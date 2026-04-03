"""Graph assembly pipeline: sync registry nodes, load/resolve/filter edges, score confidence.

Transforms the disconnected graph (355 nodes, 0 edges) into a fully connected
knowledge graph (582 nodes, 800+ edges) by:
1. Syncing all registry concepts as graph nodes
2. Loading edges from per-file extraction JSONs
3. Fuzzy-resolving dangling endpoints via ConceptRegistry.resolve()
4. Filtering low-quality "supports" edges and deduplicating
5. Assigning cross-source confidence scores
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from knowledge.graph import KnowledgeGraph
from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept, Edge, RelationType

logger = logging.getLogger(__name__)


def sync_registry_to_graph(registry: ConceptRegistry, kg: KnowledgeGraph) -> int:
    """Add all registry concepts as graph nodes.

    Returns:
        Number of concepts synced.
    """
    count = 0
    for concept in registry.all_concepts():
        if concept.id not in kg.graph.nodes:
            kg.add_concept(concept)
        count += 1
    return count


def load_edges_from_extractions(extracted_dir: Path) -> list[dict]:
    """Walk all extraction JSONs, collect edges, tag each with source_file.

    Skips files whose name starts with '_' (snapshots/metadata).
    """
    edges: list[dict] = []
    for json_file in sorted(extracted_dir.rglob("*.json")):
        if json_file.name.startswith("_"):
            continue
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Skipping unreadable file: %s", json_file)
            continue
        for e in data.get("edges", []):
            e["source_file"] = str(json_file.relative_to(extracted_dir))
            edges.append(e)
    return edges


def resolve_dangling_edges(
    edges: list[dict], registry: ConceptRegistry
) -> tuple[list[dict], list[dict]]:
    """Resolve dangling edge endpoints via fuzzy matching.

    For each edge, if source_id or target_id is not in registry,
    attempt fuzzy resolution via registry.resolve().

    Returns:
        (resolved_edges, unresolvable_edges)
    """
    resolved: list[dict] = []
    unresolvable: list[dict] = []
    # Cache resolution results to avoid repeated fuzzy lookups
    resolve_cache: dict[str, str | None] = {}

    for edge in edges:
        src = edge["source_id"]
        tgt = edge["target_id"]
        ok = True

        # Resolve source
        if registry.get(src) is None:
            if src not in resolve_cache:
                resolve_cache[src] = registry.resolve(
                    src.replace("_", " "), threshold=70
                )
            resolved_src = resolve_cache[src]
            if resolved_src is None:
                ok = False
            else:
                edge = dict(edge)  # copy to avoid mutating original
                edge["source_id"] = resolved_src
                edge["resolution"] = f"source fuzzy: {src} -> {resolved_src}"

        # Resolve target
        if ok:
            if registry.get(edge["target_id"]) is None:
                orig_tgt = tgt
                if tgt not in resolve_cache:
                    resolve_cache[tgt] = registry.resolve(
                        tgt.replace("_", " "), threshold=70
                    )
                resolved_tgt = resolve_cache[tgt]
                if resolved_tgt is None:
                    ok = False
                else:
                    edge = dict(edge) if not isinstance(edge.get("resolution"), str) else edge
                    edge = dict(edge)
                    edge["target_id"] = resolved_tgt
                    prev = edge.get("resolution") or ""
                    edge["resolution"] = (
                        f"{prev}; target fuzzy: {orig_tgt} -> {resolved_tgt}"
                        if prev
                        else f"target fuzzy: {orig_tgt} -> {resolved_tgt}"
                    )

        if ok:
            resolved.append(edge)
        else:
            unresolvable.append(edge)

    logger.info(
        "Edge resolution: %d resolved, %d unresolvable",
        len(resolved),
        len(unresolvable),
    )
    return resolved, unresolvable


# Pattern for generic co-occurrence evidence that should be filtered
_GENERIC_EVIDENCE_RE = re.compile(
    r"co-?occurring\s+(in\s+)?video|co-?occurring", re.IGNORECASE
)


def filter_and_deduplicate(edges: list[dict]) -> list[dict]:
    """Filter low-quality supports edges and deduplicate.

    Filtering rules:
    - Remove all self-loops (source_id == target_id)
    - "supports" edges: keep only if confidence >= 0.6 AND evidence is non-empty.
      Co-occurrence evidence is kept as a weak but valid signal; only truly empty
      or very low confidence supports are dropped.
      Supports with confidence < 0.6 OR generic "co-occurring" evidence with
      confidence < 0.6 are filtered.
    - All other relation types: keep regardless of confidence

    Deduplication:
    - Group by (source_id, target_id, relation)
    - Keep highest confidence, merge evidence strings
    """
    # Step 1: Filter
    filtered: list[dict] = []
    for e in edges:
        # Remove self-loops
        if e.get("source_id") == e.get("target_id"):
            continue
        relation = e.get("relation", "")
        if relation == "supports":
            conf = e.get("confidence", 0.5)
            evidence = e.get("evidence", "")
            # Filter: low confidence OR empty evidence
            if conf < 0.6:
                continue
            if not evidence.strip():
                continue
            # High-confidence supports with generic evidence are kept (weak signal)
            # Only truly generic + low-confidence are dropped above
        filtered.append(e)

    # Step 2: Deduplicate
    key_map: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for e in filtered:
        key = (e["source_id"], e["target_id"], e.get("relation", ""))
        key_map[key].append(e)

    merged: list[dict] = []
    for _key, group in key_map.items():
        best = max(group, key=lambda x: x.get("confidence", 0.5))
        # Merge evidence from all sources
        all_evidence = set()
        for g in group:
            ev = g.get("evidence", "")
            if ev:
                all_evidence.add(ev)
        best = dict(best)
        best["evidence"] = "; ".join(sorted(all_evidence))[:500]
        best["confidence"] = max(g.get("confidence", 0.5) for g in group)
        merged.append(best)

    return merged


def compute_confidence_scores(
    kg: KnowledgeGraph, registry: ConceptRegistry
) -> None:
    """Assign cross-source confidence scores to all graph nodes.

    Formula:
    - 3+ sources -> 1.0
    - ftt + at least one other -> 0.9
    - ftt only -> 0.8
    - 2+ non-ftt -> 0.7
    - single secondary -> 0.5
    """
    for node_id in kg.graph.nodes:
        concept = registry.get(node_id)
        if concept is None:
            continue
        sources = set(concept.sources)
        # Normalize: check for any source containing "ftt"
        has_ftt = any(s.startswith("ftt") for s in sources)

        if len(sources) >= 3:
            score = 1.0
        elif has_ftt and len(sources) >= 2:
            score = 0.9
        elif has_ftt:
            score = 0.8
        elif len(sources) >= 2:
            score = 0.7
        else:
            score = 0.5

        kg.graph.nodes[node_id]["confidence"] = score


def _load_registry_from_snapshot(registry_path: Path) -> ConceptRegistry:
    """Load a ConceptRegistry from the JSON snapshot file."""
    data = json.loads(registry_path.read_text())
    reg = ConceptRegistry()
    for item in data:
        concept = Concept(**item)
        reg.add(concept)
    return reg


def assemble_graph(
    extracted_dir: Path,
    registry_path: Path,
    graph_output_path: Path,
) -> dict:
    """Main assembly pipeline.

    1. Load registry from snapshot
    2. Create fresh KnowledgeGraph
    3. Sync all registry concepts as nodes
    4. Load edges from extraction JSONs
    5. Resolve dangling endpoints
    6. Filter and deduplicate
    7. Add edges to graph
    8. Compute confidence scores
    9. Save graph snapshot

    Returns:
        Stats dict with counts.
    """
    # 1. Load registry
    registry = _load_registry_from_snapshot(registry_path)
    logger.info("Registry loaded: %d concepts", len(registry))

    # 2. Fresh graph
    kg = KnowledgeGraph()

    # 3. Sync nodes
    synced = sync_registry_to_graph(registry, kg)
    logger.info("Synced %d concepts to graph (%d nodes)", synced, kg.node_count)

    # 4. Load edges
    raw_edges = load_edges_from_extractions(extracted_dir)
    logger.info("Loaded %d raw edges from extraction files", len(raw_edges))

    # 5. Resolve dangling
    resolved_edges, unresolvable = resolve_dangling_edges(raw_edges, registry)
    logger.info(
        "Resolved: %d, Unresolvable: %d", len(resolved_edges), len(unresolvable)
    )

    # 6. Filter and deduplicate
    final_edges = filter_and_deduplicate(resolved_edges)
    logger.info("After filter+dedup: %d edges", len(final_edges))

    # 7. Add edges to graph
    added = 0
    skipped = 0
    edge_type_counts: dict[str, int] = defaultdict(int)
    for e in final_edges:
        try:
            edge = Edge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                relation=RelationType(e["relation"]),
                confidence=e.get("confidence", 0.5),
                evidence=e.get("evidence", ""),
                source_file=e.get("source_file", "unknown"),
                resolution=e.get("resolution"),
            )
            kg.add_edge(edge)
            added += 1
            edge_type_counts[e["relation"]] += 1
        except Exception as exc:
            logger.warning("Skipping invalid edge %s: %s", e, exc)
            skipped += 1

    logger.info("Added %d edges, skipped %d", added, skipped)

    # 8. Confidence scores
    compute_confidence_scores(kg, registry)

    # 9. Save
    kg.to_json(graph_output_path)
    logger.info("Graph saved to %s", graph_output_path)

    # Compute orphan count
    import networkx as nx

    orphan_count = len(list(nx.isolates(kg.graph)))

    stats = {
        "node_count": kg.node_count,
        "edge_count": kg.edge_count,
        "raw_edge_count": len(raw_edges),
        "resolved_count": len(resolved_edges),
        "unresolvable_count": len(unresolvable),
        "filtered_edge_count": len(final_edges),
        "added_count": added,
        "skipped_count": skipped,
        "orphan_count": orphan_count,
        "edge_type_distribution": dict(edge_type_counts),
        "unresolvable_samples": [
            {"source_id": e["source_id"], "target_id": e["target_id"]}
            for e in unresolvable[:20]
        ],
    }

    return stats

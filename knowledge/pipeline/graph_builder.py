"""Build a KnowledgeGraph from all extraction results on disk.

Loads seed concepts, per-file extraction JSONs, and the user journey,
then assembles them into a queryable KnowledgeGraph with nodes and edges.
Validates edge references and logs warnings for dangling IDs.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from knowledge.graph import KnowledgeGraph
from knowledge.pipeline.seed import seed_registry_from_legacy_json
from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept, Edge, RelationType

logger = logging.getLogger(__name__)

EXTRACTED_DIR = Path(__file__).resolve().parent.parent / "extracted"


def _load_concepts_from_seed(registry: ConceptRegistry) -> list[Concept]:
    """Load seed concepts from _canonical_seed.json or run seeding."""
    seed_path = EXTRACTED_DIR / "_canonical_seed.json"
    if seed_path.exists():
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        concepts = []
        for entry in data:
            c = Concept.model_validate(entry)
            dup = registry.add(c)
            if dup is None:
                concepts.append(c)
        return concepts
    else:
        return seed_registry_from_legacy_json(registry)


def _load_registry_snapshot(registry: ConceptRegistry) -> list[Concept]:
    """Load full registry snapshot if available (from Plan 02)."""
    snap_path = EXTRACTED_DIR / "_registry_snapshot.json"
    if not snap_path.exists():
        return []
    data = json.loads(snap_path.read_text(encoding="utf-8"))
    concepts = []
    for entry in data:
        c = Concept.model_validate(entry)
        dup = registry.add(c)
        if dup is None:
            concepts.append(c)
    return concepts


def _load_extraction_jsons(
    extracted_dir: Path,
) -> tuple[list[Concept], list[Edge]]:
    """Load all per-file extraction JSONs from subdirectories."""
    concepts: list[Concept] = []
    edges: list[Edge] = []

    for json_file in sorted(extracted_dir.rglob("*.json")):
        # Skip snapshot/seed files
        if json_file.name.startswith("_"):
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", json_file, exc)
            continue

        for c_data in data.get("concepts", []):
            try:
                concepts.append(Concept.model_validate(c_data))
            except Exception as exc:
                logger.warning("Invalid concept in %s: %s", json_file.name, exc)

        for e_data in data.get("edges", []):
            try:
                edges.append(Edge.model_validate(e_data))
            except Exception as exc:
                logger.warning("Invalid edge in %s: %s", json_file.name, exc)

    return concepts, edges


def build_graph_from_extractions(
    extracted_dir: Path | None = None,
) -> tuple[KnowledgeGraph, ConceptRegistry]:
    """Build a complete KnowledgeGraph from all extraction outputs.

    1. Seeds registry from canonical seed or legacy JSON
    2. Loads registry snapshot if available (Plan 02 output)
    3. Loads all per-file extraction JSONs
    4. Adds all concepts as nodes
    5. Adds all edges, warning on dangling references
    6. Saves graph snapshot to _graph_snapshot.json

    Args:
        extracted_dir: Path to the extracted/ directory. Defaults to
            knowledge/extracted/ relative to this file.

    Returns:
        (KnowledgeGraph, ConceptRegistry) tuple
    """
    if extracted_dir is None:
        extracted_dir = EXTRACTED_DIR

    registry = ConceptRegistry()
    graph = KnowledgeGraph()

    # --- Step 1: Load seed concepts ---
    seed_concepts = _load_concepts_from_seed(registry)
    for c in seed_concepts:
        graph.add_concept(c)

    # --- Step 2: Load registry snapshot (Plan 02) ---
    snap_concepts = _load_registry_snapshot(registry)
    for c in snap_concepts:
        graph.add_concept(c)

    # --- Step 3: Load per-file extraction JSONs ---
    file_concepts, file_edges = _load_extraction_jsons(extracted_dir)

    # Add extracted concepts to registry and graph
    for c in file_concepts:
        dup = registry.add(c)
        if dup is None:
            graph.add_concept(c)
        elif c.id not in graph.graph.nodes:
            # Concept was deduped but node not yet in graph
            existing = registry.get(dup)
            if existing and dup not in graph.graph.nodes:
                graph.add_concept(existing)

    # Ensure all registry concepts are in the graph
    for c in registry.all_concepts():
        if c.id not in graph.graph.nodes:
            graph.add_concept(c)

    # --- Step 4: Add edges ---
    dangling_sources: set[str] = set()
    dangling_targets: set[str] = set()
    added_edges = 0

    # Dedup edges
    seen_edges: set[tuple[str, str, str]] = set()

    for edge in file_edges:
        key = (edge.source_id, edge.target_id, edge.relation.value)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        # Validate node existence
        if edge.source_id not in graph.graph.nodes:
            dangling_sources.add(edge.source_id)
            continue
        if edge.target_id not in graph.graph.nodes:
            dangling_targets.add(edge.target_id)
            continue

        graph.add_edge(edge)
        added_edges += 1

    if dangling_sources:
        logger.warning(
            "Edges referencing non-existent source nodes (%d unique): %s",
            len(dangling_sources),
            list(dangling_sources)[:10],
        )
    if dangling_targets:
        logger.warning(
            "Edges referencing non-existent target nodes (%d unique): %s",
            len(dangling_targets),
            list(dangling_targets)[:10],
        )

    # --- Step 5: Save graph snapshot ---
    snapshot_path = extracted_dir / "_graph_snapshot.json"
    graph.to_json(snapshot_path)

    return graph, registry


def print_graph_summary(graph: KnowledgeGraph, registry: ConceptRegistry) -> str:
    """Print and return a summary of the knowledge graph."""
    lines: list[str] = []
    lines.append(f"Nodes: {graph.node_count}")
    lines.append(f"Edges: {graph.edge_count}")

    # Edge type breakdown
    edge_types: Counter = Counter()
    for _u, _v, data in graph.graph.edges(data=True):
        rel = data.get("relation", "unknown")
        edge_types[rel] += 1
    lines.append("\nEdges by relation type:")
    for rel_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        lines.append(f"  {rel_type}: {count}")

    # Orphan nodes (zero edges)
    orphans = [n for n in graph.graph.nodes if graph.graph.degree(n) == 0]
    lines.append(f"\nOrphan nodes (zero edges): {len(orphans)}")
    if orphans[:5]:
        lines.append(f"  Examples: {orphans[:5]}")

    # Causal chain tests
    test_symptoms = ["scooping", "forearm_compensation", "elbow_flying_out"]
    lines.append("\nCausal chain queries:")
    for symptom in test_symptoms:
        if symptom in graph.graph.nodes:
            chains = graph.get_causal_chain(symptom)
            lines.append(f"  {symptom}: {len(chains)} paths")
            if chains:
                lines.append(f"    Example: {' -> '.join(chains[0][:5])}")
        else:
            lines.append(f"  {symptom}: node not found")

    summary = "\n".join(lines)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    graph, registry = build_graph_from_extractions()
    summary = print_graph_summary(graph, registry)
    print(summary)

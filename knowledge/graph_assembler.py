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


# === Plan 04-04: Anatomical enrichment functions ===


def enrich_with_muscles(kg: KnowledgeGraph, muscle_map_path: Path) -> int:
    """Integrate muscle data from concept-muscle map into graph nodes.

    For each concept in the map, update graph node attributes:
    - muscles_involved: list of English muscle names
    - active_or_passive: determined by predominant action type

    Returns:
        Number of nodes enriched.
    """
    data = json.loads(muscle_map_path.read_text())
    enriched = 0
    for concept_id, muscles in data.items():
        if concept_id not in kg.graph.nodes:
            continue
        muscle_names = [m["muscle"] for m in muscles]
        kg.graph.nodes[concept_id]["muscles_involved"] = muscle_names

        # Determine active_or_passive from predominant action
        actions = [m.get("action", "") for m in muscles]
        concentric = sum(1 for a in actions if "concentric" in a.lower())
        eccentric = sum(1 for a in actions if "eccentric" in a.lower())
        isometric = sum(1 for a in actions if "isometric" in a.lower() or "stabiliz" in a.lower())
        if concentric > eccentric and concentric > isometric:
            kg.graph.nodes[concept_id]["active_or_passive"] = "active"
        elif eccentric > concentric:
            kg.graph.nodes[concept_id]["active_or_passive"] = "passive"
        elif isometric >= concentric and isometric > 0:
            kg.graph.nodes[concept_id]["active_or_passive"] = "stabilizing"
        else:
            kg.graph.nodes[concept_id]["active_or_passive"] = "mixed"

        enriched += 1

    logger.info("Enriched %d nodes with muscle data", enriched)
    return enriched


# Keyword -> VLM-observable features mapping for technique nodes
_TECHNIQUE_VLM_KEYWORDS: dict[str, list[str]] = {
    "rotation": ["visible trunk/hip rotation angle", "shoulder-hip separation angle"],
    "hip": ["hip rotation direction and timing", "hip-shoulder separation visible"],
    "trunk": ["trunk rotation angle relative to baseline", "upper body coil visibility"],
    "wrist": ["wrist angle at contact", "wrist lag visibility", "wrist snap timing"],
    "unit_turn": ["shoulder turn relative to hips", "racket position behind body"],
    "contact": ["ball-racket contact position relative to body", "arm extension at contact"],
    "follow_through": ["racket path after contact", "finishing position relative to shoulder"],
    "grip": ["grip position on handle", "knuckle alignment visibility"],
    "stance": ["foot position and spacing", "weight distribution between feet"],
    "load": ["weight shift to back foot", "knee bend depth"],
    "swing": ["swing path direction", "racket face angle through swing"],
    "elbow": ["elbow position relative to torso", "elbow extension angle"],
    "shoulder": ["shoulder rotation angle", "shoulder position at contact"],
    "knee": ["knee bend angle", "knee drive direction"],
    "arm": ["arm position relative to body", "arm extension at contact point"],
    "racket": ["racket face angle", "racket head speed indicators", "racket path"],
    "leg": ["leg drive direction", "push-off leg extension"],
    "toss": ["ball position at peak", "toss arm extension"],
    "coil": ["trunk coil angle", "shoulder turn depth"],
    "weight": ["weight transfer direction", "center of gravity shift"],
    "balance": ["body balance at contact", "recovery position"],
    "backswing": ["backswing path and height", "racket take-back position"],
    "forward": ["forward swing initiation point", "forward weight shift"],
    "acceleration": ["racket acceleration visible through blur/speed", "segment sequencing"],
    "deceleration": ["follow-through path length", "body deceleration posture"],
    "pronation": ["forearm rotation angle", "racket face rotation after contact"],
    "supination": ["forearm position in backswing", "palm orientation"],
    "lag": ["wrist lag angle behind forearm", "racket lag behind hand"],
    "whip": ["sequential segment acceleration visible", "whip-like motion path"],
    "press": ["chest press forward motion", "arm adduction toward midline"],
    "chest": ["chest opening/closing angle", "pectoral engagement visible"],
    "slot": ["arm slot position", "racket drop into slot"],
}

# Biomechanics keyword -> VLM features
_BIOMECH_VLM_KEYWORDS: dict[str, list[str]] = {
    "kinetic_chain": ["sequential segment activation visible", "proximal-to-distal sequencing"],
    "angular_momentum": ["rotation speed visible at each segment"],
    "elastic": ["stretch-shortening visible in muscle groups"],
    "torque": ["joint rotation angle and speed"],
    "force": ["ground reaction force visible through leg extension"],
    "energy": ["energy transfer visible through segment sequencing"],
}


def annotate_vlm_features(kg: KnowledgeGraph) -> int:
    """Populate VLM-detectable features on graph nodes by category.

    - Symptoms: description IS the VLM feature
    - Techniques: keyword-based mapping to observable features
    - Biomechanics: joint angles and segment positions
    - Drills: no VLM features (instructions, not visible)

    Returns:
        Number of nodes annotated.
    """
    annotated = 0
    for nid, data in kg.graph.nodes(data=True):
        category = data.get("category", "")
        desc = (data.get("description") or "").lower()
        name = (data.get("name") or "").lower()
        node_id = nid.lower()

        if category == "symptom":
            # Symptom description IS a VLM feature
            original_desc = data.get("description", "")
            if original_desc:
                kg.graph.nodes[nid]["vlm_features"] = [original_desc]
                annotated += 1

        elif category == "technique":
            features: list[str] = []
            seen: set[str] = set()
            # Check name, id, and description against keyword map
            text = f"{node_id} {name} {desc}"
            for keyword, vlm_feats in _TECHNIQUE_VLM_KEYWORDS.items():
                if keyword in text:
                    for f in vlm_feats:
                        if f not in seen:
                            features.append(f)
                            seen.add(f)
            # Limit to top 3 most relevant
            if features:
                kg.graph.nodes[nid]["vlm_features"] = features[:3]
                annotated += 1

        elif category == "biomechanics":
            features = []
            seen = set()
            text = f"{node_id} {name} {desc}"
            # Try biomechanics-specific keywords first
            for keyword, vlm_feats in _BIOMECH_VLM_KEYWORDS.items():
                if keyword in text:
                    for f in vlm_feats:
                        if f not in seen:
                            features.append(f)
                            seen.add(f)
            # Also try technique keywords (many biomechanics concepts relate)
            for keyword, vlm_feats in _TECHNIQUE_VLM_KEYWORDS.items():
                if keyword in text:
                    for f in vlm_feats:
                        if f not in seen:
                            features.append(f)
                            seen.add(f)
            if features:
                kg.graph.nodes[nid]["vlm_features"] = features[:3]
                annotated += 1

        # Drills, mental_model, connection: no VLM features

    logger.info("Annotated %d nodes with VLM features", annotated)
    return annotated


def add_visible_as_edges(kg: KnowledgeGraph) -> int:
    """Create visible_as edges connecting technique/biomechanics concepts to symptom nodes.

    For nodes with VLM features, find symptom nodes whose description has keyword
    overlap, and add a visible_as edge.

    Returns:
        Number of visible_as edges added.
    """
    # Collect symptom nodes with tokenized descriptions
    symptom_tokens: dict[str, set[str]] = {}
    symptom_descs: dict[str, str] = {}
    for nid, data in kg.graph.nodes(data=True):
        if data.get("category") == "symptom":
            desc = (data.get("description") or "").lower()
            tokens = set(re.findall(r"[a-z_]{3,}", desc))
            symptom_tokens[nid] = tokens
            symptom_descs[nid] = desc

    added = 0
    for nid, data in kg.graph.nodes(data=True):
        if data.get("category") not in ("technique", "biomechanics"):
            continue
        vlm_feats = data.get("vlm_features", [])
        if not vlm_feats:
            continue

        # Tokenize VLM features for matching
        feat_text = " ".join(vlm_feats).lower()
        feat_tokens = set(re.findall(r"[a-z_]{3,}", feat_text))

        # Also use node name/id tokens
        name_tokens = set(re.findall(r"[a-z_]{3,}", f"{nid} {(data.get('name') or '').lower()}"))
        all_tokens = feat_tokens | name_tokens

        for symptom_id, stokens in symptom_tokens.items():
            # Need meaningful overlap (not just common words)
            overlap = all_tokens & stokens - {"the", "and", "for", "from", "with", "that", "this", "not"}
            if len(overlap) >= 2:
                # Check edge doesn't already exist
                existing = kg.graph.get_edge_data(nid, symptom_id, key="visible_as")
                if existing is None:
                    edge = Edge(
                        source_id=nid,
                        target_id=symptom_id,
                        relation=RelationType.VISIBLE_AS,
                        confidence=0.7,
                        evidence=f"VLM feature overlap: {', '.join(sorted(overlap)[:5])}",
                        source_file="graph_assembler:add_visible_as_edges",
                    )
                    kg.add_edge(edge)
                    added += 1

    logger.info("Added %d visible_as edges", added)
    return added


def build_why_explanation(
    kg: KnowledgeGraph, concept_id: str, muscle_profiles_path: Path
) -> dict | None:
    """Build a 'why' explanation chain for a concept.

    Returns dict: {concept, muscles: [{name, function, role}], physics, visible_symptoms}
    or None if concept not found.
    """
    if concept_id not in kg.graph.nodes:
        return None

    data = kg.graph.nodes[concept_id]

    # Load muscle profiles for detail lookup
    profiles = json.loads(muscle_profiles_path.read_text())
    profile_map = {p["name"]: p for p in profiles}

    # Build muscle details
    muscles_involved = data.get("muscles_involved", [])
    muscle_details = []
    for muscle_name in muscles_involved:
        profile = profile_map.get(muscle_name, {})
        muscle_details.append({
            "name": muscle_name,
            "function": profile.get("function", ""),
            "role": "primary",  # Default; could be refined from concept_muscle_map
        })

    # Physics: from description + causal chain context
    physics = data.get("description", "")
    # Enrich with causal predecessors
    predecessors = [
        (u, d)
        for u, _, d in kg.graph.in_edges(concept_id, data=True)
        if d.get("relation") == "causes"
    ]
    if predecessors:
        causes_text = "; ".join(
            f"{u}: {kg.graph.nodes.get(u, {}).get('description', '')[:80]}"
            for u, _ in predecessors[:3]
        )
        physics = f"{physics} [Caused by: {causes_text}]"

    # Visible symptoms: from node's vlm_features + connected symptom nodes
    visible_symptoms: list[str] = list(data.get("vlm_features", []))
    # Add symptoms from visible_as edges
    for _, target, edata in kg.graph.out_edges(concept_id, data=True):
        if edata.get("relation") == "visible_as":
            target_data = kg.graph.nodes.get(target, {})
            desc = target_data.get("description", "")
            if desc and desc not in visible_symptoms:
                visible_symptoms.append(desc)
    # Also check cause edges to symptoms
    for _, target, edata in kg.graph.out_edges(concept_id, data=True):
        if edata.get("relation") == "causes":
            target_data = kg.graph.nodes.get(target, {})
            if target_data.get("category") == "symptom":
                desc = target_data.get("description", "")
                if desc and desc not in visible_symptoms:
                    visible_symptoms.append(desc)

    return {
        "concept": {
            "id": concept_id,
            "name": data.get("name", ""),
            "description": data.get("description", ""),
        },
        "muscles": muscle_details,
        "physics": physics,
        "visible_symptoms": visible_symptoms,
    }


def update_registry_snapshot(kg: KnowledgeGraph, registry_path: Path) -> int:
    """Update registry snapshot with muscles_involved and vlm_features from graph.

    Returns:
        Number of concepts updated.
    """
    data = json.loads(registry_path.read_text())
    updated = 0
    for concept in data:
        cid = concept.get("id", "")
        if cid in kg.graph.nodes:
            node_data = kg.graph.nodes[cid]
            muscles = node_data.get("muscles_involved", [])
            vlm = node_data.get("vlm_features", [])
            active_passive = node_data.get("active_or_passive")
            changed = False
            if muscles:
                concept["muscles_involved"] = muscles
                changed = True
            if vlm:
                concept["vlm_features"] = vlm
                changed = True
            if active_passive:
                concept["active_or_passive"] = active_passive
                changed = True
            if changed:
                updated += 1

    registry_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("Updated %d concepts in registry snapshot", updated)
    return updated

# Phase 4: Graph Assembly & Anatomical Layer - Research

**Researched:** 2026-04-03
**Domain:** Knowledge graph construction, edge assembly, anatomical mapping, graph validation
**Confidence:** HIGH

## Summary

Phase 4 transforms 355 disconnected concept nodes and 1,665 extracted (but unloaded) edges into a fully connected, validated knowledge graph with anatomical depth. The critical discovery is that **the graph snapshot currently has 0 edges** -- all 355 nodes are orphans. Edges exist in per-file extraction JSONs (1,665 total, 837 with both endpoints in the registry) but were never loaded into the graph. The biomechanics anatomy data is also extremely sparse: only 6 concepts with muscle data in the graph, 0 VLM features on any node. The 6 diagnostic chains from Phase 3 are manually supplemented (not from video extraction) and have 2 validation warnings for missing registry concepts.

The work breaks into four logical streams: (1) edge assembly from existing extractions, (2) graph validation and cleanup, (3) anatomical enrichment from biomechanics source files, and (4) VLM feature annotation and "why" chain construction.

**Primary recommendation:** Load the 837 valid edges first, then resolve the 828 dangling edges by fuzzy-matching their endpoint IDs to registry concepts, then enrich the graph with anatomical data by re-processing the biomechanics book source Markdown files (the extracted JSONs are nearly empty).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GRAPH-01 | Complete concept graph with all sources merged, deduplicated, and connected | 1,665 edges available in extraction files; 837 valid, 828 need endpoint resolution |
| GRAPH-02 | Causal edge chains validated (no orphan nodes, no cycles in diagnostic paths) | KnowledgeGraph.get_causal_chain() already has cycle detection; need orphan elimination pass |
| GRAPH-03 | Every diagnostic chain has: entry symptom -> branching logic -> root cause(s) -> drill(s) -> check criteria | 6 chains exist; need expansion from video edges + domain knowledge |
| GRAPH-04 | Cross-source confidence scoring (FTT-only=high, multi-source=very high, single secondary=medium) | Registry already tracks sources per concept; scoring formula can be computed |
| ANAT-01 | Map each canonical concept to involved muscles (from biomechanics book) | Only 6/355 nodes have muscles; biomechanics source MDs have rich data but extraction was poor |
| ANAT-02 | Map each muscle to: function, training methods, common failures | 33 unique muscles found in biomechanics extractions; need structured muscle profiles |
| ANAT-03 | Connect anatomical data to VLM-detectable features | 0/355 nodes have vlm_features; need systematic annotation |
| ANAT-04 | Build "why" explanations: concept -> muscle -> physics -> visible symptom | Requires ANAT-01 + ANAT-03 + causal edges all complete |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| networkx | 3.6.1 | Directed multigraph backend | Already in use, installed |
| pydantic | v2 | Schema validation for Concept, Edge, DiagnosticChain | Already in use |
| rapidfuzz | (installed) | Fuzzy matching for dangling edge resolution | Already used in ConceptRegistry |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.10.8 | Graph visualization for debugging | Validation and sanity-check visualizations |
| json (stdlib) | - | Graph serialization via node_link_data | Already the persistence format |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib for graph viz | pyvis (interactive HTML) | Prettier but extra dependency; matplotlib sufficient for debugging |
| Manual muscle mapping | LLM-assisted enrichment | LLM can map technique concepts to muscles systematically |

## Architecture Patterns

### Current State Summary

```
Registry:    582 concepts (504 FTT, 49 TPA, 11 biomechanics, 37 user_experience)
Graph:       355 nodes, 0 edges (ALL orphans)
Extractions: 1,665 edges across 69 files (837 valid endpoints, 828 dangling)
Edge types:  1,533 "supports", 132 "causes" (heavily skewed to supports)
Muscles:     6/355 nodes have muscles_involved (1.7%)
VLM:         0/355 nodes have vlm_features (0%)
Diag chains: 6 manually created, 0 from video extractions
Biomech:     5 extracted JSONs with only 6 concepts (extraction was poor)
```

### Pattern 1: Edge Assembly Pipeline

**What:** Load edges from extraction files, resolve dangling endpoints, deduplicate, and add to graph.

**Steps:**
1. Iterate all extraction JSONs, collect edges
2. For edges where both source_id and target_id are in registry: add directly
3. For dangling edges: use `ConceptRegistry.resolve()` (fuzzy match, threshold=70) to map unknown IDs to canonical IDs
4. Deduplicate: same source_id + target_id + relation = merge (keep highest confidence, combine evidence)
5. Load into KnowledgeGraph via `add_edge()`

**Example:**
```python
from knowledge.registry import ConceptRegistry
from knowledge.graph import KnowledgeGraph
from knowledge.schemas import Edge

def resolve_edge(edge_data: dict, registry: ConceptRegistry) -> Edge | None:
    """Resolve dangling edge endpoints via fuzzy matching."""
    src = edge_data["source_id"]
    tgt = edge_data["target_id"]
    
    if registry.get(src) is None:
        resolved = registry.resolve(src.replace("_", " "), threshold=70)
        if resolved is None:
            return None  # Unresolvable
        src = resolved
    
    if registry.get(tgt) is None:
        resolved = registry.resolve(tgt.replace("_", " "), threshold=70)
        if resolved is None:
            return None
        tgt = resolved
    
    return Edge(
        source_id=src,
        target_id=tgt,
        relation=edge_data["relation"],
        confidence=edge_data.get("confidence", 0.5),
        evidence=edge_data.get("evidence", ""),
        source_file=edge_data.get("source_file", "unknown"),
    )
```

### Pattern 2: Confidence Scoring Formula

**What:** Compute cross-source confidence for each concept based on source agreement.

**Formula:**
```python
def compute_confidence(concept: Concept) -> float:
    sources = set(concept.sources)
    if len(sources) >= 3:
        return 1.0   # Very high: multi-source agreement
    if "ftt" in sources and len(sources) >= 2:
        return 0.9   # High: FTT + at least one other
    if "ftt" in sources:
        return 0.8   # High: FTT-only
    if len(sources) >= 2:
        return 0.7   # Medium-high: multiple non-FTT
    return 0.5        # Medium: single secondary source
```

### Pattern 3: Anatomical Enrichment via LLM

**What:** The biomechanics book extraction JSONs are nearly empty (6 concepts from 5 files). The source Markdown files in docs/research/ (24-28 series) have rich anatomical data. Re-process these with a focused prompt to extract muscle-to-concept mappings.

**Approach:** For each technique/biomechanics concept in the registry, use the biomechanics source files to determine:
- Which muscles are involved
- Whether the muscle action is active or passive for this concept
- What visible failure looks like when the muscle is weak/inactive

This can be done programmatically by:
1. Reading the biomechanics Markdown files
2. Building a muscle-function lookup table
3. Mapping concepts to muscles based on body segment involvement

### Pattern 4: VLM Feature Annotation

**What:** For each concept (especially symptoms and techniques), define what a VLM can visually detect.

**Categories of VLM-detectable features:**
- **Spatial:** "elbow above/below shoulder", "racket angle at contact"
- **Temporal:** "hip rotates before shoulder", "wrist lag before contact"
- **Absence:** "no visible trunk rotation", "non-dominant arm not used"
- **Comparison:** "arm position different from reference"

**Method:** Systematic annotation by concept type:
- Symptoms (75 in graph): each symptom IS a VLM feature by definition
- Techniques (245 in graph): map to observable body positions/movements
- Biomechanics (21 in graph): map to joint angles and segment positions

### Pattern 5: Graph Visualization for Debugging

**What:** Simple matplotlib + networkx visualization to verify graph structure.

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_subgraph(kg: KnowledgeGraph, center_id: str, depth: int = 2):
    """Visualize neighborhood around a concept."""
    nodes = {center_id}
    frontier = {center_id}
    for _ in range(depth):
        new_frontier = set()
        for n in frontier:
            new_frontier.update(kg.graph.predecessors(n))
            new_frontier.update(kg.graph.successors(n))
        nodes.update(new_frontier)
        frontier = new_frontier
    
    subgraph = kg.graph.subgraph(nodes)
    pos = nx.spring_layout(subgraph, k=2)
    
    # Color by edge type
    edge_colors = {
        "causes": "red", "supports": "blue", "prevents": "green",
        "drills_for": "purple", "visible_as": "orange",
    }
    
    nx.draw(subgraph, pos, with_labels=True, node_size=300, font_size=6)
    plt.savefig(f"graph_debug_{center_id}.png", dpi=150, bbox_inches="tight")
```

### Anti-Patterns to Avoid

- **Loading all 1,665 edges without validation:** Many "supports" edges from video co-occurrence are low-quality noise ("concept A appeared in same video as concept B"). Filter aggressively.
- **Treating "supports" edges as causal:** 92% of edges are "supports" (co-occurrence), only 8% are "causes". Do not use "supports" edges in diagnostic chain traversal.
- **Re-running Gemini for biomechanics enrichment:** The source Markdown files already contain the data. Parse them directly rather than sending to Gemini API.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cycle detection in causal paths | Custom DFS | `KnowledgeGraph.get_causal_chain()` already has cycle-safe traversal | Already implemented and tested |
| Fuzzy endpoint resolution | String similarity from scratch | `ConceptRegistry.resolve()` with rapidfuzz | Already implemented with tuned thresholds |
| Graph serialization | Custom JSON format | `nx.node_link_data()` / `nx.node_link_graph()` | Already the standard in KnowledgeGraph.to_json/from_json |
| Orphan detection | Manual iteration | `nx.isolates(graph)` | NetworkX built-in |
| Connected components | Custom BFS | `nx.weakly_connected_components(graph)` | NetworkX built-in |

## Common Pitfalls

### Pitfall 1: "Supports" Edge Flooding
**What goes wrong:** Loading all 1,533 "supports" edges creates a near-complete graph where everything connects to everything, making diagnostic traversal meaningless.
**Why it happens:** Video co-occurrence edges say "these concepts appeared in the same video" -- this is weak evidence of relationship.
**How to avoid:** Filter "supports" edges by confidence threshold (>= 0.7) or by requiring explicit evidence text (not just "co-occurring in video"). Alternatively, weight them much lower than "causes" edges.
**Warning signs:** Average node degree > 10 after edge loading.

### Pitfall 2: Dangling Edge Silent Failures
**What goes wrong:** 828 edges reference concept IDs not in the registry. If silently dropped, we lose ~50% of extracted relationships.
**Why it happens:** Extraction produced concept IDs that differ from canonical registry IDs (e.g., "hip_rotation_timing" vs "hip_rotation").
**How to avoid:** Fuzzy-resolve endpoints before dropping. Log all unresolvable edges for manual review. Accept that some data loss is inevitable.
**Warning signs:** Edge count after loading is < 500 (should target 800+).

### Pitfall 3: Empty Anatomical Data
**What goes wrong:** Biomechanics extraction produced only 6 concepts from 5 source files. Relying on these JSONs for ANAT-01 through ANAT-04 will produce nearly empty anatomical mappings.
**Why it happens:** The extraction prompt for biomechanics files was optimized for concept extraction, not muscle-function mapping.
**How to avoid:** Process the raw biomechanics Markdown files (24-28 series) directly with a muscle-focused extraction approach. Build a muscle lookup table first, then map concepts to muscles.
**Warning signs:** < 50% of technique concepts have muscles after enrichment.

### Pitfall 4: Diagnostic Chain Coverage Gap
**What goes wrong:** Only 6 diagnostic chains exist. Video extractions produced 0 additional chains. Phase 3 noted these were "manually supplemented."
**Why it happens:** Diagnostic chains require structured branching logic that LLM extraction rarely produces cleanly from unstructured video content.
**How to avoid:** Build diagnostic chains from the causal edge network rather than expecting them from extraction. Use graph traversal: find all symptom nodes, trace backward through "causes" edges, trace forward through "drills_for" edges. Construct DiagnosticChain objects programmatically.
**Warning signs:** < 15 diagnostic chains after Phase 4 (should target 20+).

### Pitfall 5: Missing 227 Registry Concepts from Graph
**What goes wrong:** Registry has 582 concepts but graph has only 355 nodes. 227 concepts are in the registry but not the graph.
**Why it happens:** Some extraction passes added to registry but not to graph, or graph was built from a subset.
**How to avoid:** First task should sync registry to graph -- add all missing concepts as nodes.

## Code Examples

### Loading Edges from Extraction Files
```python
import json
from pathlib import Path
from collections import defaultdict

def load_all_edges(extracted_dir: Path) -> list[dict]:
    """Collect edges from all extraction JSON files."""
    edges = []
    for json_file in extracted_dir.rglob("*.json"):
        if json_file.name.startswith("_"):
            continue
        data = json.loads(json_file.read_text())
        for e in data.get("edges", []):
            e["source_file"] = str(json_file.relative_to(extracted_dir))
            edges.append(e)
    return edges

def deduplicate_edges(edges: list[dict]) -> list[dict]:
    """Merge duplicate edges (same src+tgt+relation), keep best confidence."""
    key_map = defaultdict(list)
    for e in edges:
        key = (e["source_id"], e["target_id"], e["relation"])
        key_map[key].append(e)
    
    merged = []
    for key, group in key_map.items():
        best = max(group, key=lambda x: x.get("confidence", 0.5))
        # Combine evidence from all sources
        all_evidence = "; ".join(set(e.get("evidence", "") for e in group if e.get("evidence")))
        best["evidence"] = all_evidence[:500]  # Truncate
        best["confidence"] = max(e.get("confidence", 0.5) for e in group)
        merged.append(best)
    return merged
```

### Graph Validation
```python
import networkx as nx

def validate_graph(kg: KnowledgeGraph) -> dict:
    """Run validation checks, return report."""
    g = kg.graph
    report = {}
    
    # Orphan nodes
    orphans = list(nx.isolates(g))
    report["orphan_count"] = len(orphans)
    report["orphan_ids"] = orphans[:20]  # Sample
    
    # Weakly connected components
    components = list(nx.weakly_connected_components(g))
    report["component_count"] = len(components)
    report["largest_component"] = len(max(components, key=len)) if components else 0
    
    # Cycle detection in causal subgraph
    causal_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("relation") == "causes"]
    causal_subgraph = nx.DiGraph(causal_edges)
    cycles = list(nx.simple_cycles(causal_subgraph))
    report["causal_cycles"] = len(cycles)
    report["cycle_details"] = cycles[:5]
    
    # Edge type distribution
    from collections import Counter
    report["edge_types"] = dict(Counter(d.get("relation") for _, _, d in g.edges(data=True)))
    
    return report
```

### Diagnostic Chain Generation from Graph
```python
def generate_diagnostic_chains(kg: KnowledgeGraph) -> list[DiagnosticChain]:
    """Build diagnostic chains by traversing causal edges from symptom nodes."""
    g = kg.graph
    chains = []
    
    # Find all symptom nodes
    symptom_nodes = [n for n, d in g.nodes(data=True) if d.get("category") == "symptom"]
    
    for symptom_id in symptom_nodes:
        # Trace backward through causes
        causal_paths = kg.get_causal_chain(symptom_id, cause_type="causes")
        if not causal_paths:
            continue
        
        # Collect root causes (terminal nodes in causal paths)
        root_causes = list({path[-1] for path in causal_paths})
        
        # Find drills connected to root causes
        drills = set()
        for rc in root_causes:
            for _, target, data in g.out_edges(rc, data=True):
                if data.get("relation") == "drills_for":
                    drills.add(target)
        
        # Build check sequence from intermediate nodes
        # ...
        
        chains.append(DiagnosticChain(
            id=f"dc_{symptom_id}",
            symptom=g.nodes[symptom_id].get("description", ""),
            symptom_zh=g.nodes[symptom_id].get("name_zh", ""),
            symptom_concept_id=symptom_id,
            check_sequence=[],  # Populated from path analysis
            root_causes=root_causes,
            drills=list(drills),
            priority=2,
        ))
    
    return chains
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flat concept list | Directed multigraph with typed edges | Phase 1 (schemas.py) | Enables causal chain traversal |
| Per-video isolated concepts | Canonical registry with fuzzy dedup | Phase 1-2 | 582 merged concepts from 100+ sources |
| Manual diagnostic chains | Graph-derived chains | Phase 4 (this phase) | Scalable chain generation |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/test_knowledge_graph.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GRAPH-01 | All valid edges loaded into graph | integration | `pytest tests/test_graph_assembly.py::test_edge_loading -x` | Wave 0 |
| GRAPH-02 | No orphan nodes, no causal cycles | unit | `pytest tests/test_graph_assembly.py::test_validation -x` | Wave 0 |
| GRAPH-03 | Diagnostic chains have required structure | unit | `pytest tests/test_graph_assembly.py::test_diagnostic_chains -x` | Wave 0 |
| GRAPH-04 | Confidence scores computed correctly | unit | `pytest tests/test_graph_assembly.py::test_confidence_scoring -x` | Wave 0 |
| ANAT-01 | Concepts mapped to muscles | integration | `pytest tests/test_graph_assembly.py::test_muscle_mapping -x` | Wave 0 |
| ANAT-02 | Muscle profiles with function/training/failures | unit | `pytest tests/test_graph_assembly.py::test_muscle_profiles -x` | Wave 0 |
| ANAT-03 | VLM features populated | integration | `pytest tests/test_graph_assembly.py::test_vlm_features -x` | Wave 0 |
| ANAT-04 | "Why" explanations traversable | integration | `pytest tests/test_graph_assembly.py::test_why_chains -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_knowledge_graph.py tests/test_graph_assembly.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_graph_assembly.py` -- covers GRAPH-01 through GRAPH-04, ANAT-01 through ANAT-04
- [ ] Edge loading, validation, diagnostic chain, confidence, and anatomical test functions

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| networkx | Graph operations | Yes | 3.6.1 | -- |
| pydantic | Schema validation | Yes | v2 | -- |
| rapidfuzz | Fuzzy matching | Yes | (installed) | -- |
| matplotlib | Graph visualization | Yes | 3.10.8 | -- |
| pytest | Testing | Yes | (installed) | -- |
| graphviz | Alternative viz | No | -- | Use matplotlib + networkx |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** graphviz (use matplotlib).

## Open Questions

1. **How aggressive should "supports" edge filtering be?**
   - What we know: 1,533 "supports" edges, many are just co-occurrence noise
   - What's unclear: Optimal threshold -- confidence >= 0.7? Require non-generic evidence text?
   - Recommendation: Start with confidence >= 0.6 AND evidence not matching "Co-occurring in video" pattern; tune based on average node degree

2. **Should the 227 missing registry concepts be added to the graph?**
   - What we know: Registry=582, Graph=355, difference=227
   - What's unclear: Why they were excluded -- quality filtering or bug?
   - Recommendation: Add them all; orphan nodes are better than missing concepts (edges may connect them later)

3. **How many diagnostic chains should Phase 4 target?**
   - What we know: 6 exist, 75 symptom nodes in graph
   - What's unclear: Not every symptom needs a full chain; some are too niche
   - Recommendation: Target 15-25 chains covering the most common/important forehand issues

4. **How to handle the 2 validation warnings in diagnostic chains?**
   - What we know: `trunk_sequencing` and `racket_drop` referenced but not in registry
   - What's unclear: Are these valid concepts that should be added, or naming mismatches?
   - Recommendation: Resolve via fuzzy match first; add as new concepts if truly missing

## Sources

### Primary (HIGH confidence)
- `knowledge/extracted/_graph_snapshot.json` -- actual graph state: 355 nodes, 0 edges
- `knowledge/extracted/_registry_snapshot.json` -- 582 concepts with category/source breakdown
- `knowledge/extracted/ftt_video_diagnostic_chains.json` -- 6 chains, 2 warnings
- `knowledge/schemas.py`, `knowledge/registry.py`, `knowledge/graph.py` -- code analysis
- All 69 extraction files with edges (counted and analyzed)

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- architecture patterns
- `.planning/research/PITFALLS.md` -- known pitfalls

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already installed and in use
- Architecture: HIGH - existing code analyzed, exact gap counts verified
- Pitfalls: HIGH - discovered through actual data analysis (0 edges, 828 dangling, etc.)
- Anatomical enrichment: MEDIUM - biomechanics source quality needs verification during execution

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable domain, no external API changes expected)

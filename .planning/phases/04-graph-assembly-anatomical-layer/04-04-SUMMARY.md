---
phase: 04-graph-assembly-anatomical-layer
plan: 04
subsystem: knowledge-graph
tags: [anatomy, muscles, vlm-features, graph-enrichment, explanation-chains]

requires:
  - phase: 04-01
    provides: "Assembled knowledge graph with 582 nodes and 869 edges"
  - phase: 04-02
    provides: "275 concept-muscle mappings and 32 muscle profiles"
provides:
  - "Graph with 276/582 nodes having muscles_involved (English names)"
  - "207/582 nodes with VLM-detectable features populated"
  - "39 visible_as edges connecting techniques to observable symptoms"
  - "build_why_explanation() producing 166 valid concept->muscle->physics->symptom chains"
  - "Updated registry snapshot with enrichment data on 350 concepts"
affects: [vlm-diagnostics, coaching-engine, diagnostic-chains]

tech-stack:
  added: []
  patterns: ["keyword-rule-mapping for VLM feature annotation", "causal-chain-traversal for explanation building"]

key-files:
  created: []
  modified:
    - knowledge/graph_assembler.py
    - knowledge/extracted/_graph_snapshot.json
    - knowledge/extracted/_registry_snapshot.json
    - tests/test_anatomical.py

decisions:
  - "Symptom descriptions used directly as VLM features (they ARE the visible observation)"
  - "Keyword-rule mapping for technique VLM features (30+ keyword categories)"
  - "visible_as edges use 2+ token overlap threshold for quality matching"
  - "Pre-existing Chinese muscle names in snapshot preserved for unmapped nodes"

metrics:
  duration: "4min"
  completed: "2026-04-03"
  tasks_completed: 1
  tasks_total: 1
  files_modified: 4
---

# Phase 04 Plan 04: Anatomical Graph Enrichment Summary

Integrated muscle mappings (from Plan 02) and VLM-detectable features into the assembled knowledge graph, built traversable "why" explanation chains connecting concept to muscle to physics to visible symptom.

## What Was Done

### Task 1: Integrate muscle data + VLM features into graph and registry

Added 5 new functions to `knowledge/graph_assembler.py`:

1. **enrich_with_muscles()**: Loads `_concept_muscle_map.json`, updates 275 graph nodes with English muscle names and active/passive classification based on predominant muscle action type.

2. **annotate_vlm_features()**: Populates VLM-observable features on 207 nodes:
   - Symptoms (27/27 = 100%): description IS the VLM feature
   - Techniques: keyword-to-feature mapping with 30+ keyword categories covering rotation, wrist, contact, follow-through, grip, stance, etc.
   - Biomechanics: joint angles and segment position features

3. **add_visible_as_edges()**: Creates 39 `visible_as` edges connecting technique/biomechanics concepts to symptom nodes via keyword overlap matching (2+ token threshold).

4. **build_why_explanation()**: Builds explanation chains for any concept, returning structured data: concept info, muscle details (name + function + role from profiles), physics context (description + causal predecessors), and visible symptoms (VLM features + connected symptom nodes). 166 valid chains produced.

5. **update_registry_snapshot()**: Propagates muscles_involved, vlm_features, and active_or_passive from graph nodes back to the registry JSON. 350 concepts updated.

## Key Metrics

| Metric | Before | After |
|--------|--------|-------|
| Nodes with muscles | 13/582 | 276/582 (47.4%) |
| Nodes with VLM features | 0/582 | 207/582 (35.6%) |
| Technique nodes with muscles | ~6/396 | 200/396 (50.5%) |
| Symptom nodes with VLM | 0/27 | 27/27 (100%) |
| visible_as edges | 0 | 39 |
| Valid why-chains | 0 | 166 |
| Total graph edges | 869 | 884 |

## Tests

6 new test classes with 7 tests added to `tests/test_anatomical.py` (total: 17 tests, all passing):
- TestEnrichGraphWithMuscles (2 tests): coverage + English name validation
- TestVlmFeatureAnnotation (2 tests): symptom + technique VLM features
- TestVisibleAsEdges (1 test): visible_as edge existence
- TestWhyChainTraversal (2 tests): chain count + structure validation
- TestRegistrySnapshotUpdated (1 test): registry propagation

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all data paths are wired to real extracted data.

## Commits

| Hash | Message |
|------|---------|
| 4949566 | test(04-04): add failing tests for graph muscle enrichment and VLM features |
| baab5e4 | feat(04-04): integrate muscle data + VLM features into graph and registry |

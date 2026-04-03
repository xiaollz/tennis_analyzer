---
phase: 01-schema-infrastructure
plan: 02
subsystem: infra
tags: [rapidfuzz, networkx, fuzzy-dedup, knowledge-graph, pydantic]

requires:
  - phase: 01-schema-infrastructure/01
    provides: Pydantic v2 models (Concept, Edge, RelationType, DiagnosticChain)
provides:
  - ConceptRegistry with fuzzy dedup via rapidfuzz (token_sort_ratio)
  - KnowledgeGraph wrapping NetworkX MultiDiGraph with causal chain traversal
  - JSON serialization/deserialization for graph persistence
affects: [02-ftt-extraction, 03-secondary-sources, 04-graph-assembly]

tech-stack:
  added: [rapidfuzz]
  patterns: [fuzzy-dedup-registry, networkx-multidigraph-wrapper, node-link-data-serialization]

key-files:
  created:
    - knowledge/registry.py
    - knowledge/graph.py
    - tests/test_knowledge_registry.py
    - tests/test_knowledge_graph.py
  modified:
    - knowledge/__init__.py

key-decisions:
  - "English-only fuzzy matching for dedup (Chinese names display-only)"
  - "token_sort_ratio scorer handles word-order variations (hip rotation vs rotation of hips)"
  - "Dedup threshold 85 for add, resolve threshold 70 for fuzzy lookup"

patterns-established:
  - "Registry pattern: canonical store with fuzzy dedup prevents concept explosion"
  - "Graph wrapper pattern: typed add/query methods over NetworkX for domain safety"
  - "TDD red-green for infrastructure modules"

requirements-completed: [INFRA-04, INFRA-05]

duration: 4min
completed: 2026-04-03
---

# Phase 01 Plan 02: Registry & Graph Summary

**ConceptRegistry with rapidfuzz fuzzy dedup (threshold 85) and KnowledgeGraph wrapping NetworkX MultiDiGraph with causal chain traversal and JSON roundtrip**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-03T02:19:46Z
- **Completed:** 2026-04-03T02:24:05Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- ConceptRegistry detects near-duplicate concepts via token_sort_ratio >= 85, preventing concept explosion
- KnowledgeGraph supports multiple typed edges between same node pair (MultiDiGraph) and cycle-safe causal chain traversal
- JSON roundtrip via node_link_data preserves all nodes, edges, and attributes
- Full test suite: 26 tests across 3 files (schemas + registry + graph) all green

## Task Commits

Each task was committed atomically:

1. **Task 1: Install rapidfuzz and build ConceptRegistry with fuzzy dedup** - `870ccbc` (feat)
2. **Task 2: Build KnowledgeGraph wrapper with causal chain queries and JSON roundtrip** - `3d1ab28` (feat)

## Files Created/Modified
- `knowledge/registry.py` - ConceptRegistry with fuzzy dedup via rapidfuzz
- `knowledge/graph.py` - KnowledgeGraph wrapping NetworkX MultiDiGraph
- `knowledge/__init__.py` - Updated exports to include ConceptRegistry and KnowledgeGraph
- `tests/test_knowledge_registry.py` - 9 tests for dedup, resolve, false-positive avoidance
- `tests/test_knowledge_graph.py` - 8 tests for add, multi-edge, causal chain, JSON roundtrip

## Decisions Made
- English-only fuzzy matching for dedup (Chinese names are display-only, too many valid translations)
- token_sort_ratio scorer chosen over ratio/partial_ratio for word-order invariance
- Dedup threshold 85 (strict) for add, resolve threshold 70 (looser) for fuzzy lookup

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all functionality is fully wired.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Registry and graph are ready to receive extracted concepts from Phase 02 (FTT extraction)
- All public APIs exported from knowledge package __init__.py
- Gemini API proxy compatibility still needs verification (affects Phase 3)

---
*Phase: 01-schema-infrastructure*
*Completed: 2026-04-03*

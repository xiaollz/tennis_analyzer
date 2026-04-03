---
phase: 05-output-generation-vlm-engine
plan: 02
subsystem: vlm-engine
tags: [jinja2, vlm-prompt, knowledge-graph, two-pass-architecture, diagnostic-chains]

# Dependency graph
requires:
  - phase: 04-graph-assembly
    provides: KnowledgeGraph with 582 nodes, 884 edges, 18 diagnostic chains
provides:
  - VLMPromptCompiler class for two-pass VLM prompt generation
  - get_symptom_subgraph method for BFS subgraph extraction
  - 3 Jinja2 templates (system_prompt, symptom_checklist, diagnostic_deep)
  - Static coaching content preserved from _FTT_SYSTEM_PROMPT
affects: [05-03, vlm-analyzer-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [two-pass-vlm-architecture, jinja2-template-prompt-generation, hybrid-static-dynamic-prompt]

key-files:
  created:
    - knowledge/output/vlm_prompt.py
    - knowledge/templates/vlm/system_prompt.md.j2
    - knowledge/templates/vlm/symptom_checklist.j2
    - knowledge/templates/vlm/diagnostic_deep.j2
    - tests/test_vlm_prompt.py
  modified:
    - knowledge/graph.py
    - knowledge/output/__init__.py

key-decisions:
  - "Hybrid prompt architecture: static Jinja2 templates for coaching voice + dynamic graph-backed diagnostic injection"
  - "Pass 1 budget ~990 chars (18 chains), well under 3K limit; Pass 2 dynamic ~1.4K chars per 3 chains, well under 10K"
  - "Budget enforcement via line-boundary truncation rather than node-level removal (sufficient for current 18-chain set)"

patterns-established:
  - "Two-pass VLM: Pass 1 symptom scan -> Pass 2 targeted diagnostic"
  - "Jinja2 FileSystemLoader for knowledge/templates/vlm/ directory"
  - "get_symptom_subgraph BFS with diagnostic_types filter (causes, visible_as, drills_for)"

requirements-completed: [VLM-01, VLM-03, VLM-04, VLM-05, OUT-04]

# Metrics
duration: 6min
completed: 2026-04-03
---

# Phase 05 Plan 02: VLM Prompt Compiler Summary

**Two-pass VLM prompt compiler with Jinja2 templates: Pass 1 symptom scan (~990 chars) + Pass 2 targeted diagnostic injection (~1.4K dynamic chars), preserving all 16 principles and frame-by-frame coaching guide**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-03T08:30:35Z
- **Completed:** 2026-04-03T08:37:08Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- VLMPromptCompiler with compile_system_prompt, compile_pass1_prompt, compile_pass2_prompt
- get_symptom_subgraph method on KnowledgeGraph for BFS 2-hop subgraph extraction (causes/visible_as/drills_for edges)
- 3 Jinja2 templates decomposing the 650-line _FTT_SYSTEM_PROMPT into static + dynamic parts
- All 18 diagnostic chains available for both Pass 1 and Pass 2
- Budget enforcement: Pass 1 at 990 chars (budget 3K), static prompt at 5362 chars, Pass 2 dynamic well under 10K

## Task Commits

Each task was committed atomically:

1. **Task 1: Add get_symptom_subgraph + VLM templates** - `01e23f9` (feat)
2. **Task 2: VLMPromptCompiler + tests (TDD RED)** - `11efd25` (test)
3. **Task 2: VLMPromptCompiler + tests (TDD GREEN)** - `2cf9e80` (feat)

## Files Created/Modified
- `knowledge/output/vlm_prompt.py` - VLMPromptCompiler class with two-pass prompt generation
- `knowledge/templates/vlm/system_prompt.md.j2` - Static coaching content (frame guide, 16 principles, drills, output format)
- `knowledge/templates/vlm/symptom_checklist.j2` - Pass 1 compact symptom category list
- `knowledge/templates/vlm/diagnostic_deep.j2` - Pass 2 targeted diagnostic injection template
- `knowledge/graph.py` - Added get_symptom_subgraph method
- `knowledge/output/__init__.py` - Added VLMPromptCompiler export
- `tests/test_vlm_prompt.py` - 14 tests covering all requirements

## Decisions Made
- Hybrid prompt: static templates preserve coaching voice verbatim, dynamic sections inject only detected symptoms
- Pass 1 uses numbered categories sorted by priority for reliable VLM response parsing
- Budget enforcement uses line-boundary truncation (sufficient for current chain count)
- Subgraph extraction filters to only diagnostic edge types (causes, visible_as, drills_for)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all data paths are wired to real diagnostic chains and graph data.

## Next Phase Readiness
- VLMPromptCompiler ready for integration into vlm_analyzer.py (Plan 05-03)
- All 18 chains supported in both passes
- Templates can be extended with additional chains without code changes

---
*Phase: 05-output-generation-vlm-engine*
*Completed: 2026-04-03*

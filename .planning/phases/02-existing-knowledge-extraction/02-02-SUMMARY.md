---
phase: 02-existing-knowledge-extraction
plan: 02
subsystem: knowledge-pipeline
tags: [extraction, markdown-parsing, regex, muscle-mapping, knowledge-graph, biomechanics]

requires:
  - phase: 02-existing-knowledge-extraction
    plan: 01
    provides: Seeded ConceptRegistry with 105 concepts, extraction pipeline scaffolding
provides:
  - 32 JSON extraction files from all research Markdown files
  - Registry snapshot with 297 concepts for Plan 03 to consume
  - 8 handler functions for per-file-type extraction
  - Muscle-to-concept mappings from biomechanics files
affects: [02-existing-knowledge-extraction, 03-video-extraction]

tech-stack:
  added: []
  patterns: [per-section-muscle-propagation, arrow-chain-edge-extraction, synthesis-first-processing-order]

key-files:
  created:
    - knowledge/pipeline/handlers.py
    - knowledge/extracted/_registry_snapshot.json
    - knowledge/extracted/synthesis/13_synthesis.json
    - knowledge/extracted/biomechanics/24_biomechanics_ch1_ch8.json
  modified:
    - knowledge/pipeline/extractor.py
    - tests/test_extraction.py

key-decisions:
  - "Concept name filtering requires at least 2 alphabetic chars and min 4 chars for new concepts to avoid junk from numbered headers"
  - "Muscles propagated to concepts via bold-term references in biomechanics sections, not just header-level matching"
  - "Processing 32 files (not 31) since 04_ftt_blog has _1 and _2 variants"
  - "Edge count lower than expected (4 new edges) due to strict registry-based resolution; most arrow chains reference Chinese terms not matching English registry"

patterns-established:
  - "Handler dispatch: FILE_HANDLERS dict maps filename prefixes to typed handler functions"
  - "Processing order: synthesis first to establish canonical names, then primary sources, then videos, then biomechanics"
  - "Muscle propagation: bold terms in biomechanics sections get muscles attached to their resolved concepts"

requirements-completed: [EXIST-01, EXIST-03, EXIST-04]

duration: 11min
completed: 2026-04-03
---

# Phase 02 Plan 02: Research File Extraction Summary

**Extracted 32 research Markdown files into 297 registry concepts with muscle-to-concept mappings via 8 typed handler functions and correct synthesis-first processing order**

## Performance

- **Duration:** 11 min
- **Started:** 2026-04-03T03:26:24Z
- **Completed:** 2026-04-03T03:37:27Z
- **Tasks:** 2
- **Files modified:** 35

## Accomplishments
- Implemented 8 extraction handlers (synthesis, ftt_book, ftt_blog, ftt_videos, tpa_videos, ftt_specific, biomechanics, generic) with proper edge confidence by source type
- Processed all 32 research files in correct order producing 32 JSON outputs across 8 category subdirectories
- Registry contains 297 unique concepts (within 150-300 target), 13 concepts with muscles_involved populated
- Test suite expanded to 23 passing tests covering EXIST-01, EXIST-03, EXIST-04

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement extraction handlers** - `7b9e82c` (feat)
2. **Task 2: Run full pipeline and update tests** - `4f4d276` (feat)

## Files Created/Modified
- `knowledge/pipeline/handlers.py` - 8 handler functions with muscle table parsing, arrow chain edge extraction, FTT mapping cross-references
- `knowledge/pipeline/extractor.py` - Updated with real handler imports, PROCESSING_ORDER list, save_extraction_results, save_registry_snapshot, __main__ runner
- `knowledge/extracted/_registry_snapshot.json` - 297 concepts as JSON array
- `knowledge/extracted/synthesis/*.json` - 4 synthesis extraction files
- `knowledge/extracted/ftt_book/*.json` - 2 FTT book extraction files
- `knowledge/extracted/ftt_blog/*.json` - 6 FTT blog extraction files
- `knowledge/extracted/ftt_specific/*.json` - 6 FTT specific deep-dive files
- `knowledge/extracted/ftt_videos/*.json` - 3 FTT video analysis files
- `knowledge/extracted/tpa/*.json` - 5 TPA video/kinetic chain files
- `knowledge/extracted/biomechanics/*.json` - 5 biomechanics textbook files
- `knowledge/extracted/misc/*.json` - 1 YouTube notes file
- `tests/test_extraction.py` - 23 passing tests, 2 skipped stubs

## Decisions Made
- Filtered junk concepts from numbered headers by requiring min 2 alpha chars and 4 char min length for new concept names
- Propagated muscles to concepts via bold-term matching (not just header-level), resulting in 13 concepts with muscle data
- Accepted 32 files (not 31 as plan estimated) since 04_ftt_blog has two parts

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Filtered junk concepts from numbered section headers**
- **Found during:** Task 2 (full pipeline run)
- **Issue:** Headers like "1.2", "4 /", "2-4" were creating garbage concepts with names like "n4", "n1_2"
- **Fix:** Added alpha character count filter (min 2 alpha chars) and raised min name length to 4 chars
- **Files modified:** knowledge/pipeline/handlers.py
- **Verification:** Registry count dropped from 314 (over limit) to 297 (within 150-300 range)
- **Committed in:** 4f4d276 (Task 2 commit)

**2. [Rule 1 - Bug] Fixed test including non-research JSON files**
- **Found during:** Task 2 (test run)
- **Issue:** user_journey/learning.json (from parallel plan 02-03) has different schema (source vs source_file), causing test failure
- **Fix:** Scoped test to research extraction categories only
- **Files modified:** tests/test_extraction.py
- **Committed in:** 4f4d276 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
- Edge extraction yield is lower than expected (4 new edges from arrow chains) because most arrow chains in the Chinese research files use Chinese terms that don't fuzzy-match the English-only registry. This is acceptable for now; Plan 03's cross-source dedup may improve resolution.

## Known Stubs
None - all handlers are fully implemented.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Registry snapshot with 297 concepts ready for Plan 03 consumption
- All 32 research files extracted and structured
- 13 concepts have muscle mappings from biomechanics files
- Edge extraction infrastructure in place for future enhancement

---
*Phase: 02-existing-knowledge-extraction*
*Completed: 2026-04-03*

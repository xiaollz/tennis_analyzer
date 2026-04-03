---
phase: 02-existing-knowledge-extraction
plan: 01
subsystem: knowledge-pipeline
tags: [pydantic, rapidfuzz, knowledge-graph, json-migration, extraction-pipeline]

requires:
  - phase: 01-knowledge-schema-design
    provides: Pydantic Concept/Edge schemas, ConceptRegistry with fuzzy dedup
provides:
  - Seeded ConceptRegistry with 105 canonical concepts from 3 legacy JSON files
  - Extraction pipeline scaffolding with file-type handler dispatch
  - Canonical seed snapshot (_canonical_seed.json) for reproducible seeding
affects: [02-existing-knowledge-extraction, 03-video-extraction]

tech-stack:
  added: []
  patterns: [legacy-json-migration, filename-prefix-handler-dispatch, snake-case-id-generation]

key-files:
  created:
    - knowledge/pipeline/__init__.py
    - knowledge/pipeline/seed.py
    - knowledge/pipeline/extractor.py
    - knowledge/extracted/_canonical_seed.json
    - tests/test_extraction.py
  modified: []

key-decisions:
  - "User journey problems use P01-style IDs (problem_p01) instead of Chinese-derived snake_case for stability"
  - "Breakthroughs use B01-style IDs (breakthrough_b01) since descriptions are Chinese-only"
  - "TPA concept names parsed via '/' split to separate English from Chinese"
  - "IDs starting with digits get 'n' prefix to satisfy snake_case regex"
  - "Accepted 105 concepts (above plan estimate of 60-100) due to unique user journey items"

patterns-established:
  - "Legacy JSON migration: process sources in priority order (FTT > TPA > user journey)"
  - "Duplicate handling: merge aliases/sources into existing concept on registry collision"
  - "Handler dispatch: FILE_HANDLERS dict maps filename prefixes to handler functions"

requirements-completed: [EXIST-04]

duration: 6min
completed: 2026-04-03
---

# Phase 02 Plan 01: Legacy JSON Seed Migration Summary

**Migrated 105 canonical concepts from 3 legacy JSON files into Pydantic-validated ConceptRegistry with cross-source fuzzy dedup and extraction pipeline scaffolding**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-03T03:16:21Z
- **Completed:** 2026-04-03T03:22:39Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 5

## Accomplishments
- Seeded ConceptRegistry with 105 canonical concepts: 48 FTT core, 20 TPA kinetic chain (4 deduped), 21 user problems, 16 user breakthroughs
- Built category mapping from Chinese category strings to ConceptType enum values
- Created extraction pipeline with 7 handler stubs and filename-prefix dispatch
- Established test infrastructure with 13 passing tests and 4 skipped stubs for future plans

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `3462d18` (test)
2. **Task 1 GREEN: Implementation** - `fe2cc32` (feat)

## Files Created/Modified
- `knowledge/pipeline/__init__.py` - Package init for extraction pipeline
- `knowledge/pipeline/seed.py` - Legacy JSON migration: CATEGORY_MAP, to_snake_id, seed_registry_from_legacy_json, save_seed_snapshot
- `knowledge/pipeline/extractor.py` - ExtractionResult dataclass, 7 handler stubs, FILE_HANDLERS dispatch, get_handler, run_extraction
- `knowledge/extracted/_canonical_seed.json` - 105 serialized Concept objects (seed snapshot)
- `tests/test_extraction.py` - 13 passing tests + 4 skipped stubs covering EXIST-01 through EXIST-05

## Decisions Made
- User journey items use stable P01/B01-based IDs instead of transliterating Chinese descriptions
- Accepted 105 concepts (plan estimated 60-100) since the 37 user journey items are genuinely unique
- IDs starting with digits (e.g., "40ms Release Window") get 'n' prefix to satisfy Pydantic regex
- Cross-source dedup successfully merged 4 TPA concepts with FTT equivalents via fuzzy matching

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed digit-prefix IDs violating Pydantic validation**
- **Found during:** Task 1 GREEN phase
- **Issue:** TPA concept "40ms Release Window" produced ID "40ms_release_window" which fails `^[a-z][a-z0-9_]*$` regex
- **Fix:** Added 'n' prefix for IDs starting with digits
- **Files modified:** knowledge/pipeline/seed.py
- **Verification:** All 105 concepts pass Pydantic validation

**2. [Rule 1 - Bug] Fixed empty IDs from Chinese-only user journey names**
- **Found during:** Task 1 GREEN phase
- **Issue:** User journey problem names like "V形 Scooping（向上捞球）" produced empty snake_case IDs when no Latin characters present
- **Fix:** Used stable P01/B01-based IDs (problem_p01, breakthrough_b01)
- **Files modified:** knowledge/pipeline/seed.py
- **Verification:** All user journey concepts have valid IDs

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for Pydantic validation correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed bugs above.

## Known Stubs
- `knowledge/pipeline/extractor.py`: 7 handler functions (extract_ftt_book, extract_ftt_blog, extract_ftt_videos, extract_tpa_videos, extract_biomechanics, extract_user_journey, extract_generic) return empty ExtractionResult. These are intentional scaffolding stubs to be implemented in Plans 02 and 03.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Registry seeded and ready for Markdown extraction (Plan 02)
- Handler dispatch in place; each handler just needs implementation
- Test infrastructure ready; skipped stubs provide clear targets for future test coverage

---
*Phase: 02-existing-knowledge-extraction*
*Completed: 2026-04-03*

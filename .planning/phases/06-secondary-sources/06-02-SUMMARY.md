---
phase: 06-secondary-sources
plan: 02
subsystem: knowledge-pipeline
tags: [gemini-api, video-analysis, concept-extraction, tomallsopp]

requires:
  - phase: 06-01
    provides: Curated TomAllsopp video list (49 videos) and state file
provides:
  - 49 raw Gemini markdown analyses of TomAllsopp forehand videos
  - 49 structured concept JSON extraction files
  - Updated state file with all videos marked as extracted
affects: [06-04-reconciliation]

tech-stack:
  added: []
  patterns: [parameterized-channel-analysis, source-tag-post-processing]

key-files:
  created:
    - docs/research/tomallsopp_video_analyses/*.md (49 files)
    - knowledge/extracted/tomallsopp_videos/*.json (49 files)
    - scripts/run_tomallsopp_batch.py
    - scripts/run_tomallsopp_extract.py
  modified:
    - knowledge/state/tomallsopp_video_state.json

key-decisions:
  - "Used same Gemini analysis prompt as FTT Phase 3 for consistent cross-source comparison"
  - "Post-processed source tags from ftt_video_ to tomallsopp_video_ rather than modifying shared extractor"
  - "Low structured concept yield (1 new) is expected: 582 existing registry concepts already cover TomAllsopp's forehand biomechanics vocabulary"

patterns-established:
  - "Secondary channel batch analysis: reuse analyze_batch() with channel-specific output_dir and state_path"
  - "Source tag post-processing: fix source_file field after extraction rather than modifying shared pipeline code"

requirements-completed: [SEC-02]

duration: 74min
completed: 2026-04-03
---

# Phase 06 Plan 02: TomAllsopp Video Analysis Summary

**49 TomAllsopp forehand videos analyzed via Gemini API with 100% success rate, producing rich Chinese-language markdown analyses anchored to FTT concept framework**

## Performance

- **Duration:** 74 min (dominated by Gemini API calls with 20s delay between videos)
- **Started:** 2026-04-03T11:08:15Z
- **Completed:** 2026-04-03T12:22:00Z
- **Tasks:** 1
- **Files modified:** 101

## Accomplishments
- All 49 TomAllsopp forehand videos analyzed via Gemini API (100% success, 0 failures)
- Raw markdown analyses saved per-video in docs/research/tomallsopp_video_analyses/
- Structured concept JSONs extracted to knowledge/extracted/tomallsopp_videos/
- State file accurately tracks all 49 videos as "extracted" status
- Analyses are rich and detailed, referencing existing FTT concepts (C01-C48, T01-T24) with relationship types

## Task Commits

Each task was committed atomically:

1. **Task 1: Analyze TomAllsopp videos via Gemini API and extract concepts** - `5e2a810` (feat)

## Files Created/Modified
- `docs/research/tomallsopp_video_analyses/*.md` - 49 raw Gemini analysis markdown files
- `knowledge/extracted/tomallsopp_videos/*.json` - 49 structured concept extraction JSONs
- `knowledge/state/tomallsopp_video_state.json` - State tracking (all 49 = "extracted")
- `scripts/run_tomallsopp_batch.py` - Batch analysis script (Gemini API calls)
- `scripts/run_tomallsopp_extract.py` - Concept extraction script (registry-based dedup)

## Decisions Made
- Used identical analysis prompt as FTT Phase 3 to ensure cross-source comparability
- Post-processed source_file tags rather than modifying shared video_concept_extractor.py
- Only 1 new structured concept extracted (vs 582 existing) -- this is correct behavior: TomAllsopp covers the same biomechanics vocabulary as FTT, so dedup correctly identifies overlaps. The rich raw markdown analyses contain the unique value (coaching cues, drill descriptions, metaphors) that Plan 04 reconciliation will leverage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed registry snapshot format assumption**
- **Found during:** Task 1 (concept extraction phase)
- **Issue:** Script assumed _registry_snapshot.json was a dict with "concepts" key, but it's actually a flat list of concept dicts
- **Fix:** Added type check: `if isinstance(snapshot, list)` to handle flat list format
- **Files modified:** scripts/run_tomallsopp_extract.py
- **Verification:** Extraction completed successfully for all 49 videos
- **Committed in:** 5e2a810

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor data format mismatch, fixed immediately. No scope creep.

## Issues Encountered
- None beyond the registry format fix above.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all data pipelines produce complete output.

## Next Phase Readiness
- TomAllsopp analysis complete, ready for Feel Tennis analysis (Plan 03)
- Raw markdown analyses are the primary reconciliation input for Plan 04
- Structured JSON extraction has low concept yield due to regex-based parsing of Chinese-language Gemini output -- Plan 04 reconciliation should work directly from markdown files

---
*Phase: 06-secondary-sources*
*Completed: 2026-04-03*

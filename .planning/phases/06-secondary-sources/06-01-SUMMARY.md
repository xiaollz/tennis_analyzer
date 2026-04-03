---
phase: 06-secondary-sources
plan: 01
subsystem: pipeline
tags: [yt-dlp, video-curation, secondary-channels, tomallsopp, feeltennis]

requires:
  - phase: 03-video-extraction
    provides: video_state.py VideoEntry schema, load_state/save_state API
provides:
  - Curated forehand video lists for TomAllsopp (49) and Feel Tennis (46)
  - State JSON files for both channels with all-pending status
  - generate_tomallsopp_state() and generate_feeltennis_state() functions
affects: [06-02, 06-03, 06-04]

tech-stack:
  added: []
  patterns: [parameterized channel state generation via _generate_channel_state helper]

key-files:
  created:
    - knowledge/pipeline/secondary_videos.py
    - knowledge/state/tomallsopp_video_state.json
    - knowledge/state/feeltennis_video_state.json
    - tests/test_secondary_state.py
  modified: []

key-decisions:
  - "Strict forehand-keyword title filtering from yt-dlp enumeration: 49 TomAllsopp + 46 Feel Tennis from 878 total"
  - "Hardcoded curated lists in secondary_videos.py (not dynamic yt-dlp calls) for reproducibility"

patterns-established:
  - "Pattern: _generate_channel_state() factory for any new channel addition"
  - "Pattern: FTT_OVERLAP_IDS and PAID_CONTENT_IDS exclusion sets for cross-channel dedup"

requirements-completed: [SEC-01, SEC-03]

duration: 6min
completed: 2026-04-03
---

# Phase 06 Plan 01: Video List Curation Summary

**Title-keyword curated 95 forehand videos (49 TomAllsopp + 46 Feel Tennis) from 878 total, with state files ready for Gemini API batch processing**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-03T10:59:11Z
- **Completed:** 2026-04-03T11:05:18Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Curated 49 TomAllsopp forehand-technique videos from 307 total channel videos
- Curated 46 Feel Tennis forehand-technique videos from 571 total channel videos
- Cross-channel duplicate (1-g1OD8gh-I) and paid content (CLEjGDGEGaA) excluded
- State JSON files created with VideoEntry schema, all status=pending
- 10 tests covering list size, schema, dedup, exclusions all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Curate video lists and create secondary_videos.py** - `8ad00b3` (test) + `922cf74` (feat)
2. **Task 2: Generate state JSON files for both channels** - `05f8875` (feat)

## Files Created/Modified
- `knowledge/pipeline/secondary_videos.py` - Curated video lists + state generation functions for both channels
- `knowledge/state/tomallsopp_video_state.json` - 49 pending forehand videos from TomAllsopp
- `knowledge/state/feeltennis_video_state.json` - 46 pending forehand videos from Feel Tennis
- `tests/test_secondary_state.py` - 10 tests for list validation, schema, dedup, exclusions

## Decisions Made
- Used strict "forehand" title keyword as primary filter, supplemented by biomechanics terms (kinetic chain, topspin, pronation, etc.) for TomAllsopp
- Excluded volley-only, serve-only, backhand-only, child/beginner, analysis/makeover videos
- Hardcoded curated lists rather than dynamic yt-dlp for reproducibility (per RESEARCH.md anti-pattern guidance)
- TomAllsopp: 49 videos covering technique, kinetic chain, contact point, grip, rotation, lag
- Feel Tennis: 46 videos covering feel-based technique, wrist action, topspin, stance, biomechanics

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all data is real yt-dlp enumeration output, no placeholders.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both state files ready for Plan 02 (TomAllsopp Gemini batch analysis) and Plan 03 (Feel Tennis batch analysis)
- generate_*_state() functions available for re-generation if curation adjustments needed
- 95 total videos at ~20s delay = ~32 min API time, well within budget

---
*Phase: 06-secondary-sources*
*Completed: 2026-04-03*

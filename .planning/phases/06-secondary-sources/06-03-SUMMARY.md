---
phase: 06-secondary-sources
plan: 03
subsystem: knowledge-extraction
tags: [gemini-api, youtube-video-analysis, feeltennis, concept-extraction]

requires:
  - phase: 06-01
    provides: Curated Feel Tennis video list and state file
provides:
  - 14 raw Gemini markdown analyses for Feel Tennis forehand videos
  - 14 structured concept JSON extraction files
  - Updated Feel Tennis video state with accessibility status
affects: [06-04-reconciliation]

tech-stack:
  added: []
  patterns: [extended-proxy-timeout, members-only-detection, source-tag-postprocessing]

key-files:
  created:
    - docs/research/feeltennis_video_analyses/ (14 markdown files)
    - knowledge/extracted/feeltennis_videos/ (14 JSON files)
    - scripts/batch_feeltennis.py
  modified:
    - knowledge/state/feeltennis_video_state.json

key-decisions:
  - "31/46 Feel Tennis videos are members-only (YouTube channel membership required) -- marked as failed with descriptive error"
  - "Extended proxy timeout to 600s (10min) to handle slow video processing after proxy CPU overload"
  - "Source tag feeltennis_video_{id} via post-processing of ftt_video_ prefix in extracted concepts"
  - "1 long video (vfsS9JAAdMc, 1160s) consistently timed out -- marked as proxy_timeout"

patterns-established:
  - "Members-only detection: use yt-dlp --skip-download to verify video accessibility before API calls"
  - "Extended timeout pattern: 600s timeout for video analysis when proxy is under load"

requirements-completed: [SEC-04]

duration: 234min
completed: 2026-04-03
---

# Phase 06 Plan 03: Feel Tennis Video Analysis Summary

**Analyzed 14/15 accessible Feel Tennis forehand videos via Gemini API; 31/46 videos are members-only and inaccessible**

## Performance

- **Duration:** 234 min (mostly waiting for proxy recovery and API rate limits)
- **Started:** 2026-04-03T11:09:55Z
- **Completed:** 2026-04-03T15:03:00Z
- **Tasks:** 1
- **Files modified:** 30

## Accomplishments
- Analyzed 14 out of 15 publicly accessible Feel Tennis forehand videos via Gemini API
- Extracted structured concepts from all 14 analyses (1 new concept, 2 new edges -- most concepts already in 582-node registry)
- Discovered 31/46 videos are YouTube channel members-only content, documented with descriptive error messages
- Created batch processing script with extended timeout and manual retry logic

## Task Commits

Each task was committed atomically:

1. **Task 1: Analyze Feel Tennis videos and extract concepts** - `0bbd674` + `8067d6b` (feat)
   - First commit: 13 videos analyzed + extracted
   - Second commit: 14th video + analysis markdown files + final state

**Plan metadata:** [pending]

## Files Created/Modified
- `docs/research/feeltennis_video_analyses/*.md` (14 files) - Raw Gemini analysis per video
- `knowledge/extracted/feeltennis_videos/*.json` (14 files) - Structured concept extraction per video
- `knowledge/state/feeltennis_video_state.json` - State tracking (14 extracted, 31 members-only, 1 proxy-timeout)
- `scripts/batch_feeltennis.py` - Batch processing script with extended timeout

## Decisions Made
- 31/46 Feel Tennis videos require YouTube channel membership -- these were marked as failed with "members-only" error rather than attempting repeated retries
- Extended proxy timeout from default to 600s to handle post-TomAllsopp-batch proxy CPU overload
- Used yt-dlp accessibility check to separate members-only from genuinely accessible videos
- Source tag uses "feeltennis_video_" prefix (via post-processing) instead of modifying shared extractor code

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Members-only video detection**
- **Found during:** Task 1 (batch analysis)
- **Issue:** 31/46 curated Feel Tennis videos are YouTube channel members-only, causing Gemini API 504/503 errors
- **Fix:** Used yt-dlp --skip-download to identify accessible vs members-only videos; updated state with descriptive errors
- **Files modified:** knowledge/state/feeltennis_video_state.json
- **Verification:** All 15 accessible videos attempted, 14 succeeded
- **Committed in:** 0bbd674, 8067d6b

**2. [Rule 3 - Blocking] Proxy CPU overload after TomAllsopp batch**
- **Found during:** Task 1 (initial analysis attempts)
- **Issue:** Proxy returned "system cpu overloaded" (503) after the parallel TomAllsopp batch of 49 videos saturated resources
- **Fix:** Extended HTTP timeout to 600s, added manual retry with exponential backoff, waited for proxy recovery
- **Files modified:** scripts/batch_feeltennis.py
- **Verification:** 14/15 accessible videos successfully analyzed after proxy recovery

**3. [Rule 1 - Bug] Registry snapshot format mismatch**
- **Found during:** Task 1 (concept extraction phase)
- **Issue:** Registry snapshot is a list, not a dict with "concepts" key; extraction crashed with AttributeError
- **Fix:** Added format detection to handle both list and dict snapshot formats
- **Files modified:** scripts/batch_feeltennis.py (inline extraction code)
- **Verification:** All 14 videos extracted successfully

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** Members-only restriction reduced analyzable videos from 46 to 15. This is an external platform limitation, not a pipeline failure. 14/15 accessible videos (93%) were successfully processed.

## Issues Encountered
- Proxy "system cpu overloaded" after TomAllsopp batch -- resolved by waiting ~30 minutes for recovery
- Video vfsS9JAAdMc (1160s) consistently times out even with 600s client timeout -- likely too long for proxy to process
- Low new concept yield (1 concept, 2 edges from 14 videos) because most Feel Tennis concepts already exist in the 582-node FTT registry

## Known Stubs
None -- all analysis and extraction files contain complete data.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 14 Feel Tennis video analyses ready for reconciliation in Plan 04
- Combined with TomAllsopp's 49 analyses, total secondary source corpus is 63 videos
- Low unique concept count from Feel Tennis (most overlap with existing FTT registry) confirms the FTT-primary approach is correct
- Reconciliation can proceed with available data; members-only gap does not block Plan 04

## Self-Check: PASSED

- Analysis directory: FOUND (14 files)
- Extraction directory: FOUND (14 files)
- State file: FOUND
- Commit 0bbd674: FOUND
- Commit 8067d6b: FOUND
- SUMMARY.md: FOUND

---
*Phase: 06-secondary-sources*
*Completed: 2026-04-03*

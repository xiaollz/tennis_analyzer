---
phase: 03-ftt-video-extraction
plan: 03
subsystem: knowledge-extraction
tags: [gemini-api, video-analysis, concept-extraction, forehand]
dependency_graph:
  requires: [03-01]
  provides: [batch1-analyses, batch1-extractions, batch1-state]
  affects: [03-05, 03-06]
tech_stack:
  added: []
  patterns: [enhanced-table-parser, per-plan-state-slice, checkpointed-batch-processing]
key_files:
  created:
    - scripts/analyze_ftt_batch1.py
    - docs/research/ftt_video_analyses/*.md (12 files)
    - knowledge/extracted/ftt_videos/*.json (12 files)
    - knowledge/state/batch1_state.json
  modified: []
decisions:
  - Enhanced concept extractor to parse Gemini table output format (V-ID rows)
  - Used per-plan state slice (batch1_state.json) to avoid race conditions with parallel batches
  - 20-second delay between API calls for proxy rate limit safety
metrics:
  duration: 33min
  completed: 2026-04-03
  tasks: 1
  files: 26
requirements: [FTT-03]
---

# Phase 03 Plan 03: Batch 1 Forehand Video Analysis Summary

Analyzed 12 forehand-priority FTT videos via Gemini API with proxy, extracting 81 structured concepts and 251 edges using enhanced table-format parser.

## What Was Done

### Task 1: Analyze Batch 1 videos via Gemini API (12 forehand-priority)

Created `scripts/analyze_ftt_batch1.py` that:
1. Verifies proxy base_url at startup (packyapi.com confirmed)
2. Processes 12 forehand technique + biomechanics videos sequentially
3. Saves raw markdown analysis BEFORE extraction (data safety)
4. Checkpoints state after each video (batch1_state.json)
5. Extracts structured concepts via enhanced table parser

**Results:**
- 12/12 videos analyzed successfully (0 failures)
- 12 raw markdown files saved (6-8KB each, total ~85KB)
- 81 concepts extracted, 251 edges created
- Average 6.75 concepts per video

**Videos processed:**
| # | Video ID | Title | Concepts | Edges |
|---|----------|-------|----------|-------|
| 1 | pWzyP-xfLfU | The Secret to Lag is on Your Handle | 8 | 28 |
| 2 | McCb-RfYd0w | Magic of Non-Dominant Side on Forehand | 6 | 15 |
| 3 | JIMgI3jiVns | How Shoulder Rotation Syncs Contact | 5 | 11 |
| 4 | 5KdScDKxVSI | Shoulder Adduction Transform Contact | 7 | 22 |
| 5 | Am8j1Zw5KrE | Shoulder Adduction Unlocks Forehand | 7 | 23 |
| 6 | xLs469ZVMPU | Fix Forehand Over-Rotation - 3 Techniques | 9 | 36 |
| 7 | ExkBtFRhUWY | Magic of Single-Foot Forehand Training | 5 | 10 |
| 8 | wFOy0RKWBTg | Swing OUT on the Forehand (31s short) | 6 | 15 |
| 9 | 5jHCDc44SQM | Abdominal Corkscrew ft. Branstine | 5 | 11 |
| 10 | hNVbbPEob3g | Chest Engagement Controls Racket Face | 7 | 21 |
| 11 | UB6SbA_KX9E | Proper Trunk Sequencing | 9 | 37 |
| 12 | wFIrPMutzRo | 2 Secrets to Rotational Power | 7 | 22 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Enhanced concept extractor for Gemini output format**
- **Found during:** Task 1 verification
- **Issue:** The existing `video_concept_extractor.py` regex patterns (`**Term:**` format) did not match Gemini's actual output format (markdown tables with `| V01-01 | Name EN | Name ZH | ... |` rows)
- **Fix:** Added `extract_from_gemini_analysis()` function in batch script that parses: (a) concept table rows, (b) existing concept references, (c) teaching points, (d) drill methods from section 7
- **Files modified:** scripts/analyze_ftt_batch1.py
- **Commit:** 7d30c8f
- **Impact:** Concepts went from 2 to 81 (40x improvement)

**2. [Rule 3 - Blocking] Missing files in worktree**
- **Found during:** Task 1 startup
- **Issue:** Worktree was on older commit missing knowledge/ directory and .planning/
- **Fix:** Rebased worktree onto latest main (cdb4b87)
- **Impact:** All required pipeline modules available

**3. [Rule 3 - Blocking] Untracked config files not in git**
- **Found during:** Task 1 startup
- **Issue:** `config/youtube_api_config.json` and `docs/knowledge_graph/video_analysis_prompt.md` not committed to git
- **Fix:** Copied from main repo (these contain API keys and should stay untracked)
- **Impact:** Script can load API config and analysis prompt

**4. [Rule 1 - Bug] Chinese characters in concept IDs**
- **Found during:** Re-extraction
- **Issue:** Teaching point and drill names containing Chinese characters failed Pydantic `^[a-z][a-z0-9_]*$` ID validation
- **Fix:** Generate IDs from English-only characters; fall back to `tp_{video_id}_{n}` for all-Chinese names
- **Files modified:** scripts/analyze_ftt_batch1.py

## Decisions Made

1. **Per-plan state slice:** Used `batch1_state.json` instead of the shared `ftt_video_state.json` to avoid race conditions with parallel batch plans (03-04, 03-05)
2. **Enhanced extraction in script:** Added the enhanced Gemini table parser directly in the batch script rather than modifying the shared `video_concept_extractor.py` module, to avoid breaking other plans
3. **Force-add docs:** Used `git add -f` for analysis markdown files since `docs/` is in `.gitignore` but `docs/research/**` is explicitly un-ignored (git needed the force flag for subdirectory creation)

## Known Stubs

None. All 12 videos have real analysis content and non-empty concept extractions.

## Verification Results

```
Raw analyses saved: 12/12
Extraction JSONs: 12/12
batch1_state: 12 extracted, 0 failed
Sample (pWzyP-xfLfU): 8 concepts, 28 edges
Videos with empty concepts: 0
ALL CHECKS PASSED
```

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 7d30c8f | feat(03-03): analyze 12 forehand-priority FTT videos via Gemini API |

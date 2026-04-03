---
phase: 03-ftt-video-extraction
plan: 02
subsystem: knowledge-extraction
tags: [text-processing, concept-extraction, ftt-videos, pydantic, regex]

requires:
  - phase: 03-01
    provides: "video_concept_extractor.py, video_state.py, ConceptRegistry pipeline"
provides:
  - "33 per-video JSON files with structured tennis concepts"
  - "batch0_state.json tracking all 33 existing analyses as extracted"
  - "extract_existing_ftt_videos.py runnable extraction script"
affects: [03-04-registry-merge, 03-05-diagnostic-chains]

tech-stack:
  added: []
  patterns: ["multi-pattern regex extraction", "fallback concept mining", "per-plan state slice"]

key-files:
  created:
    - scripts/extract_existing_ftt_videos.py
    - knowledge/extracted/ftt_videos/{33 video_id}.json
    - knowledge/state/batch0_state.json
  modified: []

key-decisions:
  - "5 extraction patterns: bold-colon terms, lettered sub-sections, coaching cues, standalone bold, bullet-bold"
  - "Fallback extraction from section headers and capitalized multi-word terms when primary patterns yield < 2 concepts"
  - "Per-plan state slice (batch0_state.json) instead of shared ftt_video_state.json to avoid parallel write conflicts"

patterns-established:
  - "Multi-pattern regex extraction: 5 complementary patterns with fallback cascade"
  - "Per-plan state slice: batch0_state.json isolates this plan's state from parallel plans"

requirements-completed: [FTT-02]

duration: 3min
completed: 2026-04-03
---

# Phase 3 Plan 2: Re-extract Structured Concepts from 33 Existing FTT Video Analyses

**242 real tennis concepts extracted from 33 existing video analyses via 5-pattern regex pipeline, replacing shallow video-title placeholders**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-03T05:47:22Z
- **Completed:** 2026-04-03T05:50:17Z
- **Tasks:** 1/1
- **Files created:** 35

## Accomplishments

### Task 1: Extract concepts from 33 existing video analyses

Created `scripts/extract_existing_ftt_videos.py` with enhanced multi-pattern extraction:

1. **Pattern 1 - Bold-colon terms:** `**Term：** description` -- primary technique/biomechanics extraction
2. **Pattern 2 - Lettered sub-sections:** `**A. Title (Subtitle)**` -- forehand variant patterns
3. **Pattern 3 - Coaching cues:** `**"Cue text"**` -- mental models and feel prompts
4. **Pattern 4 - Standalone bold:** `**Capitalized Term**` -- supplementary concepts
5. **Pattern 5 - Bullet-bold:** `* **Term:** desc` -- variant bullet format

Fallback logic: if < 2 concepts extracted, mines section headers then capitalized multi-word terms.

**Results:**
- 33 per-video JSON files created (replacing old shallow 09_ftt_videos_{1,2,3}.json files)
- 242 new concepts (vs. old 4 video-title placeholders)
- 976 co-occurrence edges
- Min 2 concepts per video, avg 7.3 per video
- Registry dedup via ConceptRegistry.resolve() at threshold 70
- Category auto-detection via keyword matching (biomechanics, drill, symptom, mental_model, technique)

| Metric | Value |
|--------|-------|
| Videos processed | 33 |
| New concepts | 242 |
| Edges | 976 |
| Min concepts/video | 2 |
| Max concepts/video | 20 |
| Registry concepts (before) | 297 |
| Zero API calls | Yes |

## Commits

| Hash | Message |
|------|---------|
| 0537318 | feat(03-02): extract structured concepts from 33 existing FTT video analyses |

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all extraction produces real data from analysis text.

## Self-Check: PASSED

- scripts/extract_existing_ftt_videos.py: FOUND
- knowledge/state/batch0_state.json: FOUND
- Per-video JSON files: 36 (33 new + 3 legacy)
- Commit 0537318: FOUND

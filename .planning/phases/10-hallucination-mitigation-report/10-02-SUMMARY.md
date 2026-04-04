---
phase: "10"
plan: "02"
subsystem: report
tags: [diagnostic-journey, report-generation, narrative, backward-compatibility]
dependency_graph:
  requires: [09-02]
  provides: [diagnostic_journey_narrative, journey_report_section]
  affects: [report/report_generator.py]
tech_stack:
  added: []
  patterns: [narrative-generation, optional-section, backward-compatible-extension]
key_files:
  created:
    - tests/test_diagnostic_journey.py
  modified:
    - report/report_generator.py
decisions:
  - "Journey section uses narrative round labels: 初步扫描/针对性观察/根因确认"
  - "Observations capped at 5 per round in narrative to avoid verbosity"
  - "Description text capped at 100 chars per observation for readability"
  - "include_journey parameter defaults to True for backward-compatible opt-out"
  - "Raw IDs (hyp_dc_, obs_r1_) never appear in narrative output"
metrics:
  duration: 3min
  completed: "2026-04-04"
  tasks_completed: 2
  files_changed: 2
---

# Phase 10 Plan 2: Diagnostic Journey Report Section Summary

**One-liner:** Chinese narrative "diagnostic journey" section showing multi-round reasoning process (hypotheses tested, eliminated, confirmed) with backward-compatible report integration.

## What Was Built

### RG-01: Diagnostic Journey Narrative
- `_format_diagnostic_journey()` static method on ReportGenerator
- Takes serialized DiagnosticSession dict, produces List[str] Markdown lines
- Per-round rendering: initial hypotheses, observations with frame refs, hypothesis updates
- Final summary: confirmed/eliminated hypotheses + convergence score

### RG-02: Narrative Style
- Round labels in Chinese: 初步扫描, 针对性观察 (第N轮), 根因确认
- Observations rendered as "frame: description -> judgment" (not raw JSON)
- Hypothesis updates as "确认/排除/调整: name - reason"
- No raw IDs visible to user

### RG-03: Backward Compatibility
- `_vlm_section()` detects `diagnostic_session` key in vlm_result
- Present: renders journey section before root_cause_tree
- Absent: existing v1.0 behavior unchanged
- Legacy issue-list format still works

### RG-04: Optional Journey Section
- `include_journey` parameter (default True) on `_vlm_section()`
- `include_journey=False` suppresses journey section entirely

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1-2  | 79197de | Journey narrative + report integration + backward compat |

## Deviations from Plan

None - plan executed exactly as written.

## Test Results

15 tests passing:
- 6 generation tests (RG-01)
- 3 narrative style tests (RG-02)
- 4 backward compatibility tests (RG-03)
- 2 optional flag tests (RG-04)

## Known Stubs

None.

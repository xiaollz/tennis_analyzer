---
phase: 05-output-generation-vlm-engine
plan: 03
subsystem: vlm-engine
tags: [vlm, two-pass, knowledge-graph, prompt-engineering, diagnostic-chains]

requires:
  - phase: 05-02
    provides: VLMPromptCompiler with compile_pass1_prompt/compile_pass2_prompt API
provides:
  - Two-pass VLM analysis in VLMForehandAnalyzer (Pass 1 symptom scan, Pass 2 targeted diagnostics)
  - Configurable two_pass_enabled flag with single-pass fallback
  - system_prompt parameter on all provider backends
affects: [report-generation, vlm-config]

tech-stack:
  added: []
  patterns: [two-pass-vlm-analysis, provider-agnostic-system-prompt, auto-load-knowledge-graph]

key-files:
  created: []
  modified:
    - evaluation/vlm_analyzer.py
    - tests/test_vlm_prompt.py

key-decisions:
  - "Auto-load graph from knowledge/extracted/ when no explicit graph/chains passed to init"
  - "Pass 1 response parsing uses regex number extraction mapped to priority-sorted chain IDs"
  - "Single-pass fallback on Pass 1 failure or empty detection (graceful degradation)"
  - "system_prompt param added to all 3 provider backends with None default preserving backward compat"

patterns-established:
  - "Two-pass VLM: Pass 1 compact checklist -> Pass 2 targeted diagnostics with graph context"
  - "Provider backends accept optional system_prompt, defaulting to _FTT_SYSTEM_PROMPT"

requirements-completed: [VLM-02]

duration: 6min
completed: 2026-04-03
---

# Phase 5 Plan 3: Two-Pass VLM Integration Summary

**Two-pass VLM analysis: Pass 1 symptom scan with compact checklist, Pass 2 targeted deep analysis with graph-backed diagnostic chains, single-pass fallback for backward compatibility**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-03T08:41:05Z
- **Completed:** 2026-04-03T08:47:10Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Integrated VLMPromptCompiler two-pass system into VLMForehandAnalyzer.analyze_swing
- Pass 1 sends compact symptom checklist (~2K chars), parses VLM response into chain IDs
- Pass 2 sends static coaching prompt + targeted diagnostic context for detected symptoms only
- Single-pass fallback preserves current behavior when knowledge graph unavailable
- Added system_prompt parameter to all 3 provider backends (OpenAI-compatible, Anthropic, Gemini)
- Config flag two_pass_enabled (default True) controls mode selection
- Auto-loads graph and chains from knowledge/extracted/ when not explicitly provided

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for two-pass integration** - `f5a48a2` (test)
2. **Task 1 GREEN: Implement two-pass VLM analysis** - `fd4df28` (feat)

## Files Created/Modified
- `evaluation/vlm_analyzer.py` - Added two-pass VLM analysis flow, _call_vlm dispatcher, _parse_symptom_response, _build_user_text, system_prompt on backends
- `tests/test_vlm_prompt.py` - Added TestTwoPassVLMIntegration with 9 tests covering two-pass flow, single-pass fallback, pass1 parsing, config flag, provider params

## Decisions Made
- Auto-load compiler from default extracted paths when graph/chains not explicitly provided -- reduces integration friction
- Parse pass1 response via regex number extraction (simple, robust) rather than fuzzy symptom name matching
- On pass1 failure or empty detection, fall back to single-pass (not error) -- graceful degradation per research Open Question 1
- system_prompt defaults to None in provider backends, falling through to _FTT_SYSTEM_PROMPT -- zero breaking changes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_single_pass_fallback auto-loading real graph**
- **Found during:** Task 1 GREEN phase
- **Issue:** analyzer_no_compiler fixture was auto-loading the real graph from knowledge/extracted/, causing it to enter two-pass mode when it should test single-pass
- **Fix:** Monkeypatched _try_auto_load_compiler to return None in the no-compiler fixture
- **Files modified:** tests/test_vlm_prompt.py
- **Verification:** test_single_pass_fallback passes with exactly 1 VLM call
- **Committed in:** fd4df28

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test fixture fix only, no scope creep.

## Issues Encountered
- Pre-existing test failure in tests/test_extraction.py (missing source_file key in BbGzWTp5pCM.json) -- unrelated to this plan, not addressed

## Known Stubs
None -- all integration points are wired to real VLMPromptCompiler and KnowledgeGraph implementations.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 5 complete: all 3 plans delivered (template system, prompt compiler, two-pass integration)
- Knowledge graph is now operational end-to-end in VLM analysis pipeline
- Ready for Phase 6+ (report generation, UI integration, etc.)

---
*Phase: 05-output-generation-vlm-engine*
*Completed: 2026-04-03*

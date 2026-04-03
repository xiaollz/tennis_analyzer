---
phase: 07-user-integration
type: research
discovery_level: 0
---

# Phase 7 Research: User Integration

## Discovery Level: 0 (Skip)

All work follows established codebase patterns. No new external dependencies needed.

## Current State Analysis

### What Exists

1. **User journey extraction** (`knowledge/extracted/user_journey/learning.json`):
   - 66 concepts, 130 edges from 22 training sessions (2026-03-03 to 2026-04-01)
   - Quality is LOW: many concepts are malformed (`c_unnamed`, `n6`, `tom_pronation` as category "symptom")
   - Edge types limited to `causes` and `supports`
   - Concept IDs don't link to canonical registry (e.g., `v_scooping` vs registry's `scooping_motion`)

2. **Canonical registry**: 732 concepts, 869+ edges, 18 diagnostic chains
   - Well-structured with English+Chinese names, VLM features, muscle mappings
   - User journey concepts were bulk-inserted in Phase 2 but never reconciled

3. **VLM two-pass system** (`evaluation/vlm_analyzer.py`):
   - Pass 1: symptom checklist scan (~2K chars)
   - Pass 2: static coaching prompt + dynamic diagnostic chains (~10K budget)
   - `VLMPromptCompiler.compile_pass2_prompt(detected_chain_ids)` is the injection point
   - No user context awareness currently

4. **Training records** (`docs/record/learning.md`): 1332 lines, 22+ sessions
   - Rich structured data: causal chains, FTT theory links, breakthroughs, cues
   - Each session has date, discoveries, problems, progress notes

### Architecture Approach

**USER-01: Link records to graph**
- Don't re-extract all 66 concepts. Instead, build a `UserProfile` that:
  - Maps each training session to relevant canonical concept IDs (via registry.resolve())
  - Tracks per-concept status: `struggling`, `improving`, `mastered`, `regressed`
  - Records breakthrough dates and key cues per concept
- Output: `knowledge/extracted/user_journey/user_profile.json`

**USER-02: Personalized VLM diagnostics**
- Add a `compile_user_context(profile, detected_chain_ids)` method to VLMPromptCompiler
- Injects 3 things into Pass 2 prompt:
  1. User's current struggle areas (concept IDs + status)
  2. Recent breakthroughs (last 3 sessions)
  3. Known recurring issues (concepts that regressed)
- New Jinja2 template: `templates/vlm/user_context.j2`
- Budget: ~1.5K chars for user context within the 10K dynamic budget

**USER-03: Training plan generation**
- New module: `knowledge/output/training_plan.py`
- Inputs: UserProfile + KnowledgeGraph + DiagnosticChains
- Logic:
  1. Find concepts where user status is `struggling` or `regressed`
  2. Walk graph to find `drills_for` edges targeting those concepts
  3. Prioritize drills by: (a) how many issues they address, (b) chain priority
  4. Generate ordered drill plan with rationale
- Output: Markdown training plan

### Key Design Decisions

1. **UserProfile is a separate artifact**, not embedded in the graph — keeps the knowledge graph source-agnostic
2. **Session-to-concept mapping uses fuzzy resolve** against the 732-node registry, not the malformed Phase 2 extraction
3. **User context is budget-constrained** (~1.5K chars) to avoid pushing Pass 2 over 10K
4. **Training plan is a standalone generator**, not part of VLM — it runs independently of video analysis

### Risk Assessment

- LOW risk: All patterns established, no new dependencies, small surface area
- Only concern: fuzzy matching quality for Chinese training notes against English-primary registry
  - Mitigation: Use bilingual concept names already in registry (`name` + `name_zh`)

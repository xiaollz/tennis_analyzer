---
phase: "09"
plan: "02"
title: "Hypothesis Tracker + Cross-Causal Reasoning + MultiRoundAnalyzer Integration"
subsystem: vlm-multi-round
tags: [hypothesis-tracking, confidence-scoring, cross-causal-reasoning, progressive-narrowing]
dependency_graph:
  requires: [compile_observation_directive, compile_confirmation_prompt, knowledge-graph, diagnostic-chains]
  provides: [confidence-scoring, cross-hypothesis-reasoning, knowledge-driven-diagnostic-rounds]
  affects: [multi-round-analyzer, diagnostic-session]
tech_stack:
  added: []
  patterns: [observation-confidence-scaling, causal-graph-traversal, auto-threshold-triggers]
key_files:
  created: []
  modified:
    - evaluation/vlm_analyzer.py
    - tests/test_multi_round.py
decisions:
  - "Confidence deltas: +0.15 support, -0.20 contradict, scaled by observation confidence"
  - "Auto-eliminate threshold 0.15, auto-confirm threshold 0.85"
  - "Cross-hypothesis reasoning uses forward causal traversal from confirmed root cause"
  - "Progressive narrowing tracked via status snapshots, logged when round makes no progress"
metrics:
  duration: "8min"
  completed: "2026-04-04"
  tasks_completed: 3
  tasks_total: 3
---

# Phase 9 Plan 2: Hypothesis Tracker + Cross-Causal Reasoning + MultiRoundAnalyzer Integration Summary

Observation-based confidence scoring, cross-hypothesis causal reasoning via knowledge graph, and full integration into MultiRoundAnalyzer with knowledge-driven directives replacing static pass2 prompts.

## What Was Built

### Confidence Scoring (HT-02)
- Supporting observations (judgment=YES): +0.15 * observation confidence
- Contradicting observations (judgment=NO): -0.20 * observation confidence
- Unclear observations: no change
- Auto-eliminate when confidence drops below 0.15
- Auto-confirm when confidence rises above 0.85
- Observations linked to hypotheses via directive_source field

### Cross-Hypothesis Causal Reasoning (HT-03)
- After each round, confirmed hypotheses are checked against the knowledge graph
- Forward causal traversal via `_get_downstream_concepts()` follows outgoing 'causes' edges
- If an active hypothesis's root_cause_concept_id is downstream of a confirmed root cause, it's auto-eliminated
- Graceful no-op when graph is None (backward compatible)

### Progressive Narrowing (HT-04)
- Status snapshots track hypothesis changes between rounds
- Warning logged when consecutive rounds show no status changes
- Existing convergence stagnation detection (3 identical snapshots) provides hard enforcement

### MultiRoundAnalyzer Integration
- Diagnostic rounds now use `compile_observation_directive()` instead of `compile_pass2_prompt()`
- Final/convergence rounds use `compile_confirmation_prompt()`
- Falls back to `compile_pass2_prompt()` if directive methods unavailable (backward compatible)
- Observation scoring applied before VLM hypothesis updates (both contribute)
- Cross-hypothesis reasoning runs after each round's updates
- checked_steps tracked in session as directives are generated
- Graph parameter accepted by MultiRoundAnalyzer constructor

### Tests
12 new tests: 6 for confidence scoring (support/contradict/unclear/auto-eliminate/auto-confirm/scaling), 3 for cross-hypothesis reasoning (upstream eliminates downstream, no false elimination, no crash without graph), 3 for integration (observation directive used, confirmation used, progressive narrowing).

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None -- all methods fully implemented.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1-3 | 7eec339 | All tasks committed together (tracker + integration + tests) |

## Self-Check: PASSED

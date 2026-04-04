---
phase: "09"
plan: "01"
title: "Observation Directive Templates + compile_observation_directive()"
subsystem: knowledge-vlm-prompts
tags: [jinja2, vlm-prompts, observation-directives, confirmation]
dependency_graph:
  requires: [knowledge-graph, diagnostic-chains, hypothesis-models]
  provides: [compile_observation_directive, compile_confirmation_prompt, observation_directive_template, confirmation_template]
  affects: [multi-round-analyzer]
tech_stack:
  added: []
  patterns: [jinja2-template-rendering, budget-enforcement, directive-generation]
key_files:
  created:
    - knowledge/templates/vlm/observation_directive.j2
    - knowledge/templates/vlm/confirmation.j2
  modified:
    - knowledge/output/vlm_prompt.py
    - tests/test_vlm_prompt.py
decisions:
  - "One directive per hypothesis per round to keep prompts focused and under 4K budget"
  - "Default frame 图2-4 when DiagnosticChain.vlm_frame is None"
  - "Causal chain summaries use longest path for most informative display"
metrics:
  duration: "5min"
  completed: "2026-04-04"
  tasks_completed: 4
  tasks_total: 4
---

# Phase 9 Plan 1: Observation Directive Templates + compile_observation_directive() Summary

Jinja2 templates and VLMPromptCompiler methods for knowledge-driven per-round observation directives and final confirmation prompts.

## What Was Built

### Templates
- **observation_directive.j2**: Renders per-round targeted VLM prompts with hypothesis context (name_zh, confidence), frame-specific observation questions mapped from DiagnosticStep.check + Concept.vlm_features, and strict JSON response format for observations + hypothesis_updates.
- **confirmation.j2**: Renders final confirmation round prompt with full evidence summary grouped by hypothesis, hypothesis status display, causal chain summaries, and root_cause_tree JSON output format (v1.0 compatible).

### Compiler Methods
- **compile_observation_directive(session, round_number)**: Generates < 4K char prompt from active hypotheses and unchecked diagnostic steps. Maps DiagnosticStep.check + Concept.vlm_features + DiagnosticChain.vlm_frame into specific VLM questions (KD-04). One directive per hypothesis per round for focus.
- **compile_confirmation_prompt(session)**: Generates < 6K char final round prompt with evidence summary and root_cause_tree generation instructions.
- **_generate_directives()**: Internal helper that converts active hypotheses + unchecked steps into directive dicts with frame, question_zh, vlm_features.
- **_build_causal_chain_summaries()**: Internal helper for confirmed hypothesis causal path display.

### Tests
13 new tests covering template rendering, budget enforcement, directive generation from unchecked steps, VLM feature mapping, and edge cases.

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None -- all methods are fully implemented with real template rendering.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1-4 | 5799d0a | All tasks committed together (templates + methods + tests) |

## Self-Check: PASSED

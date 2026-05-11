# Integration Changelog

## 2026-04-07 — v4.1 Recording protocol + report 5-layer support

### Files created
- `docs/record/recording_protocol.md` — user-facing 训练录像协议 (v4.1). Distilled from `camera_angles_methodology.md`. Covers main rig (斜后 45°, 1.1m, 5–6m, 60fps), secondary rig (正侧面 weekly), common mistakes, and a check-list.

### Files modified
- `report/report_generator.py`
  - Added `_format_layer()` static helper mapping `L1..L5` / `L*_NAME` → Chinese labels.
  - `_vlm_section()`: prominently displays `根因层级：Lx ...` when `root_cause_tree.layer` (or `vlm_result.root_cause_layer`) is present.
  - New "准备阶段诊断" block surfaces only when root layer is L4/L5; uses `tree.preparation_notes` or `vlm_result.preparation_diagnosis` if provided, otherwise emits a generic guidance line.
  - Raw VLM answers `<details>` block now groups by layer using `vlm_result.question_layers` (or `raw_answer_layers`) — falls back to flat list if mapping absent (backward compatible).

### Test results
- `pytest tests/ -k report` → **11 passed, 0 failed**.
- Full suite has 16 unrelated pre-existing failures (extraction JSON missing, kpi registry, vlm_prompt) — none touch report_generator. Confirmed by inspecting failure stacks.
- `python -c "from report.report_generator import *; print('OK')"` → OK.

### Backward compatibility
- All new behavior is gated on optional dict keys (`layer`, `question_layers`, `preparation_notes`, etc.). If diagnosis output omits these fields, the report renders exactly as before.

### Notes / open items
- The diagnosis engine still needs to actually populate `layer` / `question_layers` for the new sections to fire. Until then the report falls back to v4.0 layout — no breakage.

---

## 2026-04-07 — v4.2 Preparation/Footwork concepts + top-down diagnostic flow

Companion to the report-side v4.1 work above. The diagnosis engine now actually emits layer-tagged root causes, so the L4/L5 report blocks added in v4.1 will fire.

### Files created
- `knowledge/extracted/preparation_footwork_concepts.json` — 32 new concept nodes + 35 causal edges. Each node carries `layer` (L1-L5) and cites the source research doc.

### Files modified
- `evaluation/diagnosis_engine.py`
  - `OBSERVATION_TO_CONCEPT`: +32 keyword rules for prep/footwork problems.
  - `_CONCEPT_LAYER` dict + `_LAYER_ORDER` + `_get_concept_layer()` — maps every diagnostic concept (existing and new) to L1..L5.
  - `_find_earliest_layer_problem()` — top-down "earliest layer wins" helper.
  - `_trace_root_causes()` now calls `_find_earliest_layer_problem()` first; falls back to legacy frequency+severity logic for unlabeled concepts. Backward compatible.
  - `_load_graph_data()` merges in `preparation_footwork_concepts.json` so name lookups + forward edges resolve.
  - `_CONCEPT_TO_FIX`: +32 single-cue, on-court drills (Tom Allsopp / Tomaz / FTT).
  - `_CONCEPT_TO_MUSCLE`: +10 muscle activation cues for the most load-bearing prep concepts (split, pivot, late unit turn, arm-only turn, scapular glide, left shoulder forward, hip-shoulder separation, triple bend, X stretch).
  - `Q_DIRECT_MAPPING`: +Q21..Q35 covering split timing/form/missing, pivot, footwork, recovery, unit turn timing, arm-only vs whole-body, left hand position, hip-shoulder separation, stop-start, racket-back-by-bounce, triple bend, stance width.
  - `_CONCEPT_TO_RECURRING_KEYWORDS`: +prep concepts.

### New concept IDs (32, layer-tagged)
- L5 Footwork (13): `prep01_late_split_step`, `prep02_no_split_step`, `prep03_split_step_too_high`, `prep04_no_pivot`, `prep05_late_pivot`, `prep06_choppy_steps`, `prep07_no_recovery_step`, `prep19_narrow_stance`, `prep20_no_triple_bend`, `prep21_no_landing_in_motion`, `prep22_neutral_to_semi_open_failed`, `prep25_no_multi_split_on_machine`, `prep29_late_first_step`
- L4 Preparation (13): `prep08_late_unit_turn`, `prep09_arm_only_unit_turn`, `prep10_no_scapular_glide`, `prep11_racket_head_dropped_early`, `prep12_left_hand_dropped_in_unit_turn`, `prep13_left_shoulder_not_forward`, `prep15_fake_separation`, `prep23_no_press_slot`, `prep24_backswinging_not_placing`, `prep26_arm_used_for_balance`, `prep27_no_x_stretch`, `prep28_grip_not_innervated`, `prep30_late_preparation_general`
- L3 Chain (1): `prep14_insufficient_hip_shoulder_separation`
- L2 Rhythm (3): `prep16_stop_start_syndrome`, `prep17_prep_not_done_by_bounce`, `prep18_no_wait_after_prep`
- L1 Contact downstream (2): `prep31_cramped_swing_no_extension`, `prep32_forced_rush_contact_behind`

### New causal chains
- Late split step → late first step → late pivot → late unit turn → prep not done by bounce → forced rush contact behind body
- Arm-only unit turn → no scapular glide → kinetic chain break / forearm compensation
- No pivot → arm-only unit turn (and downstream)
- Insufficient hip-shoulder separation → kinetic chain break (no torque → no power)
- Backswinging instead of placing → racket head dropped early → no press slot → kinetic chain break
- No triple bend → kinetic chain break
- Stop-start syndrome → arm compensation
- Late pivot → cramped swing → no forward extension

### Verification
- Smoke run: `python -c "import evaluation.diagnosis_engine as de; de.diagnose({'raw_answers': {'Q28': 'unit turn 启动得很晚...', 'Q29': '只用手臂引拍...'}}, {'arm_torso_synchrony': 0.3})"` → root cause = `prep08_late_unit_turn` (L4), correctly chosen over `prep09_arm_only_unit_turn` (L4 too) by severity, and causal chain points back to the L5 split-step parent.
- `python -m pytest tests/ -q` → **341 passed, 16 failed**. All 16 failures verified as preexisting via `git stash` baseline run (missing JSON for two newer research docs, registry/extraction tests scanning only categorized subdirs, and unrelated v2 KPI / VLM prompt drift). None caused by these changes.

### Backward compatibility
- All existing concept IDs and behaviors retained.
- New top-down strategy only fires when matched concepts have layer attributes; otherwise the original frequency+severity logic runs unchanged.
- Graph loader is additive; missing prep file is silently tolerated.


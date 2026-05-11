# VLM Prompt Changelog — 5-Layer Restructure

Date: 2026-04-07
File: `knowledge/templates/vlm/system_prompt.md.j2`

## Summary
Reorganized the VLM observation prompt from 6 phase-based blocks (SYNC / SHOULDER_TORSO / ARM_RACKET / LEFT_HAND / LOWER_BODY / DYNAMICS, Q1-Q20) into a 5-layer diagnostic structure (L1 CONTACT → L2 RHYTHM → L3 KINETIC CHAIN → L4 PREPARATION → L5 FOOTWORK) with 32 questions total.

**Compatibility**: All Q1-Q20 numbering and semantics are preserved. The diagnosis engine's hard-coded mappings (Q1, Q3, Q4, Q5b, Q6, Q7, Q8, Q9, Q11, Q12, Q13, Q14, Q16, Q17, Q19) continue to work unchanged. Q21-Q32 are new additions covering preparation cues and footwork that did not exist before.

## All 32 questions (mapping table)

| Q   | Layer | Concept                              | Source coach              | Concept ID guess          |
| --- | ----- | ------------------------------------ | ------------------------- | ------------------------- |
| Q10 | L1    | Racket face at contact               | FTT                       | contact_racket_face       |
| Q3  | L1    | Arm-chest gap (contact distance)     | FTT (contact prerequisite)| contact_distance          |
| Q6  | L1    | Torso lean at contact                | Tomaz                     | posture_lean              |
| Q20 | L1    | Finish balance                       | FTT                       | finish_balance            |
| Q18 | L2    | Overall rhythm continuity            | Tomaz / Ian               | rhythm_continuity         |
| Q1  | L2    | Arm-body sync                        | FTT / Rick Macci          | arm_body_sync             |
| Q2  | L2    | Sync breakdown phase                 | Ian Westermann            | sync_breakdown            |
| Q19 | L2    | Trunk decel around contact           | Biomech / Macci           | trunk_decel               |
| Q11 | L2    | Arm direction after contact          | FTT                       | followthrough_path        |
| Q17 | L3    | First mover (kinetic chain root)     | Macci                     | chain_first_mover         |
| Q5b | L3    | Hip-shoulder separation              | FTT / Tom Allsopp         | hip_shoulder_separation   |
| Q14 | L3    | Knee bend (loading)                  | Macci                     | leg_load                  |
| Q15 | L3    | Weight transfer                      | FTT                       | weight_transfer           |
| Q16 | L3    | Back foot at contact                 | Tom Allsopp               | back_foot_action          |
| Q4  | L4    | Shoulder level (right shoulder drop) | FTT                       | shoulder_drop             |
| Q5  | L4    | Shoulder rotation depth              | FTT                       | unit_turn_depth           |
| Q7  | L4    | Body facing at contact               | FTT                       | body_facing_contact       |
| Q8  | L4    | Hand drop magnitude                  | Tomaz                     | hand_drop                 |
| Q9  | L4    | Wrist trajectory shape               | Nikola                    | swing_path_shape          |
| Q12 | L4    | Left hand prep position              | Tomaz                     | left_hand_prep            |
| Q13 | L4    | Left hand action through contact     | FTT (pull off-arm)        | left_hand_action          |
| Q21 | L4    | Unit turn START vs ball              | Tomaz (parallel proc)     | prep_unit_turn_start      |
| Q22 | L4    | Unit turn FINISH vs ball             | Tomaz ("early then wait") | prep_unit_turn_finish     |
| Q23 | L4    | Left shoulder leads (vs right pulls) | Tomaz                     | prep_left_shoulder_leads  |
| Q24 | L4    | Racket "hold up" vs droop            | FTT YouTube binary cue    | prep_racket_hold_up       |
| Q25 | L4    | Jersey wrinkles (scapular glide)     | FTT YouTube binary cue    | prep_scapular_glide       |
| Q26 | L4    | "Place then pull" vs continuous      | FTT YouTube binary cue    | prep_place_pull           |
| Q27 | L5    | Split step visible                   | Tom Allsopp               | foot_split_visible        |
| Q28 | L5    | Split landing vs opp contact (±frames)| Tom Allsopp ("land as they hit") | foot_split_timing  |
| Q29 | L5    | First foot to move + direction       | Tom Allsopp               | foot_first_move           |
| Q30 | L5    | Right foot pivot (heel up, degrees)  | Tom Allsopp               | foot_right_pivot          |
| Q31 | L5    | Stance type at contact               | Tom Allsopp ATP/WTA       | foot_stance_type          |
| Q32 | L5    | Steps split→contact + recovery step  | Tom Allsopp               | foot_steps_recovery       |

Total: 32 questions (within ≤32 budget). 20 reused (Q1-Q20), 12 new (Q21-Q32 — Q21-Q26 prep, Q27-Q32 footwork).

## Parser changes
File: `evaluation/vlm_analyzer.py` → `_parse_observation_response()`

1. Regex updated from `^Q(\d+):` to `^Q(\d+[a-z]?):` so Q5b (and any future Q#letter suffixes) are captured. This was a latent bug — Q5b was in the prompt but never reached `raw_answers`.
2. Added two new frame groups to the returned `frames` dict:
   - `preparation` — keys: unit_turn_start_vs_ball, unit_turn_finish_vs_ball, left_shoulder_leads, racket_hold_up, scapular_glide_jersey, place_then_pull (Q21-Q26)
   - `footwork` — keys: split_step_visible, split_landing_vs_opp_contact, first_foot_to_move, right_foot_pivot, stance_at_contact, steps_and_recovery (Q27-Q32)
3. All existing frame groups (sync, shoulder_torso, arm_racket, left_hand, lower_body, dynamics) untouched.
4. `raw_answers` now contains Q1-Q20 + Q5b + Q21-Q32 = up to 33 entries; diagnosis engine consumes via `raw_answers.get("Q##")` so new questions are immediately available for new concept rules.

## Smoke test
A fake response in the new 5-layer format was fed through `_parse_observation_response()`:
- format = `observation_v2`
- 33 raw answers parsed (including Q5b, Q21, Q32)
- New `preparation` and `footwork` frame groups populated correctly
- Pre-existing `shoulder_torso.hip_follow_shoulder` (← Q5b) now correctly populated for the first time

## Files modified
- `knowledge/templates/vlm/system_prompt.md.j2` — full rewrite into 5-layer structure
- `evaluation/vlm_analyzer.py` — regex fix + 2 new frame groups in `_parse_observation_response`
- `docs/research/coach_analysis/vlm_prompt_changelog.md` — this file (new)

## Concerns
1. **Output token budget**: 32 questions × ~30 chars Chinese answer ≈ 960 chars, well within VLM output limits, but on a slow VLM the longer prompt may increase latency. Monitor.
2. **Concept IDs above are guesses** — coordinate with the diagnosis engine extension agent to confirm/rename. The `raw_answers["Q21"]`..`raw_answers["Q32"]` are the stable contract.
3. **Q21/Q22/Q28 ball-relative timing** depends on the VLM being able to identify opponent contact / ball-over-net / ball-bounce events in the clip. For clips that only show the player (no opponent or ball trajectory visible), VLM should answer "看不清" — the prompt rule #6 instructs this.
4. **Camera-angle dependence**: L5 footwork questions assume side or rear-side angle. Front-on clips will produce many "看不清" answers. The existing `camera_angle_hint` Jinja variable in the prompt template was NOT preserved because the original template did not contain Jinja blocks — it was a plain `.md.j2` with no variables. If a hint is later added, it should be inserted at the top.

# rear 2026-10-19 正后方 OBS 补测（fill，三跑多数）

> 模型 gemini-3.8-flash，inline_data；提示词 `pipeline/prompts/facts_rear_fill_v1.md`（check_neutrality 0 FAIL）。脚本 `/workspace/today_swing/analyze_rear_obs_fill.py`。
> 规则：≥2 次一致且非 unclear 才成立。**run1 = HTTP 429 → DRIFT，跳过，没编结果**；所以实际只有 2 个有效跑次，成立 = 2/2 一致。

| OBS | 题干（obs_catalog 原文） | run1 | run2 | run3 | 多数 | 同意次数 | 成立否 |
|---|---|---|---|---|---|---|---|
| OBS-10a | Back foot: (a) is the heel off the ground at contact; (b) within 0.5 s after contact does the foot swing around to the front, kick backward, or stay in place? [part (a)] | DRIFT(429) | heel_down_at_contact | heel_up_at_contact | 无多数 | 1/2 | 不成立 |
| OBS-10b | Back foot: (a) is the heel off the ground at contact; (b) within 0.5 s after contact does the foot swing around to the front, kick backward, or stay in place? [part (b)] | DRIFT(429) | stays_in_place | stays_in_place | stays_in_place | 2/2 | 成立 |
| OBS-04 | At the top of the backswing, does the racket stop for a noticeable moment (about 0.3 s or more) before the forward swing starts? | DRIFT(429) | pause_present | pause_present | pause_present | 2/2 | 成立 |
| OBS-05 | Between the ball leaving the machine/opponent and contact, does the player show a split step and small adjusting steps with the feet? | DRIFT(429) | no_steps_stands_still | few_adjust_steps | 无多数 | 1/2 | 不成立 |
| OBS-15a | Follow-through: (a) where does the racket finish relative to the non-hitting shoulder; (b) after contact does the hand extend toward the target for a stretch or wrap back to the body immediately; (c) at the end, is the hitting elbow in front of the chest? [part (a)] | DRIFT(429) | finish_above_non_hitting_shoulder | finish_above_non_hitting_shoulder | finish_above_non_hitting_shoulder | 2/2 | 成立 |
| OBS-15b | Follow-through: (a) where does the racket finish relative to the non-hitting shoulder; (b) after contact does the hand extend toward the target for a stretch or wrap back to the body immediately; (c) at the end, is the hitting elbow in front of the chest? [part (b)] | DRIFT(429) | wraps_back_to_body_immediately | wraps_back_to_body_immediately | wraps_back_to_body_immediately | 2/2 | 成立 |
| OBS-15c | Follow-through: (a) where does the racket finish relative to the non-hitting shoulder; (b) after contact does the hand extend toward the target for a stretch or wrap back to the body immediately; (c) at the end, is the hitting elbow in front of the chest? [part (c)] | DRIFT(429) | elbow_in_front_of_chest_at_end | elbow_in_front_of_chest_at_end | elbow_in_front_of_chest_at_end | 2/2 | 成立 |
| OBS-38 | Width of stance between feet at the start of forward swing relative to shoulder width | DRIFT(429) | shoulder_width_or_wider | shoulder_width_or_wider | shoulder_width_or_wider | 2/2 | 成立 |
| OBS-33 | At the contact frame, how far is the ball from the hitting-side hip, side to side? | DRIFT(429) | more_than_one_forearm_length | more_than_one_forearm_length | more_than_one_forearm_length | 2/2 | 成立 |
| OBS-36 | Between the start of the forward swing and contact, does the hitting hand first move outward away from the body, or directly toward the ball? | DRIFT(429) | directly_toward_ball | directly_toward_ball | directly_toward_ball | 2/2 | 成立 |
| OBS-32 | During the forward swing, does the hitting elbow pull linearly across the torso before arm extension? | DRIFT(429) | elbow_drags_linearly | rotational_drive | 无多数 | 1/2 | 不成立 |
| OBS-02 | At the contact frame, how far is the hitting hand/wrist from the side of the torso: upper arm touching the ribs, or a visible gap? | DRIFT(429) | visible_gap | visible_gap | visible_gap | 2/2 | 成立 |
| OBS-07a | Non-hitting arm: (a) when the shoulder turn is completed, does it reach across the front of the body; (b) does it leave the racket before or after the turn is completed; (c) at the start of the forward swing, has it already dropped to the side or does it rotate with the chest? [part (a)] | DRIFT(429) | does_not_reach_across | reaches_across_body_front | 无多数 | 1/2 | 不成立 |
| OBS-07b | Non-hitting arm: (a) when the shoulder turn is completed, does it reach across the front of the body; (b) does it leave the racket before or after the turn is completed; (c) at the start of the forward swing, has it already dropped to the side or does it rotate with the chest? [part (b)] | DRIFT(429) | leaves_racket_before_turn_complete | leaves_racket_after_turn_complete | 无多数 | 1/2 | 不成立 |
| OBS-07c | Non-hitting arm: (a) when the shoulder turn is completed, does it reach across the front of the body; (b) does it leave the racket before or after the turn is completed; (c) at the start of the forward swing, has it already dropped to the side or does it rotate with the chest? [part (c)] | DRIFT(429) | already_dropped_to_side | rotates_back_with_chest | 无多数 | 1/2 | 不成立 |
| OBS-16 | Result of each shot (spoken by the player or visible from where the ball lands / how it is struck). | DRIFT(429) | unclear | good | 无多数 | 1/2 | 不成立 |
| OBS-06 | At the start of the forward swing (after the feet are set), how are the feet oriented relative to the incoming ball direction? | DRIFT(429) | open | closed | 无多数 | 1/2 | 不成立 |
| OBS-08 | At the end of the preparation turn (before the hand starts to drop under the ball), how far are the shoulder line and hip line rotated away from the target? | DRIFT(429) | small_turn | clearly_sideways | 无多数 | 1/2 | 不成立 |

成立 9 题 / 共 18 题；多数为 unclear 0 题（单跑出现 unclear：OBS-16 run2）；其余 9 题两跑分歧，无多数。

原始输出：`/workspace/today_swing/rear_obs_fill/run{2,3}_gemini-3.8-flash.md`、`run1_DRIFT.txt`。

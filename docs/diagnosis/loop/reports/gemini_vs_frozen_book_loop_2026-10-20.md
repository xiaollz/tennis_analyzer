# Gemini 三跑 vs frozen（book-loop 2026-10-20）

模型：gemini-3.8-flash · 脚本：analyze_rear_obs_trunk.py · 未改 frozen。

| OBS | frozen | run1 | run2 | run3 | majority | vs frozen |
|---|---|---|---|---|---|---|
| OBS-18 | arm_first | unclear | arm_first | arm_first | arm_first (2/3) | **CONFIRMED** |
| OBS-25 | shoulders_lead_hips | unclear | shoulders_lead_hips | shoulders_lead_hips | shoulders_lead_hips (2/3) | **CONFIRMED** |
| OBS-09 | square_or_past | square_or_past | square_or_past | still_clearly_sideways | square_or_past (2/3) | **CONFIRMED** |
| OBS-08 | small_turn | clearly_sideways | small_turn | clearly_sideways | clearly_sideways (2/3) | **DRIFT** |
| OBS-40 | unclear | unclear | unclear | unclear | unclear (3/3) | **CONFIRMED** |
| OBS-19 | unclear | unclear | unclear | unclear | unclear (3/3) | **CONFIRMED** |
| OBS-71 | unclear | unclear | unclear | unclear | unclear (3/3) | **CONFIRMED** |
| NEW-SHOULDER_VS_HIP_PREP | shoulders_farther_than_hips | shoulders_farther_than_hips | shoulders_farther_than_hips | shoulders_farther_than_hips | shoulders_farther_than_hips (3/3) | **CONFIRMED** |

## 处理
- **不自动改 frozen / golden**。
- OBS-08：本轮 majority=`clearly_sideways`（2/3）与 frozen=`small_turn` 冲突 → 交人复核。
- OBS-18/25：run1 标 unclear，run2/3 与 frozen 一致 → 按 2/3 CONFIRMED；run1 soft DRIFT 记档。
- OBS-09：run3=`still_clearly_sideways` 少数漂移；majority 与 frozen 一致。

## 描述跑（不进事实）
- `/workspace/today_swing/rear_desc_loop_book_20261020.md`
- 教练观感把「脚钉死」当根因——与 trunk 已撤回的 feet_locked_first_priority 冲突；**只作观感，不写进 facts/frozen**。

# rear 2026-10-19 正后方 trunk OBS 补测（三跑多数）

> 模型：gemini-3.8-flash × 3；提示词 `pipeline/prompts/facts_rear_trunk_obs_v1.md`（neutrality 0 FAIL）。
> 视频：`/workspace/today_swing/rear_20261019_small.mp4`。不夹用户结论 / FLT 名。
> 采纳规则：≥2/3 一致才采纳；否则标不一致。

| 题 | 采纳值 | 票数 | 状态 |
|---|---|---:|---|
| OBS-18 | `arm_first` | 3/3 | 采纳 |
| OBS-25 | `shoulders_lead_hips` | 3/3 | 采纳 |
| OBS-40 | `unclear` | 3/3 | 采纳=看不清（机位限制） |
| OBS-09 | `square_or_past` | 3/3 | 采纳 |
| OBS-08 | `small_turn` | 2/3 | 采纳 |
| OBS-19 | `unclear` | 3/3 | 采纳=看不清（机位限制） |
| OBS-71 | `unclear` | 3/3 | 采纳=看不清（机位限制） |
| NEW-CANDIDATE shoulder_vs_hip_prep | `shoulders_farther_than_hips` | 3/3 | 采纳 |

## 解读（相对动力链主干，非新编画面）

- **OBS-18=arm_first（3/3）**：前挥启动时手臂先动 → 支持断点 C（臂独立发力）的**直接证据**（此前缺失）。
- **OBS-25=shoulders_lead_hips（3/3）**：肩带先转、骨盆几乎不动 → 支持「不是髋先扭转传力」；与用户「髋扭转不足」方向一致。
- **OBS-09=square_or_past（3/3）**：开放站位下髋线触球时已朝网。**注意**：这不等于「髋从侧对成功转回」；结合 OBS-25，更像髋全程偏开、未先加载扭转。
- **OBS-08**：2/3=`small_turn`，1/3=`clearly_sideways` → 采纳 **small_turn**（准备转肩幅度不大）。
- **NEW 肩线 vs 髋线=shoulders_farther_than_hips（3/3）**：准备末肩比髋转得更多。结合髋朝网、肩微转：是「上身有一点转」、**不是** FTT 说的「坐进后髋的 coil」。记为 NEW-OBS 候选，**未升 catalog**。
- **OBS-40 / OBS-19 / OBS-71 = unclear（3/3）**：正后方必然看不清（胸平面遮挡 / 帧率不足）。如实保留。

原始三跑：`/workspace/today_swing/rear_obs_trunk/run{1,2,3}_gemini-3.8-flash.md`


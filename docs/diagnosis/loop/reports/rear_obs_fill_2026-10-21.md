# 正后方 OBS 补测（rear fill）— 2026-10-21

> 固定样本 `rear_20261019_small.mp4`（正后方，5 拍右手正手）。分支 `cursor/rear-obs-fill-2026-10-21`，基于 main 468c979。
> **frozen / golden 都没改**；新事实只写进候选快照 `user_case/rear_20261019_facts_candidate_fill.json`。不覆盖 tree.json v1 / tree_vN。

## 1. PASS/FAIL 总表

| 检查 | 结果 |
|---|---|
| `check_neutrality.py prompts/facts_rear_fill_v1.md`（以及全部 6 份 facts_* 提示词） | **PASS**（0 FAIL） |
| 3 次独立跑 | run1 **DRIFT**（HTTP 429，跳过，没编结果）；run2、run3 OK |
| `check_trunk_fixture.py --selftest`：候选报告 + 候选 facts 对 golden | **PASS**（K01–K17 全过，10 个变异全被抓到） |
| `check_trunk_fixture.py --selftest --live`：frozen 报告 + `--live` 候选 | **PASS**，drifts=[] |
| `verify_all.py`（tree_v18 + binding/merge/manifest/neutrality） | **PASS** |
| `check_book_root.py --selftest` | **PASS**（8 个坏样例） |

附注：第一次调用 check_trunk_fixture 时参数配错了（候选报告配了 frozen 的 `--facts`），K11/K17 按设计报 FAIL（facts 的 sha 对不上）。按正确配对重跑后 PASS；门槛没放宽。

## 2. OBS 清单核对（以 book_visual_v1 + obs_catalog 为准）

- 保留任务给的 11 题：OBS-10/04/05/15/38/33/36/32/02/07/16，都在 `book_visual_v1.rear_fixture_obs` 的「可答未测」里，catalog 机位是 B 或 any。
- **新增 OBS-06**（站位朝向）：book_visual BR-CH15 标了 rear_fixture=true、机位 B，而且旧事实已有「宽开放站位」这条文字描述，可以和本次结果对照。
- **新增 OBS-08**（复测）：用来裁决第 7 项的待定问题，结果不进候选快照。
- **不加 OBS-01**：book_visual 写明「夹角本身仍需侧面」。**不加 OBS-29/30/60**：book_visual 标 true，但不在 rear_fixture_obs 列表里，留到下一轮。
- 多问句 OBS（07/10/15）按 catalog 的 a/b/c 拆成子题，每个子题只用自己的选项，另加 unclear。

## 3. 多数结论（详表见 `user_case/rear_20261019_obs_fill_consensus.md`）

有效跑次只有 2 次，所以「成立」= 2/2 一致且不是 unclear。共 18 个子题：**成立 9 题，多数为 unclear 的 0 题**，9 题两次结果不一致（OBS-16 有 1 次答 unclear）。

成立：OBS-10b `stays_in_place`、OBS-04 `pause_present`、OBS-15a `finish_above_non_hitting_shoulder`、OBS-15b `wraps_back_to_body_immediately`、OBS-15c `elbow_in_front_of_chest_at_end`、OBS-38 `shoulder_width_or_wider`、OBS-33 `more_than_one_forearm_length`、OBS-36 `directly_toward_ball`、OBS-02 `visible_gap`。
两次不一致：OBS-10a、05、32、07a/b/c、16、06、08。

## 4. 成立事实 ↔ 书节点、对粗枝 1「髋没先启动」的作用

| 事实 | 书节点（章，页码） | 对粗枝 1 |
|---|---|---|
| OBS-10b 后脚原地不动 | BR-CH18 Lead With The Hips（p.83–88，CH18-S2 脚解放髋）；BR-CH15 Stance（p.77–79，p.79 后脚绕前是转髋指示）；BR-CH23（p.110–114，CH23-S1） | **支持（弱）**：后脚没被带走、没绕到身前，书把这一点当作髋转动的外部指示。OBS-10a 脚跟离地两次答案不一致，证据不完整 |
| OBS-15b 触球后马上缠回身体 | BR-CH22 Follow-Through（p.107–109，CH22-S1）；BR-CH23-S3；BR-CH21（p.99–106） | **支持（弱，属下游）**：书里这是「强拉过身体」的表现，trunk_branches 已有 B1 下游叶子 L1_finish_wraps 对应它 |
| OBS-15a 收拍在左肩上方 / OBS-15c 肘在胸前结束 | BR-CH20（p.92–98）；BR-CH22-S2（p.107–109） | **无关 / 轻微削弱**：收拍位置正常，说明随挥没有截断。它和「髋先启动」不直接相关 |
| OBS-33 触球点离髋超过一个前臂 | BR-CH20-S2（p.92–98）；BR-CH01（p.10–19）；BR-CH24-S1（p.115–121）；BR-CH27（p.129–135） | **与两条粗枝都无关**：属于树干层面的容错和间距，结论是「离开身体」做到了（和旧事实「横向间距明显」一致） |
| OBS-02 触球时手臂和躯干之间有空隙 | BR-CH11-S1（p.63–70）；BR-CH12（p.71–73） | **无关**：躯干和手臂的夹角没被压窄，属于树干层面的正常项 |
| OBS-36 前挥手直奔球、没先往外 | BR-CH20-S1 Out, Up, and Through（p.92–98） | **与粗枝 1 无关**，但在 CH20 下是个**新的偏差信号**（书要求先往外）。不能当作髋的证据 |
| OBS-38 站距 ≥ 肩宽 | BR-CH18-S2 上游条件（p.83–88） | **削弱「站距窄导致髋转不起来」这条替代解释**；对「髋没先启动」本身是中性的（上游条件满足，问题不在站距） |
| OBS-04 引拍顶点有停顿 | BR-CH13 Inhale and Wait（p.74–75，书认为等待正常）；BR-CH11（p.63–70） | 和粗枝 1 无关。它属于 **B2 的 context 叶子**。正后方判断不了挥拍起点和来球的相对时机，所以**不据此判 B2** |

## 5. 候选快照下的粗枝状态

- B1_HIP_TRANSFER_ARM_COMPENSATION：has_facts → **has_facts**（不变；证据仍是 OBS-18/25/08/09）。
- B2_SWING_START_TOO_EARLY：pending_self_report → **pending_self_report**（不变；没有伪造时机）。
- 叶子变化：**NEW** `L2_backswing_pause`（OBS-04）由 missing 变为 fact=`pause_present`（2/2，context 角色）。
- 没有 DRIFT（live 对比 frozen：drifts=[]）。
- 注意：B1 下游叶子 L1_finish_wraps（obs=`OBS-15`，value=`b:wraps_…`）和 L1_symbolic_hip_push（obs=`OBS-10`）**没有接上**本次的 OBS-15b / OBS-10b，原因是键格式不一样（子题后缀）。叶子状态因此没变。这一处没改，列为缺口。

## 6. OBS-08 待裁决

- frozen：`small_turn`（2/3）；上一轮 live 多数：`clearly_sideways`；本轮：run2 `small_turn`、run3 `clearly_sideways`（1:1，无多数）。
- 把历次结果放一起，两个值大致各占一半，说明这一题在正后方机位下**判断不稳定**。**建议**：frozen 暂时保持 small_turn 不动；golden K03 只要求 OBS-18/25，OBS-08 不是硬证据，建议在 B1 证据里把它降为「不稳定 / 辅助」，等侧面或 45° 机位再补测，不要翻成 clearly_sideways。

## 7. 产物

- 脚本 `/workspace/today_swing/analyze_rear_obs_fill.py`（在 box 上，不在仓库里）；提示词 `pipeline/prompts/facts_rear_fill_v1.md`
- `user_case/rear_20261019_obs_fill_consensus.md`、`rear_obs_fill_run{2,3}.md`、`rear_obs_fill_run1_DRIFT.txt`
- `user_case/rear_20261019_facts_candidate_fill.json`
- `loop/reports/trunk_rear_20261019_2026-10-21_{frozen,candidate_fill}.{json,md}`、`diagnose_…`、`trunk_gate_…`、`verify_2026-10-21_rear_fill.json`

## 8. 缺口（≤3）

1. run1 遇到 429，只有 2 个有效跑次，「成立」实际是 2/2。建议补跑 1 次，凑满 3 跑。
2. 多问句 OBS 拆成子题后的键（OBS-15b/10b）和 trunk_branches 叶子的键（OBS-15 + `b:` 值）对不上，下游叶子没接上，需要统一命名（改脚本或 compose 前做映射，要先裁决）。
3. OBS-06/07/05/32/10a 两次结果不一致，OBS-08 长期不稳定；OBS-29/30/60 还没测。

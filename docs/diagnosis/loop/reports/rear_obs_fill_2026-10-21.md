# 正后方 OBS 补测（rear fill）— 2026-10-21（v2：run4 补跑 + 子题键映射）

> 固定样本 `rear_20261019_small.mp4`（正后方，5 拍右手正手）。分支 `cursor/rear-obs-fill-2026-10-21`，基于 main 468c979。
> **frozen / golden 都没改**；新事实只写进候选快照 `user_case/rear_20261019_facts_candidate_fill.json`。不覆盖 tree.json v1 / tree_vN。

## 1. PASS/FAIL 总表

| 检查 | 结果 |
|---|---|
| `check_neutrality.py prompts/facts_*.md`（6 份，含 facts_rear_fill_v1） | **PASS**（0 FAIL） |
| 跑次 | run1 **DRIFT**（429，跳过）；**run4 补跑一次成功，替代 run1**；有效 3 次（run4/run2/run3） |
| `check_trunk_fixture --selftest`：候选报告 + 候选 facts 对 golden | **PASS**（K01–K17；**11 个**变异全被抓，新增 `subkey_wrong_value`→K11） |
| `check_trunk_fixture --selftest --live`：frozen 报告 + live 候选 | **PASS**，drifts=[] |
| `verify_all.py`（tree_v18） | **PASS** |
| `check_book_root.py --selftest` | **PASS** |

frozen 报告 `trunk_rear_20261019_2026-10-21_frozen.json` 在映射改动前后字节一致（frozen 里没有子题键）。

## 2. OBS 清单核对（以 book_visual_v1 + obs_catalog 为准）

- 保留任务给的 11 题：OBS-10/04/05/15/38/33/36/32/02/07/16，都在 `book_visual_v1.rear_fixture_obs` 的「可答未测」里，catalog 机位是 B 或 any。
- **新增 OBS-06**（站位朝向）：book_visual BR-CH15 标了 rear_fixture=true、机位 B，而且旧事实已有「宽开放站位」这条文字描述，可以和本次结果对照。
- **新增 OBS-08**（复测）：用来裁决第 7 项的待定问题，结果不进候选快照。
- **不加 OBS-01**：book_visual 写明「夹角本身仍需侧面」。**不加 OBS-29/30/60**：book_visual 标 true，但不在 rear_fixture_obs 列表里，留到下一轮。
- 多问句 OBS（07/10/15）按 catalog 的 a/b/c 拆成子题，每个子题只用自己的选项，另加 unclear。

## 3. 多数结论（详表 `user_case/rear_20261019_obs_fill_consensus.md`）

3 有效跑，≥2/3 一致且非 unclear 才成立。18 子题：**成立 17**（16 进候选快照 + OBS-08 只记录），**多数 unclear 1**（OBS-16，2/3 unclear），无多数 0。

- 3/3：OBS-10b `stays_in_place`、OBS-04 `pause_present`、OBS-15a `finish_above_non_hitting_shoulder`、OBS-15b `wraps_back_to_body_immediately`、OBS-15c `elbow_in_front_of_chest_at_end`、OBS-38 `shoulder_width_or_wider`、OBS-33 `more_than_one_forearm_length`、OBS-36 `directly_toward_ball`、OBS-02 `visible_gap`。
- 2/3（本轮 run4 新增成立）：OBS-10a `heel_down_at_contact`、OBS-05 `no_steps_stands_still`、OBS-32 `rotational_drive`、OBS-07a `reaches_across_body_front`、OBS-07b `leaves_racket_after_turn_complete`、OBS-07c `rotates_back_with_chest`、OBS-06 `open`、OBS-08 `clearly_sideways`（不进快照）。

## 4. 成立事实 ↔ 书节点、对粗枝 1「髋没先启动」

| 事实 | 书节点（章，页码） | 对粗枝 1 |
|---|---|---|
| OBS-10a 触球脚跟未离地（2/3）+ OBS-10b 后脚原地（3/3） | CH18 Lead With The Hips p.83–88（CH18-S2 脚解放髋）；CH15 p.77–79（p.79 后脚绕前=转髋指示）；CH23-S1 p.110–114 | **支持**：后腿没蹬起、后脚没被带走，与「髋没先启动 / 蹬髋象征性」一致 |
| OBS-15b 触球后马上缠回（3/3） | CH22-S1 p.107–109；CH23-S3；CH21 p.99–106 | **支持（下游）**：B1 下游叶子 L1_finish_wraps |
| OBS-32 rotational_drive（2/3） | CH19 / CH11（肘拖过躯干，book_visual 挂 CH11） p.63–70 | **轻微削弱**「纯手臂拖」说法；不触及髋先后（OBS-18/25 仍为 arm_first / shoulders_lead_hips） |
| OBS-07a/b/c 非持拍手横过身体、转完才离拍、随胸转（2/3） | CH10 Unit Turn p.59–62；CH19 p.89–91；CH23 p.110–114 | **无关 / 准备正常**：上身一体转做到了，问题不在准备的手 |
| OBS-05 无分步调整步（2/3）+ OBS-06 open（2/3）+ OBS-38 站距≥肩宽 | CH18-S2 上游条件 p.83–88；CH15 p.77–79；CH27 p.129–135 | 站距/开放站位：**削弱「站距窄」替代解释**；脚下不动属上游条件（frozen 已撤回「脚下锁死是第一要害」，不升粗枝） |
| OBS-15a / 15c 收拍位置正常 | CH20 p.92–98；CH22-S2 | 无关 |
| OBS-33 离髋 >1 前臂、OBS-02 有空隙 | CH20-S2、CH01 p.10–19、CH24-S1；CH11-S1、CH12 p.71–73 | 无关（树干容错项做到了） |
| OBS-36 手直奔球 | CH20-S1 Out, Up, and Through p.92–98 | 与粗枝 1 无关；CH20 下**新偏差信号** |
| OBS-04 引拍顶点停顿 | CH13 Inhale and Wait p.74–75；CH11 | B2 context 叶子；正后方不判时机，**不据此判 B2** |

## 5. 候选快照下粗枝状态 / 叶子

- B1：has_facts → **has_facts**，confidence medium 不变。
- B2：pending_self_report → **pending_self_report**，low 不变（未伪造时机）。
- 叶子 **NEW**（相对 frozen）：
  - `L1_symbolic_hip_push`（evidence）self_report → **fact** `OBS-10a=heel_down_at_contact` 2/3（叶子 value=null → 取最小子题字母 a，确定性）。
  - `L1_finish_wraps`（downstream）legacy → **fact** `OBS-15b=wraps_back_to_body_immediately` 3/3。
  - `L2_backswing_pause`（context）missing → **fact** `OBS-04=pause_present` 3/3。
- DRIFT：无（live drifts=[]）。

**子题键映射**（最小改法，不碰 frozen / trunk_branches）：`compose_trunk_report.match_leaf` 中精确匹配失败后，叶子 `OBS-15` + `b:<v>` → 查 `OBS-15b=<v>`；叶子 value=null → 按字母序取第一个合格子题；报告叶子 `obs` 写实际子题键，原键存 `leaf_obs`，K11 仍按 (obs,value,agree) 精确核对。`check_trunk_fixture` 自检新增变异 `subkey_wrong_value`（先验证映射能接上，再把值改成另一子选项 → 必须被 K11 抓到）。

## 6. OBS-08 待裁决

三跑：run4 `clearly_sideways`、run2 `small_turn`、run3 `clearly_sideways` → **2/3 clearly_sideways**。加上一轮 live 多数也是 clearly_sideways；frozen 是 small_turn（2/3）。
**建议**：两轮独立三跑都偏 clearly_sideways，建议用户裁决把 frozen OBS-08 改为 clearly_sideways（或降为不稳定，只作辅助）；golden K03 只硬要求 OBS-18/25，任一选择都不影响 B1 状态。本轮**未改 frozen**，候选快照也没追加。

## 7. 产物

- 脚本 `/workspace/today_swing/analyze_rear_obs_fill.py`（box）；提示词 `pipeline/prompts/facts_rear_fill_v1.md`
- `user_case/rear_20261019_obs_fill_consensus.md`、`rear_obs_fill_run{2,3,4}.md`、`rear_obs_fill_run1_DRIFT.txt`、`rear_20261019_facts_candidate_fill.json`
- `pipeline/compose_trunk_report.py`（子题映射）、`pipeline/check_trunk_fixture.py`（变异 11）
- `loop/reports/trunk_rear_20261019_2026-10-21_{frozen,candidate_fill}.*`、`trunk_gate_*`、`verify_2026-10-21_rear_fill.json`

## 8. 缺口（≤3）

1. OBS-08 frozen 需用户裁决（建议见 §6）。
2. `L1_symbolic_hip_push` 叶子 value=null，映射取 OBS-10a；若要用 10b（后脚绕前）应在 trunk_branches 给叶子写明子题值（需裁决）。
3. OBS-16 正后方多数 unclear；OBS-29/30/60 未测。

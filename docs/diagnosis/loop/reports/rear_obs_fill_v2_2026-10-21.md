# 正后方 OBS 补测 v2 + OBS-08 frozen 裁决 + L1_symbolic_hip_push 显式映射 — 2026-10-21

> 固定样本 `rear_20261019_small.mp4`（正后方）。分支 `cursor/obs08-frozen-obs29-30-60`，基于 origin/main 3f320e2。不覆盖 tree.json v1 / tree_vN；holdout 视频不进证据；golden 未改。

## 1. 用户裁决落地
1. **frozen OBS-08：small_turn → clearly_sideways（2/3）**。用户（屈书毅）2026-10-21 明确授权。依据 `user_case/rear_20261019_obs_fill_consensus.md`（run4/run2/run3 = clearly_sideways / small_turn / clearly_sideways），上一轮 live 多数也是 clearly_sideways。条目带 `adjudicated` 字段记录。候选快照同步。
2. **`L1_symbolic_hip_push` 显式绑定 OBS-10 子题 (a)**：trunk_branches.json 中 value 从 null 改为 `a:heel_down_at_contact`（触球时后脚跟未离地），加 `mapping_note`，不再靠「value=null 取最小字母」的自动映射。

## 2. 新测 OBS-29 / 30 / 60（gemini-3.8-flash inline_data，提示词 `pipeline/prompts/facts_rear_fill_v2.md`，三跑独立，0 DRIFT）

| OBS | run1 | run2 | run3 | 多数 | 成立 | 书节点 |
|---|---|---|---|---|---|---|
| OBS-29 脊柱侧屈、持拍肩下沉 | no_lateral_bend | lateral_bend_present | no_lateral_bend | no_lateral_bend 2/3 | 成立 | BR-CH12 Adjusting to Contact p.71–73（CH12-S1） |
| OBS-30 对方场落点深度 | unclear | unclear | unclear | unclear 3/3 | 不成立：正后方看不到落点 | BR-CH26 p.124–128；BR-CH28 p.136–142 |
| OBS-60 前臂加速开始时胸相对网 | side_on | side_on | side_on | side_on 3/3 | 成立 | BR-CH19 p.89–91（CH19-S3） |

详表 `user_case/rear_20261019_obs_fill_v2_consensus.md`；原始输出 `user_case/rear_obs_fill_v2_run{1,2,3}.md`。

## 3. 粗枝 / 叶子影响
- **B1（髋没先启动 → 手臂代偿）**：has_facts / medium 不变（K03 只硬要求 OBS-18/25）。
  - `L1_prep_turn_small`（context，绑 OBS-08=small_turn）：fact → **missing**（frozen 已改 clearly_sideways，这片叶子不再成立）。即「准备转离幅度小」不再作为 B1 的旁证。
  - `L1_symbolic_hip_push`：frozen 下仍为 self_report（frozen 无 OBS-10a）；候选快照下 fact `OBS-10a=heel_down_at_contact` 2/3，现在为显式映射。
  - OBS-60 side_on（加速开始时胸还侧对网）：书 CH19-S3 关注的是「过早 / 过多转开」，本样本没有这个现象，与 B1 的 OBS-25 shoulders_lead_hips 不冲突；暂无叶子绑定。
  - OBS-29 no_lateral_bend：与粗枝 1 无直接关系；在 tree_v18 diagnose 的 top3 里替换进 N-C_DZADHZVESDS_C03（medium）。
- **B2（挥拍启动过早）**：pending_self_report / low 不变；三题都不涉及时机。
- 新叶子：无。修改叶子：`L1_symbolic_hip_push`（显式子题）。

## 4. 闸门
| 检查 | 结果 |
|---|---|
| check_neutrality prompts/facts_*.md（7 份） | PASS 0 FAIL |
| diagnose → compose_trunk_report（frozen_adj、fill_v2） | 生成 |
| check_trunk_fixture --selftest（frozen_adj、fill_v2） | PASS K01–K17，11 个变异全抓 |
| check_trunk_fixture --selftest --live（frozen_adj + 候选） | PASS，drifts=[] |
| verify_all.py --selftest | PASS（58 变异） |
| verify_all.py（tree_v18，episodes+manifest，同上轮口径） | PASS |
| check_book_root.py --selftest | OVERALL PASS |

注：加 `--cases ../replay/cases --gate G1` 时 G1 recall@3 0.00 FAIL，与本次改动无关（上一轮也未用该口径），未放宽。

## 5. DRIFT
无（三跑全部一次成功，无 429）。

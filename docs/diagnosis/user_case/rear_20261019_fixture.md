# 固定测试样本：正后方 2026-10-19（rear_20261019）

> 登记：2026-10-20。用户原话：「按树干 / 粗枝优化诊断系统，优化后直接跑正后方视频实测；以后该视频作固定测试样本，近期反复测 → 找问题 → 改 → 再测，形成 loop。」
> **近期 loop 只用这一份样本。** 换样本或加样本需要用户同意，并在 §7 记一行。

## 1. 样本

| 项 | 值 |
|---|---|
| 视频 | `/workspace/today_swing/rear_20261019_small.mp4`（仓库外，不入库） |
| 机位 | 正后方固定（B） |
| 内容 | 发球机低速，5 拍正手，开放式站位；触球约 00:02 / 00:05 / 00:09 / 00:13 / 00:17 |
| 持拍手 | 右手 |
| 声音 | 送模型前去声 |

## 2. 冻结的输入

| 文件 | 内容 |
|---|---|
| `rear_20261019_facts_frozen.json` | 冻结事实快照：三跑中立事实（OBS-18/25/09/08/40/19/71 + 一道 NEW 候选）+ 旧文字事实 + 用户自述 + 撤回项 + 机位盲区 |
| 来源 | `rear_20261019_obs_trunk_consensus.md`（三跑）、`/workspace/today_swing/rear_20261019_facts.md`（旧中立一遍）、用户 2026-10-20 自述 |
| 规则 | `../ftt_tree/trunk_branches.json`（树干 / 粗枝 / 叶子 / 状态规则） |
| 期望 | `rear_20261019_golden.json` |
| 接口 | `../pipeline/TRUNK_REPORT_CONTRACT.md` |

## 3. 期望的诊断结构（golden 的人话版）

- **树干**：动力链（力从哪来、怎么传、什么时候启动）。
- **粗枝 1：髋没把力传上来 → 手臂 / 小臂代偿**，状态**有事实**：
  - 证据：OBS-18 = arm_first（3/3），OBS-25 = shoulders_lead_hips（3/3）；
  - 自述叶子：小臂上抬、蹬髋象征性。只标「用户自述」，不当事实；
  - 上游条件：脚钉死；下游：随挥缠回左肩；
  - 看不清：OBS-40 / 19 / 71；
  - 削弱的旁支：准备不足（FLT-04）、被挤住（FLT-01）。
- **粗枝 2：挥拍启动过早**，状态**待测，只有用户自述**：
  - 必须出现在报告里，不能静默消失；
  - 不能给出任何时机结论；
  - 必须列出缺失事实：NEW 启动时机题、OBS-57、OBS-03，都需侧面机位。
- 两根粗枝之间只有一条**推断**连线。
- **撤回**：「击球点偏晚」「脚下锁死是第一要害」。
- **置信度**：全部「待验证」，最高到中（V5 未过）。
- **只改一件事**：「等球过了最高点、开始往下落，再启动前挥」，开头带呼吸提醒。

## 4. 机位盲区（正后方看不清，不许硬判）

OBS-03 击球点前后、OBS-23 膝角深浅、OBS-57 转体时拍头相对球的高度、OBS-12 小臂角度（catalog 侧面题）、OBS-13 重心、OBS-28 拍面、OBS-35 拍头滞后、启动时机 NEW 题，以及实测三跑都看不清的 OBS-40 / 19 / 71。

## 5. loop 怎么跑

1. 改规则或脚本，比如 `trunk_branches.json`、`compose_trunk_report.py`、OBS 问法。
2. `compose_trunk_report.py`：用 frozen 快照生成报告。
3. `check_trunk_fixture.py`：对照 golden 判 K01–K17，必须全 PASS。
4. 重新抽事实时（可选）：对同一视频用中立提示词再跑 3 次，用 `--live` 比对。DRIFT 交人复核，不判 FAIL。
5. `verify_all.py`（tree_v18）照常跑，不放宽。
6. 报告和门闩结果存进 `loop/reports/`，commit message 带 PASS / FAIL 行。

## 6. 近期要补的事实（补到就更新快照，需用户确认）

- 正后方：NEW 小臂上抬（手相对肘），NEW 启动时机（试问、只作参考），OBS-10、OBS-04。
- 侧面（下次加机位）：NEW 启动时机、OBS-57、OBS-03、OBS-12、OBS-13、OBS-23。

## 7. 变更记录

- 2026-10-20：登记；冻结快照 v1；golden v1。


## 7b. book-loop 2026-10-20
- golden `ban_terms` += 「送髋」（加严；与 trunk banned_cue_extra / config ban_patterns 中文项对齐）。
- trunk 文案 book-grounded；**未改** branch id / OBS 期望。
- live OBS-08 DRIFT 见 `loop/reports/gemini_vs_frozen_book_loop_2026-10-20.md`；frozen 未改。

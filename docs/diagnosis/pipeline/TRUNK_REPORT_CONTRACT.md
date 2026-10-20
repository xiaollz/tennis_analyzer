# 接口约定：compose_trunk_report.py + check_trunk_fixture.py（2026-10-20）

> 设计：Claude Code（规则与文档）｜实现、verify、e2e、commit：执行者。
> 目的：把 `diagnose.py` 的扁平 top-3 和事实表，按「树干 → 粗枝 → 叶子」重写成给人看的报告；再用固定样本的 golden 判 PASS / FAIL。
> 规则全部放在数据文件里，脚本不写死业务判断：
> - `ftt_tree/trunk_branches.json`：树干、粗枝、叶子角色、状态规则、下一次只改一件事；
> - `user_case/rear_20261019_golden.json`：固定样本的期望结果。
> 不动的东西：`verify_all` 的阈值、`tree.json` v1、tree_v18。holdout 原话只能出现在报告的知识引用里，不得写进任何 tree evidence。

---

## 1. compose_trunk_report.py

### 调用

```
python3 compose_trunk_report.py \
  --branches ../ftt_tree/trunk_branches.json \
  --facts    ../user_case/rear_20261019_facts_frozen.json \
  [--diagnose diagnose_out.json]        # 可选：diagnose.py 的输出，只作附录，不决定粗枝
  [--tree    ../ftt_tree/tree_v18.json] # 可选：只用来校验 tree_nodes 存在
  --out-json ../loop/reports/trunk_rear_20261019_<tag>.json \
  --out-md   ../loop/reports/trunk_rear_20261019_<tag>.md
```

纯规则，不调用任何模型。同样的输入必须产生**逐字节相同**的输出：key 排序固定，不写时间戳；时间放在文件名的 `<tag>` 里。

### 处理规则（按顺序）

1. **事实过滤**。只有满足以下全部条件的 `facts` 条目才算「合格事实」：
   - `source` ∈ {model, pose, manual}；
   - `agree` ≥ 2/3；
   - `value` ≠ `unclear`。

   以下内容都**不能**让粗枝变成 `has_facts`：
   - `candidate_facts`（NEW 候选题）：可以作为叶子显示，标「候选」；
   - `legacy_facts`：只能挂成 upstream_condition / downstream / excluded_siblings；
   - `self_reports`：只能让粗枝变成 `pending_self_report`。
2. **叶子匹配**。逐个粗枝、逐片叶子匹配，每片叶子得到一个状态：
   - `fact`：合格事实，且 obs 和 value 都对上；
   - `candidate`：候选事实；
   - `legacy`：旧文字事实，按 `legacy_key` 匹配；
   - `self_report`：按 `self_report_key` 匹配；
   - `not_visible`：叶子的 role 是 not_visible，或者事实值是 `unclear`；
   - `missing`：以上都没有。
3. **粗枝状态**。按 `status_rules` 判定：
   - 至少一片 role=evidence 的叶子是 `fact` → `has_facts`；
   - 否则，有 `self_report` → `pending_self_report`；
   - 否则 → `pending_untested`。

   **粗枝永远不能从报告里消失。**
4. **层级约束**。顶层只能有 `trunk_branches.json` 里列出的粗枝。叶子按 role 挂到所属粗枝下面：
   - evidence / context / upstream_condition / downstream / not_visible；
   - excluded_siblings 写成「削弱 FLT-xx」。
5. **待测粗枝**。粗枝是 `pending_*` 时：
   - 输出 `missing_facts`，照抄 `missing_facts_if_pending`，带机位；
   - **不得**为其 evidence 叶子编出任何值。
6. **撤回**。`retracted_claims` 原样进报告的「撤回」一节，并带上理由。它们的 key 不得出现在任何叶子或证据里。
7. **置信度**。V5 未通过时：
   - 每个结论都标 `validation_label = "待验证"`；
   - 置信度上限为 `medium`；
   - 一律不给 `high`。
8. **下一次只改一件事**。只输出一条，照抄 `next_one_thing`，开头带呼吸提醒。输出前对所有处方文字做禁令扫描：命中推肘 / 降拍 / 旋后 / 伸肘就直接 exit 1。
9. **diagnose 附录**（可选）。`diagnose.py` 的 top 列表原样附在「脚本原始排名」里，只作参考，不改变粗枝和状态。

### 输出 JSON（最小字段）

```json
{
  "fixture": "rear_20261019",
  "trunk": {"id": "TRUNK_KINETIC_CHAIN", "zh": "..."},
  "branches": [
    {"id": "B1_...", "status": "has_facts", "confidence": "medium", "validation_label": "待验证",
     "leaves": [{"id": "L1_arm_first", "role": "evidence", "state": "fact", "obs": "OBS-18", "value": "arm_first", "agree": "3/3"}],
     "excluded_siblings": [{"weakens": "FLT-01", "legacy_key": "lateral_spacing_ok"}],
     "missing_facts": []}
  ],
  "branch_links": [{"from": "B2_...", "to": "B1_...", "kind": "inferred"}],
  "retracted": [{"key": "contact_late", "reason": "..."}],
  "next_one_thing": {"breath": "...", "zh": "...", "source": "dZadhzVEsds:Q02", "protocol": "...", "verify": ["..."]},
  "diagnose_appendix": [],
  "meta": {"v5_passed": false, "branches_file_sha256": "...", "facts_file_sha256": "..."}
}
```

MD 版按 `user_case/rear_20261019_trunk.md` 的结构写成中文：树干一句 → 粗枝 1 → 粗枝 2 → 两根粗枝的关系 → 撤回 → 下一次只改一件事 → 附录。

---

## 2. check_trunk_fixture.py（loop 门闩）

### 调用

```
python3 check_trunk_fixture.py --report <trunk_report.json> \
  --golden ../user_case/rear_20261019_golden.json \
  --facts  ../user_case/rear_20261019_facts_frozen.json \
  [--live <new_facts.json>]        # live 模式，见下
  --out ../loop/reports/trunk_gate_rear_20261019_<tag>.json
```

每条检查输出 PASS / FAIL；只要有一条 FAIL，exit 1。

### 检查清单

| ID | 检查 | FAIL 条件 |
|---|---|---|
| K01 | 树干 | `trunk.id` ≠ golden |
| K02 | 粗枝集合 | 顶层粗枝 id 集合 ≠ `branch_ids_exact`，包括多出来的或少了的 |
| K03 | 粗枝 1 状态 | ≠ `has_facts`，或证据里缺 `OBS-18=arm_first` / `OBS-25=shoulders_lead_hips` |
| K04 | 粗枝 2 存在且待测 | 粗枝 2 缺失（静默消失），或状态 ≠ `pending_self_report` |
| K05 | 不伪造时机 | 粗枝 2 的 evidence 叶子里，`must_not_have_fact_values_for` 中任何一项出现 state=fact |
| K06 | 缺失事实列出 | 粗枝 2 的 `missing_facts` 缺 golden 列出的任何一项，或机位不是 S |
| K07 | 层级 | `never_top_level` 中任何一项出现在顶层；或脚钉死不在 B1 的 upstream_condition 下；或随挥缠肩不在 B1 的 downstream 下 |
| K08 | 撤回 | `retracted` 缺 `contact_late` / `feet_locked_first_priority`；或这两个 key 出现在任何叶子 / 证据里 |
| K09 | 看不见的不硬判 | OBS-40/19/71 不在 not_visible 状态；或被当作证据 |
| K10 | 自述不当事实 | `L1_forearm_lift`、`L1_symbolic_hip_push` 的 state 是 fact，而输入里并没有对应的合格事实 |
| K11 | 无编造事实 | 报告里任何 state=fact 的叶子，在输入 facts 里找不到同 obs、同 value、同 agree 的条目 |
| K12 | 置信度与标签 | 任何粗枝 confidence = high，或 validation_label ≠ 待验证（V5 未过时） |
| K13 | 只改一件事 | `next_one_thing` 不是恰好 1 条；或缺呼吸提醒 |
| K14 | 禁令 | 报告 JSON 和 MD 的处方文字里命中禁令词（用 `pipeline/config.json` 的 ban_patterns） |
| K15 | holdout | 报告里的 tree 写回字段（如有）引用了 holdout id。知识引用允许，但必须标「holdout」 |
| K16 | 确定性 | 用同样的输入跑两次 compose，两个 JSON 的 sha256 不同 |
| K17 | 输入未被改 | `facts_frozen.json` 的 sha256 ≠ 报告 meta 里记录的值 |

### live 模式（以后重跑事实时用）

loop 里重新抽事实后，用 `--live new_facts.json` 再跑一遍：
- K01–K17 的**结构类**检查（K02、K04、K07、K08、K11–K17）照样必须 PASS；
- 事实值和 frozen 快照不同时，输出 `DRIFT` 清单（例如「OBS-18：arm_first → nearly_simultaneous」）。DRIFT **不判 FAIL**，交人复核；
- 只有用户确认 DRIFT 后，才能更新 frozen 快照和 golden，并在 fixture 文档里记一行。

### 自检变异（必须全部被抓，接进 verify_all 自检或单独跑）

至少包括以下 10 个：
1. 删掉粗枝 2 → K02/K04
2. 给 B2 的 NEW-SWING_START_VS_BOUNCE 伪造一个 fact → K05/K11
3. 把脚钉死提成顶层粗枝 → K02/K07
4. 撤回列表丢掉 contact_late → K08
5. 把 contact_late 挂成 B2 的叶子 → K08
6. OBS-40 unclear 被当证据 → K09
7. 把 forearm_lift 标成 fact → K10
8. B1 置信度改 high → K12
9. next_one_thing 改成含「伸肘」的句子 → K14
10. facts_frozen 改一个值但不更新 meta → K17

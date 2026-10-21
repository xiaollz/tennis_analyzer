# E2E：rear_20261019 × tree_v18 × trunk→branches（2026-10-20 Asia/Shanghai）

> 固定样本 loop 首轮门闩实测。不放宽 verify_all；不覆盖 tree.json v1；未重烧 Gemini（用 frozen 事实）。

## 命令（可重复）

```bash
cd /workspace/wt_diag/docs/diagnosis/pipeline

# 1) diagnose（扁平，仅附录）
python3 diagnose.py ../ftt_tree/tree_v18.json \
  ../user_case/rear_20261019_facts_frozen.json \
  > ../loop/reports/diagnose_rear_20261019_2026-10-20_v18.json

# 2) 树干→粗枝→叶子报告
python3 compose_trunk_report.py \
  --branches ../ftt_tree/trunk_branches.json \
  --facts    ../user_case/rear_20261019_facts_frozen.json \
  --diagnose ../loop/reports/diagnose_rear_20261019_2026-10-20_v18.json \
  --tree     ../ftt_tree/tree_v18.json \
  --out-json ../loop/reports/trunk_rear_20261019_2026-10-20_v18.json \
  --out-md   ../loop/reports/trunk_rear_20261019_2026-10-20_v18.md

# 3) golden 门闩 + 10 变异自检
python3 check_trunk_fixture.py \
  --report  ../loop/reports/trunk_rear_20261019_2026-10-20_v18.json \
  --golden  ../user_case/rear_20261019_golden.json \
  --facts   ../user_case/rear_20261019_facts_frozen.json \
  --branches ../ftt_tree/trunk_branches.json \
  --selftest \
  --out ../loop/reports/trunk_gate_rear_20261019_2026-10-20_v18.json

# 4) verify_all（树侧，无 G1/V5 gate）
cd /workspace/wt_diag
python3 docs/diagnosis/pipeline/verify_all.py \
  --episodes docs/diagnosis/episodes \
  --tree docs/diagnosis/ftt_tree/tree_v18.json \
  --manifest docs/diagnosis/loop/manifest.json \
  --binding docs/diagnosis/ftt_tree/obs_binding_v18.json \
  --merge docs/diagnosis/ftt_tree/mech_merge_v18.json \
  --tree-v1 docs/diagnosis/ftt_tree/tree_base_v18.json \
  --report docs/diagnosis/loop/reports/verify_2026-10-20_tree_v18.json
```

## VERIFY

| 项 | 结果 |
|---|---|
| `verify_2026-10-20_tree_v18.json` | **PASS**（selftest 58 mutations PASS；check_tree 0 FAIL） |
| trunk gate K01–K17 | **PASS**（0 fails） |
| trunk selftest 10 mutations | **PASS**（全部抓住） |
| compose 确定性（双跑 sha256） | **相同** |
| V5 / G1 | **本轮 skip**（如实；不上新片段） |

## Golden 对照

| 期望 | 结果 |
|---|---|
| 1. 先认树干=动力链，再认粗枝；禁止 OBS 平铺并列第一要害 | **PASS** — trunk=`TRUNK_KINETIC_CHAIN`；顶层仅 B1/B2 |
| 2. 粗枝1 触发（OBS-18 arm_first + OBS-25 shoulders_lead_hips），标「有事实」 | **PASS** — status=`has_facts`，两证据均 3/3 |
| 3. 粗枝2 存在但「待测/仅自述」；不伪造时机；不静默消失 | **PASS** — status=`pending_self_report`；missing 列 NEW-SWING_START_VS_BOUNCE / OBS-57 / OBS-03（机位 S） |
| 4. 脚钉死、随挥缠肩只挂上游/下游叶子 | **PASS** — L1_feet_planted=upstream_condition；L1_finish_wraps=downstream |
| 5. 旧「击球偏晚」不得再出现 | **PASS** — retracted 含 `contact_late`；叶子/证据中无该结论 |

## diagnose.py 附录（不决定粗枝）

扁平 top-3（tree_v18 + frozen facts）：`N-C_2NN9TYMECZU_C06` medium、`N-C_UB6SBA_KX9E_C04` medium、`N-C_PUN6QIIYU_4_C03` low。仅附录，叙事以 trunk 报告为准。

## 缺口（可执行，最多 5）

1. **侧面补拍**：对同一挥拍加侧面机位，补 NEW-SWING_START_VS_BOUNCE、OBS-57、OBS-03（粗枝2 从自述升级为有事实的前提）。
2. **正后方 NEW-OBS 试问（不升 catalog）**：手相对肘（小臂上抬）；OBS-10、OBS-04。若 API 429 则跳过并记 DRIFT。
3. **promote 流程**：两道 NEW 题（小臂上抬 / 启动时机）按 promote 审核后才入 catalog；本 loop 不自动升。
4. **V5 盲测仍 FAIL/skip**：对人交付继续标「待验证」；不得为 PASS 把 B1 confidence 改 high。
5. **live 模式**：下次重抽事实用 `check_trunk_fixture.py --live`；事实漂移只出 DRIFT 交人复核，不自动改 frozen/golden。

## 产出路径

- 固定样本登记：`user_case/rear_20261019_fixture.md`
- 规则：`ftt_tree/trunk_branches.json` + `pipeline/TRUNK_REPORT_CONTRACT.md`
- 报告：`loop/reports/trunk_rear_20261019_2026-10-20_v18.{json,md}`
- 门闩：`loop/reports/trunk_gate_rear_20261019_2026-10-20_v18.json`

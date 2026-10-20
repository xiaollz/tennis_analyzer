# E2E：rear_20261019 × tree_v18 × book-loop（2026-10-20 Asia/Shanghai）

> Phase A–D 完整 loop。书根基优化 trunk 文案；不覆盖 tree.json v1；不放宽 verify_all；Gemini ≥3 中立 OBS；未自动改 frozen。

## VERIFY（PASS/FAIL）

| 项 | 结果 |
|---|---|
| Claude tennis_gap 续聊 + `book_grounded_opt_2026-10-20.md` | **PASS** |
| trunk gate K01–K17 | **PASS**（0 fails；双跑一致） |
| trunk selftest 10 mutations | **PASS**（全部抓住） |
| verify_all tree_v18（58 mutations） | **PASS** |
| compose 确定性（双跑 sha256） | **相同** `33d90bf2127d5620…` |
| Gemini 中立 OBS ×3 | **PASS**（有产物；见 drift 表） |
| `--live` vs frozen | **PASS gate** + **DRIFT 列出**（未改 frozen） |
| V5 / G1 | **本轮 skip**（如实） |

## 命令（可重复）

```bash
cd /workspace/wt_diag/docs/diagnosis/pipeline
python3 diagnose.py ../ftt_tree/tree_v18.json \
  ../user_case/rear_20261019_facts_frozen.json \
  > ../loop/reports/diagnose_rear_20261019_2026-10-20_book_loop.json
python3 compose_trunk_report.py \
  --branches ../ftt_tree/trunk_branches.json \
  --facts    ../user_case/rear_20261019_facts_frozen.json \
  --diagnose ../loop/reports/diagnose_rear_20261019_2026-10-20_book_loop.json \
  --tree     ../ftt_tree/tree_v18.json \
  --out-json ../loop/reports/trunk_rear_20261019_2026-10-20_book_loop.json \
  --out-md   ../loop/reports/trunk_rear_20261019_2026-10-20_book_loop.md
python3 check_trunk_fixture.py \
  --report  ../loop/reports/trunk_rear_20261019_2026-10-20_book_loop.json \
  --golden  ../user_case/rear_20261019_golden.json \
  --facts   ../user_case/rear_20261019_facts_frozen.json \
  --branches ../ftt_tree/trunk_branches.json \
  --selftest \
  --out ../loop/reports/trunk_gate_rear_20261019_2026-10-20_book_loop.json
cd /workspace/wt_diag
python3 docs/diagnosis/pipeline/verify_all.py \
  --episodes docs/diagnosis/episodes \
  --tree docs/diagnosis/ftt_tree/tree_v18.json \
  --manifest docs/diagnosis/loop/manifest.json \
  --binding docs/diagnosis/ftt_tree/obs_binding_v18.json \
  --merge docs/diagnosis/ftt_tree/mech_merge_v18.json \
  --tree-v1 docs/diagnosis/ftt_tree/tree_base_v18.json \
  --report docs/diagnosis/loop/reports/verify_2026-10-20_book_loop.json
# live drift（可选）
cd docs/diagnosis/pipeline
python3 check_trunk_fixture.py \
  --report ../loop/reports/trunk_rear_20261019_2026-10-20_book_loop.json \
  --golden ../user_case/rear_20261019_golden.json \
  --facts  ../user_case/rear_20261019_facts_frozen.json \
  --live   ../user_case/rear_20261019_facts_live_book_loop.json \
  --out    ../loop/reports/trunk_gate_live_book_loop_2026-10-20.json
```

## 书根基改动摘要

- B1 改写为「髋肩反向扭转张力（coil）不足 → 手臂往上兜」；禁用「送髋」。
- trunk / kinetic_chain 同步；`config.json` ban_patterns 加中文「送髋」（英文 send hip 不禁，避免误伤卡片）。
- next_one_thing 文案未改；出处补 BOOK:p75。
- 「约 45° 往前上」标用户自述非 FTT。
- 详见 `book_grounded_opt_2026-10-20.md`。

## Gemini vs frozen

见 `gemini_vs_frozen_book_loop_2026-10-20.md`。要点：OBS-18/25 **CONFIRMED**；OBS-08 **DRIFT**（live=`clearly_sideways` vs frozen=`small_turn`）交人复核；未改 frozen。

## 缺口（最多 5）

1. 侧面补拍：NEW-SWING_START_VS_BOUNCE / OBS-57 / OBS-03。
2. OBS-08 DRIFT 人工裁决是否更新 frozen。
3. V5 盲测仍未过 → 对人交付置信度上限「中 / 待验证」。
4. 6ak4 尚无已核对卡片 Qnn。
5. 教练描述跑仍易把「脚钉死」说成根因——只作观感，不进事实。

## 产出路径

- 方案：`loop/reports/book_grounded_opt_2026-10-20.md`
- 报告：`loop/reports/trunk_rear_20261019_2026-10-20_book_loop.{json,md}`
- 门闩：`loop/reports/trunk_gate_rear_20261019_2026-10-20_book_loop.json`
- 人话：`loop/reports/coach_advice_rear_20261019_book_loop.md`
- Gemini：`/workspace/today_swing/rear_obs_trunk/run{1,2,3}_gemini-3.8-flash.md`

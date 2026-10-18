# pipeline/ — FTT 抽取 + verify 闭环（v2，2026-10-18）

计划全文：`../PLAN_v2.md`。只用 Python 标准库（3.9+）；不读、不写任何密钥。

| 文件 | 作用 | 规则编号 |
|---|---|---|
| `schema/episode_card.schema.json` | 一期一张卡（字幕层 + 视频层） | — |
| `schema/tree.schema.json` | 诊断树 v1（机器可读：观测项、节点签名、鉴别、边） | — |
| `schema/facts.schema.json` | 一个片段的中立事实表（diagnose 的输入） | — |
| `schema/replay_case.schema.json` | 盲测用例 | — |
| `config.json` | 全部阈值 + 禁令词 + 中立提示词禁词（改阈值需用户同意） | — |
| `validate_cards.py` | V1 schema + 结构规则 | R01–R12 |
| `check_quotes.py` | V2 原话在字幕中 + 时间戳 ±10 s + 字幕 sha 未变 | Q00–Q04 |
| `check_tree.py` | V3 树：证据可解析且已核对、签名有约束、可区分、无环、不引用 holdout | T01–T12 |
| `check_rerun.py` | V4 两次抽取一致性（按字幕时间锚比对）/ 事实多次运行一致率 | — |
| `diagnose.py` | 确定性匹配：事实 → top-3 根因 + 证据 + 置信度 + 下次只改一件事 | — |
| `blind_replay.py` | V5 盲测回放：静音片段 → 能否召回教练标注的问题；门槛 G1/G2 | B01–B03 |
| `check_neutrality.py` | V6 提示词中立性 | N01 |
| `check_manifest.py` | V7 循环清单与真实产物一致 | M01–M07 |
| `build_tree.py` | 确定性脚本：已核对卡片 → `ftt_tree/tree.json`（症状→根因→解法/角度；规则 B1–B7 见脚本头；跳过 holdout/低置信度/draft） | B1–B7 |
| `textsim.py` | 同机制判定（TF-IDF 余弦，build_tree 合并 ≥0.40 + 同击球类型） | — |
| `blind_replay_text.py` | 补充盲测：holdout 字幕 → Gemini 从树里检索 → 脚本判分（不是 PLAN 的静音片段 V5） | — |
| `compose_tree_report.py` | 把 verify_all + 文本盲测结果合成 loop/reports 报告 | — |
| `prompts/tree_retrieve_v1.md` `tree_judge_v1.md` | 文本盲测的检索 / 答案键提示词（中立，不含用户结论或概念） | — |
| `verify_all.py` | 一键总判；**先跑自检**（好样例必须过 + 22 个变异必须被抓） | — |
| `prompts/ep_s1_v1.md` `ep_s2_v1.md` `facts_v2.md` | 字幕层抽取 / 视频层补充 / 中立事实 | — |
| `fixtures/` | 合成样例（不是 FTT 内容），供自检 | — |

```bash
cd docs/diagnosis/pipeline
python3 verify_all.py --selftest                       # 第 0 步：必须 PASS
python3 verify_all.py --episodes ../episodes --tree ../ftt_tree/tree.json \
  --manifest ../loop/manifest.json --cases ../replay/cases \
  --transcript-root /workspace/ftt_subs --gate G1 --report ../loop/reports/verify_$(date +%F_%H%M).json
```

**状态（2026-10-18）**：脚本写完后只做了逐行人工核对；**还没有实际运行过**（本会话执行 Python 需要授权）。执行者第一件事就是跑 `--selftest`，失败先修脚本，再开始循环。

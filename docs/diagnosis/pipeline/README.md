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

### tree v2 追加（2026-10-18）
| 文件 | 作用 |
|---|---|
| `obs_catalog.json` | OBS-01…18 的中立英文题 + 取值 + 层级 + 机位（由 observables/*.md 转写） |
| `obs_bind.py` / `prompts/obs_bind_v1.md` `obs_gapfill_v1.md` `obs_newcluster_v1.md` | 模型绑定 OBS（2 次运行，两次都列为必需才算 required）→ `ftt_tree/obs_binding.json`；NEW-OBS 候选聚类 |
| `mech_merge.py` / `prompts/mech_group_v1.md` | 同击球类型机制分组（2 次运行取交集，只分组不改写）→ `ftt_tree/mech_merge.json` |
| `build_tree_v2.py` | 确定性：v1 树 + 绑定 + 合并 → `ftt_tree/tree_v2.json` |
| `check_obs_binding.py` / `check_mech_merge.py` | O1–O11 / M01–M09，已接入 verify_all（`--tree-v1 --binding --merge`） |
| `blind_replay_text_v2.py` / `prompts/tree_retrieve_v2.md` | 文本盲测新变体（根因独立检索，无整体弃权）；旧脚本与旧结果不动 |
| `compose_tree_v2_report.py` | 合成 `loop/reports/verify_*_tree_v2.json` 与 `ftt_tree/mech_merge_review.md` |

### tree v3 追加
| 文件 | 作用 |
|---|---|
| `promote_new_obs.py` | NEW-OBS 簇 → 家族（模型分组，两次一致）→ 按规则选 5–8 个升级为 OBS-19…，写 catalog + observables/*.md + `ftt_tree/new_obs_promotion.json` |
| `compose_tree_v3_report.py` | 合成 `loop/reports/verify_*_tree_v3.json`（v2→v3 前后对比） |
| 环境变量 `TV_WORK` `TV_BIND_OUT` `TV_MERGE_OUT` | 重跑绑定/合并时改工作目录与输出文件（默认仍是 v2 路径） |

### tree v4 追加（2026-10-18）
| 文件 | 作用 |
|---|---|
| `promote_new_obs.py approve --pick <binding.json>:<candidate cid> --note ...` | 用户人工批准的**单期例外**升级（OBS-26…29）：题目/取值逐字取自模型候选，写 catalog + observables + `new_obs_promotion.json` 的 `user_approved_single_episode_exceptions`；跨期规则本身未改 |
| `obs_bind.py` agg 变更 | 两次都 required 的同一 OBS 若列了 ≥2 个不同取值（模型的本意是「任一」），不可能同时成立 → 降为 supporting；`check_obs_binding` O10 同步复算（更严，不放宽） |
| `compose_tree_v4_report.py` | 合成 `loop/reports/verify_*_tree_v4.json`（v3→v4 前后对比，含三个 T08 复核组） |

### tree v5 追加（2026-10-18 夜）

| 脚本 | 作用 |
|---|---|
| `s1_round.py` | 批量 Stage1（ep_s1_v6 原流程：2 次 → final → 第 3 次 → final → consensus；缺 `norm.tsv` 时由 `.en.vtt` 生成，字幕不入库） |
| `s1_manifest.py` | 按卡与 `rerun_adjudication.json` 写回 `loop/manifest.json`（无判断） |
| `video_probe.py` + `prompts/facts_video_v1.md` | 视频层：YouTube URL `fileData` + 起止偏移，不下载（结论见 `loop/video_layer_feasibility.md`） |
| `gate_cc_t08.py` | 校验 Claude Code 对 T08 对的评审（提案不入树） |
| `compose_tree_v5_report.py` | 合成 `loop/reports/verify_*_tree_v5.json` |

环境变量：`TV_TREE_V1`（绑定 / 合并所用骨架）、`TV_PROMO_OUT`、`TV_MIN_PROMOTE`、`BT_TREE` / `BT_MM` / `BT_W2` / `BT_TAG`。

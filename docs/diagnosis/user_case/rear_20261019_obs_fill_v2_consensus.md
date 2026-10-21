# rear 2026-10-19 正后方 OBS 补测 v2（OBS-29 / 30 / 60，三跑多数）

> 模型 gemini-3.8-flash，inline_data；提示词 `pipeline/prompts/facts_rear_fill_v2.md`（check_neutrality 0 FAIL；只有 catalog 原题干 + 选项 + unclear，无结论、无知识库概念、无候选问题）。脚本 `/workspace/today_swing/analyze_rear_obs_fill_v2.py`（与 v1 同法，仅换题号）。
> 三跑独立并行，**0 次 429 / 0 DRIFT**。规则：3 次里 ≥2 次一致且非 unclear 才成立。原始输出 `rear_obs_fill_v2_run{1,2,3}.md`。

| OBS | 题干（obs_catalog 原文） | run1 | run2 | run3 | 多数 | 同意次数 | 成立否 | 书节点（book_root_v1 / book_visual_v1） |
|---|---|---|---|---|---|---|---|---|
| OBS-29 | During the forward swing into contact, does the spine display lateral side-bending with the hitting shoulder dipping lower than the non-hitting shoulder? | no_lateral_bend | lateral_bend_present | no_lateral_bend | no_lateral_bend | 2/3 | 成立 | BR-CH12 Adjusting to Contact p.71–73（CH12-S1） |
| OBS-30 | Depth of ball bounce in the opponent court | unclear | unclear | unclear | unclear | 3/3 | **不成立（正后方看不到落点）** | BR-CH26 p.124–128（CH26-S1）；BR-CH28 p.136–142（CH28-S1） |
| OBS-60 | Degree of chest rotation relative to net line at the start of forward arm acceleration | side_on | side_on | side_on | side_on | 3/3 | 成立 | BR-CH19 p.89–91（CH19-S3） |

成立 2 题（OBS-29、OBS-60）进候选快照 `rear_20261019_facts_candidate_fill.json`；OBS-30 记为 majority_unclear，不进 facts。

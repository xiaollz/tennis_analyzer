# V5 静音盲测（2026-10-19，用户批准按段下载）

> 模型：`gemini-3.8-flash`（Micu）；提示词 `facts_v2`（中立，经 `check_neutrality`）；本地 mp4 `ffmpeg -an` 后 inline。
> 树：`tree_v9.json`。**不抽 holdout 进树**；阈值未放宽。

## 1. 下载的片段（约 30 s，仅所需段）

共 **10** 段 raw+muted（仓库外 `/workspace/s1_work/video/`；不入 git）。用例/facts：`docs/diagnosis/replay/muted_v5/`。

| 片段 | holdout | 标签 | 事实抽取 |
|---|:---:|---|---|
| Am8j1Zw5KrE_70-100 | 是 | Q04 induced | 3 次，空 strokes |
| dx8aGSIo24w_39-69 | 否 | Q04 | 3 次，可用 |
| wVa4XQPcaqs_92-122 | 否 | Q08 | 2 次（1×429） |
| 42BfbKsTGb4_87-117 | 否 | Q09 | 3 次，多数 unclear |
| 9NirSL59an0_172-202 | 否 | Q08 | 3 次，空 strokes |
| RVfmDk-iEwo_98-128 | 否 | Q04 | 3 次，空 strokes |
| Psidjei5BnI_108-138 | 否 | Q06 | 3 次，可用 |
| V-QkILd4V-w_102-132 | 否 | Q03 | 3 次，可用 |
| 1W1rope5l0k_76-106 | 否 | Q06 | 3 次，空 strokes |
| PfDqnff_vYA_48-78 | 否 | Q04 | 3 次，可用 |

## 2. Holdout 覆盖

- holdout 20 期里有 verified+demonstrator 卡的 **仅 1**（Am8j1Zw5KrE）。其余 19 无卡，未抽进树。
- 本轮 10 用例 = 1 holdout + 9 非 holdout（管道验证）；**不能**宣称 holdout 泛化。

## 3. blind_replay @ G1（10 用例）

```
cases 10 >= 10 → PASS
episodes 10 >= 5 → PASS
recall@3 0.00 >= 0.60 → FAIL
negatives 0（缺阴性对照）→ FAIL
abstain_rate 0.70
```

整体 **G1 FAIL**。有召回尝试的 3 例均未命中期望节点（错配到其它 cause）。报告：`blind_replay_v5_muted_2026-10-19.json`。

## 4. 结论与阻塞

- 按段下载 + 去声 + inline **可行**（android client 绕过部分 429）。
- V5 召回为 0：部分片段无可见连续正手（空 strokes）；有事实时 diagnose 与教练标签节点不对齐；缺阴性对照。
- 阻塞：holdout 缺卡（19/20）；需阴性 correct_demo 用例；空 strokes 片段需换时段或人工核画面。

# V5 静音盲测 第三轮（muted_v7 + tree_v13），2026-10-19 Asia/Shanghai

> 模型 `gemini-3.8-flash`；提示词 `facts_v3`（中立）；20–40s 段、`ffmpeg -an`；**未**抽 holdout 进树；阈值未放宽。
> 用例：`replay/muted_v7/`；重放：`loop/reports/blind_replay_v7_muted_2026-10-19.json`。
> **管线修复**：`verify_all.py` 在 `--gate G1|G2` 且未传 `--cases` 时，默认挂 `replay/muted_v6/cases`（避免再出现 cases=0 四门全 FAIL 的假坏）。

## 1. 结果

| | v6（tree_v10/12） | v7（tree_v13） |
|---|---:|---:|
| 用例 / 期 | 10 / 7 | **10 / 9** |
| 阳性 / 阴性 | 6 / 4 | **6 / 4** |
| 空挥拍片段 | 1 | **0**（本轮阳性全有挥拍） |
| abstain | 0.17 | **0.00** |
| 阴性高置信误报 | 0/4 | **0/4** |
| recall@3 | 0.00 | **0.00（G1 仍 FAIL）** |

G1：cases / episodes / negatives 三项 PASS；recall@3 FAIL。

## 2. 本轮相对 v6 改了什么

- 按「节点已有 required、且 required 在 S/A 机位可用」优先选标签；避开教练纯讲解时间戳（如 pQ793 650–680 三次确认无挥拍，已弃）。
- 新下 6 段 + 复用 v7/v6 已有去声段；事实抽取 3 次/段。
- 仍几乎全是 **behind / oblique**（FTT 素材里真正 side 极少）；模型对相机词多数标 `other`。

## 3. 为什么有挥拍、有 OBS-55 等新签名，recall 仍是 0

1. **事实取值对不上故障 required**（主因）。例：`PUN6 350-380` 期望 `N-C_PUN6QIIYU_4_C01`（required `OBS-55=inside_baseline`），三次一致给出 `behind_baseline` → 诊断正确不命中。标签引文在讲「球偏低/难打」，片段里人看起来仍在底线后。
2. **斜拍/背后 + 习惯阈值**：B 层需 S 的观察项大量被门控丢掉；`strokes` 不到 ~80% 的也被丢 → 剩下的多为泛用项（如 OBS-04 / OBS-20），排名被 `LBDQ C02` 等「常见签名」占满（与 v6 相同的恒定答案问题）。
3. **期望节点签名仍偏窄**：6 个阳性里，有的 required 在当前机位根本不可见；有的与片段内容方向相反。

## 4. 仍缺什么素材（要 recall 实质上升）

- **正侧面（S）**、全身、连续 ≥5 拍、**示范者正在犯标签所述故障** 的 20–40s 去声段（不要教练讲话口型时间戳）。
- 阴性对照：同机位、无明显故障的正确示范。
- 可选：接 pose 以给出 high 置信（目前无 pose，阴性「高置信误报」几乎必过）。

## 5. 硬约束

tree.json v1 未覆盖；阈值未放宽；holdout 未入树；禁令口令未写入；未下整集。

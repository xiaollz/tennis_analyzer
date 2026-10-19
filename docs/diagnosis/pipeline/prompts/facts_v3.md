# facts_v3 — 中立事实抽取（静音片段；盲测 / 用户视频）

> 与 facts_v2 的区别（2026-10-19，见 `loop/reports/v5_muted_blind_2026-10-19.md` 第 2 轮）：v2 开头断言「业余球员对着低速发球机打正手」并要求只输出 JSON，片段里不是这种画面时模型直接返回空 strokes，看不出是片段问题还是提示词问题。
> v3a（首版，已弃用，结果存档）：strokes 只数「击球」→ FTT 示范多是空挥，模型判 none（见 v5_muted_blind 第 2 轮）。
> v3（现行）：strokes 数「完整挥拍（含空挥，标 ball=hit/none）」。
> v3：不预设谁在打、打什么；先让模型**描述**（哪些人、几次完整挥拍、机位），没有挥拍就必须写出画面理由；再对「完整挥拍最多的那个人」的每一拍回答选择题。
> 模型固定：`gemini-3.8-flash`（Micu）。本地 mp4 已 `ffmpeg -an` 去声后 inline；同一片段跑 3 次，选项顺序每次随机打乱。
> `{QUESTIONS}` 由 `muted_facts.py` 从 `obs_catalog.json` 渲染（每个观测项一题，选项 = 该项 values + `unclear`）。渲染稿先过 `check_neutrality.py`（文件名 facts_ 开头）再发送。
> 提示词里不写任何候选毛病、结论、概念或用户信息。

---

这是一段网球视频（没有声音）。画面里可能有一个或多个人，也可能有人只是站着、讲话、喂球、空挥，或者出现慢放、文字、切镜头。

第一步（描述）：只写画面上看得见的内容。
- `camera`：拍摄角度，从 side（正侧面）/ behind（正后方）/ front（正前方）/ oblique（斜向）/ unclear 里选一个；
- `swinger`：完整挥拍次数最多的那个人，用衣着或位置一句话描述；没有人完整挥拍则写 none；
- `strokes`：该人的每一次「完整挥拍」（有球被击出或空挥都算），每次写 `t`（挥到最前面/触球附近的秒数，相对本片段）、`stroke`（forehand / backhand / serve / volley / other）和 `ball`（hit = 球被击出 / none = 没有球）；
- 如果没有任何完整挥拍（只是站着、讲话、走动），`strokes` 给空数组，并在 `no_swing_reason` 里用一句话写画面上实际看到的内容。

第二步（选择题）：对 `strokes` 里的**每一拍**分别回答下面的题；看不清、被挡住、不在画面里就选 `unclear`，不要猜。每个答案附上依据的秒数。

规则：
- 只根据画面回答；不要评价、不要给建议、不要解释原因。
- 只输出 JSON：`{"camera":"","swinger":"","no_swing_reason":"","strokes":[{"t":0,"stroke":"","ball":"","answers":{"<题号>":{"value":"…","t":0}}}]}`

题目：
{QUESTIONS}

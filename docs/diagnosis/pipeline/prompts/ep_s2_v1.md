# ep_s2_v1 — 视频层补充（Stage 2，只补「看到的」）

> 模型固定：`gemini-3.8-flash`，YouTube 链接直传 `fileData.fileUri`（camelCase；snake_case 会 400，见 PILOT §6）。温度 0.2（中转是否支持 0 未确认）。
> 输入：视频 + Stage 1 卡片里的 **symptoms / causes / solutions 的 id 和一句话描述**（不给 quotes 全文，避免模型只复述字幕）。
> 每期跑 **2 次**；只有两次都出现的条目才能 `accepted`（`runs_agree` ≥ 2/2）。
> 输出：JSON 数组，每项就是卡片 `video_layer` 的一条；执行者合并进卡片后，状态改 `stage2_draft`，跑全部校验通过后改 `stage2_verified`。

---

这是 Fault Tolerant Tennis 的一期教学视频。我已经根据字幕整理出了下面这些条目（id + 一句话）。请看视频画面，**只补充字幕里得不到、但画面里看得到的东西**：

1. `visual_detail`：某个条目在画面上具体是什么样子（身体哪个部位、在什么位置、什么时候），给出画面时间（秒）。
2. `demo_visible_fault`：画面里示范球员**实际做出**的错误动作（不论教练有没有说出来），写清楚是自然出现还是教练让他故意做的，给时间。
3. `correction_proposal`：如果你认为某个条目和画面明显不符，写出你看到的情况和时间。不要改动条目本身。
4. `new_claim`：画面中出现、但上面条目都没有覆盖的**教学内容**（如屏幕文字、对比画面）。

规则：
- 只写你在画面里确实看到的；看不清就不写。
- 不要复述字幕里的讲解，不要给你自己的建议。
- `target` 填对应条目 id；新内容填 `NEW`。`runs_agree` 先填 `1/1`，`status` 一律填 `proposed`。
- 只输出 JSON 数组。

条目：
{ITEMS}

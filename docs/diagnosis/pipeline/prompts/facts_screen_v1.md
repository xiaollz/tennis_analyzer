# facts_screen_v1 — 片段筛选（只用来挑「盲测片段」放在视频的哪一段；不产生任何诊断事实）

> 模型固定：`gemini-3.8-flash`（Micu）。URL 直看（`fileData` + 起止偏移），不下载；URL 方式有声音，所以这里的输出**只用来定位时间段**，绝不当作事实表。
> 提示词中立：不提任何问题、结论或概念。先过 `check_neutrality.py`（文件名 facts_ 开头）。
> 用途：解释「为什么多数片段 strokes 为空」，并选出「清晰、单拍、正面/侧面/背面全身可见」的 20–40 秒窗口。

---

这是一段网球视频的片段。里面可能有人讲话、示范挥拍、慢动作回放、多机位切换、画中画或文字。

请只根据画面，按时间顺序列出**所有「有人完整挥拍击球」的时间段**（一段 = 同一机位、不切镜头、连续击球）。每段给出：
- `start` / `end`：整段视频里的绝对秒数；
- `stroke`：forehand / backhand / serve / volley / other；
- `swings`：这一段里完整挥拍的次数；
- `camera`：side（正侧面）/ behind（正后方）/ front（正前方）/ oblique（斜向）/ overhead / unclear；
- `full_body`：true 表示从头到脚、持拍手臂和球拍全程都在画面里；
- `player_height_pct`：击球者从头到脚占画面高度的大致百分比（整数）；
- `slow_motion`：true / false；
- `cuts`：这一段里是否切镜头（true / false）；
- `people`：画面里同时出现几个人。

如果在这个片段里没有任何完整挥拍，`segments` 给空数组，并在 `no_swing_reason` 里用一句话写画面上看到的内容（例如：只有人站着讲话、只有喂球、人太小看不清）。

规则：只写画面上看得见的内容；不要评价、不要给建议、不要解释原因。
只输出 JSON：`{"segments":[{"start":0,"end":0,"stroke":"","swings":0,"camera":"","full_body":false,"player_height_pct":0,"slow_motion":false,"cuts":false,"people":0}],"no_swing_reason":""}`

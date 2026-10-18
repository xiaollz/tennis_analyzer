# facts_video_v1 — 视频层中立事实（YouTube URL 直看；先描述后回答，不诊断）

> 模型固定：`gemini-3.8-flash`（Micu），`fileData.fileUri` = YouTube URL + `videoMetadata` 起止偏移；**URL 方式无法去掉音轨**（见 `loop/video_layer_feasibility.md`），所以本提示词只能要求「只用画面」，不能保证。
> `{QUESTIONS}` 由执行者从 `obs_catalog.json` 渲染：每个观测项一题，选项 = 该项 `values` + `unclear`，选项顺序每次随机打乱。渲染稿必须先过 `check_neutrality.py`（文件名 facts_ 开头）再发送。
> 提示词里不写任何候选毛病、结论或概念。

---

这是一段网球视频的片段。里面可能有人示范挥拍、慢动作回放、画中画或文字。

第一步：按时间顺序描述画面上看得见的内容（谁在做什么、机位、是否慢放），每条写 mm:ss。

第二步：回答下面的选择题。每题针对画面里被示范 / 被击球的那个动作（如果出现多次，分别回答并注明时间）。

规则：
- 只根据画面回答；不要使用你听到的话。
- 看不清、不在画面里、机位不允许判断，就选 `unclear`，不要猜。
- 每个答案附上你依据的画面时间（mm:ss）。
- 不要评价、不要给建议、不要解释原因。
- 只输出 JSON：`{"description":[{"t":"mm:ss","what":"…"}],"answers":[{"q":"<题号>","value":"…","t":"mm:ss"}]}`

题目：
{QUESTIONS}

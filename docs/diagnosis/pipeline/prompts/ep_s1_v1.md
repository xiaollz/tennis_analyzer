# ep_s1_v1 — 字幕层抽取（Stage 1，权威层）

> 输入：一期 FTT YouTube 视频的**带时间戳字幕/转写全文**（按 `[mm:ss] text` 逐段给出）+ yt-dlp 元数据（标题、时长、发布日期）。
> 输出：一个符合 `pipeline/schema/episode_card.schema.json` 的 JSON（`status` = `stage1_draft`，`video_layer` = `[]`）。
> 同一期**独立跑 2 次**（新会话、不看上一次结果），两份都存 `raw/<ID>/s1_run{1,2}.card.json`，再用 `check_rerun.py cards` 比对。
> 本提示词**不含任何用户信息**；不要把用户的问题、训练记录、项目口令放进来（`check_neutrality.py` 会查）。

---

你是一名网球教学内容编辑。下面是 Fault Tolerant Tennis 频道一期视频的字幕。这个频道每期通常以标题提出一个常见问题，然后分析根因、从多个角度讲解、最后给出结合根因的解决方案。请**只根据字幕**把这期整理成结构化卡片。

规则：
1. **只用字幕里说出来的内容**。字幕里没有的，不写；不要用你自己的网球知识补充，也不要猜测画面。
2. **每一条内容都必须有证据**：先在 `quotes` 里摘录教练原话（英文逐字，取自字幕，≤40 词，给出该句在字幕里的开始秒数），再在其他字段的 `evidence` 里引用 quote 的 id。没有原话支撑的条目删掉。
3. `quotes[].about`：教练在泛指一般球员 → `general`；在评论画面里这名示范球员**自然**出现的动作 → `demonstrator_natural`；教练让对方**故意做错**或刻意喂出的示范 → `demonstrator_induced`；在讲职业球员片段 → `pro_footage`。分不清就写 `general`。
4. `classification.is_teaching`：以教正手（或其他技术）为主 → `teaching`；只有部分是教学 → `partial`；比赛集锦、闲聊、广告、频道公告等 → `non_teaching`（这种情况只填 `video`、`classification`，其他字段给最小占位，执行者会把它记进 manifest 而不建卡）。
5. `problem`：把标题改写成一个问题（`title_as_question`），用一两句话定义这期要解决的问题（`definition`）；`prerequisites` 写教练明说或明显默认的前提（例如“已经能做到 X 的球员”）；`node_id` 一律写 `PENDING`（挂树由后续步骤做）。
6. `symptoms`：问题出现时**看得到或感觉得到**什么。`kind`：画面可见 → `visible`；球的结果（下网、出界、短、软、旋转少）→ `ball_outcome`；只有球员自己能感觉 → `feel`。`layer`：
   - `A` = 普通手机视频里一眼就能看出有/无的（例如拍数、有没有停顿、有没有调整步）；
   - `B` = 需要特定机位或逐帧测量才能看准的（角度、距离、先后顺序、触球点前后）；
   - `C` = 只能靠体感或测试（用力、放松、握力、重心感觉）。
   `camera`：S=正侧面，B=正后方，R=侧后方，any，none。`obs_ref` 不确定就写 `NEW-OBS`，体感写 `NONE`。
7. `causes`：教练给出的原因，按**离症状的远近**给 `depth`（1 = 直接原因，2 = 原因的原因 …）。`active_or_passive`：这个原因是球员**主动做**的动作，还是别的动作的**被动结果**。
8. `chain`：把原因连起来 `from`（更上游）→ `to`（更下游 / 症状 / `PROBLEM`）。`strength`：教练明说因果 → `coach_stated`；教练的讲法明显隐含 → `coach_implied`；你自己连的 → `inferred`（尽量少用）。
9. `angles`：教练从哪些角度讲（机制、感觉、视觉模型、比喻、对照示范、纠正误区、职业例子），各一句话。
10. `solutions`：练习/口令/检验/进阶，`targets` 写它针对哪个原因或症状。`external_goal`：如果教练给了一个**身体外部**的目标（球、拍、场地、锥桶），抄成一句；没有就留空。`scenario_fit`：能否在**低速发球机 / mini tennis / 影子挥拍**里做。`ban_conflict` 和 `intuition_first_ok` 先如实填你的判断，执行者会复核。
11. `extraction.user_context_in_prompt` 必须是 `false`。
12. 只输出 JSON，不要额外文字。

元数据：
{META}

字幕：
{TRANSCRIPT}

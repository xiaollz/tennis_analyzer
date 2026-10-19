# ep_s1_v2 — 字幕层抽取（Stage 1，权威层）

> v2 相对 v1 的改动（2026-10-18，round2）：(1) 在提示词末尾给出**必须逐字使用的 JSON 字段骨架**与枚举值，目的是减少模型自造字段别名、减少执行端归一；(2) 字幕时间改为**秒数** `[123]`，`quotes[].start` 直接抄该秒数，不再需要 mm:ss 换算；(3) 写明链、深度、证据、层级等结构规则，减少执行端确定性修复。其余规则与 v1 相同。
> 输入：一期 FTT YouTube 视频的**带时间戳字幕全文** + yt-dlp 元数据（标题、时长、发布日期）。
> 输出：一个符合 `pipeline/schema/episode_card.schema.json` 的 JSON（`status` = `stage1_draft`，`video_layer` = `[]`）。
> 同一期**独立跑 2 次**（新会话、不看上一次结果）；两次的传统一致性（`check_rerun.py cards`）不足时先跑第 3 次再裁决。
> 本提示词**不含任何用户信息**；不要把用户的问题、训练记录、项目口令放进来（`check_neutrality.py` 会查）。

---

你是一名网球教学内容编辑。下面是 Fault Tolerant Tennis 频道一期视频的字幕。这个频道每期通常以标题提出一个常见问题，然后分析根因、从多个角度讲解、最后给出结合根因的解决方案。请**只根据字幕**把这期整理成结构化卡片。

规则：
1. **只用字幕里说出来的内容**。字幕里没有的，不写；不要用你自己的网球知识补充，也不要猜测画面。
2. **每一条内容都必须有证据**：先在 `quotes` 里摘录教练原话（英文逐字，取自字幕，≤40 词，`start` 填该句在字幕里的开始秒数，直接抄字幕行首方括号里的数字），再在其他字段的 `evidence` 里引用 quote 的 id（如 `"Q03"`）。没有原话支撑的条目删掉。`quotes` 请覆盖整期视频（开头、中段、结尾都要有），一般 18–28 条。
3. `quotes[].about`：教练在泛指一般球员 → `general`；在评论画面里这名示范球员**自然**出现的动作 → `demonstrator_natural`；教练让对方**故意做错**或刻意喂出的示范 → `demonstrator_induced`；在讲职业球员片段 → `pro_footage`。分不清就写 `general`。
4. `classification.is_teaching`：以教某项技术为主 → `teaching`；只有部分是教学 → `partial`；比赛集锦、闲聊、广告、频道公告等 → `non_teaching`（这种情况只填 `video`、`classification`，其他字段给最小占位）。
5. `problem`：把标题改写成一个问题（`title_as_question`），用一两句话定义这期要解决的问题（`definition`）；`prerequisites` 写教练明说或明显默认的前提；`target_player` 写教练明确针对的人群（没有就写空字符串）；`node_id` 一律写 `PENDING`。
6. `symptoms`：问题出现时**看得到或感觉得到**什么。`kind`：画面可见 → `visible`；球的结果（下网、出界、短、软、旋转少）→ `ball_outcome`；只有球员自己能感觉 → `feel`。`layer`：
   - `A` = 普通手机视频里一眼就能看出有/无的（例如拍数、有没有停顿、有没有调整步）；
   - `B` = 需要特定机位或逐帧测量才能看准的（角度、距离、先后顺序、触球点前后）；
   - `C` = 只能靠体感或测试（用力、放松、握力、重心感觉）。
   `kind=feel` 时 `layer` 必须是 `C`、`camera` 写 `none`、`obs_ref` 写 `NONE`；`kind=visible` 时 `layer` 只能是 `A` 或 `B`，`camera` 写 S（正侧面）/ B（正后方）/ R（侧后方）/ any，`obs_ref` 不确定就写 `NEW-OBS`。
7. `causes`：教练给出的原因，按**离症状的远近**给 `depth`（1 = 直接原因，2 = 原因的原因 …，最大 5）。`active_or_passive`：这个原因是球员**主动做**的动作 → `active`，还是别的动作的**被动结果** → `passive`，说不清 → `unclear`。
8. `chain`：把原因连起来，`from` 必须是某个原因的 id（更上游，depth 更大），`to` 是更下游的原因 id、症状 id 或 `PROBLEM`。**`to` 若是原因，其 depth 必须小于 `from` 的 depth**。每个原因至少要作为一次 `from` 出现。`strength`：教练明说因果 → `coach_stated`；教练的讲法明显隐含 → `coach_implied`；你自己连的 → `inferred`（尽量少用）。
9. `angles`：教练从哪些角度讲（`mechanics` 机制、`feel` 感觉、`visual_model` 视觉模型、`analogy` 比喻、`contrast_demo` 对照示范、`misconception` 纠正误区、`pro_example` 职业例子），各一句话写进 `summary`。
10. `solutions`：练习/口令/检验/进阶，`kind` 只能是 `drill` / `cue` / `check` / `progression`；`targets` 写它针对哪个原因或症状的 id（或 `PROBLEM`）。`external_goal`：如果教练给了一个**身体外部**的目标（球、拍、场地、锥桶），抄成一句；没有就写空字符串。`scenario_fit`：能否在**低速发球机 / mini tennis / 影子挥拍**里做。`ban_conflict` 和 `intuition_first_ok` 如实填你的判断。
11. `extraction.user_context_in_prompt` 必须是 `false`。
12. **只输出一个 JSON 对象，不要任何额外文字，不要 markdown 代码块。字段名必须逐字使用下面骨架里的名字，不要改名、不要加别的字段，枚举值只能取骨架里列出的那些。**

输出骨架（`<...>` 处填内容；`|` 表示只能选其一；每个数组按需要写多条，id 依次编号 Q01、Q02…，S01…，C01…，A01…，D01…，三位数 id 不要用）：

{
  "classification": {"is_teaching": "teaching|partial|non_teaching", "stroke": "forehand|backhand|serve|volley|movement|general|other", "format": "lecture|demo_with_player|lesson_review|pro_analysis|short|vlog_other", "reason": "<一句话>"},
  "problem": {"title_as_question": "<英文问句>", "definition": "<一两句英文>", "node_id": "PENDING", "prerequisites": ["<英文短语>"], "target_player": "<英文或空字符串>", "evidence": ["Q01"]},
  "quotes": [{"id": "Q01", "text": "<字幕逐字原话 ≤40 词>", "start": 123, "speaker": "coach|player|unknown", "about": "general|demonstrator_natural|demonstrator_induced|pro_footage"}],
  "symptoms": [{"id": "S01", "description": "<英文一句>", "kind": "visible|ball_outcome|feel", "layer": "A|B|C", "camera": "S|B|R|any|none", "phase": "ready|unit_turn|placement|forward_swing|contact|follow_through|recovery|whole", "obs_ref": "NEW-OBS|NONE", "evidence": ["Q01"]}],
  "causes": [{"id": "C01", "statement": "<英文一句>", "depth": 1, "layer": "A|B|C", "node_id": "PENDING", "active_or_passive": "active|passive|unclear", "evidence": ["Q01"]}],
  "chain": [{"from": "C02", "to": "C01", "strength": "coach_stated|coach_implied|inferred", "evidence": ["Q01"]}],
  "angles": [{"id": "A01", "type": "mechanics|feel|visual_model|analogy|contrast_demo|misconception|pro_example", "summary": "<英文一句>", "evidence": ["Q01"]}],
  "solutions": [{"id": "D01", "kind": "drill|cue|check|progression", "targets": ["C01"], "summary": "<英文一句>", "external_goal": "<英文一句或空字符串>", "scenario_fit": "machine_low|mini_tennis|shadow|needs_feeder|match_only", "ban_conflict": {"flag": false, "terms": [], "note": ""}, "intuition_first_ok": false, "evidence": ["Q01"]}]
}

元数据：
{META}

字幕（每行 `[开始秒数] 文本`）：
{TRANSCRIPT}

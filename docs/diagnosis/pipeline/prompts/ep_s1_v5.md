# ep_s1_v5 — 字幕层抽取（Stage 1，权威层）

> v5 相对 v4 的改动（2026-10-18，round5）：v4 的原因粒度规则让 causes 条数稳定在 3–5，但 solutions 一致性仍低（round4 同口径 1-2 配对均值 solutions 0.43；wd4YRQW3TOc / E_zmENJIj4g 的 solutions 配对在 0.00–0.44 之间摆动）——同一个练习/口令被一次拆成相邻两三个桶里的几条、另一次合成一条，且解法总数摆动。改动：
> (1) 新增规则 14「解法粒度规则」：每个 30 秒桶最多一条 solution；相邻桶里讲的同一个练习/口令合并成一条（锚点取第一次给出的那一句）；solutions 总数 3 到 6 条；每条 solution 的 `targets` 至少含一个原因 id（C 开头）；写完后自查；
> (2) 其余（30 秒锚点、原因粒度规则、solution kind 定义、临时字段 `cause_anchors`/`solution_anchors` 不写进卡片）与 v4 相同；
> (3) 规则编号：v4 的 14、15 顺延为 15、16。
> 输入：一期 FTT YouTube 视频的**带时间戳字幕全文** + yt-dlp 元数据（标题、时长、发布日期）。
> 输出：一个符合 `pipeline/schema/episode_card.schema.json` 的 JSON（`status` = `stage1_draft`，`video_layer` = `[]`）。
> 同一期**独立跑 2 次**（新会话、不看上一次结果）；两次的传统一致性（`check_rerun.py cards`）不足时先跑第 3 次再裁决。
> 本提示词**不含任何用户信息**；不要把用户的问题、训练记录、项目口令放进来（`check_neutrality.py` 会查）。判别例句与粒度示例只用抽象的、与具体故障结论无关的写法。

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
10. `solutions`：教练给出的"怎么做"，`kind` 只能是 `drill` / `cue` / `check` / `progression`，定义和判定规则见下面的「solution 的 kind 定义」；`targets` 写它针对哪个原因或症状的 id（或 `PROBLEM`）。`external_goal`：如果教练给了一个**身体外部**的目标（球、拍、场地、锥桶），抄成一句；没有就写空字符串。`scenario_fit`：能否在**低速发球机 / mini tennis / 影子挥拍**里做。`ban_conflict` 和 `intuition_first_ok` 如实填你的判断。
11. `extraction.user_context_in_prompt` 必须是 `false`。
12. **30 秒锚点规则（causes 与 solutions 都适用）**：把字幕按开始秒数分成 30 秒桶，桶号 `b = floor(秒数 / 30)`（例：秒数 0–29 → 桶 0，30–59 → 桶 1，185 → 桶 6）。
    - **先**在 `cause_anchors` 里给每个原因一个锚点 `anchor_s`（教练**第一次清楚说出**这个原因的那一句的开始秒数；不用总结复述处的秒数），并写出 `bucket`（= floor(anchor_s / 30)）；**然后**才填 `causes`。`causes[i].id` 与 `cause_anchors[i].id` 一一对应。
    - 每个原因的 `evidence` **只能引用开始秒数落在该锚点桶里的 quote**（1–2 条即可），不要跨桶引用。如果某原因在别的桶里被复述，不要把复述处加进 evidence。
    - 同理，**先**在 `solution_anchors` 里给每个 solution 一个锚点（教练**第一次给出这条指令**的那一句），再填 `solutions`；每个 solution 的 `evidence` 只能来自它的锚点桶；**同一个桶里最多一条 solution**（该桶里有几个动作时，合并成一条，按主要意图选 kind）；相邻桶的同一练习/口令合并，见规则 14。
    - `quotes` 要保证每个锚点桶里至少有一条 quote。`symptoms`、`angles`、`chain` 不受锚点限制。
    - `chain.evidence` 用被指向原因的 evidence 里的 quote 即可。
13. **原因粒度规则**（只约束 `causes`，目的是让不同次抽取切出同样粒度的原因）：
    - `causes` 写 **3 到 5 个**。字幕里讲了很多个原因时，向上合并成 3–5 个；字幕确实支撑不了 3 个时，以规则 1 为准，不要编造凑数。
    - **同一个 30 秒桶里讲的同一机制，合并成一条原因**（一条原因一个桶、一个锚点）。教练在同一桶里先描述现象、再解释它为什么会这样，只算一个原因，`statement` 写成一句话，把机制写进去。
    - **同一件事不要拆成两个原因**：同一个动作的"做了什么"和"没做什么"、同一个机制的"现象描述"和"机制本身"、同一句话的两种说法，都合并成一条。只有教练把它们当作**两件不同的事**、分别在**不同的 30 秒桶**里讲时，才拆。
    - 教练在别的桶里把同一个原因再讲一遍（复述、举例、总结），**不要新增原因**，也不要把复述处加进 `evidence`（见规则 12）。
    - **每个原因必须能对应到一个 `cause_anchor`**：`cause_anchors` 与 `causes` 条数相同、id 一一对应；写不出锚点（教练第一次清楚说出它的那一句）的原因就不要列。
    - 写完 `causes` 后自查一遍：有没有两条原因的锚点落在同一个桶里？有没有两条原因的 `statement` 说的是同一件事？有，就合并；合并后不足 3 条且字幕里确有别的原因被你合并掉了，就把它们按「不同桶、不同的事」重新拆开。
14. **解法粒度规则**（只约束 `solutions`，目的是让不同次抽取切出同样粒度的解法）：
    - **每个 30 秒桶最多一条 solution**（一个桶、一个锚点、一条 solution）。同一桶里有几个动作，按主要意图合并成一条。
    - **相邻桶里讲的同一个练习或同一句口令，合并成一条 solution**：教练在连续的桶里把同一个练习继续讲下去、补充细节、换个说法重复同一句口令，只算一条；`solution_anchors` 取教练**第一次给出**它的那一句，`evidence` 只引用该锚点桶里的 quote（不要把后面相邻桶的复述加进来）。只有教练转到**另一个不同的练习/口令**时才另起一条。
    - `solutions` 写 **3 到 6 条**。字幕里讲了很多条时，向上合并成 3–6 条；字幕确实支撑不了 3 条时，以规则 1 为准，不要编造凑数。
    - **每条 solution 的 `targets` 至少包含一个原因 id（C 开头）**——它是为解决哪个原因而给的；对应不上任何一个原因的"解法"不要列（可放进 `angles`）。`targets` 里可以再加症状 id 或 `PROBLEM`，但不能只有它们。
    - **每条 solution 必须能对应到一个 `solution_anchor`**：`solution_anchors` 与 `solutions` 条数相同、id 一一对应。
    - 写完 `solutions` 后自查一遍：有没有两条 solution 的锚点落在同一个桶里？有没有两条锚点在相邻桶、说的是同一个练习或口令？有，就合并；合并后不足 3 条且字幕里确有别的、对应某个原因的练习/口令被你合并掉了，就按「不同练习」重新拆开；有没有 solution 的 `targets` 里一个原因 id 都没有？有，就补上对应的原因，补不上就删掉。
15. **solution 的 kind 定义**（按**讲者的主要意图**分类：教练说这段话，是想让学员做什么用？**一段话只标一个 kind**；一条 solution 对应一个 30 秒桶）：
    - `cue` = **一句可以对自己说的口令**：短（通常一句话以内）、现在时/祈使句、挥拍时能在脑子里默念；它不是一个活动，也不是一个判断。
      例句："Say to yourself: 'finish tall.'" / "Think: 'slow hands, fast feet.'" / "Just tell yourself to see the ball longer."
    - `drill` = **可以重复做的练习动作或训练法**：有一个明确的做法和次数/回合/道具/条件，目的是靠重复把动作练出来。
      例句："Do twenty shadow swings with your feet together." / "Have a partner feed ten balls and only hit them on the bounce." / "Hit ten balls in a row while keeping your free hand behind your back."
    - `check` = **自测或判断自己是否做对的检查方法**：重点是"做完后看/问/量一个结果，答案是对或不对"，不是为了练习本身。
      例句："Freeze at the finish and see whether your weight is on your front foot." / "If you can hold the finish for a second, you did it right." / "Film from behind and check if the racket stays inside the line."
    - `progression` = **加难或降难的变体**：明确是在一个已有的练习/口令基础上改条件（更快、更慢、更低、更高、更远、更近、加对手、去掉辅助），或明说"能做到之后下一步 / 太难就先…"。
      例句："Once that feels easy, do it on faster balls." / "If this is too hard, start from a standing position." / "Next, add a moving target."
    - 常见混淆的判定规则（按顺序用）：
      1. **口令 vs 练习**：一句话里夹着口令，但整段在讲"去做一组/一轮某个活动" → 标 `drill`；只有当这段话**本身就是**让你默念的那一句时才标 `cue`。
      2. **检查 vs 练习**：做一个动作"看结果是否出现/是否做对"→ `check`；做同一个动作"为了练熟"→ `drill`。看教练说的目的词：see whether / make sure / if you can … you did it right → `check`；practice / repeat / work on / build → `drill`。
      3. **检查 vs 口令**：问句或判断句（"Is it …?""Does it …?"）→ `check`；祈使句（"Do …""Keep …"）→ `cue`。
      4. **进阶 vs 练习**：教练是在已有某个练习的基础上改难度/条件 → `progression`；是一个全新的独立练习 → `drill`，即使它在视频后半段才出现。
      5. 教练只是在**解释机制、展示对比、讲道理**，没有让学员去做什么 → **不是 solution**，不要列；可放进 `angles`。
      6. 仍然分不清时，取这一个桶里**教练第一句给出的可执行指令**的类型。
16. **只输出一个 JSON 对象，不要任何额外文字，不要 markdown 代码块。字段名必须逐字使用下面骨架里的名字，不要改名、不要加别的字段，枚举值只能取骨架里列出的那些。键的**顺序**也按骨架：先 `cause_anchors` 再 `causes`，先 `solution_anchors` 再 `solutions`。**

输出骨架（`<...>` 处填内容；`|` 表示只能选其一；每个数组按需要写多条，id 依次编号 Q01、Q02…，S01…，C01…，A01…，D01…，三位数 id 不要用）：

{
  "classification": {"is_teaching": "teaching|partial|non_teaching", "stroke": "forehand|backhand|serve|volley|movement|general|other", "format": "lecture|demo_with_player|lesson_review|pro_analysis|short|vlog_other", "reason": "<一句话>"},
  "problem": {"title_as_question": "<英文问句>", "definition": "<一两句英文>", "node_id": "PENDING", "prerequisites": ["<英文短语>"], "target_player": "<英文或空字符串>", "evidence": ["Q01"]},
  "quotes": [{"id": "Q01", "text": "<字幕逐字原话 ≤40 词>", "start": 123, "speaker": "coach|player|unknown", "about": "general|demonstrator_natural|demonstrator_induced|pro_footage"}],
  "symptoms": [{"id": "S01", "description": "<英文一句>", "kind": "visible|ball_outcome|feel", "layer": "A|B|C", "camera": "S|B|R|any|none", "phase": "ready|unit_turn|placement|forward_swing|contact|follow_through|recovery|whole", "obs_ref": "NEW-OBS|NONE", "evidence": ["Q01"]}],
  "cause_anchors": [{"id": "C01", "anchor_s": 123, "bucket": 4}],
  "causes": [{"id": "C01", "statement": "<英文一句>", "depth": 1, "layer": "A|B|C", "node_id": "PENDING", "active_or_passive": "active|passive|unclear", "evidence": ["Q01"]}],
  "chain": [{"from": "C02", "to": "C01", "strength": "coach_stated|coach_implied|inferred", "evidence": ["Q01"]}],
  "angles": [{"id": "A01", "type": "mechanics|feel|visual_model|analogy|contrast_demo|misconception|pro_example", "summary": "<英文一句>", "evidence": ["Q01"]}],
  "solution_anchors": [{"id": "D01", "anchor_s": 123, "bucket": 4}],
  "solutions": [{"id": "D01", "kind": "drill|cue|check|progression", "targets": ["C01"], "summary": "<英文一句>", "external_goal": "<英文一句或空字符串>", "scenario_fit": "machine_low|mini_tennis|shadow|needs_feeder|match_only", "ban_conflict": {"flag": false, "terms": [], "note": ""}, "intuition_first_ok": false, "evidence": ["Q01"]}]
}

元数据：
{META}

字幕（每行 `[开始秒数] 文本`）：
{TRANSCRIPT}

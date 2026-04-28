# 知识组织框架（针对 docs/research/ 233 个 .md 的实操方案）

> 2026-04-27 创建
> 起因：用户问"AK 的 Adaptive Capacity 理论"。先澄清这件事，再给出真正能落地的方案。

---

## 0. 先澄清一件事：AK 没有"Adaptive Capacity"框架

调研后确认：**Andrej Karpathy 没有公开发表过一个叫 "Adaptive Capacity" 的框架**。这个词在学术界有明确归属：

- **Adaptive Capacity** 是 C.S. Holling（1973）和 Brian Walker 等人在**社会-生态系统韧性理论**里的核心概念，指系统在扰动下自组织、学习、调整的能力，与 Resilience、Transformability 并列三要素。属于复杂自适应系统（CAS）领域，跟 AK 无关。
  - 来源：Walker, Holling et al. *Resilience, Adaptability and Transformability in Social-Ecological Systems*（Ecology and Society, 2004）。

用户脑子里大概率是把以下三类东西混在一起了：

1. AK 真正强调过的 **first principles + 自己重建** 的学习方式
2. AK 的 **Software 1.0 / 2.0 / 3.0 + LLM as OS** 类比（讲的是计算范式，不是知识组织）
3. 系统理论里的 **Adaptive Capacity**（讲的是系统韧性，跟教学也不直接相关）

下面只用 AK **真的说过的**话，配 **真正适合我们语境的**框架。

---

## 1. Karpathy 的真实立场：与"知识结构化"相关的 5 条原则

只列有出处、能找到原文的。没找到原文的脑补一律不写。

### 原则 1 · 学习 = 主动重建，不是被动阅读

> "It took me a while to really admit to myself that just reading a book is not learning but entertainment."
> — Karpathy（karpathy.ai/tweets.html）

**对我们的含义**：FTT 书的 28 万字摘录、RTP 33 个视频笔记，如果只是 "读过一遍 + 存档"，等于没学。**必须有一个把它们重写成自己语言的产物**——这正是 `synthesis.md`、`FOREHAND_COMPLETE_TAXONOMY.md` 在做的事。

### 原则 2 · 教 / 总结 = 学的一部分，不是事后

> "teach/summarize everything you learn in your own words"
> — Karpathy（karpathy.ai/tweets.html）

**对我们的含义**：每个 FTT 视频分析文件不应该只是字幕摘录，应该有一段"用我自己的语言重述"。这跟 **Feynman Technique** 完全一致：能用大白话讲明白才算懂。

### 原则 3 · 永远先预测，再吸收

> "Ideally never absorb information without predicting it first. Then you can update both 1) your knowledge but also 2) your generative model."
> — Karpathy（karpathy.ai/tweets.html）

**对我们的含义**：看新视频前，先问自己"我猜这个教练会强调什么？" 看完再对账。这能直接落地到我们的 VLM 工作流——**让 VLM 先生成假设，再用视频验证**，而不是直接出诊断。

### 原则 4 · 项目驱动，深度优先

> "iteratively take on concrete projects and accomplish them depth wise, learning 'on demand'"
> — Karpathy（karpathy.ai/tweets.html）

**对我们的含义**：用户的训练日志（learning.md）就是一个个 "concrete project"——每次训练都是一个具体问题（4/13 Unit Turn、4/26 Bounce-Hit）。知识库的所有文件都应该最终能**回答某次训练里的某个具体问题**。读不上的文件 = 暂时无效。

### 原则 5 · "Shortification" 是错觉

> "Actual learning of anything worth learning takes time and focused mental effort"
> — Karpathy（X / Twitter，被广泛转载）

**对我们的含义**：233 个 .md 不是问题，问题是**没有一条主线把它们串起来**。简短的 "10 条要点" 只是 illusion of learning。我们需要的不是把 233 文件压缩成 1 个 cheatsheet，而是**有一个能从任意问题进入、深挖到任意细节的导航结构**。

> AK 真正讲过 "OS"、"Software 3.0"、"LLM = people spirits"，但那些是讲 LLM 计算范式的（latent.space/p/s3，X 帖 1935518272667217925），跟知识组织没有直接关系。**别套用**。

---

## 2. 三个真正适合我们的框架（组合使用）

### 框架 A · Zettelkasten（卡片盒法）—— 解决"链接缺失"

**是什么**：Niklas Luhmann 用一辈子积累 9 万张卡片的方法。两条核心原则：

1. **Atomicity（原子性）**：一张卡 = 一个想法。不要把"Unit Turn"和"击球点"塞同一个文件。
2. **Bidirectional Linking**：卡片不靠文件夹组织，靠**互相引用**。任意一条概念可以从多个上下文进入。

**为什么适合我们**：我们当前的 233 个文件里，**很多是粗粒度的**——`13_synthesis.md` 945 行，里面混着握拍、Unit Turn、击球点、kinetic chain 全部内容。当用户问"我今天 Unit Turn 又出问题了"，我得线性扫 945 行才能找到相关段落。这是**反 atomicity 的**。

**怎么用**（具体动作）：
- 不需要把 233 文件全打散——成本太高。
- **新增一层 atomic concept notes**，放在 `_meta/concepts/`，每个文件 50-150 行，只讲一个概念（如 `unit_turn.md`、`contact_point.md`、`accelerate_late.md`）。
- 每个 atomic note 在末尾给出**反向索引**：哪些 source 文件提到过它（FTT 第几节、RTP 哪个视频、TPA 哪个）。
- source 文件不动，只在 atomic note 里建链接。

来源：[Zettelkasten.de Atomicity Guide](https://zettelkasten.de/atomicity/guide/)、[Introduction to Zettelkasten](https://zettelkasten.de/introduction/)

### 框架 B · First Principles + Bloom's Psychomotor Taxonomy —— 解决"分层缺失"

**是什么**：
- **First Principles**（Musk + Karpathy 都强调）：把一个复杂问题拆到最小不可还原的物理事实，不用类比。
- **Simpson's Psychomotor Taxonomy**（Bloom 体系的运动技能版，1972）：7 级——Perception → Set → Guided Response → Mechanism → Complex Overt Response → Adaptation → Origination。

**为什么适合我们**：网球技术学习是**心理-运动**任务，不是认知任务。Bloom 原版的 Remember→Understand→Apply→Analyze→Evaluate→Create 不直接适用，但 Simpson 版本完全对得上用户的训练阶段：
- 用户当前在 **Mechanism → Complex Overt Response** 之间徘徊（4/13 Unit Turn 还在 Set 阶段，加速基座已到 Mechanism）
- 不同概念在不同 Bloom 层级，**不能用统一深度去组织**。

**怎么用**：
- 在 `FOREHAND_COMPLETE_TAXONOMY.md` 的 10 层物理维度之外，**加一个"用户掌握度"维度**。每个核心概念标注用户当前在 Simpson 哪一级。
- atomic note 里写 "First Principle" 段落：这个动作在物理层面到底是什么？（如 Unit Turn = 髋胯-肩-拍三段角动量传递的启动相位，不是"转身"）

来源：[Simpson's Psychomotor Domain](https://en.wikipedia.org/wiki/Bloom's_taxonomy)、Karpathy on physics-style first-principles thinking（dwarkesh 访谈）

### 框架 C · Feynman Technique —— 解决"理解假象"

**是什么**：选一个概念 → 用大白话讲给一个 12 岁小孩 → 卡壳的地方就是知识漏洞 → 回去补 → 再讲一遍。

**为什么适合我们**：用户已经在做了——`learning.md` 里 11 字 mantra、压飘藏顶、加速基座/减速基座，全是用户自己用大白话压出来的。但**这些用户认知散落在训练日志里，没有跟 233 个 source 文件互相挂钩**。

**怎么用**：
- 用户认知（11 字 mantra 等）应该单独放一个 `_meta/user_mental_models/` 目录。
- 每条认知双向链接到：(1) 它压缩了哪些 source 文件 (2) 它在哪几次训练记录里被验证。
- VLM 输出诊断时**优先引用用户已建立的认知词汇**，而不是 FTT 原版术语。

来源：[Farnam Street: Feynman Technique](https://fs.blog/feynman-technique/)

---

## 3. 针对 233 个 .md 的具体动作清单

### 3.1 顶层入口：`docs/research/README.md`（目前不存在，必须建）

应该长这样（**不超过 150 行**）：

```markdown
# Tennis Forehand Knowledge Base

## 我现在想干嘛？
- [我刚训练完，要复盘] → docs/record/learning.md（最近一次）
- [我有一个具体问题] → _meta/concepts/INDEX.md（按概念查）
- [我想看完整体系] → 13_synthesis.md（FTT 主线）+ FOREHAND_COMPLETE_TAXONOMY.md（10 维度）
- [我想看某个教练的所有视频] → ftt_video_analyses/ / road_to_pro_video_analyses/ / tomallsopp_video_analyses/ / feeltennis_video_analyses/
- [我在调试 VLM/诊断引擎] → coach_analysis/v4.2_INTEGRATION_SUMMARY.md + diagnostic_chains/

## 知识结构（三层）
1. **Source 层**（不动）：01-30 编号文件、video_analyses/ 子目录——原始研究
2. **Synthesis 层**（已有）：synthesis.md、TAXONOMY.md、kinetic_chain_synthesis.md——按主题压缩
3. **Atomic 层**（要建）：_meta/concepts/——每个概念一个文件，双向链接

## 用户当前状态（4/27）
（链接到 ~/.claude/projects/-Users-qsy-Desktop-tennis/memory/user_tennis_learner.md 的最新状态）
```

### 3.2 维度补充：在现有 10 层之外加 3 个

`FOREHAND_COMPLETE_TAXONOMY.md` 现有 10 层是**纯物理维度**。建议加：

| 新维度 | 含义 | 为什么需要 |
|---|---|---|
| **Bloom 掌握度** | 该概念在用户身上处于 Simpson 哪一级 | 决定 VLM 该用"教学"还是"提醒"语气 |
| **认知词汇** | 用户用什么大白话讲它 | VLM 应该用"压飘藏顶"而不是"closed stance pronation" |
| **诊断 vs 训练** | 这是观察性指标还是训练性指令 | 区分"VLM 看到什么"和"教练让你做什么" |

### 3.3 该合并的、该拆的、该加链接的

**该合并**（同一概念散落多处）：
- `arm_body_integration_solutions.md` + `arm_dominance_history.md` + `arm_trunk_connection_tips.md` + `arm_trunk_coupling_biomechanics.md` + `shoulder_dominance_fix.md` + `forearm_compensation_analysis.md` → 这 6 个全是"手臂脱离躯干"主题，应该**合并成一个 atomic concept**：`_meta/concepts/arm_trunk_decoupling.md`，原 6 个降级为 source reference。

**该拆**（一个文件混多个原子概念）：
- `13_synthesis.md`（945 行）至少包含 8 个原子概念（握拍/Unit Turn/Backswing/Forward Swing/接触/Follow-through/kinetic chain/常见错误），应该**保留它作为 master synthesis**，但把每个二级标题的内容**抽出来生成一个 atomic note**。
- `FOREHAND_COMPLETE_TAXONOMY.md`（422 行）的 10 层 = 10 个原子概念，每层应该有独立的 atomic note。

**该加双向链接**：
- 用户的 `docs/record/learning.md` ←→ 触发该思考的 source 文件（4/13 Unit Turn 应该链接到 30_unit_turn_hip_rotation.md 和 ftt_passive_arm_unit_turn.md）。
- 诊断链 `diagnostic_chains/*.md` ←→ 它依赖的 atomic concepts。
- VLM prompt 模板 ←→ 它使用的概念词汇表。

### 3.4 操作优先级（不要一次全做）

| 优先级 | 动作 | 工作量 | 收益 |
|---|---|---|---|
| P0 | 写 `docs/research/README.md` 顶层入口 | 30 分钟 | 立即解决"找不到东西"问题 |
| P0 | 把 6 个 arm_xxx 文件合并成 `_meta/concepts/arm_trunk_decoupling.md` | 1 小时 | 消除最严重的概念碎片 |
| P1 | 给 10 层 taxonomy 每层抽出 atomic note（10 个文件） | 3 小时 | 建立 atomic 层骨架 |
| P1 | 给用户 mantra/压飘藏顶/加速基座等 5-8 条认知建独立文件 + 双向链接 | 2 小时 | 把"用户语言"系统化 |
| P2 | 给 233 个 source 文件加 frontmatter（concepts:, level:, status:） | 5+ 小时 | 全文检索能按概念过滤 |

P0 + P1 ≈ 6 小时，能 80% 解决问题。P2 是 nice-to-have。

---

## 4. VLM Prompt 重写的元原则

把上面 4 个框架（AK + Zettelkasten + First Principles + Feynman）压成 5 条 VLM 输出质量准则：

### 准则 1 · 单根因（First Principles）

VLM 不应该输出"5 个问题清单"。每次只指认 **1 个根因**——它通常在 `FOREHAND_COMPLETE_TAXONOMY.md` 的 Layer 1-3（地基层）。Layer 4-6 的问题往往是 Layer 1-3 的症状，修上面没用。

> 反例：VLM 说"你 Unit Turn 晚 + 接触点低 + 手臂主导 + 没有 lag"——这是 4 个问题但可能只有 1 个根因（接触点低 → 时间不够 → 临时手臂救球）。

### 准则 2 · 用户语言优先（Feynman / 用户认知）

VLM 看到"closed stance pronation with armpit lock"时，应该输出"压飘藏顶 + 锁门"——**用户已建立的词汇**。原版 FTT 术语只在用户没建立对应认知时才用。

> 配套：建一份 `_meta/user_vocabulary.md`，VLM prompt 里强制注入。

### 准则 3 · 可验证（Karpathy 原则 4 · 项目驱动）

每条诊断必须配 **可在下次训练验证的检查点**。"调整你的 Unit Turn"不算诊断，"下次拍球弹地瞬间停画面，看肩有没有转过 90°"才算。

### 准则 4 · 不绕（AK 原则 5 · 反 Shortification）

不要用"全方位"、"综合"、"多维度"这种词把没看清的东西包装成结论。看不清就说看不清。**没诊断 ≠ 失败**，**假诊断 = 灾难**。

### 准则 5 · 区分主动 vs 被动结果（项目 CLAUDE.md 已写）

正手是旋转鞭打系统。**主动动作**（Unit Turn、加速基座下压）是用户能直接控制的；**被动结果**（lag、wiper finish、racket drop）是物理涌现的。VLM 不能让用户"主动制造 lag"——这是反物理的。

---

## 5. 一句话总结

> **不要把 233 个文件再压成 1 个总结文档**（那是 shortification）。
> **要在它们之上加一层 atomic concepts + 双向链接 + 用户词汇表**，让从任意一个具体训练问题都能 3 跳之内找到对应的物理事实和验证方法。
>
> AK 真正强调的是：**主动重建 > 被动阅读、教 = 学、预测先于吸收、深度项目驱动**。这 4 条比任何"框架"都更接近他的实际方法论。

---

## 来源

- Karpathy 引述：[karpathy.ai/tweets.html](https://karpathy.ai/tweets.html)、[X @karpathy](https://x.com/karpathy/status/1935518272667217925)、[Latent Space S3 Talk](https://www.latent.space/p/s3)
- Zettelkasten：[zettelkasten.de/atomicity/guide](https://zettelkasten.de/atomicity/guide/)、[zettelkasten.de/introduction](https://zettelkasten.de/introduction/)
- Bloom / Simpson：[Wikipedia Bloom's Taxonomy](https://en.wikipedia.org/wiki/Bloom's_taxonomy)
- Feynman Technique：[fs.blog/feynman-technique](https://fs.blog/feynman-technique/)
- Adaptive Capacity（澄清，非采用）：Walker, Holling et al. 2004 *Resilience, Adaptability and Transformability*；[Wikipedia: Complex Adaptive System](https://en.wikipedia.org/wiki/Complex_adaptive_system)

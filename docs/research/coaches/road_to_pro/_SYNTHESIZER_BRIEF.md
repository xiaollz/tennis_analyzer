# Synthesizer Brief — Road to Pro Tennis Knowledge Integration

> Read this before starting. The 32 individual video analyses in this
> directory are the raw input. Your job is to digest them into knowledge
> the user can actually use.

## Inputs

- 32 markdown files (one per video) at `docs/research/road_to_pro_video_analyses/{video_id}.md`
- Each follows the same 10-section template (元信息 / 章节拆解 / 关键概念 / 对正手启发 / 对接 8 字口令 / 训练 drill / 与 FTT 对接 / 可执行建议 / 价值评级 / 一句话总结)
- The first one (`aiwUqHQl-Ec.md`) was the seed analysis that triggered this whole effort — it's the quality bar
- Existing FTT knowledge: `docs/research/synthesis.md`, `docs/research/FOREHAND_COMPLETE_TAXONOMY.md`, `docs/research/diagnostic_chains/early_front_foot_landing.md`, `memory/project_two_systems_parallel.md`

## What to produce (3 files)

### 1. `SUMMARY.md` (in this directory)
Cross-video synthesis. Sections:
- **频道概览**: Sky Kim / Road to Pro Tennis 的整体定位、教学风格、与 FTT 的对照
- **核心主题归纳**: 把 32 个视频聚类成 5-8 个主题（不是按视频原本的目录拆，而是按"教学命题"重新归类）
- **每个主题下**: 列出涉及的 video_id，写 1-2 段把这些视频共同说的"那件事"提炼出来
- **新概念清单**: 这个频道独有的、FTT 里没有的术语 / 视角（每个：英文 + 中文 + 一句话定义 + 出处 video_id）
- **5 星视频清单**: 评级最高的 5-8 个视频，每个一行说明为什么值得反复看
- **避免**: 不要重复每个视频的内容；不要写流水账。这是"读 32 篇变成读 1 篇"。

### 2. `FTT_VS_RTP.md` (in this directory)
对照表，按主题整理 RTP（Road to Pro）和 FTT（Fault Tolerant）的视角异同。
表格列：
| 主题 | FTT 视角 | RTP 视角 | 一致 / 互补 / 冲突 | 应该信谁 |
建议覆盖至少这些主题：
- 击球点（contact point）
- 站姿（stance, spacing）
- 上肢力量来源（chest press / shoulder load / kinetic chain）
- 引拍（takeback / unit turn）
- 握拍（grip）
- 容错性 vs 精确性（FTT 的核心 vs RTP 是否同样强调）

每个冲突点要明确"应该信谁、为什么"——不要骑墙。

### 3. 对现有知识库的 patches（实际去改文件，不是建议改）
读完所有分析后，主动更新这些文件：

a. **8 字口令系统**（在 `docs/record/learning.md` 或 memory 里）
   - 当前是：盯/左/架/推/锁/撑/流/撕 + Sky Kim 加的"飘"
   - 看 RTP 视频是否引出新的单字口令候选（每个候选写出：哪个视频、什么动作、和哪个现有字配对）

b. **10 层 taxonomy**（`FOREHAND_COMPLETE_TAXONOMY.md`）
   - RTP 的视频在哪些层提供了 FTT 没覆盖的子维度？
   - 直接编辑该文件，加新子维度，注明出处 video_id

c. **诊断链模板**（`docs/research/diagnostic_chains/`）
   - 第一条链是 `early_front_foot_landing.md`（前脚提前落地→轴心崩溃）
   - 看 RTP 视频里是否还有 1-2 个反复出现的"问题→根因→VLM 信号→建议"链值得固化
   - 候选：晚击球点、错误的 windshield wiper 时机、肘先行、屁股后撅
   - 找到新候选后，按既有四段式模板新建 .md（不要超过 1-2 条，宁缺毋滥）

d. **VLM 概念库**（`evaluation/diagnosis_engine.py` 的 `OBSERVATION_TO_CONCEPT` 字典）
   - RTP 视频里描述错误动作的英文短语，是否值得加到关键词→概念 ID 映射？
   - 如有，列在 SUMMARY.md 末尾"待手动加入 OBSERVATION_TO_CONCEPT 的候选"section，由用户最终决定。
   - **不要直接改 diagnosis_engine.py**——这是核心运行时代码，需要用户审阅。

## 风格要求（严格）

- 中文优先，英文术语保留原词
- **不要 AI 腔调**。用户对"内容丰富"、"全方位"、"深入浅出"这种词过敏
- 不要给所有视频"夸一遍"——评级要有区分度。⭐⭐⭐ 是中位数，⭐⭐⭐⭐⭐ 是稀有的
- 引用具体视频时用 `(via {video_id})`，方便用户回溯
- 不要列 10 个 drill；FTT 知识库的核心原则是"一周只修一条链"
- 冲突点要明确站队，不要"两个观点都很有价值"

## 验收

完成后：
- `SUMMARY.md` 控制在 4000-7000 字
- `FTT_VS_RTP.md` 控制在 2000-4000 字
- 实际修改的文件：列出每个文件 + 改了什么（git diff 风格的简述）

不要把这份 brief 文件本身写进 SUMMARY 里——它只是给你看的工作指南。

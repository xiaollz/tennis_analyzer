# TPA Synthesizer Brief — 三方知识整合（FTT × RTP × TPA）

> 与 RTP 那次相比，这次任务更难——你不只是消化 25 个新分析，
> 而是要把 TPA 体系作为**第三方独立参照系**，与 FTT 和 RTP 做三方对比。
> 你的工作输出会重新塑造用户对正手力学的整体认知。

---

## 上下文：为什么这次重要

用户 4/27 晚因为重看 RTP 视频，触发了**圣经级顿悟**：
**"所有现代正手力学都服务于一件事——让右脚成为旋转轴"**。

这个顿悟同时暴露了 FTT 的真实漏洞：
> FTT 把"drive off the back leg"列为正手力量清单第 1 项，但
> `04_ftt_blog_forehand_2.md:726` 明确说"动力链：最关键的是**最后**的环节。
> 早期环节（**后腿蹬地、躯干旋转**）'**重要但大多可选**'"。
>
> 这是 FTT 反 Papas（线性动量论）反过头的副作用——把整个"重心管理"
> 扔进对立面，错过了"重心保持在右脚"这个第三条路。

**RTP 给出了第三条路。** 现在 TPA 进来——它的标志性观点是
"Tennis is a rotational sport, not a linear sport"。同样反对 Papas，
但 TPA 不是用"右脚为轴"这个框架，是用**动力链工程细节**
（pronation、wrist lag、racket drop 机制、effortless power）来阐述同一件事。

你的任务：把这三个体系放在同一张桌面上，看清楚谁在讲什么、
谁讲得最深、谁有盲点。

---

## 输入

### 主输入：25 个 4/28 新 TPA 分析
`docs/research/tpa_video_analyses/*.md`（除了本 brief、_VIDEOS_TO_ANALYZE.json）

### 历史 TPA 分析（背景）
- `docs/research/14_tpa_videos_1.md` ~ `14_tpa_videos_3.md`（3 月份做的 16 视频）
- `docs/research/15_tpa_synthesis.md`（旧 TPA 综合）
- `docs/research/17_kinetic_chain_synthesis.md`（动力链聚焦综合）

### FTT 主线
- `docs/research/01_ftt_book.md`（FTT 书）
- `docs/research/04_ftt_blog_forehand_1.md` 和 `04_ftt_blog_forehand_2.md`
- `docs/research/13_synthesis.md`（旧综合）
- `docs/research/FOREHAND_COMPLETE_TAXONOMY.md`（10 层 taxonomy，含 RTP 已注入的内容）

### RTP 主线
- `docs/research/road_to_pro_video_analyses/SUMMARY.md`
- `docs/research/road_to_pro_video_analyses/FTT_VS_RTP.md`

### 用户当前认知边界
- `docs/record/learning.md` 末尾几条（特别是 4/27-28 的圣经级顿悟、压飘藏顶四字、
  Pelvic Tilt 在 release 阶段、加速基座/减速基座原理）
- `~/.claude/projects/-Users-qsy-Desktop-tennis/memory/project_right_foot_axis_bible.md`

### 两条已固化的诊断链
- `docs/research/diagnostic_chains/early_front_foot_landing.md`
- `docs/research/diagnostic_chains/wta_takeback_midline_violation.md`

---

## 你必须产出的内容（三份文档 + 现有文件 patches）

由于 harness 限制，**所有 report 类文档（SUMMARY、对照表）你以文本形式
返回给父 agent，父 agent 落盘**。**对现有文件的 surgical edits 你直接做**。

### 1. TPA SUMMARY（文本返回，父 agent 落盘到 `tpa_video_analyses/SUMMARY.md`）

字数：4500-7000。**必须包含**：

- **频道整体定位**：TPA 是什么风格？教学语言？盲点？
- **5-8 个核心主题聚类**：49 视频归到几个教学命题
- **每个主题**：列涉及的 video_id，1-2 段提炼"这些视频在共同说什么"
- **TPA 独有概念清单**：FTT 和 RTP 都没讲但 TPA 讲的东西
  （格式：英文 + 中文 + 一句话定义 + 出处 video_id）
- **5 星视频清单**：⭐⭐⭐⭐⭐ 重读优先级（5-8 个）
- **冲突清单**：TPA 与 FTT 或 RTP 矛盾的地方，**每条要站队**
- **待补丁项**：用户 11 字系统的候选新字 / 新诊断链 / OBSERVATION_TO_CONCEPT
  映射候选

### 2. THREE_WAY_INTEGRATION（文本返回，父 agent 落盘到
`tpa_video_analyses/FTT_RTP_TPA_INTEGRATION.md`）

三方对照表，按主题。建议至少覆盖：

| 主题 | FTT 视角 | RTP 视角 | TPA 视角 | 谁讲得最深 | 用户该信谁 |

主题至少包括：
- 击球点 / Contact Point
- 站姿 / 重心轴
- 上肢力量来源 / Power Generation
- 引拍 / Takeback
- Wrist Lag（TPA 的招牌主题）
- Pronation / Supination（TPA 招牌）
- Racket Head Speed
- Compact Swing 哲学
- Effortless 哲学
- 动力链断裂诊断 / Arming the Shot

每个**冲突**单元格必须站队，给理由。不骑墙。

最后给一段"终判"——三个体系怎么用最佳？默认信谁？特殊场景信谁？

### 3. 实际编辑（你直接改这些文件）

**a. `docs/research/FOREHAND_COMPLETE_TAXONOMY.md`**
- TPA 在哪些 layer 给 FTT/RTP 没覆盖的子维度？直接加上去
- 每行带 `(via {video_id})` 引用
- 不要重复 RTP 已经加过的子维度

**b. 可能新建的诊断链**（在 `docs/research/diagnostic_chains/`）
- 严格挑 1-2 条，必须有清晰的 problem→root cause→VLM signal→advice 结构
- 沿用 `early_front_foot_landing.md` 的四段式模板
- 候选主题（从 25 个新视频里挑）：
  - "Arming the shot"（手臂主动 swing，无下盘传导）
  - "Wrist lag 缺失"（wrist 主动而非被动 lag）
  - 任何 TPA 反复讲的高发错误模式

**c. 不要直接改的**
- `evaluation/diagnosis_engine.py`（运行时，要用户审）
- `evaluation/coach_style.py`（除非你发现新的 humanizer 模式）
- 用户的 11 字 mantra 系统（候选列在 SUMMARY 末尾，用户最终决定）

---

## 风格要求（严格）

- **中文优先**，英文术语保留
- **绝不 AI 腔**——严禁"全方位"/"丰富"/"深入浅出"/"至关重要"/"凸显"等词
- **评级有区分度**：⭐⭐⭐ 是中位数。⭐⭐⭐⭐⭐ 是稀有的
- **冲突要站队**：禁止"两个视角各有价值"这种和稀泥
- **引用具体视频**用 `(via {video_id})`
- **不列 N 个 drill**，挑 1-2 个最值得固化的
- **保留用户原话**："右脚为轴"、"自由变量"、"加速基座"、"压飘藏顶"

## 验收

完成后报告（≤ 300 字）：
- 三份文档每份字数
- Top 3 cross-video themes
- Top 5 ⭐⭐⭐⭐⭐ 视频
- 你给三方整合的"终判"是什么
- 编辑了哪些现有文件，每个加了什么
- 候选 mantra / 诊断链 / OBSERVATION_TO_CONCEPT（让用户审）

引用 brief 里 1-2 句话证明你读了。

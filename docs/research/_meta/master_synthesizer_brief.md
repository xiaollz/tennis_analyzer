# 主综合 Synthesizer Brief：7 体系正手知识整合

> 这是 Phase 5 of 6。任务是把 4 个新增 P0 频道的 30 个视频分析消化成
> 跨体系的整合框架，**与已有 3 体系（FTT/RTP/TPA）做 7 路对比**，
> 产生用户希望的"网上无法复制的正手框架"。

---

## 背景

用户授权 10 小时投入，目标构建独家正手知识库 + 诊断根因框架。

**已有 3 体系**（已完整深度分析+ 综合）：
- FTT (The Fault Tolerant Forehand)：地图——10 层 taxonomy + 容错性公理
- RTP (Road to Pro Tennis / Sky Kim)：GPS——可视觉验证的脚-肘-肩物理标志
- TPA Tennis (Tom Allsopp)：发动机说明书——动力链时序 + 肌肉张力管理

**新增 4 体系**（这次新分析）：
- **Brian Gordon**（4 视频）：3D 动捕硬科学派——Type 3 forehand、内旋占 RHS ~50%
- **Tomaz Mencinger / Feel Tennis**（0 视频，API 卡住，待补）：Feel + Kinetic Chain proprioception 派
- **Nikola Aracic / Intuitive Tennis**（9 视频）：Pro Stroke Frame-by-frame + "intuitive 非思考"哲学
- **Florian Meier / Online Tennis Instruction**（9 视频）：Junior Dev 教练 + ATP 慢动作
- **RacquetFlex**（8 视频）：Effortless Power + Lag/Snap + Injury Prevention 派

**用户认知边界**（最新）：
- 圣经级顿悟（4/27 晚）：右脚为轴 = 一切；用户用"自由变量"描述开放动力链
- 11 字 mantra：盯/左/架/推/锁/撑/流/撕/飘/藏/压
- 三条诊断链：early_front_foot_landing、wta_takeback_midline_violation、arming_the_shot_false_lag

---

## 输入清单

**必读的新分析**（30 个）：
- `docs/research/brian_gordon_video_analyses/*.md`（4 个）
- `docs/research/intuitive_tennis_video_analyses/*.md`（9 个）
- `docs/research/online_tennis_instruction_video_analyses/*.md`（9 个）
- `docs/research/racquetflex_video_analyses/*.md`（8 个）

**必读的已有综合**：
- `docs/research/road_to_pro_video_analyses/SUMMARY.md`
- `docs/research/tpa_video_analyses/SUMMARY.md`
- `docs/research/tpa_video_analyses/FTT_RTP_TPA_INTEGRATION.md`（这次的扩展版要替代它）
- `docs/research/FOREHAND_COMPLETE_TAXONOMY.md`
- `docs/research/13_synthesis.md`（FTT 主线综合）

**用户认知边界**：
- `docs/record/learning.md` 末尾（4/27-29）
- `docs/research/diagnostic_chains/*.md`（3 条链）

---

## 产出（5 件事）

### 1. 4 个频道级 SUMMARY（文本返回，父 agent 落盘）

每个频道一份，约 1500-2500 字，落到 `docs/research/{slug}_video_analyses/SUMMARY.md`。

每份必须包含：
- 频道定位 + 教学语言风格 + 盲点
- 3-5 个核心主题聚类
- 频道独有概念清单（其他 6 体系没有的）
- ⭐⭐⭐⭐⭐ 视频清单（按价值排，要诚实区分度）
- 与其他体系的关系

### 2. MASTER_FOREHAND_FRAMEWORK.md（最重要！文本返回，父 agent 落盘）

约 6000-9000 字。**这是 7 体系（FTT/RTP/TPA + 4 新增）的总整合**，是用户期待的核心产出。

落到 `docs/research/_meta/MASTER_FOREHAND_FRAMEWORK.md`。

必须包含：

**第一部分：7 体系总览**
表格：每个体系一行，列：mental model 一句话、最强领域、最大盲点、不可替代价值。

**第二部分：N 主题对照表（替代原有 FTT_RTP_TPA_INTEGRATION）**
扩展现有 10 主题（击球点 / 站姿 / 上肢力量 / 引拍 / Lag / Pronation / RHS / Compact / Effortless / Arming）每个主题给 7 列对照。冲突明确站队。

**第三部分：新发现的真空区**
4 新体系**揭示的、之前 3 体系没碰过**的领域。比如：
- Brian Gordon 的内旋数据可能改写 RHS 来源公式
- RacquetFlex 的 injury prevention 视角是否是真空区？
- Intuitive Tennis 的 "non-thinking" 哲学是否补 FTT 容错论？

**第四部分：用户当前认知边界 vs 7 体系**
对照用户已建立的圣经级顿悟、11 字、3 诊断链——
- 哪些被新体系**强化**？（多体系印证 = 高置信）
- 哪些被新体系**修正**？（与用户既有认知冲突的）
- 哪些是新体系**带来的全新角度**？

**第五部分：用户实战中怎么用最佳（按场景）**
扩展现有 INTEGRATION 文档的"默认信谁"决策表到 7 体系。

**第六部分：终判**
用户当前阶段（圣经级顿悟+三条诊断链固化期），新体系应该怎么进。
最高优先级是哪个？哪个先放着不动？

### 3. 直接编辑 `docs/research/FOREHAND_COMPLETE_TAXONOMY.md`

加新子维度，每行带 `(via {video_id})` 引用。
重点：Brian Gordon 的硬数据应该影响哪些 layer？
Intuitive Tennis 的 frame-by-frame 视角是否补 L8（视觉/注意力）？

### 4. 候选新诊断链（最多 1-2 条）

如果新体系揭示了一条**反复出现的"问题→根因→VLM 信号→建议"链**，
新建在 `docs/research/diagnostic_chains/`，用四段式模板。

候选主题（从新分析里挑）：
- "Trying to consciously think the swing"（Intuitive 的反命题）
- "Internal rotation deficit / 内旋幅度不足"（Brian Gordon 数据）
- "Late kinetic chain firing / 动力链时序错乱"（RacquetFlex）

宁缺毋滥。如果不够"清晰可固化"就不建，只在 SUMMARY 里列候选。

### 5. 候选 mantra / OBSERVATION_TO_CONCEPT（不直接改）

列在 MASTER 文档末尾，让用户决定是否纳入：
- 11 字系统的新候选字（每个标 video_id 出处 + 推荐/保留意见）
- VLM `OBSERVATION_TO_CONCEPT` 候选（关键词 → 概念 ID + severity）

---

## 风格要求（严格）

- **中文优先**，英文术语保留
- **绝不 AI 腔**——禁用"全方位"/"丰富"/"至关重要"/"凸显"等词
- **评级有区分度**：⭐⭐⭐ 是中位数。⭐⭐⭐⭐⭐ 稀有
- **冲突要站队**：禁止"两个视角各有价值"和稀泥
- **保留用户原话**："右脚为轴"、"自由变量"、"加速基座"、"压飘藏顶"
- **humanizer 风格**：教练讲人话不是 AI 报告
- 不要把 brief 本身写进任何 SUMMARY

## 验收

报告（≤ 400 字）：
- 5 份产出每份字数
- Top 3 跨体系新发现（4 新体系揭示的真空区）
- Top 5 ⭐⭐⭐⭐⭐ 视频
- 7 体系整合的"终判"是什么
- 候选 mantra 推荐哪一个？候选诊断链建了几条？

引用 brief 1-2 句证明读了。

# 冗余审计 (2026-04-28)

> 审计范围：`/Users/qsy/Desktop/tennis/docs/research/`，共 233 个 .md，3.1 MB（实测，非 2.79 MB）。
> 审计方法：扫目录结构 → 抽样阅读高度疑似文件的开头/TOC/重复段落 → 比对日期与基线引用关系。
> **本审计不删除任何文件，只列清单。最终保留/删除/归档由人工决定。**

---

## 类别 1: 完全重叠（建议删除约 24 个文件，~210 KB）

这一类是同一份内容存在两个副本，新版完整覆盖旧版。

### 1.1 TPA 视频：12 个视频在 `tomallsopp_video_analyses/` 和 `tpa_video_analyses/` 同时存在

`tpa_video_analyses/`（2026-04-28，新批 25 个）重新分析了 12 个 4/3 分析过的视频，新版每个文件 8-15KB，旧版每个 6-7KB，内容更深、更接近用户当前认知。

| 保留（新批 4/28） | 删除（旧批 4/3） | 文件大小对比 |
|---|---|---|
| `tpa_video_analyses/UVrZoQ70wxU.md` | `tomallsopp_video_analyses/UVrZoQ70wxU.md` | 15068B vs 6945B |
| `tpa_video_analyses/ubFJi2M3AMM.md` | `tomallsopp_video_analyses/ubFJi2M3AMM.md` | 10698B vs 7505B |
| `tpa_video_analyses/dw4hymptl9k.md` | `tomallsopp_video_analyses/dw4hymptl9k.md` | 10883B vs 7727B |
| `tpa_video_analyses/FtGqOcmlWLY.md` | `tomallsopp_video_analyses/FtGqOcmlWLY.md` | — |
| `tpa_video_analyses/LU9yamZPOnw.md` | `tomallsopp_video_analyses/LU9yamZPOnw.md` | — |
| `tpa_video_analyses/M1umUwuPe0w.md` | `tomallsopp_video_analyses/M1umUwuPe0w.md` | — |
| `tpa_video_analyses/O1i9y5NSoig.md` | `tomallsopp_video_analyses/O1i9y5NSoig.md` | — |
| `tpa_video_analyses/OBjVdy1MS44.md` | `tomallsopp_video_analyses/OBjVdy1MS44.md` | — |
| `tpa_video_analyses/azVf6CyDfVk.md` | `tomallsopp_video_analyses/azVf6CyDfVk.md` | — |
| `tpa_video_analyses/ftyfZXr3Zcw.md` | `tomallsopp_video_analyses/ftyfZXr3Zcw.md` | — |
| `tpa_video_analyses/muxc0h0YAJg.md` | `tomallsopp_video_analyses/muxc0h0YAJg.md` | — |
| `tpa_video_analyses/wWWDqBKwO3U.md` | `tomallsopp_video_analyses/wWWDqBKwO3U.md` | — |

**节省**：12 文件 × 7KB ≈ **~84 KB**。

注：另外 37 个 `tomallsopp_video_analyses/` 文件没有对应的新版，建议保留（它们是 4/3 唯一来源）。

### 1.2 `09_ftt_videos_1/2/3.md` ↔ `ftt_video_analyses/`（同一批 30+ 视频，两种聚合方式）

`09_ftt_videos_1.md` (49KB) + `09_ftt_videos_2.md` (48KB) + `09_ftt_videos_3.md` (75KB) = 三个 monolithic 文件，**内容**与 `ftt_video_analyses/` 目录下 39 个 per-video 文件等价（抽样 `wd4YRQW3TOc` 段落开头：完全一致）。后者按视频 ID 拆分，更易索引。

| 保留 | 删除 | 理由 |
|---|---|---|
| `ftt_video_analyses/*.md`（39 个 per-video） | `09_ftt_videos_1.md` `09_ftt_videos_2.md` `09_ftt_videos_3.md` | 同内容，per-video 更易引用且 12_synthesis 已抽取核心 |

**节省**：3 文件 × ~57KB = **~172 KB**。

### 1.3 `FTT_VS_RTP.md` 已被 `FTT_RTP_TPA_INTEGRATION.md` 完全包含

`tpa_video_analyses/FTT_RTP_TPA_INTEGRATION.md`（4/28，212 行）逐节扩充了 `road_to_pro_video_analyses/FTT_VS_RTP.md`（4/27，126 行），加入了 TPA 第三方对照。每个章节标题完全对齐。

| 保留 | 删除 | 理由 |
|---|---|---|
| `tpa_video_analyses/FTT_RTP_TPA_INTEGRATION.md` | `road_to_pro_video_analyses/FTT_VS_RTP.md` | 后者是前者的 strict subset |

**节省**：~8 KB。

---

## 类别 2: 同主题旧版（建议删除/合并约 7 个文件，~110 KB）

### 2.1 `13_synthesis.md`（3/14，39KB）vs `12_ftt_videos_synthesis.md`（3/17，36KB）

`12` 在开头明确写道："对比基线：13_synthesis.md。一、视频中的新概念（13_synthesis.md 中未覆盖的）"——`12` 是 `13` 的**增量补丁**，不是替代。但 `15_tpa_synthesis.md` 又把 `13` + `12` 都当基线，再加 TPA 增量。**这意味着用户要回答任何一个问题都得读 13 + 12 + 15 + 17 四个文件**。

**建议**：
- 不删除 `13`（它是基础综合）。
- **将 `12` 和 `17_kinetic_chain_synthesis.md`（11KB）的"新概念"段落上移到 `13` 的相应章节内**，合并后只保留 `13_synthesis.md` 一个 master synthesis。
- `15_tpa_synthesis.md`（16KB，3/23）已被 `tpa_video_analyses/SUMMARY.md`（22KB，4/28）和 `FTT_RTP_TPA_INTEGRATION.md` 覆盖——**建议删除 `15`**。

### 2.2 FTT 博客 04 系列 vs `ftt_backswing_complete.md` / `ftt_forward_swing_complete.md`

- `04_ftt_blog_forehand_1.md`（31KB，3/14）+ `04_ftt_blog_forehand_2.md`（33KB，3/14）= 28 篇博客的原始翻译笔记。
- `ftt_backswing_complete.md`（32KB，4/9）+ `ftt_forward_swing_complete.md`（32KB，4/9）= 同一批博客 + 书 + 视频的**主题再组织**（按"引拍/前挥"切片，每条信息标注来源）。

`04_*` 是按"博文 1, 博文 2, ..."的原序，`ftt_*_complete` 是按"什么是引拍/手臂/胸部/..."的主题序。**两者不严格冗余**——`04_*` 是按博客原文索引（适合查"某篇博客讲了什么"），`ftt_*_complete` 是按主题查询（适合回答用户问题）。

**建议**：
- 保留 `ftt_*_complete.md`（用户问题的主入口）。
- `04_*` 降级为只读原始资料；可考虑挪到 `_archive/` 或重命名为 `_raw_04_*`。

### 2.3 RTP/Tom 旧综合 vs 新综合

| 旧 | 新 | 处理 |
|---|---|---|
| `15_tpa_synthesis.md`（3/23，16KB）| `tpa_video_analyses/SUMMARY.md`（4/28，22KB）| 删除旧版（见 2.1） |
| `14_tpa_videos_1/2/3.md`（3/22-3/26，91KB 合计）| `tpa_video_analyses/*.md` 24 文件 + SUMMARY | 与类别 1.2 同模式：旧批是 monolithic, 新批是 per-video，建议删除旧批合集 |
| `16_tpa_kinetic_chain_1/2.md`（3/24，41KB）+ `17_kinetic_chain_synthesis.md`（11KB） | TPA SUMMARY 主题 A "减速产生加速" 已抽取核心 | `17_synthesis` 应在合并到 `13_synthesis` 后删除；`16_*` 视为原始视频笔记，可保留或归档 |

**潜在节省**：删除 `14_*`（91KB）+ `15_*`（16KB）+ `17_*`（11KB） = **~118 KB**。

### 2.4 三个 arm/shoulder 主题文件相互覆盖

| 文件 | 大小 | 日期 | 主题 |
|---|---|---|---|
| `arm_dominance_history.md` | 12KB | 4/10 | 用户训练历史时间线 |
| `arm_trunk_connection_tips.md` | 15KB | 4/9 | 教练口令 + 球员引用 |
| `arm_trunk_coupling_biomechanics.md` | 22KB | 4/9 | 解剖学/筋膜吊索 |
| `arm_body_integration_solutions.md` | 19KB | 4/10 | 教练 Tier 1/Tier 2 口令 + drill |
| `shoulder_dominance_fix.md` | 16KB | 4/10 | 三角肌为何抢戏 + 替代肌 |
| `ftt_passive_arm_unit_turn.md` | 15KB | 4/10 | FTT 视角：手臂被动 |
| `up_and_out_mechanism.md` | 21KB | 4/11 | up/out 是主动还是被动 |
| `19_forearm_compensation_analysis.md` | 58KB | 3/27 | 用户小臂代偿完整分析 |

**冗余分析**：
- `arm_trunk_connection_tips.md` 和 `arm_body_integration_solutions.md` 都列教练口令 Tier 表，主题高度重叠。一篇侧重"球员/教练采访引用"，一篇侧重"针对 Unit Turn 时手臂独立"。建议合并为一份"arm_trunk_cues_and_drills.md"。
- `shoulder_dominance_fix.md` 和 `ftt_passive_arm_unit_turn.md` 都讲"为什么三角肌抬臂错"，前者偏解剖学，后者偏 FTT 引文。建议合并。
- `19_forearm_compensation_analysis.md`（58KB，3/27）是用户问题专题，但用户的认知已迭代到 4/01 "里程碑突破"——这个文件的"当前状态评估"早已过时。**建议归档为 history（保留供回溯，但标记为已过时）**。

**潜在节省**：合并 4 个文件 → 2 个，节省 **~30 KB**；归档 19_* 不算节省。

---

## 类别 3: 工作过程文件（建议归档到 `_archive/`，约 4 个文件，~64 KB）

这些是分析任务的中间产物，技术内容已被 SUMMARY/INTEGRATION 文件吸收。

| 文件 | 大小 | 用途 | 处理建议 |
|---|---|---|---|
| `road_to_pro_video_analyses/_TASK_PLAN.md` | 2.4KB | 4/27 任务计划 | 归档 |
| `road_to_pro_video_analyses/_VIDEOS_TO_ANALYZE.json` | 22KB | 视频清单 + URL | 归档（可作为 video index 备份） |
| `road_to_pro_video_analyses/_SYNTHESIZER_BRIEF.md` | 4.6KB | 综合者使用说明 | 归档 |
| `tpa_video_analyses/_VIDEOS_TO_ANALYZE.json` | 32KB | 同上 | 归档 |
| `tpa_video_analyses/_SYNTHESIZER_BRIEF.md` | 6.3KB | 同上 | 归档 |
| `coach_analysis/integration_changelog.md` | 6KB | v4.1 集成变更日志 | 归档（系统已到 v4.2） |
| `coach_analysis/v4.2_INTEGRATION_SUMMARY.md` | 7.5KB | v4.2 整合报告 | 归档（同上） |
| `coach_analysis/vlm_prompt_changelog.md` | 7.4KB | VLM prompt 变更 | 归档 |

**归档总量**：~88 KB（不算"删除"，挪到 `_archive/` 子目录）。

---

## 类别 4: 段落级冗余（建议合并，挑出最严重的 5 处）

### 4.1 "动力链时序（毫秒级）"在 5+ 处独立讲

同一份"地面 → 髋（-75ms）→ 躯干（-57ms）→ 腕（-55ms）→ 拍头（-40ms）→ 触球"的表格/序列，出现在：

- `13_synthesis.md` Part 1.2（动力链完整序列）
- `forward_swing_body_mechanics.md` §1.1-1.2
- `arm_trunk_coupling_biomechanics.md`（有局部）
- `28_biomechanics_problem_solutions.md`
- `16_tpa_kinetic_chain_1.md` / `17_kinetic_chain_synthesis.md`
- `unit_turn_footwork_deep_dive.md` §1
- `ftt_forward_swing_complete.md`

**建议**：把"标准动力链时序毫秒级"做成一个**单一权威表**（建议放在 `13_synthesis.md`），其他文件只引用、不重复。

### 4.2 Unit Turn 概念在 6+ 处独立讲

- `30_unit_turn_hip_rotation.md`（2KB，专题）
- `unit_turn_footwork_deep_dive.md`（23KB，深度专题）
- `coach_analysis/tom_allsopp_unit_turn.md`
- `ftt_passive_arm_unit_turn.md`
- `ftt_backswing_complete.md` 二、§2.1
- `13_synthesis.md` Part 1.3
- `footwork/SYNTHESIS.md`（Phase T3）

**建议**：`30_unit_turn_hip_rotation.md`（2KB 小文件，仅讲"髋部跟着转"）的内容可并入 `unit_turn_footwork_deep_dive.md`，删除前者。

### 4.3 教练口令 Tier 表（"手不走身体转""左肩往前推"等）

- `arm_body_integration_solutions.md` §1（Tier 1 + Tier 2）
- `arm_trunk_connection_tips.md` §2（Tier 1 + Tier 2/3）

两份 Tier 表内容 ~80% 重叠，仅来源标注略不同。**建议合并为一份 master cue table**。

### 4.4 "为什么三角肌会抢戏"

- `shoulder_dominance_fix.md` 1.1（解剖学解释）
- `ftt_passive_arm_unit_turn.md`（FTT 引文）
- `arm_dominance_history.md`（用户时间线）
- `arm_body_integration_solutions.md`（修正方案）

四个文件都从不同视角讨论同一现象，每个开头都重新定义"三角肌默认调用"。**建议合并为一份 "arm_dominance_problem_complete.md"**，包含历史 + 解剖 + FTT 引文 + 修正方案 4 段。

### 4.5 击球 4 坐标 / 10 层 taxonomy

- `FOREHAND_COMPLETE_TAXONOMY.md` Layer 1（击球点 4 坐标）
- 在 `FTT_RTP_TPA_INTEGRATION.md` §1 又复述了一遍
- `unit_turn_footwork_deep_dive.md` §1 时序内也贴了一遍击球点参数

不算严重重复，但可以让 `FOREHAND_COMPLETE_TAXONOMY.md` 成为唯一定义源。

---

## 类别 5: 可疑过时（建议人工审视）

这些文件没有明确冗余，但日期/状态可疑：

| 文件 | 日期 | 状态 | 建议 |
|---|---|---|---|
| `19_forearm_compensation_analysis.md` | 3/27 | "状态：核心问题未解决" | 用户已 4/01 突破。归档或在头部加 **过时标记** |
| `arm_dominance_history.md` | 4/10 | 时间线只到 4/04 | 是否合并到 learning.md，或作为 history snapshot |
| `coach_analysis/feel_tennis_preparation.md` | — | 部分视频 403 members-only | 加 disclaimer 即可，不删 |
| `huberman_motor_learning.md` | 4/9 | 18KB 神经科学单文件 | 单一来源不冲突，保留 |
| `02_revolutionary_tennis.md` | 3/14 | RT 是 FTT 的对照系统 | 保留作为 dissent 参考 |

---

## 总结

| 类别 | 文件数 | 估算节省 |
|---|---:|---:|
| **完全重叠** 建议删除 | ~16 | **~264 KB** |
| **同主题旧版** 建议删除/合并 | ~7 | **~150 KB** |
| **工作过程文件** 建议归档 | ~8 | ~88 KB（不释放 disk，但脱离视野） |
| **段落级冗余** 建议合并 | 跨 5 主题 | ~30-50 KB |
| **过时标记** 建议加 header | ~3 | 0 KB |

**保守估算节省**：删除 + 合并 = **~410-450 KB**（约占 3.1MB 总量的 13-14%），文件数从 233 降到约 200。

**最高优先级三件事**：
1. **删除 `09_ftt_videos_1/2/3.md`** —— 最大单笔节省（172 KB），且已被 per-video + synthesis 覆盖。
2. **删除 12 个 TPA 重复视频文件**（在 `tomallsopp_video_analyses/` 中已被 4/28 新版覆盖）—— 84 KB。
3. **合并 `13_synthesis.md` + `12_ftt_videos_synthesis.md` + `17_kinetic_chain_synthesis.md`** 成一份 master synthesis ——避免每次回答都得读 3-4 份综合文件。

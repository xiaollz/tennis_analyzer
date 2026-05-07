# HSA (Horizontal Shoulder Adduction) Master Index

> **目的**：把项目里所有跟"水平肩内收 / 胸大肌驱动 / 胸肱角闭合"相关的文档、视频、代码、记忆、训练协议**统一到一个入口**。先看这个文件，再去深读子文档。
>
> **为什么需要这个 index**：本项目积累的 100+ 文档使用了 **12+ 种不同名称**描述同一个物理动作（press slot / chest fire / 胸推肘 / 撕 / 横拉 / windshield wiper / lasso / pec drive 等），导致每次找资料都要绕路。HSA 是这个动作的**生物力学正名**——所有概念到此收敛。

---

## 0. 一句话定义

**HSA = 大臂（肱骨）与躯干（胸腔）之间夹角的主动闭合**，由胸大肌驱动，是正手单一关节最大力量贡献（45-48% 前向 RHS / 与 ISR 合占 65% 接触速度）。

---

## 0.5 学术权威加持（5/4 加入）

**Tennis Science** (Elliott/Reid/Crespo 2015, University of Chicago Press) 8 章 deep read 完成，整合到本项目 KB。HSA 框架在本书中得到 ITF/UWA/Tennis Australia 三方权威的 peer-reviewed 数据确认：
- Ch2 p.34: ISR ≈ 40% 接触瞬间 RHS（serve）
- Ch2 p.34: pre-stretch SSC = +10-20% speed
- Ch2 p.34: 引拍顶点停顿 1s = -50% SSC 弹性能量
- Ch2 p.44: 蹬地 GRF 业余 1.7×BW vs 高水平 2.1×BW
- **Ch2 p.50: "IR may only occur after impact"** — ISR 是副产物，不是主动动作（5/15 模型核心支撑）
- **Ch2 p.138 kinetic chain 节点 5+6+8: scapular retraction + scapulohumeral rhythm + long axis rotation** = Phase 1 + Phase 2 整套体系的学术骨架
- **Ch7 p.147: Seated row + Shoulder ER at 90°** = Phase 1 上背 isometric 的负重训练协议
- Ch7 p.152 Kibler: "shoulder is funnel for energy flow from legs/trunk to racket arm"
- Ch7 p.142 Kibler: 网球肘 = 太多腕旋前 + 击球点过后 + 弱肩肌（HSA 失败的精确临床描述）

主索引：[`tennis_science_book/MASTER_INTEGRATION.md`](./tennis_science_book/MASTER_INTEGRATION.md)

---

## 1. 起源与时间线

| 日期 | 事件 | 文件 |
|---|---|---|
| 2026-04-30 上午 | 用户首次发现"胸推肘"体感 | `docs/record/learning.md §4/30 上午` |
| 2026-04-30 晚 | 发现肩胛槽是上身轴心（HSA 的发射台） | `memory/project_scapular_slot_bible.md` |
| 2026-05-02 | "撕"字命名 ISR 释放（HSA 的下游表现） | `learning.md §5/2 突破` |
| 2026-05-03 早 | 辛纳发力模型定型（仍没识别 HSA 是本体） | `learning.md §5/3 entry` |
| 2026-05-03 晚 | **HSA 命名 + 胸大肌全程参与连续模型** | `learning.md §5/3 晚 entry` |
| 2026-05-03 工程化 | 4 个研究文档 + `hsa_detector.py` + F7 集成 | 本文档 |

---

## 2. 子研究文档（深度阅读顺序）

### Tier 1 — 必读（按先后顺序）

| # | 文件 | 字数 | 核心 |
|---|---|---|---|
| 1 | [hsa_biomechanics_deep_dive.md](./hsa_biomechanics_deep_dive.md) | ~4700 | 解剖、肌肉激活时序、Sasaki 2022 IMU 数据、Kovacs 综述、HSA→ISR→pronation 三联耦合的力学解释、伤病学 |
| 2 | [hsa_coaches_alternative_naming.md](./hsa_coaches_alternative_naming.md) | ~3500 | 12+ 教练系统命名映射，谁讲了机制（Gordon + FTT），谁只讲了表象 |
| 3 | [hsa_youtube_survey.md](./hsa_youtube_survey.md) | ~3000 | 跨频道视频 Tier S/A/B/C 分级，前 10 优先观看清单（90-120min 总时长） |
| 4 | [hsa_local_kb_audit.md](./hsa_local_kb_audit.md) | ~2700 | 本地知识库现状 + 12+ 种历史命名 + 升级路径 |

### Tier 2 — FTT 原始材料

| 类型 | 来源 | 链接 |
|---|---|---|
| 文章 | Hugh Clarke "Accelerate Late Like Sinner" | https://faulttoleranttennis.com/accelerate-late-like-jannik-sinner/ |
| 文章 | Hugh Clarke "The Forehand Press Slot" | https://faulttoleranttennis.com/the-forehand-press-slot/ |
| 视频 | FTT "Shoulder Adduction Unlocks the Tennis Forehand" | https://www.youtube.com/watch?v=Am8j1Zw5KrE |
| 视频 | FTT "Shoulder Adduction Will Transform Your Forehand Contact" | https://www.youtube.com/watch?v=5KdScDKxVSI |
| 转录 | Gemini 原生分析 | `~/.gemini/transcripts/2026-05-03_Am8j1Zw5KrE.txt` 与 `_5KdScDKxVSI.txt` |

### Tier 3 — Brian Gordon / TennisPlayer

| 文件 | 内容 |
|---|---|
| [Four Pillars of ATP Type III](https://www.tennisplayer.net/public/biomechanics/brian_gordon/four_pillars_atp_type_iii/) | Pillar 4 = ESR→ISR transition（HSA 的另一面） |
| [Realities of the Straight Arm Forehand](https://www.tennisplayer.net/public/biomechanics/brian_gordon/straight_arm_forehand/) | Type 3 / 直臂模型，HSA 在直臂下的几何特殊性 |
| `docs/research/brian_gordon_video_analyses/` | 全套本地分析 |

---

## 3. 命名统一表

| HSA 视角的精确含义 | 历史命名 | 出处 |
|---|---|---|
| 胸肱角主动闭合 | "press slot" | FTT |
| 胸大肌向心收缩驱动 | "chest fire" / "chest engagement" | FTT |
| 胸肌发力把肘往前推 | "胸推肘" | 用户 4/30 上午突破 |
| 大臂跨过胸前 | "pull across body" / "横拉" | FTT 视频 1 |
| 拍头收到非持拍侧 | "windshield wiper finish" | 多个教练 |
| 拍头甩到左肩外 | "lasso" / "buggy whip" | Nadal/Federer 描述 |
| 击球瞬间 ISR 释放 | "撕" | 用户 5/2 突破 |
| 胸肌作为肘前推的 fixed point | "scapular slot" | 用户 4/30 晚突破 |
| 关闭夹角的几何动作 | "closing the angle" | FTT 视频 2 |
| HSA 引发的肩内旋 | "ISR" / "肩内旋" | 解剖学 |
| ISR 引发的小臂内旋 | "pronation" / "旋前" | 解剖学（**下游副产品**） |

**统一规则**：从今日起，文档中遇到上述任一术语，应在首次出现时显式标注 `(= HSA)` 或 `(HSA 的下游)`。不要孤立使用。

---

## 4. 失败模式分类（与 `hsa_detector.py` 对齐）

| 模式 ID | 描述 | 接触瞬间 HSA 角 | 总闭合幅度 | 跨胸 | 体感线索 |
|---|---|---|---|---|---|
| `healthy` | 健康 | 45-80° | ≥25° | 完成 | 胸肌"充血"，球被甩出去 |
| `no_closure` | 完全无闭合 | >85° | <15° | 否 | 大臂保持外展，纯靠转体扫 |
| `static` | 静态扫球 | 70-85° | <10° | — | 整段角度几乎不变 |
| `early_closure` | 闭合过早 | <45° | 大 | 否 | 肘已贴身，纯推球 |
| `late_closure` | 闭合过晚 | >70° | 中 | 接触后才完成 | 球离拍后才感觉胸 fire |
| `insufficient_cross_body` | 闭合 OK 但跨胸不足 | 45-80° | ≥25° | 否 | 拍头停在右侧或正前方 |

---

## 5. 代码集成

### 检测模块
- **`evaluation/hsa_detector.py`**：从 2D pose 关键点计算 HSA 指标
  - `hsa_angle_2d()`：单帧 HSA 角度
  - `compute_hsa_trajectory()`：时间序列
  - `compute_hsa_velocity()`：闭合速度
  - `cross_body_finish_distance_2d()`：跨胸完成度
  - `classify_closure_pattern()`：5 种失败模式分类
  - `compute_health_score()`：0-100 健康分
  - `detect_hsa()`：主入口
- **测试**：`tests/test_hsa_detector.py` (17 用例覆盖 5 失败模式 + 健康 + 边界)

### Foundation Layer 集成
- **`evaluation/foundation_layer.py` F7_hsa**：第 7 个地基项，priority=1
- 输入指标：`hsa_total_closure_deg`, `hsa_angle_at_contact`, `hsa_closure_pattern`, `cross_body_finish`, `hsa_health_score`
- VLM 信号：`Q39` (闭合幅度) + `Q40` (跨胸完成)

### VLM Prompt 集成
- **`knowledge/templates/vlm/system_prompt.md.j2`**：新增 Q39/Q40
- **`evaluation/vlm_analyzer.py`**：F7 PASS/FAIL 判定标签 + 解析器
- 输出 schema：`extra_observations.hsa_closure_visible` + `cross_body_finish_visible`

---

## 6. 训练协议（按熟练度阶段）

### 阶段 0：体感建立（1-3 天，对镜不持拍）
- **左手按右胸大肌** + **右手做横拉空挥** → 直到能触摸到胸肌"充血"收缩
- 来源：FTT 视频 `Am8j1Zw5KrE` [00:00-00:30]
- 检验：感觉到胸肌**主动 contract**，而不是被动跟随

### 阶段 1：HSA 隔离（1-2 周，发球机或喂球）
- **静态无转体击球**：双脚不动，纯用 HSA 击球
- 目标：没转体也能打出 decent ball
- 来源：FTT 视频 `5KdScDKxVSI` [03:40]
- 检验：球能穿透，胸肌酸（不是手臂酸）

### 阶段 2：反向工程握拍（1 天调整）
- 不拿拍找最自然的 HSA 路径
- 拿拍保持 HSA 终点姿态，调整握拍直到弦床 slightly closed
- 来源：FTT 视频 `5KdScDKxVSI` [01:05]
- 检验：握拍**适应** HSA 路径，不是相反

### 阶段 3：HSA + 转体整合（2-4 周）
- 加入 Unit Turn → HSA 释放
- 转体作为**放大器**，不是发力源
- 检验：胸肌"全程张力曲线"——从 Unit Turn 10% 到 contact 100%

### 阶段 4：实战调用（4-8 周）
- 不同球速、不同来球方向都能调 HSA
- 实战检验：球质明显改善（速度+穿透）+ 胸肌每天酸（不是手臂）

---

## 7. 视觉/几何 markers（VLM 检测要点）

### 帧序列对比
| 帧 | 看什么 | 健康 | 失败 |
|---|---|---|---|
| 引拍顶点 | 大臂相对躯干横线角度 | 90-100° | 同 |
| 接触前 1-2 帧 | 角度 + 闭合速度 | 60-80° + 闭合中 | >85° + 静态 |
| 接触瞬间 | 角度 | 45-80° | >90° 或 <35° |
| 接触后 3 帧 | 拍头位置 | 朝左前方 | 朝右后方 |
| 随挥末端 | 持拍腕 vs 非持拍肩 | 越过非持拍肩 | 停在右侧 |

### 误识别警告
- 不要把**纯转体扫球**误判为 HSA：转体让 RHS 增加但 HSA 角度不变 → static 模式
- 不要把**孤立 pronation** 误判为 HSA 完成：手腕翻 + 大臂没动 → 上游缺失
- 不要把 **windshield wiper finish** 自动当 HSA 健康：可能是接触后才闭合（late_closure）

---

## 8. 与现有 Foundation 的层级关系

```
F1-F4: FTT 4 项（priority 0，地基）
       ↓ 失败时阻断上层分析
       
F5: 右脚轴（priority 1，下肢轴心 / HSA 能量入口）
F6: 肩胛槽（priority 1，上身轴心 / HSA 发射台）
       ↓ 都建立后
       
F7: HSA（priority 1，驱动引擎，使用 F5+F6 作为支撑）
       ↓ 释放
       
ISR + pronation （HSA 的物理副产品，不需独立训练）
```

**关键原则**：F7 不是地基，是引擎。不能跳过 F5/F6 直接练 F7——没有右脚轴 → 无能量入口；没有肩胛槽 → HSA 反推后倒。

---

## 9. 已知限制与开放问题

### 2D pose 测量局限
- HSA 真正发生在 3D 倾斜轴上，2D 投影会失真
- 摄像机角度（侧面 vs 正面）影响角度估算
- COCO 17 没有胸骨/锁骨关键点 → 用肩线代替（近似）

### 未解决
- HSA 角速度峰值（°/s）的健康阈值——需要更多职业球员视频校准
- HSA 闭合发生在哪个 phase 的最优 timing——目前用"主体在接触前"作为粗略判据
- 非右手球员的镜像处理在 detector 里已做但未实测验证
- 与 ESR 加深的耦合度量——HSA 启动是否依赖 ESR 深度，无量化模型

### 后续工作
- 用 OpenPose 25 关键点（带胸骨）重新校准
- 收集 10+ 职业球员视频建立 HSA 角度时间曲线 reference band
- HSA + 击球深度 + 球速 三者相关性的实测分析
- 把 HSA 检测嵌入 `analysis/kinematic_calculator.py` 做实时帧分析

---

## 10. 用户原创洞察（FTT 视频里没明讲的）

1. **HSA 在 Unit Turn 阶段就已工作（10% 张力）作为"胸肌抑制器"**——防止大臂过度后撤
   - 物理：胸大肌轻度等长/离心收缩，把大臂"拴"在槽里
   - 价值：把"管住大臂"从约束类（"别让大臂动"）变成驱动类（"用胸肌按住"）
2. **胸肌全程张力曲线** vs FTT "chest fire" 的点状描述
   - Unit Turn 10% → forward swing 30% → contact 100% → follow-through 余势
   - 胸肌**从未断电**

这两条作为**用户原创**记入 `learning.md §5/3 晚 entry`，外部材料未明确表述。

---

## 11. 快速查找

- **代码**: `evaluation/hsa_detector.py`, `evaluation/foundation_layer.py` F7
- **测试**: `tests/test_hsa_detector.py`
- **VLM**: `knowledge/templates/vlm/system_prompt.md.j2` Q39/Q40, `evaluation/vlm_analyzer.py` F7 prompt block
- **memory**: `~/.claude/projects/-Users-qsy-Desktop-tennis/memory/project_hsa_engine.md`
- **学习日志**: `docs/record/learning.md §2026-05-03 晚`

---

## 12. JUL Tennis & Golf 频道整合（5/7 加入）

JUL 频道（21 支视频深度扫描）= **HSA 体系的外部权威背书 + 物理硬件层补完**。

### 12.1 物理硬件层补完（4 个核心新概念）

#### a) Hypothenar Eminence（小指鱼际丘）—— 双外旋锁定的硬件实现
- 位置：手掌靠小指那一侧的肉垫
- 机制：**球拍重量挂在这里 → 神经允许 deep ER 不触发防御性 IR**
- 来源：JUL `zPSEjRrDPQw`
- 跟用户体系：5/15 "双外旋锁定" 的硬件层

#### b) Index Finger 是 IR 抢跑的硬件开关
- 来源：JUL `zf6dBOucjqg` [00:46]
- 原话：*"If the weight is on the index finger, you will get early internal rotation."*
- 修正：握拍重心从食指根 → 移到小指鱼际丘
- 跟用户体系：5/16 IR 抢跑根因的**末端硬件解释**

#### c) Three-Layer Classification（三层分类）
- 轴心层：Spine Axis / Leg Axis / 伪 Unit Turn
- 驱动层：Pitching / Batting
- 结构层：Whole Body Swing / Hip Turn Swing
- 来源：JUL Djokovic + Nadal 系列
- 用途：业余 → 精英升级路线图

#### d) Ruler Test（直尺测试）—— 真假 Unit Turn 硬指标
- 来源：JUL Nadal 2 [3:33]
- 原话：*"arm line behind or on the shoulder-to-shoulder line until a little bit before the ball impact"*
- 用途：**可量化的 Unit Turn 检测标准**

### 12.2 概念映射表（JUL ↔ 用户体系）

| 用户体系 | JUL 等价 |
|---|---|
| 5/3 HSA（胸大肌闭合） | shoulder horizontal abduction |
| 5/6 推肘禁令 | F ∝ distance（HSA 是空间不是动作） |
| 5/15 双外旋锁定 | **Chinook Pose** + Hypothenar Eminence |
| 5/15 上身 Wrap | One Mass + Bio-rope |
| 5/16 IR 抢跑诊断 | Suppress IR instinct（5 种等价描述）|
| 4/13 左手拉离 | Seesaw mechanism |
| 4/27 右脚轴 | Spine axis |
| 4/30 上身槽 | Pull back upper arms（Chinook Pose Step 2）|

### 12.3 文档位置

`docs/research/jul_tennis_videos/`
- `MASTER_SYNTHESIS.md`（必先读）
- `federer_series_synthesis.md`
- `rubber_arm_series_synthesis.md`
- `djokovic_nadal_concepts_synthesis.md`
- `deep_mechanism_synthesis.md`
- 21 支独立视频分析

### 12.4 使用规则（重要——遵守 5/6 Intuition-First 协议）

> ✅ JUL 是 reasoning reference / 5 秒视觉重启工具
> ❌ JUL 不是新 cue 来源——不要加进训练 list
> ❌ 不要因为 JUL 表述去改造现有 cue（同构 ≠ 冲突）

最有用的应用：**球场卡住时看 5 秒慢动作**（Sinner / Djokovic）—— 视觉锚定，不进 reasoning。

---

## 13. ESR 根因主参考（5/8 加入）

5/8 用户终极诊断：**ESR (External Shoulder Rotation) 不在 Unit Turn 第一帧主动启动 = 项目级根因 #2，是 IR 抢跑根因 #1 的对偶面**。

ESR 是 HSA 框架的**物理上游**：
- HSA = Phase 2 主动驱动（胸大肌闭合）
- **ESR = Phase 1 主动 trigger**（抑制 IR 群 + 储 SSC）

5/8 用户原创洞察：**ESR 作用 1（抑制 IR 抢跑）> 作用 2（蓄力）**——把 ESR 从"蓄力机制"重新定义为"刹车机制"。这是项目级理论贡献。

完整文档：[`docs/research/esr_root_cause_master.md`](./esr_root_cause_master.md)（5000-7000 字主参考 + 11 章节 + 5 附录）

回答任何 ESR / IR 抢跑 / 大臂飘 / 球软 / 后倒 / 推肘代偿 / 节奏断 / 内上髁炎相关问题前，**第一句必须问**：

> **"你刚才那一拍，第一帧 ESR 启动了吗？拍头朝天还是朝右？"**

跟本 HSA 索引的精确边界：
- HSA 是末端释放（Phase 2 末 10%）
- ESR 是前置 trigger（Phase 1 第一帧）
- 没有 ESR → HSA 等于"主动推肘"（5/6 禁令场景）
- ESR 到位 → HSA 自动发生（被动闭合）

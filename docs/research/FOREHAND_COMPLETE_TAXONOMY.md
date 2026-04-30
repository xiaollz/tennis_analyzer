# 正手技术完整分类框架（10 层模型 + Bourne L11 训练辅助层）

> 2026-04-27 创建（10 层力学）
> 2026-04-29 追加：Stephen Bourne《One Minute Tennis Forehand Solution》集成 → 新增 Layer 11（训练辅助 cue 层）
> 起因：用户指出"击球点"维度被完全漏掉，这暴露了诊断系统**没有完整的维度清单**。
> 本文是补救——列出**所有**正手相关的独立维度，作为后续诊断的强制 checklist。

---

## 总原则

任何一次正手分析，**必须**遍历下面 10 层力学维度。任何一层缺失就是**不完整诊断**。
Layer 11（Bourne 训练辅助 cue）是横切层，给 L1-L10 提供压力下的回滚工具集，与 11 字实战系统形成两层结构。

> **从下往上逐层依赖**：地基（Layer 1-2）不对，上面（Layer 5-6）做得再好也没用。
> 之前我反复在 Layer 4-6 修，根因在 Layer 1，等于刷墙不修地基。

---

## Layer 1 · 几何 / 空间（Spatial Geometry）⭐⭐⭐⭐⭐ 最底层

> **被漏掉的层**——本次反思的核心。

### 维度

| 维度 | 含义 | 测量方法 | 理想值 |
|---|---|---|---|
| **击球点 · 高度** | 击球瞬间球的垂直高度 | 视频侧拍 + 身体参照 | 腰-胸（90-115cm）|
| **击球点 · 横距** | 球距持拍侧身体的水平距离 | 视频后拍 + 拍长 70cm 参照 | 40-60cm |
| **击球点 · 纵深** | 球相对前胯的前后位置 | 视频侧拍 | 前胯前方 30-50cm |
| **击球点 · 角度** | 球相对身体中线的俯视夹角 | 顶视图或推算 | 45° |
| **站位距离** | 准备位时人离来球落点的距离 | 视频前拍 | 拍长 + 一臂（约 1.4m）|
| **击球区** | 上述四个坐标定义的"舒适区"体积 | 多次击球的散点 | 30cm × 30cm × 30cm 立方 |
| **接近角** | 你向球移动的轨迹相对来球轨迹的角度 | 视频俯视 | 不要平行接近，应斜切 |

### 失败模式

- 击球点偏低（球在大腿高）
- 击球点贴身（横距 < 30cm）
- 击球点偏后（在身体侧面或后方）
- 站位太近（被球挤）

### ⭐ "击球手始终在身体右侧"（Bourne 共同性 #1）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.18b)

引拍 → 前挥 → 接触 → 收尾全过程，**持拍手不允许越过身体中线进入左半边**（针对右手球员）。
这是 Bourne 五大共同性的第一条，与现有 Layer 6 的 Midline Rule 印证：拍头跨过中线之前不许向后倒；
拍**手**则是更严格的——整个挥拍周期都该在右侧。

**几何含义**：击球点的横距（持拍侧水平距离）有下限（不许贴身/越过中线），但更重要的是**手的轨迹是单侧的**。
手越过中线 = 抱球意象崩了 + WTA-style takeback midline violation 的诊断阳性证据。

### 知识库当前覆盖

- ✅ FTT 视频 wVa4XQPcaqs "How to Find Your Perfect Contact Point"（已分析）
- ✅ FTT 视频 Am8j1Zw5KrE "Forehand Contact Zone"（已分析）
- ✅ FTT 视频 ExkBtFRhUWY "Probing the Ball on One Foot"（已分析）
- ✅ Bourne p.18b "hand stays on right side"（印证 Midline Rule + 中线诊断链）
- ❌ **诊断引擎里没有击球点 KPI**
- ❌ **VLM prompt 里没有问"击球点位置"**
- ❌ **学习记录里直到 4/26 才出现**

---

## Layer 2 · 时间 / 时序（Temporal）⭐⭐⭐⭐⭐

### 维度

| 维度 | 含义 | 测量 | 理想值 |
|---|---|---|---|
| **Unit Turn 时机** | 多早开始转身 | 球落地瞬间 vs 肩转角度 | 球弹地时肩已转 90° |
| **Bounce-Hit 节奏** | 球弹地到击球的时间 | 帧数 / fps | ~0.5-0.7s（业余）|
| **加速时机** | 拍头加速最快是哪一刻 | 拍头速度曲线峰值帧 | 触球前 50ms（Sinner 式 late accel）|
| **引拍峰值** | 拍头到引拍最高点的时机 | 视频帧标 | 球弹地之前 |
| **击球释放** | 主动加速段持续多长 | 帧 | 短脉冲，~3-4 帧 60fps |

### 失败模式

- Unit Turn 晚（球到了才转）
- 加速太早（在引拍顶点就发力，触球已减速）
- Backswing 还在向后时球已到

### 知识库当前覆盖

- ✅ FTT JIMgI3jiVns "Forehand Timing Secret"
- ✅ FTT GsHkML2mVEI "Backswing Timing"
- ✅ FTT 博客 "Accelerate Late, Like Sinner"
- ⚠️ 学习记录里偶尔提到（4/26 凌晨 Bounce-Hit drill）
- ❌ **诊断引擎不计算 Unit Turn 启动时机**
- ❌ **VLM prompt 不问"什么时候开始转身"**

---

## Layer 3 · 移动 / 步法（Footwork）⭐⭐⭐⭐

### 维度

| 维度 | 含义 |
|---|---|
| **分腿垫步**（Split Step）| 对手击球时的弹跳准备 |
| **第一步方向** | 哪个脚先动、向哪个方向（必须同侧脚先动 → 锁髋 45°，via Im2JyVN8Rn0）|
| **调整步**（Adjustment Steps）| 接近球时的小步调整 |
| **加载脚位置**（Loading Foot）| 上步还是后撤步 |
| **前脚落点** | 前脚相对于来球落点的位置 |
| **回位**（Recovery）| 击球后的位移恢复 |
| **Heel-to-Toe 落地**（前脚滚动）| 必须脚跟先着地 → 滚到前掌；前掌着地 = 踩刹车切断动量（via ZM7DYfi17no）|
| **后腿先弯**（Back Knee Bend First）| 处理短/低球时由后腿先蹲，不要弯腰前倾（via ZM7DYfi17no / LVtxz0fba18）|
| **Carioca Step + 后腿后踢** | 浅球随球上网时维持闭合肩线的步法组合（via XFdjh_EROwM）|
| **Foot Speed 等级**（1-10）| 必须 ≥ 挥速等级 + 1，否则扭矩链断（via dCRj8MSNOUg）|
| **三种步序模式**（RHL / RLH / R-R）| 球高 → R-Hit-L（左脚击球后才落地）；球低 → R-L-Hit（左脚先踩降重心）；极端被动 → R-to-R（同侧脚为轴跳起）。**球高度直接决定步序选哪一支**（via e-KdPNRv9Ls / ftyfZXr3Zcw）|

### 失败模式

- 没有 Split Step
- 第一步方向错（背对球先转）
- 调整步不够 → 球到了人没到
- 前脚落得太靠近球（或太远）

### 知识库当前覆盖

- ✅ FOmz8Wjv3DQ "Federer's Footwork"
- ✅ `docs/research/footwork/SYNTHESIS.md`
- ⚠️ 学习记录中频繁但碎片
- ❌ **诊断引擎没有 footwork 模块**

---

## Layer 4 · 站姿 / 站位（Stance + 轴心稳定性）⭐⭐⭐⭐⭐

### 维度

| 维度 | 选项 |
|---|---|
| **站姿类型** | 封闭 / 中性 / 半开放 / 完全开放 |
| **前后脚距离** | 太近 / 适中 / 太远 |
| **脚的指向** | 前脚 45°、后脚垂直底线（半开放标准）|
| **重心分布**（准备时）| 70% 后脚（半开放标准）|
| **重心分布**（击球时）| 切换到 70% 前脚（线性站姿要求）|
| **膝盖弯曲** | Double-bend 站姿深度 |
| **轴心稳定性**（Axis Stability）⭐ | 击球瞬间重心轴是否平移 |
| **前脚落地时机**（Front Foot Landing Time）⭐ | 击球前 / 击球瞬间 / 击球后 |

### ⭐ 轴心稳定性 —— 击球点几何崩溃的真正机制

> 来源：Sky Kim "Too close? Too Far? Spacing fix" (Road to Pro Tennis)
> URL: https://www.youtube.com/watch?v=aiwUqHQl-Ec

### ⭐ 站姿动态切换（Stance Branching）—— 半开放是母板

> 来源：RTP 综合（via QNksnW6cq-4 / 8LsLG8ZOa1g）

半开放站姿不是静态选择，而是**保留击球前 0.1 秒切换到开放/关闭的选择权**：
- 快球 → "Lifting and Pivoting"（左脚虚起、右脚轴转）= 开放式
- 慢/低球 → "Kick the right foot back"（左脚上步同时右脚后踢补偿）= 关闭式
- 默认必须 Semi-Open；过早死锁站姿是大忌。

### ⭐ 同侧第一步（Same-Side First Step）—— Unit Turn 的物理触发器

> 来源：Sky Kim "First Step After Split" (via Im2JyVN8Rn0)

分腿垫步落地后，**靠近来球方向那只脚必须先动**（正手时右脚先动）。这一步不是为跑而是为"锁髋 45°"，否则后续 Unit Turn 会变成全身一起转的"旋转"而非分离的"扭矩"。失败模式：先动远端脚做交叉步 → 髋直接转 90° → 引拍过大 → 击球点必偏后。

### ⭐ Pivot, Don't Jump —— 旋转是脚跟外旋出来的

> 来源：TPA Tom Allsopp "Forehand Rotation - Pivot, Don't Jump" (via ooEX4wIA8l4)

旋转的物理引擎不是"主动跳起来腾空"，而是**右脚脚尖钉地、脚跟外旋（Pivot）**。
跳是结果，不是手段——当 Pivot 产生的扭矩 + 推力大到地面摩擦力抓不住身体时，腾空才会作为副产品发生。
失败模式：业余球员为了"用上身体"主动蹬跳，离地瞬间地面反作用力消失 → 旋转动能反而下降。
检查指标：随挥结束时回看右鞋底——应该有圆弧形旋转痕迹，**不是**向上的蹬踏痕迹。

### ⭐ Sit-Stand 垂直循环 —— 站姿是动态发条不是静态形状

> 来源：OTI James Ludlow "Do NOT Stay Down" (via hGw1-pzsCK8) +
> RacquetFlex "Modern Forehand Leg Drive" (via WB1Dx59xfrI)

正手的下肢机制不是"保持低重心稳定"，而是 **Sit down (加载) → Stand up (释放)** 的垂直循环。
"Stay Down" 是行业迷思——它会切断垂直方向的 GRF 释放，导致只能靠手臂发力。
**量化标准（RFlex WB1Dx59xfrI）**：
- 站距 = 肩宽 + 1.5-3 ft（明显宽于"自然站立"）
- 击球时头比站立时低 6-12 in（明显的垂直位移）
- 击球瞬间双膝从弯曲转向伸直 = Stand up 释放正在发生

失败模式：双脚踩死整个击球过程，头部高度无变化 → 整条动力链垂直分量为零，全靠水平旋转和手臂代偿。
**检查指标**：录像对比击球前一帧（蹲位）和击球瞬间一帧（站位）——头部应有明显的向上位移（≥10cm 业余可接受，≥15cm 进阶水平）。

### ⭐ V-Formation 侧向加载 —— 压不是垂直下蹲是侧向折叠

> 来源：Intuitive Tennis Nikola "How to Load the Forehand" (via O2Bb1JA1ajA)

加载阶段的躯干不是直立的，而是**向击球侧（右侧）侧弯**——胸部和右大腿形成 V 字，
右侧腹外斜肌+髋屈肌群被强行拉长。这是 Sit-Stand 循环的**水平面分量**，
和垂直方向的 Sit down 同时发生。

**几何标准**：
- 胸骨方向：偏向右髋（不是垂直地面）
- 右肋骨与右大腿根部距离：< 一掌（可见的侧向折叠）
- 左脚自然飘起：是 V-Formation 的副产品（重心不平衡），不是主动跳

**与"飘"字诀的因果关系**：左脚离地不是"为了让右脚成轴"，是"V-Formation 释放后躯干被动伸直、由于重心偏右产生的平衡补偿位移"。这把"飘"从主动动作降格为物理副产品——更省力、更稳定。

### ⭐ Bourne 站姿命题 —— 半开放是默认，不是"开放"（印证 + 加固）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.18)

Bourne 在站姿章节开炮反对"像职业一样全开放"的教学口号——**职业绝大多数正手用的是半开放**：
- **双脚连线偏底线 30-45°**（不是双脚平行底线那种全开放）
- 全开放减少 coiling，适合 redirect power，不适合 generate power
- 双脚平行底线时髋/左腿/左脚旋转被锁，coil 容量受限

**反传统命题**："Open stance like the pros" 是教学误区，连巡回教练也在传——直接打脸。

**与现有 Stance Branching 的关系**：印证"半开放是母板"——Bourne 给的 30-45° 数字是这个母板的几何下限。
更激进的 Open / Neutral 都是从这个母板临场切换，不是预设站姿。

### ⭐ "重量装外侧脚 / 右脚承重"（Bourne 共同性 #2，与用户 4/27 圣经字面级对齐）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.18)
>
> 原文："The fundamental commonality is that the player loads the weight into the right foot and the right side of the body!"

这是 Bourne 五大共同性的核心——**站姿可以是 Semi-Open / Open / Neutral，但重量必须落到外侧脚（右脚）和身体右侧**。
**与用户 4/27 圣经"右脚为轴 = 一切"字面级对齐**——这是这本书对用户最高强度的印证。

**力学含义**：
- 这条不是"站姿选项"，是站姿之上的元规则
- 与 Layer 4 现有的 "Hit the ball off of one foot, not two"（Sky Kim 单脚击球）完全一致
- 与 V-Formation 侧向加载（向右侧弯）完全一致——重心落到右侧 = V 的几何必然
- 是"压"字诀的解剖学基础

### ⭐ Athletic Height —— 站姿宽度+头部高度的量化标准

> 来源：RacquetFlex "Modern Forehand Leg Drive" (via WB1Dx59xfrI)

之前 3 体系（FTT/RTP/TPA）只说"宽站、低位"，从不给数字。RacquetFlex 引用 Pat Dougherty 给出**两个具体参数**：
- **站距**：肩宽 + 1.5-3 ft（即两脚间距比"自然站立"宽 1.5 个肩宽以上）
- **击球时头部高度**：比站立时低 6 in 到 1 ft（约 15-30cm）

**力学根因（F1 赛车类比）**：宽轮距 + 低重心 = 不翻车 = 大脑允许腿部全力输出。站太窄或太高会触发"防摔倒"反射，自动掐断腿部发力。

**录像验证**：在球场地面找两条标志线（场内边线、双打边线）作为参照，量化两脚位置。

**核心命题**：
> 你"被球挤到"的力学根因不是手的问题，是**重心轴在击球瞬间发生了非法位移**。

**机制**：

```
理想情况：
  右脚为轴心 → 身体绕轴旋转 → 球进入预设的 45° 位置 → 击球
  → 轴心从未移动 → 击球点 4 坐标稳定

错误情况（你目前的状态）：
  右脚加载 → 想推重心到左脚 → 左脚提前踩死 → 重心轴前移
  → 原本预设的击球点 A 变成了"相对于身体偏后偏近"
  → 你感觉"球突然变快了"或"自己离球太近了"
  → 实际上是**你自己冲进了球的空间**
```

**Sky Kim 的标志性建议**：**"Hit the ball off of one foot, not two."**

- 击球瞬间**只用右脚（后脚）支撑**
- 左脚**虚点地面甚至悬空**
- 等击球**完成后**左脚才落地（"Land after hitting"）

**这和 FTT 的 "撑" 是完美互补**：
- FTT "撑" = 胸部主动收缩
- Sky "Axis" = 击球时身体不要前冲
- **两个一起做才完整**——胸推但身体没冲过去

### 失败模式（新增）

- 左脚踩太早 → 轴心前移 → 击球点缩小
- 屁股向后撅 → 轴心已塌 → 在伸手够球
- 双脚同时承重 → 轴心 = 平移 + 旋转 → 击球点不可重复

### 知识库当前覆盖

- ✅ 4/26 晚 + 凌晨学习记录（半开放切换决策）
- ✅ Sky Kim 视频（aiwUqHQl-Ec）已分析
- ❌ "站姿决策树"（什么球用什么站姿）—— 待补
- ❌ **诊断引擎不识别站姿类型**
- ❌ **诊断引擎不检测轴心位移**（Axis shift）

---

## Layer 5 · 身体姿态（Body Posture）⭐⭐⭐⭐

### 维度

| 维度 | 测量 |
|---|---|
| **脊柱倾角** | 垂直 / 前倾 / 后倾几度 |
| **头部稳定** | 触球瞬间头是否动 |
| **肩线倾斜** | 击球肩相对非击球肩的高度 |
| **髋线水平** | 双髋是否平 |
| **核心张力** | 腹肌 + 背肌共同收缩 |

### 失败模式

- 后仰（4/26 晚发现）
- 头跟着挥拍移动（"看不到"球）
- 击球肩塌（应高于非击球肩）

### ⭐ 头微倾右（Head Leads Right） —— 头是重心方向舵

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.20 RP5 + p.24)
>
> 原文："Remember, the body will follow the head."

引拍/抱球阶段，**头微倾向身体右侧**（持拍侧）。这是 Bourne 五大共同性的第五条，
也是其他体系（FTT/RTP/TPA/Brian Gordon/RacquetFlex/Intuitive/OTI）都没强调过的独家维度。

**力学含义**：
- 头偏右 → 重心右倾 → 击球瞬间重心从右到左 → 产生旋转能量与角动量
- 头不只是"看球"工具，**还是重心方向舵**
- 这给现有"盯"字（视觉锚定）补了一个新维度——头是位置而不只是注视方向

**与 Layer 1 "击球手在右侧"+ Layer 4 "重量到右脚"形成闭环**：手右、脚右、头右——三个右构成抱球意象的几何骨架。

### 知识库当前覆盖

- ✅ 4/26 晚学习记录（rear-lean）
- ✅ FTT 体系强调脊柱稳定
- ✅ Bourne p.20 / p.24（"head leads body" 头领身原则，新增维度）
- ⚠️ KPI 中有 SpineConsistencyKPI、HeadStabilityAtContactKPI（部分覆盖）
- ❌ **VLM prompt 不问"头是否微倾右侧"**

---

## Layer 6 · 上半身机制（Upper Body Mechanics）⭐⭐⭐⭐

> **这是我之前过度聚焦的层**。

### 维度

| 维度 | 含义 |
|---|---|
| **Unit Turn 整体** | 肩 + 髋 + 大臂作为单位转动 |
| **X-Factor** | 肩比髋多转 30°（分离）|
| **胸部参与**（Chest Press）| 胸大肌等长 → 向心收缩 |
| **背阔肌连接**（Lat sling）| 大臂粘躯干 |
| **前锯肌包裹**（Wrap）| 肩胛骨贴近 |
| **左臂后压**（Off-Arm Pull）| 非持拍手向后/向身体方向反作用 |
| **送肘**（Elbow Lead）| 肘部领先于手 |
| **Hip Locking**（右胯反向锁定）| 引拍时右胯不跟肩转，做投石机底座（via 9ihq4WFCWy0）|
| **Hide the Elbow**（藏肘视觉标准）| 引拍合格 = 对手视角看不到右肘（via va005XuoBEU）|
| **Midline Rule**（中线原则）| 拍头跨过身体中线之前不许向后倒（via _pB-WTQGSp4）|
| **L-shape Neck Hold**（L 型托拍颈）| 左手虎口托拍颈而非抓拍柄，强制 Unit Turn 充分（via CZhncV-DYUw）|
| **Lat-Trap Antagonism**（背阔肌主动 / 斜方肌放松）| 用背阔肌下拉强制沉肩，对抗紧张耸肩（via enu0Cl7boJ0）|
| **Shoulder Freeze / 躯干减速**（Slow down to accelerate）| 击球瞬间近端关节（肩/躯干）必须出现明显的减速制动 → 远端拍头才能完成"超车"加速。匀速旋转的躯干 = 拍头无加速度。（via 1fyiKRioGR0 / A8XXmrdIdbc）|
| **Right Shoulder in Front 检验**（右肩超前）| 击球瞬间右肩转到左肩前方（相对于底线） = 旋转充分释放的视觉验证。半西方握拍下，胸口不正对球网就根本拿不到正确击球点（via LU9yamZPOnw / ahlffa-Am9U / muxc0h0YAJg）|
| **Hand-Elbow-Shoulder Layering**（手肘肩阶梯）| 击球瞬间空间纵深递进：手在肘前 + 肘在肩前。任何一级倒挂 → 拍面失去支撑 → 手腕被迫代偿。这是"撑"字诀的工程版定义（via OBjVdy1MS44 / muxc0h0YAJg / ahlffa-Am9U）|
| **Elbow Forward Action 同步 Supination**（肘前移与旋后同步）| Racket Drop（旋后）发生时肘部必须**已经在向前走**——否则就是"假 Lag"，肘部留在身后会让击球点必然偏后（via ygbZ8aONhRI / hiujcyG1Bkk）|
| **Co-contraction 解锁**（拮抗肌互锁）| 为模仿 Double-Bend 形态而锁死肘角度 → 二头/三头肌同时收缩 = 同时踩油门和刹车 → 力量内耗。"Double-Bend 是动态切片，不是静态目标"（via UVrZoQ70wxU）|
| **Horizontal Adduction（水平内收）作为第二级加速** | 核心旋转提供基础速度（×1），击球瞬间大臂主动通过胸大肌+三角肌前束向胸前内收 → 把基础速度放大到（×10）。**这是 TPA "被动手臂"哲学的重要修正**：手臂在前臂层是被动的，但大臂层必须主动内收（via RacquetFlex BZDjG-GuhVs）|
| **NDA 平行底线**（非持拍手几何标志）| Unit Turn 完成时左手必须与底线**重合**（不是"指侧网"也不是"指球"）。指错方向会导致肩转停在 70-80°，无法激活背部大肌群（via OTI rdvLQo4Eb1M）|
| **Reactive Brake / NDA-as-brake**（左肘急收作刹车片）| 击球瞬间左肘向身体猛收 → 角动量守恒 → 拍头自动以 1.5x 切线加速度甩出。这是"撕"的力学原理（via OTI bMR1esW_hCw）|
| **Trunk Dissociation / Shoulder Tilt**（高球处理用肩轴倾斜）| 处理胸部以上高球时，**整个躯干向击球侧倾斜**让肩线对齐高球，禁止单独抬手臂（避免 Shoulder Impingement）。腋下夹角必须保持 45-60° 恒定（via RacquetFlex UNSLArLm0nc）|

### ⭐ "左臂最终平行于网"（Bourne 共同性 #4） —— NDA 几何标志的印证版

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.14)
>
> 原文："Many athletes, such as Djokovic, hold on to their opposing hand longer, but **the left arm being parallel to the net is the pre-requisite.**"

Bourne 五大共同性之一：**不论左手扶拍多久（Djokovic 式扶得久 / 简化版直接放到位），
最终左臂必须平行于网**。这是引拍完成的硬几何条件。

**与现有 NDA 平行底线（OTI rdvLQo4Eb1M）的印证**：OTI 给的标准是"左臂与底线重合"，
Bourne 给的是"左臂平行于网"——**底线和网平行，所以这两个表述指向同一几何**，
两个体系独立得出同一结论 = 高置信度。

**Bourne 强化的执行细节**：
- 这个动作要做得"strong and definite"——不是飘出去，是有力地伸到位
- 给击球一个一致的起点（"clear and precise point to begin from every time"）

### ⭐ 肩转量分时间分辨率：45° → 65° → 90°+ → 180°

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.13 / p.20 / p.19)

Bourne 在书的不同位置给出不同肩转数字——**不是矛盾，是不同时间点的快照**：

| 时间点 | 肩转角度（相对网）| 来源 |
|---|---|---|
| Initial Unit Turn 启动期 | 约 **45°** | Bourne p.13 |
| 引拍中段 / "抱球"那一帧 | 约 **65°** | Bourne p.20 RP4 |
| Coil 完成 / 前挥起始 | **至少 90°**（甚至 110°+）| Bourne p.19 + Brian Gordon |
| 前挥结束时（后肩面网）| 累计转动 **180°** | Bourne p.19 |

**Bourne 反传统命题**（p.13 强）：
> "The common and traditional advice to immediately 'take your racket back,' with the independent motion of the arm is **flawed. Avoid doing it.**"

立刻拉拍（手臂独立后撤）是**错的**——拍是身体转动**带回去**的，不是手拉回去的。
这条直接命中诊断链 `arming_the_shot_false_lag` 的根因层。

**关键警告**（p.23）：
> "A very common cause of inconsistency in the Forehand drive is a lack of or incorrect shoulder turn."

正手不稳定的最常见原因——肩转不足或角度不对。

### ⭐ 现代正手球速/旋转量化基线

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.19 + p.37)

| 指标 | 现代职业 | 经典挥拍（Sampras/Henman/Agassi 时代）|
|---|---|---|
| 球速 | **100 mph+** | 较低 |
| 上旋 | **3000 rpm+** | ~2000 rpm（约 2/3）|

**力学根因**（Bourne p.37-38）：
- 旋转量来源是**挥拍路径**（universal pronated rotational path），不是握法
- Del Potro / Federer 用偏 eastern 握法仍能完整 pronate → 印证握法不是 spin 主因
- 经典挥拍拍面 "on edge" 通过 extension（朝上或朝前），不做满 pronation
- 现代 = pronated rotational path + 聚酯弦 = 旋转爆炸

**这给 Layer 10 输出控制提供了量化天花板**：业余球员的 spin/speed 目标可参照 60-70% 职业值。

### 知识库当前覆盖

- ✅ 大量（21_ftt_chest_engagement, 22_scapular_glide, 4/22-26 学习记录）
- ✅ 概念网络 C04, C19, C21, C36 等
- ✅ **VLM prompt 重点覆盖**
- ✅ Bourne p.13 / p.14 / p.19 / p.20 / p.23 印证 Unit Turn + 肩转 + 左臂平行网 + 反对立刻拉拍

---

## Layer 7 · 手臂结构（Arm Structure）⭐⭐⭐

### 维度

| 维度 | 选项 |
|---|---|
| **击球臂形态** | 直臂（≥165°）/ 接近直臂（150-165°）/ Double-bend（< 130°）|
| **手腕状态** | Lag（被动后翻）vs 主动 |
| **肘部位置**（触球时）| 在身体前方 vs 在身体侧方 |
| **拍面角度** | 闭合 / 中性 / 开放 |
| **拍头滞后**（Wrist Lag）| 拍头落后手腕的角度 |
| **握拍微观角度**（Grip Angle）| 斜跨（Djokovic 式 → 必须 Double-Bend）vs 锤式（Nadal 式 → 必须 Straight-Arm）（via 6TdUOe5nswI / 8oXhd48MSCs）|
| **食指扳机指间距**（Pistol vs Hammer）| 食指张开成钩（Pistol，释放 pronation）vs 四指并拢（Hammer，锁死手腕）（via 2fZmWDOVoRA / IaCikOeS0oQ）|
| **Lag 时拍面方向** | 安全：手腕 Extension（后仰，掌心向前）/ 危险：Ulnar Deviation（侧折，拍面扣地）（via dzAIDEiFBv4）|
| **拍面闭合方式**（高短球时）| 安全：Vertical Closing（靠击球点前移）/ 危险：Parallel Closing（靠手腕翻面盖球）（via HWM4clOrylA）|
| **Lag 来源**（Supination vs Wrist Extension）| 真 Lag = 旋转启动后**前臂被动旋后**带出的动态张力 / 假 Lag = 引拍时手腕主动后撇 → 手臂提前 frozen-out → 击球瞬间拍面晃动（via M1umUwuPe0w / ubFJi2M3AMM / wWWDqBKwO3U / hiujcyG1Bkk）|
| **Snap-Snap 双响**（Lag-Pronation 双脉冲）| 完整动力链 = 触球前一响（被动 Supination 进入 Slot）+ 触球瞬间一响（主动 Pronation 释放）。两响之间的时间差决定爆发力。只有第一响而 Hold 住手腕往前推 = 减速伞（via O1i9y5NSoig / tGA__q2qLco / 1fyiKRioGR0）|
| **手部刚性 vs 手臂松弛**（"Loose with arm, not with hand"）| 握拍像握锤子（食指扳机指钩稳）+ 大臂/前臂关节是松的。把"放松"和"放手"分开：放松的是肌肉，不是握力（via ubFJi2M3AMM）|
| **空间-高度补偿律**（Spacing scales with ball height）| 球越高，站位距离必须**指数级**拉远（高于腰部时约 3 倍）。低球贴身没事，高球贴身必 Jammed——因为高球无法靠重力让手臂自然下垂伸展（via A7a8Ibci9MM）|
| **ISR vs Shoulder Flexion 区分** ⭐ | **正手"刷球"上旋的核心引擎是 ISR（Internal Shoulder Rotation）——大臂在肩窝里旋转**，由胸大肌+背阔肌发力。业余球员的"低到高刷"实际是 Shoulder Flexion（肩屈）——三角肌前束发力，肌群弱、力量小。**这是两种完全不同的肌群和动作模式**（via Brian Gordon CV3mDt7I2Ls + RacquetFlex WiZE3es5mEw / BZDjG-GuhVs）|
| **3 点钟拍头预加载** | 引拍最深点拍头应指向身体右侧 3 点钟方向（不是后挡墙、不是地面），目的是让大臂处于内旋预拉伸状态，触发 SSC 牵张反射储能。指向其他方向 = 失去 ISR 加载行程（via RacquetFlex WiZE3es5mEw）|
| **5 点钟引拍出口（Brian Gordon Clock Model）** | 引拍最深点应在身体右后方 5 点-5 点半位置（俯视视角），超过 6 点（指向后挡墙）即 Joint Rotation 过度，丢失肌肉弹性最优区。这给"藏"提供了可量化的深度上限（via Brian Gordon zac_u3TxxDo）|
| **Decoupling（解耦）** | 速度向量（躯干旋转 + 水平内收）和旋转向量（SIR）的物理来源完全分离——直臂状态下两个引擎并行不互相干扰。**这修正"流→撕"是连续序列的旧理解，应该是并行**（via Brian Gordon CV3mDt7I2Ls）|

### ⭐ Bourne 反传统：手不要主动压低于球（修正"低到高"教条）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.32-33)
>
> 原文："The hand may, at most, fall a few inches below the ball. However, on occasion, it never actually descends below it."
> "**The hand is secondary in importance and the height of the racket head is the most important factor.**"

**反传统命题**（直接打脸传统"low to high"教法）：
- 拍头朝下指向场地不是手主动落下来的，是**手臂从肩关节向后旋转**（即 ISR 加载 + windshield wiper 的极端形态）造成的
- "Low to high" 是**被动结果**，不是主动 cue
- **手最多比球低几英寸，有时根本不低于球**

**修正用户认知**：如果用户脑子里有"主动把手压到球下方"的意图——直接打掉。
拍头下落是 ISR + 重力的被动结果，不是主动指令。

**与 Layer 7 现有 ISR vs Shoulder Flexion 区分的印证**：
- 业余的"低到高刷"实际是 Shoulder Flexion（肩屈，三角肌前束发力）
- 真正发动机是 ISR（肩内旋）
- Bourne 用更口语的"手不用压低"表达同一件事——肩做工，手不动

### ⭐ No Wrist Snap + Nadal 腕角恒定（Bourne 重锤印证）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.38)
>
> 原文："There is no 'wrist snap'. You cannot snap the wrist forward if it is not relaxed.
> In fact, **Nadal keeps his wrist angle almost constant throughout the entire forward swing.**
> His extreme pronation is not a 'wrist snap'—it is created by the extreme upward and outward motion."

**反传统命题**（直接反对 wrist snap 教法）：
- Nadal 的腕角度从前挥到 follow-through **几乎恒定**
- Pronation 是肩+前臂的旋转（即 ISR），**不是腕的鞭打**
- "你松不下来根本就 snap 不了"——这条逻辑直接否定主动 wrist snap 的可能性

**与现有 Lag 来源（真 vs 假）+ Snap-Snap 双响的关系**：印证"两响都不是腕主动",
是被动 Supination 进入 Slot + 主动 Pronation 释放（Pronation 来自 ISR，不是手腕）。
**有冲突时以 Brian Gordon 为准**——Bourne p.48 解剖部分用"wrist responsible for rotation"
是含糊措辞，**作者实战立场（p.38）才是真主张**。

### ⭐ 反对"主动 across body"（Bourne）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.35-36)
>
> 原文："Bend the elbow sharply and pull the racket across before making contact... **This is false.
> The fundamental shape of the hitting arm structure is maintained well into the forward swing
> and out toward the extension point.**"

**反传统命题**：Across（横扫过身体到左肩）**只在接触之后**才出现，**接触前不允许主动横切**。
Hitting arm 的形态保持到 extension point 才开始变。

**修正用户潜在意图**：如果用户挥拍时脑子里有"挥到左肩"的意图，要在心里把它推迟到接触后。
接触前所有意图都应该是 inside-out（从身体内侧向外延展），不是横向切割。

### ⭐ Forward Swing 三维（up + out + across）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.35)
>
> 原文："Three forward motions are made by the hand and racket: **upward, outward, and also across.**
> These are three clearly and distinctly different dimensions."

挥拍前向运动是三维分量的合成：
1. **Upward**（向上）—— 提供 spin 与高度
2. **Outward**（向外）—— inside-out 的核心，从身体内侧向场地外侧延展
3. **Across**（横切）—— 仅在接触之后

**起点条件**：手到眼高 + 手到左侧躯干边缘 = forward swing 启动。
这给 Layer 2 时序提供了一个新的可观测帧标——比"何时开始加速"更早的"何时进入前挥"边界。

**与 Layer 10 的 Out / Up / Through 三向量模型的关系**：FTT 三向量描述的是"力的分配比例"，
Bourne 三维描述的是"路径的几何方向"——两者正交互补，不冲突。

### ⭐ 主动入口 vs 被动末端（用户 4/30 体感印证）

> 来源：用户 2026-04-30 训练突破 + `docs/research/21_ftt_chest_engagement.md` (line 30-44, 48-92) + `docs/research/arm_trunk_coupling_biomechanics.md` (line 11-25)
>
> 用户原话："已经能逐渐感觉到胸部发力，也就是肘部往前推的感觉。"

**核心命题**：手臂不是一根均匀的棍子——**近端（胸-肩-大臂-肘）必须主动驱动，远端（小臂-腕-拍）必须被动甩出**。混淆这条边界 = arming 复发。

| 段 | 性质 | 力学 | 体感 |
|---|---|---|---|
| **胸大肌 + 背阔肌**（驱动源）| 主动 | ISR + 肩水平内收发力第一站 | "胸在挤压 / 在推" |
| **大臂 + 肘**（被推段）| 主动方向、被动稳定 | 大臂住肩窝（4/29），肘被胸推往前/向上 | "肘被胸往前送" |
| **小臂 + 腕**（鞭梢段）| **永远被动** | 被前段推力甩出，pronation/wrist 是路径副产品 | "小臂自己甩过来了" |

**主动入口的精确定义**：
1. 躯干转 + 胸大肌 ISR 启动（21_ftt_chest_engagement.md 三阶段模型的 Press 阶段）
2. 把肘往前 / 向上推（不是把拍头往前推，不是把手往前推，是把**肘**推）
3. 胸-背闭环张力带稳住肱骨在肩窝里（arm_trunk_coupling_biomechanics.md 的"前后夹击"）

**被动末端的精确定义**：
1. 小臂被前段推力**甩出**——挥拍轨迹是结果，不是指令
2. Pronation 不是手腕动作，是 ISR 在末端的几何投影（与 L7 现有 Snap-Snap 双响、No Wrist Snap 一致）
3. 拍头加速只发生在接触前后那一瞬，由"胸推 → 肘领 → 小臂被甩"的鞭式延迟自动产生

**与 L7 现有维度的关系**：
- 与 **ISR vs Shoulder Flexion 区分**：补全了"ISR 是从哪里被点燃的"——胸大肌（不是凭空想象大臂自转）
- 与 **真 Lag vs 假 Lag**：主动驱动来自胸推肘，假 Lag 来自小臂主动撇手腕——两者根因都在"主动边界画错位"
- 与 **Snap-Snap 双响**：双响的物理产生条件 = 主动入口正确（胸推肘）+ 被动末端不干预（小臂不主动）

**失败模式诊断**：
- ❌ 体感是"我在用小臂挥" → arming，回到 `arming_the_shot_false_lag` 老路
- ❌ 体感是"胸在用力打球" → 胸自己用力会停在胸，没传到拍头，球软
- ✅ 体感是"我胸推了肘，小臂自己甩过来了" → 边界对了

**为什么这条要立到 L7**：L7 现有维度全部聚焦"手臂的形态/角度/方向是什么"（描述层），这条是"手臂的哪一段是主动哪一段是被动"（驱动层）——两个正交问题。形态对了但驱动反了 = 看着像 ATP 但球软。

**与 4/29 大臂住肩窝的对偶关系**：4/29 是**约束侧**（大臂不该动），4/30 是**驱动侧**（让小臂被甩出来的引擎）。两者拼起来才完整——只有约束没驱动 = 球软；只有驱动没约束 = 大臂跟着乱跑。详见 `docs/record/learning.md` 2026-04-30 entry。

> via 2026-04-30 user breakthrough + FTT chest engagement

### 知识库当前覆盖

- ✅ 4/26 早学习记录（直臂 vs Double-bend）
- ✅ Gordon Type 3 文档
- ✅ KPI ElbowAngleAtContactKPI
- ✅ Bourne p.32-33 / p.35-36 / p.38（反对主动手压低、反对主动 across、反对 wrist snap）
- ✅ 用户 4/30 主动入口体感 + 21_ftt_chest_engagement.md 三阶段 + arm_trunk_coupling_biomechanics.md 三角张力

---

## Layer 8 · 视觉 / 注意力（Vision & Attention）⭐⭐⭐⭐

> **重要的"软层"——决定你的 Cognitive 能不能转 Autonomous**。

### 维度

| 维度 | 含义 |
|---|---|
| **追球**（Ball Tracking）| 眼睛跟随球到接近击球点 |
| **四注视点**（Four Look Points）| FTT C32：弹起前 / 弹起 / 上升 / 击球 |
| **击球瞬间视线**（Quiet Eye）| 触球瞬间眼睛固定 |
| **注意力分配** | 多少注意力在身体感觉 vs 球 vs 落点 |
| **预判**（Anticipation）| 多早开始判断球的轨迹 |

### 失败模式

- 看球失败（在击球瞬间已经看下一拍）
- 注意力全在身体动作 → 反应不过来球
- 4/26 凌晨发现的"反应式 vs 主动式"心智模型

### 知识库当前覆盖

- ✅ FTT C32 四注视点（概念网络已有）
- ✅ 4/26 凌晨学习记录（注意力预算分析）
- ⚠️ 没有专门的视觉训练 drill

---

## Layer 9 · 心智模型（Mental Model）⭐⭐⭐

### 维度

| 维度 | 含义 |
|---|---|
| **Receiving vs Sending** | 找位 vs 挥拍 |
| **预设攻击区** | 是否有固定击球点期望 |
| **决定何时不打** | 让球过的判断 |
| **自我对话**（口令）| 单字触发器 vs 多字描述 |
| **错误归因** | 失误归因于动作还是位置 |

### 知识库当前覆盖

- ✅ 4/24 晚学习记录（Cognitive vs Autonomous）
- ✅ 4/26 凌晨（攻击区主动模型）
- ✅ 8 字口令系统

---

## Layer 10 · 输出控制（Outcome Control）⭐⭐⭐

### 维度

| 维度 | 控制点 |
|---|---|
| **球速** | 拍头速度 + 拍面角度 |
| **上旋**（RPM）| Out vector + 雨刷动作 |
| **方向**（Lateral）| 拍面闭合时机 + 转身停止时机 |
| **深度**（Depth）| 击球点高度 + 拍面角度 + 力度 |
| **弧度**（Arc）| Up 向量比例 |
| **Power/Spin Equation 比例**（量化 Out vs Up）| 总能量恒定，按场景分配：90/10 暴力，60/40 中庸，30/70 高弧（via U3Saz3bCPPo / x-z05u-kfXE）|
| **跳跃击球时的垂直对冲**（Vertical Energy Hedge）| 身体跳起已提供 vertical 分量 → 手必须切回 forward，避免 Up AND Up（via CrVoJL9E69Y）|

### 失败模式（输出层）

- 求稳时降低脚频和挥速 → 击球点错位 → 失误反而增多（应改为只增加 Spin 比例，via x-z05u-kfXE）
- Wiper 动作没有前置 Plow-through → 球短而无威胁（业余通病，via DDZSXrNZAgU）

### 知识库当前覆盖

- ✅ FTT 三向量模型（Out / Up / Through）
- ⚠️ 没有"打不准 → 哪个维度调整"的对照表

---

## Layer 11 · 训练辅助 Cue 层（Bourne One Minute System）⭐⭐⭐ — Bourne 这本书新增的层

> **本层是 Stephen Bourne《One Minute Tennis Forehand Solution》给整个分类框架带来的新维度。**
>
> 现有 L1-L10 都是**力学维度**（什么是对的）。Layer 11 是**教学语言学维度**（怎么把 L1-L10 的复杂约束压缩成可在赛中调用的单一意象）。
>
> 这一层不替代 L1-L10，**它是给 L1-L10 的"压力下回滚工具集"**——压力上来想不动作时回到的那个锚点。

> **结构性定位**：用户的 11 字系统（架/推/锁/撑/藏/流/撕/飘/盯/压/跟）是**实战核心 cue**。
> Layer 11 的 4 个 Bourne cue 是**训练辅助 cue**，挂在 11 字之外，形成两层：
> - **11 字管实战**（场上的瞬时触发器）
> - **4 字管训练**（练习场的几何/触觉锚点 + 压力下的 reset 流程）

### 4 个核心 Cue（每个 cue 同时满足多个 L1-L10 约束）

| # | Cue | 类型 | 对应章 | 同时满足的 L1-L10 约束 | 来源 |
|---|---|---|---|---|---|
| 1 | **"Hold onto the ball"（抱球）** | 视觉意象 | 引拍 | L1 手在右侧 + L4 重量到右脚 + L5 头微倾右 + L6 左臂平行网 + L6 肩转 65° 背向目标 + L6 反对立刻拉拍 | via Bourne p.20-25 |
| 2 | **"Power Point"（拍喉食指→拇指闭环）** | 触觉锚点 | 挥拍 | L2 引拍触发时机 + L6 unit turn 一体 + L7 inside-out 路径 + L7 完整 pronation + L7 不主动手腕 | via Bourne p.43-48 |
| 3 | **"Find / Feel / Use the BOUNCE"（拍在手中弹）** | 本体感觉 | 手腕 | L7 松握 + L7 松腕 + 压力下回滚机制（Layer 9 心智模型的物理验证器）| via Bourne p.52-55 |
| 4 | **"Don't bend knees, raise your heels"（抬脚跟）** | 注意力切换 | 发球 | 完整动力链 + 踝主动 + 膝被动 + 启动时机（拍到肩高）| via Bourne p.64 |

### Cue 1 详解：Hold onto the Ball（抱球）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.20-25)

**意象**：双手怀里抱着一个 US Open 大网球（约篮球大小），引拍过程不让球掉。

**5 个 reference points 自检**（对应 5 大共同性）：
- [ ] 拍是被肘带回去的，不是拍头先走（球没掉）= L6 elbow first
- [ ] 重心已经压到右脚外侧 = L4 right foot load
- [ ] 左臂自然伸到平行于网 = L6 NDA parallel
- [ ] 肩已经转到背离目标方向（约 65°）= L6 shoulder turn 65°
- [ ] 头微倾到身体右侧 = L5 head leads right

**核心逻辑**："5 个点都是抱球这个动作的几何必然——不是要做 5 件事，是做 1 件事得到 5 件事。"

**赛中可调用**：感到引拍变形时，picture the ball in your arms，立即纠正——不只是练习场用的。

### Cue 2 详解：Power Point（拍喉食指→拇指闭环）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.43-48)

**5 步训练协议**：
1. 拍喉（throat）位置贴有色胶布 / 橡皮泥 → 这是 Power Point
2. Ready position：左手食指自然搭在 Power Point 上
3. Take-back：左手带拍后引，**食指尽量长留在 Power Point 上**
4. 击球 + 收尾：挥拍击球，左手接拍位 → **拇指必须落到 Power Point 上**
5. Shadow：闭眼空挥多次直到 finger-to-thumb 转换稳定

**自验证**：起手食指 → 收尾拇指，每次都对。**拇指接不到 Power Point = 挥拍路径不对**——
这是 4 条诊断链（early_front_foot_landing / wta_takeback_midline_violation /
arming_the_shot_false_lag / shoulder_flexion_instead_of_isr）的同时验证器。

**加速悖论**（p.43）：用 Power Point 后，速度+10% → 旋转+15%；速度+40% → 旋转+60%。
**越用力越安全**——挥拍路径越对，能量越多走旋转通道。

**核心句**："The harder you hit your new Forehand, the more the ball goes in."

### Cue 3 详解：Find / Feel / Use the BOUNCE（拍在手中弹）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.52-55)

**核心命题**：握拍微紧 = 锁死腕部 32 条肌肉 = lag 消失 = 发力链断。**敌人不是缺 snap，是过度紧张**。

**Bounce 验证**：握拍空间略大于手的握圈 → 拍可以在手内**轻微滑动/弹跳** → 此时手和腕双双松弛。
"When you feel the bounce... your hand and wrist will both be relaxed and loose."

**这是因果反着用的诊断**——你不能直接命令"放松"，但可以通过"听 bounce"间接验证松。

**三阶段递进**：
- **Find**（离场训练）：发现 bounce 是什么感觉
- **Feel**（上场前 30 秒）：每次发球前/接球前重复一次 bounce 检查
- **Use**（高压时刻）：感到"动作没节奏"时立刻 unit turn 暂停做一次 bounce

**与 FTT 握拍微区贴合的对偶**：FTT 教"贴"，Bourne 教"松到能 bounce"——同一件事的两面，
太松失去贴合，太紧失去 lag。

**与"飘"字的关系**：BOUNCE 给"飘"字第一次提供了**可操作的验证 drill**。
"飘"是手腕末端释放的描述（结果），BOUNCE 是过程检查（原因）。

### Cue 4 详解：Don't bend knees, raise your heels（抬脚跟）

> 来源：Stephen Bourne《One Minute Tennis Forehand Solution》(via Bourne p.64)
>
> 原文："Instead of thinking about bending your knees, you must think about raising your heels."

**反传统命题**：传统教"屈膝"会把膝盖以下（小腿/跟腱/脚）锁死，动力链从中段砍断。
**膝盖不产生动量，只能引导动量**——动量来自踝+脚。

**时机锚点**（p.64）：拍头到肩高的时候开始抬跟。这是发球唯一可执行的时间 cue。

**类比**（p.65）："你坐下时不会想'我要屈膝'，膝盖只是动力链里被动弯的一环。发球同理。"

**与 Layer 4 OTI Sit-Stand 的关系**：不冲突——OTI 的 Sit 也是地面反作用力的被动结果，
跟 Bourne "膝被动弯"立场一致。OTI 更强调 Stand 释放那一段。
**Bourne 的"抬跟"是对 OTI Sit-Stand 在发球场景的口令化版本**。

**适用范围警告**：这本书自称发球解决方案是夸张了——它解决的是发球**蹬地这一个动作**。
Toss 抛歪了、握拍是 eastern 不是 continental、racket drop 不够深，光抬脚跟救不回来。

### Cue 5（候选）：**"推"（胸推肘）** ⏳ 用户 4/30 候选 cue，待 1-2 周训练验证

> **状态**：用户 2026-04-30 训练首次摸到的体感，以"训练辅助 cue"挂在 4 个 Bourne cue 之后。
> 验证 1-2 周后再决定是否升级进 11 字实战系统（候选位置在"流"之前）。

> 来源：用户 2026-04-30 训练突破 + `docs/research/21_ftt_chest_engagement.md`（FTT 胸三阶段：Attached / Press）+ `docs/research/arm_trunk_coupling_biomechanics.md`（胸-背闭环张力带）

**意象**：前挥起步那一瞬，**胸把肘往前/向上推**——不是把拍头往前推，不是把手往前推，是把**肘**推。小臂跟着自己甩过来。

**时机**：unit turn 完成 → 接触前的那一段（前挥起步）。
- 比"流"早一帧——"流"描述前挥过程的整体顺序，"推"是过程的发动机点火
- 与"撑"（左臂反向支撑）同时发生——一个是反向锚点，一个是正向驱动

**物理基础**：
- **胸大肌是 ISR 的第一发力源**（21_ftt_chest_engagement.md line 48-92 三阶段模型的 Press 阶段：胸主动缩短，把肱骨从外侧拉回身体中线）
- 与背阔肌形成"前后夹击"把肱骨锁在躯干上（arm_trunk_coupling_biomechanics.md line 23）
- 肘前移是这个力的几何投影——你不需要主动伸肘，胸推就把肘送出去

**自验证（关键体感判断）**：
- 是**被甩感**（小臂自己甩过来，没主动用力）→ ✅ 对的
- 是**主动挥感**（我在用小臂挥拍）→ ❌ arming 复发，立即停

**失败模式**：
- ❌ 胸推肘的同时**主动用小臂挥** = 两条主动力同时存在 = 链条断裂 = arming 复发
- ❌ 把"胸推肘"理解成"用胸的力量直接打球" = 胸自己用力会停在胸，传不到拍头 = 球软
- ❌ 在没有 4/29 大臂住肩窝的前提下用"推" = 没约束的驱动 = 大臂跟着乱跑

**检查方法**：
1. **体感问句**：每次挥拍后立刻自问——"刚才小臂是被甩出来的，还是我主动挥的？"
2. **球质对照**：4/29 之前 vs 加上"推"之后，球的旋转 / 深度 / 重感应该有可测差异
3. **视频核对**：肘前移方向是向前/向上为主（对），还是向场地外侧水平推出（错，那是 Shoulder Flexion 不是 ISR）

**与已有 4 cue 的关系**：
- "抱球" → 引拍阶段约束（5 个 RP 几何同时到位）
- **"推" → 前挥起步驱动（驱动源点火）** ⏳ 候选
- "Power Point" → 路径触觉验证（finger-to-thumb 闭环）
- "BOUNCE" → 手腕松弛验证（发力链不断的前提）
- "抬跟" → 发球场景动力链触发

**升级判定标准**（1-2 周后回看）：
1. 在挂球阶段稳定后，能否迁移到慢速喂球？
2. 进入正常球速后，"推"会不会变形成"主动挥小臂"？（变了 = arming 复发 = 不能升级）
3. 球质有可测改善吗？
4. 加上"推"后实战检查链是否过载？（之前是抱球 + BOUNCE + 右脚撑三件事，加"推"是四件事，是否还能在分与分之间跑完？）

**与 L7 主动 vs 被动分层小节的关系**：L7 描述"力学事实"（哪段主动哪段被动），Layer 11 的"推"提供"赛中可调用的口令"——把 L7 的边界压缩成一个动词。

### 压力下的 Reset 流程（Bourne 系统的工程化输出）

当实战变形时（用户经典卡点："空挥能做实战做不到"），按这个顺序回滚：

```
下一分前  → Bounce Discovery（30 秒）→ 验证手腕松
架拍那一秒 → 抱球意象 → 5 个 reference points 自动到位
击球瞬间  → 右脚撑死（4/27 圣经）+ 大臂住肩窝（4/29 突破）
如果还崩 → 直接走过去捡球，重新做 1-3
```

### 与现有 11 字系统的衔接

| Bourne Cue | 衔接的 11 字 | 关系 |
|---|---|---|
| 抱球 | 架 / 推 / 撑 / 藏 / 流 / 左 | 一个意象同时触发 6 个字 |
| Power Point | 流 / 撕 | 触觉验证流和撕是否完成 |
| BOUNCE | 飘 | 飘的过程检查器 |
| 抬跟 | 压 / 跟 | 压（右脚承重）的发球版触发器 |

### 知识库当前覆盖

- ✅ Bourne p.20-25 / p.43-48 / p.52-55 / p.64（4 个 cue 完整收录）
- ✅ 与用户 11 字系统形成两层结构（实战 / 训练）
- ❌ **诊断引擎不识别这 4 个 cue 的执行情况**
- ❌ **VLM prompt 不问"是否在抱球意象内"或"是否有 finger-to-thumb 闭环"**

---

# 综合：诊断 Checklist

每次正手诊断必须按顺序检查：

```
L1 几何        → 击球点 4 坐标都对吗？站位距离对吗？
   ↓
L2 时序        → Unit Turn 时机对吗？Bounce-Hit 节奏对吗？
   ↓
L3 步法        → 是否有 Split Step？调整步够吗？
   ↓
L4 站姿        → 站姿选对了吗？重心分布对吗？
   ↓
L5 姿态        → 脊柱稳定吗？后仰吗？
   ↓
L6 上半身机制  → 胸推到了吗？锁住了吗？X-Factor 出来了吗？
   ↓
L7 手臂结构    → 击球臂形态对吗？拍面角度对吗？
   ↓
L8 视觉注意力  → 追球到位了吗？注意力预算合理吗？
   ↓
L9 心智模型    → Receiving 在练吗？还是只在 Sending？
   ↓
L10 输出控制   → 想要的速度/旋转/方向出来了吗？
   ↓
L11 训练辅助 cue → 抱球 / Power Point / BOUNCE / 抬跟 任意一个能在压力下回到吗？
```

**如果 L1-L2 不对，下面所有层都是浪费**。

**L11 是横切层**——它不是"再上一层"，而是给 L1-L10 提供"压力下不掉链子"的回滚工具。诊断时 L1-L10 走完后，再用 L11 检查"这些约束有哪个 cue 能在赛中调用"。

# 现状审计：4/22-27 一周的诊断分布

| 层 | 我有覆盖吗 | 该有的深度 | 实际深度 |
|---|---|---|---|
| L1 几何 | ❌ → ✅（4/26 凌晨补） | 5 | **0 → 4** |
| L2 时序 | ⚠️ 碎片 | 5 | **2** |
| L3 步法 | ⚠️ 提了没分析 | 4 | **1** |
| L4 站姿 | ✅ 4/26 晚 | 4 | **3** |
| L5 姿态 | ⚠️ 4/26 晚 | 4 | **3** |
| **L6 上半身机制** | ✅✅✅ | 4 | **5（过度）** |
| L7 手臂结构 | ✅ 4/26 早 | 3 | **3** |
| L8 视觉注意力 | ⚠️ 4/26 凌晨 | 4 | **2** |
| L9 心智模型 | ✅ 4/24 晚 + 4/26 凌晨 | 3 | **3** |
| L10 输出控制 | ❌ 完全没碰 | 3 | **0** |
| **L11 训练辅助 cue** | 🆕 Bourne 这本书新增 | 4 | **新层，待集成进训练流程** |

**模式诊断**：
- 我**重度偏向 L6**（上半身机制），过度聚焦
- L1-L3（地基）几乎裸奔
- L8 + L10（视觉 + 输出）几乎不存在

# 修复行动

1. **VLM prompt 必须问 L1-L10 全部维度**（即将更新）
2. **诊断引擎必须能识别 L1 + L2 信号**（即将添加概念）
3. **每次新视频分析必须按 10 层 checklist 走一遍**（流程性）
4. **学习记录每周回顾时检查"哪些层一周没碰"**（监控）

---

**这个文档是地基**。以后任何正手讨论 / 诊断 / 视频分析 / 学习记录，都要先回到这 10 层 checklist。**漏一层 = 不完整 = 错。**

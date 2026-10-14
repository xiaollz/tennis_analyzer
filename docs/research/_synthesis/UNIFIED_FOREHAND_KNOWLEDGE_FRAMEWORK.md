# 统一正手知识框架（Unified Forehand Knowledge Framework）

> **状态**：可交付主框架 v1  
> **编制**：2026-10-12（Asia/Shanghai）  
> **截点**：用户学习状态截至 **2026-08-24**  
> **合并源**：`FRAMEWORK_DRAFT_FTT.md` · `FRAMEWORK_DRAFT_CROSS_COACH.md` · `FRAMEWORK_DRAFT_BOOKS.md` · `FRAMEWORK_DRAFT_USER.md`  
> **证据标签**：`VCA`（视频内容分析）· `BOOK`（书 PDF/笔记）· `SYN`（多源合成）· `META`（仅标题/播放列表，不可当主张）· `USER`（用户自验金标）  
> **硬规则**：不编造时间戳；Feel Tennis 会员视频标「可能不可及」；**严禁主动推肘口令**；Intuition-First——机制解释 ≠ 场上 cue 堆。

---

## 1. 这份框架是什么 / 不是什么

### 1.1 是什么

- **知识裁决与诊断底图**：把跨教练共识、FTT 容错机制、书本生物力学、用户自验金标挂到同一阶段骨架上，供后续诊断、训练设计与视频补看排队。
- **Intuition-First 的「Reason 层」**：Reason 设方向与边界；场上执行靠外部结果与已验体感，不靠背诵本文件。
- **证据分层文档**：每条主张尽量带 `VCA` / `BOOK` / `SYN` / `USER` / `META`；`META` 只进缺口，不进共识主张列。

### 1.2 不是什么

| 不是 | 说明 |
|------|------|
| **场上 cue 堆** | 禁止把本文当成「每球要想的 15 条口令」。用户处 Reason Plateau：≤3 cue，默认外部焦点。 |
| **唯一正确姿势说明书** | FTT 驱动光谱（肩鞭 ↔ 胸压）与 Gordon Type 2/3 并存；共性在 Press Slot / 近端→远端释放，不在引拍外观一致。 |
| **已全部实看的视频库** | synthesis 标题级要点 ≠ 本地完整 VLM；Feel 会员片标「可能不可及」。 |
| **发球数据直接搬正手** | Tennis Science 发球 ISR ~40% **不可**自动写成正手定律（正手为 selected / 综述 ~35–40%）。 |
| **自动升级 metadata** | One Minute Tennis / Feel 频道大量 `metadata_title_and_playlist_only` → 不得当处方。 |

### 1.3 一句话使用方式

> 解释世界用书本 + FTT 条件化因果；裁定「该用户下一球」用金标；设计场上任务用外部结果 + 永久禁令；补证据用缺口清单。

---

## 2. 权威裁决序

与 Hermes `04_PERMANENT_RULES.md` 及 Cross-Coach 草稿对齐：

| 序 | 权威 | 用途 |
|---:|------|------|
| 1 | **用户自验金标**（`USER`） | **永远最高**——个案下一球、体感冲突后者胜 |
| 2 | **Tennis Science**（Elliott/Reid/Crespo）+ Reid 2013 正手综述 | 学术 / 可测命题；冲突时高于教练口号 |
| 3 | **HSA 框架**（项目自有 + FTT + Gordon 整合） | 释放层统一名 |
| 4 | **JUL Tennis** | 握拍 / 鱼际丘等硬件层 |
| 5 | **FTT（Hugh Clarke）** | 容错哲学与条件化因果；**教练哲学冲突默认 FTT 胜**（除非金标覆盖） |
| 6 | **Brian Gordon** | Type 3 / ISR / 3D 动捕；**硬数据终审** |
| 7 | **Bourne（One Minute Tennis 书）** | 教学语言学 / 一分钟 cue（标 `BOOK`，≠ 频道视频） |
| 8 | **TPA / Feel（已 VCA）/ Intuitive / OTI** | 机制与体感补充 |
| 9 | **教练社区 / 论坛 / `META` 标题** | 最低；仅候选队列 |

### 2.1 冲突裁决流程（可执行）

```text
冲突出现
  → Q0: 用户是否已有自验金标覆盖？ → 是：金标胜出；停
  → Q1: 是否为 3D/学术可测命题（ISR%、肘伸展、向量解耦）？ → Gordon / Tennis Science；停
  → Q2: 是否为「语言字面 vs 机制」？（如 FTT Pull Forward）→ 机制取 TPA Throw；语言地图可保留 FTT；停
  → Q3: 是否为容错/条件化任务选择？ → FTT 胜出；停
  → Q4: 是否仅为教学意象差异、物理同构？ → 记「同构多叫法」；优先用户已验 cue
  → Q5: 证据一侧为 META / 未 VCA？ → 不裁决；进缺口清单
```

### 2.2 机制语义补丁（已记 learning）

- **球场释放 cue**：以 **TPA Throw / release** 为准；FTT “Pull Forward” 仅早期防错地图，**不作当前主 cue**。
- **关节角速度 / ISR 贡献 / 肘伸展**：Gordon 终审，不以 2D 教条压过。

---

## 3. 总览图：目标函数 + 大问题树 + 通用阶段

### 3.1 目标函数（FTT 容错优先 · `SYN`/`BOOK`）

网球不是孤立最大化拍头速度，而是在时间/空间/来球不确定下提高**预期得分并降低失误与身体风险**。

> **有效击球价值 ≈（触球质量 × 可控拍头速度 × 合法轨迹窗口 × 目标选择 × 回位准备）−（时间压力、失衡、拍面不确定性、风险成本）**

**容错性（Fault Tolerance）** = 技术在不完美准备/时机/站位下仍能产出可用球的能力。甜区接触优先于炫技动力链（力量税叙事：偏心可大幅掉速，`SYN` / FTT `KU7FHy1qQOI`）。

书本锚：出球速度约 **75% 拍速 / 25% 来球**（Tennis Science Ch2 p.32，`BOOK`）。

### 3.2 用户层大问题树（截至 2026-08-24 · `USER`）

```text
【大问题】手主导 · 手不肯把球拍交给身体（30 年投掷 GMP）
                │
  ┌─────────────┼─────────────┬──────────────┐
引拍端         降拍端        前挥端         触球端
右臂主动后拉   手动放置 drop  串行·手臂拉    刷/扶/腕翻
 ✅ 已解决     🔄 动态化中    🔄 可重复未自动  ⬇️ 下游不单练
                │              │
         有支撑落槽↑      「行进」入口 / 前向 Hit
                                    │
                         左手/Off-Arm 缺位 🔄
                         轴滑回拍底 / 拍柄滑 🔄（8/24 接口规则待球场）
```

**现行主战场一句**：已从「能否做出」进入 **转换阶段能否保持**——动态肩胛支架 + 手拍接口在转换/触球区共同保持。

### 3.3 通用阶段（二极管模型 · FTT）

准备的每一环只服务对应前挥环节（书：preparation forms a diode with forward swing）：

```text
信息接收 → Split/第一步 = Unit Turn 启动
  → Unit Turn 完成（目标：球过网前）
  → Wait（视觉等到位；快球可为 0）
  → 垂直分量建立（drop/slot；Be Patient）
  → 前挥脉冲（髋/腹/Unit Swing + Out / Throw）
  → Press Slot / 触球（晚加速）
  → 被动随挥 + 回位（圆锥中轴）
```

---

## 4. 统一阶段框架

> 每阶段列出：**目的** · **可观察点** · **主动/被动边界** · **挂载共识节点** · **用户现行状态**。

### 阶段 0 · 信息接收（Receiving）

| 项 | 内容 |
|----|------|
| **目的** | 匹配空间与时间再输出；Receiving > Sending（FTT） |
| **可观察** | Split；第一步已侧身；视线三阶段看球 |
| **主动** | 早准备；呼吸：鼻吸等待 |
| **被动/禁** | 不把「只练挥拍形状」当接收训练 |
| **共识** | C14 外部焦点；C15 正手可作弊（镜前≠球场） |
| **用户** | 默认场景：发球机最低速 + mini tennis；8/9 呼吸节奏锚 |

### 阶段 1 · Unit Turn / 准备

| 项 | 内容 |
|----|------|
| **目的** | 整身转离目标蓄旋转；把手放到可绕躯干甩出的位置；建立上背连接——**不是**持拍臂独立后拉 |
| **可观察** | 脚发起转身；双手仍相对胸前；非击球臂参与侧转；肩胛有支撑感；右脚为轴 / Sit not Push |
| **主动** | 脚+髋胸共转；拉非击球侧 / 「想左手忘右手」；ESR 第一帧校准（用户个案）；高宽+离身宽度 |
| **被动/禁** | 禁独立后拉；禁准备期主动制造腕 lag / 拍底对网；禁夹死肩胛用背提拍；禁「先转髋」主动 cue |
| **共识** | C1 Off-Arm · C2 Unit Turn · C3 右脚轴 · C8 肩胛支撑 · C10 ESR · C11 Hypothenar |
| **用户** | 引拍端后拉几何 ✅（手环游世界）；动态肩胛支架 8/18；ESR 优先重叠发生 8/23 |

### 阶段 2 · Drop / Slot（垂直分量）

| 项 | 内容 |
|----|------|
| **目的** | 拍头降到球同高或更低；进入可 Press 的槽；**不**用小臂主动「pat the dog」硬压 |
| **可观察** | 加速前手至少到球高（优选更低）；肘有空间 → Scapular Glide；有支撑落槽 ≠ 塌陷 |
| **主动** | Be Patient（等垂直建立）；高度调节：腿蹲 > 躯干倾 > 伸肘 |
| **被动/禁** | 禁主动降拍；禁拉臂贴身降拍；禁腕调高度；Pat-the-Dog 是结果非任务（TPA） |
| **共识** | C5 Lag/Flip · C8 肩胛 · 垂直→水平（FTT） |
| **用户** | 8/14 多数拍有支撑落槽；个别仍整臂先变低 → 保持率主战场 |

### 阶段 3 · 前挥释放（水平分量 / 脉冲）

| 项 | 内容 |
|----|------|
| **目的** | 短促旋转脉冲 + Out / Throw 路径；胸 Attached→Press；腿放大而非「先腿后肩」阶梯口令 |
| **可观察** | 近端结构前行；侧外空间；晚加速；胸带臂非推肘；重叠模式（非停最低点再拉手） |
| **主动** | Out / extend；Throw-release 意象；呼气挥；拉非击球肘绕身（Unit Swing） |
| **被动/禁** | lag、雨刷、旋前、ISR「撕」多为结果；**严禁推肘**；禁主动翻拍/扣腕当发动机；禁刻意控髋肩毫秒时序 |
| **共识** | C4 动力链 · C5 Lag · C6 Pronation/释放 · C7 Spacing · C9 HSA · C14 少指令 |
| **用户** | 8/14 前向 Hit 可重复未自动化；主语是「行进」；HSA 是释放层，≠ 胸部主动推 |

### 阶段 4 · 触球

| 项 | 内容 |
|----|------|
| **目的** | 在击球**区**（前+侧外）稳定拍面；穿透；上边缘领先；甜区优先 |
| **可观察** | 区内面近恒定；可接受约 10° 无意识闭合；躯干近端制动甩远端；Hit 先于 Roll |
| **主动** | 空间：球进区，触球外且前；Out + through contact |
| **被动/禁** | 禁主动翻拍纠面/造上旋；禁刷/扶/腕翻当单独练习（用户：下游结果） |
| **共识** | C7 Spacing · C9 HSA · C12 随挥诊断性（触球后） · Press Slot / top edge（FTT） |
| **用户** | 刷球 ⬇️ 不单练；V 区 seating 8/24 待球场；轴滑回拍底→腕刷 |

### 阶段 5 · 随挥 / 回位

| 项 | 内容 |
|----|------|
| **目的** | 随挥是诊断与减速，不是发力发动机；回位到对手回球角圆锥中轴 |
| **可观察** | 自然翻拍/路径；肘可在胸前结束；Land and Ready |
| **主动** | 回位拦截意识 |
| **被动/禁** | 禁为造型主动翻拍；禁「必须收到左肩」反推触球原因 |
| **共识** | C12 随挥=减速路径 · C13 站姿条件化 |
| **用户** | 随挥终点左肩 vs 头顶可作串行/重叠视觉判决（6/12） |

### 4.1 主动 / 被动边界（球场可想 vs 自动）

| 类型 | 内容 |
|------|------|
| **主动 / 有意识** | 早准备；右脚轴 / Sit；Off-Arm / 想左手；ESR 第一帧校准（个案）；垂直到位（Be Patient）；Out / Throw；呼气；回位 |
| **被动 / 自动** | 腕 lag；拍底朝网动态位；windshield wiper；前臂旋前；ISR「撕」；随挥路径；毫秒级髋→腹时序；肘前伸外形（HSA 结果） |

---

## 5. 共识节点总表（C1–C15）

> 吸收 Cross-Coach C1–C15；每行挂 FTT / 书本 / 用户金标。证据取最强可用级别。

| # | 共识主题 | 各教练叫法（源中出现过） | FTT 挂载 | 书本挂载 | 用户金标 | 证据 |
|---|----------|--------------------------|----------|----------|----------|------|
| **C1** | 非持拍侧主动 → 持拍侧被动 | Feel: Left active/right passive；FTT: 拉非击球肘；TPA: Sync arms；Bourne: 左臂平行网；JUL: Seesaw | Unit Turn 非击球臂；Place–Hold–Pull | 投掷 glove-arm 对照（非 TS 专章） | 4/9「想左手忘右手」；4/4 左侧拉=方向盘 | `USER`+`VCA`+`SYN` |
| **C2** | Unit Turn / 禁手臂独立引拍 | TPA: Unit Turn continuous；Feel: coil；Intuitive: Sync Drill；OTI: NDA；Bourne: Hold onto the ball | 整身 UT；禁独立后拉 | TS trunk foundation；Kovacs 引拍离心 | 6/10 手环游世界 ✅；6/20 肘轴引拍 | `VCA`+`BOOK`+`SYN` |
| **C3** | 右脚/外侧腿为轴，左脚不抢轴 | 用户圣经；Bourne: weight into right foot；Intuitive: V-Formation；OTI: The Post | 脚发起 UT；开/半开/中立可用 | TS leg drive / GRF；Kovacs 地面转移 | 4/27 右脚为轴；5/10 Sit not Push | `USER`+`VCA`+`BOOK` |
| **C4** | 近端先动 / 制动传远端 | TPA: Fire next link；OTI: Stopping Effect；FTT: trunk sequencing；Gordon: hips→torso→arm | Trunk 脉冲；动量交接冻结 | Proximal-to-distal；Reid：ISR 可部分偏离经典序列 | 7/14 谁承力谁先发力；感知张力不感知运动 | `VCA`+`SYN`+`BOOK` |
| **C5** | Lag / Flip 是连接产物，不可主动扣腕 | FTT: Flip / Hand Slot；TPA: controlled lag / Slap into Slot；Gordon: Flip=力转换枢纽；Bourne: Feel the Bounce | Hand Slot；禁准备期造 lag | SSC；远端多为近端结果（Reid） | 有支撑落槽 8/13–8/14 | `VCA`+`SYN` |
| **C6** | Pronation / 释放多为被动，反对 wrist snap | TPA Pron→Sup→Pron；Gordon ISR≈50% RHS（其数据语境）；Bourne: no wrist snap | 禁主动旋前；随挥旋前适应 | Reid：旋前 largely 近端结果；TS：IR may after impact | 5/11 4-Step 中 ISR撕=释放步；禁主动 ISR | `VCA`+`BOOK`+`SYN` |
| **C7** | Spacing 是力量前提 | FTT: Out + 侧向空间；TPA: 离球远一步；OTI: yoga-ball spacing | Out 向量；击球区前+侧外 | L1 几何；behind body→伤病（TS Ch7） | 击球点/被挤→先查轴与空间 | `VCA`+`SYN` |
| **C8** | 上身支撑：肩胛/背 isometric，肘不塌 | FTT: scapular support / press slot；Ascended high-set；JUL: pull back upper arms | Scapular Glide；被动臂 | TS Ch7 scapular rhythm；Kibler funnel；Kovacs 前锯/菱形 | 4/30 肩胛槽；8/18 动态支架 | `USER`+`VCA` |
| **C9** | HSA / 胸-肩闭合驱动 RHS 大头之一 | FTT: chest engagement / press；Gordon: horizontal adduction；JUL: shoulder HA 几何 | Chest Attached→Press；Press Slot | Reid H-flex ~25%；Kovacs 胸大肌内收 | 5/3 HSA；**禁推肘/胸部主动推** | `USER`+`VCA`+`SYN`+`BOOK` |
| **C10** | ESR 先于 IR：抑制抢跑 + 蓄力 | Bourne Power: ESR 章；JUL: Chinook；FTT: lag/slot 体感侧 | 垂直耐心 / ESR-SSC 侧 | TS Ch2 p.34 ESR 储能；项目 ESR 刹车排序=用户理论 | 5/8 ESR；8/23 重叠非串行 | `USER`+`BOOK`+`VCA` |
| **C11** | 握拍硬件：重量在小指鱼际 | JUL: Hypothenar | 握拍旋转轴在 V 区 | — | 5/7 Hypothenar；8/24 V 区 seating（待球场） | `USER`+`VCA` |
| **C12** | 随挥 = 减速路径 / 检查而非造型 | TPA: Box Finish；FTT: Hit-before-Roll；OTI: Post 收招 | 被动随挥；随挥诊断性 | Kovacs 离心减速；TS Ch7 | Hit 先于 Roll 8/13；6/12 终点判决 | `VCA`+`SYN`+`BOOK` |
| **C13** | 站姿条件化：开放/半开放默认 | Intuitive: Penetration；OTI/TPA/Bourne: 半开放母板；Feel 开放式（部分 VCA） | 开/半开/中立；封闭不推荐；髋能对目标 | Kovacs 开放式旋转优势≠唯一站姿 | 关闭式：左脚承重∩结构前行重叠（8/14） | `VCA`+`BOOK` |
| **C14** | 外部焦点 / 少指令执行 | Intuitive: non-thinking；Bourne: Language of Body is Feeling；Feel: feel-based；项目 Intuition Paradox | 球场短清单 ≤ 数条 | TS Ch1 类比 > 纯指令；Ch4 执行型自我对话 | 5/6 Intuition Paradox；Reason Plateau | `VCA`+`BOOK`+`USER` |
| **C15** | 正手可「作弊」→ 镜前完美球场失败 | Feel: green check mark / slap without coil；项目元问题 5/9 | Receiving vs Sending；容错目标 | Ch4 训练易进 zone | 5/9 元问题；4/9 想左手 | `USER`+`VCA` |

**共识节点数：15（C1–C15）**

### 5.1 FTT 机制节点 → 共识挂载（摘要）

FTT 草稿 24 机制节点不另立第二套编号；合并时按主题挂到 C1–C15 或阶段：

| FTT 节点簇 | 主要挂到 |
|------------|----------|
| Receiving / 容错目标 / 力量税 | 目标函数 · C14 · C15 |
| Unit Turn / 被动臂 / Place–Hold–Pull / Wait Then Fire | C2 · 阶段 1 |
| 垂直→水平 / Hand Slot / Scapular Glide | C5 · C8 · 阶段 2 |
| Out / 绕质心 / Trunk 脉冲 / Chest Press / 晚加速 | C4 · C7 · C9 · 阶段 3 |
| 拍面禁滚 / Top edge / 动量冻结 | C6 · C12 · 阶段 4 |
| 握拍轴 / 随挥诊断 / 回位圆锥 | C11 · C12 · 阶段 5 |
| 阴阳三段链（无视频 ID） | C4 · 标「无片，勿冒充已看」 |

---

## 6. 冲突与裁决表

| # | 冲突点 | 各方主张 | 裁决 |
|---|--------|----------|------|
| **X1** | Bottom-up vs Top-to-Bottom | 传统/TPA：腿→髋→肩→臂；Intuitive：Drop+Pull 自上而下，腿 Accommodate | **同构分层**：感觉入口可用 Sync Drill；能量/负荷仍承认 GRF/髋肩时序（Gordon/TPA）。不写「腿无用」。金标「右脚为轴+左侧拉」优先。 |
| **X2** | FTT “Pull Forward” vs TPA “Throw not Pull” | FTT 早期 Pull；TPA/用户后期 throw/release；Pulling Trap 已记 | **机制信 TPA Throw**；FTT Pull 不进当前球场 cue。与金标冲突则金标胜。 |
| **X3** | Stay Down vs Sit-Stand | 传统 Stay Down；OTI: Do NOT Stay Down；Bourne: Don’t bend knees, raise heels | **同构**：膝多为被动；垂直分量要起来。跟 OTI Sit-Stand + 用户 Sit not Push。FTT「轴不漂」保留。 |
| **X4** | 全程 Double Bend vs Elbow Extension | 弯肘造型教法 vs Gordon/Macci: drop 前伸肘再甩 | **硬数据信 Gordon**。教学「展」可进 reasoning；是否活跃 cue 需用户自验。用户 8/23：肘角=风格变量。 |
| **X5** | Pat-the-Dog / 极端拍头下沉 | 主动拍狗 vs TPA Gradual Squaring；FTT 有支撑进槽 | **TPA 祛魅 + FTT 有支撑进槽**；禁为求 lag 主动松散甩拍。 |
| **X6** | 肩转 65° vs 90°+ | Bourne 中段约 65°；Gordon/FTT 常谈 90°+ coil | **时间点不同，非矛盾**：中段 65°、完成 coil 再看 90°+。 |
| **X7** | 「抱球」vs 「跟后围栏握手」 | Bourne Hold onto the ball；Intuitive 等后向意象 | **意象同构**；优先用户已试且贴合右脚为轴的「抱球」。 |
| **X8** | RTP Landmark vs Intuitive 反教条 | RTP 肘 90°/标志位；Intuitive 过多指令→mechanical | **阶段裁决**：建结构用标志；实战减指令 + Intuition Paradox。 |
| **X9** | Weight Transfer「重心前移」字面 | 业余往左脚推；Feel 会员 Myth 三件套（待 VCA，**可能不可及**）；OTI/用户后腿锚定 | **临时裁决**：Sit not Push + OTI Anchor back leg；不采「往左顶髋」。待 Feel 会员 VCA 复核。 |
| **X10** | 直臂 Type 3 vs 弯臂 Type 2 | Gordon Type 3；业余弯臂常见 | **目标可取 Type 3**；容错允许 Type 2（FTT 光谱）。用户主线 HSA+ISR 对齐 Type 3 方向，不因外形未直否定链。 |

**冲突点数：10（X1–X10）**

---

## 7. 生物力学词汇对照（书本层精简）

| 术语 | 书本/标准定义 | 可见近似 | 教练口语 | 主要来源 |
|------|---------------|----------|----------|----------|
| **Kinetic chain** | 多环节按序传能；常腿→髋→躯干→肩→臂→拍 | 腿先加载，肩髋有时序差，末端后加速 | 「用腿带身子」 | TS Ch2；Kovacs Ch1 |
| **Leg drive / GRF** | 下肢蹬伸产 GRF；业余~1.7×BW vs 高水平~2.1×BW | 后腿蹬伸、重心上升 | 「蹬地」「Sit-Stand」 | TS Ch2 p.44 |
| **SSC** | 预拉伸后缩短 +约 10–20% 速度；pause ~1s 失~50% | 顶点无长挂住 | 「弹簧」 | TS Ch2 p.34 |
| **Separation / X-factor** | 肩相对髋多转约 20°–30° | 肩比髋更「关上」 | 「拧身」「线圈」 | TS Ch2 p.35 |
| **Unit turn** | 肩-髋-臂共转（教学概念） | 全身同向转，非先甩拍 | 「整体转身」 | 教练层 ↔ TS trunk |
| **ESR** | 肱骨外旋；拉伸内旋肌储能 | 拍头偏天（视握法） | 「外旋锁」 | TS Ch2 p.34；**刹车优先=项目/用户理论** |
| **ISR** | 肱骨内旋；正手文献~35–40%；发球~40% 勿直接搬 | 接触附近/后长轴内旋 | 「撕」「雨刷」* | Reid 2013；TS |
| **HSA / horizontal flexion** | 上臂水平向中线闭合；综述~25% 拍速 | 胸-肱角关闭、跨体 | 「press」「横拉」 | Reid 2013；**HSA=项目统一名** |
| **Pronation** | 前臂旋前；多为近端结果 | 拍面关闭外观 | 「转拍面」* | Reid 2013 |
| **Scapular retraction/protraction** | 肩胛后缩/前伸 | 后背夹紧 ↔ 胸廓包裹滑动 | 「夹肩胛」「槽」 | TS Ch7；Kovacs |
| **Shoulder as funnel（Kibler）** | 肩为腿/躯干能量漏斗 | 上游动则肩接住 | 「别断链」 | TS Ch7 |
| **Eccentric deceleration** | 随挥后侧离心刹车 | 可控停下、无甩伤感 | 「随挥自然停」 | Kovacs；TS Ch7 |
| **Throwing analogy** | 正手≈改良体侧投掷 | 像扔非推 | 「像投球」 | throwing KB（对照） |

\* 标星口语常把**结果**教成**主动动作**——见永久禁令。

### 7.1 书本可验证 vs 口令边界（摘要）

**可验证**：75/25 拍速贡献；SSC pause 衰减；separation 20–30°；GRF 差异；正手 ISR% 需引正手研究；远端旋前多为结果；击球点过后+过早旋前+弱肩→网球肘（TS Ch7）。

**不可升格定律**：「主动雨刷出上旋」；「ISR=40% 所以要主动内旋」；「手臂完全没用」；「随挥必须主动收到左肩」；把项目 ESR/HSA/Off-Arm **冒充** Elliott 原文章名。

---

## 8. 永久禁令与训练默认场景

### 8.1 永久禁令（合并 FTT Hall of Shame + 用户协议）

| # | 禁令 | 理由要点 | 来源层 |
|---|------|----------|--------|
| 1 | **主动推肘 / 送肘 / 肘前推作 cue** | 肘前是 HSA 结果；推肘→小肌群捅球 | `USER` 5/6；FTT chest |
| 2 | **胸部发力 / 右胸主动推当原理** | 胸是传力接口；力源在髋/躯干 | `USER` 禁令 #7 |
| 3 | **持拍臂独立后拉当「引拍」** | 破坏 Unit Turn | FTT 书 |
| 4 | **准备期主动造腕 lag / 拍底对网** | 预紧阻碍传递 | FTT Hall of Shame |
| 5 | **主动翻拍 / 主动旋前 / 主动扣腕当发动机** | 摧毁容错；伤病风险叙事 | FTT；Reid |
| 6 | **主动降拍 / 主动提前 Supination / 主动锁直肘** | 随结构前行自然出现 | `USER` 8/14+ |
| 7 | **主动夹死肩胛、用背部提拍** | 动态支架须能前伸释放 | `USER` 8/18 |
| 8 | **「先转髋」作执行 cue** | 感知张力；骨盆前转是结果 | `USER` 7/18 |
| 9 | **高位跳过垂直直接水平加速** | 合速度入地 + 腕捞 | FTT `dZadhzVEsds` |
| 10 | **把手臂拉近身体降拍 / 用腕调高度** | 破坏几何 | FTT 书 |
| 11 | **把随挥造型反推为触球原因** | 随挥在球离弦后 | FTT 书 |
| 12 | **刻意控制髋先于肩的毫秒时序** | ~25ms 不可意识控制 | FTT 博客摘录 |
| 13 | **堆 cue（>3）/ 假设实战对抗** | Intuition Paradox；Reason Plateau | `USER` |
| 14 | **呼吸作发力源 / Valsalva / 呼气硬推** | 仅节奏锚；有头痛史停练就医 | `USER` 8/9 |

### 8.2 训练默认场景

| 项 | 默认 |
|----|------|
| 场景 | **发球机最低速 + mini tennis**（不假设实战） |
| 呼吸 | 准备鼻吸；前挥至触球嘴部短促呼气 |
| 感知顶层（用户） | 左侧上背+上胸紧撑 / 右手臂彻底放松（5/11） |
| 场上只验证（现行） | 动态支架 + 手拍接口在转换与触球区是否共同保持 |
| **明确不增加** | 主动降拍、主动旋后、主动伸肘、**主动推肘** |

### 8.3 球场可想短清单（FTT + 用户现行，≤ 数条）

1. 空间：球进区，触球**外且前**  
2. 想左手 / 结构向前行进（Throw，非推肘）  
3. Be patient（垂直到位）→ Out / through  
4. 吸气放松 → 呼气挥  
5. （校准用，非整场念）ESR 第一帧 / 右脚轴是否还在  

---

## 9. 现行主战场（截至 2026-08-24）与下次验证焦点

### 9.1 现行主战场

> **转换阶段保持率**：准备稍晚、等球或近端支撑提前松开时，旧的整臂塌低 + 小臂急升会回来。已从「能否做出」进入「动态肩胛支架 + 手拍接口能否在转换和触球区共同保持」。

### 9.2 状态摘要

| 端 | 状态 |
|----|------|
| 引拍（右臂独立后拉） | ✅ 已解决 |
| 降拍（有支撑落槽） | 🔄 动态化中 |
| 前挥（重叠 / 行进） | 🔄 可重复未自动 |
| 触球刷/扶 | ⬇️ 下游不单练 |
| Off-Arm / 拍柄 seating | 🔄 与前挥同入口；8/24 待球场 |

### 9.3 下次验证焦点（只验这些）

1. **动态肩胛支架**是否在转换窗保持（不塌低、不冻结夹背）。  
2. **V 区无空隙座入**（8/24 室内金标）能否在实击中保持：行进中角度不变、握压被动增加。  
3. 失败时查**保持率**，不从零重建引拍圣经堆。

### 9.4 阶段位置

处于 **阶段 7 · 几何根因 + External Focus 重建（5/13–now）** 的执行补充段。5/13 前大量 Internal Focus「圣经」→ **机制解释可用，场上不堆 cue**。

---

## 10. 证据缺口与建议重看清单

### 10.1 证据边界（必须标清）

| 包 | 状态 | 用法 |
|----|------|------|
| FTT | 大量 `VCA`/`SYN` + 书摘；部分核心片仅 synthesis 无本地 `video_analyses` md | 主张可用；表中「建议重看」需画面复核 |
| TPA | 深 VCA + kinetic synthesis | 投掷/lag/随挥主场 |
| Feel Tennis 免费已分析 | 如 `oPc8OJkG-zw` 等可进共识 | 已挂 C1/C15 |
| **Feel Tennis 会员** | 63 待订；registry 多为 metadata | **可能不可及**；只进缺口，不当已看主张 |
| One Minute Tennis 频道 | 935× metadata；runtime_eligible: false | 书主张标 `BOOK`；频道勿当处方 |
| Bourne 书 | 已 `BOOK` 分析 | 可引用；≠「视频已证明」 |
| 阴阳三段链 | 有文无视频 ID | 不可冒充已看视频 |
| Kovacs PDF | 扫描件抽字失败 | 页码经 KB 二次摘录 |

### 10.2 建议重看 / 补 VCA（合并优先）

**A. FTT 机制中心（最多 5，来自 FTT 草稿）**

1. `0m3BMfDDShI` — Build This Foundation（Place/Hold/Pull、背连接）  
2. `dZadhzVEsds` — Hit Hard Without Missing Long（垂直→水平）  
3. `hNVbbPEob3g` — Chest Engagement（Attached/Press；禁推肘）  
4. `9NirSL59an0` — Paradoxical Topspin（Press Slot、top edge）  
5. `x_WIg59Wr9Y` — Start Swing at Contact（晚加速 / 「3 帧」主张核验）  

**B. Cross-Coach Top 补看（含 Feel 会员 · 可能不可及）**

| # | 视频 | 教练 | 服务节点 | 可达性 |
|---|------|------|----------|--------|
| 1 | Busting the Weight Transfer Myth (`kGzHlnp4fEE`) | Feel | X9、C3 | **会员 · 可能不可及** |
| 2 | Feel the Weight, Unleash the Power (`ntFG9io5BYw`) | Feel | C9、C14 | **会员 · 可能不可及** |
| 3 | Slap & Push Explained (`n-ns8luJtAQ`) | Feel | C15、C5 | **会员 · 可能不可及** |
| 4 | Connected Shoulders + Rotation (`VLoTdbA_l5o`) | Feel | C2、C8 | **会员 · 可能不可及** |
| 5 | Hip as center/axis takeback (`s7n6kSkLeCk`) | Intuitive | C2 | 公开 |
| 6 | Top-to-Bottom kinetic chain (`LbFEmpfYMhA`) | Intuitive | X1、C4 | 公开 |
| 7 | Do NOT Stay Down / Sit-Stand (`hGw1-pzsCK8`) | OTI | X3、C3 | 公开 |
| 8 | Slap Into the Slot (`0l5QKGubb_Q`) | TPA | C5、X5 | 公开 |
| 9 | Racket flip + back stability (`wzLMv4ASg9s`) | FTT | C5、C8 | 公开 |
| 10 | Macci + Gordon Elbow Extension / Type 3 (`n4Cq4VaiFTw`) | Gordon | X4、C6、C9 | 公开 |

**C. Feel 已有本地分析、合并时需统一证据标签**（路径 `docs/research/feel_tennis/feeltennis_video_analyses/`）：  
`0Mf8SFX_LuI`, `2D7UlPQHce4`, `5LOKkHpFpFU`, `5rTG-uxWmsg`, `8b96lTo4zKA`, `9KRYA9ZlYmM`, `Auem1-8t3rE`, `EWrlCzQV4Hk`, `MO01CaN6lFc`, `cwwKLLw9mII`, `guUg4hVI1AE`, `kBDKMAHrzNo`, `mnv0O2Z3MWc`, `vcWAEcF6klU`。

**D. 数值勿升格为测量真理**：40ms / 3 帧 / 50% 力量税等 → `external_validation_needed`。

---

## 11. 文件索引（草稿与源路径）

### 11.1 本合成目录

| 文件 | 角色 |
|------|------|
| `docs/research/_synthesis/UNIFIED_FOREHAND_KNOWLEDGE_FRAMEWORK.md` | **本文 · 主框架** |
| `docs/research/_synthesis/UNIFIED_FRAMEWORK_README.md` | 用户导读 |
| `docs/research/_synthesis/FRAMEWORK_DRAFT_FTT.md` | FTT 层草稿（24 机制节点） |
| `docs/research/_synthesis/FRAMEWORK_DRAFT_CROSS_COACH.md` | 跨教练草稿（C15 / X10） |
| `docs/research/_synthesis/FRAMEWORK_DRAFT_BOOKS.md` | 书本/生物力学层草稿 |
| `docs/research/_synthesis/FRAMEWORK_DRAFT_USER.md` | 用户层草稿（金标 23；截点 2026-08-24） |

### 11.2 权威与金标

| 路径 | 用途 |
|------|------|
| `docs/hermes_context_export/01_CORE_THEORY_SUMMARY.md` | 核心理论一页纸 |
| `docs/hermes_context_export/04_PERMANENT_RULES.md` | 永久规则 / 权威序 |
| `docs/hermes_context_export/05_USER_SELF_VERIFIED.md` | 用户自验金标 |
| `docs/hermes_context_export/06_VIDEO_INDEX.md` | 视频入口 |
| `docs/research/FOREHAND_COMPLETE_TAXONOMY.md` | L1–L10(+L11) 对照维度 |

### 11.3 FTT

| 路径 | 用途 |
|------|------|
| `…/fault_tolerant_tennis/foundation/first_principles_knowledge_tree.md` | P0–P7 |
| `docs/research/ftt/12_ftt_videos_synthesis.md` | 30 片索引 |
| `docs/research/ftt/ftt_backswing_complete.md` / `ftt_forward_swing_complete.md` | 准备 / 前挥 |
| `docs/research/ftt/ftt_vertical_horizontal_decomposition.md` | 二维分解 |
| `docs/research/ftt/01_ftt_book.md` | 书摘与禁令 |
| `docs/research/ftt/18_ftt_build_foundation.md` / `20–23_*.md` | 专题 |
| `docs/research/ftt/video_analyses/` | 单片分析（质量不一） |

### 11.4 书本

| 路径 | 用途 |
|------|------|
| `books/academic/tennis_science_2015.pdf` | TS 原文（可抽字） |
| `docs/research/tennis_science/book/ch2_technique.md` 等 | 章节 KB |
| `docs/research/tennis_science/tennis_science_paper_reid_elliott_2013.md` | 正手贡献综述 |
| `docs/research/tennis_science/24–28_biomechanics_*.md` | Kovacs 经 KB |
| `books/STRUCTURE.md` | 书权威优先级 |

### 11.5 跨教练与项目理论

| 路径 | 用途 |
|------|------|
| `docs/research/tpa/15_tpa_synthesis.md` | TPA |
| `docs/research/feel_tennis/` | Feel 分析与缺口 |
| `docs/research/coaches/*` | Intuitive / OTI / Gordon / JUL 等 SUMMARY |
| `docs/research/hsa_master_index.md` · `esr_root_cause_master.md` · `non_hitting_arm_master.md` | 项目理论（非书原话） |
| `COACH_VIDEO_HALL_OF_FAME.md` | 高优先级推理参考（≠ 自动球场 cue） |

---

## 12. 交付元数据

| 字段 | 值 |
|------|-----|
| 主文档 | `/workspace/tennis_analyzer/docs/research/_synthesis/UNIFIED_FOREHAND_KNOWLEDGE_FRAMEWORK.md` |
| 导读 | `/workspace/tennis_analyzer/docs/research/_synthesis/UNIFIED_FRAMEWORK_README.md` |
| 共识节点 | **15**（C1–C15） |
| 冲突点 | **10**（X1–X10） |
| 用户金标（草稿计） | **23**（球场/录像向 22 + 8/24 待验 1） |
| 现行主战场 | 转换阶段保持率：动态肩胛支架 + 手拍接口 |
| **重看状态（2026-10-12）** | 13 公开片 `rewatch_complete`（subs+frames）；见 §13 |
| **Feel 缺口** | 4 会员片仍 **可能不可及**（`kGzHlnp4fEE` / `ntFG9io5BYw` / `n-ns8luJtAQ` / `VLoTdbA_l5o`）；不进共识主张 |
| 重看合成 | `/workspace/tennis_rewatch/notes/REWATCH_SYNTHESIS.md` |
| 补丁摘要 | `docs/research/_synthesis/REWATCH_PATCH_NOTES_2026-10-12.md` |
| 未做 | 编造时间戳；把 META/会员未看片当主张；主动推肘口令；整库重爬视频；Micu |


---

## 13. 2026-10-12 重看补丁（subs+frames）

> **性质**：条目级补丁，**非**整篇重写。权威序不变（用户金标 > Tennis Science > FTT > 其他）。  
> **合成源**：`/workspace/tennis_rewatch/notes/REWATCH_SYNTHESIS.md` §4（17 条）。  
> **单片证据**：`/workspace/tennis_rewatch/notes/{id}_rewatch.md`。

### 13.1 方法说明

| 项 | 说明 |
|----|------|
| 管线 | 本机 **yt-dlp** 下载 + 字幕抽取 + **ffmpeg** 抽帧；对照既有 synthesis / 阶段骨架做 VCA 笔记 |
| 不可用 | Cursor `watchVideo` 区域不可用；**未使用 Micu** |
| 完成 | **13** 公开片状态 `rewatch_complete`（含 2 片无字幕、仅视觉：`wzLMv4ASg9s`、`x_WIg59Wr9Y` → 口播细节勿升格硬主张） |
| Feel | **4** 会员片仍不可及（membership required）；**不进共识主张列**，只保留缺口 |
| 透镜 | 手主导 → 交给身体；主战场=转换期动态肩胛支架；**严禁主动推肘口令** |

### 13.2 十七条补丁 → C/X / 阶段（逐条）

| # | 落点 | 补丁内容（短） | 证据路径 |
|---|------|----------------|----------|
| 1 | **C8** | **操作定义**：肘空间 + 上背托住（`0m3B`/`9Nir`）= 动态肩胛支架的可观察代理。转换窗查：肘是否塌进、上背是否提前放松。 | [`0m3BMfDDShI_rewatch.md`](/workspace/tennis_rewatch/notes/0m3BMfDDShI_rewatch.md) · [`9NirSL59an0_rewatch.md`](/workspace/tennis_rewatch/notes/9NirSL59an0_rewatch.md) |
| 2 | **C9** | Chest lag 禁区 + Attached→**短程末端 Press**（`hNVbb`）。**明示：press ≠ 推肘 / 胸部主动推**（上游稳定下的前送加载，发生在躯干减速末端）。 | [`hNVbbPEob3g_rewatch.md`](/workspace/tennis_rewatch/notes/hNVbbPEob3g_rewatch.md) · [`9NirSL59an0_rewatch.md`](/workspace/tennis_rewatch/notes/9NirSL59an0_rewatch.md) · [`hqganBxzQNM_rewatch.md`](/workspace/tennis_rewatch/notes/hqganBxzQNM_rewatch.md) |
| 3 | **C5** | 并列三条进槽路径，**均禁**主动松散造 lag / 硬翻腕：（a）有支撑进槽（FTT）；（b）条件化 slap 入槽（TPA，因人）；（c）被动 flip（Gordon）。 | [`0l5QKGubb_Q_rewatch.md`](/workspace/tennis_rewatch/notes/0l5QKGubb_Q_rewatch.md) · [`n4Cq4VaiFTw_rewatch.md`](/workspace/tennis_rewatch/notes/n4Cq4VaiFTw_rewatch.md) · [`wzLMv4ASg9s_rewatch.md`](/workspace/tennis_rewatch/notes/wzLMv4ASg9s_rewatch.md) · [`9NirSL59an0_rewatch.md`](/workspace/tennis_rewatch/notes/9NirSL59an0_rewatch.md) |
| 4 | **C5 / C12** | 击球段 slap/bang **vs** 随挥滚动（`9Nir`）；随挥**不作**发力发动机；接触停转可作传能完成诊断（`hGw1`）。 | [`9NirSL59an0_rewatch.md`](/workspace/tennis_rewatch/notes/9NirSL59an0_rewatch.md) · [`hGw1-pzsCK8_rewatch.md`](/workspace/tennis_rewatch/notes/hGw1-pzsCK8_rewatch.md) · [`x_WIg59Wr9Y_rewatch.md`](/workspace/tennis_rewatch/notes/x_WIg59Wr9Y_rewatch.md) |
| 5 | **阶段 2–3** | Into-out × 垂直分量须在正确 **slot 高度窗口**同步 + **晚加速**（`dZad`；`x_WI` 视觉互证）。错槽则弹性脑不信任。 | [`dZadhzVEsds_rewatch.md`](/workspace/tennis_rewatch/notes/dZadhzVEsds_rewatch.md) · [`x_WIg59Wr9Y_rewatch.md`](/workspace/tennis_rewatch/notes/x_WIg59Wr9Y_rewatch.md) |
| 6 | **阶段 2** | 低高杠杆排序：**slant 优先于主动 drop**（`0m3B`）；胸高球勿滥 drop。 | [`0m3BMfDDShI_rewatch.md`](/workspace/tennis_rewatch/notes/0m3BMfDDShI_rewatch.md) |
| 7 | **X1** | 证据升级：`LbFE`（top-down）+ `hGw1`/`n4Cq`（sit-stand / ground-up）本批均 `rewatch_complete` → 冲突表可标「**已双轨实看**」。同构分层维持；能量叙事不写「腿无用」。 | [`LbFEmpfYMhA_rewatch.md`](/workspace/tennis_rewatch/notes/LbFEmpfYMhA_rewatch.md) · [`hGw1-pzsCK8_rewatch.md`](/workspace/tennis_rewatch/notes/hGw1-pzsCK8_rewatch.md) · [`n4Cq4VaiFTw_rewatch.md`](/workspace/tennis_rewatch/notes/n4Cq4VaiFTw_rewatch.md) |
| 8 | **X3** | 证据升级：`hGw1` 实看完成——Stay-down 迷思破除；维持 **Sit–Stand** + 用户 Sit not Push。 | [`hGw1-pzsCK8_rewatch.md`](/workspace/tennis_rewatch/notes/hGw1-pzsCK8_rewatch.md) |
| 9 | **X4** | 证据升级：`n4Cq` 实看完成。教学「展肘 / Elbow extension」可进 **reasoning**；活跃 cue 仍需用户自验；金标「肘角=风格变量」保留。**与禁推肘并列——Gordon extension ≠ 推肘口令**。 | [`n4Cq4VaiFTw_rewatch.md`](/workspace/tennis_rewatch/notes/n4Cq4VaiFTw_rewatch.md) |
| 10 | **X5** | 证据升级：`0l5Q` + `TgC` + `n4Cq` tap-dog → 统一「**后置 / 自然，勿主动卡住**」；FTT 有支撑进槽并行。 | [`0l5QKGubb_Q_rewatch.md`](/workspace/tennis_rewatch/notes/0l5QKGubb_Q_rewatch.md) · [`TgCZr6aRa1Q_rewatch.md`](/workspace/tennis_rewatch/notes/TgCZr6aRa1Q_rewatch.md) · [`n4Cq4VaiFTw_rewatch.md`](/workspace/tennis_rewatch/notes/n4Cq4VaiFTw_rewatch.md) |
| 11 | **C14** | 最小可比赛调整原则（`TgC`）。感觉探针列表 +=：负重水平半圆（`hqgan`）、两指握、侧抛、first-move 腕角（`s7n6`）。 | [`TgCZr6aRa1Q_rewatch.md`](/workspace/tennis_rewatch/notes/TgCZr6aRa1Q_rewatch.md) · [`hqganBxzQNM_rewatch.md`](/workspace/tennis_rewatch/notes/hqganBxzQNM_rewatch.md) · [`s7n6kSkLeCk_rewatch.md`](/workspace/tennis_rewatch/notes/s7n6kSkLeCk_rewatch.md) · [`dZadhzVEsds_rewatch.md`](/workspace/tennis_rewatch/notes/dZadhzVEsds_rewatch.md) |
| 12 | **C15** | Slap **双面**：TPA 解锁工具（卡死后僵手、条件化）vs Intuitive 有害甩腕（`s7n6`）。适用边界写入 Hall of Caution——镜前/感觉 ≠ 球场结构。 | [`0l5QKGubb_Q_rewatch.md`](/workspace/tennis_rewatch/notes/0l5QKGubb_Q_rewatch.md) · [`s7n6kSkLeCk_rewatch.md`](/workspace/tennis_rewatch/notes/s7n6kSkLeCk_rewatch.md) |
| 13 | **C11 / 接口** | First-move 腕角（肘上/手下，`s7n6`）+ 点腕手-拍静帧（`wzLM` 视觉）挂**转换期**检查；先稳住手接口，再谈交给身体。 | [`s7n6kSkLeCk_rewatch.md`](/workspace/tennis_rewatch/notes/s7n6kSkLeCk_rewatch.md) · [`wzLMv4ASg9s_rewatch.md`](/workspace/tennis_rewatch/notes/wzLMv4ASg9s_rewatch.md) |
| 14 | **主战场** | 句式维持：「转换期动态肩胛支架 + 手拍接口」。**观察清单五条**见下 §13.3。 | 合成 §4 #14；多片互证 |
| 15 | **永久禁令复核** | 本批全文**未**引入任何主动推肘口令。注释：FTT **press** / TPA **肘领先**（slap 入槽解锁）/ Gordon **extension** = **均非推肘**。 | 全批 notes；对照 §8.1 #1 |
| 16 | **缺口保留** | Feel 四会员片不变（见 §13.4）。两无字幕片（`wzLM`/`x_WI`）待有字幕或二次精读再升证据级。C10 ESR 本批无专片，仍靠既有 Bourne/JUL/用户金标。 | §10.1–10.2；合成 §2.3 |
| 17 | **文件挂载** | 各 `notes/{id}_rewatch.md` 作为对应 C/X 的 `VCA` 证据路径写入本表索引（本条即索引承诺；**勿**整篇重写正文）。 | `/workspace/tennis_rewatch/notes/*_rewatch.md` |

### 13.3 主战场观察清单（五条）

在「动态肩胛支架 + 手拍接口」保持率检查时，按序可观察：

1. **肘空间**——转换窗肘是否塌进肋侧 / 贴死身体  
2. **上背托住**——上背是否提前放松（支架先垮）  
3. **胸附着非胸 lag**——Attached→短程末端 press；禁胸推 / 推肘外形当任务  
4. **高度窗口后再 into-out**——垂直分量 / slot 高度到位后，再叠水平 into-out（勿高位直接横砍）  
5. **接触区晚加速**——准备 ≠ 加速；proximal-to-distal 末端再加速  

### 13.4 Feel 会员四片（仍不可及 · 不进主张）

| ID | 标题 | 关联缺口 |
|----|------|----------|
| `kGzHlnp4fEE` | Busting Weight Transfer Myth | X9、C3 |
| `ntFG9io5BYw` | Feel the Weight | C9、C14 |
| `n-ns8luJtAQ` | Slap & Push | C15、C5 |
| `VLoTdbA_l5o` | Connected Shoulders | C2、C8 |

状态：meta 标注 membership required；本批未下载 / 未 VCA。临时裁决（如 X9 Sit not Push）不变。

### 13.5 十三片证据索引（`rewatch_complete`）

| ID | 教练 | 主要挂到 | 笔记 |
|----|------|----------|------|
| `0m3BMfDDShI` | FTT | C8、阶段 2（slant） | `/workspace/tennis_rewatch/notes/0m3BMfDDShI_rewatch.md` |
| `0l5QKGubb_Q` | TPA | C5、C15、X5 | `/workspace/tennis_rewatch/notes/0l5QKGubb_Q_rewatch.md` |
| `hNVbbPEob3g` | FTT | C9（禁胸 lag / press≠推肘） | `/workspace/tennis_rewatch/notes/hNVbbPEob3g_rewatch.md` |
| `hqganBxzQNM` | FTT | C9、C14（负重探针） | `/workspace/tennis_rewatch/notes/hqganBxzQNM_rewatch.md` |
| `TgCZr6aRa1Q` | TPA | C14、X5 | `/workspace/tennis_rewatch/notes/TgCZr6aRa1Q_rewatch.md` |
| `hGw1-pzsCK8` | OTI | X3、C3、C12 | `/workspace/tennis_rewatch/notes/hGw1-pzsCK8_rewatch.md` |
| `LbFEmpfYMhA` | Intuitive | X1、C1、C4 | `/workspace/tennis_rewatch/notes/LbFEmpfYMhA_rewatch.md` |
| `s7n6kSkLeCk` | Intuitive | C11、C15 | `/workspace/tennis_rewatch/notes/s7n6kSkLeCk_rewatch.md` |
| `9NirSL59an0` | FTT | C8、C9、C5/C12 | `/workspace/tennis_rewatch/notes/9NirSL59an0_rewatch.md` |
| `x_WIg59Wr9Y` | FTT | 阶段 2–3（晚加速；**无字幕**） | `/workspace/tennis_rewatch/notes/x_WIg59Wr9Y_rewatch.md` |
| `dZadhzVEsds` | FTT | 阶段 2–3（高度窗口） | `/workspace/tennis_rewatch/notes/dZadhzVEsds_rewatch.md` |
| `n4Cq4VaiFTw` | Macci/Gordon | X4、X5、C5（extension≠推肘） | `/workspace/tennis_rewatch/notes/n4Cq4VaiFTw_rewatch.md` |
| `wzLMv4ASg9s` | FTT | C5、C8、C11（**无字幕**） | `/workspace/tennis_rewatch/notes/wzLMv4ASg9s_rewatch.md` |

完整合成与各片 2–4 句核心见：`/workspace/tennis_rewatch/notes/REWATCH_SYNTHESIS.md`。

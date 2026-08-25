# One Minute Tennis：Drill、感觉线索与机制解释知识树（证据分层版）

**源频道：** [One Minute Tennis](https://www.youtube.com/@oneminutetennis)
**版本：** 0.1，2026-08-15
**索引范围：** 935 条公开视频（794 条常规视频、141 条 Shorts）。

> **本文件的证据边界。** One Minute Tennis 具有高密度、碎片化的教学形式。当前知识树将频道的官方播放列表归类和视频标题转化为**可检索的训练原子候选**，而非将标题改写为视频中的教学主张、练习步骤或生物力学事实。具体 drill 流程、感觉线索、机制解释、适用条件与时间码必须在获得视频内容证据后追加。

## 1. 原子化原则

频道视频的正确建模不应是“视频 A = 一条知识”，而应是：

```text
来源视频 / 播放列表
  → 训练原子候选（drill / cue / mechanism / error_correction / tactical_decision）
  → 条件（技术族、球况、水平、限制）
  → 目标变量（动作、球路、位置、决策或感知）
  → 可观察结果 / 练习成功标准
  → 证据（标题、官方播放列表、字幕、视频画面、人工审阅）
```

| 原子类型 | 在训练系统中的作用 | 当前可确认的证据 | 内容补齐后必须新增 |
|---|---|---|---|
| `drill` | 定义一个可执行的练习任务 | 官方“Drills and Exercises”播放列表或标题中的 drill/exercise/practice | 起始姿势、步骤、次数、成功标准、变式与禁忌 |
| `cue` | 提供简短动作/感觉提示 | 标题中的 feel/hack/tip/key/simple 等词 | 原话、目标阶段、不要误解的边界 |
| `mechanism` | 连接人体、球拍、球路或决策因果 | 标题中的 why/explained/kinetic/lag/pronation 等词 | 明确主张、证据类型、替代解释和条件 |
| `error_correction` | 将常见错误映射到可测试的修正 | 标题中的 fix/mistake/myth/stop/wrong 等词 | 错误定义、可见信号、优先纠正动作与复测 |
| `tactical_decision` | 处理球况、风险和选择 | 标题中的 strategy/when/doubles/singles/opponent 等词 | 球况输入、候选选择、风险与得分目标 |

## 2. 频道的确定性课程骨架

频道的官方播放列表提供了最强的主题归类证据：**Forehand、Service、Backhand、Volley、Movement、Drills and Exercises、Video Analysis 与 shorts**。其中有 823 条视频属于至少一个官方播放列表；一条视频可能出现在多个列表，因此这是课程关系而不是不重叠分类。

```text
A. 技术动作与球拍—球接口
├─ A1 Forehand（官方播放列表：504条成员映射）
│  ├─ 接触、拍头速度、滞后、挥拍路径、旋转与球路
│  ├─ 站位、下肢加载、恢复与力量
│  └─ 感觉线索/“hack”与错误纠正候选
├─ A2 Service（188条成员映射）
│  ├─ 起始、抛球、站位、投掷路径与击球
│  ├─ 旋前、拍头速度、旋转、二发与恢复
│  └─ 节奏、压力下发球与 drill 候选
├─ A3 Backhand（131条成员映射）
│  ├─ 单反/双反、力量、接触与路径
│  └─ 切削、变线、接发与纠错候选
├─ A4 Volley（38条成员映射）
│  ├─ 挥拍路径、间距、截击与半截击
│  └─ 过顶、高球与上网衔接
└─ A5 球拍—球接口跨技术主题
   ├─ 握拍、手腕、拍头、接触、路径
   └─ 旋转、旋前、lag、power、touch
B. 移动、空间与恢复
├─ B1 Movement（46条成员映射）
│  ├─ split-step、第一步、站位、平衡和间距
│  ├─ 高深球、短球、进攻球与上网
│  └─ recovery 与连续击球位置组织
└─ B2 开放/半开放/关闭站位的条件化选择
C. Drill 与技能学习
├─ C1 Drills and Exercises（17个官方视频条目）
│  ├─ Athletic Ready Position
│  ├─ Attacking short balls / Drills for Attacking Tennis
│  ├─ Closed and open stance drills / Semi Open Stance Exercises
│  ├─ Coordination Drills / Footwork for deep high balls
│  ├─ HUNTING THE BALL / LOAD AND EXPLODE / POWER and TOUCH
│  ├─ Serve recovery / How to RECOVER Like the Pro's
│  └─ Grip change / Big points serve 等
├─ C2 视频标题驱动的练习候选（46条）
└─ C3 学习与感觉线索：feel、simple、hack、key、easy
D. 诊断与解释
├─ D1 Video Analysis（官方播放列表；需补齐成员的内容证据）
├─ D2 mechanism 候选：why、understand、explained、kinetic、lag、pronation
└─ D3 error correction 候选：fix、mistake、myth、stop、trouble、wrong
E. 比赛决策与心理表现
├─ E1 球路、对手、单打/双打、风险与得分策略
└─ E2 练习、信心、感觉、注意与学习迁移
```

## 3. “Drills and Exercises”官方播放列表：优先内容补齐队列

以下主题直接来自频道官方的 `Drills and Exercises` 播放列表，因此适合成为最先获得人工观看笔记或合规视频分析的 17 个练习候选。标题中重复的条目可能是不同视频或复用资源；系统应以 `video_id` 区分。

| 标题级 drill 候选 | 训练域 | 内容证据补齐时要提取的字段 |
|---|---|---|
| Athletic Ready Position | B1 准备姿势与启动 | 脚距、膝髋、拍位、split-step时机、常见错误 |
| Attacking short balls | B1/C/E 进攻短球 | 识别短球、进入步法、击球目标、恢复选择 |
| Drills for Attacking Tennis | C1 进攻训练 | drill设置、进攻触发、得分/失误标准、变式 |
| Closed and open stance drills | B2/A 技术—站位 | 球况、站位切换、重心、球拍路径与边界 |
| Coordination Drills | C1 协调训练 | 难度梯度、节奏、手眼/脚眼要求、可量化标准 |
| Footwork for deep high balls | B1 高深球移动 | 预判、后撤/侧移、间距、击球和恢复 |
| HUNTING THE BALL | B1 主动接近来球 | “hunt”在视频中的定义、起步时点、误用风险 |
| LOAD AND EXPLODE | A/B 力量时序 | 加载部位、爆发方向、球况与禁忌 |
| OPEN AND CLOSED STANCE DRILLS | B2 站位选择 | 站位—来球—目标映射、drill步骤和检验 |
| POWER and TOUCH DRILLS | A/C 力量与手感 | 两种输出的区分、控制变量、反馈方式 |
| Semi Open Stance Exercises | B2 半开放站位 | 支撑脚、旋转、接触与恢复条件 |
| Serve recovery | A2/B 发球后恢复 | 发球后的第一步、场地位置、下拍准备 |
| How to RECOVER Like the Pro's | B1 恢复 | 击球后脚步与决策、不同球型恢复 |
| How to change Grips Perfectly and Fast | A5 握拍转换 | 当前握拍、目标握拍、转换路径和可观察验证 |
| Hit your best serves on the BIG POINTS | A2/E 压力发球 | 压力情境、可执行例行程序、目标与安全边界 |

**重要说明：** 表中“训练域”和“待提取字段”是知识工程的采集模板；标题未提供 drill 的真实步骤，系统不得在尚无内容证据时自动生成步序或处方。

## 4. 训练原子的组合逻辑

真正有价值的短视频内容往往通过组合而发挥作用。例如，一个“forehand cue”不应孤立下发，而应在 `球况 → 技术目标 → 机制 → cue → drill → 复测` 这一链条中被调用。

| 组合链 | 系统问题 | 所需原子 | 当前状态 |
|---|---|---|---|
| 技术纠错链 | “我正手晚了/感觉拍头不出来” | 错误可见信号 → 机制候选 → 感觉线索 → 单变量 drill → 视频复测 | 以标题/播放列表建立候选队列，待内容证据 |
| 球况适配链 | “深高球或短球怎么处理” | 来球条件 → 移动/站位 → 击球路径 → 恢复 → 战术选择 | 官方 Movement/Drill名单提供强主题索引 |
| 发球表现链 | “发球怎么更快且下一拍有准备” | 站位/投掷/旋前/恢复 → 关键线索 → 压力下练习 | Service列表与drill标题提供候选 |
| 手感与控制链 | “如何兼顾力量和 touch” | 球拍—球接口 → 输出目标 → 触球反馈 → drill变量 | 官方 POWER and TOUCH 主题待内容补齐 |
| VLM复盘链 | “视频里哪个环节可先观察” | 事件切分 → 可见变量 → 假设节点 → 补拍 → 训练测试 | 仅能定义观察槽位，不能凭标题诊断 |

## 5. VLM观察槽位

| 技术域 | 必须切分的事件 | 直接可见变量 | 不能从单机位视频自动断言 |
|---|---|---|---|
| 发球 | 准备、抛球、奖杯位、拍头低点、接触、落地、恢复 | 脚/手事件时序、拍头路径代理、接触点、恢复第一步 | 内部关节力矩、精确握压、伤病因果 |
| 正手 | 单位转体、下降、接触、随挥、恢复 | 站位、球—躯干距离、拍头/手部关系、恢复方向 | 唯一“正确”lag或挥拍样式 |
| 反手 | 准备、支撑、接触、随挥 | 支撑脚、躯干朝向代理、拍头路径、击球区 | 肌肉激活和拍面三维角度（无标定） |
| 截击/过顶 | 启动、位置调整、拦截、接触、恢复 | 距网、拍头轨迹、步数、接触相对位置 | 反应意图、战术沟通 |
| 步法/drill | 信号出现、split-step、第一步、支撑、击球后恢复 | 事件时间、第一步方向、脚步数量、身体—球距离代理 | 未出现的替代选择或疲劳水平 |

## 6. 证据升级规则

| 状态 | 允许呈现 | 不允许呈现 | 升级所需证据 |
|---|---|---|---|
| `metadata_title_and_playlist_only` | 主题、标题、播放列表归属、候选原子类型 | 视频内主张、动作步骤、机制解释、时间码 | 字幕、人工笔记或可审计音画分析 |
| `video_explicit` | 明确口述/演示的主张与时间码 | 超越视频的科学外推 | 视频原文/演示复核 |
| `visual_inference` | 可见姿态、球拍、球路与时序观察 | 将视觉相关性写成因果 | 多机位或额外情境证据 |
| `human_reviewed` | 经教练审阅的训练单元与注释 | 掩盖原始证据和版本 | 审阅人、日期、修改记录 |

## References

[1]: https://www.youtube.com/@oneminutetennis "One Minute Tennis 频道主页"

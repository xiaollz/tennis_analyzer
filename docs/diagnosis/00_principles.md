# 00 FTT 诊断原则（优先于所有故障卡）

> 依据：《The Fault Tolerant Forehand》原书（John Kumpf, 2021）`books/ftt_fault_tolerant_forehand/original.pdf`，页码 = 印刷页码；笔记 `docs/research/ftt/01_ftt_book.md` 的行号只作定位。
> 标注见 `README.md` §2。本文件每条都写「原书怎么说 → 诊断时怎么用」。

---

## P1 容错：不稳定通常不是「小误差」造成的，而是挥拍扛不住小误差

- 原书：「Most forehand inconsistency is not proximally caused by small inaccuracies, but rather by a swing which lacks the fault tolerance to handle them.」【已核对 p.13】
- 原书：容错的挥拍允许「a little early, or a little late… racket tilt off by four or five degrees… set up a little too close to the ball, or a little too far from it」球仍然进【已核对 p.14–15】。
- 笔记定位：`01_ftt_book.md:11–15`（内容与原书一致）。
- **诊断时怎么用**：一拍里偶尔偏近、偏晚不算故障；**同一种失误跨拍反复出现**（5 拍里 ≥4 拍），才说明挥拍本身缺容错，才进候选。

## P2 网球基本定理（不可让步的两条）

- 原书：「1. The racket makes a 90–135° angle with the forearm　2. The racket strikes the ball in front of the body (in a pinch, in front of the forearm)」，「Any shot which follows these two rules will work.」【已核对 p.24】
- 原书：调整拍子位置靠腿、躯干，必要时胸大肌；前臂和手腕的小肌肉只用来稳定【已核对 p.25】。
- 笔记定位：`01_ftt_book.md:27–38`。注意：笔记 `:40–44` 关于「前臂旋内/旋外肌群稳定拍面」的生物力学解释，在原书 p.24–25 **没有找到**【笔记归纳 / 未核对出处】，本目录不采用。
- **诊断时怎么用**：任何候选故障，先问它是否违反了这两条；违反的排在「风格」项前面。

## P3 治因不治症：先修「成因」，不是先修「最早的阶段」

- 原书：教练常常只看到一个可见的错误（脚站错、手腕不对、球下网）就去纠正那个错误；「these observed flaws are merely symptoms of an underlying systemic issue」；例子：触球时手腕直，原因是握拍不够放松，所以说「把手腕弯起来」没用，只会变成紧的弯手腕【已核对 p.17】。
- 原书：「We diagnose the cause and then help the student with the cause, rather than simply relaying the clues themselves back to the student」【已核对 p.18】。
- 笔记定位：笔记**没有收录**这一节（序言 p.16–19）。
- **与 PLAN.md「按阶段越靠前越优先」的关系（重要）**：
  - 原书的教学顺序是**从触球开始**：「all of the body movement that gets you to the strike is fundamentally subordinate to the strike itself」【已核对 p.21】；准备「solely a means to an end」【已核对 p.57】；触球部分是「all the information that's universally applicable」，其余是让挥拍更好的增强项【已核对 p.47】。
  - 所以「先修上游」在本目录的定义是：**先修因果链上的成因**，而不是机械地先修时间上更早的阶段。
  - 「准备 > 前挥 > 触球 > 随挥」只作为**两个候选证据同样强时的打破平局规则**【推断，来自 PLAN.md C3】，不能压过 P2。

## P4 结果优先：先问球飞成什么样，再问身体

- 原书：「Versatile striking requires answering three questions, in order: 1. What do I want the ball flight to look like? 2. What contact creates that ball flight? 3. How do I orient my body to produce that contact?」很多人从第 3 问开始【已核对 p.35】。
- 原书用结果来查错：上旋不够 → 可能是没有向外挥（out）【已核对 p.98】；低于 80% 成功率的击球「mostly useless」【已核对 p.125】；随挥本身不影响击球，「merely a diagnostic tool」【已核对 p.107】。
- 笔记定位：`01_ftt_book.md:426`（随挥是诊断工具）、`:548`（80%）。p.35 的三问笔记未收。
- **诊断时怎么用**：每次诊断的第一步是看结果锚点（口播：好 / 网 / 长 / 偏 / 挤，见 OBS-16）。没有结果数据的诊断，置信度上限为「低」。

## P5 主动 vs 被动：只纠正主动动作，不去「摆」被动位置

- 原书：看电视有两道信息屏障——看不出动作的精确时机；看不出哪些是有意做的、哪些是被动结果【已核对 p.64】。
- 原书：拍底指向网是「dynamic position; it should not be entered volitionally」【已核对 p.64】；手腕滞后只在前挥中被动出现，准备时强行做出来会妨碍传力【已核对 p.65–66】。
- 原书：「Racket rotation about the forearm through the hitting zone is automatic」；手腕滞后、拍子旋转、以及前挥启动后的整条拍路都是 dynamic positions【已核对 p.96–97】。
- 原书：随挥绕到身体另一侧是触球后的被动结果【已核对 p.108】。
- 笔记定位：`01_ftt_book.md:379–390` 的主动/被动表格【笔记归纳】，内容与上面各页一致。
- **诊断时怎么用**：如果一个观测事实描述的是被动位置（滞后、拍底方向、雨刷、随挥形状），它只能当**线索**往上追成因，不能直接变成「去做出这个位置」的处方。这也是本项目禁止主动「推肘 / 降拍 / 旋后 / 伸肘」口令的 FTT 依据之一。

## P6 必须纠正 vs 个人风格

- 原书：对错只在确定了目标之后才存在；髋扭转对「全身参与产生拍速」是 fundamental，没有个人偏好的余地；但在挡回 130 mph 发球时是 non-essential【已核对 p.53】。
- 原书：教练的两种失败——不必要地纠正个人风格；把真正的缺陷当成风格放过【已核对 p.55】。
- 原书给的「不是风格」的例子：做爆发式挥拍时转动不足；高速击球时手腕不放松【已核对 p.55】。
- 原书给的「风格 / 视情境」例子：手的位置、脚的位置、握拍「right in one context and wrong in another」【已核对 p.55】；握拍「mostly a red herring」【已核对 p.46】；站位「doesn't matter that much」（但封闭式要小心）【已核对 p.77–78】；引拍时肘伸不伸、什么时候伸「is up to you」，弯肘到底「a matter of personal preference」【已核对 p.66、p.69】；准备时手的高度「anywhere from chest to head height」【已核对 p.62】；肩转多少由各人决定【已核对 p.59、p.119】。
- 笔记定位：`01_ftt_book.md:711–726`。其中 `:720`「手肘弯曲程度是 discretionary」这个例子**不在原书 p.53–55**，出处应改为 p.66、p.69–70。
- **诊断时怎么用**（本项目的目标设定：发球机低速下的正常节奏上旋正手）：

| 先判为「必须纠正」 | 先判为「风格，不单独定罪」 |
|---|---|
| 违反基本定理（P2） | 触球肘角（弯或直）单独出现 |
| 触球不在身前、不离开身体（书 p.116）| 握拍类型 |
| 前挥时髋/胸转动不足（书 p.55）| 站位类型（封闭式除外，见 FLT-05） |
| 高速挥拍时手腕紧（书 p.55）| 准备时手的高度、肩转幅度 |
| 主动翻拍（书 p.105）| 脚步模式（四种都可，书 p.84–87）|

  「风格」项只有在**和必须纠正项一起出现、并能解释结果**时，才作为证据挂上去。

## P7 执行时不做肌肉级思考

- 原书：「During the stroke production itself, almost all muscle level analysis should be ignored.」技术分析是为了「observe and adjust… after the fact」【已核对 p.51–52】。
- 原书：动力链先后时序「thinking about this timing during stroke production is almost definitely useless」【已核对 p.90】；口令应借用扭、扔、跳这类现成动作，而不是关节角度【已核对 p.110】；同时加两件新事会难得多【已核对 p.112】。
- 笔记定位：笔记未收 p.51–52 这段。
- **诊断时怎么用**：诊断用的推理留在我们这边；给用户的只能是一个外部目标（与 CLAUDE.md Intuition-First 一致）。

## P8 准备越简单越好；准备阶段的错误多半是「多做了」

- 原书：「In general, the simpler the preparation, the better.」准备只有两个目标：转离目标、让手处在能绕身体自由甩出的位置【已核对 p.57–58】。
- 原书：「Nearly all of the forehand-killing backswing mistakes aren't caused by players failing to take preparation steps… they're almost always caused by players doing extra things」【已核对 p.80】。
- 笔记定位：`01_ftt_book.md:89–99`、`:246`。
- **诊断时怎么用**：准备阶段的候选，优先找「多出来的动作」（手臂往后拉、主动摆滞后、拍底指网），而不是「缺了哪个步骤」。

## P9 风格过滤和结果锚点之后，才进诊断树

诊断步骤（与 PLAN.md C3 第 3 步一致）：
1. 结果锚点（P4，OBS-16）。
2. 剔除只出现 1–2 拍的偶发现象（P1）。
3. 剔除单独出现的风格项（P6）。
4. 剩下的事实查 `index/obs_to_fault.md` → 候选故障 → 沿成因链往上追（P3）。
5. 每个候选写清「什么证据会推翻它」。

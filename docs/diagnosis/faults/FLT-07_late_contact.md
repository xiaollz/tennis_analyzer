# FLT-07 击球晚（Late contact）

- 阶段：触球｜层级：结果层（上游是启动时机或视觉追踪）
- 必须纠正 or 风格：**必须**——违反基本定理第 2 条「in front of the body」（书 p.24【已核对】）；「If… you start hitting the ball late, or start catching it too close to you, then any power gained… will be offset」（p.116【已核对】）。
- 出处强度：故障本身**书有直接出处**；上游成因里只有「启动信号是视觉线索」和「看球到拍线」有书证，「晚」与它们之间的因果边是【推断】。

## 1. 症状（用户可报告）
- 下网偏多（→ 下网）；球偏向身体同侧/直线方向（→ 方向偏）【推断，书未写方向规律】
- 打框、误击（→ 打框）——书把它和「没看球到触球」联系，不是和「晚」联系（p.76）
- 「仓促」「来不及」（→ 节奏断/启动晚）

## 2. 可观测证据
| OBS | 机位 | 帧 | 测量方式 | 可靠度 |
|---|---|---|---|---|
| OBS-03 触球点相对前髋 | S | 触球帧 | Pose + 抽帧 | 不稳定（3/4，R 机位）|
| OBS-04 顶点停顿 | 任意，≥60 fps | 引拍顶点 → 启动 | 模型（有无）+ Pose（时长）| 有无 2/2、10/10 拍；时长未测准；**单独不支持本卡**（书 p.75）|
| OBS-16 口播结果（网 / 框 的比例）| 音轨 | — | 用户口播 | 未测 |

## 3. 成因链
- 上游候选（鉴别诊断）：
  1. 前挥启动晚：书说启动信号是视觉线索——「once the ball gets to a certain location, you initiate your swing」（p.75【已核对】）。信号点选得太晚 → 晚【推断】。
  2. 视觉追踪不足：看球到离开拍线能让大脑在挥拍中做小调整，「It's physically impossible to initiate the swing perfectly every time」（p.76【已核对】）。书把不看球联系到误击、打框；联系到「晚」是【推断】。
  3. 站位/步法：笔记 `:698` 写「没有视觉追踪或站位问题」【笔记归纳】，原书没有这一句；站位与晚的关系【推断】。
  4. 引拍顶点停顿后重启慢 → FLT-03【推断，来源仅 10/17 R3 模型整体分析】。
- 下游后果：不在身前 → 被挤 → FLT-01（p.116 把「late」与「too close」并列【已核对】）；触球时转体的切线力不指向目标（p.107：back-to-front 力「needs to be pointing at our target at the moment of contact」【已核对】）。
- 易混淆：
  - 有翻拍时「晚 → 下网、早 → 出界」（p.38、p.105【已核对】）——那是 FLT-09 的失误特征，不能只凭「下网多」判晚。
  - 来球快时等待为零（p.75）；发球机低速下来球慢，等待长是正常的，不等于晚。

## 4. FTT 原话 / 出处（主干）
- 「2. The racket strikes the ball in front of the body (in a pinch, in front of the forearm)」— 书 p.24【已核对】｜笔记 `:27–38`、`:46–49`
- 「Your cue for when to initiate the forward swing is not that a fixed amount of time has passed… Rather, it is a visual cue.」— 书 p.75【已核对】｜笔记 `:208`
- 「nearly all of your visual focus belongs on the ball from the time when it is struck by your opponent, to the time when it leaves your own strings」— 书 p.76【已核对】｜笔记 `:213–224`
- 「If, as an experiment, you force yourself to not watch the ball through contact, you'll be amazed at just how often you mishit the ball or frame it.」— 书 p.76【已核对】
- 笔记 `:698`「击球太晚｜没有视觉追踪或站位问题」【笔记归纳】。

## 5. TPA 辅证 / Bourne 技巧
- 本轮未挂接。

## 6. 确认 / 排除测试（发球机低速，10 球）
> 开始前：准备时鼻吸，启动前挥至触球时用嘴短促呼气。（不屏气；若前额/枕部痛、眩晕、恶心，当天停止并就医评估。）
- 前提：**必须有 S 机位**，否则本卡只能标「无法判断」。
- 做法：5 球照平时 + 5 球「一直看着球，直到它离开拍线」（书 p.76），每球口播结果。
- 确认：第一组 OBS-03 有 ≥4/5 拍触球在身体侧面或身后，且第二组触球点前移、口播「好」更多 → 支持「晚」且上游与视觉追踪有关。
- 部分确认：第一组 ≥4/5 拍偏晚，但第二组无改善 → 「晚」成立，上游不是视觉追踪，转查启动时机（FLT-03 的测试）或 FLT-04。
- 排除：第一组 OBS-03 已有 ≥4/5 拍在前髋之前 → 不晚，关闭本卡，转查 FLT-01（横向间距）。

## 7. 练习与感受提示（Intuition-First）
- 外部目标：**「一直看着球，直到它离开拍线。」**（书 p.76）

## 8. 禁忌
- 本项目永久禁止主动「推肘 / 降拍 / 旋后 / 伸肘」类口令，本卡不得把它们作为处方。
- 不用「数节拍 / 固定时间启动」：书 p.75 说启动信号是视觉线索，不是计时。
- 不把「顶点停顿」本身当成晚的证据（书 p.75，见 FLT-03）。

## 9. 用户关联
- `user_case/problem_register.md#`（待建）｜learning.md `#oct17-court-clip`

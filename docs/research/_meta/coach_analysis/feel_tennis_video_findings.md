# Feel Tennis (Tomaz Mencinger) — 视频实地观看发现

> 方法：用 Gemini 3 Flash 真实观看 8 个 Feel Tennis Instruction 频道视频，抽取 Tomaz 在视频里**实际做出的动作、指向的部位、说出的口语提示词**。不是从博客文字反推。
> 原始 JSON：`knowledge/extracted/coach_videos_v2/feel_tennis_{video_id}.json`
> 抽取脚本：`scripts/watch_feel_tennis_videos.py`

## 已观看视频清单

| # | Topic | Video ID | Title |
|---|---|---|---|
| 1 | Unit Turn | vcWAEcF6klU | Tennis Forehand Unit Turn - It's Not A Backswing |
| 2 | Backswing Illusion | guUg4hVI1AE | Deconstructing A Tennis Forehand Backswing |
| 3 | Wrist Lag / Slap | 2D7UlPQHce4 | Tennis Forehand Wrist Action: Slap vs Snap Explained |
| 4 | Modern Forehand 概览 | 9KRYA9ZlYmM | Modern Tennis Forehand Technique In 8 Steps |
| 5 | Fundamentals | 5LOKkHpFpFU | How To Hit A Tennis Forehand - 3 Simple Concepts |
| 6 | Contact Point & Timing | MO01CaN6lFc | Tennis Forehand Contact Point And How To Find It |
| 7 | Hip Rotation / Open Stance | Auem1-8t3rE | Why Every Tennis Forehand Starts With An Open Stance |
| 8 | Classic vs Modern | 0Mf8SFX_LuI | Classic Tennis Forehand vs Modern Forehand Technique |

**Note**：Split Step / Non-Dominant Arm / Improve Timing 三个主题对应的视频在 feeltennis_video_state 中为 "failed"（403 members-only 或代理无法访问），未能观看。以已观看的 8 个视频覆盖准备→引拍→击球→手腕释放→站位→分段鞭打的全链路。

---

## 1. Tomaz 视觉教学的共同模式（跨 8 个视频浮现）

### 1.1 "心理意象→物理动作"是 Tomaz 的核心桥梁
他从不用生物力学术语解释，而是给一个日常物件的意象，让学员用身体去复刻这个意象的感觉：

| 意象 | 对应动作 | 视频 |
|---|---|---|
| 拍打地毯除尘（老奶奶打挂毯） | 手腕 slap 释放 | #3 Wrist Action |
| 橡皮筋被拉伸后弹回 | 身体分段蓄力+鞭打 | #8 Classic vs Modern |
| 蓝色方块整体转入球 | 躯干刚体转动拦截球 | #5 Fundamentals |
| 保龄球 vs 掷铁饼 | 挥拍路径直线 vs 圆周 | #4 Modern Forehand |
| 沿拍框边缘掉拍 | racket drop 重力辅助 | #4 Modern Forehand |
| 挤压并滚动球 | 击球时拍面与球的作用感 | #4 Modern Forehand |
| 双脚在律动"跳舞" | ready state 弹性 | #4 Modern Forehand |

### 1.2 他反复使用"视觉错觉"作为诊断框架
在 #1 Unit Turn 和 #2 Backswing Illusion 两个视频中，他**字面上**说 "visual illusion"：
- 学员以为自己的球拍在后摆中走了很大一段距离 → 以为手臂在向后用力
- 事实上：手臂只做了很小的抬起，大位移来自躯干转动
- 他的教学手段：**消除"向后"这个心理暗示词** → 让学员嘴上喊 "Left!"（左肩向前）

### 1.3 他用"物理约束+外部刺激"诱导正确形态
而不是口头纠正。典型例子：
- #1 Unit Turn：站在学员身后**推他的肩膀**测试稳定性，逼学员自己感受下背部是否稳
- #6 Contact Point：用**球网**作为物理障碍，或**从背后抛球**，强迫学员在前方拦截
- #3 Wrist Action：让学员在球网前**拍击球网顶部**，模拟拍地毯

这是 Tomaz 最独特的价值——他不纠错，他**改变环境让错误无法发生**。

### 1.4 "主动 / 拦截" vs "被动 / 等待"是反复出现的二分法
- #5: "Intercept the ball" 不要等球到
- #6: "Intercept the ball" "Don't respond to speed with speed"
- #7: "Read the ball, flight data" 在右腿上收集数据再决定
- #1: "There is no backward, only forward"

这不是具体动作，是**决策框架**：Tomaz 要的是"我主动冲进去拦它"的心态。

---

## 2. 每个视频的独家视觉细节（博客看不到的）

### #1 Unit Turn (vcWAEcF6klU)
- **独家动作**：Tomaz 走到学员 Alan 身后，在 Alan 准备引拍时**用手推他的肩膀**。站不稳 = 背部没绷紧 = Unit Turn 失败。
- **独家口令**：让学员在击球前**大声喊 "Left!"** —— 用语言强制大脑关注左侧启动。
- **身体指向**：下背部（稳定）、左肩（引导）、腹外斜肌（核心发力）。

### #2 Backswing Illusion (guUg4hVI1AE)
- **独家视觉**：他**在屏幕上用方框标记胸部/躯干**作为转动轴心，同时把手和前臂也框起来展示"相对位移极小"。这是动画标注，不是自然拍摄。
- 提出"误解 (Misinterpreting)"框架：问题出在**对自身动作的主观解读**，不是动作本身。

### #3 Wrist Action: Slap vs Snap (2D7UlPQHce4)
- **独家意象**：老奶奶拿塑料拍打挂在杆上的地毯除尘。这是非常具象的日常画面。
- **独家练习**：站在球网前**把球网当地毯拍**。用实体物件让学员感受手腕释放的弹性。
- **明确的错误分类**：(a) 推球（手腕锁死）(b) 过度释放（手腕在随挥时外翻受伤）。中间才是正解。

### #4 Modern Forehand 8 Steps (9KRYA9ZlYmM)
- 非常密集的 8 个小意象，其中最独特：
  - **"食指分开支撑"**：握拍时食指要有独立的支撑感（不是五指并拢）
  - **"沿拍框边缘掉拍"**：racket drop 不是主动下拉，而是让拍沿一个边缘滑下去，重力辅助
  - **"保龄球 vs 掷铁饼"**：发力是直线扔保龄球，不是横向转圈掷铁饼
  - **"挤压并滚动球"**：击球瞬间的感觉是 compress + roll，不是击打
  - **"左手接住球拍"**：随挥是为了**释放肩转**，左手接拍是目的不是装饰

### #5 3 Simple Concepts (5LOKkHpFpFU)
- **独家视觉**：把整个身体**可视化为一个"蓝色方块"**（可能是叠加图层/动画）。方块整体移动进球 → 手臂是被动挂件。
- "Turn the whole body into the ball" —— 不是转身 + 挥拍，是身体**走进球**。

### #6 Contact Point (MO01CaN6lFc)
- **独家手段**：限制学员的后摆（物理阻挡），或**从学员身后抛球**让他来不及大后摆 → 被迫前接。
- **金句**：「Don't respond to speed with speed」—— 来球快不是你该挥拍快的理由。
- 用地面标记画出"击球区"，让学员有明确的空间目标。

### #7 Open Stance (Auem1-8t3rE)
- **独家概念**：「Flight data」—— 把读球比作飞行数据采集。站在右腿上**等数据够了再动**。
- **核心规则**：默认开放站位 → 只有在判断球短且慢后**才**迈步转为中性站位。开放站位是"时间缓冲"而非"风格选择"。
- **独家视觉**：他用**计时器**（字面上的计时器动画）展示"你其实只有零点几秒"。

### #8 Classic vs Modern (0Mf8SFX_LuI)
- **独家二分**：Classic = "One unit"（整体僵硬）；Modern = "Segmented"（分段）。与很多人把 Unit Turn 当成"整体"的理解相反 —— 准备期是整体，发力期必须分段。
- **身体指向顺序**：胯与肩分离（第一段）→ 胸肌拉伸（手臂滞后）→ 前臂与手腕分离（拍头滞后）。**三段分离、三段弹回。**
- **警告**：Tomaz 明确提醒业余球员**不要追求极端的分段鞭打**，先稳定。这个平衡观点在他的博客里没有被强调。

---

## 3. 针对用户当前状态（摆锤时机 + 肘在前）的映射

参考 `docs/record/learning.md`，用户当前痛点是"摆锤时机"和"肘在前"。Feel Tennis 视频里最直接相关的处方：

1. **摆锤时机 → 早转 + 等 + 拦截**（#1 + #6）
   - 用 "Left!" 口令触发早转（眼睛看到球飞向正手侧就喊）
   - 转完在右腿上**收集 flight data**（#7），不要急着挥
   - 看到球进入击球区再"拦截" —— 拦截不是追，是主动上前堵

2. **肘在前 → 蓝色方块+保龄球**（#5 + #4）
   - 不要想肘的位置，想"整块身体转进去"
   - 发力路径像保龄球直线推出，而不是掷铁饼横甩 —— 直线推必然肘领先
   - 随挥用**左手接住球拍**作为肩转完整的标志

3. **手腕不紧 → 拍地毯**（#3）
   - 在球网前拍网顶，专门练手腕从 lag 到 slap 的释放
   - 避免两个极端：锁死推球 / 外翻过度

---

## 4. 给 VLM 提示词 / 诊断引擎的 Top 5 可用发现

### 4.1 "左肩位移 > 右臂位移" 是 Unit Turn 正确性的可直接检测特征
Tomaz 在 #1 明确示范：如果准备期右臂先动、左肩没动，就是错的。VLM 在 `prep_start → prep_complete` 之间可以：
- 计算 `left_shoulder.x_displacement / right_wrist.x_displacement`
- 若比值 < 0.3，Unit Turn 失败（手臂主导引拍）

### 4.2 准备结束后的"等待帧"是节奏问题的决定性信号
Tomaz 在 #6 + #7 反复强调 "wait for flight data"。早转 ≠ 早挥。VLM 可以：
- 检测 `prep_complete_frame → forward_swing_start_frame` 之间的角速度低谷
- 若两帧几乎紧贴（无低谷），判定为"早转早挥 / 没有等"
- 这恰是 Tomaz 治疗"打晚"的核心 —— 拆开准备和挥拍

### 4.3 击球点的"拦截 vs 被追"可通过拍头运动方向检测
Tomaz 在 #6 说 "racket and ball going toward each other"。VLM 可以：
- 在 `contact_frame` 检测拍头速度向量与来球向量的夹角
- 角度接近 180°（相向）= 拦截型；角度 < 90°（同向）= 被追型（打晚）
- 这比单纯看"击球点在身前 vs 身侧"更鲁棒

### 4.4 分段鞭打的正确顺序是"髋 → 肩 → 肘 → 腕"的峰值时序，不是同时
Tomaz 在 #8 明确指出**三段分离**。VLM 可以：
- 计算 hip / shoulder / elbow / wrist 的角速度峰值帧
- 正确顺序：`hip_peak < shoulder_peak < elbow_peak < wrist_peak`（每段比上一段晚 2-5 帧 @60fps）
- 任何同时或逆序都是"Classic one-unit"僵硬模式
- **注意 Tomaz 的平衡观点**：业余球员分段差距太大也是问题，需要设定合理窗口

### 4.5 手腕释放的"slap"轨迹特征：拍头加速呈脉冲而非平滑曲线
Tomaz 在 #3 强调 slap ≠ smooth push ≠ extreme snap。VLM 可以：
- 在 `pre_contact → early_followthrough` 检测拍头线速度
- 正确：脉冲式（1-2 帧内急剧上升后回落） = slap
- 错误A：平滑上升（推球，手腕锁死）
- 错误B：持续加速到随挥（腕过度释放，易伤）
- 脉冲宽度可用作"放松程度"代理指标

---

## 5. 失败 / 未覆盖

| 主题 | 状态 | 备注 |
|---|---|---|
| Split Step Timing | ❌ 未观看 | 候选视频全部 403 failed (members-only) |
| Non-Dominant Arm 专题 | ❌ 未观看 | 候选视频全部 failed。但 #4 Modern 8 Steps 里涵盖了 "左手接球拍" |
| Improve Timing | ❌ 未观看 | 候选视频 failed。但 #6 Contact Point + #7 Open Stance 已覆盖节奏话题 |
| 8 个已选视频 | ✅ 全部成功 | 0 failures, 8/8 extracted |

所有 API 调用 1-2 次 503/504 重试后成功。响应时间约 40-90 秒/视频。总运行时间约 10 分钟。

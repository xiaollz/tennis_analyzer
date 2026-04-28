# 名教练沟通风格调研

> 编制日期：2026-04-27
> 目的：把"教练怎么对学员说话"作为一个独立维度研究。不是教什么内容，是怎么把内容塞进学员脑子里。
> 方法：四来源混合——(1) 我们已分析过的 RTP/TPA/FeelTennis 视频里教练原话，(2) 公开访谈与文章里的方法论自述，(3) 反例：我们当前 VLM 输出的语段，(4) 三方教练学文献对失败模式的归纳。
> 引用纪律：所有打了双引号的话都来自具名出处（per-video 分析或公开链接）。我自己提炼的归纳不打引号，标"作者归纳"。

---

## 共同模式（7 条横切观察）

### 1. 几乎都是"先展示对的物理图景，再点错"

不是"你错在 X"开头，是"这一拍真正发生的是 X，你以为发生的是 Y"开头。

- **Sky Kim**（aiwUqHQl-Ec）："When you step with the left foot, your contact point shifts... you feel like you have to be literally jammed to hit the ball." 先讲机制（左脚踩了会发生什么），最后才落到"jammed"这个学员主观能感知的词。
- **Tom Allsopp**（A7a8Ibci9MM）："The takeback, the swing path, everything looks pretty good. You get a little bit jammed as you're striking the ball." 先承认 3 件做得对的，再说 1 件错的，在同一句里。
- **Mouratoglou**（Olympics 访谈）：刻意"先看 assets，再看 flaws"——他认为如果一上午全是负反馈，"by the end of the week confidence is shot and the player will tune you out."

我们 VLM 输出的对照：直接 "Unit Turn 转开不足"、"小臂代偿"、"分裂步落地后才识别来球方向"——零承认。学员看完会觉得自己一无是处。

### 2. 用学员的主观感觉做坐标系，不是用术语

学员能感知的是"被挤"、"飘"、"用力但没力"、"球轻"。教练用这些词，不用"23° hip-shoulder separation"。

- **Sky Kim**："you feel like you have to be literally jammed"——直接借用"jammed"这个学员词汇。不解释 jammed 等价于 contact point shift，先共情，机制讲解放在后半句。
- **Tom Allsopp**（BH24bdGmTaM）："the racket is an extension of my hand"——把硬件感觉化。
- **Tomaz Mencinger**（个人页面）："feel, timing, rhythm and ball judgment"——他把这四个明确列为 key skills，整个频道围着"feel"这个不可量化的词转。

我们的 VLM："肩部转开只有 26°，转体严重不足"——26° 学员看不见也摸不着。

### 3. 一次只塞一个心智操作（mental operation）

业余教练学文献的一致结论（mytenniscoaching.com、tennisfitness.com 多篇）：**One thing per session.** 给两件就等于零件。

- **Sky Kim** shorts 风格：每个 30-60 秒视频只攻一个命题（One Foot Hit / Hip Locking / Hide the Elbow）。长视频 5-12 分钟也只讲一件事，从三个角度反复说同一件。
- **Tom Allsopp**（O1i9y5NSoig）整段视频核心就一个动作："Have the wrist lay the racket back AFTER we've rotated our body." 全段所有补充都为这一句服务。
- **Brian Gordon**（tennisplayer.net 访谈风格）："Don't worry about 'snapping' the wrist—or not. Make the right positions in the swing and let the rest take care of itself." 直接禁止学员同时管两件事。

我们 VLM 单球报告里同时说：Unit Turn 转开不足 + 大臂胸部空隙 + 脊柱不倾斜 + 同步性 0.71 + 26° + 154°——五件事并列，且没主次标注。

### 4. 命令式而非描述式动词

教练直接给动作。"你要做 X"，不是"X 应该被做"。

- **Tom Allsopp**："I want to look at the forehand like a throwing motion."（azVf6CyDfVk）"Have the wrist lay the racket back."（O1i9y5NSoig）几乎全是 imperative + first person.
- **Sky Kim**："The foot closer to the direction that you're moving into has to move first."（Im2JyVN8Rn0）—— "has to" 不是 "should"。
- **Brian Gordon**："Make the right positions in the swing." 命令式。

我们 VLM："核心问题是 Unit Turn 转开不足。其余问题可能是这个核心问题的下游表现。"——全是被动判定语气，没人在说话。

### 5. 类比要绑物理机制，不要孤立飞

类比泛滥是业余频道的通病。一线教练用类比，但每个类比背后压着一个具体力学事实，学员一旦试动作就会撞上那个事实。

- **Tom Allsopp**："Tennis is a rotational sport, not a linear sport."（muxc0h0YAJg）—— rotational 这个词后面立刻跟"because the kinetic chain works through twist not push"，类比立刻落地到具体动作（开放式站位、髋部带肩）。
- **Sky Kim**："use the catapult effect with the tension that you create with your body"（9ihq4WFCWy0）—— catapult 这个比喻的下一句必带"hip locks while shoulder turns"，物理对应清楚。
- **Tom Allsopp**（azVf6CyDfVk）："I want to look at the forehand like a throwing motion. Right, left, throw."—— throw 这个隐喻立刻拆成 right-left-throw 的脚步顺序。

反例：劣质教学常说"像鞭子一样甩"——但什么是鞭根、鞭梢、什么是 lag 时机，全没有。

### 6. 留白：教练知道什么时候停下不讲

最难的一条。Tom Allsopp、Sky Kim 都有大量"我不告诉你为什么，你回去试就懂了"的段落。

- **Sky Kim**（-4YJ_0Ya2lM）："Footwork 5-6 vs Swing 3."—— 不解释为什么要 5-6 不是 6-7，让学员去试出来。
- **Tom Allsopp** 的 reverse forehand 段（-HFgyYQOALM）："Their coaches said not to do it. I said you have to. And once they started doing it, they were able to hit more variety and faster."—— 不论证为什么，引用结果。
- **Mouratoglou** 公开论调："A tactical tip at a crucial moment can make a big difference."—— 他承认大部分时间不说话，只在关键时刻给 1 句。

我们 VLM 反例：每个发现都强行追加机制段+知识库映射段+训练历史段+推荐练习段——把每一条信息都铺到底，没有留白。

### 7. 收尾给一件事，最多绑一个验证点

业余教练学文献、Mouratoglou、Sky Kim、Tom Allsopp 共同点：结尾不是清单，是一个"回去就做这个"的指令。

- **Tom Allsopp**（BH24bdGmTaM 收尾）："the racket is an extension of my hand"——不是清单。一个意象。
- **Sky Kim** drill 模板：一个动作 + 一个验证（"check your follow-through height"、"feel weight on right foot only"）。
- **业余教练学**（mytenniscoaching.com）："One development area at a time. From the player's perspective it is challenging to focus on more than one or two aspects."

我们 VLM 反例：报告结尾把 muscle activation guide + 训练历史 + 推荐练习 + 多条根因链 全列一遍。

---

## 各家的标志性手法

### Sky Kim (RTP)

**句式**：先讲业余痛点（学员能共鸣的主观感觉）→ 解释一个被忽视的物理机制 → 给一条极端的反向 cue。

**温度**：冷静、解析、但带"我也是从那走过来的"的同理。

**例句**：
- "When you step with the left foot, your contact point shifts... you feel like you have to be literally jammed to hit the ball."（aiwUqHQl-Ec）
- "You actually lose power by misdirecting the torque created by your body."（Im2JyVN8Rn0）
- "You need tension to build natural power... use the catapult effect with the tension that you create with your body."（9ihq4WFCWy0）

**好在哪**：每句都有"主观感觉 + 物理事实 + 隐含动作改变"三层。学员读完知道自己感觉的是什么、为什么这么感觉、要改什么。

**可学的具体手法**：
1. **借词共情**：先用学员脑子里已有的词（jammed、被挤、没力），再翻译成物理。
2. **极端 cue**：他不说"略微 X"，他说"literally only stand on right foot"。极端 cue 学员才记得住，因为日常打球只能做到 70%，给 100% 才能出 70%。

### Tom Allsopp (TPA)

**句式**：断言 + 立刻对比 + 即时演示。"It's X, not Y. And here's why."

**温度**：确信、教练范、有点"我打过你没打过"的权威感，但语速从容。

**例句**：
- "Tennis is a rotational sport, not a linear sport."（muxc0h0YAJg）—— 一句断言开场，全 video 围着这一句证据链转。
- "I want to look at the forehand like a throwing motion... Right, left, throw."（azVf6CyDfVk）
- "The takeback, the swing path, everything looks pretty good. You get a little bit jammed as you're striking the ball."（A7a8Ibci9MM）—— 三个肯定 + 一个否定的 1:3 比例。
- "When we pull something, things get a little bit tight. And when you use a throwing motion, things get released."（A7a8Ibci9MM）—— 对照式机制。

**好在哪**：
- 1:3 比例反馈（三句肯定一句否定）让学员愿意听。
- 对照式机制（pull vs throw）比单方向解释好记。
- "I want" 而不是 "you should"——把建议变成自己的判断，学员更易接受。

**可学的具体手法**：
1. **断言式开场**：第一句就钉一个命题（"Tennis is X not Y"），后面所有内容都为它服务。
2. **1:3 反馈节奏**：每提一个问题，前面或后面绑三个具体的"已经做对的"。
3. **第一人称 want/feel**：用"我希望""我觉得"代替"你应该"。

### Patrick Mouratoglou

**句式**：整体性、关系性，少给具体技术。但凡说技术必裹一层情境（"in this kind of point you want to..."）。

**温度**：导师感、长期视角。

**核心方法**（来自 Olympics 访谈、CoachTube blog）：
- "If a player is only going to hear criticism... by the end of the week confidence is shot and they will tune you out."
- 先看 assets 再看 flaws，"strengthen what she's already great at."
- "A tactical tip at a crucial moment can make a big difference."—— 不抢话，留 99% 时间不说话，只在关键点出一句。

**好在哪**：他承认教练话多 = 教练废。他用"tune out"这个词描述学员对话太多教练的反应，是非常清醒的元认知。

**可学的具体手法**：
1. **少话原则**：明文规定每次只给 1 个建议，用"沉默预算"逼自己挑最重要的那条。
2. **资产清单**：每次反馈强制包含一条"你之前练对的，别动它"——和我们 coach_style.py 里的"承认已对"完全对齐。

### Tomaz Mencinger (Feel Tennis)

**句式**：渐进式、问题导向。"如果你感觉 X，试试 Y。"

**温度**：温和、像耐心的物理老师。

**核心方法**（来自他自己的 about 页面 和 podcast）：
- 关键技能列表：**"feel, timing, rhythm and ball judgment"**——他刻意用四个不可量化的词。
- "Effortlessly and with more joy on the court"——目标导向是体验，不是技术参数。
- 学员评价："speaks clearly and teaches in digestible bites"——他自己以这个被引用为荣。

**好在哪**：他知道大部分业余学员不需要更多技术细节，需要更少。他的频道密度是行业里最低的之一，但留存率最高。

**可学的具体手法**：
1. **小口喂**：一次只讲一个动作的一个细节，下一节课再讲下一个。"digestible bites" 是他的产品定位。
2. **不可量化词**：feel、rhythm、judgment——这些词学员能立刻接住，"23° rotation" 学员接不住。

### Brian Gordon (Biomechanics)

**句式**：研究者口吻，但教学时反而最克制——他越懂越不让学员想细节。

**例句**：
- "Don't worry about 'snapping' the wrist—or not. Make the right positions in the swing and let the rest take care of itself."（tennisplayer.net）
- "There is no conscious forward wrist snap; the muscles... are actually resisting the forward wrist joint motion."

**好在哪**：他把"研究者知道的"和"教练该说的"严格切开。他的研究知道有 17 个变量，他给学员讲只讲 4 个 pillar。这是最稀缺的能力。

**可学的具体手法**：
1. **明禁某些有意识动作**："Don't worry about X" 是有效指令，比"do Y" 还有效，因为它阻止学员用错的心智模型。
2. **位置 > 肌肉动作**：教学员到达哪个位置（geometry），不教学员怎么发力（muscle action）。

### Chris Lewit

**句式**：演讲式、引经据典、academic。

**核心理念**（来自 New York Tennis Magazine、Long Island Tennis Magazine）：
- "To be a champion in anything in life, you have to be endlessly curious and always working to get better every day."
- 强调"rhythmic, fluid and elastic technique"——和 Mencinger 接近。
- 他的书 *Winning Pretty* 把"看起来漂亮"作为标题，这本身就是一种沟通策略：把技术正确包装成审美。

**可学的具体手法**：
1. **审美化包装**：把"对的技术"称为"漂亮的"，学员的内驱力会自然对齐。FTT 也用"effortless"做同样的事。

### Jeff Salzenstein

主要在 YouTube 走"前 ATP 给业余讲底层框架"的路线。从访谈和 wikipedia 看，他的方法论叫 "Own Your Zone"，把表现 = 心理状态调节。技术内容上没有 Sky Kim 和 Tom Allsopp 那么有标志性的句式，可学的不多。本调研不重点列。

---

## 我们当前 VLM 输出 vs 名教练对比

取自 `/Users/qsy/Desktop/tennis/storage/diagnoses/41ae13dd6128_c000/report.md`，原文未改。

| 主题 | 我们的输出 | Sky/Tom 等会怎么说 |
|---|---|---|
| 开场（第 1 球诊断） | "第 1 球暴露了一个关键问题：分裂步落地后才识别来球方向。" | Sky 风：*"你这一拍最让我担心的是，落地之前你眼睛已经看到球了，但身体还在等。这中间差了 1 帧——比赛里 1 帧就是球落点偏 30cm。"* |
| 量化数据 | "肩部转开只有 26°（正常应>60°），转体确实严重不足；最小膝角 154°（应≤140°），下肢承载偏弱。" | Tom 风：*"你的转体只到这（指 26°），但你需要到这（指 60°）。差了一倍。看起来一样，做出来差一倍。"* —— 用相对而非绝对数字。 |
| 多问题处理 | 同时列 Unit Turn 26° + 大臂胸部空隙 + 脊柱不倾斜 + 同步性 0.71，无主次。 | Mouratoglou 风：只挑一条，其他的明文说"不动"。我们 coach_style.py 已经写了"单根因"原则，但 VLM 实际输出违反了。 |
| 训练历史提示 | "⚠ 训练历史提醒：「小臂代偿」在你的训练中出现过 9 次（...2026-03-15、2026-03-26、2026-03-27），已解决。解决方案：4/1 通过腋下贴住..." | Sky 风：*"小臂代偿那个，你 4/1 已经修了。这次它没出现，是因为你已经会用腋下了。今天不动它。"*——一句话承认，转头就过。 |
| 收尾 | "推荐练习：落地前先看球。做法：Quentin/Federer footwork drill：起跳瞬间眼睛+躯干必须已经朝向来球方向，落地即第一步。镜前 shadow 20 次。原理：落地后才转 = 浪费 1-2 帧 = 第一步永远晚。" | Tom 风：*"这周就练一个：落地前先看球。Shadow 20 次，每次只看你的眼睛动得有没有比脚早。其他都别动。"* |
| 整体语气 | "根因层级：L5 步伐与站位"、"准备阶段诊断"、"muscle_activation_guide preparation:" | 名教练几乎不会出现这种带下划线的内部 token、层级名词。 |

具体哪些段落是 AI 腔的："核心问题是「Unit Turn 转开不足」。其余问题可能是这个核心问题的下游表现。"——"其余问题可能是 X 的下游表现"是 AI 标准模板化推理句，没有人这么说话。教练会说"剩下那几个看着像问题，其实是 Unit Turn 没转够带出来的，今天别管"。

---

## 应纳入 coach_style.py 的新原则

这一节是产出。当前 coach_style.py 已有 8 条原则。下面列建议新增的，每条都标"为什么 + 来源"。

### 1. 第 9 条："允许使用学员的原话作为坐标系"

**理由**：Sky Kim 全部诊断都从"jammed"、"feel like you can't hit"、"your hand feels disconnected"这种学员主观词汇切入，再翻译成物理。我们当前 coach_style.py 第 2 条只说"把用户的主观感觉翻译成物理"，但没明说**先用学员词、再翻译**这个顺序。顺序错了反馈就生硬。

**来源**：RTP aiwUqHQl-Ec、A7a8Ibci9MM；TPA A7a8Ibci9MM。

### 2. 第 10 条："1:3 比例——每提 1 个问题，必须前置 3 个具体的'你做对的'"

**理由**：Tom Allsopp（A7a8Ibci9MM 整段）的标准模板就是 "The takeback, the swing path, everything looks pretty good. You get a little bit jammed..."。Mouratoglou 公开理论同样：负反馈 > 50% 学员就 tune out。我们当前 coach_style.py 第 1 条只说"承认已对"，没规定**比例**。1:3 是可执行的硬约束。

**来源**：TPA A7a8Ibci9MM；Olympics 访谈 Mouratoglou。

### 3. 第 11 条："禁止显式列出内部分类 token"

**理由**：当前 VLM 输出里出现"L5 步伐与站位"、"muscle_activation_guide preparation"、"v2_late_split_recognition"——这些是知识库内部 ID，泄漏到学员可见层就立刻露馅是 AI。教练讲话不会蹦出层级编号。

**来源**：我们自己的 report.md 第 16/26/35-39 行。

### 4. 第 12 条："数字用相对差，不用绝对值，除非学员已经懂"

**理由**：Tom Allsopp 演示时几乎不说"应该 90°"，他说"应该到这（指演示位置），你只到这（指当前位置）"。Sky Kim 的视频里数字几乎只在他自己讲解机制时出现，对学员的指令里出现得很少。"26°"对学员是死的，"差了一半"是活的。

**来源**：TPA 多个视频；RTP 8LsLG8ZOa1g、Im2JyVN8Rn0。

### 5. 第 13 条："留白——每次输出至少有一段不解释，只下指令"

**理由**：Sky Kim "Footwork 5-6 vs Swing 3" 不解释。Tom Allsopp "I said you have to" 不论证。当前我们的输出每条发现都铺机制+知识图谱+练习+原理，留白为零。学员的认知负荷被拉满，结果一条都记不住。

**来源**：RTP -4YJ_0Ya2lM、-HFgyYQOALM；业余教练学文献（mytenniscoaching.com）"talking too much"。

### 6. 第 14 条："明禁式 cue 和肯定式 cue 至少 1:1"

**理由**：Brian Gordon 经典："Don't worry about snapping the wrist." 明禁某个错的心智模型，比给一个新动作更有效。我们当前输出全是"做 X"，几乎没有"别想 X"——但学员脑子里那个错模型不被显式禁掉，新动作就会立刻被旧模型污染。

**来源**：tennisplayer.net Brian Gordon；TPA Gv7sF5DKK5E"Don't flip"系列；FeelTennis MO01CaN6lFc"Don't respond to speed with speed"。

### 7. 第 15 条："沉默预算——每个报告硬性砍掉 30% 字数"

**理由**：Mouratoglou 的"a tactical tip at a crucial moment"——他做的恰恰是少说。我们当前 report.md 一个球的诊断 60 行；名教练 60 秒视频也就 12-15 句。同等信息量我们多 5 倍。多余的 4 倍是 AI 强制铺底层、强制完整闭环造成的。

**做法**：在 prompt 里加"输出必须比初稿短 30%，砍掉的应该是机制铺垫和重复的根因解释"。

**来源**：Olympics 访谈 Mouratoglou；mytenniscoaching.com "talking too much"。

### 8. 第 16 条："训练历史用一句话扫过，不展开"

**理由**：当前我们 VLM 输出的训练历史段："⚠ 训练历史提醒：「小臂代偿」在你的训练中出现过 9 次（...2026-03-15、2026-03-26、2026-03-27），已解决。解决方案：4/1 通过腋下贴住（手臂焊在胸上）+动力链完整串通解决。最终标志：'完全感受不到胳膊在发力'。"—— 这一段把数据库记录原样吐出。教练应该说"小臂代偿那个你 4/1 已经修好了，今天没出现，下面只看 X"。**承认 + 翻篇**，不要复述病史。

**来源**：作者归纳 + Mouratoglou "look at assets first" 原则的反向应用。

---

## 反例：从我们 VLM 输出挑 3 段典型 AI 腔

### 反例 1（来自 41ae13dd6128_c000/report.md 第 22 行）

> "从视频中观察到以下问题：Unit Turn 转开不足（图 preparation）、在小跳腾空时，球员的视线已经锁定来球方向，但躯干落地后才开始大幅转向。、大臂内侧和胸部侧面有明显空隙（约一个拳头以上），相比准备阶段，间距明显变大。、击球瞬间脊柱与地面接近垂直，没有明显的侧向倾斜。"

**为什么读起来是机器写的**：
- 用顿号串连四个并列观察，没有主次。教练只会挑一个，其他不提。
- "从视频中观察到以下问题"是标准 LLM 模板开场，没有任何主体感。
- 句子之间的句号 + 顿号混用（"...大幅转向。、大臂..."）暴露了拼接逻辑。

**Tom Allsopp 风改写**：*"我看你这一拍，前后挥都没毛病。问题在准备——你落地的时候身体还没转开。就这一件，今天只修这个。"*

### 反例 2（同 report 第 116 行）

> "根因分析：这些问题的最上游根因是「小臂代偿」。因果链路：「动力链断裂」导致「小臂代偿」。也就是说，你看到的表面症状（准备阶段上半身（肩部和手臂整体）最先开始转动。、是先放到一个位置停顿，然后在前挥时突然下沉拉出的"两步式"动作。）其实是上游问题的下游表现。"

**为什么是 AI 腔**：
- "因果链路"、"上游"、"下游"、"表面症状"——这些是诊断系统的内部行话。
- "也就是说"是 LLM 自带的过渡词，没人讲话用这个词。
- 嵌套括号引用观察项目，结构上是 JSON 不是话语。

**Sky Kim 风改写**：*"你这一拍小臂在自己干活，根上是动力链断了——你身体没把力传到胳膊，胳膊只能自己甩。修法不是管胳膊，是让身体先动。"*

### 反例 3（同 report 第 26 行）

> "muscle_activation_guide preparation: 腹外斜肌+背阔肌通过离心收缩储能形成的肩髋分离角，是后续蹬地转髋能爆发出力量的物理前提。你 Unit Turn 应该感觉到左侧腹斜肌被拉开、背部张力上升、右大腿后侧绷紧——如果只感觉肩膀在转但腹部没有拉伸感，说明转开幅度不够，核心弹性势能没建立。"

**为什么是 AI 腔**：
- 内部 token 名 `muscle_activation_guide preparation` 直接泄漏到学员侧。
- "离心收缩储能"、"核心弹性势能"——这些是教科书词，不是教练口语。
- 一段把三个肌群点完，没有取舍。

**Mouratoglou 风改写**：*"Unit Turn 你只盯一个感觉：左腹被拉开。其他都别想。"*

---

## 报告（≤200 字）

**找到的共同模式**：(1) 先讲对的再讲错的；(2) 借学员主观词做坐标系；(3) 一次一个心智操作；(4) 命令式 + 第一人称；(5) 类比必绑物理事实；(6) 留白——会的越多说得越少；(7) 收尾给一件事。

**最值得抄的 1-2 个手法**：
1. **Tom Allsopp 的 1:3 反馈比例**——每提 1 个问题，前面绑 3 个"你做对的"。可执行、可验证、立刻不像 AI。
2. **Brian Gordon 的明禁式 cue**——"Don't worry about X." 比"do Y"更能阻断错心智模型。我们当前输出 0% 是这种句式。

**我们最不像教练的 1 个具体毛病**：内部分类 token 直接泄漏（"L5 步伐与站位"、"muscle_activation_guide preparation"、"v2_late_split_recognition"）。这是一句话之内立刻露馅是 AI 的最致命特征。修起来也最简单——后处理时把所有 snake_case 和层级编号过滤掉就行。

---

## 来源

- Sky Kim / RTP 已分析视频（per-video docs in `/docs/research/road_to_pro_video_analyses/`）
- Tom Allsopp / TPA 已分析视频（`/docs/research/tpa_video_analyses/`、`/docs/research/tomallsopp_video_analyses/`）
- Tomaz Mencinger / Feel Tennis（`/docs/research/feeltennis_video_analyses/` + [feeltennis.net/about-me](https://www.feeltennis.net/about-me/)）
- Patrick Mouratoglou — [Olympics.com 访谈](https://www.olympics.com/en/news/patrick-mouratoglou-tennis-coach-training-methods-serena-williams)、[CoachTube blog](https://coachtube.com/tennis/articles/4-lessons-patrick-mouratoglou-learned-from-coaching-serena-williams)
- Brian Gordon — [tennisplayer.net author page](https://www.tennisplayer.net/author/brian-gordon-phd/)、[Tennis Center for Performance Research](https://tennisperformanceresearch.com/dr-brian-gordon/)
- Chris Lewit — [chrislewit.com](https://chrislewit.com/coaching/)、[New York Tennis Magazine 报道](https://newyorktennismagazine.com/article/chris-lewit-aims-to-modernize-tennis-teaching/)
- Jeff Salzenstein — [Tennis Evolution about](https://tennisevolution.com/about-jeff/)
- 反例数据：`/Users/qsy/Desktop/tennis/storage/diagnoses/41ae13dd6128_c000/report.md`
- 业余教练学失败模式：[mytenniscoaching.com Top 5 Mistakes](https://mytenniscoaching.com/2022/08/17/top-5-coaching-mistakes-and-how-to-avoid-them/)、[tennisfitness.com](https://www.tennisfitness.com/blog/tennis-tips-and-costly-coaching-mistakes)

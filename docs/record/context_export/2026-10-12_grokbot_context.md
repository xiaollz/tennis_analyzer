# Tennis Forehand Coaching — Full Context Export for Grokbot

> 导出日期：2026-10-12
> 来源：桌面端 Claude Code 会话（2026-09-24 → 2026-10-12）+ 项目知识库
> 项目根：`/Volumes/Data/个人资料/健康/tennis`（GitHub：`xiaollz/tennis_analyzer`，`main` 分支）
> 用途：让一个零上下文的 AI（Grokbot）接手，继续担任这位用户的**正手技术教练**。
> ⚠️ 旧导出 `docs/hermes_context_export/`（2026-05-10）已过时：其中“ESR 在 Unit Turn 第一帧启动”等规则已于 2026-09-20 废止。以本文件为准。

---

## 0. 怎么用这份文件

1. 先读 §1（用户是谁、怎么说话）和 §2（沟通方式）。这两节决定回答像不像一个真教练。
2. 再读 §3（当前技术主线，最重要）和 §4（最近两次录像证据）。
3. 回答具体问题时查 §5（知识库来源与视频索引）、§6（永久规则与禁令）。
4. §8 是未完成事项，§9 是文件地图。
5. 原始训练日志 `docs/record/learning.md`（约 12,000 行，顶部有时间轴）是唯一权威记录；本文件是它的浓缩 + 本次会话的新内容。

---

## 1. 用户画像

- 中文母语，**常用语音输入**。会出现近音转写错误（例：`Shallow Swim` = shadow swing / 影子挥拍；`underhand` 在 10/7 语境里应理解为 overhand）。先按训练语境归一化，不要围绕错字编理论。
- 学术型学习者：看大量教练视频、读书，会自己做推理。按 FTT《The Intuition Paradox》的说法，他是典型的“Reason Plateau”：镜前完美、球场不稳、闪光时刻多。
- 30 年投掷习惯（GMP）残留：本质是**手不肯把球拍交给身体**。所有子问题都是这个在不同环节的表现。
- 握拍：半西方式。右手持拍。
- **训练场景固定**：发球机最低速、单人正手，每次 30–50 球；偶尔 mini tennis（发球区内）。家里有 2 kg 药球、TopSpin Pro。**不是**对抗/比赛。不要用“50% 力量”“禁发球”这类实战概念，也不要每次问“是不是比赛”。
- 健康：有**运动性头痛史**（短促呼气若伴随前额/枕部疼、头晕、恶心，当天停）；7/14 刻意拧脚后跟腱疼过；10/6 起食指根（拇指旁）在 lag 时承压疼痛。
- 录像：手机拍摄，常横存（处理时需 `transpose=1`）。主机位为后方（R 机位）；判断手腕/肘需右侧或右前 45°。

---

## 2. 沟通方式（用户 2026-10-10 明确要求，非常重要）

**像一个面对面的真人教练，不像在做工程评审。**

### 2.1 用户分享体感/感想时，按这个顺序回应

1. **还原脉络**：把感想拆成几部分，推测他前面练了什么、怎么得出这个感受、暴露了什么不足。
2. **以不足为重点做交叉论证**：拿这些不足，对照知识库、FTT、TPA 等视频总结，写出逻辑推演过程（录像证据 + 教练原话 + 他自己的旧记录）。
3. **理清整条逻辑树**：始终把新发现挂回一以贯之的主线（见 §3 “蓄—带—刹—释放”）。
4. **最后抽出关键点反馈**：哪些和他的感想一致，哪些要换个说法、再验证。

### 2.2 禁止的句式

- “大方向对，但后半句要改”“对的部分 / 要改的部分”“结论：…”这类评审式开头。
- 一上来就给根因链、多 Block 训练计划。

### 2.3 哪些话要重复，哪些不要

- **每个练习方案前固定只提一句**：“准备时鼻吸，启动前挥至触球时用嘴短促呼气。”（用户 8/9 确认的节奏锚；呼气是节奏和张力阀门，不是发力来源。）
- **不要每次附上**：“不屏气、不憋劲；头痛/头晕/恶心就停；食指/跟腱/膝/髋疼就停”。用户已经知道。只在相关时提：他报告疼痛、练习含爆发或负重、出现新风险。用户原话：“你觉得正常教练会这样唠叨吗？”

### 2.4 其他风格要求

- 中文回答，先说结论性的感受再展开，但不要用工程化标题腔。
- 区分四类信息：**用户自述 / 画面可见事实 / 教练原话 / 推论**。单帧不反推唯一肌肉、主被动或击球早晚。
- 引用视频要给标题或 ID、具体时间段；没有可靠时间戳不要编。
- 场上 cue 遵守 **Intuition-First**：上场只给**一个外部目标**（通常是“球深落进目标区”），身体口令只用于影子挥拍/在家找体感。
- 用户报症状时，默认第一反应是问“球去哪了？目标是什么？”（项目 CLAUDE.md 规则）。
- 首次交付“训练视频分析报告”时，技术总结要写成**一个完整连贯的自然段**（项目 CLAUDE.md 规则）；之后的追问回到正常对话格式。

---

## 3. 当前技术主线（截至 2026-10-12）

### 3.1 一以贯之的逻辑树：蓄 → 带 → 刹 → 释放

这是 10/9–10/10 和用户一起整理出来、他认可的框架（用户原话：“蓄带刹这三段总结得很好。我之前没有刹车”）。

```text
蓄：髋臀加载，膝与脚尖成一条桩（9/23、10/1 已建立）
 ↓
带：身体先走，手臂贴着身体被带着走，不提前伸出去
    · 10/2 录像：手臂推一部分就横扫 → 这里断
    · 10/7 录像：拍子在远离身体处吊着落 → 也是这里断
    · 10/6 食指疼 → “带”断之后的末端症状
 ↓
刹：髋拉住 + 左手收回胸前，身体停在大致面朝球网（用户 3/21、4/4 旧记录早已有）
 ↓
释放：手臂和拍子被甩出去，先 Hit 后 Roll
```

用户 10/9 的体会：“身体在整个过程中都要参与，手臂只是一部分，甚至要把手臂的感觉盖过去。这样能解锁 effortless power。之前身体的力只传到 lag 之后那几毫秒，后面都是手臂。”
教练校准：方向成立；但**不是 lag 之后要更极度地扭转**——录像里躯干触球后 1–2 帧已背对镜头，扭转量够，缺的是**刹**。深扭发生在“蓄”，到“刹”时身体反而停住，手臂才被甩出。刹车**不用主动停躯干**，用户 4/4 记录的方法仍成立：左手收回胸口，身体自然停（FTT Power Checklist #5 “Pull the non-hitting elbow”）。

### 3.2 本会话按时间的发现链

| 日期 | 用户发现 / 问题 | 校准后的理解 | 状态 |
|---|---|---|---|
| 9/23 | 灵熙髋部隔离练习后：“膝盖也在转、脚也在动，力没传上去，所以手捞球” | 支撑腿做桩、骨盆在桩上转，脚是被转髋带走的结果 | learning.md ⭐，已 commit |
| 9/24–9/25 | 开放式站位脚尖/重心/臀 vs 腘绳肌感受；“力点在脚尖、碾烟灰、带膝盖，不协调” | 旋转发生在脚和地面而不是髋关节，力的方向水平。Tomaz：“hip is the boss，拖着脚” | 已回答 |
| 9/25 | 突破：“膝盖和脚尖一条线、相对固定，力量存在髋臀，释放由髋带动” | 与 Tomaz、加治遥右臀加载、7/17“坐进右臀”一致 | 药球练法已给 |
| 10/1 | 转髋三要点：①髋转非腿脚转、力存于臀 ②身体前倾 ③上身笔直水平不后带 | ②③合并为“躯干角度不变”；①是找体感用语，实打只盯落点；“三点做好其他自然到位”是待验证假设 | learning.md ⭐，commit `5ea8bef` |
| 10/2 | 录像：“手臂推一部分剩下扫过去；小臂代偿；触球时手远高于拍头” | 逐帧：手高拍头低出现在**触球后** 1–3 帧；触球后手臂横向往右伸；躯干触球后 1–2 帧背对镜头 | 见 §4.1 |
| 10/2–10/3 | 猜测：pivot 在拍底 vs V 区 → “没顶肘” → “漏了 shoulder adduction” → “只用大臂、胸带大臂，小臂被动” | 顶肘 3/6 试过已废弃，且禁止主动推/顶肘；HSA 顺序是先向外再内收（FTT `wFOy0RKWBTg` 原字幕）；“近端主动、远端被动”=用户 4/30 记录 | 已回答 |
| 10/6 | **食指承重**：lag 时拍子重量和角度压在拇指旁的食指上，食指疼；拍子下降时“彻底倒了”，手腕变形。用户认为关键是“以手腕为轴旋转球拍、手腕不变形” | 这是**第三次**出现（3/30、5/7 JUL）。修正：手腕背伸角度可被惯性**被动拉深**；要避免的是**被压塌/折向别处、靠食指当支点撬拍** | 待写入 learning.md |
| 10/7 | 新录像 3 拍 | 拍头落到 6 点时手臂已近乎伸直、远离身体，拍子吊在手上→长杠杆压食指；松尾要求“靠近身体时落下，落完再加速” | 见 §4.2 |
| 10/7（另一会话） | Overhand / Horizontal Half-Circle：近端稳定、前臂不抢跑 | FTT `PwkGLkRm73o`、`hqganBxzQNM`；Pose 数据：触球前约 0.1s 手腕垂直上移 78–107px、肘 41–60px、肩仅 8–14px | learning.md ⭐（未 commit 前由另一会话写入，本次一并同步） |
| 10/9–10/10 | “身体贯穿整个节奏”；承认“之前没有刹车” | 见 §3.1 | 待写入 learning.md |

### 3.3 当前优先级

1. **先修“带”和握拍接口**（最末端、最易验证：食指疼不疼）。
2. **再练“刹”**（后方机位就能看：触球后是否立刻背对球网）。
3. 下肢“蓄”已基本建立，保持即可。

### 3.4 已给出的下一次训练（不要再叠新 cue）

开头一句：准备时鼻吸，启动前挥至触球时用嘴短促呼气。

**练“带”（10/7 方案）**
- 在家影子挥拍 2 组 × 6：拍子靠近身体，放松，让拍头在**手伸出去之前、身体右侧**落下；检查拍头最低时手离右胯 ≤ 一拳。
- 发球机最低速 30 球（3 × 10）：唯一外部目标“球深落进目标区”；每组后自问“食指根有没有被压住？”；右侧/右前 45° 录第 2、3 组。
- 验证：侧机位上拍头最低点出现在手臂伸直之前；30 球后食指不疼；第 3 组 ≥ 6/10 进目标区。

**练“刹”（10/10 方案）**
- 在家影子挥拍 2 组 × 6：转到触球点时左手收回左胸前，胸口大致朝球网，停 1 秒。
- 发球机最低速 30 球（3 × 10）：外部目标同上；后方机位即可。
- 验证：触球后胸口停在面朝球网附近、左臂收在胸前未甩到左后方；第 3 组 ≥ 6/10 进目标区；食指不疼。

**另一会话（10/7）给的负重桥接**（与上面不冲突，可二选一）：10 球基线 → 两轮（0.5 kg 轻物慢挥 3 次 → 球拍空挥 3 次 → 5 球）→ 10 球保留测试；最后 10 球 ≥ 7 球声音扎实且落进目标区、食指不疼。食指疼时跳过负重；2 kg 药球只能双手抱慢转。

---

## 4. 录像证据（人工逐帧复核，未跑 Pose/KPI 的除外注明）

### 4.1 2026-10-02 `videos/favorites_merged_1790942683.542797.MP4`

- 1214×2160@60fps，17.4s，横存需 `transpose=1`；后方机位；4 拍（触球约 3.0s / 7.0s / 10.9s / 15.0s）。
- **触球帧**：球刚压上拍线时，拍头仍高于手，拍子约 30–45° 倾斜。
- **触球后 1–3 帧（30–100ms）**：拍子迅速放平，接着拍头掉到手下，手高于拍头（用户看到的就是这一段）。
- 触球时手在肚子前、肘弯着；**触球后**手臂才伸直，且向右侧横向伸出，不沿出球方向。
- 躯干触球后 1–2 帧已背对镜头。
- 不能判断：pivot 是拍底还是 V 区（需侧机位）；球落点。

### 4.2 2026-10-07 `videos/favorites_merged_1791362372.5078259.mov`

- 540×960@60fps，10.9s，横存需 `transpose=1`；后方机位；3 拍（约 1.3s / 5.0s / 9.0s）。
- **引拍**：拍子竖立身体右侧、肘在肋旁，稳定，没有提前躺下（比以前好）。
- **前挥前段**（约第 5–6 帧@30fps）：拍子翻成横躺腰后。
- **前挥中段（约 130ms）**：手臂几乎伸直、向右前方伸出，**拍子竖直吊在手下、拍头指地，球尚未到**（第 1 拍 60fps 放大第 5–11 帧最清楚）。
- **触球附近**：拍子从下往上、往左急甩，手抬高，躯干同时转背对镜头；随挥左臂甩向左后方。
- 与 10/2 比：手臂伸展更开（进步），但拍头下落位置从靠近身体挪到远离身体。
- 另一会话补充（Gemini + Pose，报告 `reports/2026-10-07_wrist_elbow_review/`，gitignored）：候选触球窗口约 F93–F96、F319–F322、F554–F557；手相对肩明显上升、肘角收小（124–136°→更小）。首轮 Gemini 触球时间偏早，第二轮经人工帧提示修正，不算两个独立验证。
- 不能判断：手腕是背伸被拉深（可接受）还是退回中立/屈曲（不对）；球的落点和深度。**10/7 最后问用户“这三拍球落在哪？深还是浅？”尚未得到回答。**

---

## 5. 知识来源与本会话核对过的视频

### 5.1 引用权威排序（项目 CLAUDE.md）

1. Tennis Science（Elliott/Reid/Crespo 2015）— `docs/research/tennis_science_book/MASTER_INTEGRATION.md`
2. HSA 框架 — `docs/research/hsa_master_index.md`
3. JUL Tennis & Golf — `docs/research/jul_tennis_videos/MASTER_SYNTHESIS.md`（reasoning 参考，不是新 cue 来源）
4. FTT（Johnny Kumpf）— 书 *The Fault Tolerant Forehand* + 网站 + YouTube
5. Brian Gordon（TennisPlayer.net）
6. Bourne（One Minute Tennis）
7. Kibler / Ellenbecker
8. 论坛（Reddit/TalkTennis）仅补充
9. 用户自身突破 — 与权威冲突时以权威为准

> 原则：FTT 是重要参考，但不能为了符合 FTT 改写其他教练原话或忽略反证。不同教练有矛盾时保留原话与情境，不强行统一。

### 5.2 本会话核对过的关键视频

**下肢 / 转髋**
- FTT 书 pp.84–87：脚的四种释放模式（右前掌 pivot 左脚抬；双脚跟抬；hop/kickback；双脚离地）；双脚钉死时扭力压进膝踝。
- FTT `ui-RFRdQKRk`（2026）：“the unit doesn't include your hips”；后侧臀加载后 pops；08:08–09:32 单前脚药球“NOT trying to throw… not use our hands… underhand release… twist, twist, and let it go”。⚠️ 旧书“先转髋、主动收臀”与 2026 说法（先感到髋即挥拍已散）矛盾，取新版。
- FTT `6ak4ZScH_gc`：“we perceive tension”（感知张力不感知运动）。
- TPA `IhLcK-ScJ1k`（脚跟自然抬、“碾烟头”、脚尖保持接触）；`RaRstufkeJ0`（后脚尖击球结束前不离地）；`e-KdPNRv9Ls`（高球 R→R，低球 R→L）；`azVf6CyDfVk`（开放式卡在右腿；3:09–5:43 上身引拍未完下肢已开始前移）。
- Feel Tennis `c9glLgadcas`（Tomaz，How To Get Power From Your Hips）：03:52–05:10 “foot is the boss”是错的，“hip should be the boss and drag the foot”；11:49–13:10 开放式髋活动空间约一半，stay on the right leg；09:00–10:30 搭档推手练习。**未入库**。
- Tennis Classic 加治遥 `gipNOFLO7BI`（打ち負けないフォアハンドは「お尻」で打つ，2026-09-11）：02:30–02:55 臀部比球稍低、向前并往 45° 抬；03:10–04:28 2 kg 药球右臀“グッと”加载，出手与臀部撞出同步，扔完右脚可被带出；05:49–08:31 保持前倾、目光留在击球点（⚠️ 针对她自己的腰椎过伸/后仰倾向，见 `XFdNxuVQzoo`）；08:00 自述 45° 是“斜め上”。仅字幕核实。**未入库**。
- 灵熙集训营髋部隔离练习（小红书 `6a145d60000000003503abcd`）— `docs/research/coaches/lingxi_camp/README.md`。

**手臂 / HSA / 释放**
- FTT `wFOy0RKWBTg`（原字幕已核，30s）：“that first vector is out… if I release the racket before I actually adduct my shoulder, the racket should not go towards the ball… that's a volley… then because I go out… I can now adduct my shoulder through the ball.” 丢拍实验：前挥刚启动松手，拍子应飞向右侧。
- FTT `Am8j1Zw5KrE`、`5KdScDKxVSI`（HSA；Gemini 稿 5/16）：内收本质横向，靠侧身角度让触球段变成“由后向前”；触球区是弧线上正对目标的一小段；无转体 HSA 纯化练习。
- FTT `wzLMv4ASg9s`《Why Your Racket Doesn't Flip》（Gemini 稿 8/13，三轮审计）：00:00–01:17 先向外；01:17–03:01 拍柄座进手上部；03:01–04:45 不要横拉肘过身体，肘内侧朝胸口方向（只作解释，不作主动 cue）；04:45–08:43 早期下三指松、近触球收紧、1:1 耦合；09:57–15:08 **Hit 先于 Roll**，费德勒触球时只用约 25% 雨刮；15:08–18:22 背部稳定（去耸肩、肩胛 in and down、等长支撑保持到触球）；28:21–29:49 侧向空间；**29:49–34:25 躯干减速脉冲，twist not spin，“hips hold the swing”，33:35 感受手被“卡住”释放自己发生**。
- FTT `pWzyP-xfLfU`（Pivot Point）：轴心在 V 区时拍头后摆伴随拍底前移；用户 6/5 球场验证。
- FTT `PwkGLkRm73o`、`hqganBxzQNM`（另一会话 10/7 核验，见 learning.md 10/7 条目）。

**手腕 / 握拍 / 拍头下落**
- 松尾友貴 `xYSGZkU5si4`《ストロークは「手首」が9割》（2026-04-26；Gemini 2026-10-06 + 日文字幕）：①手腕柔软、前臂不用力、手臂“使わされてる”，拍头因脱力而落，6→9 点雨刮，只在触球时握紧；②肩的转动，**不要拉右肩，右肘一直在身体前**；③**身体别转过头**，“正面向いて止まる”，前脚臀部蹬起/开放式蹬轴心脚，“ピタッ”停住，头固定。⚠️ 雨刮时机与 FTT 25% 说法矛盾，保留原话。**未入库**。
- 松尾友貴 `3I0LWgIIUgg`《フォアの基礎 3つの鉄則》（Gemini 2026-10-06）：00:54–01:17 击球前拍头必须落下（ヘッドダウン）；02:00–03:30 拍子靠近身体架、放松，拍头落到 6 点；06:23 **拍头完全落下之后才加速**。两期都**没提**食指/虎口，也**没说**“手腕形状不变”。**未入库**。
- Intuitive Tennis `s7n6kSkLeCk`《How to Get Wrist Stability for More Forehand Power & Control》（Gemini 2026-06-17）：01:43–01:57 错误示范=拍头小臂过早下落躺在身后、手腕回中立→触球补 slap；02:00–03:50 肘稍架起、手相对低，“By bringing the elbow up, you create this angle in the wrist”；“It doesn't have to be that rigid”。用户 6/20 定为“肘轴引拍圣经”。⚠️ 引用的分析文件 `docs/research/coaches/intuitive_tennis/s7n6kSkLeCk.md` **不存在**。
- JUL `zf6dBOucjqg`：“If the weight is on the index finger, you will get early internal rotation.” 承重应在小指侧鱼际（hypothenar）。见 `docs/research/jul_tennis/videos/deep_mechanism_synthesis.md:51`。
- Tom Allsopp `-JU73Jxxj9c`：引拍顶点有意识松开食指、中指压力。

### 5.3 用户自己的旧记录（回答时要主动连起来）

| 主题 | 位置（learning.md 约行号） | 内容 |
|---|---|---|
| 刹车 | 3/21（≈697–770） | “转到 45 度就刹车，左手收胸”；症状往前扑 |
| 刹车机制 | 4/4（≈1925–1968） | 鞭根停住鞭梢加速；“你不需要主动停躯干，你需要做的是左手回收”；正确冻结 vs 根本没转 |
| 食指疼 | 3/30（≈1354） | “V 型区卡住→搭上，中指环住”；食指在主动顶 |
| 顶肘 | 3/6 → 3/13 | 试过“顶肘”后废弃；3/13“整条胳膊是一根鞭子，身体转它就跟着走”（`docs/现代正手深度解析.md:189,199`） |
| 近端/远端 | 4/29–4/30 | 大臂住肩窝（约束）+ 胸推驱动、小臂被动（`docs/research/FOREHAND_COMPLETE_TAXONOMY.md:530–560`） |
| V 区 pivot | 6/5（≈8820） | 轴心从拍底移到 V 区，降拍头自然发生 |
| 肘轴引拍 | 6/20（≈9890） | Intuitive Tennis s7n6kSkLeCk |
| 坐进右臀 | 7/17（≈10410–10428） | 臀加载、前大腿松、脚跟压实 |
| 拧脚伤跟腱 | 7/14（≈10267–10277） | 刻意拧脚 → 跟腱疼 |
| Drop 主体 | 8/28 | 背托框架整体下沉，拍头只相对落后，禁止吊腕 |
| ESR 时序纠错 | 9/20（≈11729） | Hide the Elbow → Show the Shoulder；废止第一帧主动 ESR |

---

## 6. 永久规则与禁令（来自项目 CLAUDE.md，摘要）

- **禁止主动 cue**：推肘 / 顶肘 / 送肘 / 胸推肘（替换为“蹬 + 托”“信任刚体”）；回答涉及肘时不用“推、送、推前”等主动动词。肘只作结构检查（如“手低于肘”）。
- **ESR（2026-09-20 覆盖旧协议）**：禁止 Unit Turn 第一帧主动 ESR；禁止刻意收冈下肌/小圆肌锁 IR；禁止主动前臂旋后、主动 drop、主动推肘；不用肩后侧/小圆肌酸痛当做对的标准。回答 ESR/IR/球软/大臂飘/后倒/节奏断时，先查 Unit Turn/Takeback 是否完整、手是否在身体右侧、转换期是否身体先行产生动态 lag。
- **HSA**：是发力体系的物理本体（press slot / chest fire / 撕 / 横拉 等都是它的不同视角）。项目 CLAUDE.md 要求回答 HSA 相关问题前先问“HSA 体感今天到位没？”——但须服从 Intuition-First，不变成新口令。
- **Intuition-First（最高原则）**：Reason 定方向，Intuition 执行。只有三种情况给 reasoning-heavy 回答：用户明确要求理论解释；同一失败模式重复 ≥ 3 次；用户问“应该 aim 什么目标”。其他情况短回答（< 200 字），让直觉做梯度下降。
- **糙男框架**（B 站 RacketBrothers BV1Hr4y1p7Sx）：指高 → 指肘 → 出手（拍头**经过**胸前那个点）；外部聚焦。
- 区分主动动作 vs 被动结果；容错性优先；允许动态稳定，不把支撑理解成锁死所有关节。
- 训练方案单位：**球数 + 分组 + 验证点**。
- 教练模板学组织原则，不要求每帧 1:1 复刻；上传更晚不代表动作更好。
- 尾崎里紗/尾﨑里紗（Risa Ozaki）与 Vicki、Vekic、Vic Braden 不混同。
- **learning.md 写入规则**：每次新增 entry 必须同步顶部时间轴（mermaid 图 + 阶段表 + 关键 entry 索引；⭐/🏆/⚙️ 级要加链接 + 行号；顶部“状态更新日”改日期）；commit message 必须写明时间轴同步。未经用户同意不写。

---

## 7. 工具与工作方式

- **视频分析**：
  - YouTube 字幕：`yt-dlp --skip-download --write-auto-sub --sub-lang ja|en --sub-format vtt -o /tmp/<id> <URL>`。经常 HTTP 429，单语种间隔重试；视频下载常被 bot 检查拦截。
  - Gemini 原生视频（项目根运行）：`python3 scripts/youtube_gemini_analyze.py "<URL>" --prompt "<中文提示>"`，输出自动存 `~/.gemini/transcripts/`。用户反感浪费 Gemini/token：先查本地已有稿。
  - 小红书：先用 agent-reach / OpenCLI + 本地 whisper；不替用户登录、不读浏览器 Cookie。
- **本地录像**：`ffmpeg -i <file> -vf "transpose=1,crop=W:H:X:Y,fps=30,scale=520:-1,tile=5x3" -frames:v 1 out.jpg` 做联系表（本机 ffmpeg 无 drawtext）。完整流程有 Pose/KPI（`core/`、`evaluation/`）+ Gemini + 人工复核 + 报告（`reports/`，gitignored）。
- 临时文件放 `/tmp/`。不在聊天或文件中暴露任何 API key/token。

---

## 8. 未完成事项

**等用户回答**
- 10/7 三拍球落在哪？深还是浅？

**待用户批准的写入（逐项问）**
- [ ] learning.md 新 entry：10/6–10/10“食指承重 → 拍子在远离身体处吊落 → 蓄带刹”，链接 3/21、3/30、4/4、5/7 JUL、6/20、松尾两期、10/2 与 10/7 录像；标“待侧机位验证”；同步时间轴。
- [ ] 补建 `docs/research/coaches/intuitive_tennis/s7n6kSkLeCk.md`（基于 6/17 Gemini 稿，注明单轮、时间戳未核字幕）。
- [ ] 归档松尾两期（建议 `docs/research/coaches/matsuo_tomoki/`），注明与 FTT 雨刮矛盾。
- [ ] 归档 `gipNOFLO7BI`（`docs/research/coaches/web_tennis_classic/`）、`c9glLgadcas`（`docs/research/feel_tennis/video_analyses/`）。
- [ ] 可选：10/7 录像补右侧机位对照复测后再升级结论。

**其他未提交、与本线无关的改动**：工作区有大量 `.agent/`、`.planning/`、`books/`、`docs/research/...` 的修改和删除，来源不明，**不要动**。

---

## 9. 文件地图

| 路径 | 用途 |
|---|---|
| `CLAUDE.md` | 项目规则全文（必读） |
| `docs/record/learning.md` | 训练日志 + 顶部时间轴（权威） |
| `docs/record/context_export/2026-10-12_grokbot_context.md` | 本文件 |
| `docs/research/hsa_master_index.md` | HSA 主索引 |
| `docs/research/esr_root_cause_master.md` | ESR 主文档（先读顶部 9/20 覆盖声明） |
| `docs/research/FOREHAND_COMPLETE_TAXONOMY.md` | 正手分类总表（含近端/远端边界 :530） |
| `docs/research/ftt/` | FTT 书、博客、视频分析（`video_analyses/<id>.md`） |
| `docs/research/ftt/20_ftt_grip_rotation_axis.md` | V 区 pivot、食指边界 :193 |
| `docs/research/jul_tennis/videos/deep_mechanism_synthesis.md` | JUL 食指/鱼际开关 |
| `docs/research/medicine_ball_protocol.md` | 药球协议（§10 家用不投掷） |
| `docs/research/coaches/web_tennis_classic/2026-09-10_cross_channel_review.md` | 尾崎/加治专题入口 |
| `docs/research/coaches/lingxi_camp/README.md` | 灵熙集训营 |
| `docs/research/COACH_VIDEO_HALL_OF_FAME.md` | 殿堂级视频索引 |
| `docs/research/tennis_science_book/` | Tennis Science 八章 |
| `docs/filming_protocol.md` | 拍摄机位协议（gitignored） |
| `videos/` | 用户训练录像（gitignored） |
| `reports/` | 分析报告（gitignored） |
| `~/.gemini/transcripts/` | Gemini 视频分析原稿（本机） |

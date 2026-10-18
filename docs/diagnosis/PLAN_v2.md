# PLAN v2：FTT YouTube 全量拆解 + 可验证的诊断闭环（2026-10-18）

> 范围：**只做 FTT YouTube**（TPA、Bourne 等先不管）。目标：把 FTT 每一期「一个常见问题 → 根因 → 多角度讲解 → 结合根因的解法」拆成结构化卡片，挂到一棵**有约束**的诊断树上，并用**可重跑的脚本**证明「来一段新视频，树能找出问题」。
> 本轮新增：本文件 + `pipeline/`（schema、校验脚本、提示词、自检样例）。**没有改** `faults/`、`sources/`、CLAUDE.md、AGENTS.md、learning.md；没有 git 操作；不含任何密钥。
> **状态说明**：`pipeline/` 的脚本只做了逐行人工核对，**还没有运行过**（本会话执行 Python 需要授权）。执行顺序第 0 步就是跑自检。

---

## 0. 一句话

现在的树是「把相关内容找出来贴在一起」，没有任何一条规则能判定它对还是错。v2 做三件事：
1. **字幕定框架**：每期的问题、根因、解法都必须落到字幕原话和时间戳上，脚本能核对。
2. **视频补画面**：Gemini 看视频只负责补「画面里看到了什么」，不能改字幕里说过的话。
3. **盲测**：拿 FTT 视频里**教练亲口指出示范者问题**的片段，把声音去掉，只给中立事实，看树能不能自己找回那个问题。

能召回 → 证明树能看出问题；召回不了 → 改树，不是加概念。

---

## 1. 现状：现有树为什么是「只检索、没细化约束」

| # | 缺口 | 具体位置 / 证据 | 后果 |
|---|---|---|---|
| 1 | **树是散文**：节点和边写在 mermaid 和 md 里，没有机器可读的 id 和字段 | `tree.md`、`faults/FLT-*.md` | 无法自动检查任何一条边有没有证据、两个节点能不能区分 |
| 2 | **索引是一对多的「点亮」**：一条观测会点亮 3–4 个故障，只有 +/− 方向，没有「必需 / 支持 / 排除」，也没有权重 | `index/obs_to_fault.md`：例如 OBS-02「上臂贴肋」同时 + FLT-01/02/08/10 | 任何视频都能「对上」好几个故障，等于没有诊断，只是把相关内容检索出来 |
| 3 | **鉴别只写在文字里**：FLT 卡第 3 节有「易混淆」，但没有规定用哪个观测值区分 X 和 Y | FLT-01 §3「易混淆：够过头」 | 两个故障签名相同时，系统无法给出排序 |
| 4 | **证据不可核对**：成因边写「书 p.X【已核对】」或「视频 mm:ss【未核对】」，没有可被脚本核对的原话和时间戳；FTT 视频**尚未挂接** | `tree.md` 版本记录「尚未挂接 FTT 视频」；FLT-01 §4 视频部分「全部【未核对】」 | 错误的时间戳、旧笔记里编出来的细节会被固化（PILOT §4：hqgan 旧笔记里的大量器材和重量数字，字幕里都没有） |
| 5 | **ABC 分层没有被强制**：OBS 卡写了可靠度，但诊断时仍可以拿 C 层（体感）或不可靠的 B 层观测去定罪 | `observables/OBS-02`「R 机位下判为不可靠」，但 obs_to_fault 仍把它列为 FLT-01 主要证据 | 单机位视频得出过度自信的结论（10/17 的教训） |
| 6 | **前提没有做成门槛**：FTT 每期的「默认前提」（例：q6 针对已能做到胸下压、髋稳定的球员）只写在 SRC 卡里，诊断时不检查 | `SRC-FTT-q6LrQYDvxmY.md` §3 | 处方被套到前提不成立的人身上 |
| 7 | **从没被测试过**：没有任何「已知答案」的样例去测树的召回率和误报率 | 整个 `docs/diagnosis/` 没有回放集 | 不知道树准不准，可能只是「更有条理地出错」 |
| 8 | **节点来自书末错误表 + 用户近期问题**，不是来自 FTT 每期的「一个问题」 | PLAN.md B3（10 张卡的选取依据） | FTT 自己的问题划分（标题 = 一个常见问题）没被用作骨架 |
| 9 | **抽取标签不可靠且没有字段约束**：Gemini 标的「学员问题」0 条是真的具体学员；没有「谁在说 / 是不是故意示范」字段 | PILOT §2 | 无法从视频里可靠地找到「已知答案」的片段 |
| 10 | **校验靠自述**：试点的核对率（q6 81%、hqgan 63%）是人工逐条找的，不能重跑；提示词版本、模型、字幕来源也没有被检查 | PILOT §3 | 下一批没法用同一标准复核 |

一句话：**缺的不是内容，是约束和验收**。v2 把以上 10 条全部变成 schema 字段或脚本规则（见 §4 的对照表）。

---

## 2. 任务定型

### 2.1 两条路并行，分工固定

| 层 | 来源 | 负责什么 | 权威范围 |
|---|---|---|---|
| **字幕层（Stage 1）** | 字幕/转写 + 时间戳（另一位执行者负责抓取，输出到 `/workspace/ftt_subs/<ID>.*`；优先 json3/vtt，失败时用 whisper） | 问题定义、根因链、多角度讲解、解法：**教练说了什么** | **以字幕为准**。所有成因、解法必须引用字幕原话 |
| **视频层（Stage 2）** | Gemini `gemini-3.8-flash`，YouTube 链接直传 `fileData.fileUri` | 画面细节、示范者实际做出的错误、屏幕文字：**画面上有什么** | 只对「画面」有权威；不能新增或改写「教练说过的话」 |

### 2.2 产物与目录（执行者创建；本轮不建）

```
docs/diagnosis/
├── PLAN_v2.md                 # 本文件
├── pipeline/                  # schema + 校验 + 提示词 + 自检样例（本轮已建）
├── episodes/<ID>.card.json    # 每期一张卡（只存 ≤40 词的原话摘录，不存整份字幕）
├── episodes/raw/<ID>/         # s1_run1/2.card.json、s2_run1/2.json、渲染后的提示词、调用日志（不含密钥）
├── ftt_tree/tree.json         # 诊断树 v1（机器可读，是唯一的树）
├── ftt_tree/views/            # 由 tree.json 生成的 md/mermaid 视图（不手改）
├── replay/cases/*.json        # 盲测用例（已知答案）
├── replay/facts/*.json        # 静音片段的中立事实（每片段 3 次）
└── loop/manifest.json         # FTT 频道全部视频的状态机
    loop/reports/verify_*.json # 每轮 verify_all 输出（证据，不是自述）
```
片段和完整字幕不入库（版权 + 体积），只存路径和 sha256。

### 2.3 与现有文件的关系（先减后加）

- `faults/FLT-01…10`、`index/*.md`：**冻结**，不删、不改。`tree.json` 的节点用 `flt_ref` 指回 FLT 卡。树 v1 跑通后，是否把 FLT 卡降级为「由 tree.json 生成的视图」，由用户决定（§9）。
- `sources/SRC-FTT-*.md` 三期试点卡：保留作历史记录；这三期在循环里**按 v2 流程重新抽取**，不做手工转换。
- 旧笔记 `docs/research/ftt/video_analyses/`、`tennis_foundation_series/.../video_analyses/`：**不作来源**（PILOT §7.6）。
- `prompts/fact_extract_v1.md` → 由 `pipeline/prompts/facts_v2.md` 取代（静音、选择题从 tree.json 渲染、3 次）。
- **不再**多模型投票，**不再**新写自由格式的来源卡。

---

## 3. Episode card：标准抽取 schema

完整定义：`pipeline/schema/episode_card.schema.json`。字段要点：

| 区块 | 内容 | 关键约束 |
|---|---|---|
| `video` | ID、URL、yt-dlp 标题、频道（只能是 Fault Tolerant Tennis）、时长、发布日期 | 元数据只认 yt-dlp / API |
| `classification` | `is_teaching`（teaching / partial / non_teaching）、击球类型、形式（讲解 / 带示范球员 / 课程点评 / 职业分析 / 短片 / 其他）、理由 | non_teaching **不建卡**，只在 manifest 记理由 |
| `transcript` | 路径、类型（official / auto / whisper）、sha256、覆盖率 | 覆盖率 < 0.9 不能通过；sha 变了必须重核 |
| `extraction` | Stage 1 模型、提示词版本、**≥2 个独立运行 id**；Stage 2 同理；`user_context_in_prompt` 只能是 false | 没有两次独立运行就没有一致性证据 |
| `problem` | 标题改写成的问题、定义、**前提**、目标球员、`node_id`（挂树前为 PENDING）、证据 | 前提在诊断时作为门槛使用 |
| `quotes` | 教练原话：英文逐字 ≤40 词、开始秒数、说话人、**about**（泛指 / 示范者自然出现 / 示范者故意做 / 职业片段）、来源（subtitle / video） | 其他所有字段都只能通过引用 quote id 取证；Stage 1 只允许 subtitle 来源 |
| `symptoms` | 用户视频里能看到或感到的现象：类型（画面可见 / 球的结果 / 体感）、**层级 A/B/C**、机位、阶段、`obs_ref` | 体感必须是 C；画面可见必须是 A/B，并写明机位和 OBS |
| `causes` | 根因：陈述、**深度**（1 = 直接原因，越大越上游）、层级 A/B/C、`node_id`、主动 / 被动 | 每条至少 1 句**字幕**原话 |
| `chain` | 因果边 from → to（原因 → 原因 / 症状 / PROBLEM），强度（教练明说 / 隐含 / 推断） | 往下游深度必须递减；每条原因都要接进链；链必须到达问题或症状 |
| `angles` | 多角度讲解：机制 / 感觉 / 视觉模型 / 比喻 / 对照示范 / 纠正误区 / 职业例子 | 每条有原话 |
| `solutions` | 练习 / 口令 / 检验 / 进阶：针对哪条原因（`targets`）、外部目标一句话、场景适配（低速发球机 / mini / 影子挥拍 / 需要喂球人 / 只能比赛）、**禁令冲突标记**、是否符合 Intuition-First | 文本里命中「推肘 / 降拍 / 旋后 / 伸肘」而没有标记 → 失败；有标记的不能作为可用处方 |
| `video_layer` | Stage 2 补充：画面细节 / 示范者可见错误 / 修正建议 / 新内容，带时间、两次运行一致数、状态 | Stage 1 时必须为空 |
| `conflicts` | 字幕 vs 视频（或书 vs 视频）的冲突记录与裁决 | 见 §3.2 |

**A / B / C 层定义**（沿用 10/17 与试点的实测）：
- **A**：普通手机视频里，模型能稳定判断有/无（拍数、引拍顶点有没有停顿、有没有调整步、站位大类、肘明显弯还是接近直、口播结果）。
- **B**：必须用指定机位（S 正侧面 / B 正后方）+ Pose 或抽帧才看得准（角度、距离、先后顺序、触球点前后、躯干倾角）。
- **C**：视频看不到，只能靠体感或 10 球测试（放松、握力、重心感觉、肌肉用力）。

### 3.1 两阶段规则：以谁为准

1. **先跑 Stage 1**，通过全部校验后状态才是 `stage1_verified`。之后才允许跑 Stage 2。
2. Stage 2 跑 2 次，**两次都出现**的条目才能 `accepted`（`runs_agree` ≥ 2/2）。
3. `visual_detail`、`demo_visible_fault`：一致即可 accepted，作为 B 层证据使用。
4. `correction_proposal`（认为字幕层某条与画面不符）：**不改**原条目，必须写进 `conflicts`。凡是涉及「教练说了什么」的，裁决一律 `subtitle_wins`；只有纯画面事实（例如示范者实际是开放式站位）才能 `video_accepted_visual_only`。
5. `new_claim`（画面里出现字幕没有的教学内容，例如屏幕文字）：最多是 `proposed`，进入人工复核；**不能**作为成因边的唯一证据（脚本规则 R11）。
6. FTT 书与视频冲突时两者都记（`book_vs_video_both_kept`），交用户裁决，不自行取舍（沿用 README §1）。

### 3.2 一期视频如何变成「具体问题」（统一读法）

FTT 每期的结构固定为：标题 = 一个常见问题 → 拆解根因 → 多角度讲解 → 结合根因的解法（解法同样从多个角度、针对不同层的根因）。「学员」指一对一课里的那一名示范球员，一期一般只有一个人；schema 里统一记作 demonstrator，不要求多人。拆的时候只问 5 个问题，与 schema 一一对应：
1. 这期**解决什么问题**？前提是什么？（`problem`）
2. 问题出现时，**在一段普通球员视频里能看到什么**？哪些只能感觉到？（`symptoms` + A/B/C）
3. 教练认为**为什么**会这样？原因的原因是什么？（`causes.depth` + `chain`）
4. 他**从哪几个角度**讲？（`angles`）
5. 每个解法**针对哪一层原因**？能不能在低速发球机上做？和禁令冲不冲突？（`solutions.targets` / `scenario_fit` / `ban_conflict`）

---

## 4. Verify 闭环（核心）

### 4.1 原则（对应 Pstack 的理念）

- **Prove it works**：每一轮是否通过，只看 `verify_all.py` 写出的 `loop/reports/verify_*.json`；执行者汇报时**贴这个输出**，不写「我觉得都对」。
- **Test the tester**：`verify_all.py` 每次先跑自检：好样例必须通过，22 个故意做坏的变异必须分别被对应规则抓到。校验器自己坏了，整轮就是 FAIL。
- **Encode lessons in structure**：每个已知教训都对应一条脚本规则（见 §4.3）。以后出现新的失败方式，**先加规则和变异样例，再继续循环**。
- **Subtract before add**：一棵树（tree.json）、一个视频模型、一套事实提示词、一个总判命令。

### 4.2 脚本清单与通过标准

| # | 脚本 | 检查什么 | 通过标准 |
|---|---|---|---|
| V0 | `verify_all.py --selftest` | 好样例全过；22 个变异全部被抓；diagnose 两次输出完全相同；C 层事实被丢弃 | 全部满足，否则整轮 FAIL |
| V1 | `validate_cards.py` | schema + R01–R12（§4.3） | 0 FAIL |
| V2 | `check_quotes.py` | 每句字幕原话在字幕里存在（词序 LCS 比例：官方字幕 ≥0.85，自动字幕 / whisper ≥0.75）；位置在所标时间 ±10 s 内；不超过视频时长；字幕 sha 未变 | 0 FAIL |
| V3 | `check_tree.py` | 证据可解析且所在卡已核对；不引用 holdout；签名用的观测项存在、取值合法、必需项不含 C 层；每个必须纠正的节点至少 1 个必需观测；签名相同的节点之间必须有鉴别项；`core` 边需 ≥2 期证据；无环；已核对的卡都已挂树 | 0 FAIL（WARN 进报告） |
| V4 | `check_rerun.py cards` | 同一期两次 Stage 1 抽取：按字幕时间锚（30 s 桶）比较原因、链根、解法的重合度（Jaccard）；是否教学视频的判断一致；问题节点一致 | 原因 ≥0.6、链根 ≥0.5、解法 ≥0.5；不过 → 跑第 3 次 + 人工裁决，卡里注明 |
| V4' | `check_rerun.py facts` | 同一片段 3 次事实抽取，每个观测项的一致率 | ≥2/3 才算可靠；结果回写 tree.json `observables.reliability` |
| V5 | `blind_replay.py` | 盲测回放（§4.4）：召回率、首位命中、阴性对照误报、重跑稳定性 | 门槛 G1 / G2（§4.4） |
| V6 | `check_neutrality.py` | 发给模型、看球员视频的事实提示词（渲染后）不含 FTT / 项目术语 / 用户信息；抽取提示词不含用户信息 | 0 FAIL；**不过不许发送** |
| V7 | `check_manifest.py` | manifest 的状态与真实文件一致（例：写 merged 就必须有树节点引用它；有卡必须在 manifest 里；non_teaching 必须有理由且不能有卡） | 0 FAIL |

阈值都在 `pipeline/config.json`，是**初始值**。调整需用户同意，并记入 §10 变更记录。**禁止**为了让某一期通过而放宽阈值：字幕核对不过，就回去改卡。

### 4.3 教训 → 规则对照

| 已知教训（出处） | 规则 |
|---|---|
| 提示词写进用户结论会把模型带偏（learning.md 10/17 方法纠偏） | schema `user_context_in_prompt=false`；V6 N01 |
| Gemini「原话」会被顺滑、加 gloss（PILOT §3、§6.4） | R06 原因必须有字幕原话；V2 Q04 |
| 时间戳有时标在段落起点、偏差 30–60 s（PILOT §3） | V2 Q03 ±10 s |
| 「学员问题」标签不可靠；示范多是教练故意喂出来的（PILOT §2） | `quotes.about` 四分类；R10；盲测标签必须是 demonstrator_* 的已核对原话（B01） |
| 旧笔记有大量视频里没有的细节（PILOT §4） | 证据只能是 Q/V/BOOK；R04 / T03 |
| 单次调用召回不全、标签不稳（PILOT §3 call2 vs call3） | Stage 1 ≥2 次运行；V4 |
| 单机位下模型对角度、距离、早晚的判断互相矛盾（10/17） | A/B/C 分层；R08；T04；diagnose 只用 A 层和合格的 B 层 |
| 偶发动作被当成习惯（00_principles P9） | diagnose：出现拍数 <80% 的事实不用 |
| 肘弯曲程度是风格（书 p.66、p.69–70） | 节点 `kind=style` / `must_fix=false`，diagnose 只列出、不排名 |
| 禁令（推肘 / 降拍 / 旋后 / 伸肘） | R09；T06；diagnose 只输出节点的 `next_one_thing`（已过禁令扫描） |
| 二轮「先给帧图提示再修正」不算独立验证（learning.md:11976） | 盲测事实来自静音片段 + 中立提示词，模型从不看到答案（B02） |
| 字幕抓取可能失败、被替换（PILOT §1） | `transcript.kind` + sha256；Q01；覆盖率 R12 |

### 4.4 盲测回放：怎样证明「来一个新视频能看出问题」

**用例从哪来**：FTT 视频里，教练常对着示范球员说「看他这里……」。字幕层把这类原话标为 `about=demonstrator_natural/induced`；视频层把画面上确实可见的标为 `demo_visible_fault`。这些片段的「正确答案」来自**教练本人**，而不是我们的推断。

**做法**：
1. 每个用例取该原话附近的片段（建议 [t−6 s, t+10 s]），**去掉音轨**（否则教练的话直接泄露答案）。
   - 字幕层和视频层（Stage 1/2）**都不需要下载原视频**：字幕走 yt-dlp / agent-reach，Gemini 走 YouTube 链接。只有盲测需要片段。
   - 首选：只下载这十几秒（`yt-dlp --download-sections "*t1-t2"`），用 `ffmpeg -an` 去音轨后 inline 发送。不下载整期视频。
   - 退路（完全不下载）：YouTube 链接 + `videoMetadata.startOffset/endOffset` 截段，并在提示词里要求「只看画面」。模型仍然听得到教练说话，**会泄露答案**，所以这类结果在 facts 里必须标 `audio_stripped=false`。B02 会让这些用例 FAIL，它们**不计入 G1/G2**，只能当参考。
2. 用 `facts_v2` 中立提示词跑 3 次，得到事实表。提示词里只有 tree.json 的中立观测题，不出现任何节点名、术语。
3. `diagnose.py`（确定性，不调用模型）用 tree.json 给出 top-3。
4. 命中：教练标注的节点（或其直接上游）出现在 top-3。
5. **阴性对照**：教练说「这一球做得好」的片段，或职业球员片段，期望 `NONE`。如果给出高置信故障，记为误报。
6. **Holdout**：按 `sha1(video_id) mod 5 == 0` 预先固定约 20% 的期作为保留集（在抽取前就定，避免挑选偏差）。树**不得引用**这些期（T02），它们只用来出题，模拟「从没见过的新视频」。holdout 规则目前由执行者计算，还没有脚本自动核对（待加 V8）。

**门槛**（`config.json`，初始值）：

| 门槛 | 时机 | 条件 |
|---|---|---|
| **G1** | 前约 15 期完成后 | ≥10 个用例、来自 ≥5 期；recall@3 ≥0.6；有阴性对照，且高置信误报 ≤20% |
| **G2** | 全部完成后（=「树能用」的验收） | ≥30 个用例、≥15 期；recall@3 ≥0.7；top-1 ≥0.4；高置信误报 ≤15%；重跑 top-1 稳定 ≥80% 的用例；以 holdout 为主报告 |

**局限（要说清楚）**：
- 示范者的错误常是故意做的，比用户的真实动作更夸张，所以盲测会**偏乐观**。
- FTT 视频机位不是我们规定的 S/B 机位。
- 因此 G2 通过，只说明树和事实层在 FTT 素材上可用。最终仍要在用户按规范拍的视频上校准（§6 第 9 步）；如果有真人教练的标注，那是最好的「答案」。

### 4.5 diagnose 的输出（给用户报告用）

`diagnose.py` 输出：
- 最多 3 条根因，每条带：证据（哪个观测值、来源、一致率）、缺什么证据、什么情况会推翻它、10 球确认/排除测试、来源（期 ID:原话 id）。
- 风格项（列出但不纠正）。
- 被丢弃的事实及原因。
- 下次训练：开头固定呼吸提醒（**准备时鼻吸，启动前挥至触球时用嘴短促呼气**）+ 只改一件事。置信度为低时不给「改一件事」，只给测试或补拍。

排序规则：置信度 → 同时出现的候选里**更上游的优先**（FTT「治因不治症」）→ 匹配分。匹配是纯规则，同样输入永远同样输出，所以结果的波动全部来自事实层，可以被 V4' 单独量化。

首次交付给用户的总结仍按 CLAUDE.md 写成一整段；这份结构化输出作为附件。

---

## 5. 循环（loop）设计

### 5.1 状态机（`loop/manifest.json`）

`listed → subs_ready → stage1_verified → stage2_verified → merged`；旁路：`non_teaching`（有理由、无卡）、`blocked`（有理由，例如字幕和音频都拿不到）。
manifest 的完整视频列表来自字幕执行者的 coverage 清单（`yt-dlp --flat-playlist`）。**FTT 频道一共有多少期我不知道**，以那份清单为准。

### 5.2 每一轮做什么（一轮 ≤5 期）

1. 从 manifest 取下一批 `subs_ready`（按发布日期从旧到新，holdout 期照常做卡）。
2. **Stage 1**：每期用 `ep_s1_v1` 独立跑 2 次 → `check_rerun.py cards`。不过就跑第 3 次，人工按字幕裁决，并在卡里注明。
3. 合并成一张卡 → `validate_cards.py` + `check_quotes.py`。有 FAIL 时回字幕改卡，**不放宽阈值** → 状态 `stage1_verified`。
4. **Stage 2**：`ep_s2_v1` × 2 → 合并进 `video_layer` / `conflicts` → 再跑 V1+V2 → `stage2_verified`。
5. **挂树**（只用非 holdout 期）：问题和原因映射到已有节点或新建节点；新节点必须写签名（必需 / 支持 / 排除观测）和与相似节点的鉴别项 → `check_tree.py`。合并建议写成 diff，每 10 期人工抽查一次。
6. 每满 10 期：从 holdout 期出盲测用例（片段 + 3 次事实）→ `blind_replay.py`，记录指标走势。到约 15 期时跑 G1。
7. `verify_all.py ... --report loop/reports/verify_<时间>.json`。**只有 VERIFY: PASS 才提交**。commit message 包含 VERIFY 行、新增期数和盲测指标。更新 manifest。
8. 本轮遇到新的失败方式 → 先在 `verify_all.py` 加规则 + 变异样例，自检通过后再进下一轮。

### 5.3 停止条件

manifest 里所有视频都是 `merged`、`non_teaching` 或 `blocked`（有理由）；G2 通过；最后一次 verify_all PASS。G2 不过时**不算完成**：按失败用例回头改签名、鉴别项或观测项，再测。

### 5.4 G1 / G2 不过时怎么改（按优先级）

1. 看失败用例的 `dropped_facts`：是不是关键观测一直「不可靠」？→ 改观测题的问法、补 Pose，或把该项降为 C 层。
2. 两个节点总是互相抢第一 → 补鉴别项，或合并成一个节点。
3. 节点签名太宽（误报高）→ 加排除条件。
4. 最后才考虑新增观测项。**不新增概念**。

---

## 6. 建议执行顺序（交给执行者）

| 步 | 内容 | 谁 | 预计工作量（估算） | 验收 |
|---|---|---|---|---|
| 0 | 跑 `python3 pipeline/verify_all.py --selftest`；有失败先修脚本（本轮**没有运行过**） | 执行者 | 0.5–1 h | 输出 `selftest: … PASS` |
| 1 | 建 `loop/manifest.json`：导入字幕执行者的 coverage 清单；按 sha1 规则标 holdout；列出明显的非教学视频（只按标题和时长初判，最终以 Stage 1 判断为准） | 执行者 | 1 h | V7 PASS |
| 2 | 建 `ftt_tree/tree.json` 骨架：把现有 OBS-01…18 转成 `observables`（中立问法 + 取值 + A/B/C + 机位；可靠度先填 `untested` 或 10/17 数据）；节点先为空 | 执行者 + 人工抽查 | 2–3 h | `check_tree` schema 通过（空树允许） |
| 3 | 试跑 3 期（q6LrQYDvxmY、hqganBxzQNM、ui-RFRdQKRk，按 v2 重抽），走完 Stage 1 → Stage 2 → 挂树 | 执行者 | 3–4 h | 3 张卡 V1/V2/V4 PASS；与 PILOT 的人工核对结果对比，差异写进报告 |
| 4 | 按 §5.2 循环，一轮 ≤5 期，每轮一个 commit + 一份 verify 报告 | 执行者（可开 Cloud Agent） | 估算每轮 1–1.5 h（含 Gemini 约 1 min/次、每期 4 次调用）；总期数未知 | 每轮 VERIFY PASS |
| 5 | 约 15 期后：首批盲测（需要下载片段并去掉音轨）→ G1 | 执行者 | 3–4 h | G1 PASS 或失败分析 |
| 6 | 继续循环直到全部视频完成 | 执行者 | 取决于总期数 | manifest 全部完成 |
| 7 | G2 + 生成 views（md / mermaid）替代手写 index | 执行者 | 3–5 h | G2 PASS |
| 8 | 用户决定 FLT 卡和旧 index 的去留（§9） | 用户 | — | — |
| 9 | 用户按 S/B 机位规范拍一次 → facts_v2 × 3 + Pose → diagnose → 10 球测试结案 | 用户 + 执行者 | 一次训练 + 半天 | 报告每条结论都能追到事实和来源 |

**并行**：步骤 1–2 不依赖字幕；步骤 3 起需要字幕。片段下载（步骤 5）依赖视频下载，PILOT 里在 Mac 上遇到过 403，是未解决的风险。

---

## 7. 不变的约束（执行者必读）

- 训练场景：发球机最低速、单人正手、30–50 球；偶尔 mini tennis。所有测试 ≤10 球。
- 每个给用户的练习/测试开头：**准备时鼻吸，启动前挥至触球时用嘴短促呼气。**（不屏气；若前额/枕部痛、眩晕、恶心，当天停止挥拍并就医评估。）
- 禁令：不把「推肘 / 降拍 / 旋后 / 伸肘」作为主动口令（脚本 R09/T06 会扫描）。FTT 原话里出现相近说法时，标 `ban_conflict` 如实记录，不删原话。
- Intuition-First：给用户的只有一个外部目标；树是给系统诊断用的，不是给用户场上背的。
- 密钥只从环境变量读；日志、卡片、报告里不写密钥。提交前 grep 检查。
- 版权：只存 ≤40 词的原话摘录、路径和 sha；完整字幕和片段不入库。

---

## 8. 风险

1. **片段下载**：抽取和挂树全程不需要下载视频；只有盲测需要十几秒的静音片段。PILOT 里下载整期遇到过 403，按段下载是否同样被拦**未测**。拿不到片段，V5 只能用会泄露答案的退路，G1/G2 就无法正式通过。
2. **盲测偏乐观**：示范是夸张的、机位不同（§4.4）。G2 不等于「对用户视频一定准」。
3. **事实层上限**：即使树很好，A/B 层事实不可靠时也只能输出「低置信 + 补拍」。这是如实的结果，不是失败。
4. **初始阈值是估计**：0.75 / ±10 s / recall 0.6–0.7 等都没有用真实数据校准过，前 3 期试跑后可能需要调整（需用户同意）。
5. **CLAUDE.md 与本计划不一致**：CLAUDE.md 仍要求以 Tennis Science > HSA > JUL > FTT 排序，并要求某些问题第一句先问 HSA / ESR。本计划诊断时只用 FTT。这里没有改 CLAUDE.md。

## 9. 需要用户拍板（≤3 个）

1. **诊断树以 FTT YouTube 为唯一主干**，并同步 CLAUDE.md 的权威排序和「第一句必须问 HSA / ESR」规则（至少在视频诊断流程里豁免）。
2. **盲测片段**：是否允许执行者用 yt-dlp **按段**下载盲测用的十几秒片段（每个用例一段，总共约 30–50 段，不下载整期）。抽取本身不需要下载。不同意的话，盲测只能用不静音的 YouTube 截段，会泄露答案，不能作为正式验收。
3. **G2 通过后**，FLT-01…10 和手写 index 是否降级为 tree.json 的生成视图（建议是：一棵树、一个来源）。

---

## 10. 变更记录

- 2026-10-18 v2：新建。初始阈值见 `pipeline/config.json`；脚本未运行（待第 0 步）。
- 2026-10-18 tree_v1：新增 `pipeline/build_tree.py`、`textsim.py`、`blind_replay_text.py`、`compose_tree_report.py`；`tree.schema.json` 增加**可选**字段（symptoms / solutions / angles / video_layer / quote_sources / build、node.layer）；`check_tree.py` 增加 T13（只加严，不放宽）+ 3 个变异样例（自检 33 个）。`config.json` 阈值未改。
- 2026-10-18 tree_v2：新增 `pipeline/obs_catalog.json`（OBS-01…18 中立英文题）、`obs_bind.py`（症状/问题/根因 → OBS 绑定，2 次运行 + 补问）、`mech_merge.py`（同击球类型语义机制合并，2 次运行取交集）、`build_tree_v2.py`、`check_obs_binding.py`（O1–O11）、`check_mech_merge.py`（M01–M09）、`blind_replay_text_v2.py`（根因独立检索变体）；`verify_all.py` 只加严：新增可选参数 `--tree-v1/--binding/--merge` 与 24 个变异样例（自检共 57 个）。`config.json` 阈值、`build_tree.py` 的 0.40 词面合并阈值、check_tree 规则均未改；T05 仍对无 required 的 must_fix 节点报 FAIL（未用改 must_fix 绕过）。`ftt_tree/tree.json` 保持为 v1 基线，v2 为 `ftt_tree/tree_v2.json`（未覆盖）。
- 2026-10-18 tree_v3：用户批准把 NEW-OBS 候选升级为观察项。新增 `pipeline/promote_new_obs.py`（规则：≥2 期、层 A/B、机位明确、无禁词；模型只对簇代表分组；题目与取值逐字取自候选原文）→ `obs_catalog.json` 增 OBS-19…25 + `observables/OBS-19…25_*.md` + `ftt_tree/new_obs_promotion.json`；按原方法重跑绑定与机制合并 → `obs_binding_v3.json` / `mech_merge_v3.json` / `tree_v3.json`（v1 `tree.json` 与 v2 `tree_v2.json` 均未覆盖）。`verify_all.py` 自检数据改指向 v3 产物（57 个变异仍全 PASS）。`obs_bind.py` / `mech_merge.py` / `tv2_common.py` 增加环境变量 `TV_WORK` `TV_BIND_OUT` `TV_MERGE_OUT` 与 `obs_bind.py build --draft`；规则与阈值均未改。T05 100→88，T08 3→14（未加模型编的鉴别项），仍 FAIL。

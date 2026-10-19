# docs/diagnosis/ — 诊断用知识库（v0，2026-10-18）

> 目的：把「按来源堆放」的知识，改成「看到什么现象 → 可能是哪几个原因 → 怎么区分」的形态，给视频诊断用。
> 方案全文：`PLAN.md`（本目录）。本目录只新增文件，旧资料原地保留、只做链接。

## 1. 权威顺序（诊断时谁说了算）

| 顺位 | 来源 | 在本目录里的角色 |
|---|---|---|
| 1 | **FTT 主干**：《The Fault Tolerant Forehand》原书（`books/ftt_fault_tolerant_forehand/original.pdf`）+ FTT 网站 + FTT YouTube | 唯一能「定义故障」「给成因边」的来源 |
| 2 | **TPA**（Tom Allsopp）| 辅证、练习；不新开故障卡，除非 FTT 完全没覆盖且有 ≥2 个来源一致 |
| 3 | **Bourne**（One Minute Tennis）| 技巧、练习补充；规则同 TPA |
| — | **用户自己的领悟**（learning.md 里的「圣经 / 根因」）| **一律先进问题登记表 `user_case/problem_register.md`，作为假设**，有状态（待验证 / 证实 / 排除 / 合并），不直接写进故障卡，也不再宣告「圣经」 |

裁决规则：
- 同一问题 FTT 与 TPA/Bourne 说法冲突 → 以 FTT 为准，冲突写进故障卡第 5 节，不删。
- FTT 书与 FTT 视频冲突 → 两者都记，标「FTT 内部冲突」，待用户裁决；不自行取舍。
- 书和视频都没有直接出处的成因 → 标「推断」或「单源」，不能作为第一候选。
- 注意：CLAUDE.md 当前的引用排序（Tennis Science > HSA > JUL > FTT）与本目录不同。本目录按 PLAN.md F1（用户已批准）执行；**CLAUDE.md 本轮未改**，需要另行同步。

## 2. 标注约定（所有文件通用）

| 标记 | 含义 |
|---|---|
| 【已核对 p.X】 | 已对照 `original.pdf` 第 X 页读到原文（该 PDF 的页码与印刷页码一致） |
| 【未核对】 | 没有回到原文/原视频核对过（行号、页码、时间戳、原话都适用） |
| 【笔记归纳】 | `docs/research/ftt/01_ftt_book.md` 里有，但原书没有对应段落或表格，是笔记作者的归纳 |
| 【单源】 | 只有一个来源支持这条成因边 |
| 【推断】 | 书和已核对材料里都没有直接出处，是我们的推理 |

引用格式：`书 p.83【已核对】｜笔记 01_ftt_book.md:279`。笔记行号只作定位，**以原书页码为准**。

## 3. 核对时发现的、会影响全局的事实（2026-10-18）

1. 原书扉页作者是 **John Kumpf**（Published April 13, 2021，Amazon KDP，© Fault Tolerant Tennis）【已核对 p.1】。CLAUDE.md 和 PLAN.md / 旧笔记写的「Hugh Clarke」与扉页不符；故障卡与原则里已用 John Kumpf 的保持。FTT YouTube 频道主讲人身份【未核对】（与书作者是否为同一人，本轮不臆测）。
2. 笔记 `01_ftt_book.md:688–705` 的「常见错误与纠正一览」表**原书没有**【笔记归纳】。PLAN.md B3 引用的 `:695 / :698 / :702 / :703` 等都是这张表，所以本目录改为引用原书对应页。
3. 笔记 `:720`「手肘弯曲程度是 discretionary」在原书「必须 vs 风格」一节（p.53–55）里**没有这句**；肘伸展属于个人选择的原文在 p.66、p.69–70【已核对】。
4. 原书序言 p.17–18「Incorrectly Treating the Symptom」和 p.35「先问球飞成什么样」是最直接的诊断原则，笔记里**完全没收**。已写进 `00_principles.md`。
5. 原书 p.75 把「准备好后的短暂等待」写成准备的最后一步，而不是错误。这直接影响 FLT-03（见该卡）。

## 3.1 已知冲突与限制（质量标注，2026-10-18）

### 书作者
- 扉页已核对：**John Kumpf**。CLAUDE.md / 旧笔记中的 Hugh Clarke **与扉页不符**；本目录故障卡、原则、SRC 卡以 John Kumpf 为准。
- YouTube 主讲人是谁、是否与书作者同一人：**未核对**，不写进故障卡作者栏。

### 权威顺序（与 CLAUDE.md 不同步）
- **本目录**按用户批准以 **FTT 诊断为主干**（见 §1）。
- **CLAUDE.md** 仍是 Tennis Science > HSA > JUL > FTT。
- **本轮不同步 CLAUDE.md**，等用户拍板后再改仓库根规则。

### 口令禁令出处
- CLAUDE.md 永久禁令原文只有主动「**推肘**」（5/6 加入）。
- 扩展到「**降拍 / 旋后 / 伸肘**」来自用户训练偏好（甜区记忆：禁止主动推肘 / 降拍 / 旋后 / 伸肘），**不是** CLAUDE.md 原文。
- 本目录故障卡处方继续执行扩展禁令；若与 CLAUDE.md 冲突，以本目录处方栏为准，并在此注明出处差异。

### OBS 可靠度（10/17 跑数）
- `observables/00_runs_2026-10-17.md` 的数字来自：**单机位（侧后 R）**、**提示词未存档**、**混用多模型 / 多种问法**。
- **不能当作重测信度**，只作「这批跑出来看过」的参考；正式诊断前需固定提示词版本 + 同模型重复 + 多机位。

## 4. 目录与模板位置

```
docs/diagnosis/
├── README.md                 # 本文件
├── PLAN.md                   # 方案全文（只读）
├── 00_principles.md          # FTT 诊断原则（优先于所有故障卡）
├── tree.md                   # 诊断树 v0（mermaid）
├── templates/
│   ├── FLT_template.md       # 故障卡模板（PLAN B2）
│   ├── OBS_template.md       # 观测事实卡模板（PLAN B2 末段）
│   └── SRC_template.md       # 来源卡模板（PLAN D1）
├── faults/FLT-01 … FLT-10    # 故障卡
├── observables/OBS-01 … OBS-18 + 00_runs_2026-10-17.md（可靠度数据来源说明）
├── sources/SRC-FTT-BOOK.md   # FTT 书按章来源卡
├── index/
│   ├── obs_to_fault.md       # 看到某个事实 → 候选故障
│   └── symptom_to_fault.md   # 结果/体感症状 → 候选故障
├── user_case/
│   └── problem_register.md   # learning.md 假设登记（非故障卡）
└── prompts/
    ├── README.md             # 提示词版本规则
    └── fact_extract_v1.md    # 事实抽取 v1
```

已补第一周任务 4–5：`user_case/problem_register.md`、`prompts/fact_extract_v1.md`（及 `prompts/README.md`）。
尚未建立：`user_case/sessions/`、`symptoms/`、`drills/`、`index/source_to_fault.md`。

## 5. 所有故障卡的固定规则

- **训练场景**：发球机最低速 + 单人正手；偶尔 mini tennis。确认/排除测试必须 10 球内完成。
- **每个练习/测试开头固定提醒**：准备时鼻吸，启动前挥至触球时用嘴短促呼气。（呼气只做节奏锚和张力阀门，不是发力来源；不屏气，不做 Valsalva，不用呼气硬推胸部或髋部。若伴随前额/枕部疼痛、眩晕或恶心，当天停止挥拍并就医评估。）
- **口令**：只写一句外部目标（Intuition-First），不叠身体口令。
- **禁令**：禁止主动「推肘 / 降拍 / 旋后 / 伸肘」类口令作处方。出处：CLAUDE.md 永久禁令仅「推肘」；「降拍 / 旋后 / 伸肘」为用户训练偏好扩展（见 §3.1）。
- **通用知识与个人情况分开**：故障卡第 9 节只放链接，不写用户个人结论。

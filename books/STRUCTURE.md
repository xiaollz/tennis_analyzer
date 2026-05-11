# books/ 目录结构

本目录归档项目所有参考书籍 PDF。书籍是 reasoning 层的知识来源，**不是训练 cue 来源**（按 Intuition-First 协议）。

## 目录布局

```
books/
├── bourne_one_minute_series/         # Bourne 系列（One Minute Tennis）
├── ftt_fault_tolerant_forehand/      # FTT（Fault Tolerant Forehand）
├── academic/                          # 学术教科书
├── misc/                              # 其他/单本
└── STRUCTURE.md                       # 本文档
```

---

## 各子目录内容

### academic/ — 学术教科书（最高权威）

| 文件 | 作者 / 年份 | 原文件名 |
|---|---|---|
| `tennis_science_2015.pdf` | Bruce Elliott / Machar Reid / Miguel Crespo, 2015 | `Tennis Science - How Player and Racket Work Together -- Bruce Elliott, Machar Reid, Miguel Crespo -- Illustrated, PS, 2015 -- The University of -- isbn13 9780226136400 -- de95a56ed1a0c94c2fcca603825a8a8b -- Anna's Archive.pdf` |
| `网球运动系统训练_2015.pdf` | E. Paul Roetert / Mark S. Kovacs, 2015 中文版 | `网球运动系统训练 -- E_保罗·勒特尔,马克·S·科瓦奇 -- 2015-02-01 -- 北京：人民邮电出版社 -- 9787115377173 -- 87332fa3f0aee21842cfbaa6073316a0 -- Anna's Archive.pdf` |

- **Tennis Science**（Elliott/Reid/Crespo）：ITF + UWA + Tennis Australia 三方权威，peer-reviewed 教科书。本项目最高引用源。配套知识库 `docs/research/tennis_science/book/` 8 章节 KB 文档。
- **网球运动系统训练**（Kovacs）：中文版体能系统训练教材，下肢/SSC/弹簧机制相关概念的主要中文来源。

### ftt_fault_tolerant_forehand/ — FTT 系列（Hugh Clarke）

| 文件 | 说明 | 原文件名 |
|---|---|---|
| `original.pdf` | 英文原版 | `The Fault Tolerant Forehand_ Succeed Under Imperfect Conditions_nodrm.pdf` |
| `chinese_精校版_v3.pdf` | Kindle 中英对照精校版 v3 | `The Fault Tolerant Forehand_Kindle中英对照精校版_v3.pdf` |
| `chinese_精译整理版.pdf` | 中文精译整理版 | `The Fault Tolerant Forehand_中文精译整理版.pdf` |

- FTT 是用户正手知识体系的核心理论框架（容错性优先 / 旋转鞭打 / 主动 vs 被动）。
- 配套博客 / YouTube 视频 KB：`docs/research/ftt/`

### bourne_one_minute_series/ — Bourne 一分钟系列（Stephen Bourne）

| 文件 | 说明 | 原文件名 |
|---|---|---|
| `Forehand_Solution.pdf` | 正手 | `One Minute Tennis Forehand Solu - Bourne, Stephen(1).pdf` |
| `Backhand_Solution.pdf` | 反手 | `The One Minute Tennis Backhand - Bourne, Stephen.pdf` |
| `Power_Solution.pdf` | 力量 | `The One Minute Tennis Power Sol - Bourne, Stephen.pdf` |

- 同名 YouTube 频道分析 KB：`docs/research/bourne_one_minute/`

### misc/ — 其他

| 文件 | 说明 | 原文件名 |
|---|---|---|
| `coswing.pdf` | CoSwing（拍头轨迹分析）小册子 | `CoSwing.pdf` / `coswing.pdf` |

---

## 引用优先级（按项目 5/11 规则）

回答技术问题时引用顺序：

1. **Reid/Elliott 2013** 单篇论文（最聚焦的正手综述）
   → `docs/research/tennis_science/papers/reid_elliott_crespo_2013_forehand_review.pdf` + `docs/research/tennis_science/tennis_science_paper_reid_elliott_2013.md`
2. **Tennis Science 教科书 2015**（Elliott/Reid/Crespo）→ `books/academic/tennis_science_2015.pdf`
3. **Kovacs 2015**（中文 体能/下肢）→ `books/academic/网球运动系统训练_2015.pdf`
4. **HSA 框架文档**（项目内沉淀，5/3 之后）→ `docs/research/hsa_master_index.md`
5. **FTT 体系**（Hugh Clarke）→ `books/ftt_fault_tolerant_forehand/`
6. **Bourne 系列** → `books/bourne_one_minute_series/`
7. **CoSwing** 及其他 → `books/misc/`

注：用户自身突破（learning.md）跟以上权威冲突时**以权威为准**。

---

## 推荐阅读顺序（按用户当前阶段：5/8 之后 ESR 根因诊断阶段）

新读者 / Claude session reset 时按此顺序：

1. **入门**：`books/ftt_fault_tolerant_forehand/chinese_精校版_v3.pdf`（中英对照，最易入手）
2. **理论锚定**：`books/academic/tennis_science_2015.pdf` 第 4-6 章（正手 / 上半身 / 动力链）
3. **下肢/体能**：`books/academic/网球运动系统训练_2015.pdf` 弹簧机制 / SSC 章节
4. **应用补充**：`books/bourne_one_minute_series/Forehand_Solution.pdf`
5. **拍头轨迹**：`books/misc/coswing.pdf`（短，可随时翻）

---

## 来源说明

部分 PDF 来自 Anna's Archive（标注 hash），属合法学术获取范畴。FTT / Bourne / Tennis Science 均为用户合法购买电子书版本。

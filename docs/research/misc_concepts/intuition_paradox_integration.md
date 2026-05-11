# 🧠 Intuition Paradox 整合 — 项目方向重构

> **触发文章**：FTT《The Intuition Paradox》https://faulttoleranttennis.com/the-intuition-paradox/
> **整合日期**：2026-05-06
> **重要性**：项目级方向重构。从"过度 Reasoning"切换到"Reasoning + Intuition 平衡"。

---

## 0. 一句话总览

**网球必须由 Intuition 执行，Reason 只能 set goal。教练的工作是设定方向，不是规定方法。**

---

## 1. FTT 文章核心论点（用户必读）

### 1.1 两种思维模式
- **Reason（理性）**：慢、刻意、逻辑——适合 set goal、分析、学习
- **Intuition（直觉）**：快、本能、自动——唯一能在 500ms 反应窗口执行动作

### 1.2 核心定理
> "Reason alone cannot foster improvement."

网球反应窗口 ~500ms < 意识决策时间 300-500ms → **意识根本来不及参与击球瞬间**。

### 1.3 直觉的工作机制：Gradient Descent
- 每次挥拍 → 观察结果 → 直觉自动调整下一次
- 不需要意识参与
- 处方：**"Relax, swing, calmly observe"**（Gallwey 1974）

### 1.4 Intuition Paradox（核心洞察）
- 直觉是最强工具，**但会卡在局部最低点**
- 周围每一步都更差 → 直觉判断已经最优 → 卡住
- 突破需要**先变差才能更好**——直觉本身做不到

### 1.5 两种 Plateau
| 类型 | 特征 |
|---|---|
| **Reason Plateau**（学术派）| 看遍视频、镜前完美、闪光时刻多但不稳定 |
| **Intuition Plateau**（推球手）| 靠本能赢球、UTR 8 封顶 |

→ **本项目用户是教科书 Reason Plateau 案例**。

### 1.6 解决方案
**Reason 设目标 + Intuition 执行 + Observation 循环**：
1. Reason: 设定一个值得追求的目标（goal）
2. Intuition: 在目标下执行动作（无意识）
3. Observation: 观察结果，喂给 intuition
4. Iterate: 让 gradient descent 自动优化

---

## 2. 项目自我诊断 — 过度 Reasoning

| 系统组件 | Reasoning 比重 | Intuition 比重 |
|---|---|---|
| HSA 框架 | 100% | 0% |
| Foundation Layer F1-F7 | 100% | 0% |
| diagnosis_engine.py | 100% | 0% |
| VLM Q1-Q42 | 100% | 0% |
| learning.md 圣经层级 | 100% | 0% |
| Claude 教练回答模式 | 95% | 5% |
| 每日训练计划（Block A/B/C/D）| 95% | 5% |

**用户 5/15 顿悟 → 5/16 球场失败的精确机制**：
- 我们系统给的：完美 reasoning model（HSA + ESR + Wrap + tilt + 4 根因）
- 用户镜前 100% 调用（reason mode 在 0 压力下可控）
- 上球场（500ms 窗口）→ reason 太慢 → fallback 老动作
- **本质问题**：我们一直在教学习网球知识，没在教打网球

---

## 3. 系统重构方案

### 3.1 核心规则升级

**新最高级原则**（写进 CLAUDE.md）：

> **Reasoning 用于设定方向，Intuition 用于执行动作。教练的工作是 set goal，不是 prescribe method。**

### 3.2 Outcome-First 协议（替换现有 Diagnosis-First）

**用户报症状时，新回答顺序**：

```
Step 1（必做）：问 outcome
  "刚才球去哪了？" / "目标是什么？"

Step 2：判断是否需要介入
  outcome ≈ aim → 不修任何东西
  outcome 偏离 aim → 进 step 3

Step 3：goal vs method
  先问目标对不对（reason 工作）
  目标对但 outcome 偏 → method 调整

Step 4（最后）：method 调整
  ONE cue（不是 5 个）
  允许下一拍仍然不对
```

**只有 3 种情况切回 reasoning-heavy 模式**：
1. 用户明确要求理论解释
2. 多次相同失败模式（intuition 卡 local minimum）
3. 用户主动问 "应该 aim 什么"（goal-setting 是 reason 工作）

### 3.3 训练方法重构

**删掉**：
- ❌ 1000 次完美技术重复
- ❌ Block A/B/C/D 切片训练
- ❌ 镜前 100 次自检作为主训练

**降级**（保留但不主导）：
- 🟡 镜前作为体感唤醒（10 min/天）
- 🟡 哑铃影子挥作为力量训练

**新增（成为主体）**：
- ✅ **Outcome-only training**：设球质目标 → 100 球 → 只看落点 → 不分析挥拍
- ✅ **Variable feed**：发球机随机 / 找人对打
- ✅ **Mistake tolerance**：允许 30-40% 失败率
- ✅ **Constraint-led drills**：环境约束自动产生正确动作

### 3.4 KB 重构（并行结构，不删除）

```
现有 reasoning 体系（保留作 reference）：
  hsa_master_index.md
  foundation_layer.py F1-F7
  diagnosis_engine.py
  
新加 Intuition 体系（建中）：
  intuition_first_coaching_protocol.md
  outcome_metrics.md（球质 KPI）
  goal_setting_templates.md
  variable_feed_drill_library.md
  observation_journal_template.md
```

### 3.5 Claude 教练默认模式切换

| 触发 | 旧默认（reasoning） | 新默认（intuition） |
|---|---|---|
| 用户："今天打不好" | 长篇根因分析 | "球去哪了？目标是什么？" |
| 用户："我哪里错了" | 列 5 条根因 | "你观察到什么？" |
| 用户："想顿悟" | 给新 cue/概念 | "今天没顿悟也行，继续打 100 球观察" |
| 用户："该练什么" | 多 Block 计划 | "打 100 球，aim 是 X" |

---

## 4. 立即可执行的训练协议（替换 5/16 4 段计划）

### 4.1 新模板：Outcome-First 训练日

**45-60 分钟，分 3 段**：

#### 段 1：体感唤醒（10 min, 不是技术训练）
- 镜前 30 次空挥
- **只想一个**当前最深的体感（今天可以是"小臂转上去"）
- 不自检通过率

#### 段 2：Outcome 训练（30 min, 主体）
- 发球机喂球 / 找人对打
- 设**一个**球质目标（明天可以是"球落在底线前 2 米的对角线区域"）
- 打 100 球，**只看球的落点**
- 不分析挥拍
- 失败 30-40% 完全正常

#### 段 3：Observation 复盘（5 min）
- 写下：100 球里多少落到目标区？
- 写下：球**普遍**偏哪里？（不是分析"为什么"）
- 不写技术分析

### 4.2 当日给 Claude 的反馈（极简）

只发 3 个数字 + 1 句观察：
```
落点命中率: ___%
偏向: 长 / 短 / 左 / 右
今天的整体感觉（1 句）: 
```

**不要**发：
- 技术细节描述
- 哪一步做错了
- 想改的下一个 cue

---

## 5. 我（Claude 教练）的承诺修正

### 5.1 默认回答模式
从 5/6 起，用户说"今天打不好"——我**第一句永远是**：
> "球去哪了？aim 的目标是什么？"

**不再**第一句给：
- 根因分析
- 新 cue
- 多 Block 计划

### 5.2 长篇 reasoning 的触发条件
**只有 3 种情况**触发 reasoning-heavy 回答：
1. 用户明确说"给我讲解 X 的力学原理"
2. 用户描述同一失败模式 ≥ 3 次（intuition 卡 local minimum）
3. 用户主动问"我下一阶段应该 aim 什么目标"

其他所有情况 → **outcome-first**短回答（< 200 字）。

### 5.3 跟现有"Diagnosis-First"规则的关系

5/15 我承诺过 "症状 → 一句话诊断 + KB 证据"。**这条规则保留，但优先级降到 step 4**——只有 outcome 偏离且 goal 正确时才启用。

---

## 6. 用户的下一阶段路径

### 短期（5/6-5/9, 4 天）
- **完全停止**新概念学习
- **完全停止**镜前 100 次自检
- **每天**一段 outcome-only 训练（30 min）
- **每天**3 数字反馈
- 我**只**做 outcome-first 回答

### 中期（5/10-5/30, 20 天）
- 持续 outcome 训练
- 让 intuition 做 gradient descent
- 我观察用户 outcome 是否在改善
- 如果**3 周后**某个具体 outcome 仍然没改善 → 才回 reasoning 介入

### 长期
- 用户从 reason plateau → 进入 intuition 主导模式
- 圣经体系作为**reference knowledge**保留，不再作为**training target**

---

## 7. 等待 Part 2 + Part 3

文章末尾：

> "This is part 1 of a three part series explaining how to meaningfully improve in tennis, covering Intuition, Reason, and Feedback. Parts 2 and 3 coming soon!"

→ 当 Part 2 (Reason) + Part 3 (Feedback) 发布时，立即整合。本文档预留 §8 给后续整合。

---

## 8. 后续整合（待补）

- [ ] FTT Part 2: Reason 的具体作用 — 何时该用、如何设置 goal
- [ ] FTT Part 3: Feedback 系统 — 如何观察、如何让 intuition 学到
- [ ] Tim Gallwey 《Inner Game of Tennis》深读（用户已有此书）
- [ ] 整合 Wulf《Attention and Motor Skill Learning》—— external focus 是 intuition 的入口

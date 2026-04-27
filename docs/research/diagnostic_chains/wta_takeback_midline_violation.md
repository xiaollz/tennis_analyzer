# 诊断链：拍头过早倒后 → 张力丢失 → 击球虚软（Midline Rule Violation）

> 第二条诊断链。来源：Sky Kim "ATP vs WTA Forehand" (via _pB-WTQGSp4) +
> "Hide the Elbow" (via va005XuoBEU) + Backswing 重定义 (via 7_7I7sTBYTQ)。
> 模板沿用 `early_front_foot_landing.md` 四段式。

---

## 1. 问题与背景

**学员主诉**：
> "动作看起来很大、引拍也够，但球软绵绵没有穿透力。教练总说我没用上身体，但我感觉我已经在用力了。"

**力学根因**（不是手力的问题，是**拍头指向时机**的问题）：

```
理想（ATP 风格）：
  Unit Turn 时拍头抗拒向后倒 → 拍头跨过身体中线后才允许向后翻转
  → 身体已经转 90°，拍头还指斜前方 → 巨大的方向差 = 巨大的张力
  → 释放 = 鞭打效应 = 重球

错误（WTA 风格用在男子业余）：
  Unit Turn 第一时间拍头领先向后倒 → 肘 + 肩同步锁死在引拍位
  → 整条手臂提前 pre-locked → 没有方向差 = 没有张力
  → 释放 = 旋转拖拽 = 推球
```

**关键认知**：业余男选手感觉的"我已经引拍很大了"是**几何上的大**，但**张力为零**。
RTP 把这种情况叫 "Arming the shot"——只用胳膊抡，没用身体。

---

## 2. 分析维度（按 10 层 taxonomy）

主要在 **L2（时序）** 和 **L6（上半身机制）** 的多个新子维度：

| 维度 | 测量 | 理想 | 错误 |
|---|---|---|---|
| **拍头过中线时机**（Midline Rule，via _pB-WTQGSp4）| 拍头跨过身体中线相对于身体转动 90° 的时机 | 之后 | 之前 |
| **Hide the Elbow**（via va005XuoBEU）| 引拍完成时从对手视角能否看到右肘 | 看不到（藏在躯干后） | 完全可见 |
| **Hip Locking**（via 9ihq4WFCWy0）| 右胯随肩转动的角度 | 0-15°（独立锁定）| ≥45°（跟肩同步转）|
| **Backswing 加速段长度**（via 7_7I7sTBYTQ）| 从拉拍最高点到击球点之间的拍头加速距离 | 30-50cm | <10cm（减速）|

副症状会出现在 **L1（几何）**：击球点常被挤到偏后偏近（因为引拍过深→必须等更久才回到击球点）。
副症状会出现在 **L10（输出）**：球速慢、弧线低（无 plow-through），落点浅。

→ **L1/L10 是表象，L2 + L6 是根因。修 L2 + L6，下游自动好。**

---

## 3. VLM 信号（候选加入 `OBSERVATION_TO_CONCEPT`）

VLM 描述里出现以下任一表述时，触发对应概念：

| VLM 关键词 | 概念 ID | severity |
|---|---|---|
| 拍头第一时间向后倒 / racket tip went back early / wta-style takeback / tip leads first | `L2_takeback_tip_leads` | 0.8 |
| 引拍时肘部完全可见 / elbow visible from front / arming the shot / loose elbow flapping | `L6_elbow_not_hidden_unit_turn` | 0.8 |
| 髋随肩同步转动 / hip turn matches shoulder / no x-factor separation | `L6_hip_locks_failed` | 0.85 |
| 引拍后无加速段 / no acceleration before contact / racket decelerated into contact | `L2_no_pre_contact_accel` | 0.7 |
| 引拍合格的反例：肘藏 + 中线后倒 + 髋锁 | `L6_torque_loaded_correctly` | 0.0（正向） |

`_CONCEPT_LAYER` 把前三个映射到 L2/L6，top-down 推理会优先报 L2/L6 而不是 L1/L10——避免在表面"击球点偏后"层重复修而修不到根。

---

## 4. 给学员的建议

### 单字口令候选

**藏**——意思是"引拍要藏住右肘"。

学员既有的 8 字口令（盯/左/架/推/锁/撑/流/撕 + 飘）里，"架"管引拍框架。**"藏"和"架"配对**：架定外形，藏定深度。
检查方式只需一句话："引拍到位时，对手能不能看到我的右肘？"——能看到就是不及格。

### 渐进 drill（30 分钟一个 session）

```
0-5 min   对镜侧身空挥 × 20。每次 Unit Turn 完成时，
          镜子里看不见自己的右肘 = 通过；看得到 = 重做
5-15 min  Hip Lock + Tip Forward 喂球 × 30。
          引拍时强制让拍头指向斜前方（不许指后挡墙），
          同时右胯保持正侧位（不跟着肩转）
15+ min   实战拉球。失败一次就回到镜子前空挥 5 次再上场
```

### 验证方法（唯一硬指标）

侧面录像，回看 Unit Turn 完成的那一帧，**只看一件事**：
**球拍拍头是否还在身体中线之前？**

- 拍头已倒到身后（指向后挡墙）→ 错（WTA 风格）
- 拍头还在中线前（指向斜前方甚至天空）→ 对（ATP 风格）

不看张力、不看肘部、不看击球点，**只看拍头指向**。这是因为：拍头指向是 Hip Lock 和 Hide the Elbow 两件事的下游可视化指标，看一个就够了。

### 进度基线

业余男选手第一周一般能从 0% → 25%（30 球里 7-8 球做对）。
能稳定到 60% 时，再叠加 "撕" 和 "飘" 的精细化练习。

---

## 5. 与第一条链（前脚提前落地）的关系

两条链常**同时出现**：

```
L4 前脚提前落地 → 轴心崩溃 → 击球点几何被挤
                     ↓
L2/L6 拍头过早倒 → 张力丢失 → 击球虚软
```

但**修复顺序应该是 L4 在前**——左脚不飘，整个轴心都在前冲，肩根本无法做 Hip Lock。
所以诊断引擎检测到两条链同时触发时，应该**优先报 L4**（first chain），让学员先把"飘"练扎实，再练"藏"。

**一个礼拜只修一条链。** 用户确认 L4 ≥ 70% 再开始 L2/L6。

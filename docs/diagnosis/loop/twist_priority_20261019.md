# Twist / 动力链 优先清单（2026-10-19 Asia/Shanghai）

> 依据：`loop/manifest.json`（97 视频）+ `/workspace/ftt_subs/*.norm.tsv` 标题/关键词盘点。
> 用户原话焦点：扭转 ≠ 单纯旋转；力从哪来、怎么传到拍子；髋活动度/扭转把下半身 twist 力量传上来。
> 验收：本文件存在且含 state + 优先序 + non_teaching 理由。

## 1. 扫描规则

关键词（标题或字幕命中）：`twist` / `rotate` / `coil` / `power` / `kinetic` / `hip` / `corkscrew` / `unit turn` / `side bend` / `trunk` / `torque` / `abdominal`。

优先序（用户给定）：
1. **ui-RFRdQKRk**（硬指定）
2. 直接讲 **twist≠rotate / 动力链传力**
3. **髋 / 侧弯 / 腹拧** 相关
4. 其余**正手教学** `subs_ready`
5. 发球/战术/轮椅等：可 Stage1，但挂树价值低于正手动力链；非教学标 `non_teaching`

**Holdout 注记**：`holdout=true` 可 Stage1 入库为卡（知识/主干材料），但 `build_tree` **跳过**、树内不得引用（T02）。已有先例：`Am8j1Zw5KrE` 为 holdout + stage1_verified。

## 2. 本轮 Stage1 批（≤5，务必含 ui-RFRdQKRk）→ round10

| # | id | 标题 | state | holdout | 时长 | 档 | 理由 |
|---|---|---|---|---|---:|---|---|
| 1 | `ui-RFRdQKRk` | The Pros "Rotate" Differently Than You | subs_ready | **true** | 692s | P1 | 字幕核心：coil ≠ just turning sideways；coil into hip；anchor for twist；counter-rotate |
| 2 | `KU7FHy1qQOI` | The Physics, Geometry, and Biomechanics of Power Development | subs_ready (既有 **stage1_draft**，rerun 失败) | false | 1088s | P2 | kinetic chain / twist / power；可进树 |
| 3 | `6ak4ZScH_gc` | You Can Feel This - Here's How | subs_ready (draft) | false | 614s | P3 | corkscrew / twist opposite / coil；可进树 |
| 4 | `0m3BMfDDShI` | Build This Foundation - The Rest Will Follow | subs_ready (draft) | false | 890s | P3/P4 | unit turn + twisting 基础；可进树 |
| 5 | `5jHCDc44SQM` | The Abdominal Corkscrew ft. Carson Branstine | subs_ready | **true** | 457s | P3 | abdominal corkscrew；trunk twisted farther than hips；知识用，不进树 |

失败按现流程 rerun/consensus，**不放宽**阈值。

## 3. 已 stage1_verified 的 twist 相关（勿重复抽）

| id | 标题 | 备注 |
|---|---|---|
| `UB6SbA_KX9E` | Proper Trunk Sequencing will Transform Your Tennis | trunk sequencing |
| `wFIrPMutzRo` | 2 Secrets to Rotational Power \| Side Bending + The Abdominal X | 侧弯 + 腹 X |
| `xLs469ZVMPU` | Fix Your Forehand Over-Rotation \| 3 Counter-Rotational Balance Techniques | 反旋转平衡 |
| `JIMgI3jiVns` | How Shoulder Rotation Syncs Your Contact | 肩旋转与触球 |
| `PwkGLkRm73o` | The Pros Swing Differently Than You | 动力链/臂路径（trunk 材料已引用） |
| `tzgAWr3QGv0` | 7 Steps for a Sinner-Like Forehand \| #5: Tilt! | tilt / 躯干 |
| `aPcKJDadzek` | The Pros "Step In" Differently Than You | 步进与旋转相关 |
| `uJpuDUx0jIA` / `2Nn9tyMECZU` / `wzLMv4ASg9s` 等 | （字幕含 twist/coil 命中） | 已在树 |

## 4. 下一批候选（round10 之后，按序）

### 4.1 可进树（non-holdout · subs_ready）

| 优先 | id | 标题 | 备注 |
|---|---|---|---|
| 高 | `E_zmENJIj4g` | Coiling \| Spinal Motion \| … 15 Serves | coil 概念深，但是**发球**；draft 曾失败 |
| 中 | `wd4YRQW3TOc` | 4 Tips For Effortless, Controllable Topspin | 正手教学长视频 |
| 中 | `x_WIg59Wr9Y` | Why You Should Start Your Swing at Contact | 正手 |
| 中 | `pWzyP-xfLfU` | The Pros Grip Differently From You | 握拍；动力链次要 |
| 低 | `Z9XwaXDVkZg` | The Simple Foundation of the One-Handed Backhand | 反手 |
| 低 | `FjjvhNbk3q4` / `m1NBdd3Bigg` | Serve 系列 | 发球进度 |

### 4.2 Holdout · 知识优先（Stage1 后不进树）

| id | 标题 | 档 |
|---|---|---|
| `gehRK0Y6AdQ` | The Pros "Brush" Differently From You | 刷球 vs 动力链次要 |
| `q6LrQYDvxmY` | Why Pros Hit So Far Away | 间距；holdout |
| `2J8q7sn4JLc` | Why Your Forehand Sprays… | 正手喷球 |
| `5KdScDKxVSI` | Shoulder Adduction Will Transform Your Forehand Contact | 肩内收（已有 Am8j 相关） |
| `EkzzGZkqCgc` | Why Your Racket Doesn't Flip | 与已 verified wzLM 可能重复主题 |
| `hqganBxzQNM` | Why I Give Every Student a Weight to Swing | 重量挥拍手感 |

## 5. non_teaching / 降权（本 twist 专项不优先）

| id | 标题 | 理由 |
|---|---|---|
| `-ZmnDrRFfjc` | 3 Brilliant Alcaraz Tactics… | **战术**非动力链机制 |
| `w1FakobNq1Q` | Breaking Down the Greatest Tiebreak… | 比赛拆解 / 战术 |
| `omsLx7F2Wdc` | There's Only 1 Singles Strategy… | **策略** |
| `OYf48k-cfNI` | Fault Tolerance In Action | 短片；容错概念演示，非 twist 机制课 |
| `agtIR3PuNsw` | Why Alfie Hewett Hits Harder… | **轮椅**专项（有 coil，但人群不同） |
| `Bhzd6cM-oFU` / `MWjBfarwNlA` | Alcaraz slice / BH line | 切削/线路战术；holdout |
| `BbGzWTp5pCM` | Practicing in Slow Motion… | 练习方法元讨论 |
| `jDBg4Fpf0tQ` / `QrRYuVjVibU` / `Ibkq-yBteGc` | Shorts | 极短；round9 已处理过 |
| listed 无字幕 | `dYPv0QfFrsU` / `6j33LJfl46s` 等 | state=`listed`，缺字幕，本轮不抽 |

## 6. 与用户目标的对齐

- Step B 本批产出：卡 validate+quotes PASS → manifest `stage1_verified`（holdout 卡可 verified 但不挂树）。
- Step C：仅 **non-holdout** verified 进入 obs_bind → mech_merge → `tree_v15`；VERIFY PASS 才 commit。
- Step D：`ui-RFRdQKRk` + `5jHCDc44SQM` 等卡原文喂入 `kinetic_chain_trunk.md`（即使 holdout）。
- 已有挂点：FLT-05_hips_not_leading、FLT-02_arm_dominated、FLOW §8 trunk-first、`user_case/rear_20261019_trunk.md`。

## 7. 统计快照（写清单时）

- manifest：97；stage1_verified 57；subs_ready 33；listed 7
- tree：`tree_v14`；verify_2026-10-19_tree_v14.json **pass=true**；T05/T08 已清零


---

## 8. 进度快照（2026-10-20 Asia/Shanghai，round11 后 / round12 前）

| 项 | 数 |
|---|---:|
| stage1_verified | 65 |
| subs_ready 合计 | 25 |
| 其中教学·non-holdout（可进树） | ~6（含 0m3 draft 阻塞、2 Shorts draft、3 长视频 draft） |
| 其中教学·holdout（可 Stage1 知识、不进树） | ~11 |
| 降权 non_teaching / 战术等 | ~8 |

### 0m3BMfDDShI 阻塞（跳过 round12）
- 两轮（round10/11）三跑 consensus 皆 FAIL。
- 主因：`solutions` transcript-anchor Jaccard 常 <0.5（pair 2-3 causes 已 PASS 但 solutions 0.30）。
- **无新策略且不放宽阈值** → 本轮跳过；除非改 prompt 或人工拆卡（需另批）。

### round12 拟选（≤5）
1. `Z9XwaXDVkZg` — OHBH foundation（non-holdout，可进树）
2. `FjjvhNbk3q4` — Effortless Serve（non-holdout；字幕含 twist/coil）
3. `m1NBdd3Bigg` — Serve progressions（non-holdout）
4. `2J8q7sn4JLc` — Why Your Forehand Sprays（holdout，正手）
5. `5KdScDKxVSI` — Shoulder Adduction FH Contact（holdout，正手接触）


---

## 9. 进度快照（2026-10-20，round12/13 后）

| 项 | 数 |
|---|---:|
| stage1_verified | 68 |
| tree_v17 期数（非 holdout 挂树） | 61 |
| subs_ready 合计 | 22 |
| 教学·non-holdout 仍待抽/draft | 6（含 0m3/Z9X/Fjjv/m1NB 等 **rerun 阻塞**） |
| 教学·holdout 仍待抽/draft | 9 |
| 降权战术等 | 7 |
| listed（无字幕） | 7 |

### round12（全 draft）
Z9XwaXDVkZg / FjjvhNbk3q4 / m1NBdd3Bigg / 2J8q7sn4JLc / 5KdScDKxVSI — consensus FAIL（多为 solutions Jaccard &lt;0.5；**未放宽**）。

### round13
- verified：**OYf48k-cfNI**（进树）、**hqganBxzQNM**、**EkzzGZkqCgc**（holdout 知识）
- draft：gehRK0Y6AdQ、q6LrQYDvxmY

### 阻塞共性
长教学片 solutions 锚点不稳定 → 反复 draft。短片 / 主题单一片更容易 PASS。再开 round14 建议：**优先从未抽过的较短 holdout 正手**（Fu6 / emZx / BbGz 等）或补 listed 字幕；对 Z9X/Fjjv **不要无策略重试**。

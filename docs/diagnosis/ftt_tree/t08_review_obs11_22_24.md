# T08 人工复核：OBS-11 / OBS-22 / OBS-24（tree_v3）

> 范围：`tree_v3.json` 中 T08 失败对里最集中的三组（共享 required 签名、无鉴别项）。  
> 来源：`loop/reports/verify_2026-10-18_tree_v3.json` 的 T08 明细 + `obs_binding_v3.json` + `tree_v3.json` 节点原文与字幕引用。  
> 本文件只做人工复核清单；**不改树、不合并、不编鉴别项、不改 check_tree 规则**。  
> 基线 commit：`28e4917`。

T08 规则回顾：两个非 style 节点的 `signature.required` 完全相同，且彼此没有 `differentials` → FAIL。升级 OBS-19…25 后，有 required 的根因从 15 增到 27，同一签名下的节点变多，T08 从 3 升到 14。本清单覆盖其中 **9 对 / 3 个连通组**（其余 5 对另议）。

---

## 组 1：OBS-11 = `down_then_forward_and_up`（先下再前上）

**共享 required：** `OBS-11=down_then_forward_and_up`  
**T08 对数：** 3（三角形：3 个节点两两撞车）  
**涉及击球类型：** 全部正手（forehand）

### 节点

#### A. `N-C_5_IA7C4XZFU_C02`
- **原文：** Increasing tangential/brushing racket head speed on soft strings creates unwanted vertical exit launch angle simultaneously with topspin.
- **期：** `5-Ia7c4XzfU`（This Tennis Robot Exposed My Tension Issue）
- **绑定理由（模型）：** Brushing swing path is visible as low-to-high hand motion; vertical ball launch angle requires a new observable.
- **supporting：** OBS-16=long
- **字幕摘录：**
  - Q04 @302s：「The softer your string bed, the more the velocity of the swing itself flings the ball in the same direction…」
  - Q08 @486s：「…if your equipment doesn't generate a shorter landing ball when you add an oblique element to the swing, then that correction is never going to work…」

#### B. `N-C_FXDMVI3EFNE_C01`
- **原文：** An upward brushing stroke naturally narrows the physical and temporal contact window through space, making contact extremely thin.
- **期：** `FxDmVi3EFnE`（The Truth About the Topspin Pro）
- **绑定理由：** Upward brushing describes a low-to-high hand and racket trajectory through the hitting zone.
- **supporting：** OBS-15(b)=wraps_back_to_body_immediately；OBS-16=frame_hit
- **字幕摘录：**
  - Q05 @159s：「top spin always makes contact thinner.」

#### C. `N-C_FXDMVI3EFNE_C03`
- **原文：** The player attempts to produce topspin through an excessive low-to-high swing path rather than shoulder drive and forearm rotation.
- **期：** `FxDmVi3EFnE`（与 B **同一期**）
- **绑定理由：** Excessive low-to-high swing path shows hand moving down then forward and up.
- **supporting：** OBS-15(a)=finish_above_non_hitting_shoulder；OBS-15(b)=wraps_back_to_body_immediately
- **字幕摘录：**
  - Q10 @286s：「the shoulder is much better used for that back to front driving through the ball action. And then the forearm and rotation about the forearm is the best tool to use for top spin.」

### 判断
- **A 与 B/C 不是同一机制。** A 的真正机制是「软弦床 + 刷球速度 → 出射角被甩高」——器材与出球结果；OBS-11（手的路径）只是辅证，模型自己也写了「vertical launch angle requires a new observable」。把 A 和 B/C 合并会丢器材层。
- **B 与 C 相关但不是同一机制，且同属一期。** C 是「用过度低到高路径造上旋」（动作选择）；B 是「向上刷导致击球窗口变薄」（路径的后果）。同机制合并规则本来就不允许同一视频的两条根因合并；即便允许，合并也会把「因」和「果」糊在一起。
- OBS-11 在这三处都是「看得见的共同表象」，不是「同一根因」。

### 建议：**补 OBS**（首选）/ 暂留人工定（次选）/ **不合并**
1. **不要合并**这三节点。
2. **补 OBS（优先）：** 触球瞬间球离开球拍的初始垂直出射角 / 球离开弦床后是否明显上飘（A 层或 B 层，任意或 S 机位）。这正是 A 需要、且 v2 的 NEW-OBS-48 曾出现过但只有一期所以没升级的那类。有了它，A 的 required 可换成出射角，离开 OBS-11 碰撞组。
3. **B vs C：** 暂留人工定。可选后续：
   - 给 C 加「肩/前臂驱动是否出现」类 OBS（与「纯低到高路径」对照）；或
   - 把 B 标成更偏症状、降低其 must_fix（**需你批准，本轮不做**）。
4. 在补 OBS 之前，这 3 对 T08 **如实保留 FAIL**。

---

## 组 2：OBS-24 = `wrist_flexes_rapidly`（触球前后手腕快速屈伸）

**共享 required：** `OBS-24=wrist_flexes_rapidly`  
**T08 对数：** 3（三角形）  
**涉及击球类型：** 全部反手（backhand），但子型不同：双手反拍 / 切削 / 单手反拍

### 节点

#### A. `N-C_LWS7XXQRRDW_C02`
- **原文：** Failing to engage the right hand and shoulder to pull the driving vector leaves the stroke purely wristy.
- **期：** `lwS7Xxqrrdw`（The Simple Foundation of the 2-Handed Backhand）— **双手反拍**
- **绑定理由：** Failing to drive through the shoulder and hand leaves the wrist flexing rapidly through contact.
- **supporting：** OBS-18=arm_first；OBS-15(b)=wraps_back_to_body_immediately；OBS-09=still_clearly_sideways
- **字幕摘录：**
  - Q08 @260s（demonstrator_induced）：「Without the right hand providing the driving vector, the back end can get really, really wristy.」

#### B. `N-C_NEPD7FIM7HI_C01`
- **原文：** Chopping at the slice with the hand and wrist instead of driving the stroke with the trunk, shoulder, legs, and core while maintaining the racket-forearm angle.
- **期：** `NEpD7fIM7HI`（Learning from Federer's Slice）— **切削**
- **绑定理由：** Chopping with wrist rather than driving with body shows arm-first forward swing and rapid wrist flexion.
- **supporting：** OBS-18=arm_first；OBS-15(b)=wraps_back_to_body_immediately；OBS-25=shoulders_lead_hips
- **字幕摘录：**
  - Q03 @80s：「…a lot of people they chop at their slice this racket would be fully straight here or even on the other side of the forearm that's not what we want」
  - Q04 @89s：「The slice comes from the trunk… it doesn't come from chopping with the hand and the wrist」

#### C. `N-C_PQ793MBQE50_C01`
- **原文：** The wrist is allowed to break freely through contact instead of using forearm tension to restrict rotation to a plane that keeps string angle stable.
- **期：** `pQ793MBQE50`（How Scap Retraction Powers the One-Handed Backhand）— **单手反拍**
- **绑定理由：** Allowing the wrist to break freely through contact directly corresponds to rapid wrist flexion.
- **supporting：** （无）
- **字幕摘录：**
  - Q11 @362s：「on the one hander we can't allow our wrist to just freely break the way it wants to」
  - Q12 @403s（demonstrator_induced）：「if I fully relax my hand and I let my wrist just naively break as I pull it wants to break here and that is right through the Hitting Zone which is a problem」
  - Q14 @438s：「…you are going to have to use at least a little bit of forearm tension to prevent this」

### 判断
- **表面都是「腕子乱动」，机制不同。**
  - A：缺右手/肩的驱动向量 → 挥拍变纯腕（双手反拍的驱动缺失）。
  - B：切削时用腕「砍」而不是躯干驱动，且破坏拍杆—前臂角（切削专属）。
  - C：单手反拍触球区允许腕自由折断，弦面角失控（平面约束缺失）。
- 同属 `stroke=backhand`，按现有合并规则**可以**进同一分组候选，但语义上合并会抹掉「缺驱动 / 砍削 / 触球区折腕」的差别，处方也会错。
- OBS-24 抓到的是共同表象；supporting 里 A/B 都有 OBS-18=arm_first，仍不足以区分。

### 建议：**补 OBS**（首选）/ **不合并** / 暂留人工定（次选）
1. **不要把这三节点合并成一个。**
2. **补 OBS（优先，可分两条）：**
   - **拍杆—前臂夹角是否在触球区保持**（B 层，S 机位）：直/弯折到前臂另一侧 —— 直接对应 B 的「砍」与「maintain racket-forearm angle」。
   - **触球区腕折方向是否改变弦面角**（B 层，S 机位）：平面内稳定 / 明显折断打乱弦面 —— 对应 C。  
   A 更适合用「非持拍侧手是否参与前拉」类观察（双手反拍）拉开，而不是继续堆在 OBS-24 上。
3. 在补 OBS 之前，这 3 对 T08 **如实保留 FAIL**；不建议为了消 T08 而硬合并。

---

## 组 3：OBS-22 = `extended_forward`（髋前顶 / 无髋铰链）

**共享 required：** `OBS-22=extended_forward`  
**T08 对数：** 3（三角形）  
**涉及击球类型：** 混杂 — 1 个双手反拍 + 2 个正手

### 节点

#### A. `N-C_LWS7XXQRRDW_C03`
- **原文：** Standing upright instead of establishing a double bend and coiling prevents trunk and hip integration for pace.
- **期：** `lwS7Xxqrrdw` — **双手反拍**
- **绑定理由：** Standing upright shows extended hips without hip hinge, accompanied by a small preparation turn.
- **supporting：** OBS-08=small_turn；OBS-14=more_upright_or_leaning_back；OBS-23=nearly_straight
- **字幕摘录：**
  - Q11 @393s（demonstrator_induced）：「stands up super tall. She's not able to get this nice double bend and engage the hips, the legs, and the trunk.」

#### B. `N-C_UJPUDUX0JIA_C01`（已含机制合并成员 `aPcKJDadzek:C06`）
- **原文：** Hips extending forward in front of the shoulders with knees forward and chest up (quad dominant posture) rather than maintaining a hip hinge.
- **期：** `uJpuDUx0jIA`（Why Your Forehand Keeps Disappearing）— **正手**；合并成员 `aPcKJDadzek:C06`（The Pros "Step In" Differently）亦正手
- **绑定理由：** Hips pushing forward under upright shoulders directly matches hips extended forward rather than hinged back.
- **supporting：** OBS-14=more_upright_or_leaning_back
- **字幕摘录（代表 + 成员）：**
  - uJpu Q05 @103s：「…driving through that unstable hicker is sending her body up…」
  - uJpu Q09 @344s：「…hitting almost as if we are doing a Romanian deadlift…」
  - aPcK Q05 @89s（成员，膝内扣语境）：「If this knee caves in early, you're going to lose your anchor and are going to over-rotate.」

#### C. `N-C_WFIRPMUTZRO_C02`
- **原文：** The player rotates purely in a single upright plane without hip flexion or multi-planar movement.
- **期：** `wFIrPMutzRo`（2 Secrets to Rotational Power | Side Bending + The Abdominal X）— **正手**
- **绑定理由：** Rotating without hip flexion means hips are extended forward under upright shoulders rather than hinged back.
- **supporting：** OBS-14=more_upright_or_leaning_back；OBS-14=lean_unchanged
- **字幕摘录：**
  - Q11 @288s：「…you're only allowing yourself to rotate in one plane of rotation and that's very limiting. So to unlock this engine, you have to be willing one to bend…」

### 判断
- **A（反拍）与 B/C（正手）不是同一诊断节点。** 画面上都是「站直、髋没铰」，但击球类型不同、解法上下文不同。T08 不看 stroke，所以跨类型也会撞；**合并规则却禁止跨 stroke**，所以这里不能靠机制合并消 T08。
- **B 与 C 高度同机制（正手、缺髋铰链 / 直立单平面转动）。** 上一轮语义合并（两次一致才并）**没有**把它们并进同一组——说明模型不完全确定。词面与绑定理由都指向同一观察（OBS-22）。若人工确认「同一件事」，合并合理；若认为 C 更强调「侧弯 / 多平面」而 B 强调「股四头主导前顶」，则应补区分 OBS（例如躯干侧倾变化，接近现有 OBS-14 但更专一），而不是强行并。
- B 已经是 MG03 的代表（与 aPcK:C06 合并），再并 C 等于扩这个簇。

### 建议：**暂留人工定**（B↔C）；**不合并** A 与 B/C；必要时 **补 OBS**
1. **A 与 B/C：不合并**（跨击球类型）。消 T08 的干净做法是让 A 的 required 不只依赖 OBS-22——例如强化 OBS-23（低球膝屈）或「准备姿态双弯是否建立」作为 A 的 required（需重跑绑定或人工指定，**本轮不做**）。
2. **B ↔ C：暂留人工定。** 两个选项请你拍板：
   - **合并：** 若你认为「正手缺髋铰链 / 直立单平面转」是同一根因，批准后下一轮用机制合并（或人工并）把 `N-C_WFIRPMUTZRO_C02` 并进 `N-C_UJPUDUX0JIA_C01`。这能消掉本组 3 对里的 1 对（B–C），A–B、A–C 仍在。
   - **补 OBS：** 若你认为 C 的「侧弯 / 多平面」必须分开，补「前挥中躯干是否出现侧倾/侧弯」类 OBS，把 C 的 required 换过去。
3. 在拍板之前，这 3 对 T08 **如实保留 FAIL**。

---

## 汇总

| 组 | 共享签名 | 对数 | 节点数 | 是否同一机制 | 建议 |
|---|---|---|---|---|---|
| OBS-11 先下再前上 | OBS-11=`down_then_forward_and_up` | 3 | 3（均正手；其中 2 条同属一期） | 否（器材出射角 vs 刷薄 vs 过度低到高） | **补 OBS**（出射角）；**不合并** |
| OBS-24 腕快速屈伸 | OBS-24=`wrist_flexes_rapidly` | 3 | 3（双手反拍 / 切削 / 单手反拍） | 否（缺驱动 / 砍削 / 触球区折腕） | **补 OBS**（拍杆—前臂角、弦面角相关腕折）；**不合并** |
| OBS-22 髋前顶 | OBS-22=`extended_forward` | 3 | 3（1 反拍 + 2 正手） | A≠B/C；B≈C（待你定） | A 与 B/C **不合并**；B↔C **暂留人工定**（合并或补侧弯 OBS） |

**本轮未做：** 改树、跑合并、编鉴别项、改 check_tree、下载视频、重跑盲测。  
**消 T08 的合法路径（需你下一步批准）：** 补 OBS 后重跑绑定；或批准 B↔C 合并；或批准改 T05/T08 与 stroke/鉴别的关系（那是规则变更，另议）。


# 动力链主干：力从哪来，怎样传到拍子（kinetic_chain_trunk）

> 版本：2026-10-20｜依据：已核对的 Stage 1 卡片原话 + 书页码（来自已有诊断文件）。
> 用途：诊断时**先认这条主干，再挂观测（OBS）和分支**；不是给场上背的口令。
> 引用规则：
> - 只引卡片里已核对的原话。
> - 6ak4ZScH_gc、KU7FHy1qQOI 本包只有卡片摘要、没有 Qnn 原话，下面标「卡片摘要」，不当原话用。
> - ui-RFRdQKRk、5jHCDc44SQM 是 **holdout 期**：可以作为知识写在本文件里，但**不得写进 tree 节点的 evidence**（check_tree T02）。
> 禁令：推肘 / 降拍 / 旋后 / 伸肘不作任何处方口令。

---

## 0. 一句话主干

**力从地面进腿，腿把髋「扭」起来（不是单纯转过去），躯干比髋扭得更多、更快，把这股扭力带进胸；胸转回来，手臂和拍子被动甩出。任何一环断了，手臂就会自己发力（FLT-02）。**

关键区别：**扭转（twist / coil）≠ 转身（rotate / turn sideways）**。
- 转身：髋和肩一起转向侧面，中间没有张力。
- 扭转：肩和躯干比髋转得更多，髋留住、甚至反向，中间像拧毛巾一样有张力。

> 「The secret to effortless power in tennis is to coil with your trunk. But, coiling means more than just turning sideways.」— ui-RFRdQKRk:Q01（holdout）
> 「Here I am not coiled. I'm just turned sideways. Here I'm actually coiled into this hip.」— ui-RFRdQKRk:Q02（holdout）

---

## 1. 链条六步

正后方 = 现在的固定机位（B）。「能见」只表示这个机位**有机会**看到，不等于模型已经能稳定判断；可靠度以每次实测一致率为准。

### 第 1 步　地面

| 项 | 内容 |
|---|---|
| 这一步在做什么 | 脚给地面施力，力的起点。脚的位置还决定身体能不能转开 |
| FTT 出处 | 书 p.84：双脚钉死时，完整转髋做不到 · KU7FHy1qQOI（卡片摘要）：动力链服从于「打到甜点」，先靠脚和眼睛把位置找对 |
| 可确认的 OBS | OBS-05 有无分步 / 调整步 · OBS-06 站位朝向 · OBS-38 站距与肩宽 · OBS-10 后脚（触球时脚跟是否离地；触球后脚是否被带转） |
| 正后方能见 | OBS-05、OBS-06、OBS-38、OBS-10 |
| 正后方不能见 | 脚底压力分布、蹬地有多用力（C 层，只能靠体感） |

### 第 2 步　腿的加载

| 项 | 内容 |
|---|---|
| 这一步在做什么 | 后腿弯下、坐进去，存住可以往上送的力 |
| FTT 出处 | 书 p.111：从击球腿发力蹬出，髋就不容易转不够 |
| 可确认的 OBS | OBS-23 低球触球时的膝屈 · OBS-22 髋折叠还是前顶 · OBS-13 重心位置 |
| 正后方能见 | 只能粗看「站得直不直」 |
| 正后方不能见 | 膝角深浅（OBS-23 需侧面 S）、髋折叠（OBS-22 需 S）、重心前后（OBS-13 需 S） |

### 第 3 步　髋的扭转（关键一步）

| 项 | 内容 |
|---|---|
| 这一步在做什么 | 准备时髋**不跟着**肩转走，坐进后髋形成扭力；前挥时髋先往前发，再刹住，把力交给躯干 |
| FTT 出处（准备端） | 「The unit doesn't include your hips. If I go backwards with my hips as well, then I create no tension...」— ui-RFRdQKRk:Q03（holdout）。对应卡片成因 C01（髋随 Unit Turn 一起往后转，没有反向留住、没有坐进臀）、C03（缺少等量反向的扭） |
| FTT 出处（前挥端） | 6ak4ZScH_gc（卡片摘要）：骨盆先往前发，预先拉长筋膜，然后肩超过它；体感是「把骨盆往回拽」，实际骨盆在往前走 |
| 可确认的 OBS | OBS-25 肩先还是髋先（期望 hips_lead_shoulders） · OBS-71 骨盆在前挥中是否减速（tree_v15，题面以 catalog 为准） · OBS-08 准备结束时肩线和髋线转离多少 · **NEW-OBS 候选**（见下） |
| 正后方能见 | OBS-25、OBS-08（B 机位题），OBS-71 按 catalog 所标机位 |
| 正后方不能见 | 臀部是否「坐进去」、有没有张力（C 层） |

**NEW-OBS 候选**：OBS-08 把「肩线和髋线」放在一题里问，分不出「扭」和「转」。
- 中立问法：*At the end of the preparation turn, is the shoulder line turned farther from the target than the hip line, about the same, or less?*
- 取值：`shoulders_farther_than_hips` / `about_the_same` / `shoulders_less_than_hips`
- 机位：正后方（B）

### 第 4 步　胸 / 躯干

| 项 | 内容 |
|---|---|
| 这一步在做什么 | 腹部把髋的扭力接住，躯干比髋扭得更多、转得更快，带着胸转回来，触球时胸接近正对 |
| FTT 出处 | 「secret ... is in the abdomen.」— 5jHCDc44SQM:Q02（holdout）· 躯干 / 肩线 / 胸比髋扭得更靠后 — 5jHCDc44SQM:Q03 · 「trunk moving faster than hips for pretty much the entire swing」— 5jHCDc44SQM:Q04 · 只靠髋去发、腹部没接上，挥拍就散、张力就丢 — 5jHCDc44SQM:Q05–Q06 · 书 p.30：触球时髋和胸仍侧对，手臂被卡住 |
| 可确认的 OBS | OBS-09 触球时髋是否转回 · OBS-19 躯干在触球前减速还是一直加速 · OBS-29 是否有侧屈 |
| 正后方能见 | OBS-09、OBS-19、OBS-29（都是 B 机位题）；肩线是否半开 |
| 正后方不能见 | 腹部是否「接上」（C 层）；躯干前倾变化（OBS-14 需 S） |

### 第 5 步　手臂

| 项 | 内容 |
|---|---|
| 这一步在做什么 | 手臂跟着胸被带出去，不自己先拉 |
| FTT 出处 | 书 p.83：「the answer is not to pull with your arm」· 书 p.116：「Power does not come from swinging harder with the arm」· FLT-02 |
| 可确认的 OBS | OBS-40 手臂和胸的转动是否同步 · OBS-18 哪个部位先往目标动 · OBS-32 肘是否直线拖过躯干 · OBS-36 手先往外还是直奔球 · OBS-02 触球时手离躯干多远 |
| 正后方能见 | OBS-40、OBS-18、OBS-32、OBS-36、OBS-02（都是 B 机位题） |
| 正后方不能见 | 手臂肌肉是否在主动用力（C 层） |

### 第 6 步　拍子

| 项 | 内容 |
|---|---|
| 这一步在做什么 | 拍头被甩过球，滞后和随挥都是被动结果 |
| FTT 出处 | 书 p.113（转述）：让拍子朝目标方向穿过球，挥拍在触球那一刻就算结束 |
| 可确认的 OBS | OBS-15 随挥 · OBS-16 每球结果 · OBS-35 拍头落后手多少 |
| 正后方能见 | OBS-15、OBS-16 |
| 正后方不能见 | OBS-35（需 S）、拍面角度（OBS-28 需 S）、击球点前后（OBS-03 需 S） |

---

## 2. 三个断点

断点只说「链在哪断」。断点之下挂的才是分支，不再并列成第一要害。

### 断点 A：脚钉死 / 腿没加载（第 1–2 步）
- **含义**：地面的力没进链。按书 p.84，这会让第 3 步的完整转髋做不到。
- **性质**：通常是**上游条件**，不单独当根因。在发球机慢节奏下停脚很常见，这是情境，不是动力链故障本身。
- **确认用 OBS**：OBS-05、OBS-38、OBS-10（正后方）；OBS-23（侧面）。

### 断点 B：髋没扭起来 / 胸没打开（第 3–4 步）
这是用户本人认定的要害：髋活动度和扭转没掌握，下半身的扭力上不来。细分四种：

| 子断点 | 说的是什么 | 出处 | 确认用 OBS |
|---|---|---|---|
| B1 | 准备时髋跟着肩一起转走，只是「转身」，没有张力 | ui-RFRdQKRk:Q02/Q03 · C01（holdout） | NEW-OBS 候选（肩线 vs 髋线）· OBS-08 |
| B2 | 躯干没有比髋扭得更多 | 5jHCDc44SQM:Q03 · C01（holdout） | NEW-OBS 候选 |
| B3 | 髋发了，但腹部没接上，力散了 | 5jHCDc44SQM:Q05–Q06 · C02（holdout） | OBS-25 · OBS-19（间接）；「接上」本身是 C 层 |
| B4 | 前挥时髋没领先、胸没转回，触球时仍侧对 | 书 p.30 · FLT-05 · 6ak4ZScH_gc（卡片摘要） | OBS-25 · OBS-09 · OBS-71 |

### 断点 C：手臂脱离胸，自己发力（第 5 步）= FLT-02
- **含义**：主干故障的表现形式。A 或 B 断了，手臂就补上来；也可能 A、B 都没问题，只是手臂抢先。
- **确认用 OBS**：OBS-40（arm_lags_behind_chest / moves_together_with_chest）、OBS-18（arm_first）、OBS-32（elbow_drags_linearly）。
- **常见下游**：随挥马上缠回身体（OBS-15b wraps_back_to_body_immediately）、球软、不稳定（p.116）。

### 断点之间的关系

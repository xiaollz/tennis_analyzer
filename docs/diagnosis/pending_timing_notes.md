# 待核实时序要点登记（2026-10-18）

> 性质：**登记，不是结论。** 不改树、不改规则、不跑模型、不下视频。  
> 对应 learning.md：`docs/record/learning.md` 的 `2026-10-18 旋转与 supination 同步`（分支 `cursor/learning-1018-rotation-supination`，commit `25bad5f`）。

## 1. 要点（用户新补充，用户标为「非常重要」）

- **用户转述**：TPA 教练 Tom Alsop 有一集说 rotation 与 supination 应该一起——weight transfer、身体 rotation 时**能感觉到** supination，两者合起来大概就是正确时序。
- **用户自己的感觉**：身体负责旋转，胳臂只负责 supination（外旋），**不需要再往前挥或往上提**。
- **状态**：Tom Alsop 那一期**尚未拿到字幕，原话未核实**，以上全部是用户转述 / 体感。

## 2. 与禁令的关系（写法约束）

- 永久禁令：不把「旋后 / 推肘 / 降拍 / 伸肘」作为主动口令（`config.json` 的 `ban_patterns`、R09/T06 会扫描）。
- 因此本要点只能写成：**supination 是身体旋转带出的结果 / 觉察点**，不是手去做的动作指令；主语是「身体在转」，「胳臂只负责」＝不再额外加动作。
- 进入树时**不能**作为 `next_one_thing` / 练习口令；只能作为**录像里的观察项**或待核对问题。

## 3. FTT 对应证据（间接；FTT 没有专门一期讲「旋转与 supination 同步」）

| 期 | 标题 | 约时间 | 内容 | 在本库的状态 |
|---|---|---|---|---|
| `PwkGLkRm73o` | The Pros Swing Differently Than You | 0:44–1:10（`supinate` 在 1:05） | 手臂不必往上走；投掷中前臂 supinate/pronate，拍头自己 low-to-high | 无卡片（manifest：`subs_ready`，非 holdout，尚未做 stage1 抽取）；字幕在 `/workspace/ftt_subs/` |
| `2Nn9tyMECZU` | Why Pros Swing Around, not Low-to-High | 0:13–0:25 | 肩在背稳定下装载外旋，再飞过变内旋；往上是绕出来的 | 卡片 `stage1_verified`，已在 tree v1/v2/v3 |
| `JIMgI3jiVns` | How Shoulder Rotation Syncs Your Contact | 2:36–4:20 | 触球在肩开始向前滚的起点，大部分滚动在触球后 | 卡片 `stage1_verified` 但**低置信**，build_tree 已排除；**需用户核字幕** |
| `LoSrMj-Qxkc` | Their Mothers are Sisters, Their Forehands AREN'T | 约 1:10–1:35 | 肩外旋 layback → 内旋 windshield wiper | 无卡片 |
| `E_zmENJIj4g` | Coiling \| Spinal Motion \| Eye Dominance \| Arm Slot | — | 发球 | 卡片 `stage1_draft`，**未核实**，不引用 |

说明：FTT 的共同方向是「手臂不额外往上/往前去做，拍头的 low-to-high 与滚动来自旋转 + 前臂 supinate/pronate（或肩外旋→内旋）」。「旋转与 supination **同时发生**就是正确时序」这句话**只来自用户转述的 Tom Alsop**，不是 FTT 结论。

## 4. 与现有观察项的关系

| 现有项 | 它问什么 | 与本要点的关系 |
|---|---|---|
| OBS-18 | 前挥启动时哪个部位先动（髋 / 躯干 / 手臂） | **相关，不同**：OBS-18 问「谁先启动」；本要点问「手臂有没有**额外**去做」。若 OBS-18=`arm_first`，可能同时出现「手臂额外往前/往上」，但反过来不成立（身体先动，手臂仍可能额外加动作）。 |
| OBS-25 | 前挥启动时肩和髋哪个先转 | **相关，不同**：同为「顺序」类，但只比较躯干内部；不涉及前臂。与 OBS-18 已有语义重叠（见 `t08_review_obs11_22_24.md` 与 tree_v3 报告的提醒），本要点不要再往这两项上堆。 |
| OBS-11 | 前挥启动到触球手的路径（水平向前 / 先下再前上 / 先上抬） | **最接近**：「手臂额外往前 / 往上提」会反映在手的路径上；但 OBS-11 看的是手的轨迹，不区分「被旋转带着走」还是「手自己去做」。 |
| OBS-24 | 触球前后腕是否快速屈伸 | 不同：腕屈伸，不是前臂 roll。 |
| OBS-15(b) | 触球后手是向目标延伸还是马上绕回 | 弱相关（随挥形态）。 |

## 5. 可能的新观察点（候选；未升级、未进 obs_catalog.json、未跑绑定）

> 登记候选，供用户批准后再走 `promote_new_obs.py` 的「人工批准」通道（单期 / 无跨期证据，需明示批准）。题目与取值必须中立，不含处方。

- **候选 A：前臂 roll 是被身体带着，还是手臂额外加动作**  
  问法（中立英文草案）：*During the forward swing, does the hitting forearm roll (supinate) while the hand keeps travelling with the rotating torso, or does the hand/arm add extra forward or upward travel of its own?*  
  取值草案：`rolls_with_torso_no_extra_travel` / `extra_forward_travel` / `extra_upward_lift`  
  层级 / 机位草案：B / S（前后与高低在画面平面内）。  
  注意：这是**观察项**，描述「看到了什么」，不是 cue；不得写成「去旋后」。
- **候选 B（可选）：触球前后前臂旋转量**（前臂在触球后是否出现明显 supinate/pronate 滚动）——仅用于核对 FTT `PwkGLkRm73o` 一类的描述，与候选 A 的关系待定。

## 6. 待核实清单

1. **Tom Alsop 那一期**：用户提供链接 / 标题 → 取字幕 → 逐字摘录（≤40 词）+ 时间戳 → 判断他说的是「rotation 与 supination 一起」还是「supination 因 rotation 出现」。
2. **`JIMgI3jiVns` 字幕**：用户对着视频核 2:36–4:20（低置信卡片）。
3. 是否把候选 A 升级为 OBS（需用户明确批准，因为只有用户转述 + 单期间接证据）。
4. `PwkGLkRm73o` 是否做 stage1 卡（manifest 状态 `subs_ready`、非 holdout、尚无卡片；它是 FTT 唯一明说 supinate 的一期，建议优先）。
5. 在以上核实前，**不得**把本要点写入任何 `next_one_thing`、练习或节点原文。

## 7. 追加（2026-10-18 晚）：手是被动被带着 supinate，还是主动往前拉？

- **用户补充（意识层面，用户标为非常重要）**：意识里只关注 weight transfer（重心转移，右脚转到左脚），在这个过程中让手**被动地**做 supination（前臂旋外），而**不是主动往前拉**。对应 learning.md 2026-10-18 条目 §7（分支 `cursor/learning-1018-rotation-supination`）。
- **待核实项**：球场录像里，手的前臂旋转（roll）是随着重心转移 / 身体旋转**被带出来**的，还是手臂**自己额外往前拉 / 往前送**的？
- **与候选观察点 A 的关系**：这就是候选 A 的核心区分——取值草案里的 `rolls_with_torso_no_extra_travel` 对应「被动被带着」，`extra_forward_travel` 对应「主动往前拉」。本项只是给候选 A 增加一个用户意识层面的验证动机，**不改 A 的题目或取值**，也不升级为 OBS。
- **可见性提醒**：「意识里在关注什么」（注意力）从视频里**看不到**，只能看到结果（手有没有额外往前走）。因此即使录像里看到「没有额外往前」，也不能反推用户当时的注意力放在哪里；注意力部分只能靠用户自述，不进树、不进 OBS。
- **写法约束不变**：只能记为观察项 / 待核对问题，不得写成「让手被动 supinate」之类的口令，也不得进入 `next_one_thing`。
- **待办（并入第 6 节）**：6.6 拿到球场录像后，对照候选 A 逐拍标注（被带着 roll / 额外往前 / 额外往上），并与用户当拍自述的注意力对照，仅作个人记录。


## 8. 核实进展（2026-10-19）

- **主链接（高置信）**：https://www.youtube.com/watch?v=dw4hymptl9k — TPA《The Truth About Forehand Supination and Pronation》。自动字幕含 *body rotates as you supinate* 与 *transferring to my left leg, I'm supinating at the same time*。详见 `docs/diagnosis/coaches/tom_allsopp_rotation_supination_2026-10-19.md`。
- **次链接（中高）**：https://www.youtube.com/watch?v=ubFJi2M3AMM — 《Improving Your Forehand Kinetic Chain…》；*rotate the body to supinate the arm*（顺序表述）。
- 待用户打开主链接确认是否即所指那集；确认前仍不进 `next_one_thing` / 不升级 OBS。

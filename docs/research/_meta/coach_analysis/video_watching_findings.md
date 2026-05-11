# 教练视频实地观看综合发现（FTT + Tom Allsopp）

> 来源：通过 Gemini 3 Flash Preview 直接观看 10 个 YouTube 视频原片得到的视觉级笔记。
> 原始 JSON 见 `knowledge/extracted/coach_videos_v2/{coach}_{video_id}.json`。
> 本文目的：补充纯文字研究无法覆盖的视觉细节、机位选择、错对动作差异，
> 并提炼可直接写入 VLM prompt / 知识图谱的新观察点。

## 1. 视频清单

| # | 教练 | Video ID | 标题 | URL |
|---|------|----------|------|-----|
| 1 | Johnny (FTT) | 0m3BMfDDShI | Build This Foundation – The Rest Will Follow | https://www.youtube.com/watch?v=0m3BMfDDShI |
| 2 | Johnny (FTT) | pWzyP-xfLfU | Why Your Forehand Doesn't Have Lag (The Pivot Point) | https://www.youtube.com/watch?v=pWzyP-xfLfU |
| 3 | Johnny (FTT) | BbGzWTp5pCM | Why You're Good in Practice but Bad in Matches (Nadal's Secret) | https://www.youtube.com/watch?v=BbGzWTp5pCM |
| 4 | Quentin MF (FTT) | FOmz8Wjv3DQ | Roger Federer's Footwork Secrets | https://www.youtube.com/watch?v=FOmz8Wjv3DQ |
| 5 | Johnny (FTT) | wFIrPMutzRo | Rotational Power — Side Bending + X Stretch | https://www.youtube.com/watch?v=wFIrPMutzRo |
| 6 | Tom Allsopp (TPA) | Vcg_HcHaQ34 | How to get more power on your forehand (Torque & Rotation) | https://www.youtube.com/watch?v=Vcg_HcHaQ34 |
| 7 | Tom Allsopp (TPA) | CmXxvX60TOI | Early Preparation on the Forehand | https://www.youtube.com/watch?v=CmXxvX60TOI |
| 8 | Tom Allsopp (TPA) | ubFJi2M3AMM | Lag Timing — Unit Turn must finish first | https://www.youtube.com/watch?v=ubFJi2M3AMM |
| 9 | Tom Allsopp (TPA) | utZkaHi9XXM | Takeback is the result of the turn (not the arm) | https://www.youtube.com/watch?v=utZkaHi9XXM |
| 10 | Tom Allsopp (TPA) | M1umUwuPe0w | Active vs Passive Wrist on the Forehand | https://www.youtube.com/watch?v=M1umUwuPe0w |

---

## 2. 每段视频一句话摘要

1. **Build This Foundation (Johnny)** — 用 2.5 lb 重量片演示"球拍很轻但结构要稳"；核心是 Elbow Space + 背部肌肉支撑 + Unit Turn / Unit Swing；错误状态在视觉上是"准备时肘部下垂、挥拍时拍头乱晃"。
2. **Pivot Point (Johnny)** — 球拍的旋转支点必须在虎口（食指根部），不在手柄底盖；用"两指挥拍"（拇指+食指）drill 强迫支点上移，回归全握后保留触觉。
3. **Match Intensity / Multi-Split-Step (Johnny)** — 业余练习是"slow motion"，比赛级别要求站位更宽 + 持续小跳分裂步 + 强力呼气，听得见喷气声。Multi-Split-Step 是面对球机时保持神经兴奋的核心 drill。
4. **Federer Footwork (Quentin)** — Glide Step + Triple Bend（踝/膝/髋同时弯）+ 分裂步要"跳得够高、够晚"，落地前用眼睛识别球的方向以实现"落地即启动"；用弹力带 in-and-out + 单脚跳 + 双手背后侧跳来开发。
5. **Rotational Power / Side Bending (Johnny)** — 直立旋转 = 机械僵硬，正确是脊柱侧弯 + 肩髋 X 拉伸；用药球对拉 + PVC 长杆锁臂 drill 强迫躯干发力。
6. **Torque & Rotation (Tom)** — 深球时用关闭式 = 关节锁死无扭矩；正确是开放式 + 肩比髋多转 ~45°；高尔夫挥杆类比；标注"Too Sideways = No Torque"。
7. **Early Preparation (Tom)** — 错误是左手抓拍喉太久 + 球拍僵在身前；正确是连续圆弧引拍 + 高肘 + 引拍速度匹配来球速度（"slow ball, slow takeback"）。
8. **Lag Timing (Tom)** — 错误是手腕主动后撇制造 lag → 击球瞬间手臂"冻结"，拍头 wobble；正确是身体先动、手臂作为整体旋后（supination），lag 是结果不是动作。
9. **Takeback Style (Tom)** — ATP 高肘旋前 vs WTA 大弧度旋后引拍的选择应基于生理结构（肩膀宽度、手肘并拢柔韧性测试）；强迫学生模仿不适合的风格 → 动作僵硬。
10. **Active vs Passive Wrist (Tom)** — 中性手腕引拍；手腕后仰必须发生在"前挥过程"，不是"引拍过程"；用"摸狗"和"摸马"图片演示拍头下落高度随击球高度而变。

---

## 3. 跨视频洞察：纯文字研究没有捕捉到的东西

### 3.1 机位证据

- **Tom 几乎所有 Unit Turn 错误诊断都用"分屏 + 正前方 / 正侧面"组合**，而不是仓库笔记里假定的"后方斜 45°"。
  实际上他用的"画中画对比"（学生画面 + 教练同步示范）才是他真正的诊断界面。
  → 仓库 `tom_allsopp_unit_turn.md` §3 写的"Tom 的主诊断角是后方斜 45°"在 Vcg_HcHaQ34 是对的，
  但在大多数 Unit Turn 视频里 Tom 实际上是 **正前方 + 分屏侧面**。我们的 VLM prompt 不应硬性要求后方 45°。

- **Johnny FTT 几乎从不用"后方斜 45°"作为主机位**。FTT 视频以"正前方近景 + 侧面学生击球"为主，
  原因是 FTT 强调"micro-zone bonding / pivot point / 肘部空间"等**手腕与肘部局部细节**，
  这些在后方 45° 反而看不清。

- **跟拍机位 (跟随教练走路)** 在 Quentin 的 Federer 步法视频中是关键：
  侧跳 drill 必须用跟拍才能展示"双手背后 + 单脚平衡"的过程，固定机位会丢失。

### 3.2 错对对比的视觉特征（从看到的画面提炼，不是文字推断）

| 错误 | 视觉信号（从画面观察） | 出现视频 |
|------|----------------------|---------|
| Slow-motion 练习 | 等球时**身体完全静止**，没有任何小跳；脚距小于肩宽 | BbGzWTp5pCM |
| Stop-Start syndrome | 引拍到顶后球拍**真正不动**，肉眼可见停顿帧 | CmXxvX60TOI |
| Pivot 错位 | 拨动拍头时整个手柄**绕底盖**摆动而不是绕虎口 | pWzyP-xfLfU |
| 直立旋转无 side bend | 击球瞬间**脊柱与地面垂直**，没有向左侧倾 | wFIrPMutzRo |
| 关闭式深球 | 左脚跨过身体中线**朝来球方向**，髋已锁死 | Vcg_HcHaQ34 |
| 主动 wrist lag | 引拍过程中**腕部先于肩部产生角度变化** | ubFJi2M3AMM, M1umUwuPe0w |
| 强迫风格不匹配 | 学生引拍时球拍**没有越过身体中线**且节奏破碎 | utZkaHi9XXM |
| Triple Bend 缺失 | 准备站位时**双腿近乎直立**、踝几乎不弯 | FOmz8Wjv3DQ |
| 肘下垂 / 无 Elbow Space | 准备时**肘部贴肋骨**，球拍头低于胸 | 0m3BMfDDShI |

### 3.3 教练之间的"隐藏共识"

1. **三位教练都明确反对"早早把动作做完然后等球"**（FTT: Multi-Split-Step；Tom: stop-start syndrome；Johnny: 慢动作练习陷阱）。
   这是仓库笔记里写过的概念，但**通过看视频才意识到三家用的是同一类视觉信号**：
   "等球时下肢有没有持续低幅度跳动"。这是一个**单帧不可判断、必须看连续 30 帧**的特征。
2. **三位都把'手腕'描述为被动结果，但触发时机不同**：
   - Tom: 必须在 Unit Turn **完成后** 才允许 lag。
   - Johnny: 通过 Pivot Point（虎口支点）让 lag **物理上不可能用手腕主动制造**。
   - Quentin/Tom: 强调 lag 来自身体加速 + arm supination，不是 wrist。
   → 这三种说法在 VLM 的诊断输出里应**收敛到同一个 hypothesis**："手腕主动 = 上游缺失的代偿"。
3. **"球速决定引拍速度"是 Tom 视频的明确口令**，但仓库笔记没把它列为可执行规则。
   这是一个**非常实用的诊断维度**：观察学生引拍速度是否与来球速度匹配。

### 3.4 文字研究遗漏 / 偏差校正

- 仓库 `tom_allsopp_unit_turn.md` §1 列出"左手放拍颈"是 Tom 的核心 cue，
  但实际看 Tom 的 5 个视频，**他几乎不强调左手位置**（CmXxvX60TOI 提了一次但不是核心）。
  Tom 真正反复说的是 **"shoulder turn"** 和 **"continuous loop"**，左手位置更像是 FTT/Feel Tennis 的术语。
- 仓库笔记假设 Tom 用"45°+5° golf model"作为统一模型，但实际只在 Vcg_HcHaQ34 出现一次。
  其他 4 个 Tom 视频里他**根本不提具体角度数字**，而是强调"separation"的感觉。
  → VLM 不应把"分离角必须 45°+5°"当作硬阈值套到所有 Tom 风格的诊断上。
- Federer 步法视频里"分裂步落地前必须先看到球的方向"是一个**仓库笔记完全没有的时序细节**——
  这是一个真正的新观察点。

---

## 4. 应整合进 VLM Prompt / 知识图谱的新观察

### Top 5 新视觉洞察（建议直接写入 VLM prompt）

1. **"等球阶段的下肢活跃度"是练习/比赛差距的视觉决定信号。**
   要求 VLM 观察击球之间的 ~30 帧（0.5 秒）窗口：脚是否在持续低幅度跳动？站位是否 ≥ 肩宽？
   规则：连续 30 帧无脚部位移 → 触发 `slow_motion_practice`。

2. **球拍的"旋转支点"可以从单帧握拍特写判断。**
   观察手柄的握法：底部三根手指是否紧握？拇指与食指是否形成 V 字？
   如果底部手指握紧 + 手腕僵直 → 触发 `pivot_at_butt_cap`，标记为 lag 缺失的结构性原因（不是动作问题）。

3. **引拍速度应与来球速度匹配，不是越早越好。**
   要求 VLM 计算"引拍角速度 / 来球速度"比例。
   如果两者比例 > 2.0（引拍远快于来球） → 触发 `early_completion_then_wait`，
   即使动作幅度看起来标准也要标记为错误。

4. **分裂步的"落地前预判"是步法效率的关键时序。**
   观察分裂步在空中相位时，球员的躯干 / 视线方向是否已经朝向来球落点？
   如果落地后才出现明显的躯干转向 → 触发 `late_split_recognition`。

5. **脊柱"侧向倾斜角"是转动力量的核心证据。**
   击球瞬间从正面或正前方机位观察脊柱与地面的夹角。
   完全垂直（接近 90°）→ 触发 `no_side_bending` → 上传到诊断引擎作为"力量来源缺失"的根因之一。

### 知识图谱补充

- 新增概念 `V2-01: pivot_at_butt_cap` (准备 / 握拍 / structural)，causes → `lag_missing` (C11)。
- 新增概念 `V2-02: late_split_recognition` (准备 / 步法 / timing)，causes → `late_first_step`。
- 新增概念 `V2-03: takeback_speed_mismatch` (准备 / 时序 / behavioral)，causes → `early_completion`。
- 新增 cue/规则："slow ball → slow takeback；fast ball → fast takeback" 应作为时序层的诊断维度，独立于"引拍幅度是否到位"。
- 把 Tom 的"45°+5° golf model"从硬阈值降级为"风格性参考"，仅在 Vcg_HcHaQ34 / Tom-style 诊断时启用。

### Prompt 文本草稿（可直接合并到 vlm_analyzer 的 Pass 1 检查表）

```
[新增观察点]
A. 等球阶段（击球间隙 30 帧窗口）的下肢活跃度：脚是否持续小跳？站位 ≥ 肩宽？
B. 握拍特写（如有可见帧）：底部三指是否紧握？支点是否在虎口（V 字）？
C. 引拍速度 vs 来球速度比：是否 > 2.0？(早完成然后等球)
D. 分裂步空中相位：躯干/视线是否已转向来球？(落地前预判)
E. 击球瞬间脊柱-地面角：< 75° (有侧弯) / ≥ 85° (无侧弯，僵硬旋转)
```

---

## 5. 失败记录

- 第一次运行时 `tom/ubFJi2M3AMM` 因 packyapi 503（系统 CPU 过载）失败，
  等待 ~15 秒后重跑成功（70.2s 输出 1925 字符）。
- 其他 9 个视频均一次成功，每段平均处理时间 60–90 秒，
  总体在预算范围内（< 30 分钟）。

---

## 附：原始数据文件

```
knowledge/extracted/coach_videos_v2/
├── ftt_0m3BMfDDShI.json
├── ftt_BbGzWTp5pCM.json
├── ftt_FOmz8Wjv3DQ.json
├── ftt_pWzyP-xfLfU.json
├── ftt_wFIrPMutzRo.json
├── tom_CmXxvX60TOI.json
├── tom_M1umUwuPe0w.json
├── tom_Vcg_HcHaQ34.json
├── tom_ubFJi2M3AMM.json
└── tom_utZkaHi9XXM.json
```

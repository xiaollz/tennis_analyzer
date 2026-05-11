# 视频分析的拍摄角度方法论

研究目的：为基于 VLM 的正手视频诊断系统确定最优拍摄协议。
研究范围：免费 YouTube/教练博客内容，涵盖 Tom Allsopp、Feel Tennis、Online Tennis Instruction、Tennis Techie、TopspinPro、My Tennis Tools、Kovacs Institute、TCPR (Brian Gordon) 等。

---

## 1. 不同诊断目的对应的最佳角度

| 想看什么 | 推荐角度 | 原因 | 备选 |
|---|---|---|---|
| **Unit Turn 完成度**（肩转角度、非持拍手指向） | **正后方（baseline 后中线）** | 唯一能直接量化"肩线相对底线的旋转角度"和非持拍手是否横过身体的视角；侧面会被身体遮挡 | 高位 45°后方（兼顾深度） |
| **步伐时序 / 调整步**（split step、carioca、recovery） | **低位侧面**（球网延长线、地面高度 30–60 cm） | 脚离地高度、步幅、步频在侧面最清晰；低位放大垂直位移，便于看 split step 落地瞬间 | 正后方（看横向覆盖） |
| **准备阶段时机（相对来球）** | **侧面 + 必须把来球弹跳点纳入画面** | 只有侧面能同时看到"球过网→弹跳→击球"和"持拍手开始引拍"的相对时间轴 | 高位后方（也能看到来球，但深度判断差） |
| **引拍高度 / 拍头轨迹（loop vs straight）** | **正后方** | 拍头从高到低的环形路径在后方呈现为清晰的弧线；侧面会因透视压缩 | 持拍手侧斜后 45° |
| **重心位置 / 平衡（前后、左右）** | **正面（隔网对面）或正后方** | 正面看左右倾斜，后方看前后倾斜；侧面只能看其中一维 | 双机位 |
| **接触点位置（前后、高低）** | **侧面（持拍手同侧）** | 接触点相对前脚的前后距离只有侧面能准确测量 | — |
| **杆头滞后 / 旋转鞭打** | **正后方 + 高帧率** | 后方能看到"躯干转→手臂跟→拍头最后到达"的时序 | 持拍侧斜后 45° |
| **髋肩分离 (X-factor)** | **正上方俯视 或 高位正后方** | 需要同时量化髋线和肩线的水平角度差 | 正后方略高 |

**核心结论**：单一角度无法覆盖所有诊断目的。如果只能选一个，**持拍手对侧的低位侧面（含来球弹跳）** 信息密度最高；如果能选两个，**侧面 + 正后方** 是黄金组合（这是 TCPR、Kovacs Institute、Onform 的标配）。

---

## 2. 帧率要求

| 诊断目标 | 最低帧率 | 推荐 | 原因 |
|---|---|---|---|
| 整体节奏、Unit Turn 时机 | 30 fps | 60 fps | 30 fps 在快速引拍时单帧位移过大，时序判断误差 ±33 ms |
| Split step 落地瞬间 | 60 fps | 120 fps | split step 落地窗口约 50–80 ms，60 fps 仅 3–5 帧，120 fps 可定位到 ±8 ms |
| 拍头滞后、鞭打瞬间 | 120 fps | 240 fps | 前臂内旋 + 手腕释放发生在 ~30 ms 内 |
| 接触瞬间球拍形变、入射角 | 240 fps | 480 fps+ | 球与拍接触约 4–5 ms |

**对当前 VLM 系统的建议**：**60 fps 是底线，120 fps 是甜点**。VLM 不需要 240 fps（帧太多反而稀释关键信息），但 split step 与 Unit Turn 的诊断在 30 fps 下会丢失关键时序帧。来源：Onform 支持 240 fps；专业生物力学实验室使用 300 fps 以上；普通教练实践 60–120 fps 已足够诊断技术问题。

---

## 3. 距离与构图

- **画面高度**：必须包含 **完整全身 + 头顶留 0.5 m + 脚下留 0.3 m**。半身画面会丢失下肢动力链信息，是业余拍摄最常见错误。
- **画面宽度**：**至少容纳球员左右各 2 m**（覆盖侧向移动）；侧面机位还要 **额外纳入来球的最后一个弹跳点**（约球员前方 2–3 m）。
- **相机距离**：侧面机位距球员 **6–8 m**（避免广角畸变又保证全身入镜，搭配 24–35 mm 等效焦距）。
- **相机高度**：
  - 侧面/后方：**腰高（1.0–1.2 m）** —— 看动力链和重心
  - 步伐专项：**膝高（0.4–0.6 m）** —— 放大脚部位移
  - 髋肩分离：**肩高或更高（1.6 m+）** —— 减少透视压缩
- **稳定性**：必须用三脚架。手持/地面随意放置会引入抖动，破坏 VLM 对小幅动作的判断。

---

## 4. 单机位 vs 多机位

- **顶级实验室（TCPR / Kovacs Institute）**：使用 markerless 3D 动捕，8+ 同步相机 300 fps。普通用户无法复制。
- **职业教练日常**：**2 机位是标准配置**——一个侧面、一个后方，事后用 Onform/Coach's Eye 同步对比。
- **业余 + VLM 用户**：单机位可行，但要 **每次训练轮换角度**：第 1 组侧面、第 2 组后方、第 3 组 45°斜后。这样跨 session 累积出多角度数据集，VLM 可分别诊断不同维度。

---

## 5. 给用户的录制建议（最终方案）

**主协议（每次训练强制）**：
1. **机位**：手机/相机置于 **持拍手对侧的侧面**（右手球员→相机在球员左侧），三脚架，**腰高 1.1 m**，距球员 **7 m**。
2. **构图**：球员居于画面右 1/3（左手球员则居左 1/3），左侧（来球方向）留出 3 m 空间纳入弹跳点；上下留白如上。
3. **帧率**：**1080p @ 60 fps 起步**，手机若支持 1080p @ 120 fps 优先用 120 fps（iPhone "慢动作"模式即可，但要确认输出文件是 120 fps 而非播放速率被改写）。
4. **光线**：**顺光或顶光**，避免逆光（剪影会让 VLM 完全失效）；阴天最佳；强烈直射阳光下避免半身处于阴影、半身处于强光的混合状态。
5. **背景**：选择 **球场围网或单色挡布**，避免画面里有其他移动球员/观众；纯净背景能让姿态检测精度提升 20%+。
6. **着装**：**纯色上衣 + 与裤子对比明显**，便于关键点检测识别躯干和四肢。
7. **每组录满 6–10 拍**，开始前空打一拍让相机自动对焦稳定。

**辅助协议（每周至少 1 次）**：
8. **后方机位**：相机置于 **球员正后方底线后 3 m，高度 1.6 m（略高）**，专门拍 Unit Turn、髋肩分离、引拍轨迹。
9. **45°斜后机位**：作为周期性补充，用于交叉验证。

**禁忌**：
- 禁止逆光、禁止半身画幅、禁止手持、禁止 30 fps、禁止把相机放在球网柱顶端俯拍（透视失真）、禁止使用过广的鱼眼镜头。

---

## 6. 来源

- [Online Tennis Instruction — Tennis Swing Analysis](https://www.onlinetennisinstruction.com/tennis-swing/)（45° 前角为主推角度）
- [TopspinPro — Analyze your Tennis with Video](https://topspinpro.com/blog/analyze-your-tennis-with-video-for-dramatic-improvement/)（侧面看接触点和重心）
- [Tennis Techie — Best Practices for Technical Video Analysis](https://www.tennistechie.com/blog/2018/9/10best-practices-for-video-analysis)（多角度建议）
- [My Tennis Tools — Filming Tennis in Slow Motion](https://mytennistools.com/filming-tennis-in-slow-motion/)（120 fps 设备建议）
- [My Tennis Tools — How to Film a Tennis Match Part 1](https://mytennistools.com/how-to-film-a-tennis-match-part-1/)
- [Jean Galea — How to Record Your Tennis Strokes](https://jeangalea.com/record-tennis-strokes/)（相机距底线一英尺的细节）
- [Onform — Sports Video Analysis App](https://onform.com/)（240 fps 行业标准）
- [Brookside Tennis — Video Stroke Analysis](https://brooksidetennistraining.com/video-stroke-analysis)（300 fps 教练实践）
- [Kovacs Institute — Serve Screen](https://kovacsinstitute.com/servescreen.html)（多机位标准化协议）
- [Tennis Center for Performance Research — Dr. Brian Gordon](https://tennisperformanceresearch.com/dr-brian-gordon/)（markerless 3D 动捕）
- [TennisPlayer — Brian Gordon and the Simi Tennis Application](https://www.tennisplayer.net/public/biomechanics/sean_oneil/brian_gordon_simi/)
- [Personal Best Tennis — Video Analysis](https://www.personalbesttennis.com/video-analysis)

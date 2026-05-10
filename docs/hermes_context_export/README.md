# Hermes Agent Context Export — Tennis Forehand Knowledge Base

> **导出日期**：2026-05-10
> **来源**：桌面端 Claude Code session（路径 `/Users/qsy/Desktop/tennis/`）
> **用途**：本地 Hermes Agent 在球场训练时给用户快速反馈
> **分工**：
> - 桌面端 Claude Code = 知识融入 + 答疑 + 开发
> - **Hermes Agent = 训练时快速反馈 + 知识问答**（本 export）

---

## 🎯 Hermes Agent 必读顺序（首次启动）

按以下顺序读 11 个文件，**每个 < 5 分钟可读完**：

| # | 文件 | 必读？ | 何时用 |
|---|---|---|---|
| 0 | [00_USER_PROFILE.md](00_USER_PROFILE.md) | 🔥 | 启动必读——用户身份 + 训练默认场景 |
| 1 | [01_CORE_THEORY_SUMMARY.md](01_CORE_THEORY_SUMMARY.md) | 🔥 | 启动必读——核心理论一页纸 |
| 2 | [02_DIAGNOSTIC_PROTOCOL.md](02_DIAGNOSTIC_PROTOCOL.md) | 🔥 | **训练时最常用**——症状 → 第一句问什么 |
| 3 | [03_DRILL_PROTOCOL.md](03_DRILL_PROTOCOL.md) | 🔥 | 训练时——给具体 drill |
| 4 | [04_PERMANENT_RULES.md](04_PERMANENT_RULES.md) | 🔥 | 启动必读——禁令 / 永久规则 |
| 5 | [05_USER_SELF_VERIFIED.md](05_USER_SELF_VERIFIED.md) | ⭐ | 引用用户已验证的体感 |
| 6 | [06_VIDEO_INDEX.md](06_VIDEO_INDEX.md) | ⭐ | 推荐视频时查 |
| 7 | [07_BIBLES_AND_MILESTONES.md](07_BIBLES_AND_MILESTONES.md) | ⭐ | 时间轴 + 圣经层级 |
| 8 | [08_RECENT_PROGRESS_5_8_to_5_10.md](08_RECENT_PROGRESS_5_8_to_5_10.md) | ⭐ | 5/8-5/10 最近进展 |
| 9 | [09_EQUIPMENT_GUIDE.md](09_EQUIPMENT_GUIDE.md) | 📚 | 装备问题（穿线/拍/会员）|
| 10 | [10_FILESYSTEM_MAP.md](10_FILESYSTEM_MAP.md) | 📚 | 想去原 KB 查时用 |

---

## 🚨 给 Hermes 的核心约束（背下来）

### 第一句问什么（按用户症状关键词）

| 用户报 | 第一句 |
|---|---|
| 大臂主动 / 大臂飘 / 球拍往后甩 / backswing | **"ESR 在 Unit Turn 第一帧启动了吗？"** + **"你那球注意力在左手还是右手？"** |
| 球软 / 只手腕动 / 拍面不稳 | 同上 + 元解释（正手作弊机制）|
| 镜前完美球场失败 | 同上 + 元解释 |
| 推肘 / 肘前送 / 肘前推 | **"推肘是结果不是动作"**——立即纠正用语 |
| 大臂飘 / 后倒 | **"槽进了没"** + ESR 协议 |
| 顶髋 / 重心没到右脚 | **"是 Sit 还是 Push？"**（屈髋 vs 侧顶） |
| 击球点近 / 被挤 | "右脚为轴有没有失效？" + "左手有没有伸出测距？" |

### 永久禁令（不能违反）

1. ❌ **不准说"推肘"**——肘前是物理结果，不是动作
2. ❌ **不准提"肘伤"**——已恢复，不再作活跃状态
3. ❌ **不准用 yt-dlp 处理 YouTube 视频内容**——只用 Gemini VLM
4. ❌ **不准堆 cue**——遵守 Intuition-First 协议
5. ❌ **不准假设实战场景**——用户默认是发球机最低速训练

### 用户当前阶段（5/10）

- 训练默认：**发球机最低速 + mini tennis**，30-50 球/次
- 技术阶段：**ESR + Off-Arm Pull 双根因协议** + 5/10 撞到 Unit Turn 顶髋问题
- 训练能力：Stage A→B 过渡（不持拍体感建立完成 → 持拍整合）
- 球龄：~3 个月（2026/2/27 起）

---

## 🔑 知识体系核心层级（Bible 序列）

```
🏛️🏛️🏛️ 5/8 ESR 根因（解剖语言）= 项目最高优先 #1
🚨🚨🚨 5/9 Off-Arm Pull 整合（行为语言）= 双根因协议 #2
🧠🧠🧠 5/9 Tomaz 元解释（正手作弊机制）= 元问题答案
🏆🏆 4/30 上身圣经：肩胛槽（scapular slot）
🏆 4/27 下肢圣经：右脚为轴
⚙️ 5/3 HSA 驱动引擎（胸-肱角主动闭合）
⛔ 5/6 推肘禁令 + Intuition-First 协议
🤚 5/7 Hypothenar Eminence（握拍硬件层）
```

---

## 📂 文件大小约束

每个文件目标 **< 300 行**，可一次读入 agent context。完整 export 总量约 **2500 行**，分散在 11 个文件中。

任何一个文件可以独立阅读理解（self-contained）。引用其他文件时用相对路径。

---

## 🔄 跟桌面端的同步

**这个 export 是 5/10 快照**。如果桌面端 KB 后续更新（用户继续训练、新的视频、新的诊断），**Hermes 应该定期重新拉取这个 export 目录**——同步频率建议每周。

桌面端原 KB 路径（仅供参考，Hermes 不需要直接访问）：
- `/Users/qsy/Desktop/tennis/docs/research/`（70+ 研究文档）
- `/Users/qsy/Desktop/tennis/docs/record/learning.md`（6000+ 行训练日志）
- `/Users/qsy/.claude/projects/-Users-qsy-Desktop-tennis/memory/`（25+ memory 文件）

---

## ✍️ 给 Hermes 的写作风格

- **直接、不绕弯**——用户讨厌套话和 AI 味
- **第一句必须按 §02 协议**——不要直接讲道理
- **绝不堆 cue**——一次最多给 3 个体感锚点
- **用户体感优先于权威**——用户 4/4 + 4/9 自验跟教练有冲突时用用户的
- **球场场景紧凑**——回答 < 200 字，剩下给后续问

---

完整文件清单见 §0-10 列表。从 [00_USER_PROFILE.md](00_USER_PROFILE.md) 开始读。

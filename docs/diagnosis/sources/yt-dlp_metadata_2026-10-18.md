# yt-dlp 元数据核对记录（2026-10-18，用户 Mac，yt-dlp 2026.03.13，`--dump-json --skip-download`）

box 上同命令返回 HTTP 429 / 机器人验证，故在 Mac 上执行（逐条间隔约 4 秒）。下表是 yt-dlp 的输出字段（title / channel / duration / upload_date）；**以此为准，旧笔记里的标题、频道、教练名不采信**。

| ID | yt-dlp 标题 | 频道 | 时长 | 发布日期 | 官方人工字幕 | 自动英文字幕 | 旧笔记写的标题 |
|---|---|---|---|---|---|---|---|
| q6LrQYDvxmY | Why Pros Hit So Far Away | Fault Tolerant Tennis | 1036 s（17:16） | 2026-10-15 | 无 | 有 | 「Why You're Jammed, and How to Fix It」（**与 yt-dlp 不符**） |
| ui-RFRdQKRk | The Pros "Rotate" Differently Than You | Fault Tolerant Tennis | 692 s（11:32） | 2026-07-15 | 无 | 有 | 同 |
| hqganBxzQNM | Why I Give Every Student a Weight to Swing (and why it works) | Fault Tolerant Tennis | 403 s（6:43） | 2026-01-07 | 无 | 有 | 同 |
| wzLMv4ASg9s | Why Your Racket Doesn't Flip - and How to Fix It | Fault Tolerant Tennis | 2066 s（34:26） | 2026-08-12 | 无 | 有 | 同（时长 >20 分钟，未选） |
| 6ak4ZScH_gc | You Can Feel This - Here's How | Fault Tolerant Tennis | 614 s（10:14） | 2026-07-17 | 无 | 有 | 同（备选，主题是髋/肩髋分离，与 ui 重叠，未选） |
| wVa4XQPcaqs | Use The Wall to Find Your Perfect Contact | Fault Tolerant Tennis | 275 s（4:35） | 2024-07-19 | 无 | 有 | 「How to Find Your Perfect Contact Point」（不符；备选） |
| pWzyP-xfLfU | The Pros Grip Differently From You (with lesson examples) | Fault Tolerant Tennis | 1262 s（21:02） | 2026-03-27 | 无 | 有 | 同（握拍主题，不贴症状，>20 分钟，未选） |
| wFOy0RKWBTg | Swing OUT on the Forehand | Fault Tolerant Tennis | 31 s | 2024-06-25 | 无 | 有 | 「正手向外向量与肩部内收原理」/ 频道写成 Nassau Tennis（**频道与标题均不符**；31 秒短片，未选） |

其余 38 个 ID **没有用 yt-dlp 核对**（避免连续请求被限流）；旧笔记里它们的频道很多是模型猜的（TopCourt / Essential Tennis / Tennis TV…），不能直接当「官方 FTT」。

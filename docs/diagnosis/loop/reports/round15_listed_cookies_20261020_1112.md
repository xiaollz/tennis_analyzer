# round15 — listed 字幕（box Chrome cookies）

时间：Asia/Shanghai 2026-10-20 11:12  
工具：`/workspace/ftt_venv/bin/yt-dlp --cookies-from-browser chrome:/home/box/chrome-profile`  
亦试：Fork-3 profile；netscape cookies + `youtube_transcript_api`

## 结果：**成功 0 / 7**

Cookies **有效**（不再是 round14 的 IpBlocked/429 bot）。  
失败原因统一为：**视频本身无字幕 / TranscriptsDisabled**（均为 Shorts，6–48s，无 manual 也无 auto captions）。

| id | title | dur | 原因 |
|----|-------|-----|------|
| `dYPv0QfFrsU` | Forehand With/Without Scapular Activation | 12s | TranscriptsDisabled / no auto captions |
| `6j33LJfl46s` | Scapular Glide on the Tennis Forehand | 12s | TranscriptsDisabled / no auto captions |
| `97vo-kFis6c` | The Best Disguised Serve on the Tour | 14s | TranscriptsDisabled / no auto captions |
| `Cwc4YcKFZrU` | Retired WTA Pro Still Ripping the Forehand | 25s | TranscriptsDisabled / no auto captions |
| `risP4qTirj8` | WTA Pro Training with Hand Toss Forehands | 31s | TranscriptsDisabled / no auto captions |
| `XF17hgRlccw` | What Perfect Forehand Contact Looks Like | 6s | TranscriptsDisabled / no auto captions |
| `dnNOOornvek` | Retired WTA Pro Still Ripping Backhands | 48s | TranscriptsDisabled / no auto captions |

## 日志
`/workspace/ftt_subs/_listed_fetch_r15/`

## 下一步
listed 通道不再是 IP 拦，而是内容无字幕 → **不能升 subs_ready / Stage1**。  
转向 **B：非 holdout 正手 draft 人工/授权裁决**（0m3B / jDBg / QrRY 等）。

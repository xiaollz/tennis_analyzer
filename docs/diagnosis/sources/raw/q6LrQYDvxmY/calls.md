# 调用记录 q6LrQYDvxmY

- 模型：`gemini-3.8-flash`（固定）｜网关：Micu `https://www.micuapi.ai/v1beta/models/gemini-3.8-flash:generateContent`｜密钥：primary（环境变量），**未写入任何文件**；vertex 未使用。
- 提示词：`prompt_v1.txt`（**v1**，三期同一份、逐字相同；中立，不含项目术语/用户问题/故障卡名）。`temperature=0.2`，`maxOutputTokens=32000`。
- 视频输入：YouTube 链接直传 `fileData.fileUri`（camelCase）。**snake_case 的 `file_data/file_uri` 在该网关返回 400**（见 q6 的 call1）。
- 调用脚本：`../_tools/gem_call.py`。

| # | 时间（UTC+8） | 路径 | 结果 | 耗时(s) | tokens：prompt / 输出 / 思考 / 合计 | 备注 |
|---|---|---|---|---|---|---|
| 1 | 2026-10-18 10:46:14 +0800 | yt (YouTube URL, 非下载) | HTTP 400: {"error":{"message":"* ***.contents[0].parts[1].da | 4.8 | - / - / - / - | snake_case file_data 被拒（400 `parts[1].data: required oneof field`）；无输出 |
| 2 | 2026-10-18 10:47:22 +0800 | yt (YouTube URL, 非下载) | ok | 62.1 | 94548 / 2628 / 1952 / 99128 | 主用结果（`gemini_call2_output.md`） |
| 3 | 2026-10-18 11:01:09 +0800 | yt (YouTube URL, 非下载) | ok | 50.3 | 94548 / 2664 / 2820 / 100032 | 同提示词重复一次，测可重复性（`gemini_call3_repeat_output.md`） |

本期共 3 次 HTTP 调用（1 次格式错误失败，2 次成功），达到上限 3 次。

## yt-dlp / 字幕 / 转写记录（本期）
- yt-dlp 元数据：box 上 `yt-dlp --dump-json` → `HTTP 429`＋「Sign in to confirm you're not a bot」（含 `player_client=visionos` 重试一次）；改在用户 Mac（yt-dlp 2026.03.13）上取得（见 ../yt-dlp_metadata_2026-10-18.md）。
- 字幕：Mac 上 `--write-auto-subs --sub-langs en` **HTTP 429，共尝试 5 次**（间隔 25 秒～15 分钟）后停止，没有用 cookies、没有换 IP 硬刷。
- 转写：沿用 box 上 2026-10-17 已下载的 `/workspace/ftt_q6Lr/v.mp4`（1280×720，1035.6 s，与 yt-dlp 时长 1036 s 相符）；本次用 faster-whisper 重转两遍：medium.en+VAD（198 s，有 5 处空洞）与 small.en 60 秒分块（约 85 s，覆盖完整）。视频文件不入库。
- 没有抽帧：本期问题与处方主要在讲解音频里；画面（远景球场）本轮未逐帧核对。

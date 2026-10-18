# 调用记录 ui-RFRdQKRk

- 模型：`gemini-3.8-flash`（固定）｜网关：Micu `https://www.micuapi.ai/v1beta/models/gemini-3.8-flash:generateContent`｜密钥：primary（环境变量），**未写入任何文件**；vertex 未使用。
- 提示词：`prompt_v1.txt`（**v1**，三期同一份、逐字相同；中立，不含项目术语/用户问题/故障卡名）。`temperature=0.2`，`maxOutputTokens=32000`。
- 视频输入：YouTube 链接直传 `fileData.fileUri`（camelCase）。**snake_case 的 `file_data/file_uri` 在该网关返回 400**（见 q6 的 call1）。
- 调用脚本：`../_tools/gem_call.py`。

| # | 时间（UTC+8） | 路径 | 结果 | 耗时(s) | tokens：prompt / 输出 / 思考 / 合计 | 备注 |
|---|---|---|---|---|---|---|
| 1 | 2026-10-18 10:48:57 +0800 | yt (YouTube URL, 非下载) | ok | 36.8 | 63245 / 2460 / 1792 / 67497 | 唯一一次调用 |

本期共 1 次调用。

## yt-dlp / 字幕记录（本期）——**没有取到字幕、也没有取到音频**
- 元数据：Mac 上 yt-dlp 成功（见 ../yt-dlp_metadata_2026-10-18.md）。
- 字幕：Mac 上 `--write-auto-subs --sub-langs en` **HTTP 429，共 5 次**（间隔 25 秒～15 分钟）后停止。
- 音频/低清视频：Mac 上 `-f bv*[height<=360]+ba` → `HTTP 403`（android vr client，需要 PO token）；换 `player_client=visionos` 提示该版本 yt-dlp 不支持该 client。没有用 cookies，没有升级用户机器上的 yt-dlp，没有再硬试。
- 结果：**本期没有任何一手字幕/转写**。核对只能退到「旧笔记（自称基于官方自动字幕）」的二手比对，见 `secondary_check_vs_old_note.md`；来源卡里所有「字幕核对」均写「未核对」。

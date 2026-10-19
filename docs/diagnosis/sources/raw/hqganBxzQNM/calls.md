# 调用记录 hqganBxzQNM

- 模型：`gemini-3.8-flash`（固定）｜网关：Micu `https://www.micuapi.ai/v1beta/models/gemini-3.8-flash:generateContent`｜密钥：primary（环境变量），**未写入任何文件**；vertex 未使用。
- 提示词：`prompt_v1.txt`（**v1**，三期同一份、逐字相同；中立，不含项目术语/用户问题/故障卡名）。`temperature=0.2`，`maxOutputTokens=32000`。
- 视频输入：YouTube 链接直传 `fileData.fileUri`（camelCase）。**snake_case 的 `file_data/file_uri` 在该网关返回 400**（见 q6 的 call1）。
- 调用脚本：`../_tools/gem_call.py`。

| # | 时间（UTC+8） | 路径 | 结果 | 耗时(s) | tokens：prompt / 输出 / 思考 / 合计 | 备注 |
|---|---|---|---|---|---|---|
| 1 | 2026-10-18 10:48:55 +0800 | yt (YouTube URL, 非下载) | ok | 35.4 | 36876 / 1684 / 2208 / 40768 | 唯一一次调用 |

本期共 1 次调用。

## yt-dlp / 字幕记录（本期）
- 元数据与自动英文字幕均由用户 Mac 上的 yt-dlp（2026.03.13）取得：`--write-auto-subs --write-subs --sub-langs en,en-orig --sub-format vtt --skip-download`，一次成功（同批另外两期 429）。人工字幕不存在（`subtitles: []`）。
- 字幕全文留在 box `/workspace/ftt_pilot/hqganBxzQNM/`，不入库；入库的只有 `subtitle_excerpts.md` 里的片段。
- 没有下载视频、没有抽帧；示范学员的画面细节未核对。

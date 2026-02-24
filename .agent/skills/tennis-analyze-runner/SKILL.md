---
name: tennis-analyze-runner
description: Run the local tennis_analyzer CLI on Mac to analyze iCloud-synced .mp4 tennis videos and generate a single annotated output video (hybrid impact + Big3 UI). Use when the user (often via Telegram/OpenClaw) asks to analyze a specific video path/filename, analyze the latest uploaded video, or rerun with different options (impact mode, left-handed, device, metrics).
---

# Tennis Analyze Runner (for OpenClaw/Terminal)

目标：在本机 `/Users/qsy/Desktop/tennis` 项目里，从终端稳定地跑完一次视频分析，产出：
- **分析输出视频**（带骨架 + Big3 面板，可选数值 overlay）
- **离线报告**（`report.md` + 图表 + 关键击球缩略图）
并把**输出路径**回给用户。

关键点（必须遵守）：
- **不要下载任何模型**：只用本地模型 `models/yolo11m-pose.pt`
- **不要用系统 python**：必须用项目 venv 的 python：`/Users/qsy/Desktop/tennis/venv/bin/python`
- OpenClaw **无法“读取/上传”视频**：用户会先把视频放到 iCloud 同步目录（推荐 drop 目录见下）。

## 固定路径约定（不要改，除非用户明确要求）

- Repo 根目录：`/Users/qsy/Desktop/tennis`
- 输入视频 drop 目录（iCloud 同步）：`/Users/qsy/Desktop/tennis/data/videos/video`
- 输出目录：`/Users/qsy/Desktop/tennis/reports`
- 模型文件：`/Users/qsy/Desktop/tennis/models/yolo11m-pose.pt`

## 默认推荐参数（“稳、可复现、无具体数值面板”）

- `--impact-mode hybrid`：音频+姿态 two-pass（更适合离线报告与击球缩略图准确性）
- `--impact-merge-s 0.8`：去重“落地+击球”很近的重复触发
- `--impact-audio-tol 7`：允许音画有少量偏移（侧面/手机录屏更稳）
- `--big3-ui`：左上 Big3 面板（触球点→重心→随挥）
- `--no-metrics`：关闭右上角数值 metrics（避免输出一堆角度/距离）
- `--view auto`：自动判断侧面/背面视角，并对指标做 gating（默认即可）
- `-d auto`（或 Apple Silicon 可尝试 `-d mps`，失败再退回 `cpu`）
- `--report`：生成 `report.md` + 图表 + 关键击球缩略图（默认开启）

## 推荐执行方式（优先用脚本）

使用 wrapper 脚本能自动处理：
- 找最新视频
- 生成带时间戳的输出文件名（避免覆盖）
- 强制使用 venv + 本地模型
- 默认同时产出：输出视频 + 报告

脚本路径：
- `/Users/qsy/Desktop/tennis/.agent/skills/tennis-analyze-runner/scripts/run_tennis_analysis.sh`

### 1) 分析最新视频（drop 目录里最新的 mp4）

```bash
bash /Users/qsy/Desktop/tennis/.agent/skills/tennis-analyze-runner/scripts/run_tennis_analysis.sh --latest
```

### 2) 分析指定视频（给相对/绝对路径都可以）

```bash
# 相对路径（相对 repo 根）
bash /Users/qsy/Desktop/tennis/.agent/skills/tennis-analyze-runner/scripts/run_tennis_analysis.sh data/videos/video/xxxx.mp4

# 只给文件名（会自动去 drop 目录里找）
bash /Users/qsy/Desktop/tennis/.agent/skills/tennis-analyze-runner/scripts/run_tennis_analysis.sh xxxx.mp4

# 绝对路径
bash /Users/qsy/Desktop/tennis/.agent/skills/tennis-analyze-runner/scripts/run_tennis_analysis.sh /path/to/video.mp4
```

### 2.1) 用日期文件夹隔离本次上传（推荐）

```bash
# 例：把本次结果都放到 reports/2026-02-09/<video_id>/ 下，避免和历史混在一起
VID="data/videos/video/xxxx.mp4"
STEM="xxxx"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="reports/2026-02-09/${STEM}/${STEM}_combined_hybrid_${TS}.mp4"
bash /Users/qsy/Desktop/tennis/.agent/skills/tennis-analyze-runner/scripts/run_tennis_analysis.sh "$VID" --output "$OUT"
```

### 3) 常用可选参数

```bash
# 左手
... run_tennis_analysis.sh --latest --left-handed

# 纯 pose（当 hybrid 因为音轨/ffmpeg 报错时用）
... run_tennis_analysis.sh --latest --impact-mode pose

# 调大/调小去重窗口（落地和击球很近时）
... run_tennis_analysis.sh --latest --impact-merge-s 0.6
... run_tennis_analysis.sh --latest --impact-merge-s 1.0

# 调大/调小“音画容差”（侧面拍、蓝牙麦克风、剪辑合成后可能需要更大）
... run_tennis_analysis.sh --latest --impact-audio-tol 5
... run_tennis_analysis.sh --latest --impact-audio-tol 10

# Apple Silicon 加速
... run_tennis_analysis.sh --latest --device mps

# 视角强制（自动判断错时）
... run_tennis_analysis.sh --latest --view side
... run_tennis_analysis.sh --latest --view back
```

## 直接命令方式（脚本不可用时）

```bash
cd /Users/qsy/Desktop/tennis
/Users/qsy/Desktop/tennis/venv/bin/python -m tennis_analyzer.main \
  data/videos/video/xxxx.mp4 \
  -o reports/xxxx_combined_hybrid.mp4 \
  -m models/yolo11m-pose.pt \
  --impact-mode hybrid --impact-merge-s 0.8 --impact-audio-tol 7 --big3-ui --no-metrics --view auto
```

如需同时生成报告：

```bash
cd /Users/qsy/Desktop/tennis
/Users/qsy/Desktop/tennis/venv/bin/python -m tennis_analyzer.main \
  data/videos/video/xxxx.mp4 \
  -o reports/xxxx_combined_hybrid.mp4 \
  -m models/yolo11m-pose.pt \
  --impact-mode hybrid --impact-merge-s 0.8 --impact-audio-tol 7 --big3-ui --no-metrics \
  --report --report-sample-fps 4 --view auto
```

## 给用户的回复格式（Telegram 上建议这样回）

最少包含：
- 输出视频路径（绝对路径）
- 报告路径（`..._report/report.md`）
- 文件大小（可选）
- 若日志里有：impact 帧列表（可选；不要刷屏）

不要粘贴大量终端日志；只回“结果在哪、怎么打开”。

## 击球点正确性保障（必做检查 + 快速调参）

Big3 的“质量门槛”是击球帧（impact）本身要对：缩略图不对 → 指标不可信。

### 必做检查（跑完后 30 秒内完成）

1) 打开输出视频：确认每次击球时 Big3 面板会 flash，并在击球后一小段时间冻结显示
2) 打开报告：`<output>_report/report.md`
   - 检查“关键击球缩略图/impact thumbnails”是否真的是击球瞬间（不是落地、不是准备、不是挥空）

### 常见问题与推荐调参

1) **把落地当击球 / 击球被落地挤掉**
   - 增大去重窗口：`--impact-merge-s 1.0` 或 `1.2`

2) **漏了很多击球（只抓到少数几次）**
   - 放宽音画容差：`--impact-audio-tol 10` 或 `12`
   - 或临时用纯姿态：`--impact-mode pose`（先保证能出结果，再回头调 hybrid）

3) **击球帧整体偏早/偏晚（缩略图总是差 1-3 帧）**
   - 先确认视频是否经过剪辑/变速/合成（这类更容易音画偏移）
   - 尝试：
     - 更严格：`--impact-audio-tol 0`（视频音画很准时）
     - 更宽松：`--impact-audio-tol 10`（手机+蓝牙麦克风/合成视频更常见）

4) **iCloud 文件没下完导致 hybrid 音频提取失败或读取异常**
   - 先在 Finder 点“下载”，或等文件完整下载
   - 终端快速检查：`ls -lh data/videos/video | head`

## Telegram 指令解析（建议规则）

优先识别这些意图并映射到脚本参数：

- 用户说“分析最新/last/latest”：
  - 运行：`... run_tennis_analysis.sh --latest`
- 用户给了 `*.mp4` 文件名（不带路径）：
  - 运行：`... run_tennis_analysis.sh <filename.mp4>`
- 用户给了路径（`/` 开头或 `data/...`）：
  - 运行：`... run_tennis_analysis.sh <path>`
- 用户明确说“没声音/用pose/实时”：
  - 加：`--impact-mode pose`
- 用户明确说“左手”：
  - 加：`--left-handed`
- 用户明确说“要数值”：
  - 加：`--metrics`（默认是 `--no-metrics`）

如果脚本报“找不到输入视频”：
- 先在 drop 目录里列一下候选让用户选（不要猜）：
  - `ls -lt /Users/qsy/Desktop/tennis/data/videos/video | head`

## 故障排查（按优先级）

1) `No module named cv2` / `ultralytics`：
   - 说明用错 python；必须用 `.../venv/bin/python`
2) `hybrid` 报音频相关错误：
   - 先用 `--impact-mode pose` 跑完（保证能出结果）
   - 或让用户确认视频有音轨且已完整下载到本机（iCloud 文件可能还在云端）
3) 输入视频读不到/0 bytes：
   - 让用户等待 iCloud 下载完成，或在 Finder 里点“下载”

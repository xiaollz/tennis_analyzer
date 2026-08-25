# Tennis Analyzer v3

基于 YOLO Pose 的网球挥拍分析工具，支持正手/单反评估、击球检测、标注视频与 Markdown 报告生成。

## 功能

- YOLO Pose（COCO 17 点）人体关键点检测
- 自动检测击球时刻（视觉+音频）
- 正手 / 单反技术评分与 KPI 反馈
- 标注视频输出（骨架、轨迹、HUD）
- 图表与完整分析报告输出
- Gradio Web UI

## 安装

```bash
pip install -r requirements.txt
```

## 命令行

```bash
python main.py analyse --video path/to/video.mp4 --model auto
```

常用参数：

- `--stroke auto|forehand|backhand`
- `--left-handed`（默认右手）
- `--model`（`auto` 或自定义 YOLO `.pt` 权重）
- `--output-dir`
- `--joints`

## Web UI

```bash
python main.py ui --port 7860 --model auto
```

## 项目结构

```text
tennis/
├── main.py
├── core/
│   ├── pose_estimator.py
│   └── video_processor.py
├── analysis/
├── evaluation/
├── report/
├── config/
└── docs/
```

## 知识库入口

- `docs/START_HERE.md`：网球知识库阅读入口。
- `docs/record/learning.md`：个人训练时间轴、训练发现与复盘主记录。
- `docs/record/session_*.md`：按 session 归档的完整对话记录。

## License

MIT

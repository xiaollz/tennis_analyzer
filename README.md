# Tennis Pose Analyzer

基于 YOLO Pose 的网球动作分析工具，分析练习视频并输出带骨架标注和生物力学指标的视频。

## 功能

- 姿态估计：使用 YOLO11-pose 检测人体17个关键点
- 骨架可视化：在视频上叠加彩色骨架
- 生物力学指标：实时显示关节角度、髋肩分离角等
- 跨平台：支持 Mac (MPS) / Linux (CUDA) / CPU

## 🎾 AI 教练分析流程 (AI Coach Analysis Workflow)

这是获取深度技术分析报告的标准流程：

### 1. 准备视频
将你的网球练习视频（推荐正手/单反训练）放入 `data/videos/` 目录。
*例如：`data/videos/forehand_practice.mp4`*

### 2. 提取关键帧与音频
运行提取脚本，将视频分解为图像帧和音频数据。
```bash
# 语法：python3 scripts/extract_key_frames.py <视频路径> --output_dir <输出目录>
python3 scripts/extract_key_frames.py data/videos/forehand_practice.mp4 --output_dir data/processed/forehand_analysis_01
```

### 3. 生成教练报告
运行报告生成脚本，读取上一步的数据并生成 Markdown 报告。
```bash
# 语法：python3 scripts/generate_coach_report.py <数据目录>
python3 scripts/generate_coach_report.py data/processed/forehand_analysis_01
```

### 4. 查看报告
分析报告将会自动生成在 `reports/tennis_analysis_report.md`。
可以直接用 Markdown 阅读器打开查看图文并茂的分析结果。

---

## 安装

```bash
cd /Users/qsy/Desktop/tennis

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 或者安装为包
pip install -e .
```

## 使用

### 命令行

```bash
# 基本用法
python -m tennis_analyzer.main input.mp4 -o output.mp4

# 🎯 推荐：Hybrid 击球点（音频+关键点） + Big3 面板 + 动力链反馈（合并到一个视频）
# 注意：在网络受限环境下请用本地模型路径，例如 models/yolo11m-pose.pt
python -m tennis_analyzer.main input.mp4 -o output.mp4 -m models/yolo11m-pose.pt --impact-mode hybrid --big3-ui

# 如果你的视频里“落地声+击球声”很近，建议开启/加大去重窗口（默认 0.8s）
python -m tennis_analyzer.main input.mp4 -o output.mp4 -m models/yolo11m-pose.pt --impact-mode hybrid --impact-merge-s 1.2 --big3-ui

# 指定模型（更快但精度稍低）
python -m tennis_analyzer.main input.mp4 -o output.mp4 -m yolo11s-pose.pt

# 指定设备
python -m tennis_analyzer.main input.mp4 -o output.mp4 -d mps  # Mac
python -m tennis_analyzer.main input.mp4 -o output.mp4 -d cuda # NVIDIA GPU

# 不显示指标
python -m tennis_analyzer.main input.mp4 -o output.mp4 --no-metrics

# 调整置信度阈值
python -m tennis_analyzer.main input.mp4 -o output.mp4 -c 0.3
```

### 作为库使用

```python
from tennis_analyzer.core import PoseEstimator, VideoProcessor
from tennis_analyzer.visualization import SkeletonDrawer
from tennis_analyzer.analysis import BiomechanicsAnalyzer

# 初始化
estimator = PoseEstimator(model_name="yolo11m-pose.pt", device="mps")
drawer = SkeletonDrawer()
analyzer = BiomechanicsAnalyzer()

# 处理视频
for frame_idx, frame, results in estimator.predict_video("input.mp4"):
    for person in results["persons"]:
        # 绘制骨架
        frame = drawer.draw(frame, person["keypoints"], person["confidence"])

        # 计算指标
        metrics = analyzer.analyze(person["keypoints"], person["confidence"])
        print(f"Frame {frame_idx}: {metrics}")
```

## 显示的指标

| 指标 | 说明 |
|------|------|
| L/R Knee | 左/右膝盖弯曲角度 |
| L/R Elbow | 左/右肘部弯曲角度 |
| X-Factor | 髋肩分离角（发力关键） |
| Shoulder | 肩部旋转角度 |

## 模型选择

| 模型 | 精度 | 速度 | 推荐场景 |
|------|------|------|----------|
| yolo11n-pose | 较低 | 最快 | 实时预览 |
| yolo11s-pose | 中等 | 快 | 日常使用 |
| yolo11m-pose | 较高 | 中等 | **推荐** |
| yolo11l-pose | 高 | 较慢 | 精细分析 |
| yolo11x-pose | 最高 | 最慢 | 最高精度 |

## 目录结构

```
tennis/
├── scripts/                 # Python scripts for analysis and extraction
│   ├── extract_key_frames.py
│   ├── generate_coach_report.py
│   └── ...
├── data/                    # Data storage
│   ├── videos/              # Raw video files
│   ├── processed/           # Extracted frames and outputs
│   └── metadata/            # JSON metadata files
├── models/                  # ML models (YOLO weights)
├── docs/                    # Documentation
├── reports/                 # Generated analysis reports
├── tennis_analyzer/         # Core package source code
├── requirements.txt
└── README.md
```

## 后续计划

- [ ] 动作分类（正手/反手/发球/截击）
- [ ] 动作阶段检测（引拍/击球/随挥）
- [ ] 更多网球专用指标
- [ ] 与 Feel Tennis 教学要点对比
- [ ] Web 界面

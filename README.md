# Tennis Analyzer v2 — Modern Forehand Evaluation

AI-powered tennis forehand analysis based on the **Modern Forehand** framework derived from:

- **Dr. Brian Gordon** — Type 3 forehand biomechanics, straight-arm extension
- **Rick Macci** — compact unit turn, elbow mechanics, "the flip"
- **Tennis Doctor** — four non-negotiables, kinetic chain sequencing
- **Feel Tennis** — modern forehand 8-step model

## Features

- **Pose Estimation**: YOLO Pose (COCO 17-keypoint) for real-time body tracking
- **Joint Trajectory Tracking**: Track and visualise any joint's path with configurable trails
- **14 KPI Metrics** across 6 swing phases:
  - Phase 1: Preparation & Unit Turn (shoulder rotation, knee bend, spine posture)
  - Phase 3: Kinetic Chain (sequence, hip-shoulder separation, hand path linearity)
  - Phase 4: Contact Point (position, elbow angle, body freeze, head stability)
  - Phase 5: Extension & Follow-Through (forward extension, follow-through path)
  - Phase 6: Balance & Recovery (head stability, spine consistency)
- **Automatic Impact Detection**: wrist-speed peak analysis
- **Annotated Video Output**: skeleton overlay + trajectory trails + HUD
- **Comprehensive Report**: Markdown report with radar charts, KPI bar charts, coaching tips
- **Gradio Web UI**: interactive analysis interface

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Command-Line Analysis

```bash
python main.py analyse --video path/to/forehand.mp4 --output-dir ./output
```

Options:
- `--right-handed` / `--left-handed`: specify dominant hand (default: right)
- `--joints right_wrist right_elbow right_hip`: specify joints to track
- `--model yolo11m-pose.pt`: specify YOLO model

### Gradio Web UI

```bash
python main.py ui --port 7860
```

## Architecture

```
tennis_analyzer_v2/
├── config/
│   ├── keypoints.py          # COCO 17-keypoint definitions
│   └── framework_config.py   # All evaluation thresholds & weights
├── core/
│   ├── pose_estimator.py     # YOLO Pose wrapper
│   └── video_processor.py    # Video I/O with auto-rotation
├── analysis/
│   ├── trajectory.py         # Joint trajectory management & smoothing
│   └── kinematic_calculator.py  # Angle, rotation, body-plane geometry
├── evaluation/
│   ├── event_detector.py     # Impact detection & swing phase estimation
│   ├── kpi.py                # 14 KPI definitions with scoring logic
│   └── forehand_evaluator.py # Orchestration: data → metrics → scores
├── report/
│   ├── visualizer.py         # Skeleton, trajectory, chart drawing
│   └── report_generator.py   # Markdown report generation
├── main.py                   # CLI + Gradio UI entry point
└── docs/
    ├── architecture_v2.md
    └── learn_ytb/            # Reference transcripts
```

## Evaluation Framework

The evaluation is structured around 6 phases of a modern forehand:

| Phase | Weight | Key Metrics |
|-------|--------|-------------|
| Preparation & Unit Turn | 15% | Shoulder rotation (X-Factor), knee bend, spine posture |
| Loading & Lag | 10% | Wrist layback, elbow-hand drop |
| Kinetic Chain | 20% | Sequential peak ordering, hip-shoulder separation, hand path linearity |
| Contact Point | 25% | Contact position, elbow angle (straight-arm vs double-bend), body freeze, head stability |
| Extension & Follow-Through | 15% | Forward extension distance, follow-through path ratio |
| Balance & Recovery | 15% | Overall head stability, spine consistency |

Each KPI produces a 0-100 score with human-readable coaching feedback.

## Model Selection

| Model | Accuracy | Speed | Recommended Use |
|-------|----------|-------|-----------------|
| yolo11n-pose | Lower | Fastest | Real-time preview |
| yolo11s-pose | Medium | Fast | Quick analysis |
| yolo11m-pose | Higher | Medium | **Recommended** |
| yolo11l-pose | High | Slower | Detailed analysis |
| yolo11x-pose | Highest | Slowest | Maximum accuracy |

## License

MIT

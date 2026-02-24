"""Tennis Analyzer v2 — 现代正手评估系统。

Usage:
    # 命令行分析
    python main.py analyse --video path/to/video.mp4 [--right-handed] [--output-dir ./output]

    # 启动 Gradio Web UI
    python main.py ui [--port 7860]
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.framework_config import DEFAULT_CONFIG, FrameworkConfig
from config.keypoints import KEYPOINT_NAMES
from core.video_processor import VideoProcessor, VideoWriter
from core.pose_estimator import PoseEstimator
from analysis.trajectory import TrajectoryStore
from evaluation.forehand_evaluator import ForehandEvaluator, MultiSwingReport
from evaluation.event_detector import HybridImpactDetector, ImpactEvent
from report.visualizer import SkeletonDrawer, TrajectoryDrawer, ChartGenerator, JOINT_CN
from report.report_generator import ReportGenerator


# =====================================================================
# Pipeline
# =====================================================================

class ForehandPipeline:
    """端到端流水线：视频 → 姿态估计 → 击球检测 → 评估 → 报告。"""

    def __init__(
        self,
        model_name: str = "yolo11m-pose.pt",
        is_right_handed: bool = True,
        cfg: FrameworkConfig = DEFAULT_CONFIG,
        output_dir: str = "./output",
        tracked_joints: Optional[List[str]] = None,
        max_trail: int = 30,
    ):
        self.is_right_handed = is_right_handed
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_trail = max_trail

        # 核心模块
        self.estimator = PoseEstimator(model_name=model_name)
        self.skeleton_drawer = SkeletonDrawer()

        # 默认只追踪 1 个关节：持拍手腕
        default_joints = ["right_wrist"] if is_right_handed else ["left_wrist"]
        self.tracked_joints = tracked_joints or default_joints
        self._init_trajectory_drawers()

    def _init_trajectory_drawers(self):
        """初始化轨迹绘制器。"""
        self.trajectory_drawers: Dict[str, TrajectoryDrawer] = {}
        joint_colors = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0),
            (255, 165, 0), (255, 0, 0), (0, 165, 255),
        ]
        for i, jname in enumerate(self.tracked_joints[:2]):  # 最多2个
            color = joint_colors[i % len(joint_colors)]
            self.trajectory_drawers[jname] = TrajectoryDrawer(
                joint=jname, color=color, max_trail=self.max_trail, fade=True,
            )

    def run(
        self,
        video_path: str,
        progress_callback=None,
    ) -> Dict:
        """运行完整分析流水线。

        Returns
        -------
        dict:
            report : MultiSwingReport
            report_path : str (Markdown 文件)
            annotated_video_path : str
            chart_paths : dict
        """
        video_name = Path(video_path).stem
        vp = VideoProcessor(video_path)
        fps = vp.fps

        # ── 阶段 0: 检测旋转 ────────────────────────────────────────
        extra_rot = vp.detect_rotation_from_content(self.estimator)
        if extra_rot != 0:
            vp.apply_additional_rotation(extra_rot)

        # ── 阶段 1: 姿态估计 + 轨迹收集 + 击球检测 ──────────────────
        if progress_callback:
            progress_callback(0, vp.total_frames, "正在进行姿态估计...")

        keypoints_series: List[np.ndarray] = []
        confidence_series: List[np.ndarray] = []
        frame_indices: List[int] = []
        frames_raw: List[np.ndarray] = []

        store = TrajectoryStore(fps=fps)

        # 初始化音频+视觉协同击球检测器
        impact_detector = HybridImpactDetector(
            video_path=video_path,
            fps=fps,
            is_right_handed=self.is_right_handed,
            cfg=self.cfg.impact_detection,
        )

        wrist_speeds_per_frame: List[float] = []

        for frame_idx, frame in vp.read_frames():
            result = self.estimator.predict(frame)

            if result["num_persons"] > 0:
                person = self._select_person(result["persons"])
                kp = person["keypoints"]
                conf = person["confidence"]
            else:
                kp = np.zeros((17, 2), dtype=np.float32)
                conf = np.zeros(17, dtype=np.float32)

            keypoints_series.append(kp)
            confidence_series.append(conf)
            frame_indices.append(frame_idx)
            frames_raw.append(frame)

            store.update(kp, conf, frame_idx)

            # 击球检测（逐帧更新）
            _, wrist_speed = impact_detector.update(frame_idx, kp, conf)
            wrist_speeds_per_frame.append(wrist_speed)

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, vp.total_frames, "姿态估计中...")

        # 完成击球检测
        impact_events = impact_detector.finalize()

        # ── 阶段 2: 评估（多次击球独立评分）──────────────────────────
        if progress_callback:
            progress_callback(vp.total_frames, vp.total_frames, "正在评估正手技术...")

        evaluator = ForehandEvaluator(
            fps=fps,
            is_right_handed=self.is_right_handed,
            cfg=self.cfg,
        )
        report = evaluator.evaluate_multi(
            keypoints_series, confidence_series, frame_indices, impact_events,
        )

        # ── 阶段 3: 生成标注视频 ────────────────────────────────────
        if progress_callback:
            progress_callback(0, len(frames_raw), "正在生成标注视频...")

        annotated_path = str(self.output_dir / f"{video_name}_annotated.mp4")
        with VideoWriter(annotated_path, vp.width, vp.height, fps, input_path=video_path) as writer:
            # 重置轨迹绘制器
            for drawer in self.trajectory_drawers.values():
                drawer.clear()

            # 构建击球帧集合，用于标注
            impact_frame_set = set(report.impact_frames)

            for i, (frame, kp, conf) in enumerate(zip(frames_raw, keypoints_series, confidence_series)):
                # 绘制骨骼
                annotated = self.skeleton_drawer.draw(frame, kp, conf)

                # 更新并绘制轨迹（带消失时间）
                for drawer in self.trajectory_drawers.values():
                    drawer.update(kp, conf, frame_idx=frame_indices[i])
                    annotated = drawer.draw(annotated)

                # 标记击球帧
                current_frame = frame_indices[i]
                if current_frame in impact_frame_set:
                    # 找到对应的击球序号
                    swing_idx = report.impact_frames.index(current_frame)
                    self._draw_impact_marker(annotated, swing_idx + 1)

                # HUD 叠加
                annotated = self._draw_hud(annotated, current_frame, report)

                writer.write(annotated)

                if progress_callback and i % 10 == 0:
                    progress_callback(i, len(frames_raw), "写入标注视频...")

        # ── 阶段 4: 生成图表 ────────────────────────────────────────
        if progress_callback:
            progress_callback(0, 1, "正在生成分析图表...")

        chart_paths = self._generate_charts(report, store, video_name, frame_indices)

        # ── 阶段 5: 生成报告 ────────────────────────────────────────
        report_gen = ReportGenerator(output_dir=str(self.output_dir))
        report_path = report_gen.generate(report, video_name=video_name, chart_paths=chart_paths)

        if progress_callback:
            progress_callback(1, 1, "分析完成！")

        return {
            "report": report,
            "report_path": report_path,
            "annotated_video_path": annotated_path,
            "chart_paths": chart_paths,
        }

    # ── 辅助方法 ─────────────────────────────────────────────────────

    @staticmethod
    def _select_person(persons: list) -> dict:
        """选择最显著的人（最大边界框）。"""
        if len(persons) == 1:
            return persons[0]
        best = persons[0]
        best_area = 0
        for p in persons:
            if p["bbox"] is not None:
                bbox = p["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > best_area:
                    best_area = area
                    best = p
        return best

    @staticmethod
    def _draw_impact_marker(frame: np.ndarray, swing_num: int):
        """在帧上绘制击球标记。"""
        h, w = frame.shape[:2]
        text = f"IMPACT #{swing_num}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 600.0
        thickness = max(2, int(font_scale * 2))
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = int(h * 0.08) + text_size[1]
        # 背景
        cv2.rectangle(frame, (x - 10, y - text_size[1] - 10),
                       (x + text_size[0] + 10, y + 10), (0, 0, 200), -1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, frame_idx: int, report: MultiSwingReport) -> np.ndarray:
        """在帧上绘制 HUD 信息。"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 1200.0
        font_scale = max(0.4, font_scale)

        # 帧号
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 击球次数
        cv2.putText(frame, f"Swings: {report.total_swings}", (10, 50),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 平均评分
        score = report.average_score
        color = (0, 255, 0) if score >= 70 else (0, 255, 255) if score >= 50 else (0, 0, 255)
        score_text = f"Avg: {score:.0f}"
        text_size = cv2.getTextSize(score_text, font, font_scale * 1.2, 2)[0]
        cv2.putText(frame, score_text, (w - text_size[0] - 10, 30),
                    font, font_scale * 1.2, color, 2, cv2.LINE_AA)

        # 追踪关节标签
        y_offset = h - 15
        for jname, drawer in self.trajectory_drawers.items():
            cn_name = JOINT_CN.get(jname, jname)
            cv2.putText(frame, cn_name, (10, y_offset),
                        font, font_scale * 0.9, drawer.color, 1, cv2.LINE_AA)
            y_offset -= 20

        return frame

    def _generate_charts(
        self,
        report: MultiSwingReport,
        store: TrajectoryStore,
        video_name: str,
        frame_indices: List[int],
    ) -> Dict[str, str]:
        """生成所有分析图表。"""
        charts = {}
        chart_dir = self.output_dir / "charts"
        chart_dir.mkdir(exist_ok=True)

        # 多次击球对比图
        if report.total_swings > 1:
            swing_scores = [(ev.swing_index, ev.overall_score) for ev in report.swing_evaluations]
            summary_path = str(chart_dir / f"{video_name}_multi_swing.png")
            if ChartGenerator.multi_swing_summary_chart(swing_scores, summary_path):
                charts["multi_swing_summary"] = summary_path

        # 每次击球的雷达图和 KPI 条形图
        for ev in report.swing_evaluations:
            idx = ev.swing_index
            suffix = f"_{idx}" if report.total_swings > 1 else ""

            # 雷达图
            phase_scores = {p: ps.score for p, ps in ev.phase_scores.items()}
            radar_path = str(chart_dir / f"{video_name}_radar{suffix}.png")
            result = ChartGenerator.radar_chart(
                phase_scores, radar_path,
                title="各阶段评分雷达图",
                swing_idx=idx if report.total_swings > 1 else None,
            )
            if result:
                key = f"radar_{idx}" if report.total_swings > 1 else "radar"
                charts[key] = radar_path

            # KPI 条形图
            kpi_bar_path = str(chart_dir / f"{video_name}_kpi_bar{suffix}.png")
            result = ChartGenerator.kpi_bar_chart(
                ev.kpi_results, kpi_bar_path,
                title="KPI 评分详情",
                swing_idx=idx if report.total_swings > 1 else None,
            )
            if result:
                key = f"kpi_bar_{idx}" if report.total_swings > 1 else "kpi_bar"
                charts[key] = kpi_bar_path

        # 关节轨迹图和速度曲线
        impact_frames_list = report.impact_frames
        for jname in self.tracked_joints[:2]:
            traj = store.get(jname)
            positions = traj.get_positions(smoothed=True)
            cn_name = JOINT_CN.get(jname, jname)

            if len(positions) > 2:
                traj_path = str(chart_dir / f"{video_name}_{jname}_trajectory.png")
                ChartGenerator.joint_trajectory_chart(
                    positions, traj.frame_indices, jname,
                    traj_path, impact_frames=impact_frames_list,
                )
                charts[f"trajectory_{jname}"] = traj_path

            speeds = traj.get_speeds(smoothed=True)
            if len(speeds) > 2:
                speed_path = str(chart_dir / f"{video_name}_{jname}_speed.png")
                ChartGenerator.speed_profile_chart(
                    speeds, traj.frame_indices[1:], jname,
                    speed_path, impact_frames=impact_frames_list,
                )
                charts[f"speed_{jname}"] = speed_path

        return charts


# =====================================================================
# Gradio UI
# =====================================================================

def build_gradio_ui(pipeline: ForehandPipeline):
    """构建 Gradio Blocks 界面。"""
    import gradio as gr

    with gr.Blocks(
        title="网球分析器 v2 — 现代正手评估",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🎾 网球分析器 v2 — 现代正手技术评估")
        gr.Markdown(
            "上传正手挥拍视频，系统将基于 **Modern Forehand** 理论框架 "
            "(Dr. Brian Gordon, Rick Macci, Tennis Doctor) 评估您的技术。\n\n"
            "支持多次击球独立评分，使用音频+视觉协同检测击球点。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="上传正手视频")
                with gr.Row():
                    is_right = gr.Checkbox(value=True, label="右手持拍")
                    max_trail_slider = gr.Slider(
                        minimum=10, maximum=60, value=30, step=5,
                        label="轨迹保留帧数",
                    )
                tracked_joints_input = gr.CheckboxGroup(
                    choices=[
                        "right_wrist", "left_wrist",
                        "right_elbow", "left_elbow",
                    ],
                    value=["right_wrist"],
                    label="追踪关节（最多2个）",
                )
                analyse_btn = gr.Button("开始分析", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_text = gr.Textbox(label="状态", interactive=False)
                with gr.Row():
                    overall_score = gr.Number(label="平均综合评分", interactive=False)
                    swing_count = gr.Number(label="检测到击球次数", interactive=False)

        with gr.Tabs():
            with gr.Tab("标注视频"):
                video_output = gr.Video(label="标注视频")

            with gr.Tab("评分概览"):
                radar_chart = gr.Image(label="阶段评分雷达图")
                multi_swing_chart = gr.Image(label="多次击球对比")

            with gr.Tab("KPI 详情"):
                kpi_bar_chart = gr.Image(label="KPI 评分条形图")
                kpi_table = gr.Dataframe(
                    headers=["KPI", "阶段", "评分", "评级", "测量值", "反馈"],
                    label="KPI 结果",
                )

            with gr.Tab("关节轨迹"):
                trajectory_gallery = gr.Gallery(label="轨迹图表", columns=2)

            with gr.Tab("速度曲线"):
                speed_gallery = gr.Gallery(label="速度曲线图表", columns=2)

            with gr.Tab("完整报告"):
                report_md = gr.Markdown(label="完整报告")
                report_file = gr.File(label="下载报告")

        def run_analysis(video, right_handed, tracked_joints, max_trail_val):
            if video is None:
                return "请上传视频。", 0, 0, None, None, None, None, [], [], [], "", None

            # 重新配置 pipeline
            pipeline.is_right_handed = right_handed
            pipeline.max_trail = int(max_trail_val)
            pipeline.tracked_joints = (tracked_joints[:2] if tracked_joints else
                                       (["right_wrist"] if right_handed else ["left_wrist"]))
            pipeline._init_trajectory_drawers()

            try:
                result = pipeline.run(video)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"错误: {e}", 0, 0, None, None, None, None, [], [], [], "", None

            report = result["report"]
            charts = result["chart_paths"]

            # KPI 表格（汇总所有击球）
            kpi_rows = []
            for ev in report.swing_evaluations:
                prefix = f"[第{ev.swing_index + 1}次] " if report.total_swings > 1 else ""
                for k in ev.kpi_results:
                    val_str = f"{k.raw_value:.2f}" if k.raw_value is not None else "无数据"
                    phase_cn = ReportGenerator.PHASE_TITLES.get(k.phase, k.phase)
                    kpi_rows.append([
                        f"{prefix}{k.kpi_id} {k.name}",
                        phase_cn,
                        f"{k.score:.0f}",
                        k.rating,
                        val_str,
                        k.feedback,
                    ])

            # 轨迹和速度图
            traj_images = [v for k, v in charts.items() if k.startswith("trajectory_")]
            speed_images = [v for k, v in charts.items() if k.startswith("speed_")]

            # 报告文本
            report_text = ""
            if Path(result["report_path"]).exists():
                report_text = Path(result["report_path"]).read_text(encoding="utf-8")

            # 雷达图（显示第一次击球的，或唯一的）
            radar_img = charts.get("radar") or charts.get("radar_0")
            multi_img = charts.get("multi_swing_summary")

            # KPI 条形图
            kpi_bar_img = charts.get("kpi_bar") or charts.get("kpi_bar_0")

            return (
                "分析完成！",
                report.average_score,
                report.total_swings,
                result["annotated_video_path"],
                radar_img,
                multi_img,
                kpi_bar_img,
                kpi_rows,
                traj_images,
                speed_images,
                report_text,
                result["report_path"],
            )

        analyse_btn.click(
            fn=run_analysis,
            inputs=[video_input, is_right, tracked_joints_input, max_trail_slider],
            outputs=[
                status_text, overall_score, swing_count,
                video_output,
                radar_chart, multi_swing_chart,
                kpi_bar_chart, kpi_table,
                trajectory_gallery, speed_gallery,
                report_md, report_file,
            ],
        )

    return demo


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="网球分析器 v2 — 现代正手评估")
    subparsers = parser.add_subparsers(dest="command")

    # analyse 子命令
    analyse_parser = subparsers.add_parser("analyse", help="分析正手视频")
    analyse_parser.add_argument("--video", required=True, help="视频文件路径")
    analyse_parser.add_argument("--right-handed", action="store_true", default=True)
    analyse_parser.add_argument("--left-handed", action="store_true", default=False)
    analyse_parser.add_argument("--output-dir", default="./output")
    analyse_parser.add_argument("--model", default="yolo11m-pose.pt")
    analyse_parser.add_argument("--joints", nargs="+", default=None,
                                help="追踪的关节 (如 right_wrist right_elbow)")
    analyse_parser.add_argument("--max-trail", type=int, default=30,
                                help="轨迹保留帧数（默认30）")

    # ui 子命令
    ui_parser = subparsers.add_parser("ui", help="启动 Gradio Web UI")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--model", default="yolo11m-pose.pt")
    ui_parser.add_argument("--share", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "analyse":
        is_right = not args.left_handed
        pipeline = ForehandPipeline(
            model_name=args.model,
            is_right_handed=is_right,
            output_dir=args.output_dir,
            tracked_joints=args.joints,
            max_trail=args.max_trail,
        )

        def progress(current, total, msg):
            pct = current / max(total, 1) * 100
            print(f"\r[{pct:5.1f}%] {msg}", end="", flush=True)

        print(f"正在分析: {args.video}")
        result = pipeline.run(args.video, progress_callback=progress)
        print()
        report = result["report"]
        print(f"检测到击球次数: {report.total_swings}")
        print(f"平均综合评分: {report.average_score:.0f}/100")
        for ev in report.swing_evaluations:
            print(f"  第{ev.swing_index + 1}次击球: {ev.overall_score:.0f}/100 ({ev.arm_style})")
        print(f"报告: {result['report_path']}")
        print(f"标注视频: {result['annotated_video_path']}")

    elif args.command == "ui":
        pipeline = ForehandPipeline(model_name=args.model)
        demo = build_gradio_ui(pipeline)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

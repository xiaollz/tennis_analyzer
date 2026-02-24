"""Tennis Analyzer v2 — Modern Forehand Evaluation.

Usage:
    # Command-line analysis
    python main.py analyse --video path/to/video.mp4 [--right-handed] [--output-dir ./output]

    # Launch Gradio web UI
    python main.py ui [--port 7860]
"""

from __future__ import annotations

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.framework_config import DEFAULT_CONFIG, FrameworkConfig
from config.keypoints import KEYPOINT_NAMES
from core.video_processor import VideoProcessor, VideoWriter
from core.pose_estimator import PoseEstimator
from analysis.trajectory import TrajectoryStore
from evaluation.forehand_evaluator import ForehandEvaluator, EvaluationReport
from evaluation.event_detector import ImpactDetector
from report.visualizer import SkeletonDrawer, TrajectoryDrawer, ChartGenerator
from report.report_generator import ReportGenerator


# =====================================================================
# Pipeline
# =====================================================================

class ForehandPipeline:
    """End-to-end pipeline: video → pose → evaluation → report."""

    def __init__(
        self,
        model_name: str = "yolo11m-pose.pt",
        is_right_handed: bool = True,
        cfg: FrameworkConfig = DEFAULT_CONFIG,
        output_dir: str = "./output",
        tracked_joints: Optional[List[str]] = None,
    ):
        self.is_right_handed = is_right_handed
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Core modules
        self.estimator = PoseEstimator(model_name=model_name)
        self.skeleton_drawer = SkeletonDrawer()

        # Trajectory drawers for user-selected joints
        default_joints = ["right_wrist", "right_elbow"] if is_right_handed else ["left_wrist", "left_elbow"]
        self.tracked_joints = tracked_joints or default_joints
        self.trajectory_drawers: Dict[str, TrajectoryDrawer] = {}

        # Colour palette for different joints
        joint_colors = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0),
            (255, 165, 0), (255, 0, 0), (0, 165, 255),
        ]
        for i, jname in enumerate(self.tracked_joints):
            color = joint_colors[i % len(joint_colors)]
            self.trajectory_drawers[jname] = TrajectoryDrawer(
                joint=jname, color=color, max_trail=200, fade=True,
            )

    def run(
        self,
        video_path: str,
        progress_callback=None,
    ) -> Dict:
        """Run the full pipeline.

        Parameters
        ----------
        video_path : str
            Path to the input video.
        progress_callback : callable, optional
            Called with (current_frame, total_frames, status_message).

        Returns
        -------
        dict with keys:
            report : EvaluationReport
            report_path : str (Markdown file)
            annotated_video_path : str
            chart_paths : dict
        """
        video_name = Path(video_path).stem
        vp = VideoProcessor(video_path)
        fps = vp.fps

        # ── Phase 1: Pose estimation + trajectory collection ─────────
        if progress_callback:
            progress_callback(0, vp.total_frames, "Starting pose estimation...")

        keypoints_series: List[np.ndarray] = []
        confidence_series: List[np.ndarray] = []
        frame_indices: List[int] = []
        frames_raw: List[np.ndarray] = []

        store = TrajectoryStore(fps=fps)

        for frame_idx, frame in vp.read_frames():
            result = self.estimator.predict(frame)

            if result["num_persons"] > 0:
                # Select the largest person (by bounding-box area)
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

            # Update trajectory drawers
            for drawer in self.trajectory_drawers.values():
                drawer.update(kp, conf)

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, vp.total_frames, "Pose estimation...")

        # ── Phase 2: Evaluation ──────────────────────────────────────
        if progress_callback:
            progress_callback(vp.total_frames, vp.total_frames, "Evaluating forehand...")

        evaluator = ForehandEvaluator(
            fps=fps,
            is_right_handed=self.is_right_handed,
            cfg=self.cfg,
        )
        report = evaluator.evaluate(keypoints_series, confidence_series, frame_indices)

        # ── Phase 3: Generate annotated video ────────────────────────
        if progress_callback:
            progress_callback(0, len(frames_raw), "Generating annotated video...")

        annotated_path = str(self.output_dir / f"{video_name}_annotated.mp4")
        with VideoWriter(annotated_path, vp.width, vp.height, fps, input_path=video_path) as writer:
            # Reset trajectory drawers for clean re-draw
            for drawer in self.trajectory_drawers.values():
                drawer.clear()

            for i, (frame, kp, conf) in enumerate(zip(frames_raw, keypoints_series, confidence_series)):
                # Draw skeleton
                annotated = self.skeleton_drawer.draw(frame, kp, conf)

                # Update and draw trajectories
                for drawer in self.trajectory_drawers.values():
                    drawer.update(kp, conf)
                    annotated = drawer.draw(annotated)

                # Mark impact frame
                if report.swing_event.impact_frame is not None and frame_indices[i] == report.swing_event.impact_frame:
                    cv2.putText(annotated, "IMPACT", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # HUD overlay
                annotated = self._draw_hud(annotated, frame_indices[i], kp, conf, report)

                writer.write(annotated)

                if progress_callback and i % 10 == 0:
                    progress_callback(i, len(frames_raw), "Writing annotated video...")

        # ── Phase 4: Generate charts ─────────────────────────────────
        if progress_callback:
            progress_callback(0, 1, "Generating charts...")

        chart_paths = self._generate_charts(report, store, video_name, frame_indices)

        # ── Phase 5: Generate report ─────────────────────────────────
        report_gen = ReportGenerator(output_dir=str(self.output_dir))
        report_path = report_gen.generate(report, video_name=video_name, chart_paths=chart_paths)

        if progress_callback:
            progress_callback(1, 1, "Done!")

        return {
            "report": report,
            "report_path": report_path,
            "annotated_video_path": annotated_path,
            "chart_paths": chart_paths,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _select_person(persons: list) -> dict:
        """Select the most prominent person (largest bounding box)."""
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

    def _draw_hud(self, frame, frame_idx, kp, conf, report: EvaluationReport):
        """Draw a minimal heads-up display on the frame."""
        h, w = frame.shape[:2]
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Overall score
        score = report.overall_score
        color = (0, 255, 0) if score >= 70 else (0, 255, 255) if score >= 50 else (0, 0, 255)
        cv2.putText(frame, f"Score: {score:.0f}", (w - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Tracked joint labels
        y_offset = h - 20
        for jname, drawer in self.trajectory_drawers.items():
            label = jname.replace("_", " ").title()
            cv2.putText(frame, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, drawer.color, 1, cv2.LINE_AA)
            y_offset -= 18

        return frame

    def _generate_charts(
        self,
        report: EvaluationReport,
        store: TrajectoryStore,
        video_name: str,
        frame_indices: List[int],
    ) -> Dict[str, str]:
        """Generate all analysis charts."""
        charts = {}
        chart_dir = self.output_dir / "charts"
        chart_dir.mkdir(exist_ok=True)

        # Radar chart
        phase_scores = {p: ps.score for p, ps in report.phase_scores.items()}
        radar_path = str(chart_dir / f"{video_name}_radar.png")
        if ChartGenerator.radar_chart(phase_scores, radar_path):
            charts["radar"] = radar_path

        # KPI bar chart
        kpi_bar_path = str(chart_dir / f"{video_name}_kpi_bar.png")
        if ChartGenerator.kpi_bar_chart(report.kpi_results, kpi_bar_path):
            charts["kpi_bar"] = kpi_bar_path

        # Joint trajectory charts
        for jname in self.tracked_joints:
            traj = store.get(jname)
            positions = traj.get_positions(smoothed=True)
            if len(positions) > 2:
                traj_path = str(chart_dir / f"{video_name}_{jname}_trajectory.png")
                ChartGenerator.joint_trajectory_chart(
                    positions, traj.frame_indices, jname.replace("_", " ").title(),
                    traj_path, impact_frame=report.swing_event.impact_frame,
                )
                charts[f"trajectory_{jname}"] = traj_path

            # Speed profile
            speeds = traj.get_speeds(smoothed=True)
            if len(speeds) > 2:
                speed_path = str(chart_dir / f"{video_name}_{jname}_speed.png")
                ChartGenerator.speed_profile_chart(
                    speeds, traj.frame_indices[1:], jname.replace("_", " ").title(),
                    speed_path, impact_frame=report.swing_event.impact_frame,
                )
                charts[f"speed_{jname}"] = speed_path

        return charts


# =====================================================================
# Gradio UI
# =====================================================================

def build_gradio_ui(pipeline: ForehandPipeline):
    """Build and return a Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(
        title="Tennis Analyzer v2 — Modern Forehand",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🎾 Tennis Analyzer v2 — Modern Forehand Evaluation")
        gr.Markdown(
            "Upload a video of a forehand swing. The analyser will evaluate your technique "
            "based on the **Modern Forehand** framework (Dr. Brian Gordon, Rick Macci, Tennis Doctor)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Forehand Video")
                with gr.Row():
                    is_right = gr.Checkbox(value=True, label="Right-handed")
                    tracked_joints_input = gr.CheckboxGroup(
                        choices=[
                            "right_wrist", "left_wrist",
                            "right_elbow", "left_elbow",
                            "right_shoulder", "left_shoulder",
                            "right_hip", "left_hip",
                            "right_knee", "left_knee",
                            "right_ankle", "left_ankle",
                            "nose",
                        ],
                        value=["right_wrist", "right_elbow"],
                        label="Joints to Track",
                    )
                analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_text = gr.Textbox(label="Status", interactive=False)
                overall_score = gr.Number(label="Overall Score", interactive=False)

        with gr.Tabs():
            with gr.Tab("Annotated Video"):
                video_output = gr.Video(label="Annotated Video")

            with gr.Tab("Phase Scores"):
                radar_chart = gr.Image(label="Phase Radar Chart")

            with gr.Tab("KPI Details"):
                kpi_bar_chart = gr.Image(label="KPI Bar Chart")
                kpi_table = gr.Dataframe(
                    headers=["KPI", "Phase", "Score", "Rating", "Value", "Feedback"],
                    label="KPI Results",
                )

            with gr.Tab("Joint Trajectories"):
                trajectory_gallery = gr.Gallery(label="Trajectory Charts", columns=2)

            with gr.Tab("Speed Profiles"):
                speed_gallery = gr.Gallery(label="Speed Profile Charts", columns=2)

            with gr.Tab("Report"):
                report_md = gr.Markdown(label="Full Report")
                report_file = gr.File(label="Download Report")

        def run_analysis(video, right_handed, tracked_joints):
            if video is None:
                return "Please upload a video.", 0, None, None, None, [], [], [], "", None

            # Reconfigure pipeline
            pipeline.is_right_handed = right_handed
            pipeline.tracked_joints = tracked_joints if tracked_joints else (
                ["right_wrist", "right_elbow"] if right_handed else ["left_wrist", "left_elbow"]
            )
            pipeline.trajectory_drawers = {}
            joint_colors = [
                (0, 255, 255), (255, 0, 255), (0, 255, 0),
                (255, 165, 0), (255, 0, 0), (0, 165, 255),
            ]
            for i, jname in enumerate(pipeline.tracked_joints):
                color = joint_colors[i % len(joint_colors)]
                pipeline.trajectory_drawers[jname] = TrajectoryDrawer(
                    joint=jname, color=color, max_trail=200, fade=True,
                )

            try:
                result = pipeline.run(video)
            except Exception as e:
                return f"Error: {e}", 0, None, None, None, [], [], [], "", None

            report = result["report"]
            charts = result["chart_paths"]

            # KPI table
            kpi_rows = []
            for k in report.kpi_results:
                val_str = f"{k.raw_value:.2f}" if k.raw_value is not None else "N/A"
                kpi_rows.append([
                    f"{k.kpi_id} {k.name}", k.phase, f"{k.score:.0f}",
                    k.rating, val_str, k.feedback,
                ])

            # Trajectory images
            traj_images = [v for k, v in charts.items() if k.startswith("trajectory_")]
            speed_images = [v for k, v in charts.items() if k.startswith("speed_")]

            # Report markdown
            report_text = ""
            if Path(result["report_path"]).exists():
                report_text = Path(result["report_path"]).read_text(encoding="utf-8")

            return (
                "Analysis complete!",
                report.overall_score,
                result["annotated_video_path"],
                charts.get("radar"),
                charts.get("kpi_bar"),
                kpi_rows,
                traj_images,
                speed_images,
                report_text,
                result["report_path"],
            )

        analyse_btn.click(
            fn=run_analysis,
            inputs=[video_input, is_right, tracked_joints_input],
            outputs=[
                status_text, overall_score, video_output,
                radar_chart, kpi_bar_chart, kpi_table,
                trajectory_gallery, speed_gallery,
                report_md, report_file,
            ],
        )

    return demo


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Tennis Analyzer v2 — Modern Forehand")
    subparsers = parser.add_subparsers(dest="command")

    # analyse sub-command
    analyse_parser = subparsers.add_parser("analyse", help="Analyse a forehand video")
    analyse_parser.add_argument("--video", required=True, help="Path to video file")
    analyse_parser.add_argument("--right-handed", action="store_true", default=True)
    analyse_parser.add_argument("--left-handed", action="store_true", default=False)
    analyse_parser.add_argument("--output-dir", default="./output")
    analyse_parser.add_argument("--model", default="yolo11m-pose.pt")
    analyse_parser.add_argument("--joints", nargs="+", default=None,
                                help="Joints to track (e.g. right_wrist right_elbow)")

    # ui sub-command
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio web UI")
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
        )

        def progress(current, total, msg):
            pct = current / max(total, 1) * 100
            print(f"\r[{pct:5.1f}%] {msg}", end="", flush=True)

        print(f"Analysing: {args.video}")
        result = pipeline.run(args.video, progress_callback=progress)
        print()
        print(f"Overall Score: {result['report'].overall_score:.0f}/100")
        print(f"Report: {result['report_path']}")
        print(f"Annotated Video: {result['annotated_video_path']}")

    elif args.command == "ui":
        pipeline = ForehandPipeline(model_name=args.model)
        demo = build_gradio_ui(pipeline)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

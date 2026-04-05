"""Tennis Analyzer v3 — 容错型正手 & 单反评估系统。

Usage:
    # 命令行分析（自动识别正手/反手）
    python main.py analyse --video path/to/video.mp4 [--right-handed] [--output-dir ./output]

    # 指定击球类型
    python main.py analyse --video path/to/video.mp4 --stroke forehand
    python main.py analyse --video path/to/video.mp4 --stroke backhand

    # 启动 Gradio Web UI
    python main.py ui [--port 7860]
"""

from __future__ import annotations

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.framework_config import DEFAULT_CONFIG, FrameworkConfig
from config.backhand_config import DEFAULT_BACKHAND_CONFIG, BackhandConfig
from config.keypoints import KEYPOINT_NAMES
from core.video_processor import VideoProcessor, VideoWriter
from core.pose_estimator import (
    PoseEstimator,
    DEFAULT_YOLO_MODEL,
)
from analysis.trajectory import TrajectoryStore
from evaluation.forehand_evaluator import ForehandEvaluator, MultiSwingReport
from evaluation.backhand_evaluator import BackhandEvaluator
from evaluation.event_detector import HybridImpactDetector, ImpactEvent, SwingPhaseEstimator
from evaluation.stroke_classifier import StrokeClassifier, StrokeType
from evaluation.vlm_analyzer import (
    KeyframeExtractor,
    VLMForehandAnalyzer,
    create_keyframe_grid,
    save_keyframe_grid,
)
from report.visualizer import SkeletonDrawer, TrajectoryDrawer, ChartGenerator, JOINT_CN
from report.report_generator import ReportGenerator


# =====================================================================
# Pipeline
# =====================================================================

class TennisAnalysisPipeline:
    """端到端流水线：视频 → 姿态估计 → 击球检测 → 自动识别 → 评估 → 报告。

    支持正手和单反两种评估模式，可自动识别或手动指定。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_YOLO_MODEL,
        is_right_handed: bool = True,
        stroke_mode: str = "forehand",  # "auto" / "forehand" / "backhand"
        fg_cfg: FrameworkConfig = DEFAULT_CONFIG,
        bh_cfg: BackhandConfig = DEFAULT_BACKHAND_CONFIG,
        output_dir: str = "./output",
        tracked_joints: Optional[List[str]] = None,
        max_trail: int = 30,
    ):
        self.is_right_handed = is_right_handed
        self.stroke_mode = stroke_mode
        self.fg_cfg = fg_cfg
        self.bh_cfg = bh_cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_trail = max_trail
        self.model_name = model_name

        # 核心模块
        self.estimator = PoseEstimator(model_name=model_name)
        self.skeleton_drawer = SkeletonDrawer()
        self.stroke_classifier = StrokeClassifier(is_right_handed=is_right_handed)

        # 默认只追踪 1 个关节：持拍手腕
        default_joints = ["right_wrist"] if is_right_handed else ["left_wrist"]
        self.tracked_joints = tracked_joints or default_joints
        self._init_trajectory_drawers()

    def set_pose_estimator(self, model_name: str) -> None:
        """动态切换姿态模型。"""
        target_model = str(model_name).strip() if model_name is not None else ""
        if not target_model:
            target_model = "auto"
        if target_model == self.model_name:
            return
        self.estimator = PoseEstimator(model_name=target_model)
        self.model_name = target_model

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
            stroke_type : str ("forehand" / "one_handed_backhand")
            classifications : list  (每次击球的分类结果)
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
            cfg=self.fg_cfg.impact_detection,
        )

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
            impact_detector.update(frame_idx, kp, conf)

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, vp.total_frames, "姿态估计中...")

        # 完成击球检测
        impact_events = self._dedupe_overlapping_impacts(impact_detector.finalize(), store, fps)

        # ── 阶段 1.5: 自动识别击球类型 ──────────────────────────────
        if progress_callback:
            progress_callback(vp.total_frames, vp.total_frames, "正在识别击球类型...")

        # 更新分类器的惯用手设置
        self.stroke_classifier = StrokeClassifier(is_right_handed=self.is_right_handed)

        classifications = []
        detected_stroke_type = "forehand"

        if self.stroke_mode == "auto" and impact_events:
            # 自动识别每次击球类型
            impact_frames = [e.impact_frame_idx for e in impact_events]
            classifications = self.stroke_classifier.classify_all_swings(
                keypoints_series, confidence_series, frame_indices, impact_frames,
            )

            # 统计多数类型
            type_counts = {}
            for cls in classifications:
                t = cls.stroke_type.value
                type_counts[t] = type_counts.get(t, 0) + 1

            if type_counts:
                majority_type = max(type_counts, key=type_counts.get)
                if majority_type in ("one_handed_backhand", "two_handed_backhand"):
                    detected_stroke_type = "one_handed_backhand"
                else:
                    detected_stroke_type = "forehand"
        elif self.stroke_mode == "backhand":
            detected_stroke_type = "one_handed_backhand"
        else:
            detected_stroke_type = "forehand"

        is_backhand = detected_stroke_type != "forehand"

        # ── 阶段 2: 评估（多次击球独立评分）──────────────────────────
        if progress_callback:
            stroke_cn = "单反" if is_backhand else "正手"
            progress_callback(vp.total_frames, vp.total_frames, f"正在评估{stroke_cn}技术...")

        if is_backhand:
            evaluator = BackhandEvaluator(
                fps=fps,
                is_right_handed=self.is_right_handed,
                cfg=self.bh_cfg,
            )
        else:
            evaluator = ForehandEvaluator(
                fps=fps,
                is_right_handed=self.is_right_handed,
                cfg=self.fg_cfg,
            )

        report = evaluator.evaluate_multi(
            keypoints_series, confidence_series, frame_indices, impact_events,
        )

        # ── 阶段 2.5: VLM 关键帧分析（仅正手）──────────────────────
        vlm_results = []
        if not is_backhand and report.swing_evaluations:
            if progress_callback:
                progress_callback(0, 1, "正在进行 VLM 视觉分析...")
            vlm_results = self._run_vlm_analysis(
                frames_raw, frame_indices, report, video_name,
                keypoints_series=keypoints_series,
                confidence_series=confidence_series,
                video_path=video_path,
            )

        # ── 阶段 3: 生成标注视频 ────────────────────────────────────
        if progress_callback:
            progress_callback(0, len(frames_raw), "正在生成标注视频...")

        annotated_path = str(self.output_dir / f"{video_name}_annotated.mp4")
        with VideoWriter(annotated_path, vp.width, vp.height, fps, input_path=video_path) as writer:
            # 重置轨迹绘制器
            for drawer in self.trajectory_drawers.values():
                drawer.clear()

            # 构建击球帧集合
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
                    swing_idx = report.impact_frames.index(current_frame)
                    self._draw_impact_marker(annotated, swing_idx + 1)

                # HUD 叠加
                annotated = self._draw_hud(
                    annotated, current_frame, report, detected_stroke_type
                )

                writer.write(annotated)

                if progress_callback and i % 10 == 0:
                    progress_callback(i, len(frames_raw), "写入标注视频...")

        # ── 阶段 4: 生成图表 ────────────────────────────────────────
        if progress_callback:
            progress_callback(0, 1, "正在生成分析图表...")

        chart_paths = self._generate_charts(
            report, store, video_name, frame_indices, is_backhand
        )

        # ── 阶段 5: 生成报告 ────────────────────────────────────────
        report_gen = ReportGenerator(output_dir=str(self.output_dir))
        report_path = report_gen.generate(
            report, video_name=video_name,
            chart_paths=chart_paths,
            stroke_type=detected_stroke_type,
            vlm_results=vlm_results,
        )

        if progress_callback:
            progress_callback(1, 1, "分析完成！")

        return {
            "report": report,
            "report_path": report_path,
            "annotated_video_path": annotated_path,
            "chart_paths": chart_paths,
            "stroke_type": detected_stroke_type,
            "classifications": classifications,
            "vlm_results": vlm_results,
        }

    # ── VLM 分析 ──────────────────────────────────────────────────────

    def _run_vlm_analysis(
        self,
        frames_raw: List[np.ndarray],
        frame_indices: List[int],
        report: MultiSwingReport,
        video_name: str,
        keypoints_series: Optional[List[np.ndarray]] = None,
        confidence_series: Optional[List[np.ndarray]] = None,
        video_path: Optional[str] = None,
    ) -> List[Optional[Dict]]:
        """Run VLM keyframe analysis for each swing. Returns list of VLM results.

        Also saves keyframe grid images to self.output_dir/charts/.
        """
        extractor = KeyframeExtractor()
        analyzer = VLMForehandAnalyzer()
        results: List[Optional[Dict]] = []

        # Limit VLM analysis to max 6 swings (evenly sampled) to save tokens
        MAX_VLM_SWINGS = 6
        all_evals = report.swing_evaluations
        if len(all_evals) > MAX_VLM_SWINGS:
            step = len(all_evals) / MAX_VLM_SWINGS
            selected_indices = {int(i * step) for i in range(MAX_VLM_SWINGS)}
            print(f"[VLM] {len(all_evals)} 次击球，均匀采样 {MAX_VLM_SWINGS} 次进行 VLM 分析")
        else:
            selected_indices = set(range(len(all_evals)))

        for ev in all_evals:
            if ev.swing_index not in selected_indices:
                results.append(None)
                continue
            # Extract keyframes (with issue annotations if keypoints available)
            keyframes = extractor.extract(
                frames_raw, frame_indices, ev.swing_event,
                keypoints_series=keypoints_series,
                confidence_series=confidence_series,
                is_right_handed=self.is_right_handed,
            )
            if not keyframes:
                results.append(None)
                continue

            # Create and save grid
            grid = create_keyframe_grid(keyframes)
            grid_dir = self.output_dir / "charts"
            grid_dir.mkdir(exist_ok=True)
            suffix = f"_{ev.swing_index}" if report.total_swings > 1 else ""
            grid_path = str(grid_dir / f"{video_name}_keyframes{suffix}.png")
            save_keyframe_grid(grid, grid_path)

            # Compute supplementary metrics (M7/M8/M9)
            supp_metrics = None
            if keypoints_series is not None and confidence_series is not None:
                evaluator = ForehandEvaluator(
                    fps=self.fps if hasattr(self, 'fps') else 30.0,
                    is_right_handed=self.is_right_handed,
                )
                # Get fps from video
                from core.video_processor import VideoProcessor
                try:
                    evaluator.fps = len(frame_indices) / max(1, frame_indices[-1] - frame_indices[0]) * frame_indices[-1] / max(1, len(frame_indices)) if frame_indices else 30.0
                except Exception:
                    pass
                supp_metrics = evaluator.compute_supplementary_metrics(
                    keypoints_series, confidence_series, frame_indices,
                    ev.swing_event,
                )

            # 提取当前 swing 的视频片段供 VLM 视频模式使用
            swing_clip_path = None
            try:
                swing_clip_path = self._extract_swing_clip(
                    video_path, ev.swing_event, frame_indices, fps,
                )
            except Exception:
                pass  # 视频片段提取失败不影响分析，回退到关键帧

            # Call VLM (v4: pure observation → diagnosis engine does reasoning)
            mode_label = "视频" if swing_clip_path else "关键帧"
            print(f"[VLM] 正在分析第 {ev.swing_index + 1} 次击球（{mode_label}模式）...")
            vlm_result = analyzer.analyze_swing(
                grid, video_path=video_path,
                supplementary_metrics=supp_metrics,
                swing_video_path=swing_clip_path,
            )
            if vlm_result is not None:
                vlm_result["keyframe_grid_path"] = grid_path
                if supp_metrics:
                    vlm_result["supplementary_metrics"] = supp_metrics

                # 诊断引擎：VLM 视觉输出 + 量化数据交叉验证
                try:
                    from evaluation.diagnosis_engine import diagnose
                    diag_metrics = {
                        "arm_torso_synchrony": supp_metrics.get("arm_torso_synchrony") if supp_metrics else None,
                        "scooping_depth": supp_metrics.get("scooping_depth") if supp_metrics else None,
                        "scooping_detected": supp_metrics.get("scooping_detected", False) if supp_metrics else False,
                        "forward_extension": supp_metrics.get("forward_extension") if supp_metrics else None,
                        "shoulder_rotation": supp_metrics.get("shoulder_rotation") if supp_metrics else None,
                        "swing_arc_ratio": supp_metrics.get("swing_arc_ratio") if supp_metrics else None,
                    }
                    vlm_result = diagnose(vlm_result, diag_metrics)
                    contradictions = vlm_result.get("contradictions", [])
                    if contradictions:
                        print(f"[诊断] 发现 {len(contradictions)} 处VLM与量化数据矛盾")
                except Exception as exc:
                    pass  # 诊断引擎失败不影响主流程

                issue_count = len(vlm_result.get('issues', []))
                rounds = vlm_result.get('diagnostic_session', {}).get('rounds', [])
                round_info = f"（{len(rounds)}轮迭代）" if rounds else ""
                print(f"[VLM] 第 {ev.swing_index + 1} 次击球分析完成，发现 {issue_count} 个问题{round_info}")
            else:
                print(f"[VLM] 第 {ev.swing_index + 1} 次击球 VLM 分析未返回结果")
            results.append(vlm_result)

        print(f"[VLM] 总计: {sum(1 for r in results if r is not None)}/{len(results)} 次击球获得 VLM 分析")
        return results

    # ── 辅助方法 ─────────────────────────────────────────────────────

    def _extract_swing_clip(
        self,
        video_path: str,
        swing_event,
        frame_indices: List[int],
        fps: float,
    ) -> Optional[str]:
        """Extract a short video clip for the current swing using ffmpeg.

        Returns path to the clip file, or None if extraction fails.
        """
        import subprocess
        prep = swing_event.prep_start_frame or frame_indices[0]
        ft_end = swing_event.followthrough_end_frame or frame_indices[-1]

        # Add 0.3s buffer on each side
        start_sec = max(0, prep / fps - 0.3)
        duration = (ft_end - prep) / fps + 0.6

        clip_dir = self.output_dir / "clips"
        clip_dir.mkdir(exist_ok=True)
        clip_path = clip_dir / f"swing_{swing_event.impact_frame}.mp4"

        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_sec:.2f}",
            "-i", str(video_path),
            "-t", f"{duration:.2f}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-an",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode == 0 and clip_path.exists():
            return str(clip_path)
        return None

    def _dedupe_overlapping_impacts(
        self,
        impact_events: List[ImpactEvent],
        store: TrajectoryStore,
        fps: float,
    ) -> List[ImpactEvent]:
        if len(impact_events) <= 1:
            return impact_events

        wrist_key = "right_wrist" if self.is_right_handed else "left_wrist"
        wrist_traj = store.get(wrist_key)
        wrist_speeds = wrist_traj.get_speeds(smoothed=True)
        speed_frame_indices = wrist_traj.frame_indices[1:] if len(wrist_traj.frame_indices) > 1 else []
        if len(wrist_speeds) == 0 or not speed_frame_indices:
            return impact_events

        estimator = SwingPhaseEstimator(fps=fps)
        overlap_tolerance = int(round(0.05 * fps))
        provisional = []
        for ev in sorted(impact_events, key=lambda item: item.impact_frame_idx):
            swing = estimator.estimate_phases(
                impact_frame=ev.impact_frame_idx,
                wrist_speeds=wrist_speeds,
                frame_indices=speed_frame_indices,
            )
            provisional.append((ev, swing))

        clusters = []
        current_cluster = [provisional[0]]
        current_end = provisional[0][1].followthrough_end_frame or provisional[0][0].impact_frame_idx
        for item in provisional[1:]:
            ev, swing = item
            prep_start = swing.prep_start_frame or ev.impact_frame_idx
            swing_end = swing.followthrough_end_frame or ev.impact_frame_idx
            if prep_start <= current_end + overlap_tolerance:
                current_cluster.append(item)
                current_end = max(current_end, swing_end)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
                current_end = swing_end
        clusters.append(current_cluster)

        deduped = []
        for cluster in clusters:
            audio_confirmed = [ev for ev, _ in cluster if ev.audio_confirmed]
            candidates = audio_confirmed if audio_confirmed else [ev for ev, _ in cluster]
            best = max(candidates, key=lambda ev: ev.peak_speed_px_s)
            deduped.append(best)

        return sorted(deduped, key=lambda item: item.impact_frame_idx)

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
        text = f"Hit #{swing_num}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(w, h) / 1300.0)
        thickness = max(1, int(font_scale * 2))
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = 10
        y = h - 12
        cv2.rectangle(
            frame,
            (x - 6, y - text_size[1] - 6),
            (x + text_size[0] + 6, y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        report: MultiSwingReport,
        stroke_type: str = "forehand",
    ) -> np.ndarray:
        """在帧上绘制 HUD 信息。"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 1200.0
        font_scale = max(0.4, font_scale)

        # 帧号
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 击球类型
        stroke_cn = "OHB" if stroke_type != "forehand" else "FH"
        cv2.putText(frame, f"Type: {stroke_cn}", (10, 50),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 击球次数
        cv2.putText(frame, f"Swings: {report.total_swings}", (10, 75),
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
            label_name = jname.replace('_', ' ').title()
            cv2.putText(frame, label_name, (10, y_offset),
                        font, font_scale * 0.8, drawer.color, 1, cv2.LINE_AA)
            y_offset -= 20

        return frame

    def _generate_charts(
        self,
        report: MultiSwingReport,
        store: TrajectoryStore,
        video_name: str,
        frame_indices: List[int],
        is_backhand: bool = False,
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

        # KPI charts removed — VLM analysis is primary output

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


# 向后兼容别名
ForehandPipeline = TennisAnalysisPipeline


def default_report_output_dir(base_dir: str = "reports") -> str:
    """默认按日期组织输出目录，如 reports/2026-02-26。"""
    return str(Path(base_dir) / datetime.now().strftime("%Y-%m-%d"))


# =====================================================================
# Gradio UI
# =====================================================================

def build_gradio_ui(pipeline: TennisAnalysisPipeline):
    """构建 Gradio Blocks 界面。"""
    import gradio as gr

    with gr.Blocks(
        title="网球分析器 v3 — 容错型正手 & 单反评估",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🎾 网球分析器 v3 — 容错型正手 & 单反技术评估")
        gr.Markdown(
            "上传挥拍视频，系统将基于 **《容错型正手》原则模型 / One-Handed Backhand** 评估您的技术。\n\n"
            "**正手评估层**: 转开/备手与下肢准备 → 转髋带动 → 前方接触 → 向外/向前穿过 → 稳定性\n\n"
            "只保留能由身体关键点稳定估计的指标，支持自动识别正手/反手、多次击球独立评分、音频+视觉协同检测击球点。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="上传挥拍视频")
                with gr.Row():
                    is_right = gr.Checkbox(value=True, label="右手持拍")
                    stroke_mode = gr.Radio(
                        choices=["auto", "forehand", "backhand"],
                        value="forehand",
                        label="击球类型",
                        info="auto=自动识别, forehand=正手, backhand=单反",
                    )
                with gr.Row():
                    max_trail_slider = gr.Slider(
                        minimum=10, maximum=60, value=30, step=5,
                        label="轨迹保留帧数",
                    )
                with gr.Row():
                    model_input = gr.Textbox(
                        value="auto",
                        label="姿态模型",
                        info=f"auto 或自定义 YOLO 权重（示例: {DEFAULT_YOLO_MODEL}）",
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
                    detected_type = gr.Textbox(label="识别击球类型", interactive=False)

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

        def run_analysis(
            video,
            right_handed,
            stroke,
            tracked_joints,
            max_trail_val,
            model_name,
        ):
            if video is None:
                return ("请上传视频。", 0, 0, "", None, None, None, None,
                        [], [], [], "", None)

            # 每次分析都按当天日期归档输出
            pipeline.output_dir = Path(default_report_output_dir())
            pipeline.output_dir.mkdir(parents=True, exist_ok=True)

            # 重新配置 pipeline
            pipeline.is_right_handed = right_handed
            pipeline.stroke_mode = stroke
            pipeline.max_trail = int(max_trail_val)
            pipeline.set_pose_estimator(model_name=model_name)
            pipeline.tracked_joints = (tracked_joints[:2] if tracked_joints else
                                       (["right_wrist"] if right_handed else ["left_wrist"]))
            pipeline._init_trajectory_drawers()

            try:
                result = pipeline.run(video)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return (f"错误: {e}", 0, 0, "", None, None, None, None,
                        [], [], [], "", None)

            report = result["report"]
            charts = result["chart_paths"]
            stroke_type = result["stroke_type"]
            stroke_cn = "单手反拍" if stroke_type != "forehand" else "正手"

            # 选择正确的阶段标题
            is_bh = stroke_type != "forehand"
            phase_titles = (
                ReportGenerator.BACKHAND_PHASE_TITLES if is_bh
                else ReportGenerator.FOREHAND_PHASE_TITLES
            )

            # KPI 表格
            kpi_rows = []
            for ev in report.swing_evaluations:
                prefix = f"[第{ev.swing_index + 1}次] " if report.total_swings > 1 else ""
                for k in ev.kpi_results:
                    val_str = f"{k.raw_value:.2f}" if k.raw_value is not None else "无数据"
                    phase_cn = phase_titles.get(k.phase, k.phase)
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

            radar_img = None  # Radar chart removed
            multi_img = charts.get("multi_swing_summary")
            kpi_bar_img = charts.get("kpi_bar") or charts.get("kpi_bar_0")

            return (
                "分析完成！",
                report.average_score,
                report.total_swings,
                stroke_cn,
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
            inputs=[
                video_input,
                is_right,
                stroke_mode,
                tracked_joints_input,
                max_trail_slider,
                model_input,
            ],
            outputs=[
                status_text, overall_score, swing_count, detected_type,
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
    parser = argparse.ArgumentParser(description="网球分析器 v3 — 容错型正手 & 单反评估")
    subparsers = parser.add_subparsers(dest="command")

    # analyse 子命令
    analyse_parser = subparsers.add_parser("analyse", help="分析挥拍视频")
    analyse_parser.add_argument("--video", required=True, help="视频文件路径")
    analyse_parser.add_argument("--right-handed", action="store_true", default=True)
    analyse_parser.add_argument("--left-handed", action="store_true", default=False)
    analyse_parser.add_argument("--stroke", choices=["auto", "forehand", "backhand"],
                                default="forehand", help="击球类型 (默认=forehand), auto=自动识别")
    analyse_parser.add_argument("--output-dir", default=None,
                                help="输出目录（默认按日期写入 reports/YYYY-MM-DD）")
    analyse_parser.add_argument("--model", default="auto",
                                help="YOLO 姿态模型名称或权重路径，auto=默认模型")
    analyse_parser.add_argument("--joints", nargs="+", default=None,
                                help="追踪的关节 (如 right_wrist right_elbow)")
    analyse_parser.add_argument("--max-trail", type=int, default=30,
                                help="轨迹保留帧数（默认30）")

    # ui 子命令
    ui_parser = subparsers.add_parser("ui", help="启动 Gradio Web UI")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--model", default="auto")
    ui_parser.add_argument("--share", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "analyse":
        is_right = not args.left_handed
        output_dir = args.output_dir or default_report_output_dir()
        pipeline = TennisAnalysisPipeline(
            model_name=args.model,
            is_right_handed=is_right,
            stroke_mode=args.stroke,
            output_dir=output_dir,
            tracked_joints=args.joints,
            max_trail=args.max_trail,
        )

        def progress(current, total, msg):
            pct = current / max(total, 1) * 100
            print(f"\r[{pct:5.1f}%] {msg}", end="", flush=True)

        print(f"正在分析: {args.video}")
        print(f"击球类型: {args.stroke}")
        print(f"姿态模型: {pipeline.estimator.model_name}")
        result = pipeline.run(args.video, progress_callback=progress)
        print()

        report = result["report"]
        stroke_cn = "单手反拍" if result["stroke_type"] != "forehand" else "正手"
        print(f"识别击球类型: {stroke_cn}")
        print(f"检测到击球次数: {report.total_swings}")
        # Show VLM scores if available, otherwise fall back to KPI scores
        vlm_scores = []
        for i, vr in enumerate(result.get("vlm_results", [])):
            if vr and vr.get("score") is not None:
                vlm_scores.append(vr["score"])
                print(f"  第{i+1}次击球: {vr['score']}/100（VLM）")
            elif i < len(report.swing_evaluations):
                print(f"  第{i+1}次击球: {report.swing_evaluations[i].overall_score:.0f}/100")
        if vlm_scores:
            avg = sum(vlm_scores) / len(vlm_scores)
            print(f"VLM 平均评分: {avg:.0f}/100")
        else:
            print(f"平均综合评分: {report.average_score:.0f}/100")
        print(f"报告: {result['report_path']}")
        print(f"标注视频: {result['annotated_video_path']}")

    elif args.command == "ui":
        pipeline = TennisAnalysisPipeline(
            model_name=args.model,
            output_dir=default_report_output_dir(),
        )
        demo = build_gradio_ui(pipeline)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

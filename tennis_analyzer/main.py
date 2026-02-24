"""Tennis Pose Analyzer - CLI entry point."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .core.pose_estimator import PoseEstimator
from .core.video_processor import VideoProcessor, VideoWriter, detect_rotation_from_pose, rotate_frame
from .core.smoother import KeypointSmoother, MetricsSmoother
from .visualization.skeleton import SkeletonDrawer, WristTracker
from .visualization.overlay import OverlayRenderer
from .visualization.big3_ui import Big3UIRenderer
from .analysis.biomechanics import BiomechanicsAnalyzer
from .analysis.audio_impact import TwoPassImpactDetector
from .analysis.kinetic_chain import KineticChainManager
from .analysis.pose_utils import (
    CAMERA_VIEW_BACK,
    CAMERA_VIEW_SIDE,
    CAMERA_VIEW_UNKNOWN,
    estimate_view_ratio,
)
from .config.keypoints import KEYPOINT_NAMES

# Colors for feedback (BGR)
COLOR_GOOD = (0, 255, 128)    # Green
COLOR_WARNING = (0, 0, 255)   # Red


def detect_video_rotation(video: VideoProcessor, estimator: PoseEstimator) -> int:
    """Detect video rotation by analyzing pose in first few frames."""
    if video.rotation != 0:
        return video.rotation

    rotations = []
    for frame_idx, frame in video.read_frames():
        if frame_idx >= 5:
            break

        results = estimator.predict(frame, conf=0.3)
        if results["num_persons"] > 0:
            person = results["persons"][0]
            rot = detect_rotation_from_pose(person["keypoints"], person["confidence"])
            if rot != 0:
                rotations.append(rot)

    video.cap.set(0, 0)

    if rotations:
        from collections import Counter
        most_common = Counter(rotations).most_common(1)
        if most_common:
            return most_common[0][0]

    return 0


def detect_camera_view(
    video: VideoProcessor,
    estimator: PoseEstimator,
    rotation: int,
    *,
    confidence: float = 0.5,
    max_probe_frames: int = 45,
    stride: int = 3,
    min_samples: int = 10,
    side_threshold: float = 0.38,
    back_threshold: float = 0.46,
) -> str:
    """Heuristically detect camera view (side vs back) from early pose frames.

    Uses shoulder_width_px / torso_height_px ratio:
      - Side view => smaller ratio (foreshortened shoulders)
      - Back view => larger ratio
    """
    ratios: list[float] = []

    for frame_idx, frame in video.read_frames():
        if frame_idx >= int(max_probe_frames):
            break
        if int(frame_idx) % max(1, int(stride)) != 0:
            continue

        if rotation != 0:
            frame = rotate_frame(frame, rotation)

        results = estimator.predict(frame, conf=float(confidence))
        if results.get("num_persons", 0) <= 0:
            continue

        person = results["persons"][0]
        ratio = estimate_view_ratio(person["keypoints"], person["confidence"])
        if ratio is None:
            continue
        ratios.append(float(ratio))

        if len(ratios) >= int(min_samples):
            break

    # Reset for the main processing pass.
    video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not ratios:
        return CAMERA_VIEW_UNKNOWN

    med = float(np.median(np.asarray(ratios, dtype=np.float32)))
    if med < float(side_threshold):
        return CAMERA_VIEW_SIDE
    if med > float(back_threshold):
        return CAMERA_VIEW_BACK
    return CAMERA_VIEW_UNKNOWN


def analyze_video(
    input_path: str,
    output_path: str,
    model: str = "yolo11m-pose.pt",
    device: str = "auto",
    show_metrics: bool = True,
    confidence: float = 0.5,
    right_handed: bool = True,
    slow_motion: float = 1.0,
    smoothing: float = 0.5,
    manual_rotation: int = None,
    impact_mode: str = "hybrid",  # "pose" | "hybrid"
    impact_merge_s: float = 0.8,
    impact_audio_tol_frames: int = 7,
    show_big3_ui: bool = False,
    report: bool = False,
    report_dir: str | None = None,
    report_sample_fps: float = 4.0,
    view: str = "auto",  # "auto" | "side" | "back"
):
    """Analyze a tennis video and output with pose overlay."""
    print(f"Loading video: {input_path}")
    video = VideoProcessor(input_path, auto_rotate=False)
    print(f"  Raw resolution: {video._raw_width}x{video._raw_height}")
    print(f"  FPS: {video.fps:.2f}")
    print(f"  Duration: {video.duration:.2f}s ({video.total_frames} frames)")

    print(f"\nLoading model: {model}")
    estimator = PoseEstimator(model_name=model, device=device)
    print(f"  Device: {estimator.device}")

    # Detect rotation
    if manual_rotation is not None:
        rotation = manual_rotation
        print(f"\n  Manual rotation: {rotation}°")
    else:
        print(f"\n  Detecting video orientation...")
        rotation = detect_video_rotation(video, estimator)
        if rotation != 0:
            print(f"  Auto-detected rotation: {rotation}° (will correct)")
        else:
            print(f"  No rotation needed")

    # Calculate output dimensions
    if rotation in (90, -90, 270, -270):
        output_width = video._raw_height
        output_height = video._raw_width
    else:
        output_width = video._raw_width
        output_height = video._raw_height

    print(f"  Output resolution: {output_width}x{output_height}")

    # Initialize components
    skeleton_drawer = SkeletonDrawer(line_thickness=1, point_radius=2)
    overlay_renderer = OverlayRenderer()
    biomechanics = BiomechanicsAnalyzer()
    wrist_tracker = WristTracker(max_trail_length=10, min_distance=8)

    # Optional Big3 UI (panel overlay).
    big3_ui = Big3UIRenderer() if show_big3_ui else None

    # Optional report generation (single-pass collector).
    report_collector = None
    report_dir_path = None
    if report:
        from .reporting.report_generator import ReportCollector

        report_collector = ReportCollector(
            fps=video.fps,
            total_frames=video.total_frames,
            sample_fps=float(report_sample_fps),
            camera_view=None,
        )
        report_collector.set_biomechanics_analyzer(biomechanics)

        if report_dir is None:
            # Default: sidecar folder next to the output video.
            base = Path(output_path)
            report_dir_path = Path(str(base.with_suffix("")) + "_report")
        else:
            report_dir_path = Path(report_dir)
            if not report_dir_path.is_absolute():
                report_dir_path = Path.cwd() / report_dir_path
        report_dir_path.mkdir(parents=True, exist_ok=True)
        (report_dir_path / "assets").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Impact detection mode
    # ------------------------------------------------------------------
    impact_events_by_frame = None
    if impact_mode == "hybrid":
        print("\nHybrid impact detection: audio onset + wrist speed peaks")
        two_pass = TwoPassImpactDetector(
            input_path,
            video.fps,
            estimator,
            is_right_handed=right_handed,
            require_audio=True,
            # Side-view / phone recordings can have a small audio-video offset.
            # A slightly wider tolerance reduces missed impacts without adding
            # many false positives because we still require a pose speed peak.
            audio_tolerance_frames=int(max(0, impact_audio_tol_frames)),
            pose_confidence=confidence,
            rotation=rotation,
            merge_within_s=float(impact_merge_s),
        )
        impact_events_by_frame = two_pass.get_impact_event_map()
        print(f"   Using {len(impact_events_by_frame)} scheduled impacts")

    # ------------------------------------------------------------------
    # Camera view (side vs back): used for metric gating & scale choice.
    # ------------------------------------------------------------------
    view = str(view or "auto").lower()
    if view not in ("auto", CAMERA_VIEW_SIDE, CAMERA_VIEW_BACK):
        view = "auto"

    if view == "auto":
        print("\nDetecting camera view (side vs back)...")
        camera_view = detect_camera_view(video, estimator, rotation, confidence=confidence)
        print(f"  Camera view: {camera_view}")
    else:
        camera_view = view
        print(f"\nCamera view: {camera_view} (forced)")

    if report_collector is not None:
        report_collector.camera_view = camera_view

    # Kinetic chain monitors (modular system)
    kinetic_chain = KineticChainManager(
        is_right_handed=right_handed,
        fps=video.fps,
        camera_view=camera_view,
        impact_events_by_frame=impact_events_by_frame,
    )

    # Smoothers
    keypoint_smoother = KeypointSmoother(smoothing_factor=smoothing)
    metrics_smoother = MetricsSmoother(smoothing_factor=smoothing + 0.2)

    # Output FPS
    output_fps = video.fps * slow_motion
    if slow_motion != 1.0:
        print(f"\nSlow motion: {slow_motion}x (output FPS: {output_fps:.2f})")

    print(f"Smoothing factor: {smoothing}")
    print(f"\nProcessing video...")

    with VideoWriter(output_path, output_width, output_height, output_fps, input_path=input_path) as writer:
        for frame_idx, frame in video.read_frames():
            # Apply rotation
            if rotation != 0:
                frame = rotate_frame(frame, rotation)

            # Pose estimation
            results = estimator.predict(frame, conf=confidence)

            chain_results = None

            # Process first detected person
            if results["num_persons"] > 0:
                person = results["persons"][0]

                # Smooth keypoints
                smoothed_keypoints = keypoint_smoother.smooth(
                    person["keypoints"],
                    person["confidence"]
                )

                # Draw skeleton
                frame = skeleton_drawer.draw(
                    frame,
                    smoothed_keypoints,
                    person["confidence"]
                )

                # Track wrist (always track)
                wrist_tracker.update(
                    smoothed_keypoints,
                    person["confidence"],
                    True,  # Always track
                    is_right_handed=right_handed
                )
                frame = wrist_tracker.draw(frame)

                # Update kinetic chain monitors (always analyze as forehand)
                chain_results = kinetic_chain.update(
                    smoothed_keypoints,
                    person["confidence"],
                    True,  # Always treat as forehand
                    frame_idx=frame_idx
                )

                # Report collection (angles sampled + per-impact snapshots).
                if report_collector is not None and chain_results:
                    report_collector.maybe_sample_angles(
                        frame_idx=int(frame_idx),
                        keypoints=smoothed_keypoints,
                        confidence=person["confidence"],
                    )
                    report_collector.on_frame(frame_idx=int(frame_idx), chain_results=chain_results)

                # Calculate and display metrics
                if show_metrics:
                    raw_metrics = biomechanics.analyze(
                        smoothed_keypoints,
                        person["confidence"]
                    )
                    if raw_metrics:
                        # Add kinetic chain metrics
                        if chain_results:
                            raw_metrics["Arm"] = chain_results["extension"]["angle"]
                        smoothed_metrics = metrics_smoother.smooth(raw_metrics)
                        # Add non-numeric metrics after smoothing
                        if chain_results:
                            # Avoid dumping raw spacing numbers; prefer delta-to-goal message when available.
                            sp_msg = str(chain_results.get("spacing", {}).get("message", "") or "").strip()
                            if sp_msg:
                                smoothed_metrics["Space"] = sp_msg.replace("[空间]", "").strip()
                        frame = overlay_renderer.draw_metrics(frame, smoothed_metrics)

                # Big3 UI panel (frozen after impact and updated once contact-zone is known).
                if big3_ui is not None and chain_results:
                    big3 = chain_results.get("big3") or {}
                    statuses = {
                        "contact_point": big3.get("contact_point"),
                        "weight_transfer": big3.get("weight_transfer"),
                        "contact_zone": big3.get("contact_zone"),
                    }
                    if big3.get("is_impact"):
                        frame = big3_ui.draw_impact_flash(frame)
                        big3_ui.show_results = True
                    if big3_ui.show_results:
                        big3_ui.frozen_results = statuses
                        frame = big3_ui.draw_status_panel(frame, statuses)

            # Draw frame info
            info = {
                "Frame": f"{frame_idx + 1}/{video.total_frames}",
            }
            info_position = "bottom-right" if show_big3_ui else "top-left"
            frame = overlay_renderer.draw_info_panel(frame, info, position=info_position)

            # Draw kinetic chain feedback (bottom-left, stacked)
            feedbacks = kinetic_chain.get_active_feedback()
            if feedbacks:
                # If we already show Big3 in the top-left panel, avoid duplicating them
                # in the bottom-left list to keep the UI clean.
                if show_big3_ui:
                    feedbacks = [
                        (name, msg, status)
                        for (name, msg, status) in feedbacks
                        if name not in ("contact_point", "weight_transfer", "contact_zone")
                    ]
                # Convert to (message, status) format for draw_feedback_list
                feedback_items = [(msg, status) for (_, msg, status) in feedbacks]
                frame = overlay_renderer.draw_feedback_list(frame, feedback_items, position="bottom-left")

            writer.write(frame)

            # Progress
            if (frame_idx + 1) % 30 == 0 or frame_idx == video.total_frames - 1:
                progress = (frame_idx + 1) / video.total_frames * 100
                print(f"  Progress: {progress:.1f}% ({frame_idx + 1}/{video.total_frames})", end="\r")

    print(f"\n\nOutput saved to: {output_path}")

    # Write report after the video is complete.
    if report_collector is not None and report_dir_path is not None:
        from .reporting.report_generator import ReportData, write_markdown_report

        report_data = ReportData(
            input_path=str(input_path),
            output_video_path=str(output_path),
            fps=float(video.fps),
            total_frames=int(video.total_frames),
            camera_view=getattr(report_collector, "camera_view", None),
            impacts=list(report_collector.impacts),
            angle_series=dict(report_collector.angle_series),
        )
        report_path = write_markdown_report(report=report_data, report_dir=report_dir_path)
        print(f"Report saved to: {report_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tennis Pose Analyzer - Analyze tennis videos with pose estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tennis-analyze input.mp4 -o output.mp4
  tennis-analyze input.mp4 -o output.mp4 --slow 0.5
  tennis-analyze input.mp4 -o output.mp4 --rotate 90
        """
    )

    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output video file path")
    parser.add_argument("-m", "--model", default="yolo11m-pose.pt", help="YOLO pose model")
    parser.add_argument("-d", "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("-c", "--confidence", type=float, default=0.5)
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics overlay")
    parser.add_argument("--left-handed", action="store_true", help="Player is left-handed")
    parser.add_argument("-s", "--slow", type=float, default=1.0, help="Slow motion factor (0-1)")
    parser.add_argument("--smooth", type=float, default=0.5, help="Smoothing factor (0-1)")
    parser.add_argument("-r", "--rotate", type=int, choices=[0, 90, -90, 180], default=None)
    parser.add_argument(
        "--impact-mode",
        choices=["pose", "hybrid"],
        default="hybrid",
        help="Impact detection mode: pose-only (real-time) or hybrid (audio+pose two-pass)",
    )
    parser.add_argument(
        "--impact-merge-s",
        type=float,
        default=0.8,
        help="(hybrid only) Merge close impacts within this many seconds (bounce+hit de-dup)",
    )
    parser.add_argument(
        "--impact-audio-tol",
        type=int,
        default=7,
        help="(hybrid only) Audio onset tolerance window in video frames (default: 7). Use 0 for strict matching.",
    )
    parser.add_argument(
        "--big3-ui",
        action="store_true",
        help="Show Big3 panel overlay (contact point / weight transfer / contact zone)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate an offline markdown report + charts next to the output video",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Report output directory (default: <output>_report)",
    )
    parser.add_argument(
        "--report-sample-fps",
        type=float,
        default=4.0,
        help="Sample FPS for angle curves in the report (default: 4)",
    )
    parser.add_argument(
        "--view",
        default="auto",
        choices=["auto", "side", "back"],
        help="Camera view. 'auto' uses a pose heuristic; side/back gates metrics accordingly.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.slow <= 0 or args.slow > 1.0:
        print(f"Error: Slow motion factor must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    if args.smooth < 0 or args.smooth > 1.0:
        print(f"Error: Smoothing factor must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        suffix = "_analyzed"
        if args.slow != 1.0:
            suffix += f"_slow{args.slow}"
        output_path = str(input_path.parent / f"{input_path.stem}{suffix}.mp4")

    try:
        analyze_video(
            input_path=str(input_path),
            output_path=output_path,
            model=args.model,
            device=args.device,
            show_metrics=not args.no_metrics,
            confidence=args.confidence,
            right_handed=not args.left_handed,
            slow_motion=args.slow,
            smoothing=args.smooth,
            manual_rotation=args.rotate,
            impact_mode=args.impact_mode,
            impact_merge_s=args.impact_merge_s,
            impact_audio_tol_frames=args.impact_audio_tol,
            show_big3_ui=args.big3_ui,
            report=args.report,
            report_dir=args.report_dir,
            report_sample_fps=args.report_sample_fps,
            view=args.view,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

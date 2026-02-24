#!/usr/bin/env python3
"""
Big 3 Tennis Analyzer
=====================
Focuses on the 3 most critical checkpoints for forehand technique:
1. Contact Point - Is ball hit in front of body?
2. Weight Transfer - Back foot rotated at contact?
3. Contact Zone - Strings follow forward+up after contact?

Based on Patrick Brodfeld's coaching methodology.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tennis_analyzer.core.pose_estimator import PoseEstimator
from tennis_analyzer.analysis.impact import WristSpeedImpactDetector
from tennis_analyzer.config.keypoints import KEYPOINT_NAMES


# COCO Keypoint Indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class CheckpointResult:
    """Result for a single checkpoint analysis."""
    name: str
    score: str  # "GOOD", "OK", "BAD"
    value: float  # Numeric measurement
    unit: str  # "px", "degrees", etc.
    message: str  # Human-readable explanation
    frame_idx: int  # Frame where this was measured
    frame_path: str  # Path to the frame image


@dataclass
class Big3Report:
    """Complete analysis report."""
    contact_point: Optional[CheckpointResult]
    weight_transfer: Optional[CheckpointResult]
    contact_zone: Optional[CheckpointResult]
    contact_frame_idx: int
    priority_fix: str  # Which checkpoint to fix first
    tip: str  # Coaching tip for the priority fix


class Big3Analyzer:
    """
    Analyzer focused on the Big 3 Checkpoints.
    
    Usage:
        analyzer = Big3Analyzer("data/processed/my_forehand")
        report = analyzer.analyze()
        print(report)
    """
    
    def __init__(
        self,
        data_dir: str,
        model_path: str = None,
        *,
        is_right_handed: bool = True,
        fps: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.frames_data = []
        self.is_right_handed = is_right_handed
        self.fps = float(fps) if fps and fps > 0 else 30.0
        
        # Default model path
        if model_path is None:
            model_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../models/yolo11m-pose.pt'
            ))
        
        print("🎾 Big 3 Analyzer - Loading...")
        self.pose_estimator = PoseEstimator(model_name=model_path)
        self._load_frames()
    
    def _load_frames(self):
        """Load all extracted frames and run pose estimation."""
        image_files = sorted(list(self.data_dir.glob("frame_*.jpg")))
        
        if not image_files:
            raise ValueError(f"No frames found in {self.data_dir}")
        
        print(f"📷 Analyzing {len(image_files)} frames...")
        
        for img_path in image_files:
            # Parse frame info from filename: frame_123_phase_label.jpg
            parts = img_path.stem.split('_')
            frame_idx = int(parts[1])
            phase_label = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
            
            # Load image and run pose estimation
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            results = self.pose_estimator.predict(img)
            
            if results["persons"]:
                self.frames_data.append({
                    "frame_idx": frame_idx,
                    "phase_label": phase_label,
                    "keypoints": np.array(results["persons"][0]["keypoints"]),
                    "confidence": np.array(results["persons"][0]["confidence"]),
                    "image_path": str(img_path),
                    "image": img
                })
        
        print(f"✅ Loaded {len(self.frames_data)} frames with valid poses")

    def _dominant_wrist_idx(self) -> int:
        return RIGHT_WRIST if self.is_right_handed else LEFT_WRIST

    def _dominant_hip_idx(self) -> int:
        return RIGHT_HIP if self.is_right_handed else LEFT_HIP

    def _back_ankle_idx(self) -> int:
        # Heuristic: for right-hander forehand the back foot is usually the right foot.
        return RIGHT_ANKLE if self.is_right_handed else LEFT_ANKLE

    def _infer_forward_sign(self, end_idx: int, window: int = 7) -> float:
        """Infer forward swing direction (+1 or -1) from recent wrist dx."""
        wrist_idx = self._dominant_wrist_idx()
        start = max(0, end_idx - window)
        xs = []
        for f in self.frames_data[start : end_idx + 1]:
            conf = f["confidence"]
            if conf[wrist_idx] > 0.5:
                xs.append(float(f["keypoints"][wrist_idx][0]))
        if len(xs) < 3:
            return 1.0
        dx = np.diff(np.asarray(xs, dtype=np.float32))
        med = float(np.median(dx))
        if abs(med) < 1e-3:
            return 1.0
        return 1.0 if med > 0 else -1.0
    
    def _find_contact_frame(self) -> int:
        """
        Find the contact frame using the unified impact detector.

        Primary:
        - Wrist speed peak detector (deceleration after acceleration), using the
          *original* frame indices from filenames to compute dt.

        Fallback (if no peak found):
        - Frame where the dominant wrist is most "in front" of the dominant hip,
          along the inferred forward axis.
        """
        det = WristSpeedImpactDetector(
            fps=self.fps,
            is_right_handed=self.is_right_handed,
            cooldown_frames=int(round(self.fps * 0.45)),  # avoid double triggers
            max_frame_gap=int(round(self.fps * 0.5)),  # extracted swing frames are often ~0.2s apart
            min_peak_speed_px_s=250.0,  # extracted frames are sparse -> lower threshold
            min_peak_speed_sw_s=1.5,
            peak_over_baseline_ratio=1.2,
        )

        events: List[tuple] = []
        for i, frame in enumerate(self.frames_data):
            event, _speed = det.update(frame["frame_idx"], frame["keypoints"], frame["confidence"])
            if event is not None:
                score = event.peak_speed_sw_s if event.peak_speed_sw_s is not None else event.peak_speed_px_s
                events.append((score, event.impact_frame_idx))

        if events:
            _score, impact_frame_idx = max(events, key=lambda t: t[0])
            # Map to the closest extracted frame.
            idx = int(np.argmin([abs(f["frame_idx"] - impact_frame_idx) for f in self.frames_data]))
            return idx

        # Fallback: wrist most in front of hip (along inferred forward axis)
        wrist_idx = self._dominant_wrist_idx()
        hip_idx = self._dominant_hip_idx()
        contact_idx = 0
        best = -float("inf")
        for i, frame in enumerate(self.frames_data):
            kps = frame["keypoints"]
            conf = frame["confidence"]
            if conf[wrist_idx] < 0.5 or conf[hip_idx] < 0.5:
                continue
            forward_sign = self._infer_forward_sign(i)
            delta = (float(kps[wrist_idx][0]) - float(kps[hip_idx][0])) * forward_sign
            if delta > best:
                best = delta
                contact_idx = i
        return contact_idx
    
    def analyze_contact_point(self, contact_idx: int) -> CheckpointResult:
        """
        Checkpoint 1: Contact Point
        Measures how far in front of the body the ball is hit.
        
        Method: Compare wrist X position to hip X position at contact.
        Good: Wrist 50+ pixels ahead of hip
        OK: Wrist 0-50 pixels ahead
        Bad: Wrist behind hip (negative)
        """
        frame = self.frames_data[contact_idx]
        kps = frame["keypoints"]
        conf = frame["confidence"]

        wrist_idx = self._dominant_wrist_idx()
        hip_idx = self._dominant_hip_idx()

        # Need wrist and hip
        if conf[wrist_idx] < 0.5 or conf[hip_idx] < 0.5:
            return CheckpointResult(
                name="Contact Point",
                score="UNKNOWN",
                value=0,
                unit="px",
                message="Could not detect wrist or hip",
                frame_idx=frame["frame_idx"],
                frame_path=frame["image_path"]
            )

        forward_sign = self._infer_forward_sign(contact_idx)
        wrist_x = float(kps[wrist_idx][0])
        hip_x = float(kps[hip_idx][0])
        delta_px_forward = (wrist_x - hip_x) * forward_sign

        # Convert to cm if shoulders are confident.
        pixels_per_cm = None
        if conf[LEFT_SHOULDER] > 0.35 and conf[RIGHT_SHOULDER] > 0.35:
            sw_px = float(np.linalg.norm(kps[RIGHT_SHOULDER] - kps[LEFT_SHOULDER]))
            if sw_px > 1e-6:
                pixels_per_cm = sw_px / 41.0

        delta_cm = (delta_px_forward / pixels_per_cm) if pixels_per_cm else None
        
        # Determine score
        if delta_cm is not None:
            if 20.0 <= delta_cm <= 40.0:
                score = "GOOD"
                message = f"✅ 击球点舒服在前方 +{delta_cm:.0f}cm ({delta_px_forward:+.0f}px)"
            elif delta_cm >= 10.0:
                score = "OK"
                message = f"⚠️ 击球点略靠后 +{delta_cm:.0f}cm（目标 20-40cm）"
            else:
                score = "BAD"
                message = f"❌ 触球太晚 +{delta_cm:.0f}cm（目标 20-40cm）"
        else:
            if delta_px_forward >= 50:
                score = "GOOD"
                message = f"✅ Contact point in front +{delta_px_forward:.0f}px"
            elif delta_px_forward >= 0:
                score = "OK"
                message = f"⚠️ Contact point could be more in front (+{delta_px_forward:.0f}px)"
            else:
                score = "BAD"
                message = f"❌ Late contact! {abs(delta_px_forward):.0f}px behind hip"
        
        return CheckpointResult(
            name="Contact Point",
            score=score,
            value=delta_px_forward,
            unit="px",
            message=message,
            frame_idx=frame["frame_idx"],
            frame_path=frame["image_path"]
        )
    
    def analyze_weight_transfer(self, contact_idx: int) -> CheckpointResult:
        """
        Checkpoint 2: Weight Transfer
        Checks if back foot is rotated/lifted at contact (showing weight transfer).
        
        Method: Compare back ankle Y position between preparation and contact.
        Also check hip rotation (shoulder-hip angle change).
        """
        frame = self.frames_data[contact_idx]
        kps = frame["keypoints"]
        conf = frame["confidence"]
        
        # Get first frame as reference (preparation)
        prep_frame = self.frames_data[0]
        prep_kps = prep_frame["keypoints"]
        prep_conf = prep_frame["confidence"]
        
        ankle_idx = self._back_ankle_idx()
        hip_idx = self._dominant_hip_idx()

        # Check back ankle
        if conf[ankle_idx] < 0.5 or prep_conf[ankle_idx] < 0.5:
            return CheckpointResult(
                name="Weight Transfer",
                score="UNKNOWN",
                value=0,
                unit="px",
                message="Could not detect ankle",
                frame_idx=frame["frame_idx"],
                frame_path=frame["image_path"]
            )
        
        # Ankle Y change (lower Y = higher position in image coords)
        ankle_y_prep = float(prep_kps[ankle_idx][1])
        ankle_y_contact = float(kps[ankle_idx][1])
        ankle_lift = ankle_y_prep - ankle_y_contact  # Positive = lifted
        
        # Hip forward check (camera direction aware).
        forward_sign = self._infer_forward_sign(contact_idx)
        hip_x_prep = float(prep_kps[hip_idx][0]) if prep_conf[hip_idx] > 0.5 else 0.0
        hip_x_contact = float(kps[hip_idx][0]) if conf[hip_idx] > 0.5 else 0.0
        hip_forward = (hip_x_contact - hip_x_prep) * forward_sign
        
        # Combined score
        transfer_score = ankle_lift + hip_forward * 0.5
        
        if transfer_score >= 30:
            score = "GOOD"
            message = f"✅ Great weight transfer! Back foot engaged, hip rotated"
        elif transfer_score >= 10:
            score = "OK"
            message = f"⚠️ Some weight transfer. Try to engage back foot more"
        else:
            score = "BAD"
            message = f"❌ Flat-footed! No weight transfer detected"
        
        return CheckpointResult(
            name="Weight Transfer",
            score=score,
            value=transfer_score,
            unit="score",
            message=message,
            frame_idx=frame["frame_idx"],
            frame_path=frame["image_path"]
        )
    
    def analyze_contact_zone(self, contact_idx: int) -> CheckpointResult:
        """
        Checkpoint 3: Contact Zone
        Checks if the racket follows through forward+up after contact.
        
        Method: Track wrist trajectory for 3-5 frames after contact.
        Good: Forward (X increasing) + Up (Y decreasing)
        Bad: Sideways only or downward
        """
        # Get frames after contact
        post_contact_frames = self.frames_data[contact_idx:contact_idx + 5]
        
        if len(post_contact_frames) < 2:
            frame = self.frames_data[contact_idx]
            return CheckpointResult(
                name="Contact Zone",
                score="UNKNOWN",
                value=0,
                unit="trajectory",
                message="Not enough frames after contact",
                frame_idx=frame["frame_idx"],
                frame_path=frame["image_path"]
            )
        
        wrist_idx = self._dominant_wrist_idx()

        # Track wrist movement
        wrist_positions = []
        for frame in post_contact_frames:
            kps = frame["keypoints"]
            conf = frame["confidence"]
            if conf[wrist_idx] > 0.5:
                wrist_positions.append(kps[wrist_idx])
        
        if len(wrist_positions) < 2:
            frame = self.frames_data[contact_idx]
            return CheckpointResult(
                name="Contact Zone",
                score="UNKNOWN",
                value=0,
                unit="trajectory",
                message="Could not track wrist after contact",
                frame_idx=frame["frame_idx"],
                frame_path=frame["image_path"]
            )
        
        # Calculate trajectory
        start_pos = wrist_positions[0]
        end_pos = wrist_positions[-1]

        forward_sign = self._infer_forward_sign(contact_idx)
        delta_x = (end_pos[0] - start_pos[0]) * forward_sign  # Forward motion
        delta_y = start_pos[1] - end_pos[1]  # Upward motion (inverted Y)
        
        # Determine score
        frame = post_contact_frames[0]
        
        if delta_x > 20 and delta_y > 20:
            score = "GOOD"
            message = f"✅ Perfect contact zone! Forward (+{delta_x:.0f}px) and up (+{delta_y:.0f}px)"
        elif delta_x > 20:
            score = "OK"
            message = f"⚠️ Good forward motion. Add more upward brush for topspin"
        elif delta_y > 20:
            score = "OK"
            message = f"⚠️ Good upward motion. Needs more forward penetration"
        else:
            score = "BAD"
            message = f"❌ Poor contact zone. Wrist going sideways, not forward+up"
        
        return CheckpointResult(
            name="Contact Zone",
            score=score,
            value=delta_x + delta_y,
            unit="px",
            message=message,
            frame_idx=frame["frame_idx"],
            frame_path=frame["image_path"]
        )
    
    def analyze(self) -> Big3Report:
        """Run full Big 3 analysis and return report."""
        print("\n🔍 Running Big 3 Analysis...")
        
        # Step 1: Find contact frame
        contact_idx = self._find_contact_frame()
        print(f"📍 Contact frame identified: {self.frames_data[contact_idx]['frame_idx']}")
        
        # Step 2: Analyze each checkpoint
        contact_point = self.analyze_contact_point(contact_idx)
        weight_transfer = self.analyze_weight_transfer(contact_idx)
        contact_zone = self.analyze_contact_zone(contact_idx)
        
        # Step 3: Determine priority fix
        priority_fix = "Contact Point"  # Default
        tip = "Shorten backswing, prepare earlier with unit turn"
        
        # Priority order: Contact Point > Weight Transfer > Contact Zone
        if contact_point.score == "BAD":
            priority_fix = "Contact Point"
            tip = "Shorten backswing, prepare earlier with unit turn"
        elif weight_transfer.score == "BAD":
            priority_fix = "Weight Transfer"
            tip = "Drive back hip forward, let back foot rotate naturally"
        elif contact_zone.score == "BAD":
            priority_fix = "Contact Zone"
            tip = "Extend through the ball, aim forward+up not sideways"
        elif contact_point.score == "OK":
            priority_fix = "Contact Point"
            tip = "Move contact point further in front for more control"
        
        return Big3Report(
            contact_point=contact_point,
            weight_transfer=weight_transfer,
            contact_zone=contact_zone,
            contact_frame_idx=self.frames_data[contact_idx]["frame_idx"],
            priority_fix=priority_fix,
            tip=tip
        )
    
    def print_report(self, report: Big3Report):
        """Print a formatted report to console."""
        print("\n" + "=" * 60)
        print("🎾 BIG 3 TENNIS TECHNIQUE ANALYSIS")
        print("=" * 60)
        
        for i, checkpoint in enumerate([report.contact_point, report.weight_transfer, report.contact_zone], 1):
            if checkpoint:
                score_emoji = {"GOOD": "✅", "OK": "⚠️", "BAD": "❌"}.get(checkpoint.score, "❓")
                print(f"\n{i}. {checkpoint.name.upper():<20} {score_emoji} {checkpoint.score}")
                print(f"   {checkpoint.message}")
                print(f"   📷 Frame: {checkpoint.frame_idx}")
        
        print("\n" + "-" * 60)
        print(f"🎯 PRIORITY FIX: {report.priority_fix}")
        print(f"💡 TIP: {report.tip}")
        print("=" * 60)
    
    def save_report(self, report: Big3Report, output_dir: str = None):
        """Save report as markdown file."""
        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports'))
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "big3_summary.md")
        
        with open(output_path, 'w') as f:
            f.write("# 🎾 Big 3 Tennis Technique Analysis\n\n")
            f.write(f"**Contact Frame:** {report.contact_frame_idx}\n\n")
            
            f.write("## Checkpoints\n\n")
            f.write("| # | Checkpoint | Score | Details |\n")
            f.write("|---|------------|-------|----------|\n")
            
            for i, cp in enumerate([report.contact_point, report.weight_transfer, report.contact_zone], 1):
                if cp:
                    score_emoji = {"GOOD": "✅", "OK": "⚠️", "BAD": "❌"}.get(cp.score, "❓")
                    f.write(f"| {i} | {cp.name} | {score_emoji} {cp.score} | {cp.value:.1f}{cp.unit} |\n")
            
            f.write(f"\n## Priority Fix\n\n")
            f.write(f"**Focus on:** {report.priority_fix}\n\n")
            f.write(f"**Coaching Tip:** {report.tip}\n")
        
        print(f"\n📄 Report saved to: {output_path}")
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Big 3 Tennis Analyzer - Focus on what matters most")
    parser.add_argument("data_dir", type=str, help="Directory containing extracted frames")
    parser.add_argument("--output", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()
    
    # Run analysis
    analyzer = Big3Analyzer(args.data_dir)
    report = analyzer.analyze()
    
    # Print and save report
    analyzer.print_report(report)
    analyzer.save_report(report, args.output)


if __name__ == "__main__":
    main()

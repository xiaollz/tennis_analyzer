#!/usr/bin/env python3
"""
Big 3 Video Player (Hybrid Impact Detection)
=============================================
Uses AUDIO + WRIST VELOCITY for robust impact detection.

Workflow:
1. First pass: Scan video to find verified impacts (audio onset + velocity peak)
2. Second pass: Calculate Big 3 at each verified impact
3. Render: Display frozen Big 3 results after each impact

Usage:
    python3 scripts/big3_video_player.py video.mp4 --output annotated.mp4
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tennis_analyzer.core.pose_estimator import PoseEstimator
from tennis_analyzer.analysis.audio_impact import TwoPassImpactDetector
from tennis_analyzer.analysis.big3_monitors import (
    ContactPointMonitor, WeightTransferMonitor, ContactZoneMonitor, Status
)
from tennis_analyzer.visualization.big3_ui import Big3UIRenderer
from tennis_analyzer.visualization.skeleton import SkeletonDrawer, WristTracker


class Big3VideoPlayer:
    """
    Video player with robust Big 3 analysis.
    Uses two-pass impact detection (audio + velocity).
    """
    
    def __init__(self, video_path: str, 
                 model_path: str = None,
                 is_right_handed: bool = True):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        
        if not self.video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {video_path}")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.1f}, Frames: {self.frame_count}")
        
        # Default model path
        if model_path is None:
            model_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../models/yolo11m-pose.pt'
            ))
        
        # Initialize pose estimator
        print("🔧 Loading pose estimator...")
        self.pose_estimator = PoseEstimator(model_name=model_path)
        
        # Two-pass impact detection (audio + velocity)
        print("🔍 Running two-pass impact detection...")
        self.impact_detector = TwoPassImpactDetector(
            video_path, self.fps, self.pose_estimator, is_right_handed
        )
        self.impact_frames = self.impact_detector.get_impact_frames()
        
        # Initialize visualization
        self.skeleton_renderer = SkeletonDrawer()
        self.wrist_tracker = WristTracker(max_trail_length=20)
        self.ui_renderer = Big3UIRenderer()
        
        # Big 3 settings
        self.is_right_handed = is_right_handed
        
        # Pre-calculate Big 3 for each verified impact
        self.impact_results = {}
        if self.impact_frames:
            print("📊 Calculating Big 3 for each impact...")
            self._calculate_big3_for_impacts()
        
        # State
        # 0-based frame index for consistency with OpenCV and TwoPassImpactDetector
        self.current_frame = 0
        self.paused = False
        self.show_skeleton = True
        self.impact_flash_frames = 0
        
        print("✅ Ready!")
    
    def _calculate_big3_for_impacts(self):
        """Calculate Big 3 metrics at each verified impact frame."""
        for impact_frame in self.impact_frames:
            # Collect frames around impact
            start_frame = max(0, impact_frame - 10)
            end_frame = min(self.frame_count, impact_frame + 10)
            
            self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Initialize monitors
            cp_monitor = ContactPointMonitor(self.is_right_handed)
            wt_monitor = WeightTransferMonitor(self.is_right_handed)
            cz_monitor = ContactZoneMonitor(self.is_right_handed)
            
            # Collect results
            cp_at_impact = None
            wt_at_impact = None
            cz_last = None
            
            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = self.video.read()
                if not ret:
                    break
                
                results = self.pose_estimator.predict(frame)
                if not results["persons"]:
                    continue
                
                person = results["persons"][0]
                keypoints = np.array(person["keypoints"])
                confidence = np.array(person["confidence"])
                
                # Update monitors
                cp_result = cp_monitor.update(keypoints, confidence)
                wt_result = wt_monitor.update(
                    keypoints,
                    confidence,
                    frame_idx=frame_num,
                    is_impact=(frame_num == impact_frame),
                    impact_frame_idx=impact_frame,
                )
                
                # At impact frame, capture results
                if frame_num == impact_frame:
                    cp_at_impact = cp_result
                    wt_at_impact = wt_result
                    cz_monitor.signal_impact(frame_num)
                    # Seed contact zone with the actual impact frame wrist position.
                    cz_last = cz_monitor.update(keypoints, confidence)
                
                # Post-impact: update contact zone
                if frame_num > impact_frame:
                    cz_last = cz_monitor.update(keypoints, confidence)
            
            # Final contact zone result after the post-impact window.
            cz_at_impact = cz_last
            
            # Store results
            if cp_at_impact:
                self.impact_results[impact_frame] = {
                    "contact_point": cp_at_impact,
                    "weight_transfer": wt_at_impact,
                    "contact_zone": cz_at_impact,
                }
        
        print(f"   Calculated Big 3 for {len(self.impact_results)} impacts")
    
    def _get_current_result(self):
        """Get the Big 3 result for the current frame (from most recent impact)."""
        current_result = None
        for impact_frame in sorted(self.impact_results.keys()):
            if self.current_frame >= impact_frame:
                current_result = self.impact_results[impact_frame]
        return current_result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        # Pose estimation
        results = self.pose_estimator.predict(frame)
        
        keypoints = None
        confidence = None
        
        if results["persons"]:
            person = results["persons"][0]
            keypoints = np.array(person["keypoints"])
            confidence = np.array(person["confidence"])
        
        # Check for impact flash
        is_impact = self.impact_detector.is_impact_frame(self.current_frame)
        
        if is_impact:
            self.impact_flash_frames = 5
        
        show_flash = self.impact_flash_frames > 0
        if self.impact_flash_frames > 0:
            self.impact_flash_frames -= 1
        
        # Draw skeleton
        if self.show_skeleton and keypoints is not None:
            frame = self.skeleton_renderer.draw(
                frame, keypoints, confidence,
                color_override=(100, 255, 100)
            )
        
        # Draw wrist trail
        if keypoints is not None:
            self.wrist_tracker.update(keypoints, confidence, 
                                      is_stroke=True, 
                                      is_right_handed=self.is_right_handed)
        frame = self.wrist_tracker.draw(frame)
        
        # Get frozen Big 3 result
        frozen_result = self._get_current_result()
        
        if frozen_result:
            self.ui_renderer.frozen_results = frozen_result
            self.ui_renderer.show_results = True
        
        frame = self.ui_renderer.render_overlay(
            frame,
            statuses=frozen_result or {},
            phase="击球" if is_impact else "",
            frame_num=self.current_frame,
            total_frames=self.frame_count,
            show_impact_flash=show_flash,
            is_actual_impact=is_impact
        )
        
        return frame
    
    def run(self, window_name: str = "Big 3 Tennis Analysis"):
        """Run interactive video player."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\n🎮 Controls:")
        print("   SPACE - Play/Pause")
        print("   S     - Toggle skeleton")
        print("   R     - Reset")
        print("   Q/ESC - Quit\n")
        
        while True:
            if not self.paused:
                ret, frame = self.video.read()
                if not ret:
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    self.ui_renderer.reset()
                    continue

                frame = self.process_frame(frame)
                cv2.imshow(window_name, frame)
                self.current_frame += 1
            
            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('s'):
                self.show_skeleton = not self.show_skeleton
            elif key == ord('r'):
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                self.ui_renderer.reset()
        
        cv2.destroyAllWindows()
        self.video.release()
    
    def export(self, output_path: str) -> str:
        """Export video with Big 3 overlay and audio."""
        import subprocess
        import tempfile
        
        print(f"\n📦 Exporting to: {output_path}")
        
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, self.fps, 
                                  (self.width, self.height))
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.ui_renderer.reset()
        frame_idx = 0
        
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            
            # Keep 0-based indexing consistent with OpenCV and TwoPassImpactDetector.
            self.current_frame = frame_idx
            frame = self.process_frame(frame)
            writer.write(frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                progress = frame_idx / self.frame_count * 100
                print(f"   Progress: {progress:.1f}%")
        
        writer.release()
        
        # Merge with audio
        print("🔊 Adding audio...")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', self.video_path,
                '-c:v', 'libx264',   # Use H.264 codec for compatibility
                '-pix_fmt', 'yuv420p', # Ensure pixel format is supported by all players
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Exported: {output_path}")
            else:
                import shutil
                shutil.move(temp_video, output_path)
        except:
            import shutil
            shutil.move(temp_video, output_path)
        
        try:
            os.remove(temp_video)
        except:
            pass
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Big 3 Tennis Analysis (Hybrid Impact Detection)"
    )
    parser.add_argument("video", type=str, help="Video file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path")
    parser.add_argument("--left-handed", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    
    args = parser.parse_args()
    
    player = Big3VideoPlayer(args.video, is_right_handed=not args.left_handed)
    
    if args.output:
        player.export(args.output)
        if args.no_display:
            return
    
    if not args.no_display:
        player.run()


if __name__ == "__main__":
    main()

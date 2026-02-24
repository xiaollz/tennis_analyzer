
import json
import numpy as np
import statistics
import os
import sys
import glob
from pathlib import Path
import math

# Add the parent directory to sys.path so we can import tennis_analyzer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class EnhancedCoachAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.frames_data = []
        self.audio_context = {}
        self.load_data()

    def load_data(self):
        # Load audio context
        audio_path = self.data_dir / "audio_context.json"
        if audio_path.exists():
            with open(audio_path, 'r') as f:
                self.audio_context = json.load(f)
        
        from tennis_analyzer.core.pose_estimator import PoseEstimator
        
        print("Loading YOLO model for analysis...")
        self.pose_estimator = PoseEstimator(model_name=os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yolo11m-pose.pt')))
        
        image_files = sorted(list(self.data_dir.glob("frame_*.jpg")))
        
        print(f"Analyzing {len(image_files)} frames...")
        for img_path in image_files:
            parts = img_path.stem.split('_')
            frame_idx = int(parts[1])
            phase_label = "_".join(parts[2:]) 
            
            import cv2
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            results = self.pose_estimator.predict(img)
            
            frame_data = {
                "frame_idx": frame_idx,
                "phase_label": phase_label,
                "persons": results["persons"],
                "image_path": str(img_path)
            }
            self.frames_data.append(frame_data)

    def analyze_mechanics(self):
        if not self.frames_data:
            return None

        # Containers for metrics
        knee_flexions = []
        elbow_angles = []
        
        contact_frame = None
        min_hand_shoulder_dist = float('inf')
        
        for i, frame in enumerate(self.frames_data):
            if not frame['persons']: continue
            person = frame['persons'][0]
            kps = person['keypoints'] 
            conf = person['confidence']
            
            # 1. Knee Flexion (Right Knee)
            # COCO: 12=R_Hip, 14=R_Knee, 16=R_Ankle
            if conf[12] > 0.5 and conf[14] > 0.5 and conf[16] > 0.5:
                angle = self._calculate_angle(kps[12], kps[14], kps[16])
                # Check frames that are NOT follow_through or recovery for loading
                if frame['phase_label'] not in ['follow_through', 'recovery']:
                    knee_flexions.append(angle)

            # 2. Elbow Angle & Contact Heuristics
            # COCO: 6=R_Shoulder, 8=R_Elbow, 10=R_Wrist
            if conf[6] > 0.5 and conf[8] > 0.5 and conf[10] > 0.5:
                angle = self._calculate_angle(kps[6], kps[8], kps[10])
                elbow_angles.append(angle)
                
                # Check for Contact Frame candidates
                # Priority 1: Explicit Label
                if frame['phase_label'] == 'contact':
                    contact_frame = frame
                
                # Priority 2: Forward Swing + Wrist furthest X (assuming L->R play)
                # Or just use the last frame labeled 'forward_swing' before 'follow_through'
                if not contact_frame and frame['phase_label'] == 'forward_swing':
                     # Keep updating; the last forward swing frame is likely closest to contact
                     # before it transitions to follow through.
                     # Better yet, let's look for the max extension or specific geometric cues
                     pass

        # Heuristic Backup for Contact: Pick the last 'forward_swing' frame
        if not contact_frame:
            forward_swing_frames = [f for f in self.frames_data if f['phase_label'] == 'forward_swing']
            if forward_swing_frames:
                contact_frame = forward_swing_frames[-1]

        return {
            "knee_flexion_min": min(knee_flexions) if knee_flexions else None,
            "contact_frame": contact_frame,
            "elbow_angles": elbow_angles
        }

    def _calculate_angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def generate_report(self):
        metrics = self.analyze_mechanics()
        if not metrics:
            print("❌ No valid metric data extracted.")
            return

        report_lines = []
        report_lines.append(f"\n# 🎾 Tennis Technique Analysis Report")
        report_lines.append(f"**Analysis Mode:** Adaptive Frame Rate + Audio Context")
        
        if self.audio_context:
            report_lines.append(f"🔊 **Audio Track:** Extracted ({len(self.audio_context.get('audio_base64', ''))} bytes)")

        # 1. Loading Phase
        min_knee = metrics.get('knee_flexion_min')
        report_lines.append(f"\n## 1. Power Loading (Knee Bend)")
        if min_knee:
            report_lines.append(f"- **Max Knee Flexion:** {min_knee:.1f}°")
            if min_knee > 150:
                report_lines.append("   ❌ **Issue:** Legs too straight (Insufficient Loading).")
                report_lines.append("   💡 **Fix:** \"Sit in the chair\" before you swing.")
            elif 130 <= min_knee <= 150:
                report_lines.append("   ✅ **Status:** Good athletic stance.")
            else:
                report_lines.append("   ⚠️ **Note:** Very deep knee bend detected.")
        else:
            report_lines.append("   ⚠️ Could not detect knees during preparation/swing.")

        # 2. Contact Point
        contact = metrics.get('contact_frame')
        report_lines.append(f"\n## 2. Contact Point Analysis")
        if contact:
            person = contact['persons'][0]
            kps = person['keypoints']
            r_wrist_x = kps[10][0]
            r_shoulder_x = kps[6][0]
            
            delta = r_wrist_x - r_shoulder_x
            
            report_lines.append(f"- **Best Est. Contact Frame:** {contact['frame_idx']} ({contact['phase_label']})")
            report_lines.append(f"- **Extension:** Wrist is {abs(delta):.1f}px {'ahead of' if delta > 0 else 'behind'} shoulder")
            
            if delta > 0:
                report_lines.append("   ✅ **Status:** Good contact point in front of body.")
            else:
                report_lines.append("   ❌ **Issue:** Late contact (Jamming).")
                report_lines.append("   💡 **Fix:** shorten backswing, meet the ball earlier.")
                
            c_elbow = self._calculate_angle(kps[6], kps[8], kps[10])
            report_lines.append(f"- **Arm Structure:** Elbow angle {c_elbow:.1f}°")
            if 100 < c_elbow < 160:
                report_lines.append("   ℹ️ **Style:** Double Bend / Semi-straight (Modern Standard)")
            elif c_elbow >= 160:
                report_lines.append("   ℹ️ **Style:** Straight Arm (High Leverage)")
            else:
                report_lines.append("   ❌ **Issue:** Arm too bent ( cramped stroke ).")
        else:
            report_lines.append("   ⚠️ No contact frame candidates found.")
            
        full_report = "\n".join(report_lines)
        print(full_report)
        
        # Save to file
        output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../reports/tennis_analysis_report.md"))
        with open(output_file, "w") as f:
            f.write(full_report)
        print(f"\n✅ Report saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate tennis analysis report from extracted frames.")
    parser.add_argument("data_dir", type=str, help="Directory containing the extracted frames (e.g., data/processed/extracted_frames_adaptive/video_name)")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if os.path.exists(data_dir):
        analyzer = EnhancedCoachAnalyzer(data_dir)
        analyzer.generate_report()
    else:
        print(f"Data directory not found: {data_dir}")

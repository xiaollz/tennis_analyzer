
import argparse
import sys
import os
import cv2
import json
import base64
import subprocess
from pathlib import Path
import time

# Add the parent directory to sys.path so we can import tennis_analyzer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tennis_analyzer.core.video_processor import VideoProcessor
from tennis_analyzer.core.pose_estimator import PoseEstimator
from tennis_analyzer.core.action_classifier import ActionClassifier, PhaseDetector, StrokePhase, StrokeType

def extract_audio(video_path):
    """Extract audio from video and return as base64 string."""
    try:
        # Use ffmpeg to extract audio to a temporary wav file
        temp_audio_path = Path(video_path).with_suffix(".temp.wav")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",          # No video
            "-acodec", "pcm_s16le", # PCM 16-bit little endian
            "-ar", "16000", # 16kHz
            "-ac", "1",     # Mono
            str(temp_audio_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        if not temp_audio_path.exists():
            print("Warning: Audio extraction failed, no output file.")
            return None
            
        with open(temp_audio_path, "rb") as f:
            audio_data = f.read()
            
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        
        # Cleanup
        temp_audio_path.unlink()
        
        return base64_audio
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file with adaptive frame rate.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/extracted_frames')), help="Directory to save extracted frames.")
    parser.add_argument("--model", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yolo11m-pose.pt')), help="YOLO model to use.")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using model: {args.model}")

    try:
        # Initialize components
        processor = VideoProcessor(str(video_path))
        pose_estimator = PoseEstimator(model_name=args.model)
        classifier = ActionClassifier()
        phase_detector = PhaseDetector()
        
        # Audio Extraction
        print("Extracting audio context...")
        audio_b64 = extract_audio(video_path)
        if audio_b64:
            audio_context = {
                "video_name": video_path.name,
                "audio_base64": audio_b64,
                "format": "wav",
                "sample_rate": 16000
            }
            with open(output_dir / "audio_context.json", "w") as f:
                json.dump(audio_context, f)
            print("Audio context saved.")
        
        # Adaptive Frame Extraction
        fps = processor.fps
        target_interval_default = 1.0  # 1 FPS default
        target_interval_high_precision = 0.2 # 5 FPS for swings
        
        last_save_time = -10.0 # Initialize to save first frame
        saved_count = 0
        
        print(f"Video FPS: {fps}")
        print("Starting adaptive extraction...")
        
        for frame_idx, frame in processor.read_frames():
            current_time = frame_idx / fps
            
            # 1. Pose Estimation
            results = pose_estimator.predict(frame)
            
            is_swinging = False
            stroke_phase = StrokePhase.RECOVERY
            stroke_type = StrokeType.UNKNOWN
            
            if results["num_persons"] > 0:
                person = results["persons"][0] # Assume primary person is index 0
                kpts = person["keypoints"]
                conf = person["confidence"]
                
                # 2. Action Classification
                stroke_type, _ = classifier.classify(kpts, conf)
                
                # 3. Phase Detection
                stroke_phase = phase_detector.detect(kpts, conf, stroke_type, fps)
                
                # Determine if we are in a high-interest phase
                if stroke_phase in [StrokePhase.FORWARD_SWING, StrokePhase.CONTACT, StrokePhase.FOLLOW_THROUGH]:
                    is_swinging = True
            
            # Determine target interval
            target_interval = target_interval_high_precision if is_swinging else target_interval_default
            
            # Save frame if enough time has passed
            if current_time - last_save_time >= target_interval:
                output_filename = output_dir / f"frame_{frame_idx:04d}_{stroke_phase.value}.jpg"
                cv2.imwrite(str(output_filename), frame)
                saved_count += 1
                last_save_time = current_time
                
                if saved_count % 10 == 0:
                     print(f"Saved {saved_count} frames (Time: {current_time:.1f}s, Phase: {stroke_phase.value})...", end='\r')
            
        print(f"\nFinished! Extracted {saved_count} frames.")
        print(f"Frames saved to: {output_dir.absolute()}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

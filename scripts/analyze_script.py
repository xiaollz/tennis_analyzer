
import json
import numpy as np
import sys
import os

# Add the current directory to sys.path so we can import tennis_analyzer
sys.path.append(os.getcwd())

from tennis_analyzer.core.pose_estimator import PoseEstimator
from tennis_analyzer.core.video_processor import VideoProcessor
from tennis_analyzer.analysis.biomechanics import BiomechanicsAnalyzer

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj

def main():
    video_path = "/Users/qsy/Desktop/tennis/video/ac51f1737d6fff40e86bddc6bc39552b.mp4"
    output_path = "/Users/qsy/Desktop/tennis/analysis_results.json"
    
    print(f"Analyzing video: {video_path}")
    
    try:
        processor = VideoProcessor(video_path)
        # Use a faster model for quick analysis if accuracy is sufficient, but 
        # user wants "analysis" so 'm' (medium) or 'l' (large) is better. 
        # default is 'm' in PoseEstimator.
        estimator = PoseEstimator(model_name="yolo11m-pose.pt") 
        analyzer = BiomechanicsAnalyzer()
    except Exception as e:
        print(f"Error initializing components: {e}")
        return

    results_data = []
    
    frame_count = 0
    # Process every frame
    for frame_idx, frame in processor.read_frames():
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...")
            
        pose_results = estimator.predict(frame)
        
        frame_data = {
            "frame_idx": frame_idx,
            "persons": []
        }
        
        if "persons" in pose_results:
            for person in pose_results["persons"]:
                keypoints = person["keypoints"]
                confidence = person["confidence"]
                
                metrics = analyzer.analyze(keypoints, confidence)
                
                # Calculate Body Center
                body_center = analyzer.get_body_center(keypoints, confidence)
                
                person_data = {
                    "keypoints": numpy_to_python(keypoints),
                    "confidence": numpy_to_python(confidence),
                    "metrics": metrics,
                    "body_center": body_center
                }
                frame_data["persons"].append(person_data)
        
        results_data.append(frame_data)
        
    print(f"Finished processing {frame_count} frames.")
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, default=numpy_to_python)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

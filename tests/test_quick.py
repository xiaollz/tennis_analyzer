#!/usr/bin/env python3
"""Quick test script for tennis analyzer."""

import importlib.util

# This file is primarily an integration sanity-check script (model + webcam/image).
# In CI / minimal environments, the heavy deps might not be installed.
if importlib.util.find_spec("ultralytics") is None:
    import pytest

    pytest.skip("ultralytics not installed; skipping integration script", allow_module_level=True)

import cv2
import numpy as np
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis_analyzer.core import PoseEstimator
from tennis_analyzer.visualization import SkeletonDrawer, OverlayRenderer
from tennis_analyzer.analysis import BiomechanicsAnalyzer


def test_with_image(image_path: str = None):
    """Test pose estimation with an image."""
    print("Initializing components...")
    estimator = PoseEstimator(model_name="yolo11m-pose.pt", device="mps")
    drawer = SkeletonDrawer()
    overlay = OverlayRenderer()
    analyzer = BiomechanicsAnalyzer()

    if image_path and Path(image_path).exists():
        print(f"Loading image: {image_path}")
        frame = cv2.imread(image_path)
    else:
        # Create a test image with text
        print("No image provided, using webcam for single frame test...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam. Please provide an image path.")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture frame")
            return

    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    print("Running pose estimation...")

    results = estimator.predict(frame, conf=0.5)
    print(f"Detected {results['num_persons']} person(s)")

    for i, person in enumerate(results["persons"]):
        print(f"\nPerson {i+1}:")
        frame = drawer.draw(frame, person["keypoints"], person["confidence"])

        metrics = analyzer.analyze(person["keypoints"], person["confidence"])
        for name, value in metrics.items():
            print(f"  {name}: {value:.1f}°")

        frame = overlay.draw_metrics(frame, metrics)

    # Save result
    output_path = "output/test_result.jpg"
    Path("output").mkdir(exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to test image")
    args = parser.parse_args()

    test_with_image(args.image)

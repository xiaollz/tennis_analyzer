"""YOLO pose wrapper for pose estimation."""

import numpy as np
from pathlib import Path
from ultralytics import YOLO


class PoseEstimator:
    """Wrapper for YOLO pose estimation model."""

    def __init__(self, model_name: str = "yolo11m-pose.pt", device: str = "auto"):
        """
        Initialize pose estimator.

        Args:
            model_name: YOLO pose model name (yolo11n-pose, yolo11s-pose, yolo11m-pose, etc.)
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        # Ultralytics will attempt an online download if the weight file is missing.
        # In this repo we keep weights under ./models/; resolve that path first.
        self.model = YOLO(self._resolve_model_path(model_name))

    def _resolve_model_path(self, model_name: str) -> str:
        p = Path(str(model_name))
        if p.exists():
            return str(p)

        # Repo layout: <root>/tennis_analyzer/core/pose_estimator.py
        # -> <root>/models/<weights>.pt
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / p.name,
            root / "models" / p.name,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)

        # Fall back to whatever Ultralytics understands (may download if online).
        return str(model_name)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def predict(self, frame: np.ndarray, conf: float = 0.5) -> dict:
        """
        Run pose estimation on a single frame.

        Args:
            frame: BGR image as numpy array
            conf: Confidence threshold

        Returns:
            Dictionary with keypoints and metadata
        """
        results = self.model.predict(
            frame,
            conf=conf,
            device=self.device,
            verbose=False
        )

        return self._parse_results(results[0])

    def _parse_results(self, result) -> dict:
        """Parse YOLO results into structured format."""
        output = {
            "persons": [],
            "num_persons": 0,
        }

        if result.keypoints is None or len(result.keypoints) == 0:
            return output

        keypoints_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

        for person_idx, kpts in enumerate(keypoints_data):
            person = {
                "keypoints": kpts[:, :2],  # (17, 2) x, y coordinates
                "confidence": kpts[:, 2],   # (17,) confidence scores
                "bbox": None,
            }

            # Get bounding box if available
            if result.boxes is not None and len(result.boxes) > person_idx:
                person["bbox"] = result.boxes[person_idx].xyxy.cpu().numpy()[0]

            output["persons"].append(person)

        output["num_persons"] = len(output["persons"])
        return output

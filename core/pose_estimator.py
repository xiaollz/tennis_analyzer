"""YOLO Pose wrapper for human pose estimation.

This module is intentionally kept thin: it wraps the Ultralytics YOLO pose
model and returns structured per-person keypoint data.  All downstream
analysis (angles, trajectories, evaluation) lives in other modules.
"""

import numpy as np
from pathlib import Path
from typing import Optional


class PoseEstimator:
    """Wrapper for YOLO pose estimation model (COCO 17 keypoints)."""

    def __init__(self, model_name: str = "yolo11m-pose.pt", device: str = "auto"):
        from ultralytics import YOLO

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = YOLO(self._resolve_model_path(model_name))

    # ── model path resolution ────────────────────────────────────────
    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        p = Path(model_name)
        if p.exists():
            return str(p)
        root = Path(__file__).resolve().parents[1]
        for candidate in [root / p.name, root / "models" / p.name]:
            if candidate.exists():
                return str(candidate)
        return str(model_name)  # fallback: let Ultralytics download

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── inference ─────────────────────────────────────────────────────
    def predict(self, frame: np.ndarray, conf: float = 0.5) -> dict:
        """Run pose estimation on a single BGR frame.

        Returns
        -------
        dict with keys:
            persons : list[dict]   – each dict has 'keypoints' (17,2), 'confidence' (17,), 'bbox'
            num_persons : int
        """
        results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)
        return self._parse_results(results[0])

    @staticmethod
    def _parse_results(result) -> dict:
        output = {"persons": [], "num_persons": 0}
        if result.keypoints is None or len(result.keypoints) == 0:
            return output

        kp_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
        for idx, kpts in enumerate(kp_data):
            person = {
                "keypoints": kpts[:, :2],   # (17, 2)
                "confidence": kpts[:, 2],    # (17,)
                "bbox": None,
            }
            if result.boxes is not None and len(result.boxes) > idx:
                person["bbox"] = result.boxes[idx].xyxy.cpu().numpy()[0]
            output["persons"].append(person)

        output["num_persons"] = len(output["persons"])
        return output

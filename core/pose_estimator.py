"""YOLO pose-estimation wrapper with COCO-17 output."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


COCO_KEYPOINT_COUNT = 17
DEFAULT_YOLO_MODEL = "yolo11m-pose.pt"


class PoseEstimator:
    """YOLO inference wrapper with COCO-17 compatible output."""

    _YOLO_MODEL_ALIASES: Dict[str, str] = {
        "": DEFAULT_YOLO_MODEL,
        "auto": DEFAULT_YOLO_MODEL,
        "default": DEFAULT_YOLO_MODEL,
        "yolo": DEFAULT_YOLO_MODEL,
        "yolo11m": DEFAULT_YOLO_MODEL,
        "yolo11m-pose": DEFAULT_YOLO_MODEL,
        "yolo11m-pose.pt": DEFAULT_YOLO_MODEL,
    }

    def __init__(self, model_name: str = DEFAULT_YOLO_MODEL, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.backend = "yolo"
        self.model_name = self._resolve_model_name(model_name)
        self._init_yolo()

    @classmethod
    def _resolve_model_name(cls, model_name: Optional[str]) -> str:
        raw = "" if model_name is None else str(model_name).strip()
        normalized = raw.lower().replace("_", "-").replace(" ", "")
        return cls._YOLO_MODEL_ALIASES.get(normalized, raw)

    def _init_yolo(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "未安装 ultralytics。请先执行: pip install -r requirements.txt"
            ) from exc

        resolved = self._resolve_existing_file(self.model_name)
        model_path = resolved if resolved is not None else self.model_name
        self.model = YOLO(model_path)
        # Ultralytics on some macOS/MPS setups can stall; fallback to CPU for robustness.
        self._yolo_device = "cpu" if self.device == "mps" else self.device

    @staticmethod
    def _resolve_existing_file(model_name: str) -> Optional[str]:
        p = Path(model_name)
        if p.exists():
            return str(p)
        root = Path(__file__).resolve().parents[1]
        for candidate in (root / p.name, root / "models" / p.name):
            if candidate.exists():
                return str(candidate)
        return None

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

    def predict(self, frame: np.ndarray, conf: float = 0.5) -> dict:
        return self._predict_yolo(frame, conf=conf)

    def _predict_yolo(self, frame: np.ndarray, conf: float = 0.5) -> dict:
        output = {"persons": [], "num_persons": 0}

        results = self.model(
            source=frame,
            conf=max(0.05, float(conf)),
            verbose=False,
            device=self._yolo_device,
        )
        if not results:
            return output

        result = results[0]
        keypoints_obj = getattr(result, "keypoints", None)
        if keypoints_obj is None or getattr(keypoints_obj, "xy", None) is None:
            return output

        keypoints_xy = keypoints_obj.xy
        if hasattr(keypoints_xy, "cpu"):
            keypoints_xy = keypoints_xy.cpu().numpy()
        keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
        if keypoints_xy.ndim != 3 or keypoints_xy.shape[0] == 0:
            return output

        keypoints_scores = getattr(keypoints_obj, "conf", None)
        if keypoints_scores is not None:
            if hasattr(keypoints_scores, "cpu"):
                keypoints_scores = keypoints_scores.cpu().numpy()
            keypoints_scores = np.asarray(keypoints_scores, dtype=np.float32)
            if keypoints_scores.ndim == 3 and keypoints_scores.shape[-1] == 1:
                keypoints_scores = keypoints_scores[..., 0]
        else:
            keypoints_scores = np.ones(
                (keypoints_xy.shape[0], keypoints_xy.shape[1]),
                dtype=np.float32,
            )

        boxes_xyxy = None
        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is not None and getattr(boxes_obj, "xyxy", None) is not None:
            boxes_xyxy = boxes_obj.xyxy
            if hasattr(boxes_xyxy, "cpu"):
                boxes_xyxy = boxes_xyxy.cpu().numpy()
            boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)

        for idx in range(keypoints_xy.shape[0]):
            kp = keypoints_xy[idx]
            score_row = keypoints_scores[idx] if idx < keypoints_scores.shape[0] else None
            if score_row is None:
                score_row = np.ones(kp.shape[0], dtype=np.float32)

            kp, score_row = self._align_to_coco17(kp, score_row)
            if conf > 0 and float(np.max(score_row)) < conf:
                continue

            bbox = None
            if boxes_xyxy is not None and idx < boxes_xyxy.shape[0]:
                bbox = boxes_xyxy[idx][:4]
            if bbox is None:
                bbox = self._extract_bbox(kp, score_row)

            output["persons"].append({
                "keypoints": kp,
                "confidence": score_row,
                "bbox": bbox,
            })

        output["num_persons"] = len(output["persons"])
        return output

    @staticmethod
    def _align_to_coco17(keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if keypoints.shape[0] == COCO_KEYPOINT_COUNT:
            return keypoints.astype(np.float32), scores.astype(np.float32)

        # Fallback for non-17-keypoint models: keep first min(n, 17) entries.
        aligned_kp = np.zeros((COCO_KEYPOINT_COUNT, 2), dtype=np.float32)
        aligned_scores = np.zeros(COCO_KEYPOINT_COUNT, dtype=np.float32)
        n = min(COCO_KEYPOINT_COUNT, keypoints.shape[0])
        aligned_kp[:n] = keypoints[:n]
        aligned_scores[:n] = scores[:n]
        return aligned_kp, aligned_scores

    @staticmethod
    def _extract_bbox(keypoints: np.ndarray, scores: np.ndarray) -> Optional[np.ndarray]:
        valid = scores > 0.05
        points = keypoints[valid] if np.any(valid) else keypoints
        if points.size == 0:
            return None

        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)

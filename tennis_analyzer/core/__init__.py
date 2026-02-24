from .video_processor import VideoProcessor, VideoWriter
from .action_classifier import ActionClassifier, StrokeType, PhaseDetector, StrokePhase
from .smoother import KeypointSmoother, MetricsSmoother, ActionSmoother

# NOTE: PoseEstimator depends on heavy runtime deps (ultralytics/torch). Keep the
# core package importable in lightweight environments (unit tests, docs).
try:
    from .pose_estimator import PoseEstimator  # type: ignore
except Exception:  # pragma: no cover
    PoseEstimator = None  # type: ignore

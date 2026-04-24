"""Audio-driven video segmentation — 把整段练习视频按击球声切成小片段。"""

from segmentation.segmenter import SoundBasedSegmenter, ClipSpec, SegmentationResult

__all__ = ["SoundBasedSegmenter", "ClipSpec", "SegmentationResult"]

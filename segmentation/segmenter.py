"""音频驱动的视频分段器。

接收一段练习视频，按击球声把视频切成若干 2-3 秒的片段。每个片段
包含 1-2 次击球（按时间近邻合并），并为每个片段生成缩略图。

依赖：ffmpeg（系统二进制）+ OpenCV + 已有的 AudioOnsetDetector。
"""

from __future__ import annotations

import os
import json
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from evaluation.event_detector import AudioOnsetDetector


# ── 数据结构 ────────────────────────────────────────────────────────

@dataclass
class ClipSpec:
    """单个片段的元数据。"""
    clip_id: str
    video_id: str
    index: int                      # 在视频内的序号（0 开始）
    start_s: float                  # 片段起始秒
    end_s: float                    # 片段结束秒
    impact_times_s: List[float]     # 击球时刻（秒，可能 1-2 个）
    onset_strength: float           # 平均 onset 强度（越大越可能是击球）
    clip_path: str                  # 片段 mp4 路径
    thumbnail_path: str             # 缩略图 jpg 路径
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationResult:
    video_id: str
    video_path: str
    fps: float
    duration_s: float
    total_onsets: int
    clips: List[ClipSpec] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "total_onsets": self.total_onsets,
            "clips": [c.to_dict() for c in self.clips],
            "error": self.error,
        }


# ── 核心分段器 ──────────────────────────────────────────────────────

class SoundBasedSegmenter:
    """按击球声把视频切成 2-3 秒的片段。

    Parameters
    ----------
    pre_s : float
        每个片段首个击球声之前保留的时间（秒），用于 Unit Turn / 引拍
    post_s : float
        每个片段最后一个击球声之后保留的时间（秒），用于随挥
    group_gap_s : float
        相邻击球声小于该间隔时合并到同一片段（默认 1.2s）
    max_hits_per_clip : int
        单个片段内允许的最大击球数（默认 2）
    onset_threshold_std : float
        音频 onset 的标准差阈值倍数，越大越严格（默认 3.0）
    dominance_window_s : float
        簇内主导峰窗口：任意 N 秒窗口内，只保留最强 onset，其余视为
        回声/噪声。N=0 关闭过滤（默认 0.7s）。
    dominance_ratio : float
        簇内非主导峰的强度上限比例：弱于主导峰 dominance_ratio 倍的
        onset 被丢弃。默认 0.65（即 < 65% 主导强度的视为回声）。
    short_video_force_single : float
        若视频总时长 ≤ 该阈值，强制只产出 1 个片段（取最强 onset 为锚）。
        防止用户拍的"单次击球"短视频被误切。默认 4.0s。
    """

    def __init__(
        self,
        pre_s: float = 1.5,
        post_s: float = 1.8,
        group_gap_s: float = 1.5,
        max_hits_per_clip: int = 2,
        onset_threshold_std: float = 3.0,
        dominance_window_s: float = 0.7,
        dominance_ratio: float = 0.65,
        short_video_force_single: float = 4.0,
    ):
        self.pre_s = pre_s
        self.post_s = post_s
        self.group_gap_s = group_gap_s
        self.max_hits_per_clip = max_hits_per_clip
        self.onset_threshold_std = onset_threshold_std
        self.dominance_window_s = dominance_window_s
        self.dominance_ratio = dominance_ratio
        self.short_video_force_single = short_video_force_single

    # ── 主入口 ─────────────────────────────────────────────────

    def segment(
        self,
        video_path: str,
        video_id: str,
        output_dir: str,
        progress_cb=None,
    ) -> SegmentationResult:
        """执行分段并导出所有片段 + 缩略图。

        Parameters
        ----------
        video_path : str
            原视频路径
        video_id : str
            本次视频的唯一 ID（用于命名片段）
        output_dir : str
            片段与缩略图输出目录（会自动创建 clips/ 子目录）
        progress_cb : Callable[[float, str], None] | None
            进度回调 (0-1, 描述)
        """
        output_dir = Path(output_dir)
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        def _progress(p: float, msg: str):
            if progress_cb:
                try:
                    progress_cb(p, msg)
                except Exception:
                    pass

        # 1) 读取视频元信息
        _progress(0.05, "读取视频元信息")
        fps, total_frames, duration_s = self._probe_video(video_path)
        if fps <= 0 or duration_s <= 0:
            return SegmentationResult(
                video_id=video_id,
                video_path=video_path,
                fps=fps,
                duration_s=duration_s,
                total_onsets=0,
                error="无法读取视频元信息",
            )

        # 2) 音频 onset 检测
        _progress(0.15, "检测击球声")
        detector = AudioOnsetDetector(
            video_path=video_path,
            fps=fps,
            threshold_std=self.onset_threshold_std,
        )
        if not detector.available or not detector.onset_frames:
            return SegmentationResult(
                video_id=video_id,
                video_path=video_path,
                fps=fps,
                duration_s=duration_s,
                total_onsets=0,
                error="未检测到击球声（视频可能无音频或声音过弱）",
            )

        # 帧号 → 秒
        onset_times: List[float] = sorted(f / fps for f in detector.onset_frames)
        strengths: Dict[float, float] = {
            f / fps: detector.onset_strength.get(f, 0.0)
            for f in detector.onset_frames
        }

        # 3) 过滤"回声/噪声" onset（簇内主导峰过滤）
        _progress(0.22, "过滤回声")
        onset_times = self._filter_ghost_onsets(onset_times, strengths)

        # 4) 短视频特例：≤ 阈值时强制只产出 1 个片段
        if duration_s <= self.short_video_force_single and onset_times:
            anchor = max(onset_times, key=lambda t: strengths.get(t, 0.0))
            onset_times = [anchor]

        # 5) 按时间间隔分组 → 每组变成一个片段
        _progress(0.30, "合并相邻击球")
        groups: List[List[float]] = []
        current: List[float] = []
        for t in onset_times:
            if not current:
                current = [t]
                continue
            if (t - current[-1]) <= self.group_gap_s and len(current) < self.max_hits_per_clip:
                current.append(t)
            else:
                groups.append(current)
                current = [t]
        if current:
            groups.append(current)

        # 4) 逐组导出片段
        total = len(groups)
        clips: List[ClipSpec] = []
        for i, impacts in enumerate(groups):
            pct = 0.30 + 0.60 * (i / max(total, 1))
            _progress(pct, f"导出片段 {i+1}/{total}")

            start_s = max(0.0, impacts[0] - self.pre_s)
            end_s = min(duration_s, impacts[-1] + self.post_s)
            # 避免片段过短（< 1.2s）
            if (end_s - start_s) < 1.2:
                end_s = min(duration_s, start_s + 1.2)

            clip_id = f"{video_id}_c{i:03d}"
            clip_path = clips_dir / f"{clip_id}.mp4"
            thumb_path = clips_dir / f"{clip_id}.jpg"

            ok = self._export_clip(video_path, str(clip_path), start_s, end_s)
            if not ok:
                continue

            # 缩略图：在第一个击球时刻附近取一帧
            impact_frame = int(impacts[0] * fps)
            self._extract_thumbnail(video_path, str(thumb_path), impact_frame, fps)

            avg_strength = float(np.mean([strengths.get(t, 0.0) for t in impacts]))
            clips.append(ClipSpec(
                clip_id=clip_id,
                video_id=video_id,
                index=i,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                impact_times_s=[round(t, 3) for t in impacts],
                onset_strength=round(avg_strength, 4),
                clip_path=str(clip_path),
                thumbnail_path=str(thumb_path),
                duration_s=round(end_s - start_s, 3),
            ))

        _progress(1.0, f"完成，共 {len(clips)} 个片段")

        return SegmentationResult(
            video_id=video_id,
            video_path=video_path,
            fps=fps,
            duration_s=duration_s,
            total_onsets=len(onset_times),
            clips=clips,
        )

    # ── 内部工具 ────────────────────────────────────────────────

    def _filter_ghost_onsets(
        self,
        onset_times: List[float],
        strengths: Dict[float, float],
    ) -> List[float]:
        """簇内主导峰过滤：任意 dominance_window_s 内只保留最强 onset。

        球拍击球后常有 50-300ms 的回声 / 球落地 / 网响等次级声峰，
        这些 onset 会被检测器同样捕获。此函数在每个 onset 周围划一个
        窗口，若有更强的 onset 存在，且当前 onset 的强度 < 主导峰 ×
        dominance_ratio，则丢弃。
        """
        if self.dominance_window_s <= 0 or len(onset_times) <= 1:
            return list(onset_times)

        kept: List[float] = []
        for t in onset_times:
            s = strengths.get(t, 0.0)
            # 找窗口内的主导强度
            dominant_strength = s
            for t2 in onset_times:
                if t2 == t:
                    continue
                if abs(t2 - t) <= self.dominance_window_s:
                    s2 = strengths.get(t2, 0.0)
                    if s2 > dominant_strength:
                        dominant_strength = s2
            # 若自身强度低于主导峰 × ratio，认为是回声
            if dominant_strength > 0 and s < dominant_strength * self.dominance_ratio:
                continue
            kept.append(t)
        return kept

    def _probe_video(self, video_path: str) -> tuple[float, int, float]:
        """返回 (fps, total_frames, duration_s)。失败返回 (0,0,0)。"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0, 0, 0.0
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if fps <= 0:
                fps = 30.0
            duration = total_frames / fps if total_frames > 0 else 0.0
            return fps, total_frames, duration
        except Exception:
            return 0.0, 0, 0.0

    def _export_clip(
        self, src: str, dst: str, start_s: float, end_s: float
    ) -> bool:
        """用 ffmpeg 剪出一段视频（含音频）。"""
        duration = max(0.1, end_s - start_s)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-i", src,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            dst,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1024
        except Exception:
            return False

    def _extract_thumbnail(
        self, src: str, dst: str, frame_idx: int, fps: float
    ) -> bool:
        """在指定帧附近截取一张缩略图。"""
        try:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                return False
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return False
            # 缩放到长边不超过 720
            h, w = frame.shape[:2]
            scale = 720.0 / max(h, w) if max(h, w) > 720 else 1.0
            if scale < 1.0:
                frame = cv2.resize(
                    frame, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            return bool(cv2.imwrite(dst, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]))
        except Exception:
            return False


# ── 便利函数 ────────────────────────────────────────────────────────

def segment_video(
    video_path: str,
    video_id: str,
    output_dir: str,
    progress_cb=None,
    **kwargs,
) -> SegmentationResult:
    """便利包装：一次性分段，返回结果。"""
    seg = SoundBasedSegmenter(**kwargs)
    return seg.segment(video_path, video_id, output_dir, progress_cb=progress_cb)

"""击球事件自动检测模块。

采用 **音频 + 视觉协同** 检测击球点：
    1. **音频检测**: 从视频中提取音频，检测能量突变（击球声）。
    2. **视觉检测**: 追踪手腕速度峰值（加速→减速模式）。
    3. **协同验证**: 只有当音频峰值和视觉峰值在时间上吻合时，才确认为击球。
       如果没有音频，则回退到纯视觉检测。

支持检测视频中的 **多次击球**，每次击球独立标记。
"""

from __future__ import annotations

import os
import wave
import bisect
import tempfile
import subprocess
from dataclasses import dataclass, field
from collections import deque, Counter
from typing import Optional, Tuple, List, Dict, Deque

import numpy as np

from config.keypoints import KEYPOINT_NAMES
from config.framework_config import ImpactDetectionConfig, DEFAULT_CONFIG


# ── 数据结构 ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImpactEvent:
    """检测到的击球事件。"""
    impact_frame_idx: int
    trigger_frame_idx: int
    peak_speed_px_s: float
    peak_speed_sw_s: Optional[float]
    peak_velocity_unit: Tuple[float, float]
    audio_confirmed: bool = False


@dataclass
class SwingEvent:
    """完整的挥拍事件：准备 → 击球 → 随挥。"""
    swing_index: int = 0
    prep_start_frame: Optional[int] = None
    impact_frame: Optional[int] = None
    followthrough_end_frame: Optional[int] = None
    impact_event: Optional[ImpactEvent] = None


# ── 音频击球检测 ────────────────────────────────────────────────────

class AudioOnsetDetector:
    """从视频音频中检测击球声（能量突变）。

    使用轻量级的 RMS 能量差分方法，无需 librosa 等重依赖。
    """

    def __init__(
        self,
        video_path: str,
        fps: float = 30.0,
        sample_rate: int = 22050,
        frame_length: int = 1024,
        hop_length: int = 512,
        threshold_std: float = 3.0,
        min_separation_s: float = 0.08,
        min_time_s: float = 0.35,
    ):
        self.video_path = video_path
        self.fps = max(fps, 1.0)
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold_std = threshold_std
        self.min_separation_s = min_separation_s
        self.min_time_s = min_time_s

        self.onset_frames: List[int] = []
        self.onset_strength: Dict[int, float] = {}
        self.strong_onset_frames: set = set()
        self.available = False

        self._detect()

    def _extract_audio(self) -> Optional[str]:
        """从视频中提取音频到临时 WAV 文件。"""
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = [
            'ffmpeg', '-y', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate), '-ac', '1',
            temp_wav
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                os.remove(temp_wav)
                return None
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            try:
                os.remove(temp_wav)
            except OSError:
                pass
            return None
        return temp_wav

    def _load_wav_mono(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """读取 WAV 文件为单声道浮点数组。"""
        with wave.open(wav_path, "rb") as wf:
            sr = int(wf.getframerate())
            n_ch = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            if sampwidth != 2:
                raise ValueError(f"不支持的 WAV 位深={sampwidth}")
            raw = wf.readframes(wf.getnframes())
        x = np.frombuffer(raw, dtype=np.int16)
        if n_ch > 1:
            x = x.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
        return x.astype(np.float32) / 32768.0, sr

    def _detect(self):
        """执行音频能量突变检测。"""
        wav_path = self._extract_audio()
        if not wav_path:
            return
        try:
            x, sr = self._load_wav_mono(wav_path)
            if x.size < self.frame_length * 3:
                return

            # 高通滤波（一阶差分）强调瞬态
            x = np.diff(x, prepend=x[:1])

            frame_len = self.frame_length
            hop = self.hop_length
            eps = 1e-8
            n_frames = 1 + max(0, (x.size - frame_len) // hop)
            if n_frames <= 3:
                return

            rms = np.empty(n_frames, dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                frame = x[start: start + frame_len]
                rms[i] = float(np.sqrt(np.mean(frame * frame) + eps))

            env = np.log(rms + eps)
            novelty = np.maximum(0.0, np.diff(env, prepend=env[:1]))

            thr = float(np.mean(novelty) + self.threshold_std * np.std(novelty))
            min_sep = max(1, int((self.min_separation_s * sr) / hop))

            onset_audio_frames: List[int] = []
            onset_strengths: List[float] = []
            last = -10_000
            for i in range(1, novelty.size - 1):
                if (i - last) < min_sep:
                    continue
                v = float(novelty[i])
                if v <= thr:
                    continue
                if v >= float(novelty[i - 1]) and v >= float(novelty[i + 1]):
                    onset_audio_frames.append(i)
                    onset_strengths.append(v)
                    last = i

            # 转换为视频帧号
            onset_times_s = [i * hop / sr for i in onset_audio_frames]
            if self.min_time_s > 0:
                keep = [t >= self.min_time_s for t in onset_times_s]
                onset_times_s = [t for t, k in zip(onset_times_s, keep) if k]
                onset_strengths = [v for v, k in zip(onset_strengths, keep) if k]

            strength_by_frame: Dict[int, float] = {}
            for t, v in zip(onset_times_s, onset_strengths):
                fr = int(round(t * self.fps))
                prev = strength_by_frame.get(fr)
                if prev is None or v > prev:
                    strength_by_frame[fr] = v

            self.onset_strength = strength_by_frame
            self.onset_frames = sorted(strength_by_frame.keys())
            self._infer_strong_onsets()
            self.available = bool(self.onset_frames)

        except Exception:
            pass
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def _infer_strong_onsets(self):
        """使用 Otsu 阈值推断强击球声。"""
        self.strong_onset_frames = set()
        if not self.onset_strength:
            return
        strengths = np.array(list(self.onset_strength.values()), dtype=np.float32)
        if strengths.size < 6:
            return

        s_min, s_max = float(np.min(strengths)), float(np.max(strengths))
        if s_max - s_min < 1e-4:
            return

        # Otsu 阈值
        nbins = 64
        hist, bin_edges = np.histogram(strengths, bins=nbins, range=(s_min, s_max))
        hist = hist.astype(np.float64)
        total = float(hist.sum())
        if total <= 0:
            return
        p = hist / total
        omega = np.cumsum(p)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mu = np.cumsum(p * bin_centers)
        mu_t = float(mu[-1])
        denom = np.maximum(omega * (1.0 - omega), 1e-12)
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        k = int(np.nanargmax(sigma_b2))
        thr = float(0.5 * (bin_edges[k] + bin_edges[k + 1]))

        strong = strengths[strengths >= thr]
        weak = strengths[strengths < thr]
        if strong.size < 2 or weak.size < 2:
            return

        strong_mean = float(np.mean(strong))
        weak_mean = float(np.mean(weak))
        if weak_mean > 1e-6 and (strong_mean / weak_mean) >= 1.5:
            self.strong_onset_frames = {
                fr for fr, v in self.onset_strength.items() if v >= thr
            }

    def nearest_onset(self, frame_idx: int, tolerance: int = 3, strong_only: bool = False) -> Optional[int]:
        """查找最近的音频击球帧。"""
        if strong_only and self.strong_onset_frames:
            xs = sorted(self.strong_onset_frames)
        else:
            xs = self.onset_frames
        if not xs:
            return None
        i = bisect.bisect_left(xs, frame_idx)
        cand = []
        if 0 <= i < len(xs):
            cand.append(xs[i])
        if i - 1 >= 0:
            cand.append(xs[i - 1])
        if not cand:
            return None
        best = min(cand, key=lambda x: abs(frame_idx - x))
        return best if abs(frame_idx - best) <= tolerance else None


# ── 视觉击球检测（手腕速度峰值）────────────────────────────────────

class WristSpeedDetector:
    """实时手腕速度峰值检测。

    在帧 t 计算 wrist(t-1) → wrist(t) 的速度。
    当 speed(t-2) < speed(t-1) > speed(t) 时检测到局部极大值。
    """

    def __init__(
        self,
        fps: float = 30.0,
        is_right_handed: bool = True,
        cfg: Optional[ImpactDetectionConfig] = None,
    ):
        c = cfg or DEFAULT_CONFIG.impact_detection
        self.fps = max(fps, 1.0)
        self.is_right_handed = is_right_handed

        self.min_wrist_conf = c.min_wrist_conf
        self.cooldown_frames = c.cooldown_frames
        self.min_peak_speed_sw_s = c.min_peak_speed_sw_s
        self.min_peak_speed_px_s = c.min_peak_speed_px_s
        self.min_peak_speed_px_s_floor = c.min_peak_speed_px_s_floor
        self.peak_over_baseline_ratio = c.peak_over_baseline_ratio
        self.max_frame_gap = c.max_frame_gap

        self.wrist_idx = KEYPOINT_NAMES["right_wrist" if is_right_handed else "left_wrist"]

        self._prev_wrist: Optional[np.ndarray] = None
        self._prev_frame_idx: Optional[int] = None
        self._history_size = 7

        self._speed_px_s: Deque[float] = deque(maxlen=self._history_size)
        self._speed_sw_s: Deque[Optional[float]] = deque(maxlen=self._history_size)
        self._vel_px_s: Deque[np.ndarray] = deque(maxlen=self._history_size)
        self._sw_px_hist: Deque[float] = deque(maxlen=60)
        self._cooldown_left = 0

        self.events: List[ImpactEvent] = []

    def reset(self):
        self._prev_wrist = None
        self._prev_frame_idx = None
        self._speed_px_s.clear()
        self._speed_sw_s.clear()
        self._vel_px_s.clear()
        self._sw_px_hist.clear()
        self._cooldown_left = 0
        self.events.clear()

    def update(
        self,
        frame_idx: int,
        keypoints: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[Optional[ImpactEvent], float]:
        """处理一帧。返回 (event_or_None, 当前手腕速度)。"""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if confidence[self.wrist_idx] < self.min_wrist_conf:
            return None, 0.0

        wrist = np.asarray(keypoints[self.wrist_idx], dtype=np.float64)

        if self._prev_frame_idx is not None:
            gap = int(frame_idx) - int(self._prev_frame_idx)
            if gap <= 0 or gap > self.max_frame_gap:
                self._prev_wrist = wrist.copy()
                self._prev_frame_idx = int(frame_idx)
                self._speed_px_s.clear()
                self._speed_sw_s.clear()
                self._vel_px_s.clear()
                return None, 0.0

        if self._prev_wrist is None:
            self._prev_wrist = wrist.copy()
            self._prev_frame_idx = int(frame_idx)
            return None, 0.0

        prev_idx = int(self._prev_frame_idx) if self._prev_frame_idx is not None else int(frame_idx) - 1
        gap = max(1, int(frame_idx) - prev_idx)
        dt_s = gap / self.fps

        delta = wrist - self._prev_wrist
        speed_px_s = float(np.linalg.norm(delta)) / max(1e-6, dt_s)
        vel_px_s = delta / max(1e-6, dt_s)

        sw_px = self._shoulder_width_estimate(keypoints, confidence)
        speed_sw_s = (speed_px_s / sw_px) if (sw_px is not None and sw_px > 1e-6) else None

        self._speed_px_s.append(speed_px_s)
        self._speed_sw_s.append(speed_sw_s)
        self._vel_px_s.append(vel_px_s.astype(np.float64))

        self._prev_wrist = wrist.copy()
        self._prev_frame_idx = int(frame_idx)

        event = self._maybe_detect_peak(int(frame_idx))
        if event is not None:
            self.events.append(event)
        return event, speed_px_s

    def _shoulder_width_estimate(self, keypoints, confidence) -> Optional[float]:
        ls = KEYPOINT_NAMES["left_shoulder"]
        rs = KEYPOINT_NAMES["right_shoulder"]
        if confidence[ls] < 0.35 or confidence[rs] < 0.35:
            sw = None
        else:
            sw = float(np.linalg.norm(keypoints[rs] - keypoints[ls]))
            if sw < 12.0:
                sw = None
        if sw is not None:
            if len(self._sw_px_hist) >= 5:
                med = float(np.median(list(self._sw_px_hist)))
                if med > 1e-6 and (sw < 0.4 * med or sw > 2.5 * med):
                    sw = None
            if sw is not None:
                self._sw_px_hist.append(sw)
        if not self._sw_px_hist:
            return sw
        return float(np.median(list(self._sw_px_hist)))

    def _maybe_detect_peak(self, trigger_frame_idx: int) -> Optional[ImpactEvent]:
        if self._cooldown_left > 0 or len(self._speed_px_s) < 3:
            return None

        use_norm = all(s is not None for s in list(self._speed_sw_s)[-3:])

        if use_norm:
            speeds = [float(x) if x is not None else float("nan") for x in self._speed_sw_s]
            peak_threshold = self.min_peak_speed_sw_s
        else:
            speeds = list(self._speed_px_s)
            peak_threshold = self.min_peak_speed_px_s

        prev_prev, prev, curr = speeds[-3], speeds[-2], speeds[-1]
        if not (prev > prev_prev and prev > curr):
            return None

        peak_px_s = float(self._speed_px_s[-2])
        if use_norm and peak_px_s < self.min_peak_speed_px_s_floor:
            return None
        if not use_norm and peak_px_s < self.min_peak_speed_px_s:
            return None
        if prev < peak_threshold:
            return None

        baseline_vals = [v for v in speeds[:-2] if np.isfinite(v)]
        baseline = float(np.median(baseline_vals)) if baseline_vals else 0.0
        if baseline > 1e-6 and (prev / baseline) < self.peak_over_baseline_ratio:
            return None

        self._cooldown_left = self.cooldown_frames
        impact_frame_idx = trigger_frame_idx - 1

        peak_sw_s = float(self._speed_sw_s[-2]) if self._speed_sw_s[-2] is not None else None
        peak_vel = self._vel_px_s[-2] if len(self._vel_px_s) >= 2 else np.zeros(2)
        norm = float(np.linalg.norm(peak_vel))
        unit = (float(peak_vel[0] / norm), float(peak_vel[1] / norm)) if norm > 1e-6 else (0.0, 0.0)

        return ImpactEvent(
            impact_frame_idx=impact_frame_idx,
            trigger_frame_idx=trigger_frame_idx,
            peak_speed_px_s=peak_px_s,
            peak_speed_sw_s=peak_sw_s,
            peak_velocity_unit=unit,
            audio_confirmed=False,
        )


# ── 协同击球检测器 ──────────────────────────────────────────────────

class HybridImpactDetector:
    """音频 + 视觉协同击球检测器。

    流程：
    1. 先用 AudioOnsetDetector 分析整段音频
    2. 逐帧用 WristSpeedDetector 检测手腕速度峰值
    3. 对每个视觉峰值，检查是否有对应的音频击球声
    4. 合并、去重，输出最终的击球帧列表
    """

    def __init__(
        self,
        video_path: str,
        fps: float = 30.0,
        is_right_handed: bool = True,
        cfg: Optional[ImpactDetectionConfig] = None,
        audio_tolerance_frames: int = 3,
        merge_within_s: float = 0.8,
        min_verified_peak_speed_px_s: float = 400.0,
    ):
        self.fps = max(fps, 1.0)
        self.is_right_handed = is_right_handed
        self.audio_tolerance = audio_tolerance_frames
        self.merge_within_s = merge_within_s
        self.min_verified_peak_speed_px_s = min_verified_peak_speed_px_s

        # 音频检测
        self.audio_detector = AudioOnsetDetector(video_path, fps=fps)

        # 视觉检测
        self.wrist_detector = WristSpeedDetector(
            fps=fps, is_right_handed=is_right_handed, cfg=cfg,
        )

        # 最终结果
        self._verified_events: Dict[int, ImpactEvent] = {}

    def update(
        self,
        frame_idx: int,
        keypoints: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[Optional[ImpactEvent], float]:
        """逐帧更新。返回 (event_or_None, 当前手腕速度)。"""
        event, speed = self.wrist_detector.update(frame_idx, keypoints, confidence)
        if event is not None:
            verified = self._verify_with_audio(event)
            if verified:
                self._verified_events[event.impact_frame_idx] = verified
                return verified, speed
        return None, speed

    def _verify_with_audio(self, event: ImpactEvent) -> Optional[ImpactEvent]:
        """用音频验证视觉检测到的击球。"""
        if not self.audio_detector.available:
            # 没有音频，回退到纯视觉
            if event.peak_speed_px_s >= self.min_verified_peak_speed_px_s:
                return event
            return None

        # 检查是否有对应的音频击球声
        strong_only = bool(self.audio_detector.strong_onset_frames)
        nearest = self.audio_detector.nearest_onset(
            event.impact_frame_idx,
            tolerance=self.audio_tolerance,
            strong_only=strong_only,
        )

        if nearest is not None:
            # 音频确认
            if event.peak_speed_px_s >= self.min_verified_peak_speed_px_s * 0.7:
                return ImpactEvent(
                    impact_frame_idx=event.impact_frame_idx,
                    trigger_frame_idx=event.trigger_frame_idx,
                    peak_speed_px_s=event.peak_speed_px_s,
                    peak_speed_sw_s=event.peak_speed_sw_s,
                    peak_velocity_unit=event.peak_velocity_unit,
                    audio_confirmed=True,
                )
        else:
            # 没有音频确认，但速度非常高，仍然接受
            if event.peak_speed_px_s >= self.min_verified_peak_speed_px_s * 1.5:
                return event

        return None

    def finalize(self) -> List[ImpactEvent]:
        """完成检测，合并近距离的击球，返回最终列表。"""
        self._merge_close_impacts()
        return [self._verified_events[k] for k in sorted(self._verified_events.keys())]

    def _merge_close_impacts(self):
        """合并时间上过于接近的击球（同一次挥拍的多个峰值）。"""
        if not self._verified_events:
            return
        merge_frames = int(round(self.merge_within_s * self.fps))
        events = sorted(self._verified_events.values(), key=lambda e: e.impact_frame_idx)

        clusters: List[List[ImpactEvent]] = []
        cur = [events[0]]
        cluster_start = events[0].impact_frame_idx
        for ev in events[1:]:
            if ev.impact_frame_idx - cluster_start <= merge_frames:
                cur.append(ev)
            else:
                clusters.append(cur)
                cur = [ev]
                cluster_start = ev.impact_frame_idx
        clusters.append(cur)

        merged: Dict[int, ImpactEvent] = {}
        for cluster in clusters:
            # 优先选择音频确认的，然后选速度最高的
            audio_confirmed = [e for e in cluster if e.audio_confirmed]
            if audio_confirmed:
                best = max(audio_confirmed, key=lambda e: e.peak_speed_px_s)
            else:
                best = max(cluster, key=lambda e: e.peak_speed_px_s)
            merged[best.impact_frame_idx] = best

        self._verified_events = merged

    def get_events(self) -> List[ImpactEvent]:
        """获取当前已验证的击球事件列表。"""
        return [self._verified_events[k] for k in sorted(self._verified_events.keys())]


# ── 挥拍阶段估计 ────────────────────────────────────────────────────

class SwingPhaseEstimator:
    """估计每次击球的准备开始帧和随挥结束帧。"""

    def __init__(self, fps: float = 30.0):
        self.fps = max(fps, 1.0)

    def estimate_phases(
        self,
        impact_frame: int,
        wrist_speeds: np.ndarray,
        frame_indices: List[int],
        prev_impact_frame: Optional[int] = None,
        next_impact_frame: Optional[int] = None,
    ) -> SwingEvent:
        """给定击球帧和手腕速度序列，估计挥拍阶段。

        prev_impact_frame / next_impact_frame 用于限制阶段边界，
        避免与相邻击球重叠。
        """
        event = SwingEvent(impact_frame=impact_frame)

        if len(wrist_speeds) == 0 or len(frame_indices) == 0:
            return event

        f2p = {f: i for i, f in enumerate(frame_indices)}
        impact_pos = f2p.get(impact_frame)
        if impact_pos is None:
            return event

        # ── 准备开始：从击球帧向前回溯，直到速度降至峰值的 20%
        peak_speed = float(wrist_speeds[impact_pos]) if impact_pos < len(wrist_speeds) else 0.0
        threshold = peak_speed * 0.20
        prep_pos = impact_pos
        for i in range(impact_pos - 1, -1, -1):
            if i < len(wrist_speeds) and wrist_speeds[i] < threshold:
                prep_pos = i
                break
        prep_pos = max(0, prep_pos - int(0.3 * self.fps))

        # 不超过上一次击球的随挥结束
        if prev_impact_frame is not None:
            prev_pos = f2p.get(prev_impact_frame, 0)
            min_prep = prev_pos + int(0.2 * self.fps)
            prep_pos = max(prep_pos, min_prep)

        event.prep_start_frame = frame_indices[min(prep_pos, len(frame_indices) - 1)]

        # ── 随挥结束：从击球帧向后，直到速度降至峰值的 25%
        ft_threshold = peak_speed * 0.25
        ft_pos = min(impact_pos + 1, len(wrist_speeds) - 1)
        for i in range(impact_pos + 1, len(wrist_speeds)):
            if wrist_speeds[i] < ft_threshold:
                ft_pos = i
                break
        ft_pos = min(len(frame_indices) - 1, ft_pos + int(0.2 * self.fps))

        # 不超过下一次击球的准备开始
        if next_impact_frame is not None:
            next_pos = f2p.get(next_impact_frame, len(frame_indices) - 1)
            max_ft = next_pos - int(0.1 * self.fps)
            ft_pos = min(ft_pos, max_ft)

        event.followthrough_end_frame = frame_indices[min(ft_pos, len(frame_indices) - 1)]

        return event

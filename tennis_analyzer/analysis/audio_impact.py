#!/usr/bin/env python3
"""
Hybrid Impact Detector
======================
Combines AUDIO onset detection with WRIST SPEED PEAKS to robustly detect impacts.

A true impact requires BOTH:
1. Audio onset (sharp sound)
2. Wrist speed peak (high speed followed by deceleration)

This filters out:
- Ball bounces (sound but no wrist movement)
- Non-impact swings (movement but no sound)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Dict
import subprocess
import tempfile
import os
import wave
import bisect

from .impact import WristSpeedImpactDetector, ImpactEvent
from ..core.smoother import KeypointSmoother


@dataclass(frozen=True)
class AudioOnsetConfig:
    """Configuration for simple audio-onset detection."""

    sample_rate: int = 22050
    frame_length: int = 1024
    hop_length: int = 512
    # Peak picking
    threshold_std: float = 3.0
    min_separation_s: float = 0.08  # ~2-3 video frames at 30fps
    # Optional: ignore onsets before this time to avoid false positives from
    # camera handling noise at the beginning of a clip.
    min_time_s: float = 0.35


class HybridImpactDetector:
    """Audio onset detection + impact verification helper."""

    def __init__(self, video_path: str, fps: float = 30.0):
        self.video_path = video_path
        self.fps = float(fps) if fps and fps > 0 else 30.0
        
        # Detection results
        self.audio_onset_frames: List[int] = []
        # Map: onset_frame_idx -> onset_strength (novelty peak value).
        self.audio_onset_strength: Dict[int, float] = {}
        # Optional: inferred "strong" onsets (likely racket hits). Enabled only
        # when audio onset strengths appear clearly bimodal.
        self.strong_audio_onset_frames: set[int] = set()
        self.audio_strength_threshold: Optional[float] = None
        self.verified_impact_frames: List[int] = []

        self._audio_cfg = AudioOnsetConfig()
        self._analyze_audio_energy()
    
    def _extract_audio(self) -> Optional[str]:
        """Extract audio from video to temp WAV file."""
        # NOTE: We use mkstemp + close fd to avoid issues where ffmpeg can't overwrite
        # an already-open temp file (platform dependent).
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        cmd = [
            'ffmpeg', '-y', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1',
            temp_wav
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                return None
        except FileNotFoundError:
            # ffmpeg not installed / not in PATH
            return None
        return temp_wav
    
    def _analyze_audio_energy(self) -> None:
        """Detect sharp audio onsets using a lightweight energy-based method.

        This intentionally avoids heavy dependencies (e.g. librosa) and works with
        the default project requirements.
        """
        wav_path = self._extract_audio()
        if not wav_path:
            return

        try:
            x, sr = self._load_wav_mono(wav_path)
            if x.size < self._audio_cfg.frame_length * 3:
                return

            # Emphasize transients (simple high-pass via first difference).
            x = np.diff(x, prepend=x[:1])

            frame_len = self._audio_cfg.frame_length
            hop = self._audio_cfg.hop_length
            eps = 1e-8
            n_frames = 1 + max(0, (x.size - frame_len) // hop)
            if n_frames <= 3:
                return

            rms = np.empty(n_frames, dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                frame = x[start : start + frame_len]
                rms[i] = float(np.sqrt(np.mean(frame * frame) + eps))

            env = np.log(rms + eps)
            novelty = np.maximum(0.0, np.diff(env, prepend=env[:1]))

            thr = float(np.mean(novelty) + self._audio_cfg.threshold_std * np.std(novelty))
            min_sep = int((self._audio_cfg.min_separation_s * sr) / hop)
            if min_sep < 1:
                min_sep = 1

            onset_audio_frames: List[int] = []
            onset_audio_strength: List[float] = []
            last = -10_000
            for i in range(1, novelty.size - 1):
                if (i - last) < min_sep:
                    continue
                v = float(novelty[i])
                if v <= thr:
                    continue
                if v >= float(novelty[i - 1]) and v >= float(novelty[i + 1]):
                    onset_audio_frames.append(i)
                    onset_audio_strength.append(v)
                    last = i

            # Convert to video frames (0-based).
            onset_times_s = [i * hop / sr for i in onset_audio_frames]
            if self._audio_cfg.min_time_s and self._audio_cfg.min_time_s > 0:
                keep = [t >= float(self._audio_cfg.min_time_s) for t in onset_times_s]
                onset_times_s = [t for t, k in zip(onset_times_s, keep) if k]
                onset_audio_strength = [v for v, k in zip(onset_audio_strength, keep) if k]

            # Round to video frames and merge duplicates by max strength.
            strength_by_frame: Dict[int, float] = {}
            for t, v in zip(onset_times_s, onset_audio_strength):
                fr = int(round(t * self.fps))
                prev = strength_by_frame.get(fr)
                if prev is None or float(v) > float(prev):
                    strength_by_frame[fr] = float(v)

            self.audio_onset_strength = dict(strength_by_frame)
            # Keep a sorted unique list for fast nearest-frame queries.
            self.audio_onset_frames = sorted(strength_by_frame.keys())
            self._infer_strong_onsets()

            if self.audio_onset_frames:
                print(f"Audio onsets (energy) at frames: {self.audio_onset_frames}")
                if self.strong_audio_onset_frames and self.audio_strength_threshold is not None:
                    strong = sorted(self.strong_audio_onset_frames)
                    print(
                        f"  Strong onsets (>= {self.audio_strength_threshold:.3f}) at frames: {strong}"
                    )
        except Exception as e:
            print(f"Audio analysis failed: {e}")
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

    def _infer_strong_onsets(self) -> None:
        """Infer strong (likely racket hits) audio onsets from onset strengths.

        We only enable the strong/weak split when the strength distribution looks
        clearly bimodal. This avoids dropping legitimate hits in uniformly noisy
        audio environments.
        """
        self.strong_audio_onset_frames = set()
        self.audio_strength_threshold = None

        if not self.audio_onset_strength:
            return

        strengths = np.asarray(list(self.audio_onset_strength.values()), dtype=np.float32)
        if strengths.size < 6:
            return

        s_min = float(np.min(strengths))
        s_max = float(np.max(strengths))
        if not np.isfinite(s_min) or not np.isfinite(s_max) or (s_max - s_min) < 1e-4:
            return

        # Otsu threshold on 1D strengths.
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
        if weak_mean <= 1e-6:
            return

        def _enable(threshold: float) -> None:
            self.audio_strength_threshold = float(threshold)
            self.strong_audio_onset_frames = {
                int(fr) for fr, v in self.audio_onset_strength.items() if float(v) >= float(threshold)
            }

        # Enable only when separation is strong and overlap is limited.
        if (strong_mean / weak_mean) >= 1.7 and (
            float(np.percentile(strong, 10)) > float(np.percentile(weak, 90)) * 1.15
        ):
            _enable(thr)
            return

        # Fallback: if Otsu didn't yield a clean split, still try a conservative
        # "top onsets" threshold. This helps avoid verifying pose peaks against
        # weak bounces/footsteps (common in back-view practice clips).
        p = 75.0
        thr_p = float(np.percentile(strengths, p))
        strong_p = strengths[strengths >= thr_p]
        weak_p = strengths[strengths < thr_p]
        if strong_p.size >= 2 and weak_p.size >= 2:
            strong_mean = float(np.mean(strong_p))
            weak_mean = float(np.mean(weak_p))
            if weak_mean > 1e-6 and (strong_mean / weak_mean) >= 1.5:
                _enable(thr_p)
                return

    def _load_wav_mono(self, wav_path: str) -> tuple[np.ndarray, int]:
        with wave.open(wav_path, "rb") as wf:
            sr = int(wf.getframerate())
            n_ch = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            if sampwidth != 2:
                raise ValueError(f"Unsupported WAV sampwidth={sampwidth} (expected 16-bit)")
            raw = wf.readframes(wf.getnframes())

        x = np.frombuffer(raw, dtype=np.int16)
        if n_ch > 1:
            x = x.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
        x_f = (x.astype(np.float32) / 32768.0).copy()
        return x_f, sr

    def nearest_audio_onset(
        self,
        frame_idx: int,
        tolerance: int = 3,
        *,
        strong_only: bool = False,
    ) -> Optional[int]:
        """Return the nearest audio onset frame within tolerance (in video frames)."""
        xs_all = self.audio_onset_frames
        if not xs_all:
            return None
        strong_frames = getattr(self, "strong_audio_onset_frames", set())
        if strong_only and strong_frames:
            xs = sorted(strong_frames)
        else:
            xs = xs_all
        # Binary search to find insertion point.
        i = bisect.bisect_left(xs, int(frame_idx))
        cand = []
        if 0 <= i < len(xs):
            cand.append(xs[i])
        if i - 1 >= 0:
            cand.append(xs[i - 1])
        if not cand:
            return None
        best = min(cand, key=lambda x: abs(int(frame_idx) - int(x)))
        return best if abs(int(frame_idx) - int(best)) <= int(tolerance) else None
    
    def is_near_audio_onset(
        self, frame_idx: int, tolerance: int = 3, *, strong_only: bool = False
    ) -> bool:
        """Check if frame is near an audio onset."""
        return (
            self.nearest_audio_onset(frame_idx, tolerance=tolerance, strong_only=strong_only)
            is not None
        )

    def add_verified_impact(self, frame_idx: int) -> None:
        if frame_idx not in self.verified_impact_frames:
            self.verified_impact_frames.append(frame_idx)
    
    def is_impact_frame(self, frame_idx: int) -> bool:
        """Check if frame is a verified impact."""
        for impact_frame in self.verified_impact_frames:
            if abs(frame_idx - impact_frame) <= 1:
                return True
        return False
    
    def get_verified_impacts(self) -> List[int]:
        """Get list of verified impact frames."""
        return sorted(self.verified_impact_frames)
    
    def reset(self):
        """Reset verified impacts (audio onsets are kept)."""
        self.verified_impact_frames.clear()


class TwoPassImpactDetector:
    """
    Two-pass impact detection:
    1. First pass: Analyze entire video to find verified impacts (audio + velocity)
    2. Second pass: Use verified impacts for Big 3 calculation and display
    """
    
    def __init__(
        self,
        video_path: str,
        fps: float,
        pose_estimator,
        is_right_handed: bool = True,
        *,
        require_audio: bool = True,
        audio_tolerance_frames: int = 3,
        pose_confidence: float = 0.5,
        rotation: int = 0,
        # If we detect multiple verified impacts close in time, we treat them as the
        # same stroke and keep the "best" one (usually the true racket-ball contact)
        # to avoid reporting bounce + contact as two separate impacts.
        merge_within_s: float = 0.8,
        # Wrist peak detector settings.
        # For side-view footage it's common to have multiple speed peaks per stroke
        # (prep + bounce timing + true contact). We keep cooldown short and rely on
        # `_merge_close_impacts()` to pick the best one.
        wrist_peak_cooldown_frames: int = 4,
        wrist_peak_min_wrist_conf: float = 0.35,
        use_both_wrists: bool = True,
        # Light pose smoothing during pass-1 reduces jitter-induced false peaks.
        pose_smoothing_factor: float = 0.4,
        # Hard safety floor in px/s for accepting a verified (audio-aligned) impact.
        # Rationale: for side-view footage shoulder-width normalization can inflate
        # `sw/s` (foreshortening), making tiny wrist twitches look like "fast" peaks.
        # A modest absolute floor filters bounces/idle frames while keeping true hits.
        min_verified_peak_speed_px_s: float = 450.0,
        # Optional: allow pose-only impacts even when audio is present but no onset
        # is found near the pose peak. Disabled by default because it tends to
        # introduce false positives (practice swings, pre-bounce adjustments).
        pose_fallback_min_peak_speed_px_s: Optional[float] = None,
        # When audio is available, optionally "snap" the scheduled impact frame
        # to the audio onset (plus a tiny offset) for better visual alignment.
        snap_to_audio_onset: bool = True,
        audio_snap_offset_frames: int = 1,
    ):
        self.video_path = video_path
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.pose_estimator = pose_estimator
        self.is_right_handed = is_right_handed
        self.require_audio = bool(require_audio)
        self.audio_tolerance_frames = int(audio_tolerance_frames)
        self.pose_confidence = float(pose_confidence)
        self.rotation = int(rotation) if rotation is not None else 0
        self.merge_within_s = float(merge_within_s)
        self.wrist_peak_cooldown_frames = int(wrist_peak_cooldown_frames)
        self.wrist_peak_min_wrist_conf = float(wrist_peak_min_wrist_conf)
        self.use_both_wrists = bool(use_both_wrists)
        self.pose_smoothing_factor = float(pose_smoothing_factor)
        self.min_verified_peak_speed_px_s = float(min_verified_peak_speed_px_s)
        self.pose_fallback_min_peak_speed_px_s = (
            float(pose_fallback_min_peak_speed_px_s)
            if pose_fallback_min_peak_speed_px_s is not None
            else None
        )
        self.snap_to_audio_onset = bool(snap_to_audio_onset)
        self.audio_snap_offset_frames = int(audio_snap_offset_frames)

        # Audio verifier
        self.detector = HybridImpactDetector(video_path, self.fps)

        # Wrist speed peak detectors (primary + optional fallback wrist).
        # Note: `WristSpeedImpactDetector.is_right_handed` means which wrist to track.
        self._wrist_peaks: List[WristSpeedImpactDetector] = [
            WristSpeedImpactDetector(
                fps=self.fps,
                is_right_handed=bool(is_right_handed),
                cooldown_frames=self.wrist_peak_cooldown_frames,
                min_wrist_conf=self.wrist_peak_min_wrist_conf,
            )
        ]
        if self.use_both_wrists:
            self._wrist_peaks.append(
                WristSpeedImpactDetector(
                    fps=self.fps,
                    is_right_handed=(not bool(is_right_handed)),
                    cooldown_frames=self.wrist_peak_cooldown_frames,
                    min_wrist_conf=self.wrist_peak_min_wrist_conf,
                )
            )

        # Verified impacts as ImpactEvents (indexed by impact_frame_idx).
        self._impact_events_by_frame: Dict[int, ImpactEvent] = {}
        
        # Run first pass to find all impacts
        self._first_pass()

    def _snap_impacts_to_audio_onsets(self) -> None:
        """Replace pose-peak impact indices with (audio_onset + offset) indices.

        This is only used when audio is available. We keep the original peak
        speeds, but schedule/preview frames based on the audio onset, which tends
        to be closer to the visually-correct contact frame.
        """
        if not self.snap_to_audio_onset:
            return
        if not self.require_audio or not self.detector.audio_onset_frames:
            return
        if not self._impact_events_by_frame:
            return

        # If we inferred strong onsets, we still only snap when we have high
        # confidence that the onset corresponds to racket-ball contact.
        strong_frames = set(getattr(self.detector, "strong_audio_onset_frames", set()) or set())
        strong_only = bool(strong_frames)
        strength_by_frame = getattr(self.detector, "audio_onset_strength", {}) or {}

        # Only snap when pose peak is already very close to the onset, OR the
        # onset is classified as "strong". This avoids shifting impact thumbnails
        # to bounce/footstep onsets.
        tight_frames = 1

        snapped: Dict[int, ImpactEvent] = {}
        for ev in self._impact_events_by_frame.values():
            onset = self.detector.nearest_audio_onset(
                int(ev.impact_frame_idx),
                tolerance=self.audio_tolerance_frames,
                strong_only=strong_only,
            )
            new_idx = int(ev.impact_frame_idx)
            if onset is not None:
                dist = abs(int(ev.impact_frame_idx) - int(onset))
                _strength = float(strength_by_frame.get(int(onset), 0.0))
                should_snap = (dist <= int(tight_frames)) or (int(onset) in strong_frames)
                if should_snap:
                    new_idx = int(onset) + int(self.audio_snap_offset_frames)
            new_ev = ImpactEvent(
                impact_frame_idx=int(new_idx),
                trigger_frame_idx=int(new_idx) + 1,
                peak_speed_px_s=float(ev.peak_speed_px_s),
                peak_speed_sw_s=float(ev.peak_speed_sw_s) if ev.peak_speed_sw_s is not None else None,
                peak_velocity_unit=tuple(ev.peak_velocity_unit),
            )
            prev = snapped.get(int(new_idx))
            if prev is None or float(new_ev.peak_speed_px_s) > float(prev.peak_speed_px_s):
                snapped[int(new_idx)] = new_ev

        self._impact_events_by_frame = snapped

    def _infer_bounce_frames(self) -> None:
        """Infer a best-effort bounce onset frame for each scheduled impact.

        This is used by downstream Big3 logic (e.g., "prep before bounce").

        Notes:
        - Only available when audio onsets exist (hybrid mode).
        - Bounce onset is heuristic: we pick the latest onset BEFORE the hit
          onset within a short time window.
        """
        if not self.require_audio or not self.detector.audio_onset_frames:
            return
        if not self._impact_events_by_frame:
            return

        onsets = sorted(set(int(x) for x in self.detector.audio_onset_frames))
        if not onsets:
            return

        # Typical bounce->hit is within ~0.2-0.9s; keep a conservative window.
        max_gap_frames = int(round(1.2 * self.fps))
        min_gap_frames = max(1, int(round(0.08 * self.fps)))  # avoid same-onset duplicates

        updated: Dict[int, ImpactEvent] = {}
        for k, ev in self._impact_events_by_frame.items():
            hit_onset = self.detector.nearest_audio_onset(
                int(ev.impact_frame_idx),
                tolerance=self.audio_tolerance_frames,
                strong_only=False,  # bounce is often weaker than hit
            )
            if hit_onset is None:
                updated[int(k)] = ev
                continue

            i = bisect.bisect_left(onsets, int(hit_onset))
            if i <= 0:
                updated[int(k)] = ev
                continue

            bounce = int(onsets[i - 1])
            gap = int(hit_onset) - int(bounce)
            if gap < min_gap_frames or gap > max_gap_frames:
                updated[int(k)] = ev
                continue

            updated[int(k)] = ImpactEvent(
                impact_frame_idx=int(ev.impact_frame_idx),
                trigger_frame_idx=int(ev.trigger_frame_idx),
                peak_speed_px_s=float(ev.peak_speed_px_s),
                peak_speed_sw_s=float(ev.peak_speed_sw_s) if ev.peak_speed_sw_s is not None else None,
                peak_velocity_unit=tuple(ev.peak_velocity_unit),
                bounce_frame_idx=int(bounce),
            )

        self._impact_events_by_frame = updated

    def _merge_close_impacts(self) -> None:
        """Merge multiple verified impacts that happen close in time.

        Why:
        - In real drills, audio has multiple onsets per ball (bounce, hit, net, etc.)
        - Pose wrist-speed can have multiple peaks during one stroke (prep + hit)
        We keep only the strongest/most plausible impact within a short window.
        """
        if not self._impact_events_by_frame:
            return

        merge_within_frames = int(round(max(0.0, self.merge_within_s) * self.fps))
        if merge_within_frames <= 0:
            return

        events = sorted(
            self._impact_events_by_frame.values(), key=lambda e: int(e.impact_frame_idx)
        )

        # ------------------------------------------------------------------
        # Prefer audio-cluster merging when audio exists and is required.
        #
        # Rationale:
        # - A single stroke often produces 2 audio onsets (bounce + contact).
        # - Pose peaks may occur slightly before/after the onset and can create
        #   long chains of peaks. Clustering by *audio onsets* is more stable
        #   than clustering by pose peak timestamps.
        # ------------------------------------------------------------------
        if self.detector.audio_onset_frames and self.require_audio:
            # If strong-onset filtering is enabled, merge/select using those to
            # avoid "bounce-only" onsets generating impacts.
            strong_frames = getattr(self.detector, "strong_audio_onset_frames", set())
            if strong_frames:
                onsets = sorted(set(int(x) for x in strong_frames))
            else:
                onsets = sorted(set(int(x) for x in self.detector.audio_onset_frames))
            onset_clusters: List[List[int]] = []
            cur_on: List[int] = [onsets[0]]
            for o in onsets[1:]:
                if int(o) - int(cur_on[-1]) <= merge_within_frames:
                    cur_on.append(int(o))
                else:
                    onset_clusters.append(cur_on)
                    cur_on = [int(o)]
            onset_clusters.append(cur_on)

            # Group pose events by their nearest onset (within tolerance).
            events_by_onset: Dict[int, List[ImpactEvent]] = {}
            for ev in events:
                onset = self.detector.nearest_audio_onset(
                    int(ev.impact_frame_idx), tolerance=self.audio_tolerance_frames
                )
                if onset is None:
                    continue
                events_by_onset.setdefault(int(onset), []).append(ev)

            selected: Dict[int, ImpactEvent] = {}
            for cluster in onset_clusters:
                cluster_events: List[ImpactEvent] = []
                for o in cluster:
                    cluster_events.extend(events_by_onset.get(int(o), []))
                if not cluster_events:
                    continue

                # ------------------------------------------------------------------
                # Side-view fix: audio onset clusters often contain BOTH bounce+hit.
                # We want the *contact* onset, which is usually the later one, and
                # we want the pose peak to be *tightly* aligned to that onset.
                #
                # Previous logic over-weighted peak speed, which can pick a far-off
                # pose peak (often from occlusion/jitter) and yield a wrong thumbnail.
                #
                # Strategy:
                # 1) Prefer the latest onset in the cluster that has any pose peak
                #    within a tight window (<= tight_frames).
                # 2) If none exist, fall back to whichever onset has the closest
                #    pose peak within the full tolerance.
                # 3) Within an onset, pick the closest peak, then fastest.
                # ------------------------------------------------------------------
                tight_frames = int(min(3, max(0, self.audio_tolerance_frames)))
                if tight_frames <= 0:
                    tight_frames = 1

                def _best_for_onset(
                    onset_frame: int, *, max_dist: int
                ) -> Optional[ImpactEvent]:
                    if max_dist <= 0:
                        return None
                    cands = [
                        ev
                        for ev in cluster_events
                        if abs(int(ev.impact_frame_idx) - int(onset_frame)) <= int(max_dist)
                    ]
                    if not cands:
                        return None
                    return min(
                        cands,
                        key=lambda ev: (
                            abs(int(ev.impact_frame_idx) - int(onset_frame)),
                            -float(ev.peak_speed_px_s),
                            -int(ev.impact_frame_idx),
                        ),
                    )

                chosen_onset: Optional[int] = None
                chosen_best: Optional[ImpactEvent] = None

                # 1) Prefer a tightly-aligned pose peak; choose the strongest onset
                # (racket hits tend to be louder than bounces/footsteps).
                tight_candidates: List[tuple[float, int, ImpactEvent]] = []
                for o in sorted(cluster, reverse=True):
                    cand = _best_for_onset(int(o), max_dist=tight_frames)
                    if cand is None:
                        continue
                    strength_map = getattr(self.detector, "audio_onset_strength", {}) or {}
                    strength = float(strength_map.get(int(o), 0.0))
                    tight_candidates.append((strength, int(o), cand))
                if tight_candidates:
                    strength, o, cand = max(tight_candidates, key=lambda t: (t[0], t[1]))
                    chosen_onset = int(o)
                    chosen_best = cand

                # 2) Fallback: pick onset with closest pose peak in the full window.
                if chosen_best is None:
                    best_tuple: Optional[tuple[int, float, int, float, int]] = None
                    for o in sorted(cluster, reverse=True):
                        cand = _best_for_onset(int(o), max_dist=self.audio_tolerance_frames)
                        if cand is None:
                            continue
                        dist = abs(int(cand.impact_frame_idx) - int(o))
                        strength_map = getattr(self.detector, "audio_onset_strength", {}) or {}
                        strength = float(strength_map.get(int(o), 0.0))
                        # Prefer smaller dist; then stronger onset; then later onset; then faster.
                        tup = (
                            int(dist),
                            -float(strength),
                            -int(o),
                            -float(cand.peak_speed_px_s),
                            -int(cand.impact_frame_idx),
                        )
                        if best_tuple is None or tup < best_tuple:
                            best_tuple = tup
                            chosen_onset = int(o)
                            chosen_best = cand

                if chosen_best is None:
                    continue

                best = chosen_best
                selected[int(best.impact_frame_idx)] = best

            self._impact_events_by_frame = selected
            return

        clusters: List[List[ImpactEvent]] = []
        cur: List[ImpactEvent] = [events[0]]
        cluster_start = int(events[0].impact_frame_idx)
        for ev in events[1:]:
            # IMPORTANT: cluster window is measured from the *first* event in the
            # cluster (not the previous one). This prevents "chain merging" where
            # many small peaks could bridge across multiple strokes.
            if int(ev.impact_frame_idx) - int(cluster_start) <= merge_within_frames:
                cur.append(ev)
            else:
                clusters.append(cur)
                cur = [ev]
                cluster_start = int(ev.impact_frame_idx)
        clusters.append(cur)

        selected: Dict[int, ImpactEvent] = {}
        for cluster in clusters:
            if len(cluster) == 1:
                best = cluster[0]
            else:
                # Score: prefer higher wrist-speed peak, closer audio alignment, and
                # (as a final tie-breaker) later frame (bounce tends to be earlier).
                def score(ev: ImpactEvent) -> tuple[float, int, int]:
                    onset = self.detector.nearest_audio_onset(
                        int(ev.impact_frame_idx), tolerance=self.audio_tolerance_frames
                    )
                    dist = abs(int(ev.impact_frame_idx) - int(onset)) if onset is not None else 999
                    return (float(ev.peak_speed_px_s), -int(dist), int(ev.impact_frame_idx))

                best = max(cluster, key=score)

            selected[int(best.impact_frame_idx)] = best

        self._impact_events_by_frame = selected
    
    def _first_pass(self):
        """Analyze entire video to find verified impact frames."""
        import cv2
        from tennis_analyzer.core.video_processor import rotate_frame
        
        print("First pass: Finding verified impacts...")
        
        # Smooth keypoints to reduce jitter-induced false peaks.
        smoother = KeypointSmoother(smoothing_factor=self.pose_smoothing_factor)

        video = cv2.VideoCapture(self.video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if self.rotation:
                frame = rotate_frame(frame, self.rotation)

            # Pose
            results = self.pose_estimator.predict(frame, conf=self.pose_confidence)
            
            if results["persons"]:
                person = results["persons"][0]
                keypoints = np.asarray(person["keypoints"], dtype=np.float32)
                confidence = np.asarray(person["confidence"], dtype=np.float32)

                keypoints_s = smoother.smooth(keypoints, confidence, min_confidence=0.3)

                for wrist_peak in self._wrist_peaks:
                    event, _speed = wrist_peak.update(frame_idx, keypoints_s, confidence)
                    if event is None:
                        continue

                    pose_impact_idx = int(event.impact_frame_idx)

                    # Verify this *pose* impact using audio onsets when available.
                    verified = True
                    nearest = None
                    if self.detector.audio_onset_frames:
                        strong_only = bool(getattr(self.detector, "strong_audio_onset_frames", set()))
                        nearest = self.detector.nearest_audio_onset(
                            pose_impact_idx,
                            tolerance=self.audio_tolerance_frames,
                            strong_only=strong_only,
                        )
                        if nearest is None and self.require_audio:
                            verified = False
                    else:
                        # Audio missing/unavailable -> fall back to pose-only impacts.
                        verified = True

                    # Optional strong-pose fallback even when audio is present but no onset
                    # is detected near the pose peak (disabled by default).
                    if (
                        (not verified)
                        and self.require_audio
                        and self.pose_fallback_min_peak_speed_px_s is not None
                    ):
                        if float(event.peak_speed_px_s) >= float(self.pose_fallback_min_peak_speed_px_s):
                            verified = True

                    if verified:
                        # Extra guard for side-view: reject low absolute-speed peaks.
                        if float(event.peak_speed_px_s) < float(self.min_verified_peak_speed_px_s):
                            continue
                        # IMPORTANT: Keep the pose peak frame as the impact frame.
                        # Audio is used for verification only (not frame replacement),
                        # otherwise a bounce/footstep onset can shift impact earlier.
                        prev = self._impact_events_by_frame.get(pose_impact_idx)
                        if prev is None or float(event.peak_speed_px_s) > float(prev.peak_speed_px_s):
                            self._impact_events_by_frame[pose_impact_idx] = event
            else:
                # Avoid smoothing across large gaps (pose dropouts).
                smoother.reset()

            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"   Scanning: {frame_idx}/{frame_count}")
        
        video.release()
        # Merge bounce+hit duplicates into a single "best" impact.
        self._merge_close_impacts()
        # If audio is available, optionally snap scheduled impact frames to the
        # onset (plus offset) for better contact thumbnails.
        self._snap_impacts_to_audio_onsets()
        # Best-effort bounce inference (used by "prep before bounce" heuristics).
        self._infer_bounce_frames()

        verified = self.get_impact_frames()
        print(f"Found {len(verified)} verified impacts at frames: {verified}")

    def get_impact_frames(self) -> List[int]:
        """Get list of verified impact frames."""
        return sorted(self._impact_events_by_frame.keys())

    def get_impact_events(self) -> List[ImpactEvent]:
        """Get verified impact events (sorted by impact_frame_idx)."""
        return [self._impact_events_by_frame[k] for k in self.get_impact_frames()]

    def get_impact_event_map(self) -> Dict[int, ImpactEvent]:
        """Get mapping: impact_frame_idx -> ImpactEvent."""
        return dict(self._impact_events_by_frame)
    
    def is_impact_frame(self, frame_num: int) -> bool:
        """Check if this is an impact frame."""
        # Tolerate +/-1 for visualization flash.
        for idx in self.get_impact_frames():
            if abs(int(frame_num) - int(idx)) <= 1:
                return True
        return False

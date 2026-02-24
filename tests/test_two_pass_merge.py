from __future__ import annotations

from dataclasses import replace

from tennis_analyzer.analysis.audio_impact import TwoPassImpactDetector, HybridImpactDetector
from tennis_analyzer.analysis.impact import ImpactEvent


def _ev(idx: int, speed: float) -> ImpactEvent:
    return ImpactEvent(
        impact_frame_idx=idx,
        trigger_frame_idx=idx + 1,
        peak_speed_px_s=float(speed),
        peak_speed_sw_s=None,
        peak_velocity_unit=(1.0, 0.0),
    )


def test_two_pass_merges_close_impacts_keeps_strongest():
    # Construct without running __init__ (avoids cv2/ffmpeg/video scanning).
    det = TwoPassImpactDetector.__new__(TwoPassImpactDetector)
    det.fps = 30.0
    det.merge_within_s = 0.8  # ~24 frames
    det.audio_tolerance_frames = 3
    det.require_audio = True

    # Fake audio detector: onsets near both events.
    audio = HybridImpactDetector.__new__(HybridImpactDetector)
    audio.audio_onset_frames = [10, 20, 100]
    det.detector = audio

    det._impact_events_by_frame = {
        10: _ev(10, 600.0),
        20: _ev(20, 1200.0),  # should win within the cluster
        100: _ev(100, 900.0),
    }

    TwoPassImpactDetector._merge_close_impacts(det)
    assert det.get_impact_frames() == [20, 100]


def test_two_pass_merge_tie_breakers_prefer_audio_alignment_then_later():
    det = TwoPassImpactDetector.__new__(TwoPassImpactDetector)
    det.fps = 30.0
    det.merge_within_s = 0.8
    det.audio_tolerance_frames = 3
    det.require_audio = True

    audio = HybridImpactDetector.__new__(HybridImpactDetector)
    audio.audio_onset_frames = [10, 21]
    det.detector = audio

    # Same speed; event at 21 is closer to onset 21 than event at 10 is to any onset,
    # so 21 should win. If still tied, later frame should win.
    det._impact_events_by_frame = {
        10: _ev(10, 1000.0),
        21: _ev(21, 1000.0),
    }

    TwoPassImpactDetector._merge_close_impacts(det)
    assert det.get_impact_frames() == [21]


def test_two_pass_merge_prefers_late_onset_tight_alignment_over_speed():
    # Regression test for side-view: a far-off (but very fast) wrist peak should
    # not beat a tightly-aligned peak near the later (contact) onset within the
    # same onset cluster (bounce+contact).
    det = TwoPassImpactDetector.__new__(TwoPassImpactDetector)
    det.fps = 30.0
    det.merge_within_s = 0.8
    det.audio_tolerance_frames = 7
    det.require_audio = True

    audio = HybridImpactDetector.__new__(HybridImpactDetector)
    audio.audio_onset_frames = [100, 120]
    det.detector = audio

    det._impact_events_by_frame = {
        105: _ev(105, 2000.0),  # near onset 100 but far from onset 120
        121: _ev(121, 800.0),   # tightly aligned to the later onset 120
    }

    TwoPassImpactDetector._merge_close_impacts(det)
    assert det.get_impact_frames() == [121]

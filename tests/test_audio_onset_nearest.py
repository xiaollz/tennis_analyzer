from tennis_analyzer.analysis.audio_impact import HybridImpactDetector


def test_nearest_audio_onset_within_tolerance():
    # Construct without running __init__ (avoids ffmpeg dependency).
    det = HybridImpactDetector.__new__(HybridImpactDetector)
    det.audio_onset_frames = [10, 20, 40]

    assert det.nearest_audio_onset(21, tolerance=1) == 20
    assert det.nearest_audio_onset(9, tolerance=1) == 10
    assert det.nearest_audio_onset(30, tolerance=5) is None


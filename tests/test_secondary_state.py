"""Tests for secondary channel video curation and state generation.

Tests cover: TomAllsopp + Feel Tennis video lists, dedup, schema, state generation.
"""

import pytest


def test_tomallsopp_list_size():
    """TOMALLSOPP_FOREHAND_VIDEOS has 40-50 entries."""
    from knowledge.pipeline.secondary_videos import TOMALLSOPP_FOREHAND_VIDEOS
    assert 40 <= len(TOMALLSOPP_FOREHAND_VIDEOS) <= 50, (
        f"Expected 40-50 TomAllsopp videos, got {len(TOMALLSOPP_FOREHAND_VIDEOS)}"
    )


def test_tomallsopp_entry_schema():
    """Each TomAllsopp entry has video_id, title, duration keys."""
    from knowledge.pipeline.secondary_videos import TOMALLSOPP_FOREHAND_VIDEOS
    for entry in TOMALLSOPP_FOREHAND_VIDEOS:
        assert "video_id" in entry, f"Missing video_id in {entry}"
        assert "title" in entry, f"Missing title in {entry}"
        assert "duration" in entry, f"Missing duration in {entry}"
        assert isinstance(entry["video_id"], str)
        assert isinstance(entry["title"], str)
        assert isinstance(entry["duration"], int)


def test_feeltennis_list_size():
    """FEELTENNIS_FOREHAND_VIDEOS has 40-50 entries."""
    from knowledge.pipeline.secondary_videos import FEELTENNIS_FOREHAND_VIDEOS
    assert 40 <= len(FEELTENNIS_FOREHAND_VIDEOS) <= 50, (
        f"Expected 40-50 Feel Tennis videos, got {len(FEELTENNIS_FOREHAND_VIDEOS)}"
    )


def test_feeltennis_entry_schema():
    """Each Feel Tennis entry has video_id, title, duration keys."""
    from knowledge.pipeline.secondary_videos import FEELTENNIS_FOREHAND_VIDEOS
    for entry in FEELTENNIS_FOREHAND_VIDEOS:
        assert "video_id" in entry, f"Missing video_id in {entry}"
        assert "title" in entry, f"Missing title in {entry}"
        assert "duration" in entry, f"Missing duration in {entry}"
        assert isinstance(entry["video_id"], str)
        assert isinstance(entry["title"], str)
        assert isinstance(entry["duration"], int)


def test_no_cross_channel_duplicates():
    """No video_id appears in both TomAllsopp and Feel Tennis lists."""
    from knowledge.pipeline.secondary_videos import (
        TOMALLSOPP_FOREHAND_VIDEOS,
        FEELTENNIS_FOREHAND_VIDEOS,
    )
    tom_ids = {v["video_id"] for v in TOMALLSOPP_FOREHAND_VIDEOS}
    feel_ids = {v["video_id"] for v in FEELTENNIS_FOREHAND_VIDEOS}
    overlap = tom_ids & feel_ids
    assert not overlap, f"Cross-channel duplicates found: {overlap}"


def test_ftt_overlap_excluded():
    """Known FTT overlap video '1-g1OD8gh-I' is NOT in either secondary list."""
    from knowledge.pipeline.secondary_videos import (
        TOMALLSOPP_FOREHAND_VIDEOS,
        FEELTENNIS_FOREHAND_VIDEOS,
    )
    all_ids = {v["video_id"] for v in TOMALLSOPP_FOREHAND_VIDEOS} | {
        v["video_id"] for v in FEELTENNIS_FOREHAND_VIDEOS
    }
    assert "1-g1OD8gh-I" not in all_ids, "FTT overlap video should be excluded"


def test_generate_tomallsopp_state():
    """generate_tomallsopp_state() returns dict with correct schema, all pending."""
    from knowledge.pipeline.secondary_videos import generate_tomallsopp_state
    state = generate_tomallsopp_state()
    assert "channel_id" in state
    assert "total_videos" in state
    assert "videos" in state
    assert state["channel_id"] == "@TomAllsopp"
    assert 40 <= state["total_videos"] <= 50
    for vid_id, entry in state["videos"].items():
        assert entry["status"] == "pending"


def test_generate_feeltennis_state():
    """generate_feeltennis_state() returns dict with correct schema, all pending."""
    from knowledge.pipeline.secondary_videos import generate_feeltennis_state
    state = generate_feeltennis_state()
    assert "channel_id" in state
    assert "total_videos" in state
    assert "videos" in state
    assert state["channel_id"] == "@feeltennis"
    assert 40 <= state["total_videos"] <= 50
    for vid_id, entry in state["videos"].items():
        assert entry["status"] == "pending"


def test_state_matches_video_entry_schema():
    """Each video entry in generated state matches VideoEntry TypedDict schema."""
    from knowledge.pipeline.secondary_videos import generate_tomallsopp_state
    from knowledge.pipeline.video_state import VideoEntry
    required_keys = set(VideoEntry.__annotations__.keys())
    state = generate_tomallsopp_state()
    for vid_id, entry in state["videos"].items():
        entry_keys = set(entry.keys())
        missing = required_keys - entry_keys
        assert not missing, f"Video {vid_id} missing keys: {missing}"


def test_feeltennis_paid_content_excluded():
    """Feel Tennis paid content video 'CLEjGDGEGaA' is NOT in FEELTENNIS list."""
    from knowledge.pipeline.secondary_videos import FEELTENNIS_FOREHAND_VIDEOS
    feel_ids = {v["video_id"] for v in FEELTENNIS_FOREHAND_VIDEOS}
    assert "CLEjGDGEGaA" not in feel_ids, "Paid content video should be excluded"

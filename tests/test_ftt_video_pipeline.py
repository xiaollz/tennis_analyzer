"""Tests for FTT video pipeline modules: video_state, video_analyzer, video_concept_extractor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Task 1: Video state manager tests
# ---------------------------------------------------------------------------


class TestVideoState:
    """Tests for knowledge.pipeline.video_state module."""

    def test_generate_initial_state_has_73_videos(self):
        from knowledge.pipeline.video_state import generate_initial_state

        state = generate_initial_state()
        assert len(state["videos"]) == 73

    def test_generate_initial_state_33_analyzed(self):
        from knowledge.pipeline.video_state import generate_initial_state, get_videos_by_status

        state = generate_initial_state()
        analyzed = get_videos_by_status(state, "analyzed")
        assert len(analyzed) == 33

    def test_generate_initial_state_40_pending(self):
        from knowledge.pipeline.video_state import generate_initial_state, get_videos_by_status

        state = generate_initial_state()
        pending = get_videos_by_status(state, "pending")
        assert len(pending) == 40

    def test_analyzed_videos_have_analysis_file(self):
        from knowledge.pipeline.video_state import generate_initial_state

        state = generate_initial_state()
        for vid_id, entry in state["videos"].items():
            if entry["status"] == "analyzed":
                assert entry["analysis_file"] is not None, f"Analyzed video {vid_id} missing analysis_file"

    def test_pending_videos_have_no_analysis_file(self):
        from knowledge.pipeline.video_state import generate_initial_state

        state = generate_initial_state()
        for vid_id, entry in state["videos"].items():
            if entry["status"] == "pending":
                assert entry["analysis_file"] is None, f"Pending video {vid_id} should not have analysis_file"

    def test_mark_video_updates_status(self):
        from knowledge.pipeline.video_state import generate_initial_state, mark_video

        state = generate_initial_state()
        # Pick a pending video
        pending_id = [k for k, v in state["videos"].items() if v["status"] == "pending"][0]
        mark_video(state, pending_id, "analyzed", analysis_file="test.md")
        assert state["videos"][pending_id]["status"] == "analyzed"
        assert state["videos"][pending_id]["analysis_file"] == "test.md"

    def test_get_pending_videos_returns_only_pending(self):
        from knowledge.pipeline.video_state import generate_initial_state, get_videos_by_status

        state = generate_initial_state()
        pending = get_videos_by_status(state, "pending")
        for v in pending:
            assert v["status"] == "pending"

    def test_get_analyzed_videos_returns_only_analyzed(self):
        from knowledge.pipeline.video_state import generate_initial_state, get_videos_by_status

        state = generate_initial_state()
        analyzed = get_videos_by_status(state, "analyzed")
        for v in analyzed:
            assert v["status"] == "analyzed"

    def test_save_and_load_state_roundtrip(self):
        from knowledge.pipeline.video_state import generate_initial_state, save_state, load_state

        state = generate_initial_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(state, path)
            loaded = load_state(path)
            assert len(loaded["videos"]) == 73
            assert loaded["videos"] == state["videos"]

    def test_load_state_creates_initial_if_missing(self):
        from knowledge.pipeline.video_state import load_state

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            state = load_state(path)
            assert len(state["videos"]) == 73
            # File should have been created
            assert path.exists()

    def test_get_state_summary(self):
        from knowledge.pipeline.video_state import generate_initial_state, get_state_summary

        state = generate_initial_state()
        summary = get_state_summary(state)
        assert summary["analyzed"] == 33
        assert summary["pending"] == 40
        assert summary["total"] == 73

    def test_all_video_entries_have_required_fields(self):
        from knowledge.pipeline.video_state import generate_initial_state

        required_fields = {"video_id", "title", "url", "duration", "status",
                          "analysis_file", "extracted_file", "analyzed_at", "error"}
        state = generate_initial_state()
        for vid_id, entry in state["videos"].items():
            missing = required_fields - set(entry.keys())
            assert not missing, f"Video {vid_id} missing fields: {missing}"

    def test_video_urls_are_correct_format(self):
        from knowledge.pipeline.video_state import generate_initial_state

        state = generate_initial_state()
        for vid_id, entry in state["videos"].items():
            assert entry["url"] == f"https://www.youtube.com/watch?v={vid_id}"

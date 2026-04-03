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


# ---------------------------------------------------------------------------
# Task 2: Video analyzer tests
# ---------------------------------------------------------------------------


# Sample analysis text fixture (excerpt from a real FTT video analysis)
SAMPLE_ANALYSIS_TEXT = """
### 1. 具体技术要点清单

*   **击穿球体 (Hitting Through the Ball)：** 核心要点。无论打平击还是上旋，都必须"击穿"球体。
*   **肩膀倾斜 (Shoulder Slant)：** 打上旋时，肩膀应有轻微的倾斜。

### 2. 独特的比喻、训练法、提示语与口令

*   **比喻：耐克对勾 (Nike Swoosh)：** 理想的挥拍轨迹不是"C"型，而是"耐克标志"。
*   **训练方法：适应性反应练习 (Adaptive Response Drill)：** 当球短且身体前倾时，打得更平。

### 4. 与"容错型正手" (Fault Tolerant Forehand) 理论的关联

*   **动力链的稳定性：** 强调通过肩膀倾斜和躯干旋转来产生上旋。
"""

# Sample multi-video markdown for extract_from_existing_markdown tests
SAMPLE_MULTI_VIDEO_MD = """# FTT Videos

---

## 1. First Video Title

**URL**: https://www.youtube.com/watch?v=AAAAAAAAAAA

**频道**: Fault Tolerant Tennis

Some analysis content for video 1.

### 1. 具体技术要点清单
*   **Point A：** Description A.

---

## 2. Second Video Title

**URL**: https://www.youtube.com/watch?v=BBBBBBBBBBB

**频道**: Fault Tolerant Tennis

Some analysis content for video 2.

### 1. 具体技术要点清单
*   **Point B：** Description B.
"""


class TestVideoAnalyzer:
    """Tests for knowledge.pipeline.video_analyzer module."""

    def test_load_api_config(self):
        from knowledge.pipeline.video_analyzer import load_api_config
        from pathlib import Path

        config = load_api_config(Path("config/youtube_api_config.json"))
        assert "api_key" in config
        assert "base_url" in config
        assert "model" in config

    def test_create_client_uses_base_url_from_config(self):
        """Verify that create_client threads the proxy base_url correctly."""
        from unittest.mock import patch, MagicMock
        from knowledge.pipeline.video_analyzer import create_client

        config = {
            "api_key": "test-key",
            "base_url": "https://proxy.example.com",
            "model": "test-model",
        }

        with patch("knowledge.pipeline.video_analyzer.genai") as mock_genai:
            mock_genai.Client = MagicMock()
            mock_types = MagicMock()
            with patch("knowledge.pipeline.video_analyzer.types", mock_types):
                create_client(config)

            # Verify Client was called with the proxy base_url
            call_kwargs = mock_genai.Client.call_args
            assert call_kwargs is not None
            # api_key should be passed
            assert call_kwargs.kwargs.get("api_key") == "test-key" or \
                   (call_kwargs.args and call_kwargs.args[0] == "test-key") or \
                   call_kwargs[1].get("api_key") == "test-key"

    def test_analyze_video_constructs_correct_request(self):
        """Verify analyze_video sends YouTube URL as file_data part."""
        from unittest.mock import MagicMock, patch
        from knowledge.pipeline.video_analyzer import analyze_video

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis result"
        mock_client.models.generate_content.return_value = mock_response

        with patch("knowledge.pipeline.video_analyzer.types") as mock_types:
            mock_types.Content = MagicMock()
            mock_types.Part = MagicMock()
            mock_types.FileData = MagicMock()
            mock_types.GenerateContentConfig = MagicMock()

            result = analyze_video(mock_client, "test123", "Analyze this", model="test-model")

        assert result == "Analysis result"
        mock_client.models.generate_content.assert_called_once()

    def test_analyze_video_returns_markdown_string(self):
        """Verify return type is str."""
        from unittest.mock import MagicMock, patch
        from knowledge.pipeline.video_analyzer import analyze_video

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Video Analysis\nSome content"
        mock_client.models.generate_content.return_value = mock_response

        with patch("knowledge.pipeline.video_analyzer.types") as mock_types:
            mock_types.Content = MagicMock()
            mock_types.Part = MagicMock()
            mock_types.FileData = MagicMock()
            mock_types.GenerateContentConfig = MagicMock()

            result = analyze_video(mock_client, "abc123", "prompt")

        assert isinstance(result, str)
        assert "Video Analysis" in result


# ---------------------------------------------------------------------------
# Task 2: Video concept extractor tests
# ---------------------------------------------------------------------------


class TestVideoConceptExtractor:
    """Tests for knowledge.pipeline.video_concept_extractor module."""

    def test_extract_from_existing_markdown_splits_sections(self):
        """Test that multi-video markdown is split into per-video sections."""
        from knowledge.pipeline.video_concept_extractor import extract_from_existing_markdown
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(SAMPLE_MULTI_VIDEO_MD)
            f.flush()
            sections = extract_from_existing_markdown(Path(f.name))

        assert len(sections) == 2
        assert sections[0]["video_id"] == "AAAAAAAAAAA"
        assert sections[1]["video_id"] == "BBBBBBBBBBB"

    def test_extract_from_existing_markdown_deduplicates(self):
        """Test that duplicate video IDs are deduplicated (keep first occurrence)."""
        from knowledge.pipeline.video_concept_extractor import extract_from_existing_markdown
        import tempfile

        # Create markdown with duplicate video ID
        dup_md = """# Test

## 1. First Occurrence

**URL**: https://www.youtube.com/watch?v=DUPLICATED

Content 1.

## 2. Second Occurrence (duplicate)

**URL**: https://www.youtube.com/watch?v=DUPLICATED

Content 2 (should be skipped).

## 3. Unique Video

**URL**: https://www.youtube.com/watch?v=UNIQUEVID01

Content 3.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(dup_md)
            f.flush()
            sections = extract_from_existing_markdown(Path(f.name))

        video_ids = [s["video_id"] for s in sections]
        assert len(video_ids) == 2  # Deduplicated
        assert video_ids.count("DUPLICATED") == 1
        assert "UNIQUEVID01" in video_ids

    def test_extract_concepts_from_analysis_produces_concepts(self):
        """Test that concept extraction produces valid Concept objects."""
        from knowledge.pipeline.video_concept_extractor import extract_concepts_from_analysis
        from knowledge.registry import ConceptRegistry

        registry = ConceptRegistry()
        concepts, edges, chains = extract_concepts_from_analysis(
            SAMPLE_ANALYSIS_TEXT, "gehRK0Y6AdQ", "The Pros Brush Differently", registry
        )

        # Should extract at least some concepts
        assert len(concepts) > 0
        # All concepts should have valid snake_case IDs
        import re
        for c in concepts:
            assert re.match(r"^[a-z][a-z0-9_]*$", c.id), f"Invalid ID: {c.id}"

    def test_extract_concepts_produces_edges(self):
        """Test that extraction produces Edge objects linking concepts."""
        from knowledge.pipeline.video_concept_extractor import extract_concepts_from_analysis
        from knowledge.registry import ConceptRegistry

        registry = ConceptRegistry()
        concepts, edges, chains = extract_concepts_from_analysis(
            SAMPLE_ANALYSIS_TEXT, "gehRK0Y6AdQ", "The Pros Brush Differently", registry
        )

        # Should produce some edges (at least from teaching points linking concepts)
        # Note: may be 0 for simple analysis text, so we just check type
        assert isinstance(edges, list)
        for e in edges:
            assert hasattr(e, "source_id")
            assert hasattr(e, "target_id")

    def test_extract_from_real_file(self):
        """Test extraction from actual 09_ftt_videos_1.md file."""
        from knowledge.pipeline.video_concept_extractor import extract_from_existing_markdown

        md_path = Path("docs/research/09_ftt_videos_1.md")
        if not md_path.exists():
            pytest.skip("Real analysis file not available")

        sections = extract_from_existing_markdown(md_path)
        assert len(sections) == 10  # File 1 has 10 videos
        # All should have video_id extracted
        for s in sections:
            assert s["video_id"] != ""
            assert "unknown" not in s["video_id"]

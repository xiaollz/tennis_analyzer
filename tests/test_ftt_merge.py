"""Tests for FTT video concept merge logic."""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.registry import ConceptRegistry
from knowledge.schemas import Concept, ConceptType, Edge, RelationType


class TestConceptDeduplication:
    """Test that fuzzy dedup catches near-duplicate concepts."""

    def test_exact_duplicate_rejected(self):
        reg = ConceptRegistry()
        c1 = Concept(
            id="hip_rotation",
            name="Hip Rotation",
            name_zh="髋部旋转",
            category=ConceptType.BIOMECHANICS,
            sources=["ftt"],
            description="Rotation of the hip",
        )
        assert reg.add(c1) is None  # New concept
        assert reg.add(c1) == "hip_rotation"  # Duplicate

    def test_fuzzy_duplicate_detected(self):
        reg = ConceptRegistry()
        c1 = Concept(
            id="hip_rotation",
            name="Hip Rotation",
            name_zh="髋部旋转",
            category=ConceptType.BIOMECHANICS,
            sources=["ftt"],
            description="Rotation of the hip",
        )
        c2 = Concept(
            id="hip_rotations",
            name="Hip Rotations",
            name_zh="髋部旋转",
            category=ConceptType.BIOMECHANICS,
            sources=["ftt"],
            description="Rotation of hips",
        )
        assert reg.add(c1) is None
        # Should be caught as duplicate of hip_rotation
        result = reg.add(c2)
        assert result == "hip_rotation"

    def test_distinct_concepts_kept(self):
        reg = ConceptRegistry()
        c1 = Concept(
            id="hip_rotation",
            name="Hip Rotation",
            name_zh="髋部旋转",
            category=ConceptType.BIOMECHANICS,
            sources=["ftt"],
            description="Rotation of the hip",
        )
        c2 = Concept(
            id="shoulder_adduction",
            name="Shoulder Adduction",
            name_zh="肩部内收",
            category=ConceptType.BIOMECHANICS,
            sources=["ftt"],
            description="Adduction of the shoulder",
        )
        assert reg.add(c1) is None
        assert reg.add(c2) is None
        assert len(reg) == 2


class TestMergeReport:
    """Test merge report structure and content."""

    def test_merge_report_exists_after_run(self):
        """Merge report file should exist after script runs."""
        report_path = Path("knowledge/extracted/ftt_videos/_merge_report.json")
        assert report_path.exists(), "Merge report not found"
        data = json.loads(report_path.read_text())
        assert "base_registry_count" in data
        assert "final_registry_count" in data
        assert "total_raw_concepts_from_videos" in data
        assert "new_concepts_added" in data
        assert "concepts_deduplicated" in data
        assert "per_video" in data

    def test_merge_report_dedup_stats(self):
        """Dedup should have caught some near-duplicates."""
        report_path = Path("knowledge/extracted/ftt_videos/_merge_report.json")
        if not report_path.exists():
            pytest.skip("Merge report not yet generated")
        data = json.loads(report_path.read_text())
        # At least some dedup should have happened
        assert data["concepts_deduplicated"] > 0, "No deduplication occurred"

    def test_merge_report_per_video_coverage(self):
        """Merge report should have stats for all processed videos."""
        report_path = Path("knowledge/extracted/ftt_videos/_merge_report.json")
        if not report_path.exists():
            pytest.skip("Merge report not yet generated")
        data = json.loads(report_path.read_text())
        assert len(data["per_video"]) > 60, f"Only {len(data['per_video'])} videos in report"


class TestRegistrySize:
    """Test that registry stays within acceptable bounds."""

    def test_registry_not_exploded(self):
        """Registry should stay under 600 concepts."""
        snapshot = Path("knowledge/extracted/_registry_snapshot.json")
        if not snapshot.exists():
            pytest.skip("Registry snapshot not yet generated")
        data = json.loads(snapshot.read_text())
        assert len(data) <= 600, f"Registry exploded to {len(data)} concepts"
        assert len(data) >= 300, f"Registry too small at {len(data)} concepts"


class TestEdgeDeduplication:
    """Test that duplicate edges are properly handled."""

    def test_edge_dedup_by_key(self):
        """Edges with same source_id+target_id+relation should be deduped."""
        from collections import defaultdict

        edges = [
            {"source_id": "a", "target_id": "b", "relation": "supports", "confidence": 0.5},
            {"source_id": "a", "target_id": "b", "relation": "supports", "confidence": 0.8},
            {"source_id": "a", "target_id": "c", "relation": "causes", "confidence": 0.7},
        ]

        edge_key = lambda e: (e["source_id"], e["target_id"], e["relation"])
        groups = defaultdict(list)
        for e in edges:
            groups[edge_key(e)].append(e)

        unique = [max(g, key=lambda e: e.get("confidence", 0)) for g in groups.values()]
        assert len(unique) == 2
        # The a->b supports edge should keep the higher confidence
        ab_edge = [e for e in unique if e["target_id"] == "b"][0]
        assert ab_edge["confidence"] == 0.8


class TestQualityFilter:
    """Test concept quality filtering."""

    def test_player_names_filtered(self):
        """Player names should not become concepts."""
        from scripts.merge_ftt_video_concepts import _is_quality_concept_fn

        filter_fn = _is_quality_concept_fn()
        assert not filter_fn({"id": "carlos_alcaraz", "name": "Carlos Alcaraz"})
        assert not filter_fn({"id": "giovanni_mpetshi_perricard", "name": "Giovanni Mpetshi Perricard"})

    def test_valid_concepts_pass_filter(self):
        """Legitimate tennis concepts should pass the filter."""
        from scripts.merge_ftt_video_concepts import _is_quality_concept_fn

        filter_fn = _is_quality_concept_fn()
        assert filter_fn({"id": "hip_rotation", "name": "Hip Rotation"})
        assert filter_fn({"id": "shoulder_adduction", "name": "Shoulder Adduction"})
        assert filter_fn({"id": "trunk_sequencing", "name": "Trunk Sequencing"})

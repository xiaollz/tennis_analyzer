"""Tests for cross-source reconciliation logic.

Verifies that secondary source concepts (TomAllsopp, Feel Tennis) are correctly
classified as agreement, complement, or conflict against the FTT-primary registry,
and that confidence scores are updated accordingly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from knowledge.schemas import Concept, ConceptType
from knowledge.registry import ConceptRegistry


def _make_concept(
    id: str,
    name: str,
    sources: list[str] | None = None,
    confidence: float = 0.8,
    description: str = "test concept",
    category: ConceptType = ConceptType.TECHNIQUE,
    name_zh: str = "测试",
    aliases: list[str] | None = None,
) -> Concept:
    return Concept(
        id=id,
        name=name,
        name_zh=name_zh,
        aliases=aliases or [],
        category=category,
        sources=sources or ["ftt"],
        description=description,
        confidence=confidence,
    )


def _make_registry_with_ftt_concepts() -> ConceptRegistry:
    """Build a small registry with representative FTT concepts."""
    reg = ConceptRegistry()
    reg.add(_make_concept("hip_rotation", "Hip Rotation", description="Active hip rotation initiates the kinetic chain"))
    reg.add(_make_concept("wrist_lag", "Wrist Lag", description="Passive wrist lag from proper sequencing"))
    reg.add(_make_concept("unit_turn", "Unit Turn", description="Coordinated shoulder and hip turn as one unit"))
    reg.add(_make_concept("gravity_drop", "Gravity Driven Elbow Drop", description="Letting the racket drop via gravity"))
    reg.add(_make_concept("contact_point", "Contact Point", description="Optimal contact point in front of the body"))
    return reg


class TestClassifyConcept:
    """Test classify_concept() classification logic."""

    def test_agreement_when_fuzzy_match_high(self):
        """Concept matching existing FTT concept at >= 70 threshold -> agreement."""
        from knowledge.pipeline.reconciliation import classify_concept

        reg = _make_registry_with_ftt_concepts()
        secondary = _make_concept(
            "ta_hip_rotation", "Hip Rotation Technique",
            sources=["tomallsopp"],
            description="Hip rotation drives power in the forehand",
        )
        result = classify_concept(secondary, reg)
        assert result == "agreement"

    def test_complement_when_no_match(self):
        """Truly new concept with no fuzzy match -> complement."""
        from knowledge.pipeline.reconciliation import classify_concept

        reg = _make_registry_with_ftt_concepts()
        secondary = _make_concept(
            "ta_rollercoaster", "Rollercoaster Peak Dip",
            sources=["tomallsopp"],
            description="Momentarily reducing grip pressure at the top of backswing",
        )
        result = classify_concept(secondary, reg)
        assert result == "complement"

    def test_conflict_with_negation_words(self):
        """Concept description contains negation of FTT concept -> conflict."""
        from knowledge.pipeline.reconciliation import classify_concept

        reg = _make_registry_with_ftt_concepts()
        secondary = _make_concept(
            "ft_no_wrist_lag", "Wrist Lag",
            sources=["feeltennis"],
            description="You should not focus on wrist lag as it is wrong to actively create lag",
        )
        result = classify_concept(secondary, reg)
        assert result == "conflict"

    def test_ambiguous_defaults_to_complement(self):
        """When match is below threshold, default to complement (safe)."""
        from knowledge.pipeline.reconciliation import classify_concept

        reg = _make_registry_with_ftt_concepts()
        secondary = _make_concept(
            "ft_feel_swing", "Feel Based Swing Learning",
            sources=["feeltennis"],
            description="Learn the forehand through proprioceptive feel cues",
        )
        result = classify_concept(secondary, reg)
        assert result == "complement"


class TestBoostConfidence:
    """Test boost_confidence() confidence score updates."""

    def test_single_secondary_boosts_to_09(self):
        """FTT concept + 1 secondary agreement -> 0.9 confidence."""
        from knowledge.pipeline.reconciliation import boost_confidence

        reg = _make_registry_with_ftt_concepts()
        boost_confidence(reg, "hip_rotation", ["tomallsopp"])
        concept = reg.get("hip_rotation")
        assert concept.confidence == 0.9
        assert "tomallsopp" in concept.sources

    def test_two_secondaries_boost_to_095(self):
        """FTT concept + 2 secondaries -> 0.95 confidence."""
        from knowledge.pipeline.reconciliation import boost_confidence

        reg = _make_registry_with_ftt_concepts()
        boost_confidence(reg, "hip_rotation", ["tomallsopp", "feeltennis"])
        concept = reg.get("hip_rotation")
        assert concept.confidence == 0.95
        assert "tomallsopp" in concept.sources
        assert "feeltennis" in concept.sources


class TestReconcileAll:
    """Test reconcile_all() end-to-end reconciliation."""

    def _setup_extraction_dirs(self, tmp_path: Path):
        """Create mock extraction directories with per-video JSONs."""
        ta_dir = tmp_path / "tomallsopp_videos"
        ta_dir.mkdir()
        ft_dir = tmp_path / "feeltennis_videos"
        ft_dir.mkdir()

        # TomAllsopp video with agreement concept + complement concept
        ta_video = {
            "video_id": "test_ta_1",
            "concepts": [
                _make_concept(
                    "ta_hip_rotation", "Hip Rotation Power",
                    sources=["tomallsopp"],
                    description="Hip rotation drives forehand power",
                ).model_dump(),
                _make_concept(
                    "ta_rollercoaster", "Rollercoaster Peak Dip",
                    sources=["tomallsopp"],
                    description="Reducing grip pressure at backswing peak",
                ).model_dump(),
            ],
            "edges": [],
            "diagnostic_chains": [],
        }
        (ta_dir / "test_ta_1.json").write_text(json.dumps(ta_video))

        # Feel Tennis video with agreement + conflict
        ft_video = {
            "video_id": "test_ft_1",
            "concepts": [
                _make_concept(
                    "ft_unit_turn", "Unit Turn",
                    sources=["feeltennis"],
                    description="Unit turn is the foundation of modern forehand",
                ).model_dump(),
                _make_concept(
                    "ft_no_lag", "Wrist Lag",
                    sources=["feeltennis"],
                    description="You should not actively create wrist lag, it is wrong to force lag",
                ).model_dump(),
            ],
            "edges": [],
            "diagnostic_chains": [],
        }
        (ft_dir / "test_ft_1.json").write_text(json.dumps(ft_video))

        return ta_dir, ft_dir

    def test_reconcile_returns_result_with_counts(self, tmp_path):
        """reconcile_all returns ReconciliationResult with correct counts."""
        from knowledge.pipeline.reconciliation import reconcile_all

        reg = _make_registry_with_ftt_concepts()
        ta_dir, ft_dir = self._setup_extraction_dirs(tmp_path)
        result = reconcile_all(reg, ta_dir, ft_dir)

        assert result["total"] == 4  # 2 from TA + 2 from FT
        assert result["agreements"] + result["complements"] + result["conflicts"] == result["total"]
        assert result["agreements"] >= 2  # hip_rotation + unit_turn
        assert result["complements"] >= 1  # rollercoaster
        assert result["conflicts"] >= 1  # wrist lag conflict

    def test_complement_added_to_registry(self, tmp_path):
        """Complement concepts are added to registry with confidence=0.6."""
        from knowledge.pipeline.reconciliation import reconcile_all

        reg = _make_registry_with_ftt_concepts()
        initial_size = len(reg)
        ta_dir, ft_dir = self._setup_extraction_dirs(tmp_path)
        reconcile_all(reg, ta_dir, ft_dir)

        assert len(reg) > initial_size
        # Find the complement concept
        all_concepts = reg.all_concepts()
        complements = [c for c in all_concepts if c.confidence == 0.6 and "tomallsopp" in c.sources]
        assert len(complements) >= 1

    def test_conflict_logged(self, tmp_path):
        """Conflict concepts appear in conflict_log."""
        from knowledge.pipeline.reconciliation import reconcile_all

        reg = _make_registry_with_ftt_concepts()
        ta_dir, ft_dir = self._setup_extraction_dirs(tmp_path)
        result = reconcile_all(reg, ta_dir, ft_dir)

        assert len(result["conflict_log"]) >= 1
        conflict = result["conflict_log"][0]
        assert "concept_name" in conflict
        assert "source" in conflict
        assert "matched_ftt_id" in conflict

    def test_dual_source_complement_boosted_to_07(self, tmp_path):
        """If same complement concept in both channels, confidence boosted to 0.7."""
        from knowledge.pipeline.reconciliation import reconcile_all

        reg = _make_registry_with_ftt_concepts()

        ta_dir = tmp_path / "tomallsopp_videos"
        ta_dir.mkdir()
        ft_dir = tmp_path / "feeltennis_videos"
        ft_dir.mkdir()

        # Same concept name in both channels (complement, not in FTT registry)
        shared_name = "Elastic Band Effect"
        ta_video = {
            "video_id": "ta_shared",
            "concepts": [
                _make_concept(
                    "ta_elastic", shared_name,
                    sources=["tomallsopp"],
                    description="Elastic energy stored via body segment stretch",
                ).model_dump(),
            ],
            "edges": [],
            "diagnostic_chains": [],
        }
        ft_video = {
            "video_id": "ft_shared",
            "concepts": [
                _make_concept(
                    "ft_elastic", shared_name,
                    sources=["feeltennis"],
                    description="Elastic band effect from body rotation stretch",
                ).model_dump(),
            ],
            "edges": [],
            "diagnostic_chains": [],
        }
        (ta_dir / "ta_shared.json").write_text(json.dumps(ta_video))
        (ft_dir / "ft_shared.json").write_text(json.dumps(ft_video))

        reconcile_all(reg, ta_dir, ft_dir)

        # Find the elastic band concept - should be boosted to 0.7
        all_c = reg.all_concepts()
        elastic = [c for c in all_c if "elastic" in c.name.lower()]
        assert len(elastic) >= 1
        assert elastic[0].confidence == 0.7

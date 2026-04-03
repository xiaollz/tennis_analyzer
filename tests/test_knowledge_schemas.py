"""Tests for knowledge schema models: Concept, Edge, DiagnosticChain."""

import pytest
from pydantic import ValidationError

from knowledge.schemas import (
    Concept,
    ConceptType,
    DiagnosticChain,
    DiagnosticStep,
    Edge,
    RelationType,
    SourceId,
)


class TestConcept:
    def test_concept_roundtrip(self):
        """Concept with all fields -> model_dump_json -> model_validate_json -> equal."""
        c = Concept(
            id="hip_rotation",
            name="Hip Rotation",
            name_zh="转髋",
            aliases=["hip turn", "hip drive"],
            category=ConceptType.TECHNIQUE,
            sources=["ftt", "tpa"],
            description="Rotational movement of hips initiating kinetic chain",
            vlm_features=["hip angle change in frontal view"],
            muscles_involved=["gluteus maximus", "hip flexors"],
            active_or_passive="active",
            confidence=1.0,
        )
        json_str = c.model_dump_json()
        c2 = Concept.model_validate_json(json_str)
        assert c == c2
        assert c2.id == "hip_rotation"

    def test_concept_id_validation(self):
        """Uppercase ID like 'Hip Rotation' raises ValidationError; 'hip_rotation' passes."""
        with pytest.raises(ValidationError):
            Concept(
                id="Hip Rotation",
                name="Hip Rotation",
                name_zh="转髋",
                category=ConceptType.TECHNIQUE,
                description="test",
            )
        # Valid ID should pass
        c = Concept(
            id="hip_rotation",
            name="Hip Rotation",
            name_zh="转髋",
            category=ConceptType.TECHNIQUE,
            description="test",
        )
        assert c.id == "hip_rotation"

    def test_concept_id_rejects_spaces(self):
        """'hip rotation' (with space) raises ValidationError."""
        with pytest.raises(ValidationError):
            Concept(
                id="hip rotation",
                name="Hip Rotation",
                name_zh="转髋",
                category=ConceptType.TECHNIQUE,
                description="test",
            )

    def test_concept_defaults(self):
        """Concept with only required fields has empty lists for optional list fields."""
        c = Concept(
            id="hip_rotation",
            name="Hip Rotation",
            name_zh="转髋",
            category=ConceptType.TECHNIQUE,
            description="test",
        )
        assert c.aliases == []
        assert c.vlm_features == []
        assert c.muscles_involved == []
        assert c.sources == []
        assert c.active_or_passive is None
        assert c.confidence == 1.0


class TestEdge:
    def test_edge_validation(self):
        """Edge with RelationType.CAUSES roundtrips correctly."""
        e = Edge(
            source_id="hip_rotation",
            target_id="arm_lag",
            relation=RelationType.CAUSES,
            confidence=0.9,
            evidence="FTT: hip rotation initiates kinetic chain",
            source_file="13_synthesis.md",
        )
        json_str = e.model_dump_json()
        e2 = Edge.model_validate_json(json_str)
        assert e == e2
        assert e2.relation == RelationType.CAUSES

    def test_edge_all_relation_types(self):
        """All 7 RelationType values can be used in Edge."""
        for rt in RelationType:
            e = Edge(
                source_id="concept_a",
                target_id="concept_b",
                relation=rt,
                evidence="test evidence",
                source_file="test.md",
            )
            assert e.relation == rt
        assert len(RelationType) == 7


class TestDiagnosticChain:
    def test_diagnostic_chain(self):
        """DiagnosticChain with check_sequence of DiagnosticSteps roundtrips."""
        dc = DiagnosticChain(
            id="dc_arm_driven_hitting",
            symptom="Arm moves independently of trunk rotation",
            symptom_zh="手臂独立于躯干旋转运动",
            symptom_concept_id="arm_driven_hitting",
            check_sequence=[
                DiagnosticStep(
                    check="Does the arm start moving before hip rotation is visible?",
                    check_zh="手臂是否在髋部旋转可见之前开始移动?",
                    if_true="kinetic_chain_break",
                    if_false=None,
                )
            ],
            root_causes=["kinetic_chain_break", "back_tension_loss"],
            drills=["hips_hit_drill", "weighted_shadow_swing"],
            priority=1,
            vlm_frame="forward_swing_start",
        )
        json_str = dc.model_dump_json()
        dc2 = DiagnosticChain.model_validate_json(json_str)
        assert dc == dc2
        assert dc2.priority == 1

    def test_diagnostic_chain_priority_bounds(self):
        """priority=0 raises ValidationError, priority=6 raises ValidationError."""
        base = dict(
            id="dc_test",
            symptom="test",
            symptom_zh="test",
            symptom_concept_id="test",
            check_sequence=[],
            root_causes=[],
            drills=[],
            vlm_frame=None,
        )
        with pytest.raises(ValidationError):
            DiagnosticChain(**base, priority=0)
        with pytest.raises(ValidationError):
            DiagnosticChain(**base, priority=6)
        # Valid bounds
        dc1 = DiagnosticChain(**base, priority=1)
        assert dc1.priority == 1
        dc5 = DiagnosticChain(**base, priority=5)
        assert dc5.priority == 5

    def test_diagnostic_chain_id_prefix(self):
        """id must start with 'dc_', 'bad_id' raises ValidationError."""
        with pytest.raises(ValidationError):
            DiagnosticChain(
                id="bad_id",
                symptom="test",
                symptom_zh="test",
                symptom_concept_id="test",
                check_sequence=[],
                root_causes=[],
                drills=[],
                priority=1,
            )

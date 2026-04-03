"""Tests for knowledge schema models: Concept, Edge, DiagnosticChain, and multi-round VLM models."""

import json

import pytest
from pydantic import ValidationError

from knowledge.schemas import (
    Concept,
    ConceptType,
    DiagnosticChain,
    DiagnosticSession,
    DiagnosticStep,
    Edge,
    Hypothesis,
    HypothesisStatus,
    HypothesisUpdate,
    Observation,
    ObservationJudgment,
    RelationType,
    RoundResult,
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


# --- Multi-round VLM model tests ---


class TestHypothesis:
    def test_hypothesis_roundtrip(self):
        """Hypothesis with status=active, confidence=0.6, chain_id serializes and back."""
        h = Hypothesis(
            id="hyp_scooping",
            chain_id="dc_arm_driven_hitting",
            root_cause_concept_id="kinetic_chain_break",
            name="Scooping due to active lag",
            name_zh="因主动滞后导致的捞球",
            status=HypothesisStatus.ACTIVE,
            confidence=0.6,
            round_introduced=0,
        )
        data = h.model_dump()
        h2 = Hypothesis.model_validate(data)
        assert h == h2
        assert h2.status == HypothesisStatus.ACTIVE
        assert h2.confidence == 0.6
        assert h2.chain_id == "dc_arm_driven_hitting"

    def test_hypothesis_invalid_status(self):
        """Invalid status raises ValidationError."""
        with pytest.raises(ValidationError):
            Hypothesis(
                id="hyp_test",
                chain_id="dc_test",
                root_cause_concept_id="test",
                name="Test",
                name_zh="测试",
                status="bogus",
                confidence=0.5,
                round_introduced=0,
            )

    def test_hypothesis_confidence_bounds(self):
        """Confidence outside 0-1 raises ValidationError."""
        base = dict(
            id="hyp_test",
            chain_id="dc_test",
            root_cause_concept_id="test",
            name="Test",
            name_zh="测试",
            status=HypothesisStatus.ACTIVE,
            round_introduced=0,
        )
        with pytest.raises(ValidationError):
            Hypothesis(**base, confidence=1.5)
        with pytest.raises(ValidationError):
            Hypothesis(**base, confidence=-0.1)


class TestObservation:
    def test_observation_serializes(self):
        """Observation with frame, judgment, confidence, round_number serializes correctly."""
        obs = Observation(
            id="obs_r1_01",
            round_number=1,
            frame="图3",
            description="Arm starts before hip rotation",
            judgment=ObservationJudgment.YES,
            confidence=0.9,
            directive_source="dc_arm_driven_hitting.check_sequence[0]",
        )
        data = obs.model_dump()
        assert data["frame"] == "图3"
        assert data["judgment"] == "yes"
        assert data["confidence"] == 0.9
        assert data["round_number"] == 1

    def test_observation_invalid_judgment(self):
        """Invalid judgment raises ValidationError."""
        with pytest.raises(ValidationError):
            Observation(
                id="obs_test",
                round_number=0,
                frame="图1",
                description="test",
                judgment="maybe",
                confidence=0.5,
                directive_source="test",
            )


class TestRoundResult:
    def test_round_result_fields(self):
        """RoundResult stores round_number, prompt, response, observations, hypothesis_updates."""
        obs = Observation(
            id="obs_r1_01",
            round_number=1,
            frame="图3",
            description="test obs",
            judgment=ObservationJudgment.YES,
            confidence=0.8,
            directive_source="test",
        )
        hu = HypothesisUpdate(
            hypothesis_id="hyp_scooping",
            action="confirm",
            reason="Observation confirms hypothesis",
        )
        rr = RoundResult(
            round_number=1,
            prompt_sent="Analyze frame 3...",
            raw_response="Frame 3 shows...",
            observations=[obs],
            hypothesis_updates=[hu],
            timestamp="2026-04-03T14:30:00Z",
        )
        assert rr.round_number == 1
        assert len(rr.observations) == 1
        assert len(rr.hypothesis_updates) == 1
        assert rr.timestamp == "2026-04-03T14:30:00Z"


class TestDiagnosticSession:
    def test_diagnostic_session_defaults(self):
        """DiagnosticSession defaults: convergence_score=0.0, empty lists."""
        sess = DiagnosticSession(session_id="sess_test")
        assert sess.convergence_score == 0.0
        assert sess.hypotheses == []
        assert sess.observations == []
        assert sess.rounds == []
        assert sess.active_chain_ids == []
        assert sess.checked_steps == {}
        assert sess.max_rounds == 4

    def test_diagnostic_session_convergence_bounds(self):
        """convergence_score outside 0-1 raises ValidationError."""
        with pytest.raises(ValidationError):
            DiagnosticSession(session_id="sess_test", convergence_score=1.5)
        with pytest.raises(ValidationError):
            DiagnosticSession(session_id="sess_test", convergence_score=-0.1)

    def test_diagnostic_session_roundtrip(self):
        """model_dump_json -> model_validate_json preserves all fields."""
        hyp = Hypothesis(
            id="hyp_a",
            chain_id="dc_test",
            root_cause_concept_id="test_cause",
            name="Test Hyp",
            name_zh="测试假设",
            status=HypothesisStatus.CONFIRMED,
            confidence=0.9,
            supporting_observations=["obs_r1_01"],
            round_introduced=0,
            round_resolved=2,
        )
        obs = Observation(
            id="obs_r1_01",
            round_number=1,
            frame="图3",
            description="test",
            judgment=ObservationJudgment.YES,
            confidence=0.8,
            directive_source="test",
        )
        rr = RoundResult(
            round_number=1,
            prompt_sent="prompt",
            raw_response="response",
            observations=[obs],
            hypothesis_updates=[],
        )
        sess = DiagnosticSession(
            session_id="sess_20260403_143000",
            video_path="/tmp/test.mp4",
            hypotheses=[hyp],
            observations=[obs],
            rounds=[rr],
            active_chain_ids=["dc_test"],
            checked_steps={"dc_test": [0, 1]},
            convergence_score=0.75,
            max_rounds=4,
            created_at="2026-04-03T14:30:00Z",
        )
        json_str = sess.model_dump_json()
        sess2 = DiagnosticSession.model_validate_json(json_str)
        assert sess == sess2
        assert sess2.convergence_score == 0.75
        assert sess2.hypotheses[0].status == HypothesisStatus.CONFIRMED
        assert sess2.observations[0].frame == "图3"

    def test_diagnostic_session_json_size(self):
        """Session with 2 rounds, 3 hypotheses, 5 observations under 10KB."""
        hyps = [
            Hypothesis(
                id=f"hyp_{i}",
                chain_id="dc_test",
                root_cause_concept_id="cause",
                name=f"Hyp {i}",
                name_zh=f"假设{i}",
                round_introduced=0,
            )
            for i in range(3)
        ]
        obss = [
            Observation(
                id=f"obs_r1_{i:02d}",
                round_number=1,
                frame=f"图{i}",
                description=f"Observation {i}",
                judgment=ObservationJudgment.YES,
                confidence=0.8,
                directive_source="test",
            )
            for i in range(5)
        ]
        rounds = [
            RoundResult(
                round_number=r,
                prompt_sent=f"Prompt for round {r}",
                raw_response=f"Response for round {r}",
                observations=obss[:3] if r == 1 else obss[3:],
            )
            for r in range(1, 3)
        ]
        sess = DiagnosticSession(
            session_id="sess_size_test",
            hypotheses=hyps,
            observations=obss,
            rounds=rounds,
        )
        json_bytes = len(sess.model_dump_json().encode("utf-8"))
        assert json_bytes < 10240, f"JSON size {json_bytes} exceeds 10KB"

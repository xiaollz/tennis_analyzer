"""Tests for diagnostic journey report section (RG-01 through RG-04)."""

import pytest

from report.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

def _make_session_data(
    rounds=None,
    hypotheses=None,
    convergence_score=0.85,
    observations=None,
):
    """Build a minimal DiagnosticSession dict for testing."""
    return {
        "session_id": "test_sess",
        "rounds": rounds or [],
        "hypotheses": hypotheses or [],
        "observations": observations or [],
        "convergence_score": convergence_score,
    }


def _make_3_round_session():
    """Build a realistic 3-round session: scan -> diagnose -> confirm."""
    hypotheses = [
        {
            "id": "hyp_dc_scooping",
            "chain_id": "dc_scooping",
            "root_cause_concept_id": "scooping_active",
            "name": "Scooping",
            "name_zh": "拍头下压",
            "status": "eliminated",
            "confidence": 0.0,
            "round_introduced": 0,
            "round_resolved": 2,
        },
        {
            "id": "hyp_dc_arm_driven",
            "chain_id": "dc_arm_driven",
            "root_cause_concept_id": "arm_disconnection",
            "name": "Arm-driven hitting",
            "name_zh": "手臂主导击球",
            "status": "confirmed",
            "confidence": 0.9,
            "round_introduced": 0,
            "round_resolved": 3,
        },
    ]

    rounds = [
        {
            "round_number": 0,
            "prompt_sent": "Pass 1 scan",
            "raw_response": "1, 2",
            "observations": [],
            "hypothesis_updates": [],
        },
        {
            "round_number": 1,
            "prompt_sent": "Observation directive",
            "raw_response": "{}",
            "observations": [
                {
                    "id": "obs_r1_01",
                    "round_number": 1,
                    "frame": "图3",
                    "description": "黄色轨迹在图2-3有V形尖角",
                    "judgment": "yes",
                    "confidence": 0.8,
                    "directive_source": "hyp_dc_scooping",
                },
                {
                    "id": "obs_r1_02",
                    "round_number": 1,
                    "frame": "图4",
                    "description": "肘角在击球时偏小约85度",
                    "judgment": "yes",
                    "confidence": 0.7,
                    "directive_source": "hyp_dc_arm_driven",
                },
            ],
            "hypothesis_updates": [
                {
                    "hypothesis_id": "hyp_dc_scooping",
                    "action": "adjust",
                    "reason": "V形轨迹可能是下游症状",
                },
            ],
        },
        {
            "round_number": 2,
            "prompt_sent": "Confirmation prompt",
            "raw_response": "{}",
            "observations": [
                {
                    "id": "obs_r2_01",
                    "round_number": 2,
                    "frame": "图1",
                    "description": "手臂在unit turn时未与躯干同步旋转",
                    "judgment": "yes",
                    "confidence": 0.9,
                    "directive_source": "hyp_dc_arm_driven",
                },
            ],
            "hypothesis_updates": [
                {
                    "hypothesis_id": "hyp_dc_arm_driven",
                    "action": "confirm",
                    "reason": "手臂脱离身体旋转系统确认为根因",
                },
                {
                    "hypothesis_id": "hyp_dc_scooping",
                    "action": "eliminate",
                    "reason": "scooping是手臂主导的下游症状",
                },
            ],
        },
    ]

    return _make_session_data(
        rounds=rounds,
        hypotheses=hypotheses,
        convergence_score=0.9,
    )


# ---------------------------------------------------------------------------
# RG-01: Diagnostic Journey Section Exists
# ---------------------------------------------------------------------------

class TestDiagnosticJourneyGeneration:
    """RG-01: Report includes diagnostic journey showing reasoning process."""

    def test_3_round_session_generates_narrative(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)

        # Should have content
        assert len(lines) > 5
        # Should have journey header
        assert any("诊断推理过程" in l for l in lines)

    def test_empty_session_returns_empty(self):
        session_data = _make_session_data(rounds=[], hypotheses=[])
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        assert len(lines) == 0

    def test_round_0_shows_initial_hypotheses(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        # Should mention initial hypotheses
        assert "初步假设" in text or "初步扫描" in text

    def test_confirmed_hypothesis_appears_in_conclusion(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        assert "手臂主导击球" in text
        assert "确认" in text

    def test_eliminated_hypothesis_appears(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        assert "排除" in text
        assert "拍头下压" in text

    def test_convergence_score_shown(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        assert "收敛度" in text
        assert "90%" in text


# ---------------------------------------------------------------------------
# RG-02: Narrative Style (not log dump)
# ---------------------------------------------------------------------------

class TestNarrativeStyle:
    """RG-02: Journey renders as readable narrative, not raw data."""

    def test_no_raw_json_in_output(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        # Should not contain raw JSON artifacts
        assert '{"' not in text
        assert "obs_r1_01" not in text  # No raw observation IDs
        assert "hyp_dc_" not in text  # No raw hypothesis IDs

    def test_observations_have_frame_references(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        # Observations should reference frames
        assert "图" in text

    def test_round_labels_are_descriptive(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        # Should use descriptive labels, not "Round 0", "Round 1"
        assert "初步扫描" in text
        assert "根因确认" in text


# ---------------------------------------------------------------------------
# RG-03: Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """RG-03: v1.0 vlm_result without diagnostic_session works unchanged."""

    def test_v1_result_no_journey(self):
        """v1.0 result without diagnostic_session -> no journey section."""
        v1_result = {
            "root_cause_tree": {
                "root_cause": "手臂脱离",
                "downstream_symptoms": [],
                "fix": {"one_drill": "Do shadow swings"},
            },
            "overall_narrative": "Test narrative",
        }
        lines = ReportGenerator._vlm_section(v1_result, {}, 0, 1)
        text = "\n".join(lines)
        # Should have root cause tree but NOT journey
        assert "根因诊断" in text
        assert "诊断推理过程" not in text

    def test_v2_result_has_both_journey_and_tree(self):
        """v2.0 result with diagnostic_session -> journey + root_cause_tree."""
        v2_result = {
            "diagnostic_session": _make_3_round_session(),
            "root_cause_tree": {
                "root_cause": "手臂脱离",
                "downstream_symptoms": [],
                "fix": {"one_drill": "Do shadow swings"},
            },
            "overall_narrative": "Test narrative",
        }
        lines = ReportGenerator._vlm_section(v2_result, {}, 0, 1)
        text = "\n".join(lines)
        # Should have BOTH
        assert "诊断推理过程" in text
        assert "根因诊断" in text

    def test_v2_result_journey_suppressed(self):
        """include_journey=False suppresses the journey section."""
        v2_result = {
            "diagnostic_session": _make_3_round_session(),
            "root_cause_tree": {
                "root_cause": "手臂脱离",
                "downstream_symptoms": [],
                "fix": {"one_drill": "Do shadow swings"},
            },
            "overall_narrative": "Test narrative",
        }
        lines = ReportGenerator._vlm_section(v2_result, {}, 0, 1, include_journey=False)
        text = "\n".join(lines)
        # Journey suppressed, tree still present
        assert "诊断推理过程" not in text
        assert "根因诊断" in text

    def test_legacy_format_still_works(self):
        """v1.0 result with issues list (no root_cause_tree) still works."""
        legacy_result = {
            "issues": [
                {"name": "手臂主导", "severity": "高", "frame": "图3"},
            ],
            "overall_assessment": "需要改善",
        }
        lines = ReportGenerator._vlm_section(legacy_result, {}, 0, 1)
        text = "\n".join(lines)
        assert "手臂主导" in text
        assert "诊断推理过程" not in text


# ---------------------------------------------------------------------------
# RG-04: Optional / Collapsible
# ---------------------------------------------------------------------------

class TestJourneyOptional:
    """RG-04: Journey section can be hidden."""

    def test_include_journey_default_true(self):
        """Default behavior includes journey when session data present."""
        v2_result = {
            "diagnostic_session": _make_3_round_session(),
        }
        lines = ReportGenerator._vlm_section(v2_result, {}, 0, 1)
        text = "\n".join(lines)
        assert "诊断推理过程" in text

    def test_include_journey_false(self):
        """Explicitly setting include_journey=False hides it."""
        v2_result = {
            "diagnostic_session": _make_3_round_session(),
        }
        lines = ReportGenerator._vlm_section(v2_result, {}, 0, 1, include_journey=False)
        text = "\n".join(lines)
        assert "诊断推理过程" not in text

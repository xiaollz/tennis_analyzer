"""Tests for diagnostic journey formatting and backward compatibility.

Covers:
- RG-01: Diagnostic journey section generation
- RG-02: Multi-round compression
- RG-03: Backward compatibility with v1/v2 results
"""

import pytest
from report.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_data(rounds=None, hypotheses=None):
    return {
        "rounds": rounds or [],
        "hypotheses": hypotheses or [],
    }


def _make_3_round_session():
    """Create a realistic 3-round diagnostic session."""
    return _make_session_data(
        rounds=[
            {
                "round_index": 0,
                "observations": [
                    {"region": "图2-3", "judgment": "support",
                     "description": "手臂先于躯干启动"},
                ],
                "hypothesis_updates": [
                    {"hypothesis": "arm_dominant", "action": "raise_confidence",
                     "reason": "手臂抢先动作明显"},
                ],
            },
            {
                "round_index": 1,
                "observations": [
                    {"region": "图3", "judgment": "contradict",
                     "description": "拍头下压角度正常"},
                ],
                "hypothesis_updates": [
                    {"hypothesis": "excessive_pat", "action": "eliminate",
                     "reason": "拍头下坠在正常范围"},
                ],
            },
            {
                "round_index": 2,
                "observations": [
                    {"region": "图4-5", "judgment": "support",
                     "description": "肘部空间不足"},
                ],
                "hypothesis_updates": [
                    {"hypothesis": "arm_dominant", "action": "confirm",
                     "reason": "手臂脱离身体旋转系统，肘部紧贴"},
                ],
            },
        ],
        hypotheses=[
            {"name": "arm_dominant", "name_zh": "手臂主导击球",
             "status": "confirmed", "confidence": 0.92},
            {"name": "excessive_pat", "name_zh": "拍头下压",
             "status": "eliminated", "confidence": 0.10},
        ],
    )


# ---------------------------------------------------------------------------
# RG-01: Diagnostic Journey Section Exists
# ---------------------------------------------------------------------------

class TestDiagnosticJourneyGeneration:
    """RG-01: Report includes diagnostic journey showing reasoning process."""

    def test_3_round_session_generates_narrative(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)

        # Should have content (compressed: header + narrative + blank)
        assert len(lines) >= 2
        # Should have journey header
        assert any("诊断路径" in l for l in lines)

    def test_empty_session_returns_empty(self):
        session_data = _make_session_data(rounds=[], hypotheses=[])
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        assert len(lines) == 0

    def test_compressed_journey_mentions_reasoning(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        # Compressed format should mention the number of rounds
        assert "轮观察" in text or "诊断路径" in text

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

    def test_compressed_journey_is_concise(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        # Compressed format should be no more than 5 lines total
        assert len(lines) <= 5


# ---------------------------------------------------------------------------
# RG-02: Multi-round Compression
# ---------------------------------------------------------------------------

class TestMultiRoundCompression:
    """RG-02: Multi-round session is compressed to 1-2 sentence summary."""

    def test_single_round_no_prefix(self):
        session_data = _make_session_data(
            rounds=[{
                "round_index": 0,
                "observations": [],
                "hypothesis_updates": [
                    {"hypothesis": "test", "action": "confirm", "reason": "test"},
                ],
            }],
            hypotheses=[
                {"name": "test", "name_zh": "测试", "status": "confirmed"},
            ],
        )
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        # Single round should not have "N轮观察" prefix
        assert "轮观察" not in text

    def test_multi_round_has_count(self):
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        assert "3 轮观察" in text


# ---------------------------------------------------------------------------
# RG-03: Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """RG-03: Various vlm_result formats work with v5 report generator."""

    def test_v1_result_shows_root_cause(self):
        """v1.0 result without diagnostic_session -> shows root cause."""
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
        # v5 format: root cause shown inline
        assert "手臂脱离" in text
        assert "诊断路径" not in text

    def test_v2_result_no_journey_in_report(self):
        """v2.0 result with diagnostic_session -> journey NOT shown in v5 report."""
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
        # v5: journey is NOT shown in report (removed by design)
        assert "手臂脱离" in text
        assert "诊断路径" not in text

    def test_diagnostic_journey_method_still_works(self):
        """_format_diagnostic_journey still available for internal use."""
        session_data = _make_3_round_session()
        lines = ReportGenerator._format_diagnostic_journey(session_data)
        text = "\n".join(lines)
        assert "诊断路径" in text
        assert "手臂主导击球" in text

    def test_legacy_format_still_renders(self):
        """v1.0 result with issues list (no root_cause_tree) still renders."""
        legacy_result = {
            "issues": [
                {"name": "手臂主导", "severity": "高", "frame": "图3"},
            ],
            "overall_assessment": "需要改善",
        }
        lines = ReportGenerator._vlm_section(legacy_result, {}, 0, 1)
        # Should not crash; may produce minimal output
        assert isinstance(lines, list)

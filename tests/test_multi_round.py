"""Tests for MultiRoundAnalyzer orchestrator and convergence detection."""

import json
from unittest.mock import MagicMock, patch

import pytest

from knowledge.schemas import (
    DiagnosticSession,
    Hypothesis,
    HypothesisStatus,
    HypothesisUpdate,
    Observation,
    ObservationJudgment,
    RoundResult,
)


# ---------------------------------------------------------------------------
# Helpers: build mock analyzer + compiler
# ---------------------------------------------------------------------------


def _make_mock_compiler(chain_ids=None):
    """Create a mock VLMPromptCompiler with compile_pass1/pass2_prompt methods."""
    compiler = MagicMock()
    compiler.compile_pass1_prompt.return_value = "Pass 1 symptom checklist prompt"
    compiler.compile_pass2_prompt.return_value = "Pass 2 diagnostic prompt"
    compiler.compile_system_prompt.return_value = "System prompt"

    # Build chain_map with mock DiagnosticChain objects
    chain_ids = chain_ids or ["dc_arm_driven", "dc_scooping"]
    chains = []
    for cid in chain_ids:
        chain = MagicMock()
        chain.id = cid
        chain.symptom = f"Symptom for {cid}"
        chain.symptom_zh = f"症状 {cid}"
        chain.symptom_concept_id = f"symptom_{cid}"
        chain.root_causes = [f"root_{cid}"]
        chain.check_sequence = [
            MagicMock(check="Check step 1", check_zh="检查1", if_true=f"root_{cid}", if_false=None)
        ]
        chain.priority = 1
        chains.append(chain)

    compiler.chains = chains
    compiler.chain_map = {c.id: c for c in chains}
    return compiler


def _make_mock_analyzer(chain_ids=None, pass1_response="1, 2"):
    """Create a mock VLMForehandAnalyzer."""
    analyzer = MagicMock()
    analyzer.compiler = _make_mock_compiler(chain_ids)
    analyzer.two_pass_enabled = True
    analyzer._chain_id_by_number = {}

    chain_ids = chain_ids or ["dc_arm_driven", "dc_scooping"]
    for i, cid in enumerate(chain_ids, 1):
        analyzer._chain_id_by_number[i] = cid

    # Default _call_vlm returns pass1 response then diagnostic JSON
    call_count = [0]

    def mock_call_vlm(image_b64, user_text, system_prompt=None):
        call_count[0] += 1
        if call_count[0] == 1:
            # Round 0 (pass1 symptom scan)
            return pass1_response
        # Subsequent rounds: return observation JSON
        return json.dumps({
            "observations": [
                {
                    "id": f"obs_r{call_count[0] - 1}_01",
                    "round_number": call_count[0] - 1,
                    "frame": "图3",
                    "description": "Arm leads body rotation",
                    "judgment": "yes",
                    "confidence": 0.85,
                    "directive_source": "check_step_0",
                }
            ],
            "hypothesis_updates": [
                {
                    "hypothesis_id": f"hyp_{chain_ids[0]}",
                    "action": "confirm",
                    "reason": "Observation supports hypothesis",
                }
            ],
        })

    analyzer._call_vlm = MagicMock(side_effect=mock_call_vlm)
    analyzer._parse_symptom_response.return_value = chain_ids
    return analyzer


# ---------------------------------------------------------------------------
# Convergence tests (unit tests on _check_convergence)
# ---------------------------------------------------------------------------


class TestConvergenceCheck:
    """Test convergence detection logic independently."""

    def _make_orchestrator_with_session(self, hypotheses, rounds=None):
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        analyzer = _make_mock_analyzer()
        mra = MultiRoundAnalyzer(analyzer, max_rounds=4)
        mra.session = DiagnosticSession(
            session_id="test_sess",
            hypotheses=hypotheses,
            rounds=rounds or [],
            max_rounds=4,
        )
        return mra

    def test_converge_when_top_confidence_ge_08(self):
        """Convergence returns True when top hypothesis confidence >= 0.8."""
        hyps = [
            Hypothesis(
                id="hyp_1", chain_id="dc_arm_driven", root_cause_concept_id="root_1",
                name="Arm driven", name_zh="手臂主导",
                status=HypothesisStatus.ACTIVE, confidence=0.85,
                round_introduced=0,
            ),
            Hypothesis(
                id="hyp_2", chain_id="dc_scooping", root_cause_concept_id="root_2",
                name="Scooping", name_zh="捞球",
                status=HypothesisStatus.ACTIVE, confidence=0.3,
                round_introduced=0,
            ),
        ]
        mra = self._make_orchestrator_with_session(hyps)
        assert mra._check_convergence() is True

    def test_converge_when_one_active_hypothesis_remains(self):
        """Convergence returns True when only 1 active hypothesis left."""
        hyps = [
            Hypothesis(
                id="hyp_1", chain_id="dc_arm_driven", root_cause_concept_id="root_1",
                name="Arm driven", name_zh="手臂主导",
                status=HypothesisStatus.ACTIVE, confidence=0.6,
                round_introduced=0,
            ),
            Hypothesis(
                id="hyp_2", chain_id="dc_scooping", root_cause_concept_id="root_2",
                name="Scooping", name_zh="捞球",
                status=HypothesisStatus.ELIMINATED, confidence=0.0,
                round_introduced=0, round_resolved=1,
            ),
        ]
        mra = self._make_orchestrator_with_session(hyps)
        assert mra._check_convergence() is True

    def test_converge_when_no_change_for_2_rounds(self):
        """Convergence returns True when hypothesis statuses unchanged for 2 rounds."""
        hyps = [
            Hypothesis(
                id="hyp_1", chain_id="dc_arm_driven", root_cause_concept_id="root_1",
                name="Arm driven", name_zh="手臂主导",
                status=HypothesisStatus.ACTIVE, confidence=0.6,
                round_introduced=0,
            ),
            Hypothesis(
                id="hyp_2", chain_id="dc_scooping", root_cause_concept_id="root_2",
                name="Scooping", name_zh="捞球",
                status=HypothesisStatus.ACTIVE, confidence=0.5,
                round_introduced=0,
            ),
        ]
        # 3 rounds with no status changes
        rounds = [
            RoundResult(round_number=0, prompt_sent="p", raw_response="r"),
            RoundResult(round_number=1, prompt_sent="p", raw_response="r"),
            RoundResult(round_number=2, prompt_sent="p", raw_response="r"),
        ]
        mra = self._make_orchestrator_with_session(hyps, rounds=rounds)
        # Need at least 3 rounds to detect "no change for 2 consecutive rounds"
        # We store status snapshots during run; for this unit test, simulate
        # by setting _status_snapshots
        mra._status_snapshots = [
            {("hyp_1", "active"), ("hyp_2", "active")},  # after round 0
            {("hyp_1", "active"), ("hyp_2", "active")},  # after round 1
            {("hyp_1", "active"), ("hyp_2", "active")},  # after round 2
        ]
        assert mra._check_convergence() is True

    def test_no_converge_when_multiple_active_below_08(self):
        """Convergence returns False when 2+ active hypotheses all below 0.8."""
        hyps = [
            Hypothesis(
                id="hyp_1", chain_id="dc_arm_driven", root_cause_concept_id="root_1",
                name="Arm driven", name_zh="手臂主导",
                status=HypothesisStatus.ACTIVE, confidence=0.6,
                round_introduced=0,
            ),
            Hypothesis(
                id="hyp_2", chain_id="dc_scooping", root_cause_concept_id="root_2",
                name="Scooping", name_zh="捞球",
                status=HypothesisStatus.ACTIVE, confidence=0.5,
                round_introduced=0,
            ),
        ]
        mra = self._make_orchestrator_with_session(hyps)
        # Only 1 round snapshot => can't detect 2-round stagnation
        mra._status_snapshots = [
            {("hyp_1", "active"), ("hyp_2", "active")},
        ]
        assert mra._check_convergence() is False


# ---------------------------------------------------------------------------
# Full run() integration tests (with mocked VLM)
# ---------------------------------------------------------------------------


class TestMultiRoundRun:
    """Test the full run() loop with mocked VLM calls."""

    def test_run_exits_after_max_rounds(self):
        """run() exits after max_rounds=4 even if not converged."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        # Set up mock that never converges (all adjusts, no confirms)
        chain_ids = ["dc_arm_driven", "dc_scooping"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)

        call_count = [0]

        def never_converge_vlm(image_b64, user_text, system_prompt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "1, 2"  # pass1 symptom scan
            return json.dumps({
                "observations": [],
                "hypothesis_updates": [
                    {"hypothesis_id": "hyp_dc_arm_driven", "action": "adjust", "reason": "Need more data"},
                ],
            })

        analyzer._call_vlm = MagicMock(side_effect=never_converge_vlm)

        mra = MultiRoundAnalyzer(analyzer, max_rounds=4)
        session = mra.run("fake_b64_image")

        assert isinstance(session, DiagnosticSession)
        # Round 0 + up to 4 diagnostic rounds = at most 5 rounds total
        # But max_rounds=4 means at most 4 diagnostic rounds (1-4) + round 0
        assert len(session.rounds) <= 5
        assert session.final_result is not None

    def test_run_creates_initial_hypotheses_from_pass1(self):
        """run() creates hypotheses from Pass 1 detected chain IDs."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven", "dc_scooping"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)
        mra = MultiRoundAnalyzer(analyzer, max_rounds=1)  # 1 round to exit quickly
        session = mra.run("fake_b64_image")

        assert len(session.hypotheses) == 2
        hyp_ids = {h.chain_id for h in session.hypotheses}
        assert "dc_arm_driven" in hyp_ids
        assert "dc_scooping" in hyp_ids

    def test_run_calls_vlm_once_per_round(self):
        """run() calls _call_vlm once per round (1 for round 0, 1 per diagnostic round)."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)
        mra = MultiRoundAnalyzer(analyzer, max_rounds=1)
        session = mra.run("fake_b64_image")

        # Round 0 (pass1) + round 1 (diagnostic, confirms -> exits)
        assert analyzer._call_vlm.call_count >= 2

    def test_round_0_uses_compile_pass1_prompt(self):
        """Round 0 calls compile_pass1_prompt (same as existing Pass 1)."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)
        mra = MultiRoundAnalyzer(analyzer, max_rounds=1)
        mra.run("fake_b64_image")

        analyzer.compiler.compile_pass1_prompt.assert_called_once()

    def test_session_rounds_correct_length(self):
        """session.rounds has correct length after run completes."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)
        mra = MultiRoundAnalyzer(analyzer, max_rounds=2)
        session = mra.run("fake_b64_image")

        # At least round 0 + 1 diagnostic round
        assert len(session.rounds) >= 2

    def test_hypothesis_confidence_increases_on_confirm(self):
        """Hypothesis confidence >= 0.8 when action='confirm'."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)
        mra = MultiRoundAnalyzer(analyzer, max_rounds=2)
        session = mra.run("fake_b64_image")

        # The mock sends confirm for dc_arm_driven
        confirmed = [h for h in session.hypotheses if h.chain_id == "dc_arm_driven"]
        assert len(confirmed) == 1
        assert confirmed[0].confidence >= 0.8

    def test_hypothesis_eliminated_on_eliminate_action(self):
        """Hypothesis status changes to eliminated when action='eliminate'."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven", "dc_scooping"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)

        call_count = [0]

        def eliminate_vlm(image_b64, user_text, system_prompt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "1, 2"
            return json.dumps({
                "observations": [],
                "hypothesis_updates": [
                    {"hypothesis_id": "hyp_dc_scooping", "action": "eliminate", "reason": "No evidence"},
                    {"hypothesis_id": "hyp_dc_arm_driven", "action": "confirm", "reason": "Strong evidence"},
                ],
            })

        analyzer._call_vlm = MagicMock(side_effect=eliminate_vlm)

        mra = MultiRoundAnalyzer(analyzer, max_rounds=2)
        session = mra.run("fake_b64_image")

        scooping = [h for h in session.hypotheses if h.chain_id == "dc_scooping"]
        assert len(scooping) == 1
        assert scooping[0].status == HypothesisStatus.ELIMINATED
        assert scooping[0].confidence == 0.0

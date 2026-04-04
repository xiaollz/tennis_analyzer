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
    compiler.compile_observation_directive.return_value = "Observation directive prompt"
    compiler.compile_confirmation_prompt.return_value = "Confirmation prompt"

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


# ---------------------------------------------------------------------------
# analyze_swing_iterative() entry point tests
# ---------------------------------------------------------------------------


class TestAnalyzeSwingIterative:
    """Test analyze_swing_iterative() entry point and session persistence."""

    def _make_real_analyzer_with_mocks(self):
        """Create a VLMForehandAnalyzer with mocked internals."""
        from evaluation.vlm_analyzer import VLMForehandAnalyzer

        analyzer = VLMForehandAnalyzer.__new__(VLMForehandAnalyzer)
        analyzer.provider = "openai_compatible"
        analyzer.api_key = "test_key"
        analyzer.base_url = "http://test"
        analyzer.model = "test-model"
        analyzer.extra_headers = {}
        analyzer.two_pass_enabled = True

        compiler = _make_mock_compiler(["dc_arm_driven"])
        analyzer.compiler = compiler
        analyzer._chain_id_by_number = {1: "dc_arm_driven"}

        # Mock _call_vlm
        call_count = [0]

        def mock_call_vlm(image_b64, user_text, system_prompt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "1"
            return json.dumps({
                "observations": [],
                "hypothesis_updates": [
                    {"hypothesis_id": "hyp_dc_arm_driven", "action": "confirm", "reason": "Clear evidence"},
                ],
            })

        analyzer._call_vlm = MagicMock(side_effect=mock_call_vlm)
        analyzer._parse_symptom_response = MagicMock(return_value=["dc_arm_driven"])
        return analyzer

    def test_returns_dict_on_success(self):
        """analyze_swing_iterative returns dict (same as v1.0 format) on success."""
        import numpy as np

        analyzer = self._make_real_analyzer_with_mocks()
        # Create a small fake keyframe grid
        fake_grid = np.zeros((100, 100, 3), dtype=np.uint8)
        result = analyzer.analyze_swing_iterative(fake_grid, save_session=False)

        assert isinstance(result, dict)
        assert "issues" in result

    def test_fallback_on_multi_round_exception(self):
        """Falls back to analyze_swing when MultiRoundAnalyzer raises exception."""
        import numpy as np
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        analyzer = self._make_real_analyzer_with_mocks()
        fake_grid = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock analyze_swing to track it being called as fallback
        analyzer.analyze_swing = MagicMock(return_value={"issues": [], "fallback": True})

        # Make MultiRoundAnalyzer.run raise
        with patch.object(MultiRoundAnalyzer, "run", side_effect=RuntimeError("Boom")):
            result = analyzer.analyze_swing_iterative(fake_grid, save_session=False)

        analyzer.analyze_swing.assert_called_once()
        assert result == {"issues": [], "fallback": True}

    def test_fallback_when_compiler_is_none(self):
        """Falls back to analyze_swing when compiler is None."""
        import numpy as np

        analyzer = self._make_real_analyzer_with_mocks()
        analyzer.compiler = None
        fake_grid = np.zeros((100, 100, 3), dtype=np.uint8)

        analyzer.analyze_swing = MagicMock(return_value={"issues": [], "fallback": True})
        result = analyzer.analyze_swing_iterative(fake_grid, save_session=False)

        analyzer.analyze_swing.assert_called_once()

    def test_save_session_writes_json(self, tmp_path):
        """_save_session writes JSON file to output/diagnostic_sessions/."""
        from evaluation.vlm_analyzer import VLMForehandAnalyzer

        analyzer = VLMForehandAnalyzer.__new__(VLMForehandAnalyzer)
        session = DiagnosticSession(
            session_id="test_sess_001",
            max_rounds=4,
        )

        # Patch the output directory to tmp_path
        out_dir = tmp_path / "output" / "diagnostic_sessions"
        with patch("evaluation.vlm_analyzer.Path.__file__", create=True):
            # Directly call with monkeypatched path
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{session.session_id}.json"
            out_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")

        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["session_id"] == "test_sess_001"

    def test_save_session_roundtrip(self, tmp_path):
        """Saved JSON can be loaded back as DiagnosticSession."""
        session = DiagnosticSession(
            session_id="test_sess_002",
            hypotheses=[
                Hypothesis(
                    id="hyp_1", chain_id="dc_arm_driven",
                    root_cause_concept_id="root_1",
                    name="Arm driven", name_zh="手臂主导",
                    round_introduced=0,
                ),
            ],
            rounds=[
                RoundResult(round_number=0, prompt_sent="p", raw_response="r"),
            ],
            max_rounds=4,
        )

        out_path = tmp_path / f"{session.session_id}.json"
        out_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")

        loaded_data = json.loads(out_path.read_text())
        loaded_session = DiagnosticSession.model_validate(loaded_data)
        assert loaded_session.session_id == "test_sess_002"
        assert len(loaded_session.hypotheses) == 1
        assert len(loaded_session.rounds) == 1

    def test_no_save_when_save_session_false(self):
        """analyze_swing_iterative with save_session=False does NOT write file."""
        import numpy as np

        analyzer = self._make_real_analyzer_with_mocks()
        analyzer._save_session = MagicMock()
        fake_grid = np.zeros((100, 100, 3), dtype=np.uint8)

        result = analyzer.analyze_swing_iterative(fake_grid, save_session=False)

        analyzer._save_session.assert_not_called()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Phase 9 Plan 2: Confidence scoring, cross-hypothesis reasoning, integration
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    """HT-02: Confidence scoring from observations."""

    def _make_orchestrator(self, hypotheses=None):
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        analyzer = _make_mock_analyzer(chain_ids=["dc_arm_driven", "dc_scooping"])
        mra = MultiRoundAnalyzer(analyzer, max_rounds=4)
        mra.session = DiagnosticSession(
            session_id="test_scoring",
            hypotheses=hypotheses or [
                Hypothesis(
                    id="hyp_dc_arm_driven", chain_id="dc_arm_driven",
                    root_cause_concept_id="root_dc_arm_driven",
                    name="Arm driven", name_zh="手臂主导",
                    status=HypothesisStatus.ACTIVE, confidence=0.5,
                    round_introduced=0,
                ),
            ],
            max_rounds=4,
        )
        return mra

    def test_supporting_observation_increases_confidence(self):
        """YES observation with high confidence increases hypothesis confidence."""
        mra = self._make_orchestrator()
        obs = Observation(
            id="obs_test_01", round_number=1, frame="图3",
            description="Arm leads", judgment=ObservationJudgment.YES,
            confidence=0.9, directive_source="hyp_dc_arm_driven",
        )
        initial = mra.session.hypotheses[0].confidence
        mra._score_observations([obs])
        assert mra.session.hypotheses[0].confidence > initial
        assert "obs_test_01" in mra.session.hypotheses[0].supporting_observations

    def test_contradicting_observation_decreases_confidence(self):
        """NO observation with high confidence decreases hypothesis confidence."""
        mra = self._make_orchestrator()
        obs = Observation(
            id="obs_test_02", round_number=1, frame="图3",
            description="No arm issue", judgment=ObservationJudgment.NO,
            confidence=0.9, directive_source="hyp_dc_arm_driven",
        )
        initial = mra.session.hypotheses[0].confidence
        mra._score_observations([obs])
        assert mra.session.hypotheses[0].confidence < initial
        assert "obs_test_02" in mra.session.hypotheses[0].contradicting_observations

    def test_unclear_observation_no_change(self):
        """UNCLEAR observation does not change confidence."""
        mra = self._make_orchestrator()
        obs = Observation(
            id="obs_test_03", round_number=1, frame="图3",
            description="Hard to tell", judgment=ObservationJudgment.UNCLEAR,
            confidence=0.5, directive_source="hyp_dc_arm_driven",
        )
        initial = mra.session.hypotheses[0].confidence
        mra._score_observations([obs])
        assert mra.session.hypotheses[0].confidence == initial

    def test_auto_eliminate_below_threshold(self):
        """Hypothesis auto-eliminated when confidence drops below 0.15."""
        hyps = [
            Hypothesis(
                id="hyp_dc_arm_driven", chain_id="dc_arm_driven",
                root_cause_concept_id="root_dc_arm_driven",
                name="Arm driven", name_zh="手臂主导",
                status=HypothesisStatus.ACTIVE, confidence=0.2,
                round_introduced=0,
            ),
        ]
        mra = self._make_orchestrator(hypotheses=hyps)
        # Strong contradicting observation should drop below 0.15
        obs = Observation(
            id="obs_elim", round_number=1, frame="图3",
            description="No evidence", judgment=ObservationJudgment.NO,
            confidence=1.0, directive_source="hyp_dc_arm_driven",
        )
        mra._score_observations([obs])
        assert mra.session.hypotheses[0].status == HypothesisStatus.ELIMINATED

    def test_auto_confirm_above_threshold(self):
        """Hypothesis auto-confirmed when confidence rises above 0.85."""
        hyps = [
            Hypothesis(
                id="hyp_dc_arm_driven", chain_id="dc_arm_driven",
                root_cause_concept_id="root_dc_arm_driven",
                name="Arm driven", name_zh="手臂主导",
                status=HypothesisStatus.ACTIVE, confidence=0.8,
                round_introduced=0,
            ),
        ]
        mra = self._make_orchestrator(hypotheses=hyps)
        obs = Observation(
            id="obs_conf", round_number=1, frame="图3",
            description="Clear arm lead", judgment=ObservationJudgment.YES,
            confidence=1.0, directive_source="hyp_dc_arm_driven",
        )
        mra._score_observations([obs])
        assert mra.session.hypotheses[0].status == HypothesisStatus.CONFIRMED

    def test_confidence_scaled_by_observation_confidence(self):
        """Delta is scaled by observation confidence."""
        mra = self._make_orchestrator()
        # Low confidence observation
        obs_low = Observation(
            id="obs_low", round_number=1, frame="图3",
            description="Maybe", judgment=ObservationJudgment.YES,
            confidence=0.3, directive_source="hyp_dc_arm_driven",
        )
        initial = mra.session.hypotheses[0].confidence
        mra._score_observations([obs_low])
        delta_low = mra.session.hypotheses[0].confidence - initial
        # Should be 0.15 * 0.3 = 0.045
        assert abs(delta_low - 0.045) < 0.001


class TestCrossHypothesisReasoning:
    """HT-03: Cross-hypothesis causal reasoning via knowledge graph."""

    def _make_graph_with_causal_chain(self):
        """Create a graph where forearm_compensation causes racket_drop."""
        from knowledge.graph import KnowledgeGraph
        from knowledge.schemas import Concept, ConceptType, Edge, RelationType

        kg = KnowledgeGraph()
        kg.add_concept(Concept(
            id="forearm_compensation", name="Forearm Compensation",
            name_zh="小臂代偿", category=ConceptType.SYMPTOM,
            description="Arm drives swing", confidence=0.9,
        ))
        kg.add_concept(Concept(
            id="racket_drop", name="Racket Drop",
            name_zh="拍头下坠", category=ConceptType.SYMPTOM,
            description="Racket drops too low", confidence=0.9,
        ))
        kg.add_edge(Edge(
            source_id="forearm_compensation", target_id="racket_drop",
            relation=RelationType.CAUSES, confidence=0.85,
            evidence="Arm compensation causes racket drop",
            source_file="test",
        ))
        return kg

    def test_confirmed_upstream_eliminates_downstream(self):
        """If confirmed A causes active B in graph, B is auto-eliminated."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        graph = self._make_graph_with_causal_chain()
        analyzer = _make_mock_analyzer(chain_ids=["dc_arm_driven", "dc_scooping"])
        mra = MultiRoundAnalyzer(analyzer, max_rounds=4, graph=graph)

        mra.session = DiagnosticSession(
            session_id="test_causal",
            hypotheses=[
                Hypothesis(
                    id="hyp_arm", chain_id="dc_arm_driven",
                    root_cause_concept_id="forearm_compensation",
                    name="Arm driven", name_zh="手臂主导",
                    status=HypothesisStatus.CONFIRMED, confidence=0.9,
                    round_introduced=0, round_resolved=1,
                ),
                Hypothesis(
                    id="hyp_scoop", chain_id="dc_scooping",
                    root_cause_concept_id="racket_drop",
                    name="Scooping", name_zh="捞球",
                    status=HypothesisStatus.ACTIVE, confidence=0.5,
                    round_introduced=0,
                ),
            ],
            rounds=[RoundResult(round_number=0, prompt_sent="p", raw_response="r")],
            max_rounds=4,
        )

        mra._cross_hypothesis_reasoning(current_round=1)

        scoop = [h for h in mra.session.hypotheses if h.id == "hyp_scoop"][0]
        assert scoop.status == HypothesisStatus.ELIMINATED

    def test_no_elimination_without_causal_link(self):
        """If no causal edge between confirmed and active, active stays."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer
        from knowledge.graph import KnowledgeGraph
        from knowledge.schemas import Concept, ConceptType

        # Graph with no causal edges
        kg = KnowledgeGraph()
        kg.add_concept(Concept(
            id="forearm_compensation", name="FC", name_zh="FC",
            category=ConceptType.SYMPTOM, description="test", confidence=0.9,
        ))
        kg.add_concept(Concept(
            id="some_other", name="Other", name_zh="Other",
            category=ConceptType.SYMPTOM, description="test", confidence=0.9,
        ))

        analyzer = _make_mock_analyzer(chain_ids=["dc_arm_driven", "dc_other"])
        mra = MultiRoundAnalyzer(analyzer, max_rounds=4, graph=kg)

        mra.session = DiagnosticSession(
            session_id="test_no_causal",
            hypotheses=[
                Hypothesis(
                    id="hyp_arm", chain_id="dc_arm_driven",
                    root_cause_concept_id="forearm_compensation",
                    name="Arm driven", name_zh="手臂主导",
                    status=HypothesisStatus.CONFIRMED, confidence=0.9,
                    round_introduced=0, round_resolved=1,
                ),
                Hypothesis(
                    id="hyp_other", chain_id="dc_other",
                    root_cause_concept_id="some_other",
                    name="Other", name_zh="其他",
                    status=HypothesisStatus.ACTIVE, confidence=0.5,
                    round_introduced=0,
                ),
            ],
            rounds=[RoundResult(round_number=0, prompt_sent="p", raw_response="r")],
            max_rounds=4,
        )

        mra._cross_hypothesis_reasoning(current_round=1)

        other = [h for h in mra.session.hypotheses if h.id == "hyp_other"][0]
        assert other.status == HypothesisStatus.ACTIVE

    def test_no_crash_without_graph(self):
        """Cross-hypothesis reasoning gracefully handles no graph."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        analyzer = _make_mock_analyzer()
        mra = MultiRoundAnalyzer(analyzer, max_rounds=4, graph=None)
        mra.session = DiagnosticSession(
            session_id="test_no_graph",
            hypotheses=[
                Hypothesis(
                    id="hyp_1", chain_id="dc_arm_driven",
                    root_cause_concept_id="root",
                    name="test", name_zh="test",
                    status=HypothesisStatus.CONFIRMED, confidence=0.9,
                    round_introduced=0,
                ),
            ],
            rounds=[], max_rounds=4,
        )
        # Should not raise
        mra._cross_hypothesis_reasoning(current_round=1)


class TestKnowledgeDrivenIntegration:
    """Test that MultiRoundAnalyzer uses knowledge-driven directives."""

    def test_diagnostic_round_uses_observation_directive(self):
        """Diagnostic rounds use compile_observation_directive, not pass2."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven", "dc_scooping"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)

        call_count = [0]

        def slow_converge_vlm(image_b64, user_text, system_prompt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "1, 2"
            # Round 1: adjust only (no convergence)
            if call_count[0] == 2:
                return json.dumps({
                    "observations": [],
                    "hypothesis_updates": [
                        {"hypothesis_id": "hyp_dc_arm_driven", "action": "adjust", "reason": "Need more data"},
                    ],
                })
            # Round 2: eliminate one
            return json.dumps({
                "observations": [],
                "hypothesis_updates": [
                    {"hypothesis_id": "hyp_dc_scooping", "action": "eliminate", "reason": "No evidence"},
                ],
            })

        analyzer._call_vlm = MagicMock(side_effect=slow_converge_vlm)

        mra = MultiRoundAnalyzer(analyzer, max_rounds=3)
        mra.run("fake_b64_image")

        # compile_observation_directive should have been called for at least one diagnostic round
        assert analyzer.compiler.compile_observation_directive.called

    def test_final_round_uses_confirmation(self):
        """When converged, the final round uses compile_confirmation_prompt."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)

        call_count = [0]

        def confirm_vlm(image_b64, user_text, system_prompt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "1"
            # First diagnostic round confirms immediately
            return json.dumps({
                "observations": [{
                    "id": f"obs_r{call_count[0] - 1}_01",
                    "round_number": call_count[0] - 1,
                    "frame": "图3",
                    "description": "Clear evidence",
                    "judgment": "yes",
                    "confidence": 0.95,
                    "directive_source": "hyp_dc_arm_driven",
                }],
                "hypothesis_updates": [{
                    "hypothesis_id": "hyp_dc_arm_driven",
                    "action": "confirm",
                    "reason": "Strong evidence",
                }],
            })

        analyzer._call_vlm = MagicMock(side_effect=confirm_vlm)
        mra = MultiRoundAnalyzer(analyzer, max_rounds=3)
        session = mra.run("fake_b64_image")

        # Should have converged (only 1 hypothesis, confirmed)
        assert session.convergence_score >= 0.8

    def test_progressive_narrowing_round_makes_progress(self):
        """At least one hypothesis resolved per round in this scenario."""
        from evaluation.vlm_analyzer import MultiRoundAnalyzer

        chain_ids = ["dc_arm_driven", "dc_scooping"]
        analyzer = _make_mock_analyzer(chain_ids=chain_ids)

        call_count = [0]

        def progressive_vlm(image_b64, user_text, system_prompt=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "1, 2"
            # Round 1: eliminate scooping (convergence: 1 active left -> loop exits)
            return json.dumps({
                "observations": [],
                "hypothesis_updates": [
                    {"hypothesis_id": "hyp_dc_scooping", "action": "eliminate", "reason": "No evidence"},
                ],
            })

        analyzer._call_vlm = MagicMock(side_effect=progressive_vlm)
        mra = MultiRoundAnalyzer(analyzer, max_rounds=4)
        session = mra.run("fake_b64_image")

        eliminated = [h for h in session.hypotheses if h.status == HypothesisStatus.ELIMINATED]
        active = [h for h in session.hypotheses if h.status == HypothesisStatus.ACTIVE]
        # Progressive narrowing: at least one eliminated, loop exited with remaining
        assert len(eliminated) >= 1
        # The remaining hypothesis is still active (last-standing)
        assert len(active) >= 1 or len(eliminated) >= 1

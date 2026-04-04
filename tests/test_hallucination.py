"""Tests for hallucination mitigation mechanisms (HM-01 through HM-04)."""

import pytest

from knowledge.schemas import (
    DiagnosticSession,
    Observation,
    ObservationJudgment,
)
from evaluation.hallucination_mitigation import (
    validate_anchoring,
    detect_contradictions,
    cross_validate_with_kinematics,
    collect_reobserve_candidates,
    build_reobserve_prompt_fragment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(
    obs_id: str = "obs_r1_01",
    round_number: int = 1,
    frame: str = "图3",
    description: str = "Elbow angle is wide at contact",
    judgment: str = "yes",
    confidence: float = 0.8,
    directive_source: str = "hyp_dc_arm_driven",
) -> Observation:
    return Observation(
        id=obs_id,
        round_number=round_number,
        frame=frame,
        description=description,
        judgment=ObservationJudgment(judgment),
        confidence=confidence,
        directive_source=directive_source,
    )


def _session(observations=None, contradictions=None) -> DiagnosticSession:
    return DiagnosticSession(
        session_id="test_sess",
        observations=observations or [],
        contradictions=contradictions or [],
    )


# ---------------------------------------------------------------------------
# HM-01: Observation Anchoring
# ---------------------------------------------------------------------------

class TestObservationAnchoring:
    """HM-01: Every observation must reference a frame and have a real description."""

    def test_well_anchored_observation_passes(self):
        obs = _obs(frame="图3", description="Elbow angle appears wide at about 130 degrees")
        unanchored = validate_anchoring([obs])
        assert len(unanchored) == 0
        assert obs.confidence == 0.8  # unchanged
        assert obs.is_anchored is True

    def test_no_frame_reference_flagged(self):
        obs = _obs(frame="", description="The swing looks rushed and incomplete")
        unanchored = validate_anchoring([obs])
        assert len(unanchored) == 1
        assert obs.confidence == 0.0
        assert obs.is_anchored is False

    def test_short_description_flagged(self):
        obs = _obs(frame="图2", description="yes")
        unanchored = validate_anchoring([obs])
        assert len(unanchored) == 1
        assert obs.confidence == 0.0
        assert obs.is_anchored is False

    def test_english_frame_reference_passes(self):
        obs = _obs(frame="Frame 4", description="Shoulder rotation is incomplete at this point")
        unanchored = validate_anchoring([obs])
        assert len(unanchored) == 0

    def test_invalid_frame_pattern_flagged(self):
        obs = _obs(frame="somewhere", description="The arm is doing something visible here")
        unanchored = validate_anchoring([obs])
        assert len(unanchored) == 1

    def test_multiple_observations_mixed(self):
        good = _obs(obs_id="good", frame="图1", description="Clear unit turn visible with full shoulder rotation")
        bad = _obs(obs_id="bad", frame="", description="hmm")
        unanchored = validate_anchoring([good, bad])
        assert len(unanchored) == 1
        assert unanchored[0].id == "bad"


# ---------------------------------------------------------------------------
# HM-02: Contradiction Detection
# ---------------------------------------------------------------------------

class TestContradictionDetection:
    """HM-02: Cross-round contradiction detection."""

    def test_same_frame_opposite_judgment_detected(self):
        """Scenario 1: Same frame, same feature, yes->no = contradiction."""
        prior = _obs(obs_id="obs_r1_01", round_number=1, frame="图3",
                     judgment="yes", directive_source="hyp_dc_scooping")
        new = _obs(obs_id="obs_r2_01", round_number=2, frame="图3",
                   judgment="no", directive_source="hyp_dc_scooping")

        session = _session(observations=[prior])
        contradictions = detect_contradictions(session, [new])

        assert len(contradictions) == 1
        assert contradictions[0]["obs_new"] == "obs_r2_01"
        assert contradictions[0]["obs_prior"] == "obs_r1_01"
        # Both confidences should be reduced
        assert prior.confidence <= 0.3
        assert new.confidence <= 0.3

    def test_different_frame_no_contradiction(self):
        """Scenario 2: Different frames should not trigger contradiction."""
        prior = _obs(obs_id="obs_r1_01", frame="图1", judgment="yes",
                     directive_source="hyp_dc_arm")
        new = _obs(obs_id="obs_r2_01", frame="图4", judgment="no",
                   directive_source="hyp_dc_arm")

        session = _session(observations=[prior])
        contradictions = detect_contradictions(session, [new])
        assert len(contradictions) == 0

    def test_same_frame_same_judgment_no_contradiction(self):
        """Same frame, same judgment = agreement, not contradiction."""
        prior = _obs(obs_id="obs_r1_01", frame="图3", judgment="yes",
                     directive_source="hyp_dc_scooping")
        new = _obs(obs_id="obs_r2_01", frame="图3", judgment="yes",
                   directive_source="hyp_dc_scooping")

        session = _session(observations=[prior])
        contradictions = detect_contradictions(session, [new])
        assert len(contradictions) == 0

    def test_triple_contradiction_all_flagged(self):
        """Scenario 3: Three observations on same feature, alternating judgments."""
        r1 = _obs(obs_id="obs_r1_01", round_number=1, frame="图2",
                  judgment="yes", directive_source="hyp_dc_scooping")
        r2 = _obs(obs_id="obs_r2_01", round_number=2, frame="图2",
                  judgment="no", directive_source="hyp_dc_scooping")

        session = _session(observations=[r1, r2])
        # R2 contradicts R1 (already recorded)
        # Now R3 contradicts R2
        r3 = _obs(obs_id="obs_r3_01", round_number=3, frame="图2",
                  judgment="yes", directive_source="hyp_dc_scooping")
        contradictions = detect_contradictions(session, [r3])
        # R3 contradicts R2 (no->yes flip)
        assert len(contradictions) >= 1


# ---------------------------------------------------------------------------
# HM-03: Quantitative Cross-Validation
# ---------------------------------------------------------------------------

class TestQuantitativeCrossValidation:
    """HM-03: VLM vs YOLO kinematic data cross-validation."""

    def test_elbow_wide_overridden_when_tight(self):
        """VLM says 'elbow wide' but YOLO measures 85 degrees -> override."""
        obs = _obs(description="Elbow angle is wide and extended at contact")
        metrics = {"elbow_angle": 85.0}

        overridden = cross_validate_with_kinematics([obs], metrics)
        assert len(overridden) == 1
        assert obs.judgment == ObservationJudgment.NO
        assert obs.override_reason is not None
        assert "85" in obs.override_reason

    def test_elbow_tight_overridden_when_wide(self):
        """VLM says 'elbow tight' but YOLO measures 140 degrees -> override."""
        obs = _obs(description="Elbow is tight and bent at impact")
        metrics = {"elbow_angle": 140.0}

        overridden = cross_validate_with_kinematics([obs], metrics)
        assert len(overridden) == 1
        assert obs.judgment == ObservationJudgment.NO

    def test_no_override_when_consistent(self):
        """VLM says 'elbow wide' and YOLO measures 135 degrees -> no override."""
        obs = _obs(description="Elbow angle is wide and extended")
        metrics = {"elbow_angle": 135.0}

        overridden = cross_validate_with_kinematics([obs], metrics)
        assert len(overridden) == 0
        assert obs.judgment == ObservationJudgment.YES  # unchanged

    def test_wrist_sharp_drop_overridden_when_smooth(self):
        """VLM says 'sharp wrist drop' but YOLO shows minimal drop."""
        obs = _obs(description="Wrist drops sharply in a V-shape pattern")
        metrics = {"wrist_below_elbow": 0.02}

        overridden = cross_validate_with_kinematics([obs], metrics)
        assert len(overridden) == 1
        assert obs.judgment == ObservationJudgment.NO

    def test_no_metrics_no_override(self):
        """No supplementary metrics -> no cross-validation."""
        obs = _obs(description="Elbow angle is wide")
        overridden = cross_validate_with_kinematics([obs], None)
        assert len(overridden) == 0

    def test_nested_metrics_extraction(self):
        """Metrics nested under phase key should still be found."""
        obs = _obs(description="Elbow angle is wide at contact")
        metrics = {"contact": {"elbow_angle": 80.0}}

        overridden = cross_validate_with_kinematics([obs], metrics)
        assert len(overridden) == 1

    def test_non_elbow_observation_not_overridden(self):
        """Observation about hip rotation shouldn't be affected by elbow data."""
        obs = _obs(description="Hip rotation appears incomplete in this frame")
        metrics = {"elbow_angle": 85.0}

        overridden = cross_validate_with_kinematics([obs], metrics)
        assert len(overridden) == 0


# ---------------------------------------------------------------------------
# HM-04: Re-observation Candidates
# ---------------------------------------------------------------------------

class TestReobserveCandidates:
    """HM-04: Low-confidence and flagged observations trigger re-observation."""

    def test_low_confidence_collected(self):
        obs = _obs(obs_id="low_conf", confidence=0.3)
        session = _session()
        candidates = collect_reobserve_candidates(session, [obs], confidence_threshold=0.5)
        assert "low_conf" in candidates

    def test_high_confidence_not_collected(self):
        obs = _obs(obs_id="high_conf", confidence=0.9)
        session = _session()
        candidates = collect_reobserve_candidates(session, [obs], confidence_threshold=0.5)
        assert "high_conf" not in candidates

    def test_unanchored_collected(self):
        obs = _obs(obs_id="unanchored", confidence=0.8)
        obs.is_anchored = False
        session = _session()
        candidates = collect_reobserve_candidates(session, [obs])
        assert "unanchored" in candidates

    def test_contradicted_collected(self):
        obs = _obs(obs_id="new_obs", confidence=0.8)
        session = _session(contradictions=[
            {"obs_new": "new_obs", "obs_prior": "old_obs", "frame": "3", "feature": "test"},
        ])
        candidates = collect_reobserve_candidates(session, [obs])
        assert "new_obs" in candidates
        assert "old_obs" in candidates

    def test_reobserve_prompt_fragment(self):
        obs = _obs(obs_id="re_obs", confidence=0.3, frame="图2",
                   description="Something about the elbow movement pattern")
        session = _session(observations=[obs])
        fragment = build_reobserve_prompt_fragment(session, ["re_obs"])
        assert "图2" in fragment
        assert "重新观察" in fragment
        assert len(fragment) > 0

    def test_empty_candidates_empty_fragment(self):
        session = _session()
        fragment = build_reobserve_prompt_fragment(session, [])
        assert fragment == ""


# ---------------------------------------------------------------------------
# Integration: All checks together
# ---------------------------------------------------------------------------

class TestHallucinationMitigationIntegration:
    """Integration test: all 4 checks work together on a realistic scenario."""

    def test_full_pipeline(self):
        """A round with mixed observations: some anchored, some not,
        one contradicting prior, one overridden by kinematics."""
        # Prior observation from round 1
        prior = _obs(
            obs_id="obs_r1_01", round_number=1, frame="图3",
            description="Elbow angle is wide and fully extended",
            judgment="yes", confidence=0.8,
            directive_source="hyp_dc_scooping",
        )
        session = _session(observations=[prior])
        session.supplementary_metrics = {"elbow_angle": 85.0}

        # New observations from round 2
        new_obs = [
            # Good observation
            _obs(obs_id="obs_r2_01", round_number=2, frame="图1",
                 description="Unit turn shows full shoulder rotation",
                 judgment="yes", confidence=0.9,
                 directive_source="hyp_dc_unit_turn"),
            # Unanchored (no frame)
            _obs(obs_id="obs_r2_02", round_number=2, frame="",
                 description="ok", judgment="yes", confidence=0.7,
                 directive_source="hyp_dc_arm"),
            # Contradicts prior on same frame
            _obs(obs_id="obs_r2_03", round_number=2, frame="图3",
                 description="Elbow appears tight and bent at contact",
                 judgment="no", confidence=0.7,
                 directive_source="hyp_dc_scooping"),
        ]

        # Step 1: Anchoring (HM-01)
        unanchored = validate_anchoring(new_obs)
        assert len(unanchored) == 1  # obs_r2_02
        assert new_obs[1].confidence == 0.0

        # Step 2: Kinematic cross-validation (HM-03)
        overridden = cross_validate_with_kinematics(new_obs, session.supplementary_metrics)
        # obs_r2_01 mentions unit turn (no elbow) -> not overridden
        # obs_r2_02 has 0 confidence -> mentions nothing relevant
        # obs_r2_03 says "tight" but YOLO says 85 (tight IS correct) -> no override
        # Actually, 85 < 90 so "tight" is correct -> no override
        assert len(overridden) == 0

        # Step 3: Contradiction detection (HM-02)
        contradictions = detect_contradictions(session, new_obs)
        # obs_r2_03 (no) contradicts obs_r1_01 (yes) on frame 图3, same directive
        assert len(contradictions) == 1

        # Step 4: Re-observation candidates (HM-04)
        session.contradictions = contradictions
        candidates = collect_reobserve_candidates(session, new_obs)
        # obs_r2_02 (unanchored, conf=0), obs_r2_03 (contradicted, conf<=0.3)
        # obs_r1_01 (contradicted prior)
        assert "obs_r2_02" in candidates
        assert "obs_r2_03" in candidates

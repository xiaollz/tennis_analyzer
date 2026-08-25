from evaluation.diagnosis_engine import (
    _build_quant_evidence,
    _build_quant_summary,
    _map_observations_to_concepts,
    _map_via_q_direct,
    _validate_with_metrics,
)


def _concept_ids(matches):
    return {match["mapped_concept"] for match in matches}


def test_negated_problem_terms_do_not_create_problem_concepts():
    observations = [
        {
            "text": "髋部先于胸部启动，胸部中性传力，无胸肌主动挤压鼓起，F10 PASS。",
            "frame": None,
            "field": "hip_vs_chest_power_source",
        },
        {
            "text": "自然落入 slot，未见 upright_lowering_no_separation 或 late_racket_head_collapse。",
            "frame": None,
            "field": "racket_hold_up",
        },
        {
            "text": "手臂与身体未见脱节现象。",
            "frame": None,
            "field": "asymmetry_origin",
        },
        {
            "text": "引拍到前挥一气呵成，无停顿。",
            "frame": None,
            "field": "place_then_pull",
        },
    ]

    concepts = _concept_ids(_map_observations_to_concepts(observations))

    assert "chest_active_trap_5_12" not in concepts
    assert "prep11b_no_racket_drop_separation" not in concepts
    assert "problem_p04" not in concepts
    assert "prep18_no_wait_after_prep" not in concepts


def test_real_active_chest_statement_still_maps_to_problem():
    observations = [
        {
            "text": "胸肌主动挤压鼓起，胸部发力抢在髋部之前，F10 FAIL。",
            "frame": None,
            "field": "hip_vs_chest_power_source",
        }
    ]

    concepts = _concept_ids(_map_observations_to_concepts(observations))

    assert "chest_active_trap_5_12" in concepts


def test_q_direct_ignores_negated_independent_arm_signal():
    matches = _map_via_q_direct(
        {
            "raw_answers": {
                "Q1": "F3 PASS：躯干与手臂同步，未见手臂独立抢跑。",
            }
        }
    )

    concepts = _concept_ids(matches)
    assert "arm_body_connected" in concepts
    assert "problem_p03" not in concepts


def test_zero_lag_correlation_does_not_claim_independent_arm():
    metrics = {"arm_torso_synchrony": -0.21}

    validation = _validate_with_metrics([], metrics)
    evidence = " ".join(_build_quant_evidence(metrics))
    summary = _build_quant_summary(metrics)

    assert not validation["confirmed"]
    assert "手臂独立" not in evidence
    assert "手臂独立" not in summary

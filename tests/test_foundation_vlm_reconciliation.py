from evaluation.foundation_layer import check_foundations


def _status(results, foundation_id):
    return next(item for item in results if item["id"] == foundation_id)


def test_f3_explicit_pass_is_not_overridden_by_zero_lag_metric():
    results = check_foundations(
        {"Q1": "F3 PASS：躯干传力带动手臂，无独立抢跑。"},
        {"arm_body_sync_score": -0.21},
    )

    f3 = _status(results, "F3_back_glue")
    assert f3["status"] == "pass"
    assert not f3["should_block_downstream"]


def test_f3_metric_without_video_evidence_is_uncertain():
    results = check_foundations({}, {"arm_body_sync_score": 0.1})

    f3 = _status(results, "F3_back_glue")
    assert f3["status"] == "uncertain"


def test_f4_explicit_pass_ignores_negated_failure_word():
    results = check_foundations(
        {
            "Q23": (
                "F4 PASS：非持拍侧主动引导整体 Unit Turn，"
                "非右肩独立拉拍。"
            ),
            "Q5b": "F4 PASS：肩髋保持整体转动。",
        },
        {},
    )

    f4 = _status(results, "F4_unit_action")
    assert f4["status"] == "pass"
    assert not f4["should_block_downstream"]


def test_explicit_fail_remains_a_failure():
    results = check_foundations(
        {"Q1": "F3 FAIL：手臂在躯干启动前独立抢跑。"},
        {},
    )

    f3 = _status(results, "F3_back_glue")
    assert f3["status"] == "fail"
    assert f3["should_block_downstream"]

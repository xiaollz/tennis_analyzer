from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_static_vlm_prompt_contains_push_throw_ratio_model():
    text = (ROOT / "knowledge/templates/vlm/system_prompt.md.j2").read_text()
    assert "Throw / Push / Through / Pulling ratio" in text
    assert "pulling_trap_hand_or_elbow" in text
    assert "dominant_push_control" in text
    assert "healthy_throw_through_blend" in text
    assert "shoulder_pulling_locked_arm" in text
    assert "forearm_supination_slot_missing" in text
    assert "Push 不是 Pulling 的同义词" in text


def test_hardcoded_vlm_prompt_contains_push_throw_ratio_model():
    text = (ROOT / "evaluation/vlm_analyzer.py").read_text()
    assert "Throw / Push / Through / Pulling ratio" in text
    assert "pulling_trap_hand_or_elbow" in text
    assert "dominant_push_control" in text
    assert "healthy_throw_through_blend" in text
    assert "shoulder_pulling_locked_arm" in text
    assert "forearm_supination_slot_missing" in text
    assert "Push 不是 Pulling 的同义词" in text


def test_static_vlm_prompt_contains_no_backswing_illusion_q51():
    text = (ROOT / "knowledge/templates/vlm/system_prompt.md.j2").read_text()
    assert "Q51:" in text
    assert "No Backswing Illusion" in text
    assert "upper_arm_independent_backswing" in text
    assert "upper_arm_connected_no_backswing" in text
    assert "大臂相对躯干独立后拉" in text


def test_hardcoded_vlm_prompt_contains_no_backswing_illusion_model():
    text = (ROOT / "evaluation/vlm_analyzer.py").read_text()
    assert "No Backswing Illusion" in text
    assert "upper_arm_independent_backswing" in text
    assert "upper_arm_connected_no_backswing" in text
    assert "大臂相对躯干独立后拉" in text

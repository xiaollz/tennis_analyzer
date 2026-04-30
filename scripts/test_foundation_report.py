"""Smoke test for the Foundation Layer report integration.

Synthesizes minimal vlm_results + a MultiSwingReport-shaped object,
runs ReportGenerator.generate(), prints the first 80 lines.

Exercises three scenarios:
  1. All 3 swings P0 fail (architecture violation)
  2. Mixed: 1 pass, 1 P0 fail, 1 VLM-unavailable (uncertain)
  3. All pass
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.diagnosis_engine import diagnose
from report.report_generator import ReportGenerator


def make_swing_eval(idx: int):
    return SimpleNamespace(
        swing_index=idx,
        overall_score=70.0,
        kpi_results=[],
        swing_event=SimpleNamespace(impact_frame=100 + idx * 30),
    )


def make_report(n_swings: int):
    return SimpleNamespace(
        total_swings=n_swings,
        average_score=70.0,
        swing_evaluations=[make_swing_eval(i) for i in range(n_swings)],
        impact_frames=[100 + i * 30 for i in range(n_swings)],
    )


def make_vlm_result_p0_fail():
    """A swing where F1 + F4 P0 foundations fail (with strong keyword hits)."""
    return {
        "raw_answers": {
            "Q1": "手臂先动，和身体不同步",
            "Q9": "拍头先下坠后上升，呈明显 V 形",
            "Q23": "右肩单独后拉，左肩没参与",
            "Q24": "拍头早垂，下沉到腰部",
            "Q26": "动作流畅",
            "Q5b": "肩髋同步，大约 30 度",
            "Q15": "重心保持后侧",
            "Q16": "右脚 pivot 明显",
            "Q31": "半开放站姿",
            "Q3":  "大臂与胸侧间距稳定",
            "Q25": "球衣有褶皱",
        },
        "issues": [],
        "score": 60,
    }


def make_vlm_result_pass():
    return {
        "raw_answers": {
            "Q1": "手臂与躯干基本同步，左肩有引导",
            "Q9": "拍头从高位平滑过渡，无下坠",
            "Q23": "左肩主动后推带动右肩",
            "Q24": "拍头维持在胸口高度",
            "Q26": "动作一气呵成，无停顿",
            "Q5b": "肩髋差约 30 度",
            "Q15": "重心在挥拍启动时同步转移",
            "Q16": "右脚 pivot 拧转保持承重",
            "Q31": "半开放站姿",
            "Q3":  "大臂与胸侧间距稳定不扩大",
            "Q25": "非持拍侧球衣有拉伸褶皱",
        },
        "issues": [],
        "score": 88,
    }


def make_vlm_result_unavailable():
    return {
        "raw_answers": {},
        "issues": [],
        "vlm_unavailable": True,
        "fallback": True,
    }


def good_metrics():
    return {
        "arm_torso_synchrony": 0.7,
        "scooping_depth": 0.0,
        "scooping_detected": False,
        "forward_extension": 0.6,
        "shoulder_rotation": 90.0,
        "swing_arc_ratio": 1.2,
        "min_knee_angle": 140.0,
    }


def bad_metrics():
    return {
        "arm_torso_synchrony": 0.2,
        "scooping_depth": 0.4,
        "scooping_detected": True,
        "forward_extension": 0.3,
        "shoulder_rotation": 80.0,
        "swing_arc_ratio": 0.8,
        "min_knee_angle": 165.0,
    }


def run_scenario(name: str, vlm_inputs, metric_inputs):
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {name}")
    print('=' * 60)

    diagnosed = []
    for vi, mi in zip(vlm_inputs, metric_inputs):
        diagnosed.append(diagnose(vi, mi))

    rep = make_report(len(diagnosed))

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator(output_dir=tmpdir)
        path = gen.generate(
            rep, video_name=f"test_{name}",
            chart_paths={},
            stroke_type="forehand",
            vlm_results=diagnosed,
        )
        text = Path(path).read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            print(f"  {i:3d} | {line}")
        print(f"  ... ({len(lines)} total lines)")


if __name__ == "__main__":
    run_scenario(
        "all_p0_fail",
        [make_vlm_result_p0_fail(), make_vlm_result_p0_fail(), make_vlm_result_p0_fail()],
        [bad_metrics(), bad_metrics(), bad_metrics()],
    )
    run_scenario(
        "mixed",
        [make_vlm_result_pass(), make_vlm_result_p0_fail(), make_vlm_result_unavailable()],
        [good_metrics(), bad_metrics(), {}],
    )
    run_scenario(
        "all_pass",
        [make_vlm_result_pass(), make_vlm_result_pass()],
        [good_metrics(), good_metrics()],
    )

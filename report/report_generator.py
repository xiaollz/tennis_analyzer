"""Markdown report generator for Modern Forehand evaluation.

Produces a structured, human-readable Markdown report from an
``EvaluationReport``, optionally embedding chart images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from evaluation.forehand_evaluator import EvaluationReport
from evaluation.kpi import KPIResult


class ReportGenerator:
    """Generate a Markdown evaluation report."""

    PHASE_TITLES = {
        "preparation": "Phase 1: Preparation & Unit Turn",
        "loading": "Phase 2: Loading & Racket Drop",
        "kinetic_chain": "Phase 3: Kinetic Chain & Forward Swing",
        "contact": "Phase 4: Contact Point",
        "extension": "Phase 5: Extension & Follow-Through",
        "balance": "Phase 6: Balance & Recovery",
    }

    PHASE_ORDER = ["preparation", "loading", "kinetic_chain", "contact", "extension", "balance"]

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report: EvaluationReport,
        video_name: str = "unknown",
        chart_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate the full Markdown report and return its file path."""
        chart_paths = chart_paths or {}
        lines: List[str] = []

        # ── Header ───────────────────────────────────────────────────
        lines.append("# Modern Forehand Analysis Report")
        lines.append("")
        lines.append(f"**Video**: {video_name}  ")
        lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
        lines.append(f"**Arm Style Detected**: {report.arm_style}  ")
        if report.swing_event.impact_frame is not None:
            lines.append(f"**Impact Frame**: {report.swing_event.impact_frame}  ")
        if report.swing_event.impact_event:
            lines.append(f"**Peak Wrist Speed**: {report.swing_event.impact_event.peak_speed_px_s:.0f} px/s  ")
        lines.append("")

        # ── Overall Score ────────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## Overall Score")
        lines.append("")
        score = report.overall_score
        grade = self._score_to_grade(score)
        lines.append(f"### {score:.0f} / 100  ({grade})")
        lines.append("")

        # Radar chart
        if "radar" in chart_paths:
            lines.append(f"![Phase Scores]({chart_paths['radar']})")
            lines.append("")

        # ── Phase Scores Summary Table ───────────────────────────────
        lines.append("## Phase Scores Summary")
        lines.append("")
        lines.append("| Phase | Score | Rating |")
        lines.append("|-------|-------|--------|")
        for phase in self.PHASE_ORDER:
            if phase in report.phase_scores:
                ps = report.phase_scores[phase]
                rating = self._score_to_grade(ps.score)
                title = self.PHASE_TITLES.get(phase, phase)
                lines.append(f"| {title} | {ps.score:.0f} | {rating} |")
        lines.append("")

        # ── Detailed KPI Results by Phase ────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## Detailed KPI Analysis")
        lines.append("")

        for phase in self.PHASE_ORDER:
            if phase not in report.phase_scores:
                continue
            ps = report.phase_scores[phase]
            title = self.PHASE_TITLES.get(phase, phase)
            lines.append(f"### {title}")
            lines.append("")

            for kpi in ps.kpis:
                lines.append(f"#### {kpi.kpi_id} — {kpi.name}")
                lines.append("")
                if kpi.rating == "n/a":
                    lines.append(f"> *{kpi.feedback}*")
                else:
                    val_str = self._format_value(kpi.raw_value, kpi.unit)
                    lines.append(f"- **Score**: {kpi.score:.0f}/100 ({kpi.rating})")
                    lines.append(f"- **Measured**: {val_str}")
                    lines.append(f"- **Feedback**: {kpi.feedback}")
                lines.append("")

            # Phase-specific chart
            chart_key = f"phase_{phase}"
            if chart_key in chart_paths:
                lines.append(f"![{title}]({chart_paths[chart_key]})")
                lines.append("")

        # ── KPI Bar Chart ────────────────────────────────────────────
        if "kpi_bar" in chart_paths:
            lines.append("---")
            lines.append("")
            lines.append("## All KPI Scores")
            lines.append("")
            lines.append(f"![KPI Scores]({chart_paths['kpi_bar']})")
            lines.append("")

        # ── Coaching Summary ─────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## Coaching Summary")
        lines.append("")
        lines.extend(self._coaching_summary(report))
        lines.append("")

        # ── Methodology Note ─────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## Methodology")
        lines.append("")
        lines.append("This analysis is based on the **Modern Forehand** framework derived from:")
        lines.append("- **Dr. Brian Gordon** — Type 3 forehand biomechanics, straight-arm extension")
        lines.append("- **Rick Macci** — compact unit turn, elbow mechanics, \"the flip\"")
        lines.append("- **Tennis Doctor** — four non-negotiables, kinetic chain sequencing")
        lines.append("- **Feel Tennis** — modern forehand 8-step model")
        lines.append("")
        lines.append("Pose estimation is performed using YOLO Pose (COCO 17-keypoint model). "
                      "All metrics are computed from 2D keypoint trajectories and are subject to "
                      "camera-angle limitations. For best results, use a side-view recording at 60+ FPS.")
        lines.append("")

        # Write to file
        report_path = self.output_dir / f"forehand_report_{video_name}.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 85:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 50:
            return "Fair"
        if score >= 30:
            return "Needs Work"
        return "Poor"

    @staticmethod
    def _format_value(value, unit: str) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if "degrees" in unit or "°" in unit:
                return f"{value:.1f}°"
            if "ratio" in unit:
                return f"{value:.2f}"
            if "norm" in unit or "torso" in unit:
                return f"{value:.3f}"
            return f"{value:.2f} {unit}"
        return f"{value} {unit}"

    def _coaching_summary(self, report: EvaluationReport) -> List[str]:
        """Generate prioritised coaching tips from KPI results."""
        lines = []

        # Strengths (top 3 scores)
        valid = [k for k in report.kpi_results if k.rating != "n/a"]
        if not valid:
            lines.append("*Insufficient data for a coaching summary.*")
            return lines

        sorted_kpis = sorted(valid, key=lambda k: k.score, reverse=True)

        lines.append("### Strengths")
        lines.append("")
        for kpi in sorted_kpis[:3]:
            if kpi.score >= 60:
                lines.append(f"- **{kpi.name}** ({kpi.score:.0f}): {kpi.feedback}")
        lines.append("")

        # Areas for improvement (bottom 3 scores)
        lines.append("### Areas for Improvement")
        lines.append("")
        for kpi in sorted_kpis[-3:]:
            if kpi.score < 70:
                lines.append(f"- **{kpi.name}** ({kpi.score:.0f}): {kpi.feedback}")
        lines.append("")

        # Top priority drill
        worst = sorted_kpis[-1]
        lines.append("### Priority Drill")
        lines.append("")
        drill = self._suggest_drill(worst)
        lines.append(f"Focus on **{worst.name}** — {drill}")

        return lines

    @staticmethod
    def _suggest_drill(kpi: KPIResult) -> str:
        """Suggest a drill based on the weakest KPI."""
        drills = {
            "P1.1": "Practice unit turns with a resistance band around your torso. Turn until your back faces the net.",
            "P1.4": "Do split-step-to-loaded-stance drills. Focus on bending your knees as you set up for the shot.",
            "P1.3": "Shadow swing in front of a mirror, keeping your head at a constant height.",
            "KC3.1": "Use the 'step-rotate-swing' drill: step with the front foot, rotate hips, then let the arm follow.",
            "KC3.2": "Practice hip-lead drills: rotate your hips toward the target while keeping your shoulders closed.",
            "KC3.4": "Swing along a line on the ground. Your racket should follow the line through the contact zone.",
            "C4.1": "Toss a ball and catch it at arm's length in front of your hip. That's your ideal contact point.",
            "C4.2": "Shadow swing focusing on arm extension at contact. For straight-arm: reach out fully. For double-bend: maintain a firm L-shape.",
            "C4.3": "Practice 'chest to target' drill: at contact, your chest should face the net and stop rotating.",
            "C4.4": "Keep your eyes on the contact point even after hitting. Count to 1 before looking up.",
            "E5.1": "After contact, push your hand toward the target for 2-3 feet before letting it rise.",
            "E5.2": "Finish with the racket over your opposite shoulder. The path should go forward first, then up.",
            "B6.1": "Place a book on your head during shadow swings. It should stay balanced throughout.",
            "B6.2": "Practice swings with a focus on keeping your belt buckle at a constant height.",
        }
        return drills.get(kpi.kpi_id, "Work on this area with targeted practice drills.")

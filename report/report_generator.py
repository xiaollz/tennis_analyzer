"""Markdown 报告生成器 — 支持正手 & 单反评估。

生成结构化的中文 Markdown 报告，支持多次击球独立评分。
自动根据挥拍类型选择对应的阶段标题和方法论说明。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from evaluation.forehand_evaluator import MultiSwingReport, SwingEvaluation
from evaluation.kpi import KPIResult


class ReportGenerator:
    """生成中文 Markdown 评估报告 — 支持正手 & 单反。"""

    # ── 正手阶段标题 ─────────────────────────────────────────────────
    FOREHAND_PHASE_TITLES = {
        "preparation": "阶段一：准备 & 转体",
        "loading": "阶段二：蓄力 & 落拍",
        "kinetic_chain": "阶段三：动力链 & 前挥",
        "contact": "阶段四：击球点",
        "extension": "阶段五：延伸 & 随挥",
        "balance": "阶段六：平衡 & 恢复",
    }
    FOREHAND_PHASE_ORDER = [
        "preparation", "loading", "kinetic_chain", "contact", "extension", "balance"
    ]

    # ── 单反阶段标题 ─────────────────────────────────────────────────
    BACKHAND_PHASE_TITLES = {
        "ohb_preparation": "阶段一：准备 & 侧身转体",
        "ohb_backswing": "阶段二：引拍 & L形杠杆",
        "ohb_kinetic_chain": "阶段三：动力链 & 前挥",
        "ohb_contact": "阶段四：击球点 & 手臂伸展",
        "ohb_extension": "阶段五：ATA收拍 & 保持侧身",
        "ohb_balance": "阶段六：平衡 & 恢复",
    }
    BACKHAND_PHASE_ORDER = [
        "ohb_preparation", "ohb_backswing", "ohb_kinetic_chain",
        "ohb_contact", "ohb_extension", "ohb_balance",
    ]

    # 兼容旧代码
    PHASE_TITLES = FOREHAND_PHASE_TITLES
    PHASE_ORDER = FOREHAND_PHASE_ORDER

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report: MultiSwingReport,
        video_name: str = "unknown",
        chart_paths: Optional[Dict[str, str]] = None,
        stroke_type: str = "forehand",
    ) -> str:
        """生成完整的 Markdown 报告并返回文件路径。

        Parameters
        ----------
        stroke_type : str
            "forehand" 或 "one_handed_backhand"
        """
        chart_paths = chart_paths or {}
        is_backhand = stroke_type != "forehand"
        phase_titles = self.BACKHAND_PHASE_TITLES if is_backhand else self.FOREHAND_PHASE_TITLES
        phase_order = self.BACKHAND_PHASE_ORDER if is_backhand else self.FOREHAND_PHASE_ORDER
        stroke_cn = "单手反拍" if is_backhand else "现代正手"

        lines: List[str] = []

        # ── 标题 ─────────────────────────────────────────────────────
        lines.append(f"# {stroke_cn}技术分析报告")
        lines.append("")
        lines.append(f"**视频**: {video_name}  ")
        lines.append(f"**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
        lines.append(f"**击球类型**: {stroke_cn}  ")
        lines.append(f"**检测到击球次数**: {report.total_swings}  ")
        lines.append("")

        # ── 综合评分概览 ─────────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## 综合评分概览")
        lines.append("")

        if report.total_swings > 0:
            lines.append(f"### 平均综合评分：{report.average_score:.0f} / 100  ({self._score_to_grade(report.average_score)})")
            lines.append("")

            if report.total_swings > 1:
                lines.append("| 击球次序 | 综合评分 | 评级 | 击球帧 | 音频确认 |")
                lines.append("|----------|----------|------|--------|----------|")
                for ev in report.swing_evaluations:
                    impact_f = ev.swing_event.impact_frame if ev.swing_event.impact_frame is not None else "—"
                    audio = "✓" if (ev.swing_event.impact_event and ev.swing_event.impact_event.audio_confirmed) else "—"
                    grade = self._score_to_grade(ev.overall_score)
                    lines.append(f"| 第{ev.swing_index + 1}次 | {ev.overall_score:.0f} | {grade} | {impact_f} | {audio} |")
                lines.append("")

                if "multi_swing_summary" in chart_paths:
                    lines.append(f"![各次击球评分对比]({chart_paths['multi_swing_summary']})")
                    lines.append("")
        else:
            lines.append("未检测到有效击球。以下仅评估可用的姿态数据。")
            lines.append("")

        # ── 每次击球详细分析 ─────────────────────────────────────────
        for ev in report.swing_evaluations:
            lines.extend(self._swing_section(
                ev, chart_paths, report.total_swings,
                phase_titles, phase_order, stroke_cn,
            ))

        # ── 综合教练建议 ─────────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## 综合教练建议")
        lines.append("")
        lines.extend(self._coaching_summary(report, is_backhand))
        lines.append("")

        # ── 方法论说明 ───────────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## 分析方法")
        lines.append("")
        if is_backhand:
            lines.append("本分析基于 **Modern One-Handed Backhand** 理论框架，综合以下来源：")
            lines.append("- **Dr. Brian Gordon** — 单反生物力学、L形杠杆系统")
            lines.append("- **Rick Macci** — 侧身转体、非持拍手平衡、ATA收拍")
            lines.append("- **Tennis Doctor** — Inside-Out 路径、保持侧身原则")
            lines.append("- **Feel Tennis** — 单反整体协调、步法与时机")
        else:
            lines.append("本分析基于 **Modern Forehand** 理论框架，综合以下来源：")
            lines.append("- **Dr. Brian Gordon** — Type 3 正手生物力学、直臂延伸")
            lines.append("- **Rick Macci** — 紧凑转体、肘部力学、「翻转」技术")
            lines.append("- **Tennis Doctor** — 四大不可妥协原则、动力链顺序")
            lines.append("- **Feel Tennis** — 现代正手8步模型")
        lines.append("")
        lines.append("姿态估计使用 YOLO Pose (COCO 17关键点模型)。"
                      "所有指标均基于2D关键点轨迹计算，受相机角度限制。"
                      "建议使用侧面视角、60+FPS 录制以获得最佳分析效果。")
        lines.append("")

        # 写入文件
        type_tag = "单反分析报告" if is_backhand else "正手分析报告"
        report_path = self.output_dir / f"{type_tag}_{video_name}.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    # ── 单次击球详细分析 ─────────────────────────────────────────────

    def _swing_section(
        self,
        ev: SwingEvaluation,
        chart_paths: Dict[str, str],
        total_swings: int,
        phase_titles: Dict[str, str],
        phase_order: List[str],
        stroke_cn: str,
    ) -> List[str]:
        lines = []
        lines.append("---")
        lines.append("")

        if total_swings > 1:
            lines.append(f"## 第 {ev.swing_index + 1} 次击球分析")
        else:
            lines.append("## 击球详细分析")
        lines.append("")

        lines.append(f"**综合评分**: {ev.overall_score:.0f} / 100  ({self._score_to_grade(ev.overall_score)})  ")
        lines.append(f"**击球类型**: {stroke_cn}  ")
        if ev.arm_style:
            lines.append(f"**手臂风格**: {ev.arm_style}  ")
        if ev.swing_event.impact_frame is not None:
            lines.append(f"**击球帧**: {ev.swing_event.impact_frame}  ")
        if ev.swing_event.prep_start_frame is not None:
            lines.append(f"**准备开始帧**: {ev.swing_event.prep_start_frame}  ")
        if ev.swing_event.followthrough_end_frame is not None:
            lines.append(f"**随挥结束帧**: {ev.swing_event.followthrough_end_frame}  ")
        if ev.swing_event.impact_event and ev.swing_event.impact_event.peak_speed_px_s:
            lines.append(f"**手腕峰值速度**: {ev.swing_event.impact_event.peak_speed_px_s:.0f} px/s  ")
            if ev.swing_event.impact_event.audio_confirmed:
                lines.append("**音频确认**: ✓ 已确认  ")
        lines.append("")

        # 雷达图
        radar_key = f"radar_{ev.swing_index}" if total_swings > 1 else "radar"
        if radar_key in chart_paths:
            lines.append(f"![阶段评分雷达图]({chart_paths[radar_key]})")
            lines.append("")

        # 阶段评分汇总表
        lines.append("### 各阶段评分")
        lines.append("")
        lines.append("| 阶段 | 评分 | 评级 |")
        lines.append("|------|------|------|")
        for phase in phase_order:
            if phase in ev.phase_scores:
                ps = ev.phase_scores[phase]
                grade = self._score_to_grade(ps.score)
                title = phase_titles.get(phase, phase)
                lines.append(f"| {title} | {ps.score:.0f} | {grade} |")
        lines.append("")

        # KPI 详细分析
        lines.append("### KPI 详细分析")
        lines.append("")

        for phase in phase_order:
            if phase not in ev.phase_scores:
                continue
            ps = ev.phase_scores[phase]
            title = phase_titles.get(phase, phase)
            lines.append(f"#### {title}")
            lines.append("")

            for kpi in ps.kpis:
                lines.append(f"**{kpi.kpi_id} — {kpi.name}**")
                lines.append("")
                if kpi.rating == "无数据" or kpi.rating == "n/a":
                    lines.append(f"> *{kpi.feedback}*")
                else:
                    val_str = self._format_value(kpi.raw_value, kpi.unit)
                    lines.append(f"- **评分**: {kpi.score:.0f}/100 ({kpi.rating})")
                    lines.append(f"- **测量值**: {val_str}")
                    lines.append(f"- **反馈**: {kpi.feedback}")
                lines.append("")

        # KPI 条形图
        bar_key = f"kpi_bar_{ev.swing_index}" if total_swings > 1 else "kpi_bar"
        if bar_key in chart_paths:
            lines.append(f"![KPI 评分详情]({chart_paths[bar_key]})")
            lines.append("")

        return lines

    # ── 辅助方法 ─────────────────────────────────────────────────────

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 85:
            return "优秀"
        if score >= 70:
            return "良好"
        if score >= 50:
            return "一般"
        if score >= 30:
            return "待改进"
        return "较差"

    @staticmethod
    def _format_value(value, unit: str) -> str:
        if value is None:
            return "无数据"
        if isinstance(value, float):
            if "度" in unit or "°" in unit:
                return f"{value:.1f}°"
            if "比" in unit or "ratio" in unit or "R²" in unit:
                return f"{value:.2f}"
            if "归一化" in unit or "标准差" in unit or "norm" in unit:
                return f"{value:.3f}"
            return f"{value:.2f} {unit}"
        return f"{value} {unit}"

    def _coaching_summary(self, report: MultiSwingReport, is_backhand: bool = False) -> List[str]:
        """从所有击球中提取综合教练建议。"""
        lines = []

        all_kpis: List[KPIResult] = []
        for ev in report.swing_evaluations:
            all_kpis.extend(ev.kpi_results)

        valid = [k for k in all_kpis if k.rating not in ("无数据", "n/a")]
        if not valid:
            lines.append("*数据不足，无法生成教练建议。*")
            return lines

        # 按 KPI ID 分组，取平均分
        kpi_avg: Dict[str, List[float]] = {}
        kpi_map: Dict[str, KPIResult] = {}
        for kpi in valid:
            kpi_avg.setdefault(kpi.kpi_id, []).append(kpi.score)
            kpi_map[kpi.kpi_id] = kpi

        avg_scores = [(kpi_id, float(sum(scores) / len(scores)), kpi_map[kpi_id])
                      for kpi_id, scores in kpi_avg.items()]
        avg_scores.sort(key=lambda x: x[1], reverse=True)

        # 优势
        lines.append("### 技术优势")
        lines.append("")
        for _, avg, kpi in avg_scores[:3]:
            if avg >= 50:
                lines.append(f"- **{kpi.name}**（平均 {avg:.0f} 分）：{kpi.feedback}")
        lines.append("")

        # 改进方向
        lines.append("### 需要改进")
        lines.append("")
        for _, avg, kpi in avg_scores[-3:]:
            if avg < 80:
                lines.append(f"- **{kpi.name}**（平均 {avg:.0f} 分）：{kpi.feedback}")
        lines.append("")

        # 首要训练建议
        worst_id, worst_avg, worst_kpi = avg_scores[-1]
        lines.append("### 首要训练建议")
        lines.append("")
        drill = self._suggest_drill(worst_kpi, is_backhand)
        lines.append(f"重点改进 **{worst_kpi.name}** — {drill}")

        return lines

    @staticmethod
    def _suggest_drill(kpi: KPIResult, is_backhand: bool = False) -> str:
        """根据最弱 KPI 建议训练方法。"""
        forehand_drills = {
            "P1.1": "练习整体转体（Unit Turn）：用弹力带绕在躯干上，转体直到背部面向球网。",
            "P1.4": "做分步→蓄力站位练习：重点在准备击球时弯曲膝盖。",
            "P1.3": "对着镜子做影子挥拍，保持头部高度不变。",
            "KC3.1": "使用「踏步→转髋→挥臂」三步练习：前脚踏步，转动髋部，然后让手臂自然跟随。",
            "KC3.2": "练习髋部领先：向目标方向转动髋部，同时保持肩膀关闭。",
            "KC3.4": "沿地面的一条线挥拍，球拍应在击球区沿直线运动。",
            "C4.1": "抛球并在髋部前方一臂距离处接住，那就是理想击球点。",
            "C4.2": "做影子挥拍，专注于击球时的手臂伸展。直臂型：充分伸展；双弯型：保持稳固的L形。",
            "C4.3": "练习「胸部面向目标」：击球时胸部应面向球网并停止旋转。",
            "C4.4": "击球后保持眼睛注视击球点，数到1再抬头。",
            "E5.1": "击球后，将手向目标方向推送60-90厘米，然后再让球拍上升。",
            "E5.2": "随挥结束时球拍应在对侧肩膀上方。路径应先向前，再向上。",
            "B6.1": "影子挥拍时在头上放一本书，全程保持平衡。",
            "B6.2": "练习挥拍时保持腰带扣高度不变。",
        }
        backhand_drills = {
            "BP1.1": "练习侧身转体：背对球网转体，非持拍手托住拍喉引导。",
            "BP1.2": "做分步→蓄力站位练习：重点在准备击球时弯曲膝盖。",
            "BP1.3": "引拍时非持拍手始终托住拍喉，直到开始前挥才释放。",
            "BP1.4": "对着镜子做影子挥拍，保持脊柱直立。",
            "BB2.1": "练习L形引拍：肘部弯曲约90°，球拍头指向上方/后方。",
            "BK3.1": "使用「踏步→转髋→挥臂」三步练习。",
            "BK3.2": "练习髋部领先：向目标方向转动髋部，同时保持肩膀关闭。",
            "BK3.3": "沿地面的一条线挥拍，球拍应在击球区沿直线运动（Inside-Out）。",
            "BC4.1": "抛球并在身体前方一臂距离处接住，那就是理想击球点。",
            "BC4.2": "击球时手臂应充分伸展，肘部接近完全伸直。",
            "BC4.3": "练习击球时胸部保持侧向，不要过度旋转。",
            "BC4.4": "非持拍手向后伸展，形成T字形平衡。",
            "BC4.5": "击球后保持眼睛注视击球点，数到1再抬头。",
            "BE5.1": "随挥结束时球拍应在持拍侧肩膀上方（ATA位置）。",
            "BE5.2": "击球后保持侧身姿态，不要急于转回正面。",
            "BB6.1": "影子挥拍时在头上放一本书，全程保持平衡。",
            "BB6.2": "练习挥拍时保持腰带扣高度不变。",
        }
        drills = backhand_drills if is_backhand else forehand_drills
        return drills.get(kpi.kpi_id, "针对此项进行专项训练。")

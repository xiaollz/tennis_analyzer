"""Markdown 报告生成器 v3 — 支持容错型正手原则模型 & 单反评估。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from config.framework_config import DEFAULT_CONFIG
from config.backhand_config import DEFAULT_BACKHAND_CONFIG
from evaluation.forehand_evaluator import MultiSwingReport, SwingEvaluation
from evaluation.kpi import KPIResult


class ReportGenerator:
    """生成中文 Markdown 评估报告 — 支持容错型正手原则模型 & 单反。"""

    FOREHAND_PHASE_TITLES = {
        "unit_turn": "阶段一：转开、备手与下肢准备",
        "chain": "阶段二：转髋带动与解旋顺序",
        "contact": "阶段三：前方接触与手臂结构",
        "through": "阶段四：向外、向前穿过",
        "stability": "阶段五：头部与躯干稳定",
    }
    FOREHAND_PHASE_ORDER = ["unit_turn", "chain", "contact", "through", "stability"]

    # ── 单反阶段标题 ─────────────────────────────────────────────────
    BACKHAND_PHASE_TITLES = {
        "ohb_preparation":   "阶段一：准备 & 侧身转体",
        "ohb_backswing":     "阶段二：引拍 & L形杠杆",
        "ohb_kinetic_chain": "阶段三：动力链 & 前挥",
        "ohb_contact":       "阶段四：击球点 & 手臂伸展",
        "ohb_extension":     "阶段五：ATA收拍 & 保持侧身",
        "ohb_balance":       "阶段六：平衡 & 恢复",
    }
    BACKHAND_PHASE_ORDER = [
        "ohb_preparation", "ohb_backswing", "ohb_kinetic_chain",
        "ohb_contact", "ohb_extension", "ohb_balance",
    ]

    # 兼容旧代码
    PHASE_TITLES = FOREHAND_PHASE_TITLES
    PHASE_ORDER = FOREHAND_PHASE_ORDER

    # ── 训练处方库（v3 新增）──────────────────────────────────────────
    FOREHAND_DRILLS = {
        "UT1.1": {
            "drill": "影子挥拍只做转开和备手，不急着前挥，先把上身完整带走。",
            "feel": "感觉胸口、肩膀和拍手一起转开，而不是手单独往后拿。",
            "cue": "Turn as a unit. 准备越简单，比赛里越容错。",
        },
        "UT1.2": {
            "drill": "做 split step 后轻微坐下，再开始准备，练腿先承担高度。",
            "feel": "感觉来球再低一点，你也不用先伸手去捞。",
            "cue": "Low ball? Use legs and trunk first.",
        },
        "UT1.3": {
            "drill": "慢速影子挥拍，保持头和脊柱高度稳定。",
            "feel": "像整个人围绕一根中轴旋转，不是弯腰去找球。",
            "cue": "Posture first, compensation later is a losing trade.",
        },
        "KC2.1": {
            "drill": "准备后先把髋部解回来，再让上身跟上。",
            "feel": "像髋先开门，肩随后被带开。",
            "cue": "Hip leads, shoulders follow.",
        },
        "KC2.2": {
            "drill": "做慢动作分解：转开 -> 等球 -> 髋先启动 -> 手再进击球区。",
            "feel": "前挥不是手去拉，而是身体把手送出去。",
            "cue": "Do not pull the forward swing with the hand.",
        },
        "C3.1": {
            "drill": "drop feed 到理想击球点，专打身体前方。",
            "feel": "主观上把球点再拿前一点。",
            "cue": "Always in front, or as close to always as possible.",
        },
        "C3.2": {
            "drill": "做站定喂球，专练不被球挤住，也不主动够球。",
            "feel": "击球时手臂和身体之间有工作空间。",
            "cue": "Not jammed, not reaching. Leave room for the swing to work.",
        },
        "C3.3": {
            "drill": "保持你自然的击球手臂样式，但要求每次接触都稳定重复。",
            "feel": "不是摆职业姿势，而是让结构在普通球上总能工作。",
            "cue": "Structure matters more than imitation.",
        },
        "T4.1": {
            "drill": "沿一条想象中的通道穿过击球区，别在接触区乱绕。",
            "feel": "手通过球区像在走轨道。",
            "cue": "The hand path through contact should stay clean.",
        },
        "T4.2": {
            "drill": "击球后先把手送向目标，再自然收拍。",
            "feel": "不是打一碰就收，而是先穿过去。",
            "cue": "Through the ball before the finish happens.",
        },
        "T4.3": {
            "drill": "击球后让手同时向前、向身体外侧送出去。",
            "feel": "像把球从自己身体边上推出去。",
            "cue": "Out and through, not just up.",
        },
        "S5.1": {
            "drill": "击球后继续盯住接触点半拍。",
            "feel": "头越安静，最后时刻越能自动修正。",
            "cue": "Quiet head, clearer contact.",
        },
        "S5.2": {
            "drill": "做慢速连续挥拍，要求头部高度尽量不变。",
            "feel": "头部像整拍的稳定锚点。",
            "cue": "Stability gives tolerance.",
        },
        "S5.3": {
            "drill": "低球时先降腿和躯干，不要先弯腰捞手。",
            "feel": "不同高度来球，身体框架尽量相似。",
            "cue": "Use the body to organize the contact.",
        },
    }

    BACKHAND_DRILLS = {
        "BP1.1": {
            "drill": "练习侧身转体：背对球网转体，非持拍手托住拍喉引导。",
            "feel": "感觉像是用背部面向球网，肩膀旋转超过 90°。",
            "cue": "Rick Macci: 「Show your back to the net.」",
        },
        "BP1.2": {
            "drill": "做分步→蓄力站位练习：重点在准备击球时弯曲膝盖。",
            "feel": "感觉大腿有明显的蓄力感。",
            "cue": "「Load the legs」— 腿部是力量的源泉。",
        },
        "BP1.3": {
            "drill": "引拍时非持拍手始终托住拍喉，直到开始前挥才释放。",
            "feel": "感觉非持拍手在引导整个引拍过程。",
            "cue": "非持拍手是单反的「方向盘」。",
        },
        "BP1.4": {
            "drill": "对着镜子做影子挥拍，保持脊柱直立。",
            "feel": "感觉头顶有一根线向上拉。",
            "cue": "脊柱是旋转的轴心。",
        },
        "BB2.1": {
            "drill": "练习L形引拍：肘部弯曲约90°，球拍头指向上方/后方。",
            "feel": "感觉像是把球拍「挂」在身后。",
            "cue": "「L-shaped lever」— 这是单反力量的杠杆系统。",
        },
        "BK3.1": {
            "drill": "使用「踏步→转髋→挥臂」三步练习。",
            "feel": "感觉力量从脚底传递到手臂。",
            "cue": "动力链：地面→腿→髋→肩→臂。",
        },
        "BK3.2": {
            "drill": "练习髋部领先：向目标方向转动髋部，同时保持肩膀关闭。",
            "feel": "感觉像拧毛巾。",
            "cue": "「Hip leads」— 髋部是引擎。",
        },
        "BK3.3": {
            "drill": "沿地面的一条线挥拍，球拍应在击球区沿直线运动（Inside-Out）。",
            "feel": "感觉手从身体内侧向外侧推出。",
            "cue": "「Inside-Out path」— 这是单反力量的关键路径。",
        },
        "BC4.1": {
            "drill": "抛球并在身体前方一臂距离处接住，那就是理想击球点。",
            "feel": "感觉手臂完全伸展，有充分的空间。",
            "cue": "「Contact with full extension」— 单反必须在手臂完全伸直时击球。",
        },
        "BC4.2": {
            "drill": "击球时手臂应充分伸展，肘部接近完全伸直。",
            "feel": "感觉像推门一样。",
            "cue": "单反的力量来自完全伸展的杠杆。",
        },
        "BC4.3": {
            "drill": "练习击球时胸部保持侧向，不要过度旋转。",
            "feel": "感觉身体像一面墙，手臂穿过墙壁击球。",
            "cue": "「Stay sideways」— 保持侧身是单反的核心原则。",
        },
        "BC4.4": {
            "drill": "非持拍手向后伸展，形成T字形平衡。",
            "feel": "感觉像飞机的两个翅膀。",
            "cue": "非持拍手的反向平衡是单反稳定性的关键。",
        },
        "BC4.5": {
            "drill": "击球后保持眼睛注视击球点，数到1再抬头。",
            "feel": "感觉头部是固定的轴心。",
            "cue": "头部稳定 = 击球稳定。",
        },
        "BE5.1": {
            "drill": "随挥结束时球拍应在持拍侧肩膀上方（ATA位置）。",
            "feel": "感觉像是「亮出腋窝」。",
            "cue": "「Air The Armpit」— 这是单反随挥的标志性位置。",
        },
        "BE5.2": {
            "drill": "击球后保持侧身姿态，不要急于转回正面。",
            "feel": "感觉身体保持侧向，只有手臂在运动。",
            "cue": "「Stay sideways through contact」— 过早转身会损失力量。",
        },
        "BB6.1": {
            "drill": "影子挥拍时在头上放一本书，全程保持平衡。",
            "feel": "感觉头部是整个身体的锚点。",
            "cue": "头部稳定是所有好击球的基础。",
        },
        "BB6.2": {
            "drill": "练习挥拍时保持腰带扣高度不变。",
            "feel": "感觉像在一个固定高度的平面上旋转。",
            "cue": "脊柱一致性反映核心稳定性。",
        },
    }

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report: MultiSwingReport,
        video_name: str = "unknown",
        chart_paths: Optional[Dict[str, str]] = None,
        stroke_type: str = "forehand",
        vlm_results: Optional[List[Optional[Dict]]] = None,
    ) -> str:
        """生成完整的 Markdown 报告并返回文件路径。"""
        import os
        chart_paths = chart_paths or {}
        rel_chart_paths = {}
        for k, p in chart_paths.items():
            try:
                rel_chart_paths[k] = Path(os.path.relpath(p, start=str(self.output_dir))).as_posix()
            except Exception:
                rel_chart_paths[k] = str(p)
        chart_paths = rel_chart_paths
        
        is_backhand = stroke_type != "forehand"
        phase_titles = self.BACKHAND_PHASE_TITLES if is_backhand else self.FOREHAND_PHASE_TITLES
        phase_order = self.BACKHAND_PHASE_ORDER if is_backhand else self.FOREHAND_PHASE_ORDER
        drills = self.BACKHAND_DRILLS if is_backhand else self.FOREHAND_DRILLS
        stroke_cn = "单手反拍" if is_backhand else "现代正手"

        lines: List[str] = []

        # ── 标题 ─────────────────────────────────────────────────────
        lines.append(f"# {stroke_cn}技术分析报告")
        lines.append("")
        lines.append(f"**视频**: {video_name}  ")
        lines.append(f"**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
        lines.append(f"**击球类型**: {stroke_cn}  ")
        lines.append(f"**检测到击球次数**: {report.total_swings}  ")
        if not is_backhand:
            lines.append(f"**评估模型**: 容错型正手原则模型 (v4)  ")
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

                # 一致性分析
                lines.extend(self._consistency_analysis(report))
                lines.append("")

                if "multi_swing_summary" in chart_paths:
                    lines.append(f"![各次击球评分对比]({chart_paths['multi_swing_summary']})")
                    lines.append("")
        else:
            lines.append("未检测到有效击球。以下仅评估可用的姿态数据。")
            lines.append("")

        # ── 每次击球详细分析 ─────────────────────────────────────────
        _vlm = vlm_results or []
        for ev in report.swing_evaluations:
            vlm_data = _vlm[ev.swing_index] if ev.swing_index < len(_vlm) else None
            lines.extend(self._swing_section(
                ev, chart_paths, report.total_swings,
                phase_titles, phase_order, stroke_cn, drills,
                vlm_result=vlm_data,
            ))

        # ── 训练处方 ─────────────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## 训练处方")
        lines.append("")
        lines.extend(self._training_prescription(report, drills, is_backhand))
        lines.append("")

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
            lines.append("本分析以 **《容错型正手》** 的底层原则为主：")
            lines.append("- **前方接触**：球尽量在身体前方完成接触")
            lines.append("- **简单准备**：转开、备手、等待，不堆多余动作")
            lines.append("- **髋带动前挥**：手不是发动机，髋和躯干才是")
            lines.append("- **Out / Up / Through**：手向外、向上、向前穿过球")
            lines.append("- **容错优先**：先追求普通球不容易打丢")
            lines.append("")
            lines.append("**本版评分层**：转开/备手 → 转髋带动 → 前方接触 → 向外向前穿过 → 稳定性")
        lines.append("")
        lines.append("姿态估计使用 YOLO Pose（COCO 17关键点模型）。"
                      "本版只保留能由身体关键点稳定估计的原则型指标；拍面角度、真实 wrist lag、球拍与前臂夹角不直接评分。"
                      "所有指标均基于2D关键点轨迹计算，受相机角度限制。建议使用侧面视角、60+FPS 录制以获得最佳分析效果。")
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
        drills: Dict,
        vlm_result: Optional[Dict] = None,
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
        if ev.arm_style and ev.arm_style != "未知":
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

        # KPI 详细分析（含训练处方）
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

                    # 如果评分低于 70，附上训练处方
                    if kpi.score < 70 and kpi.kpi_id in drills:
                        drill_info = drills[kpi.kpi_id]
                        lines.append("")
                        lines.append(f"  > **训练处方**: {drill_info['drill']}")
                        lines.append(f"  > **体感提示**: {drill_info['feel']}")
                        lines.append(f"  > **教练提示**: {drill_info['cue']}")
                lines.append("")

        # KPI 条形图
        bar_key = f"kpi_bar_{ev.swing_index}" if total_swings > 1 else "kpi_bar"
        if bar_key in chart_paths:
            lines.append(f"![KPI 评分详情]({chart_paths[bar_key]})")
            lines.append("")

        # ── VLM 视觉分析 ─────────────────────────────────────────
        if vlm_result:
            lines.extend(self._vlm_section(vlm_result, chart_paths, ev.swing_index, total_swings))

        return lines

    # ── VLM 视觉分析章节 ───────────────────────────────────────────

    @staticmethod
    def _vlm_section(
        vlm_result: Dict,
        chart_paths: Dict[str, str],
        swing_index: int,
        total_swings: int,
    ) -> List[str]:
        import os
        lines = []
        lines.append("### VLM 视觉分析（基于 FTT 原则）")
        lines.append("")

        # Keyframe grid image
        grid_path = vlm_result.get("keyframe_grid_path")
        if grid_path and os.path.exists(grid_path):
            lines.append(f"![关键帧分析]({grid_path})")
            lines.append("")

        # Overall assessment
        if vlm_result.get("overall_assessment"):
            lines.append(f"**整体评价**: {vlm_result['overall_assessment']}")
            lines.append("")

        # Strengths
        strengths = vlm_result.get("strengths", [])
        if strengths:
            lines.append("**优点**:")
            for s in strengths:
                lines.append(f"- {s}")
            lines.append("")

        # Issues
        issues = vlm_result.get("issues", [])
        if issues:
            lines.append("**问题诊断**:")
            lines.append("")
            for issue in issues:
                severity = issue.get("severity", "中")
                severity_tag = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(severity, "⚪")
                lines.append(f"**{severity_tag} {issue.get('name', '未命名')}** — 严重程度: {severity}")
                if issue.get("description"):
                    lines.append(f"- 观察: {issue['description']}")
                if issue.get("ftt_principle"):
                    lines.append(f"- FTT 原则: {issue['ftt_principle']}")
                if issue.get("correction"):
                    lines.append(f"- 纠正建议: {issue['correction']}")
                lines.append("")

        # Priority drill
        if vlm_result.get("priority_drill"):
            lines.append(f"**最优先训练**: {vlm_result['priority_drill']}")
            lines.append("")

        return lines

    # ── 一致性分析（v3 新增）─────────────────────────────────────────

    def _consistency_analysis(self, report: MultiSwingReport) -> List[str]:
        """分析多次击球的一致性。"""
        lines = []
        if report.total_swings < 2:
            return lines

        scores = [ev.overall_score for ev in report.swing_evaluations]
        std_dev = float(__import__("numpy").std(scores))
        score_range = max(scores) - min(scores)

        lines.append("### 击球一致性分析")
        lines.append("")

        if std_dev < 5:
            lines.append(f"**一致性评级**: 优秀（标准差 {std_dev:.1f}）")
            lines.append("")
            lines.append("各次击球表现非常稳定，动作模式已经形成良好的肌肉记忆。")
        elif std_dev < 10:
            lines.append(f"**一致性评级**: 良好（标准差 {std_dev:.1f}）")
            lines.append("")
            lines.append("各次击球表现较为稳定，但仍有一定波动。建议通过重复练习进一步固化动作模式。")
        elif std_dev < 15:
            lines.append(f"**一致性评级**: 一般（标准差 {std_dev:.1f}）")
            lines.append("")
            lines.append("各次击球表现波动较大，说明动作模式尚未完全稳定。建议降低击球力度，专注于动作的一致性。")
        else:
            lines.append(f"**一致性评级**: 待改进（标准差 {std_dev:.1f}）")
            lines.append("")
            lines.append("各次击球表现差异显著，说明技术动作尚未形成稳定模式。建议从慢速影子挥拍开始，逐步建立一致的动作模式。")

        # 找出波动最大的阶段
        if report.total_swings >= 2:
            phase_stds = {}
            for phase in report.swing_evaluations[0].phase_scores:
                phase_scores = []
                for ev in report.swing_evaluations:
                    if phase in ev.phase_scores:
                        phase_scores.append(ev.phase_scores[phase].score)
                if len(phase_scores) >= 2:
                    phase_stds[phase] = float(__import__("numpy").std(phase_scores))

            if phase_stds:
                most_variable = max(phase_stds, key=phase_stds.get)
                fh_titles = self.FOREHAND_PHASE_TITLES
                bh_titles = self.BACKHAND_PHASE_TITLES
                all_titles = {**fh_titles, **bh_titles}
                title = all_titles.get(most_variable, most_variable)
                lines.append("")
                lines.append(f"**波动最大的阶段**: {title}（标准差 {phase_stds[most_variable]:.1f}）— 这是提高一致性的重点。")

        return lines

    # ── 训练处方（v3 新增）────────────────────────────────────────────

    def _training_prescription(
        self,
        report: MultiSwingReport,
        drills: Dict,
        is_backhand: bool,
    ) -> List[str]:
        """生成优先级排序的训练处方。"""
        lines = []

        # 收集所有 KPI 的平均分
        all_kpis: List[KPIResult] = []
        for ev in report.swing_evaluations:
            all_kpis.extend(ev.kpi_results)

        valid = [k for k in all_kpis if k.rating not in ("无数据", "n/a")]
        if not valid:
            lines.append("*数据不足，无法生成训练处方。*")
            return lines

        # 按 KPI ID 分组取平均
        kpi_avg: Dict[str, List[float]] = {}
        kpi_map: Dict[str, KPIResult] = {}
        for kpi in valid:
            kpi_avg.setdefault(kpi.kpi_id, []).append(kpi.score)
            kpi_map[kpi.kpi_id] = kpi

        phase_weights = (
            DEFAULT_BACKHAND_CONFIG.scoring.as_dict()
            if is_backhand else DEFAULT_CONFIG.scoring.as_dict()
        )
        weak_kpis = []
        for kpi_id, scores in kpi_avg.items():
            avg = float(sum(scores) / len(scores))
            kpi = kpi_map[kpi_id]
            deficit = max(0.0, 70.0 - avg)
            if deficit <= 0.0:
                continue
            phase_weight = float(phase_weights.get(kpi.phase, 0.1))
            priority_score = deficit * phase_weight
            weak_kpis.append((kpi_id, avg, kpi, priority_score))

        # 优先级与总分口径对齐：优先修复“低分且高权重阶段”的问题。
        weak_kpis.sort(key=lambda x: (-x[3], x[1], x[0]))

        if not weak_kpis:
            lines.append("所有指标均达到良好水平，继续保持！可以尝试在更高强度的对抗中保持技术质量。")
            return lines

        lines.append("以下是按优先级排序的训练计划，从最需要改进的指标开始：")
        lines.append("")

        for priority, (kid, avg, kpi, priority_score) in enumerate(weak_kpis[:5], 1):
            lines.append(
                f"### 优先级 {priority}：{kpi.name}（平均 {avg:.0f} 分，优先度 {priority_score:.1f}）"
            )
            lines.append("")

            if kid in drills:
                drill_info = drills[kid]
                lines.append(f"**训练方法**: {drill_info['drill']}")
                lines.append("")
                lines.append(f"**体感提示**: {drill_info['feel']}")
                lines.append("")
                lines.append(f"**教练提示**: {drill_info['cue']}")
            else:
                lines.append(f"**反馈**: {kpi.feedback}")
                lines.append("")
                lines.append("针对此项进行专项训练。")
            lines.append("")

        # 训练计划建议
        lines.append("### 建议训练计划")
        lines.append("")
        lines.append("1. **热身**（5分钟）：慢速影子挥拍，专注于“转开 -> 等球 -> 髋带动 -> 前方接触”。")
        lines.append(f"2. **专项训练**（15分钟）：重点练习上述优先级 1（{weak_kpis[0][2].name}）。")
        if len(weak_kpis) >= 2:
            lines.append(f"3. **辅助训练**（10分钟）：练习优先级 2（{weak_kpis[1][2].name}）。")
        lines.append("4. **整合练习**（10分钟）：正常速度挥拍，将专项训练融入完整动作。")
        lines.append("5. **录像对比**：每周录制一次，使用本分析器对比进步。")

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
            if "比" in unit or "ratio" in unit or "R²" in unit:
                return f"{value:.2f}"
            if "归一化" in unit or "标准差" in unit or "norm" in unit:
                return f"{value:.3f}"
            # Avoid matching "高度比" as angle; treat explicit angular units only.
            if unit in ("度", "度/秒", "度标准差") or "°" in unit:
                return f"{value:.1f}°" if unit == "度" else f"{value:.0f} {unit}"
            if "px/s²" in unit:
                return f"{value:.0f} {unit}"
            if "ms" in unit:
                return f"{value * 1000:.0f} ms"
            if "秒" in unit:
                # Avoid misleading "0 秒" for sub-second timing values.
                if abs(value) < 1.0:
                    return f"{value * 1000:.0f} ms"
                return f"{value:.2f} 秒"
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

        # 核心理念提醒
        lines.append("### 核心理念")
        lines.append("")
        if is_backhand:
            lines.append("- **保持侧身**：单反的力量来自侧身姿态和完全伸展的手臂杠杆。")
            lines.append("- **非持拍手平衡**：非持拍手的反向伸展是稳定性和力量的关键。")
            lines.append("- **L形杠杆**：引拍时保持肘部弯曲约90°，形成高效的杠杆系统。")
        else:
            lines.append("- **容错优先**：不是偶尔打一板神球，而是普通球也不容易打丢。")
            lines.append("- **前方接触**：来不及时至少也尽量把球点留在身体前侧。")
            lines.append("- **用腿和躯干组织击球**：低球和难球优先靠下肢与躯干，不先靠手补。")
            lines.append("- **向外、向上、向前**：现代正手不是只往上刷，必须同时出球、穿球、离身。 ")

        return lines

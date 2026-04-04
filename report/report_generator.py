"""Markdown 报告生成器 v4 — VLM 为核心，量化指标为辅助参考。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from config.framework_config import DEFAULT_CONFIG
from config.backhand_config import DEFAULT_BACKHAND_CONFIG
from evaluation.forehand_evaluator import MultiSwingReport, SwingEvaluation
from evaluation.kpi import KPIResult


class ReportGenerator:
    """生成中文 Markdown 评估报告 — VLM 教练分析为主，量化指标为辅。"""

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

        # ── 总体印象（取代旧的综合评分概览）─────────────────────────
        lines.append("---")
        lines.append("")
        lines.append("## 总体印象")
        lines.append("")

        if report.total_swings > 0:
            lines.append(self._qualitative_summary(report, is_backhand))
            lines.append("")

            if report.total_swings > 1:
                # 一致性分析（简化版，不含分数表）
                lines.extend(self._consistency_analysis(report))
                lines.append("")
        else:
            lines.append("未检测到有效击球。以下仅评估可用的姿态数据。")
            lines.append("")

        # ── 核心问题总结（多球时生成）─────────────────────────────────
        _vlm = vlm_results or []
        if report.total_swings > 1 and any(v for v in _vlm if v):
            lines.extend(self._multi_swing_summary(_vlm, report))

        # ── 每次击球详细分析（只展开重点球）─────────────────────────────
        highlight_indices = self._pick_highlight_swings(_vlm, report)
        for ev in report.swing_evaluations:
            vlm_data = _vlm[ev.swing_index] if ev.swing_index < len(_vlm) else None
            supp = vlm_data.get("supplementary_metrics") if vlm_data else None
            is_highlight = ev.swing_index in highlight_indices
            lines.extend(self._swing_section(
                ev, chart_paths, report.total_swings,
                phase_titles, phase_order, stroke_cn, drills,
                vlm_result=vlm_data,
                supplementary_metrics=supp,
                compact=not is_highlight,
            ))

        # ── 综合教练建议（简化版）─────────────────────────────────────
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
        lines.append("视觉分析由 VLM（视觉语言模型）基于关键帧完成，量化指标由 YOLO Pose（COCO 17关键点）辅助计算。"
                      "所有量化指标均基于2D关键点轨迹，受相机角度限制。建议使用侧面视角、60+FPS 录制以获得最佳分析效果。")
        lines.append("")

        # 写入文件
        type_tag = "单反分析报告" if is_backhand else "正手分析报告"
        report_path = self.output_dir / f"{type_tag}_{video_name}.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    # ── 总体定性总结（取代旧的分数概览）─────────────────────────────

    def _qualitative_summary(self, report: MultiSwingReport, is_backhand: bool) -> str:
        """根据整体数据生成一句定性总结，取代分数展示。"""
        avg = report.average_score
        n = report.total_swings
        stroke = "单反" if is_backhand else "正手"

        if avg >= 85:
            return f"本次共分析 {n} 次{stroke}击球，整体动作质量优秀，各环节衔接流畅，技术框架成熟稳定。"
        elif avg >= 70:
            return f"本次共分析 {n} 次{stroke}击球，整体动作框架良好，核心原则基本到位，部分环节仍有优化空间。"
        elif avg >= 50:
            return f"本次共分析 {n} 次{stroke}击球，动作框架初步成型，但多个环节需要改进以提高容错性和稳定性。"
        elif avg >= 30:
            return f"本次共分析 {n} 次{stroke}击球，技术动作存在较明显的结构性问题，建议从基础动作模式开始重建。"
        else:
            return f"本次共分析 {n} 次{stroke}击球，动作模式与目标技术差距较大，建议在教练指导下从分解动作开始练习。"

    # ── 单次击球详细分析 ─────────────────────────────────────────────

    # ── 多球总结 + 重点球选择 ────────────────────────────────────────

    @staticmethod
    def _multi_swing_summary(vlm_results: list, report) -> List[str]:
        """Generate a summary paragraph across all swings — core issues first."""
        lines = []
        lines.append("## 核心问题")
        lines.append("")

        # Collect root causes from all swings
        root_causes = []
        for v in vlm_results:
            if not v:
                continue
            tree = v.get("root_cause_tree", {})
            rc = tree.get("root_cause", "")
            if rc:
                root_causes.append(rc)
            elif v.get("issues"):
                # Legacy format: use first issue name
                root_causes.append(v["issues"][0].get("name", ""))

        if not root_causes:
            return lines

        # Count frequency of similar root causes (simple dedup by first 10 chars)
        from collections import Counter
        short_causes = [rc[:15] for rc in root_causes]
        freq = Counter(short_causes)
        top_cause_prefix = freq.most_common(1)[0][0] if freq else ""

        # Find the most representative full root cause text
        main_cause = next((rc for rc in root_causes if rc.startswith(top_cause_prefix)), root_causes[0])
        occurrence = freq.most_common(1)[0][1] if freq else 0

        lines.append(f"**{report.total_swings} 次击球中，{occurrence} 次出现同一个核心问题：{main_cause}**")
        lines.append("")

        # List which swings had which issues (compact)
        if len(freq) > 1:
            for prefix, count in freq.most_common(3):
                matching = [i + 1 for i, rc in enumerate(root_causes) if rc[:15] == prefix]
                full = next((rc for rc in root_causes if rc.startswith(prefix)), prefix)
                lines.append(f"- 第 {'、'.join(map(str, matching))} 球：{full[:50]}")
            lines.append("")

        return lines

    @staticmethod
    def _pick_highlight_swings(vlm_results: list, report) -> set:
        """Pick 1-2 most representative swings for detailed analysis."""
        if report.total_swings <= 2:
            return set(range(report.total_swings))

        scored = []
        for i, v in enumerate(vlm_results):
            if not v:
                scored.append((i, 0))
                continue
            # Prefer swings with root_cause_tree (richer analysis)
            has_tree = 1 if v.get("root_cause_tree") else 0
            issue_count = len(v.get("issues", []))
            has_journey = 1 if v.get("diagnostic_session") else 0
            score = has_tree * 3 + issue_count + has_journey * 2
            scored.append((i, score))

        scored.sort(key=lambda x: -x[1])
        # Pick top 2, but try to pick different root causes
        highlights = {scored[0][0]}
        if len(scored) > 1:
            first_rc = ""
            v0 = vlm_results[scored[0][0]]
            if v0:
                first_rc = v0.get("root_cause_tree", {}).get("root_cause", "")[:15]
            for idx, _ in scored[1:]:
                v = vlm_results[idx]
                if v:
                    rc = v.get("root_cause_tree", {}).get("root_cause", "")[:15]
                    if rc != first_rc:
                        highlights.add(idx)
                        break
            if len(highlights) < 2:
                highlights.add(scored[1][0])

        return highlights

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
        supplementary_metrics: Optional[Dict] = None,
        compact: bool = False,
    ) -> List[str]:
        lines = []
        lines.append("---")
        lines.append("")

        if total_swings > 1:
            label = f"## 第 {ev.swing_index + 1} 次击球" + (" (详细)" if not compact else "")
            lines.append(label)
        else:
            lines.append("## 击球分析")
        lines.append("")

        # ── Compact mode: one-liner summary only ──
        if compact and vlm_result:
            score = vlm_result.get("score")
            tree = vlm_result.get("root_cause_tree", {})
            rc = tree.get("root_cause", "")
            if score:
                lines.append(f"评分 {score}/100 — {rc}" if rc else f"评分 {score}/100")
            elif rc:
                lines.append(rc)
            lines.append("")
            return lines

        # ── 击球基本信息 ───────────────────────────────────────────
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
                lines.append("**音频确认**: 已确认  ")
        lines.append("")

        # ── 教练分析（VLM — 主体内容）─────────────────────────────
        if vlm_result:
            lines.extend(self._vlm_section(vlm_result, chart_paths, ev.swing_index, total_swings))

        # ── 量化辅助参考 ──────────────────────────────────────────
        metrics_lines = self._supplementary_metrics_section(
            ev, phase_titles, phase_order, supplementary_metrics,
        )
        if metrics_lines:
            lines.extend(metrics_lines)

        return lines

    # ── VLM 视觉分析章节 ───────────────────────────────────────────

    @staticmethod
    def _vlm_section(
        vlm_result: Dict,
        chart_paths: Dict[str, str],
        swing_index: int,
        total_swings: int,
        include_journey: bool = True,
    ) -> List[str]:
        import os
        lines = []
        lines.append("### 教练分析")
        lines.append("")

        # Keyframe grid image
        grid_path = vlm_result.get("keyframe_grid_path", "")
        if grid_path:
            basename = os.path.basename(grid_path)
            rel_path = f"charts/{basename}"
            lines.append(f"![关键帧分析]({rel_path})")
            lines.append("")

        # Score
        if vlm_result.get("score") is not None:
            score = vlm_result["score"]
            reasoning = vlm_result.get("score_reasoning", "")
            lines.append(f"**容错评分**: {score}/100 — {reasoning}")
            lines.append("")

        # ── Diagnostic Journey (v2.0 multi-round) ──
        diag_session = vlm_result.get("diagnostic_session")
        if diag_session and include_journey:
            lines.extend(ReportGenerator._format_diagnostic_journey(diag_session))

        # ── New format: Root Cause Tree ──
        root_tree = vlm_result.get("root_cause_tree")
        if root_tree:
            lines.extend(ReportGenerator._format_root_cause_tree(vlm_result))
        else:
            # ── Legacy format fallback ──
            lines.extend(ReportGenerator._format_legacy_issues(vlm_result))

        return lines

    # ── 诊断推理过程（v2.0 多轮诊断）────────────────────────────────────

    @staticmethod
    def _format_diagnostic_journey(session_data: Dict) -> List[str]:
        """Render the multi-round diagnostic journey as a readable narrative.

        RG-01: Shows iterative reasoning process
        RG-02: Narrative style, not log dump
        RG-04: Can be suppressed by caller via include_journey=False

        Args:
            session_data: Serialized DiagnosticSession dict with rounds,
                         hypotheses, and observations.
        """
        lines: List[str] = []
        rounds = session_data.get("rounds", [])
        hypotheses = session_data.get("hypotheses", [])

        if not rounds:
            return lines

        lines.append("### 诊断推理过程")
        lines.append("")

        # Build hypothesis lookup
        hyp_map = {h["id"]: h for h in hypotheses}

        # Narrative round labels
        _ROUND_LABELS = {
            0: "初步扫描",
            -1: "根因确认",  # sentinel for last round
        }

        for i, rnd in enumerate(rounds):
            round_num = rnd.get("round_number", i)
            is_last = i == len(rounds) - 1

            # Round header
            if round_num == 0:
                label = "初步扫描"
            elif is_last:
                label = "根因确认"
            else:
                label = f"针对性观察 (第{round_num}轮)"

            lines.append(f"**{label}:**")
            lines.append("")

            observations = rnd.get("observations", [])
            updates = rnd.get("hypothesis_updates", [])

            if round_num == 0:
                # Round 0: report initial hypothesis creation
                active_hyps = [h for h in hypotheses if h.get("round_introduced", 0) == 0]
                if active_hyps:
                    hyp_names = [h.get("name_zh") or h.get("name", "?") for h in active_hyps]
                    lines.append(f"- 初步假设: {', '.join(hyp_names)}")
                else:
                    lines.append("- 扫描未发现明显问题")
                lines.append("")
                continue

            # Intermediate/final rounds: observations + hypothesis updates
            if observations:
                for obs in observations[:5]:  # Cap at 5 observations per round
                    frame = obs.get("frame", "")
                    desc = obs.get("description", "")[:100]
                    judgment = obs.get("judgment", "")
                    judgment_cn = {"yes": "是", "no": "否", "unclear": "不确定"}.get(judgment, judgment)

                    override = obs.get("override_reason")
                    if override:
                        lines.append(f"- {frame}: {desc} -> **{judgment_cn}**（量化数据修正）")
                    else:
                        lines.append(f"- {frame}: {desc} -> **{judgment_cn}**")

            if updates:
                lines.append("")
                for upd in updates:
                    hyp_id = upd.get("hypothesis_id", "")
                    action = upd.get("action", "")
                    reason = upd.get("reason", "")[:80]
                    hyp = hyp_map.get(hyp_id, {})
                    hyp_name = hyp.get("name_zh") or hyp.get("name", hyp_id)

                    if action == "confirm":
                        lines.append(f"- **确认**: {hyp_name} — {reason}")
                    elif action == "eliminate":
                        lines.append(f"- **排除**: {hyp_name} — {reason}")
                    elif action == "adjust":
                        lines.append(f"- **调整**: {hyp_name} — {reason}")

            lines.append("")

        # Summary: final hypothesis state
        confirmed = [h for h in hypotheses if h.get("status") == "confirmed"]
        eliminated = [h for h in hypotheses if h.get("status") == "eliminated"]
        if confirmed:
            names = [h.get("name_zh") or h.get("name", "?") for h in confirmed]
            lines.append(f"**诊断结论**: {', '.join(names)}")
            if eliminated:
                elim_names = [h.get("name_zh") or h.get("name", "?") for h in eliminated]
                lines.append(f"**已排除**: {', '.join(elim_names)}")
            lines.append("")

        convergence = session_data.get("convergence_score", 0)
        rounds_count = len(rounds)
        lines.append(f"*经过{rounds_count}轮分析，收敛度: {convergence:.0%}*")
        lines.append("")

        return lines

    # ── 根因树格式（新版） ─────────────────────────────────────────────

    @staticmethod
    def _format_root_cause_tree(vlm_result: Dict) -> List[str]:
        lines = []
        tree = vlm_result["root_cause_tree"]

        # Overall narrative — 简短故事
        if vlm_result.get("overall_narrative"):
            lines.append(vlm_result['overall_narrative'])
            lines.append("")

        # Root cause — 核心问题
        lines.append(f"> **根因：{tree.get('root_cause', '')}**")
        lines.append("")
        evidence_parts = []
        if tree.get("root_cause_evidence"):
            evidence_parts.append(tree['root_cause_evidence'])
        if tree.get("root_cause_body"):
            evidence_parts.append(tree['root_cause_body'])
        if evidence_parts:
            lines.append(" ".join(evidence_parts))
            lines.append("")
        if tree.get("root_cause_feel"):
            lines.append(f"*{tree['root_cause_feel']}*")
            lines.append("")

        # Causal tree — 因果树（紧凑版）
        symptoms = tree.get("downstream_symptoms", [])
        if symptoms:
            root_short = tree.get("root_cause", "根因")[:25]
            lines.append("```")
            lines.append(f"  {root_short}")
            for i, s in enumerate(symptoms):
                connector = "├──" if i < len(symptoms) - 1 else "└──"
                cause_short = s.get("how_root_cause_creates_it", "")[:60]
                lines.append(f"  {connector} {s.get('symptom', '')} [{s.get('frame', '')}]")
                lines.append(f"  {'│' if i < len(symptoms) - 1 else ' '}   {cause_short}")
            lines.append("```")
            lines.append("")

        # Fix — 顿悟 + 训练（合并为一段）
        fix = tree.get("fix", {})
        if fix:
            fix_parts = []
            if fix.get("insight"):
                fix_parts.append(f"**{fix['insight']}**")
            if fix.get("mental_model_shift"):
                fix_parts.append(fix['mental_model_shift'])
            if fix_parts:
                lines.append(" ".join(fix_parts))
                lines.append("")
            if fix.get("one_drill"):
                lines.append(f"**练这个：** {fix['one_drill']}")
                method = fix.get("drill_method", "")
                cue = fix.get("drill_feel_cue", "")
                check = fix.get("check_criteria", "")
                if method:
                    lines.append(f"做法：{method}")
                if cue:
                    lines.append(f"口令：{cue}")
                if check:
                    lines.append(f"做对了 = {check}")
                lines.append("")

        # Secondary root cause
        secondary = vlm_result.get("secondary_root_cause")
        if secondary and secondary.get("root_cause"):
            lines.append(f"另外注意：{secondary['root_cause']}")
            sec_fix = secondary.get("fix", {})
            if sec_fix and sec_fix.get("one_drill"):
                lines.append(f"练：{sec_fix['one_drill']}")
            lines.append("")

        # Strengths（只保留核心，不要进步信号的冗余描述）
        strengths = vlm_result.get("strengths", [])
        if strengths:
            strength_texts = []
            for s in strengths:
                if isinstance(s, dict):
                    strength_texts.append(s.get('description', ''))
                else:
                    strength_texts.append(str(s))
            if strength_texts:
                lines.append(f"**做得好：** {'；'.join(strength_texts)}")
                lines.append("")

        # Kinetic chain narrative
        if vlm_result.get("kinetic_chain_narrative"):
            lines.append(f"**力量传导：** {vlm_result['kinetic_chain_narrative']}")
            lines.append("")

        return lines

    @staticmethod
    def _format_legacy_issues(vlm_result: Dict) -> List[str]:
        """Fallback for old-format VLM responses (issues list)."""
        lines = []

        if vlm_result.get("overall_assessment"):
            lines.append(f"**整体评价**: {vlm_result['overall_assessment']}")
            lines.append("")

        strengths = vlm_result.get("strengths", [])
        if strengths:
            lines.append("**优点**:")
            for s in strengths:
                if isinstance(s, dict):
                    lines.append(f"- {s.get('description', '')}")
                    if s.get("body_reason"):
                        lines.append(f"  - 身体原因: {s['body_reason']}")
                else:
                    lines.append(f"- {s}")
            lines.append("")

        issues = vlm_result.get("issues", [])
        if issues:
            lines.append("**问题诊断**:")
            lines.append("")
            for issue in issues:
                severity = issue.get("severity", "中")
                severity_tag = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(severity, "⚪")
                frame_tag = f" [{issue['frame']}]" if issue.get("frame") else ""
                lines.append(f"**{severity_tag} {issue.get('name', '未命名')}{frame_tag}** — 严重程度: {severity}")
                if issue.get("ftt_principle"):
                    lines.append(f"- FTT 原则: {issue['ftt_principle']}")
                if issue.get("action"):
                    lines.append(f"- 动作: {issue['action']}")
                if issue.get("body"):
                    lines.append(f"- 身体: {issue['body']}")
                if issue.get("feel"):
                    lines.append(f"- 感觉: {issue['feel']}")
                if issue.get("drill"):
                    lines.append(f"- 训练方法: {issue['drill']}")
                lines.append("")

        if vlm_result.get("weight_transfer"):
            lines.append(f"**重心转移分析**: {vlm_result['weight_transfer']}")
            lines.append("")
        if vlm_result.get("kinetic_chain"):
            lines.append(f"**动力链分析**: {vlm_result['kinetic_chain']}")
            lines.append("")

        drills = vlm_result.get("drills", [])
        if drills:
            lines.append("**推荐训练计划**:")
            lines.append("")
            for i, drill in enumerate(drills, 1):
                if isinstance(drill, dict):
                    lines.append(f"{i}. **{drill.get('name', '训练')}**")
                    if drill.get("method"):
                        lines.append(f"   - 方法: {drill['method']}")
                    if drill.get("purpose"):
                        lines.append(f"   - 目的: {drill['purpose']}")
                    if drill.get("cue"):
                        lines.append(f"   - 口令: {drill['cue']}")
            lines.append("")

        if vlm_result.get("priority_drill"):
            lines.append(f"**最优先训练**: {vlm_result['priority_drill']}")
            lines.append("")

        return lines

    # ── 量化辅助参考章节 ─────────────────────────────────────────────

    def _supplementary_metrics_section(
        self,
        ev: SwingEvaluation,
        phase_titles: Dict[str, str],
        phase_order: List[str],
        extra_metrics: Optional[Dict] = None,
    ) -> List[str]:
        """从 KPI 结果中提取关键量化数据，以简洁的单行形式呈现。"""
        lines = []

        # Collect valid KPIs with actual measurements
        valid_kpis = [
            kpi for kpi in ev.kpi_results
            if kpi.rating not in ("无数据", "n/a") and kpi.raw_value is not None
        ]

        # Also include any externally-supplied supplementary metrics
        has_extra = extra_metrics and len(extra_metrics) > 0

        if not valid_kpis and not has_extra:
            return lines

        lines.append("### 量化辅助参考")
        lines.append("")

        # Format each KPI as a single-line metric description
        for kpi in valid_kpis:
            val_str = self._format_value(kpi.raw_value, kpi.unit)
            note = self._metric_note(kpi)
            # Flag suspicious values that might be unreliable
            if kpi.raw_value is not None and kpi.raw_value < 0 and kpi.kpi_id == "C3.1":
                lines.append(f"- **{kpi.name}**: {val_str} — ⚠️ 可能受相机角度影响")
            elif note:
                lines.append(f"- **{kpi.name}**: {val_str} — {note}")
            else:
                lines.append(f"- **{kpi.name}**: {val_str}")

        # Append supplementary metrics (M7/M8/M9)
        if has_extra:
            sm = extra_metrics
            if sm.get("arm_torso_synchrony") is not None:
                lines.append(f"- **手臂-躯干同步性**: {sm['arm_torso_synchrony']:.2f}（{sm.get('arm_torso_sync_label', '')}）")
            if sm.get("wrist_v_detected") is not None:
                if sm["wrist_v_detected"]:
                    lines.append(f"- **手腕高度模式**: 检测到 V 形 scooping（深度 {sm.get('wrist_v_depth', 0):.2f}）")
                else:
                    lines.append("- **手腕高度模式**: 未检测到 scooping")
            if sm.get("swing_shape_label") is not None:
                lines.append(f"- **挥拍轨迹**: {sm['swing_shape_label']}（弧度比 {sm.get('swing_arc_ratio', 0):.2f}）")

        lines.append("")
        return lines

    @staticmethod
    def _metric_note(kpi: KPIResult) -> str:
        """为 KPI 生成简短的定性注释，取代分数。"""
        if kpi.score >= 85:
            return "良好"
        elif kpi.score >= 70:
            return "基本到位"
        elif kpi.score >= 50:
            return "需关注"
        elif kpi.score >= 30:
            return "明显不足"
        else:
            return "亟需改进"

    # ── 一致性分析 ─────────────────────────────────────────────────

    def _consistency_analysis(self, report: MultiSwingReport) -> List[str]:
        """分析多次击球的一致性。"""
        lines = []
        if report.total_swings < 2:
            return lines

        scores = [ev.overall_score for ev in report.swing_evaluations]
        std_dev = float(__import__("numpy").std(scores))

        lines.append("### 击球一致性")
        lines.append("")

        if std_dev < 5:
            lines.append("各次击球表现非常稳定，动作模式已经形成良好的肌肉记忆。")
        elif std_dev < 10:
            lines.append("各次击球表现较为稳定，但仍有一定波动。建议通过重复练习进一步固化动作模式。")
        elif std_dev < 15:
            lines.append("各次击球表现波动较大，动作模式尚未完全稳定。建议降低击球力度，专注于动作的一致性。")
        else:
            lines.append("各次击球表现差异显著，技术动作尚未形成稳定模式。建议从慢速影子挥拍开始，逐步建立一致的动作模式。")

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
            if unit in ("度", "度/秒", "度标准差") or "°" in unit:
                return f"{value:.1f}°" if unit == "度" else f"{value:.0f} {unit}"
            if "px/s²" in unit:
                return f"{value:.0f} {unit}"
            if "ms" in unit:
                return f"{value * 1000:.0f} ms"
            if "秒" in unit:
                if abs(value) < 1.0:
                    return f"{value * 1000:.0f} ms"
                return f"{value:.2f} 秒"
            return f"{value:.2f} {unit}"
        return f"{value} {unit}"

    def _coaching_summary(self, report: MultiSwingReport, is_backhand: bool = False) -> List[str]:
        """从所有击球中提取综合教练建议 — 简洁定性版。"""
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

        # 做得好的方面
        good = [kpi for _, avg, kpi in avg_scores[:3] if avg >= 50]
        if good:
            lines.append("**做得好的方面**: " + "、".join(kpi.name for kpi in good) + "。")
            lines.append("")

        # 最需要改进的方面
        weak = [kpi for _, avg, kpi in avg_scores[-3:] if avg < 80]
        if weak:
            lines.append("**最需要改进**: " + "、".join(kpi.name for kpi in weak) + "。")
            lines.append("")

        # 核心理念提醒
        if is_backhand:
            lines.append("记住单反的核心：保持侧身、充分伸展、非持拍手平衡。力量来自地面和转体，不是手臂。")
        else:
            lines.append("记住容错正手的核心：前方接触、髋带动前挥、向外向前穿过球。普通球不丢比偶尔一板神球更重要。")

        return lines

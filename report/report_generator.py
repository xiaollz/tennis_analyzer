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
        """生成完整的 Markdown 报告并返回文件路径。

        报告结构（v5 简化版）：
        1. 标题 + 基本信息
        2. 核心问题（多球时，自然段叙述）
        3. 每球分析：问题（含分数）→ 因果叙述 → 训练方法
        4. 量化辅助参考（所有球汇总，放最后）
        5. 下次训练叮嘱（简短）
        """
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
        _vlm = vlm_results or []

        # ── 标题 ─────────────────────────────────────────────────────
        lines.append(f"# {stroke_cn}技术分析")
        lines.append("")
        lines.append(f"**视频**: {video_name} | **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | **击球数**: {report.total_swings}")
        lines.append("")

        # ── 核心问题（多球时生成，自然段叙述）────────────────────────
        if report.total_swings > 1 and any(v for v in _vlm if v):
            lines.extend(self._multi_swing_summary(_vlm, report))

        # ── 每球分析 ─────────────────────────────────────────────────
        highlight_indices = self._pick_highlight_swings(_vlm, report)
        for ev in report.swing_evaluations:
            vlm_data = _vlm[ev.swing_index] if ev.swing_index < len(_vlm) else None
            is_highlight = ev.swing_index in highlight_indices
            lines.extend(self._swing_section(
                ev, chart_paths, report.total_swings,
                phase_titles, phase_order, stroke_cn, drills,
                vlm_result=vlm_data,
                compact=not is_highlight,
            ))

        # ── 量化辅助参考（所有球汇总，放在最后）─────────────────────
        all_metrics_lines = self._all_swings_metrics(report, _vlm)
        if all_metrics_lines:
            lines.append("---")
            lines.append("")
            lines.extend(all_metrics_lines)

        # ── 下次训练叮嘱 ─────────────────────────────────────────────
        lines.append("---")
        lines.append("")
        lines.extend(self._coaching_summary(report, is_backhand, vlm_results=_vlm))
        lines.append("")

        # 写入文件
        type_tag = "单反分析报告" if is_backhand else "正手分析报告"
        report_path = self.output_dir / f"{type_tag}_{video_name}.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    # ── 多球总结 + 重点球选择 ────────────────────────────────────────

    @staticmethod
    def _multi_swing_summary(vlm_results: list, report) -> List[str]:
        """Generate a coach-style narrative summary across all swings.

        Outputs a natural paragraph without bold formatting or lists.
        """
        lines = []
        lines.append("## 核心问题")
        lines.append("")

        # Collect root causes with swing indices
        def _strip_trailing_period(s: str) -> str:
            return s.rstrip("。.") if s else s

        swing_causes: List[tuple] = []  # (swing_1based, root_cause_text)
        for i, v in enumerate(vlm_results):
            if not v:
                continue
            tree = v.get("root_cause_tree", {})
            rc = tree.get("root_cause", "")
            if rc:
                swing_causes.append((i + 1, rc))
            elif v.get("issues"):
                swing_causes.append((i + 1, v["issues"][0].get("name", "")))

        if not swing_causes:
            return lines

        # Group by similar root causes (prefix-based dedup)
        short_to_full: Dict[str, str] = {}
        short_to_swings: Dict[str, List[int]] = {}
        for swing_num, rc in swing_causes:
            prefix = rc[:15]
            short_to_full.setdefault(prefix, rc)
            short_to_swings.setdefault(prefix, []).append(swing_num)

        # Sort groups by frequency (descending)
        groups = sorted(short_to_swings.items(), key=lambda x: -len(x[1]))
        main_prefix, main_swings = groups[0]
        main_cause = short_to_full[main_prefix]
        total = report.total_swings

        # Build a coach-style narrative paragraph (no bold, no lists)
        parts = []

        if len(main_swings) == total:
            parts.append(f"看完这 {total} 个球，每一个都指向同一件事：{_strip_trailing_period(main_cause)}。")
        elif len(main_swings) >= total * 0.6:
            swing_list = "、".join(str(s) for s in main_swings)
            parts.append(
                f"这 {total} 个球里，第 {swing_list} 球都有同一个核心问题：{_strip_trailing_period(main_cause)}。"
            )
        else:
            swing_list = "、".join(str(s) for s in main_swings)
            parts.append(
                f"第 {swing_list} 球暴露了一个关键问题：{_strip_trailing_period(main_cause)}。"
            )

        # Sentence 3: how the main problem manifests (use downstream symptoms if available)
        manifestations = []
        for swing_num, rc in swing_causes:
            if rc[:15] != main_prefix:
                continue
            v = vlm_results[swing_num - 1]
            if v and v.get("root_cause_tree", {}).get("downstream_symptoms"):
                for s in v["root_cause_tree"]["downstream_symptoms"][:2]:
                    symptom = s.get("symptom", "")
                    if symptom and symptom not in manifestations:
                        manifestations.append(symptom)
        if manifestations:
            parts.append(f"它导致了{'、'.join(manifestations[:3])}等表面现象。")

        # Sentence 4: secondary issue if exists
        if len(groups) > 1:
            sec_prefix, sec_swings = groups[1]
            sec_cause = short_to_full[sec_prefix]
            sec_list = "、".join(str(s) for s in sec_swings)
            parts.append(f"另外，第 {sec_list} 球还有一个不同的问题：{_strip_trailing_period(sec_cause)}。")

        lines.append(" ".join(parts))
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
        """生成单球分析。

        详细模式：问题（含分数）→ 因果叙述 → 训练方法
        Compact 模式：一行评分 + 根因
        """
        lines = []
        lines.append("---")
        lines.append("")

        if total_swings > 1:
            lines.append(f"## 第 {ev.swing_index + 1} 球")
        else:
            lines.append("## 击球分析")
        lines.append("")

        # ── Compact mode: one line ──
        if compact and vlm_result:
            score = vlm_result.get("score")
            tree = vlm_result.get("root_cause_tree", {})
            rc = tree.get("root_cause", "")
            if score and rc:
                lines.append(f"{score}/100 — {rc}")
            elif score:
                lines.append(f"{score}/100")
            elif rc:
                lines.append(rc)
            lines.append("")
            return lines

        # ── 详细模式：三部分结构 ──
        if vlm_result:
            lines.extend(self._vlm_section(vlm_result, chart_paths, ev.swing_index, total_swings))

        return lines

    # ── VLM 视觉分析章节（v5 简化：三部分结构）─────────────────────

    @staticmethod
    def _vlm_section(
        vlm_result: Dict,
        chart_paths: Dict[str, str],
        swing_index: int,
        total_swings: int,
    ) -> List[str]:
        """三部分结构：问题是什么 → 为什么 → 怎么解决。"""
        import os
        lines = []

        # Keyframe grid image
        grid_path = vlm_result.get("keyframe_grid_path", "")
        if grid_path:
            basename = os.path.basename(grid_path)
            rel_path = f"charts/{basename}"
            lines.append(f"![关键帧]({rel_path})")
            lines.append("")

        tree = vlm_result.get("root_cause_tree", {})

        # ── 第一部分：问题是什么（含分数）──
        score = vlm_result.get("score")
        core_diagnosis = vlm_result.get("overall_narrative", "")
        root_cause = tree.get("root_cause", "")

        if score is not None and root_cause:
            lines.append(f"**{score}/100** — {root_cause}")
        elif score is not None:
            lines.append(f"**{score}/100**")
        elif root_cause:
            lines.append(f"**{root_cause}**")
        lines.append("")

        # Core diagnosis paragraph (natural text, no bold)
        if core_diagnosis:
            lines.append(core_diagnosis)
            lines.append("")

        # Evidence line
        evidence = tree.get("root_cause_evidence", "")
        if evidence:
            lines.append(f"证据：{evidence}")
            lines.append("")

        # ── 第二部分：为什么（因果叙述段落）──
        causal = tree.get("causal_explanation", "")
        if causal:
            lines.append(causal)
            lines.append("")
        elif tree.get("downstream_symptoms"):
            # v3 fallback: build a paragraph from symptom list
            symptoms = tree["downstream_symptoms"]
            symptom_texts = []
            for s in symptoms:
                how = s.get("how_root_cause_creates_it", "")
                symptom_name = s.get("symptom", "")
                if how:
                    symptom_texts.append(f"{symptom_name}（{how}）")
                elif symptom_name:
                    symptom_texts.append(symptom_name)
            if symptom_texts:
                lines.append(f"{root_cause}导致了：{'，'.join(symptom_texts)}。")
                lines.append("")

        # Secondary root cause (inline)
        secondary = vlm_result.get("secondary_root_cause")
        if secondary and secondary.get("root_cause"):
            lines.append(f"另外注意：{secondary['root_cause']}")
            lines.append("")

        # ── 第三部分：怎么解决（训练方法）──
        fix = tree.get("fix", {})
        if fix and fix.get("one_drill"):
            lines.append(f"**练这个：{fix['one_drill']}**")
            method = fix.get("drill_method", "")
            cue = fix.get("drill_feel_cue", "")
            check = fix.get("check_criteria", "")
            if method:
                lines.append(f"做法：{method}")
            if cue:
                lines.append(f"口令：{cue}")
            if check:
                lines.append(f"做对标准：{check}")
            lines.append("")

        return lines

    # ── 诊断推理过程（保留为内部方法，报告中不再显示）──────────────────

    @staticmethod
    def _format_diagnostic_journey(session_data: Dict) -> List[str]:
        """Compress multi-round diagnostic journey into 1-2 sentence summary.

        Not shown in reports (v5), but kept for backward compatibility
        and internal use.
        """
        lines: List[str] = []
        rounds = session_data.get("rounds", [])
        hypotheses = session_data.get("hypotheses", [])

        if not rounds:
            return lines

        confirmed = [h for h in hypotheses if h.get("status") == "confirmed"]
        eliminated = [h for h in hypotheses if h.get("status") == "eliminated"]

        if not confirmed and not eliminated:
            return lines

        lines.append("**诊断路径：**")

        parts = []
        if eliminated:
            elim_names = [h.get("name_zh") or h.get("name", "?") for h in eliminated]
            parts.append(f"排除了{', '.join(elim_names)}")
        if confirmed:
            conf_names = [h.get("name_zh") or h.get("name", "?") for h in confirmed]
            confirm_reason = ""
            for rnd in reversed(rounds):
                for upd in rnd.get("hypothesis_updates", []):
                    if upd.get("action") == "confirm" and upd.get("reason"):
                        confirm_reason = upd["reason"][:60]
                        break
                if confirm_reason:
                    break
            if confirm_reason:
                parts.append(f"确认{', '.join(conf_names)}（{confirm_reason}）")
            else:
                parts.append(f"确认{', '.join(conf_names)}")

        num_rounds = len(rounds)
        prefix = f"经过 {num_rounds} 轮观察，" if num_rounds > 1 else ""
        lines.append(f"{prefix}{'，'.join(parts)}。")
        lines.append("")

        return lines

    # ── 量化辅助参考（所有球汇总）────────────────────────────────────

    def _all_swings_metrics(
        self,
        report: MultiSwingReport,
        vlm_results: list,
    ) -> List[str]:
        """把所有球的量化指标汇总到报告最后，不在每球内部显示。"""
        lines = []

        all_valid_kpis = []
        all_extra = []
        for ev in report.swing_evaluations:
            valid = [
                kpi for kpi in ev.kpi_results
                if kpi.rating not in ("无数据", "n/a") and kpi.raw_value is not None
            ]
            if valid:
                all_valid_kpis.append((ev.swing_index, valid))
            # supplementary metrics from VLM
            if ev.swing_index < len(vlm_results) and vlm_results[ev.swing_index]:
                sm = vlm_results[ev.swing_index].get("supplementary_metrics")
                if sm:
                    all_extra.append((ev.swing_index, sm))

        if not all_valid_kpis and not all_extra:
            return lines

        lines.append("## 量化辅助参考")
        lines.append("")

        for swing_idx, kpis in all_valid_kpis:
            if report.total_swings > 1:
                lines.append(f"**第 {swing_idx + 1} 球**")
            for kpi in kpis:
                val_str = self._format_value(kpi.raw_value, kpi.unit)
                note = self._metric_note(kpi)
                if kpi.raw_value is not None and kpi.raw_value < 0 and kpi.kpi_id == "C3.1":
                    lines.append(f"- {kpi.name}: {val_str} (可能受相机角度影响)")
                elif note:
                    lines.append(f"- {kpi.name}: {val_str} ({note})")
                else:
                    lines.append(f"- {kpi.name}: {val_str}")

            # Supplementary metrics for this swing
            for extra_idx, sm in all_extra:
                if extra_idx == swing_idx:
                    if sm.get("arm_torso_synchrony") is not None:
                        lines.append(f"- 手臂-躯干同步性: {sm['arm_torso_synchrony']:.2f} ({sm.get('arm_torso_sync_label', '')})")
                    if sm.get("wrist_v_detected") is not None:
                        if sm["wrist_v_detected"]:
                            lines.append(f"- 手腕高度模式: V形scooping (深度 {sm.get('wrist_v_depth', 0):.2f})")
                        else:
                            lines.append("- 手腕高度模式: 正常")
                    if sm.get("swing_shape_label") is not None:
                        lines.append(f"- 挥拍轨迹: {sm['swing_shape_label']} (弧度比 {sm.get('swing_arc_ratio', 0):.2f})")
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

    def _coaching_summary(
        self,
        report: MultiSwingReport,
        is_backhand: bool = False,
        vlm_results: Optional[List[Optional[Dict]]] = None,
    ) -> List[str]:
        """简短叮嘱：下次练什么，一句话。"""
        lines = []
        _vlm = vlm_results or []

        primary_fix = ""
        primary_root = ""
        for v in _vlm:
            if not v:
                continue
            tree = v.get("root_cause_tree", {})
            if tree.get("root_cause"):
                primary_root = tree["root_cause"]
                fix = tree.get("fix", {})
                if fix.get("one_drill"):
                    primary_fix = fix["one_drill"]
                break

        if primary_root and primary_fix:
            pr = primary_root.rstrip("。.")
            lines.append(
                f"下次训练只抓一件事：{primary_fix}。"
                f"因为{pr}是你目前所有问题的源头，解决了它，捞球、缺 Out、随挥塌陷会连带消失。"
            )
        elif primary_root:
            lines.append(
                f"下次训练专攻{primary_root}。一次只改一件事，改透了再往下走。"
            )
        else:
            lines.append("一次只改一件事，改透了再往下走。")

        return lines

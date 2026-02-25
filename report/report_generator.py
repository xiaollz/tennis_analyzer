"""Markdown 报告生成器 v3 — 支持正手8阶段 & 单反评估 + 训练处方。

v3 升级：
    - 正手从 6 阶段升级到 8 阶段
    - 新增训练处方系统（每个 KPI 都有对应的训练方法和体感提示）
    - 新增逐阶段诊断和优先级排序
    - 新增多次击球一致性分析
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from evaluation.forehand_evaluator import MultiSwingReport, SwingEvaluation
from evaluation.kpi import KPIResult


class ReportGenerator:
    """生成中文 Markdown 评估报告 — 支持正手8阶段 & 单反。"""

    # ── 正手 8 阶段标题（v3 升级）────────────────────────────────────
    FOREHAND_PHASE_TITLES = {
        "unit_turn":   "阶段一：一体化转体 (Unit Turn)",
        "slot_prep":   "阶段二：槽位准备 (Slot Preparation)",
        "leg_drive":   "阶段三：蹬转与髋部启动 (Leg Drive)",
        "torso_pull":  "阶段四：躯干与肩部牵引 (Torso Pull)",
        "lag_drive":   "阶段五：滞后与肘部驱动 (Lag & Elbow Drive)",
        "contact":     "阶段六：击球与肩内旋 (Contact & SIR)",
        "wiper":       "阶段七：雨刷式随挥 (Wiper Follow-Through)",
        "balance":     "阶段八：减速与平衡 (Deceleration & Balance)",
    }
    FOREHAND_PHASE_ORDER = [
        "unit_turn", "slot_prep", "leg_drive", "torso_pull",
        "lag_drive", "contact", "wiper", "balance",
    ]

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
        # Phase 1: Unit Turn
        "P1.1": {
            "drill": "整体转体练习：双手持拍在胸前，蹬地→转髋→转肩一气呵成，背部面向球网。",
            "feel": "感觉像是用肚脐带动整个身体旋转，而不是手臂拉拍。",
            "cue": "Rick Macci: 「Turn as a unit — hips, shoulders, racket all together.」",
        },
        "P1.2": {
            "drill": "分步→蓄力站位练习：做 Split Step 后立即屈膝降低重心，膝盖弯曲约 120°。",
            "feel": "感觉像坐在高脚凳上，大腿有明显的蓄力感。",
            "cue": "想象你在弹簧上，先压缩再释放。",
        },
        "P1.3": {
            "drill": "脊柱直立练习：头顶放一本书做影子挥拍，全程保持书不掉落。",
            "feel": "感觉头顶有一根线向上拉，脊柱始终保持自然直立。",
            "cue": "Tennis Doctor: 「Head stays still — it's the axis of rotation.」",
        },
        # Phase 2: Slot Preparation
        "S2.1": {
            "drill": "肘部后撤练习：转体后，用非持拍手推肘部向后，直到肘部在躯干后方。",
            "feel": "感觉像是把肘部「钉」在身后，胸肌有轻微拉伸感。",
            "cue": "Rick Macci: 「Elbow back, elbow back!」肘部是启动前挥的钥匙。",
        },
        "S2.2": {
            "drill": "拍头下垂练习：在 Slot 位置，放松手腕让拍头自然下垂到膝盖以下。",
            "feel": "感觉手腕像一个松弛的铰链，拍头因重力自然下垂。",
            "cue": "「Pat the Dog」— 想象你在身侧拍一只小狗的头。",
        },
        # Phase 3: Leg Drive
        "L3.1": {
            "drill": "蹬地力量练习：从屈膝位置用力蹬地向上跳，感受地面反作用力传递到髋部。",
            "feel": "感觉力量从脚底板→小腿→大腿→髋部向上传递，像弹簧释放。",
            "cue": "Dr. Gordon: 「Power starts from the ground — push the earth away.」",
        },
        "L3.2": {
            "drill": "髋部旋转速度练习：双手叉腰，快速转髋但保持肩膀不动，感受髋肩分离。",
            "feel": "感觉像拧毛巾——下半身先转，上半身被拉扯。",
            "cue": "「Hip fires first」— 髋部是整个动力链的引擎。",
        },
        # Phase 4: Torso Pull
        "T4.1": {
            "drill": "髋肩分离练习：面对墙壁，髋部贴墙转动，肩膀保持不动，感受躯干拉伸。",
            "feel": "感觉腹斜肌和背部有明显的拉伸-收缩感，像弹弓被拉开。",
            "cue": "「X-Factor」— 髋肩之间的角度差越大，储存的弹性能量越多。",
        },
        "T4.2": {
            "drill": "髋肩时序练习：慢动作挥拍，先转髋 → 停顿 → 再转肩，感受时间差。",
            "feel": "感觉肩膀是被髋部「拖」着走的，不是主动转动。",
            "cue": "Dr. Gordon: 「Hip leads by 40-80ms — this is where the power comes from.」",
        },
        # Phase 5: Lag & Elbow Drive
        "D5.1": {
            "drill": "肘部驱动练习：肘部贴近身体向前推，像出拳一样，手腕和拍头完全被动。",
            "feel": "感觉肘部像一个活塞向前推进，手腕和球拍像鞭子的末端被甩出去。",
            "cue": "Rick Macci: 「Lead with the elbow — the racket follows.」",
        },
        "D5.2": {
            "drill": "手部路径线性度练习：在地上画一条直线，沿线做影子挥拍，手腕应沿直线运动。",
            "feel": "感觉手在击球区走一条笔直的通道，不是弧线。",
            "cue": "Tennis Doctor: 「The hand path through contact should be linear.」",
        },
        # Phase 6: Contact & SIR
        "C6.1": {
            "drill": "击球点位置练习：抛球并在身体前方一臂距离处接住，那就是理想击球点。",
            "feel": "感觉击球时手臂有充分的伸展空间，不是被挤压在身体旁边。",
            "cue": "「Contact in front of the front hip」— 击球点在前脚髋部的前方。",
        },
        "C6.2": {
            "drill": "击球时手臂形态练习：直臂型充分伸展，双弯型保持稳固的L形。",
            "feel": "直臂型：感觉像推门；双弯型：感觉像甩毛巾。",
            "cue": "两种风格都正确，关键是击球时手臂形态稳定。",
        },
        "C6.3": {
            "drill": "身体刹车练习：击球时想象胸部撞上一面玻璃墙，身体突然停止旋转。",
            "feel": "感觉核心肌群突然收紧，身体停止但手臂继续向前。",
            "cue": "「Body brakes, arm accelerates」— 身体的突然减速让手臂获得鞭打效应。",
        },
        "C6.4": {
            "drill": "头部稳定练习：击球后保持眼睛注视击球点，数到1再抬头。",
            "feel": "感觉头部像一个固定的轴心，身体围绕它旋转。",
            "cue": "Federer 的头在击球后始终保持不动——这是控制的关键。",
        },
        "C6.5": {
            "drill": "肩内旋感知练习：手臂水平伸直，快速向内旋转（像开门把手），感受肩部深层肌肉。",
            "feel": "感觉是「用胸肌和后背打球」，不是手腕或前臂。肩内旋的力量来自胸大肌和背阔肌。",
            "cue": "Dr. Gordon: 「SIR is the most important power source — it's automatic if the kinetic chain is correct.」",
        },
        # Phase 7: Wiper Follow-Through
        "W7.1": {
            "drill": "前向延伸练习：击球后，将手向目标方向推送 60-90cm，然后再让球拍上升。",
            "feel": "感觉像是「穿过球」打，不是「打到球」就停。",
            "cue": "「Extend through the ball」— 延伸是旋转和深度的来源。",
        },
        "W7.2": {
            "drill": "雨刷式收拍练习：击球后，前臂快速旋前（像雨刷器），拍头从右向左扫过。",
            "feel": "感觉前臂像拧毛巾一样旋转，拍头自然从右侧扫到左侧。",
            "cue": "「Windshield wiper」— 这个动作产生重上旋，是现代正手的标志。",
        },
        # Phase 8: Balance
        "B8.1": {
            "drill": "全程头部稳定练习：在头上放一本书做完整挥拍，全程保持平衡。",
            "feel": "感觉头部是整个身体的「锚点」，所有旋转都围绕它进行。",
            "cue": "「Head is the axis」— 头部稳定是所有好击球的基础。",
        },
        "B8.2": {
            "drill": "脊柱一致性练习：练习挥拍时保持腰带扣高度不变，不要上下起伏。",
            "feel": "感觉像在一个固定高度的平面上旋转，不是上下弹跳。",
            "cue": "脊柱角度的一致性反映了核心稳定性和动作的可重复性。",
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
    ) -> str:
        """生成完整的 Markdown 报告并返回文件路径。"""
        chart_paths = chart_paths or {}
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
            lines.append(f"**评估模型**: 8 阶段 Modern Forehand (v3)  ")
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
        for ev in report.swing_evaluations:
            lines.extend(self._swing_section(
                ev, chart_paths, report.total_swings,
                phase_titles, phase_order, stroke_cn, drills,
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
            lines.append("本分析基于 **Modern Forehand v3** 理论框架（8阶段模型），综合以下来源：")
            lines.append("- **Dr. Brian Gordon** — Type 3 正手生物力学、SIR（肩内旋）、直臂延伸")
            lines.append("- **Rick Macci** — 紧凑转体、肘部后撤与驱动、「翻转」技术")
            lines.append("- **Tennis Doctor** — 四大不可妥协原则、动力链顺序、身体刹车")
            lines.append("- **Feel Tennis** — 现代正手8步模型、用身体打球")
            lines.append("")
            lines.append("**8 阶段模型**：一体化转体 → 槽位准备 → 蹬转启动 → 躯干牵引 → 滞后驱动 → 击球与SIR → 雨刷随挥 → 减速平衡")
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
        drills: Dict,
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

        avg_scores = [(kpi_id, float(sum(scores) / len(scores)), kpi_map[kpi_id])
                      for kpi_id, scores in kpi_avg.items()]
        avg_scores.sort(key=lambda x: x[1])

        # 找出需要训练的 KPI（分数 < 70）
        weak_kpis = [(kid, avg, kpi) for kid, avg, kpi in avg_scores if avg < 70]

        if not weak_kpis:
            lines.append("所有指标均达到良好水平，继续保持！可以尝试在更高强度的对抗中保持技术质量。")
            return lines

        lines.append("以下是按优先级排序的训练计划，从最需要改进的指标开始：")
        lines.append("")

        for priority, (kid, avg, kpi) in enumerate(weak_kpis[:5], 1):
            lines.append(f"### 优先级 {priority}：{kpi.name}（平均 {avg:.0f} 分）")
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
        lines.append("1. **热身**（5分钟）：慢速影子挥拍，专注于完整的 8 阶段动作链。")
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
            if "度" in unit or "°" in unit:
                return f"{value:.1f}°"
            if "比" in unit or "ratio" in unit or "R²" in unit:
                return f"{value:.2f}"
            if "归一化" in unit or "标准差" in unit or "norm" in unit:
                return f"{value:.3f}"
            if "px/s²" in unit:
                return f"{value:.0f} {unit}"
            if "°/s" in unit:
                return f"{value:.0f} {unit}"
            if "ms" in unit or "秒" in unit:
                return f"{value:.0f} {unit}"
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
            lines.append("- **用身体打球**：力量来自地面反作用力 → 蹬转 → 髋部旋转 → 躯干牵引 → 肩内旋，手臂只是传递工具。")
            lines.append("- **动力链时序**：髋部先于肩部 40-80ms 启动旋转，这个时间差是力量的来源。")
            lines.append("- **身体刹车**：击球时身体突然减速，将动量传递给手臂和球拍（鞭打效应）。")
            lines.append("- **放松手臂**：手臂越放松，鞭打效应越强。主动发力的感觉应该在胸肌和背部，不是手臂。")

        return lines

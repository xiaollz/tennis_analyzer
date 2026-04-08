"""VLM-based keyframe analysis for tennis forehand.

Supports multiple VLM providers via config/vlm_config.json:
  - openai_compatible: Qwen-VL, GPT-4o, any OpenAI-compatible proxy
  - anthropic: Claude
  - gemini: Google Gemini
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from evaluation.event_detector import SwingEvent

# Optional knowledge-graph imports (graceful degradation when unavailable)
try:
    from knowledge.output.vlm_prompt import VLMPromptCompiler
    from knowledge.graph import KnowledgeGraph
    from knowledge.schemas import DiagnosticChain
    from knowledge.schemas import (
        Hypothesis, HypothesisStatus, Observation, ObservationJudgment,
        HypothesisUpdate, RoundResult, DiagnosticSession,
    )

    _HAS_KNOWLEDGE = True
except ImportError:
    _HAS_KNOWLEDGE = False


# ── Config ──────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "vlm_config.json"


def load_vlm_config() -> Dict:
    """Load VLM configuration from config/vlm_config.json."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Strip internal doc keys
        return {k: v for k, v in cfg.items() if not k.startswith("_")}
    except Exception as exc:
        print(f"[VLM] 警告: 读取配置失败 ({exc})")
        return {}


# ── Keyframe Labels ─────────────────────────────────────────────────

KEYFRAME_LABELS: List[Tuple[str, str]] = [
    ("prep_complete", "准备完成"),
    ("forward_swing_start", "前挥启动"),
    ("pre_contact", "击球前"),
    ("contact", "击球瞬间"),
    ("early_followthrough", "随挥初期"),
    ("followthrough_end", "随挥结束"),
]


# ── KeyframeExtractor ───────────────────────────────────────────────

class KeyframeExtractor:
    """Extract 6 key frames from a single swing, with optional issue annotations."""

    # Orange colour for issue highlights (BGR)
    _ISSUE_COLOR = (0, 140, 255)

    def extract(
        self,
        frames_raw: List[np.ndarray],
        frame_indices: List[int],
        swing_event: SwingEvent,
        keypoints_series: Optional[List[np.ndarray]] = None,
        confidence_series: Optional[List[np.ndarray]] = None,
        is_right_handed: bool = True,
    ) -> List[Tuple[str, np.ndarray]]:
        if not frames_raw or not frame_indices:
            return []

        idx_map = {f: i for i, f in enumerate(frame_indices)}
        n = len(frames_raw)

        prep_frame = swing_event.prep_start_frame
        impact_frame = swing_event.impact_frame
        ft_end_frame = swing_event.followthrough_end_frame

        prep_pos = idx_map.get(prep_frame, 0) if prep_frame is not None else 0
        impact_pos = idx_map.get(impact_frame, n // 2) if impact_frame is not None else n // 2
        ft_end_pos = idx_map.get(ft_end_frame, n - 1) if ft_end_frame is not None else n - 1

        prep_pos = max(0, min(prep_pos, n - 1))
        impact_pos = max(0, min(impact_pos, n - 1))
        ft_end_pos = max(0, min(ft_end_pos, n - 1))

        positions = [
            prep_pos,
            (prep_pos + impact_pos) // 2,
            max(0, impact_pos - 3),
            impact_pos,
            min(n - 1, impact_pos + 5),
            ft_end_pos,
        ]

        # Pre-compute joint trajectories ONLY within this swing's range
        # (prep_pos → ft_end_pos) to avoid leaking trails from adjacent swings.
        swing_start = max(0, prep_pos)
        swing_end = min(ft_end_pos + 1, n)

        wrist_trail: List[Optional[Tuple[int, int]]] = []
        elbow_trail: List[Optional[Tuple[int, int]]] = []
        if keypoints_series is not None and confidence_series is not None:
            from config.keypoints import KEYPOINT_NAMES
            side = "right" if is_right_handed else "left"
            wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
            elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
            for i in range(swing_start, swing_end):
                if float(confidence_series[i][wrist_idx]) >= 0.3:
                    wrist_trail.append((int(keypoints_series[i][wrist_idx][0]), int(keypoints_series[i][wrist_idx][1])))
                else:
                    wrist_trail.append(None)
                if float(confidence_series[i][elbow_idx]) >= 0.3:
                    elbow_trail.append((int(keypoints_series[i][elbow_idx][0]), int(keypoints_series[i][elbow_idx][1])))
                else:
                    elbow_trail.append(None)

        # Max pixel distance between consecutive trail points before breaking
        # the line (prevents spurious cross-swing connections).
        _MAX_TRAIL_GAP_PX = 150

        result: List[Tuple[str, np.ndarray]] = []
        for (label, _cn), pos in zip(KEYFRAME_LABELS, positions):
            pos = max(0, min(pos, n - 1))
            frame = frames_raw[pos].copy()

            # Draw joint trajectories: sliding window of last ~15 frames (0.5s at 30fps)
            # Only show recent trail, not full accumulation from swing start.
            _TRAIL_WINDOW = 15
            trail_end_rel = max(0, min(pos - swing_start + 1, len(wrist_trail)))
            trail_start_rel = max(0, trail_end_rel - _TRAIL_WINDOW)

            # Elbow: cyan trail (windowed)
            if elbow_trail:
                pts_e = [p for p in elbow_trail[trail_start_rel:trail_end_rel] if p is not None]
                if len(pts_e) >= 2:
                    for i in range(1, len(pts_e)):
                        dx = pts_e[i][0] - pts_e[i-1][0]
                        dy = pts_e[i][1] - pts_e[i-1][1]
                        if dx*dx + dy*dy > _MAX_TRAIL_GAP_PX * _MAX_TRAIL_GAP_PX:
                            continue
                        alpha = 0.4 + 0.6 * (i / len(pts_e))
                        cv2.line(frame, pts_e[i-1], pts_e[i], (int(200*alpha), int(200*alpha), 0), 2, cv2.LINE_AA)

            # Wrist: yellow trail (windowed, drawn on top)
            if wrist_trail:
                pts_w = [p for p in wrist_trail[trail_start_rel:trail_end_rel] if p is not None]
                if len(pts_w) >= 2:
                    for i in range(1, len(pts_w)):
                        dx = pts_w[i][0] - pts_w[i-1][0]
                        dy = pts_w[i][1] - pts_w[i-1][1]
                        if dx*dx + dy*dy > _MAX_TRAIL_GAP_PX * _MAX_TRAIL_GAP_PX:
                            continue
                        alpha = 0.4 + 0.6 * (i / len(pts_w))
                        cv2.line(frame, pts_w[i-1], pts_w[i], (0, int(255*alpha), int(255*alpha)), 2, cv2.LINE_AA)

            # Annotate issues if keypoint data available
            if keypoints_series is not None and confidence_series is not None:
                frame = self._annotate_issues(
                    frame, pos, keypoints_series, confidence_series,
                    is_right_handed, label,
                )
                frame = self._draw_angle_annotations(
                    frame, pos, keypoints_series, confidence_series,
                    is_right_handed, label,
                )

            result.append((label, frame))
        return result

    def _annotate_issues(
        self,
        frame: np.ndarray,
        pos: int,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        is_right_handed: bool,
        phase_label: str,
    ) -> np.ndarray:
        """Draw orange bounding boxes around detected issues on the frame."""
        from config.keypoints import KEYPOINT_NAMES
        from analysis.kinematic_calculator import wrist_below_elbow_distance

        # Only annotate forward swing / pre-contact / contact phases
        if phase_label not in ("forward_swing_start", "pre_contact", "contact"):
            return frame

        side = "right" if is_right_handed else "left"
        wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
        elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
        shoulder_idx = KEYPOINT_NAMES[f"{side}_shoulder"]

        # Check wrist drop over a ±3 frame window (robust to single-frame noise)
        n = len(keypoints_series)
        max_drop = 0.0
        for i in range(max(0, pos - 3), min(n, pos + 4)):
            wd = wrist_below_elbow_distance(keypoints_series[i], confidence_series[i], is_right_handed)
            if wd is not None and wd > max_drop:
                max_drop = wd

        # Threshold: wrist more than 0.15 torso-heights below elbow
        if max_drop > 0.15:
            kp = keypoints_series[pos]
            conf = confidence_series[pos]
            if conf[wrist_idx] > 0.3 and conf[elbow_idx] > 0.3 and conf[shoulder_idx] > 0.3:
                pts = np.array([kp[shoulder_idx], kp[elbow_idx], kp[wrist_idx]], dtype=np.int32)
                margin = 30
                x1 = max(0, int(pts[:, 0].min()) - margin)
                y1 = max(0, int(pts[:, 1].min()) - margin)
                x2 = min(frame.shape[1], int(pts[:, 0].max()) + margin)
                y2 = min(frame.shape[0], int(pts[:, 1].max()) + margin)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self._ISSUE_COLOR, 3)

        return frame

    def _draw_angle_annotations(
        self,
        frame: np.ndarray,
        pos: int,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        is_right_handed: bool,
        phase_label: str,
    ) -> np.ndarray:
        """Draw elbow angle arc, shoulder line, and hip-shoulder separation."""
        from config.keypoints import KEYPOINT_NAMES
        from analysis.kinematic_calculator import elbow_angle

        kp = keypoints_series[pos]
        conf = confidence_series[pos]
        side = "right" if is_right_handed else "left"

        shoulder_idx = KEYPOINT_NAMES[f"{side}_shoulder"]
        elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
        wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
        l_shoulder_idx = KEYPOINT_NAMES["left_shoulder"]
        r_shoulder_idx = KEYPOINT_NAMES["right_shoulder"]
        l_hip_idx = KEYPOINT_NAMES["left_hip"]
        r_hip_idx = KEYPOINT_NAMES["right_hip"]

        conf_thr = 0.3

        # 1. Elbow angle arc on contact frame (only draw if HIGH confidence)
        if phase_label == "contact":
            elbow_conf_thr = 0.7  # Higher threshold — low confidence wrist causes wild angle errors
            if (float(conf[shoulder_idx]) >= elbow_conf_thr
                    and float(conf[elbow_idx]) >= elbow_conf_thr
                    and float(conf[wrist_idx]) >= elbow_conf_thr):
                angle = elbow_angle(kp, conf, right=is_right_handed)
                if angle is not None:
                    elbow_pt = (int(kp[elbow_idx][0]), int(kp[elbow_idx][1]))
                    shoulder_pt = np.array([kp[shoulder_idx][0], kp[shoulder_idx][1]], dtype=np.float64)
                    wrist_pt = np.array([kp[wrist_idx][0], kp[wrist_idx][1]], dtype=np.float64)
                    elbow_arr = np.array([kp[elbow_idx][0], kp[elbow_idx][1]], dtype=np.float64)

                    # Vectors from elbow to shoulder and elbow to wrist
                    v_shoulder = shoulder_pt - elbow_arr
                    v_wrist = wrist_pt - elbow_arr

                    # Compute start angle and sweep for the arc
                    start_deg = float(np.degrees(np.arctan2(v_shoulder[1], v_shoulder[0])))
                    end_deg = float(np.degrees(np.arctan2(v_wrist[1], v_wrist[0])))

                    # Normalize sweep to draw the smaller arc
                    sweep = end_deg - start_deg
                    if sweep > 180:
                        sweep -= 360
                    elif sweep < -180:
                        sweep += 360

                    # Color: green if 120-160 degrees, red otherwise
                    if 100 <= angle <= 170:
                        arc_color = (0, 255, 0)
                    else:
                        arc_color = (0, 0, 255)

                    arc_radius = 30
                    cv2.ellipse(frame, elbow_pt, (arc_radius, arc_radius),
                                0, start_deg, start_deg + sweep, arc_color, 2, cv2.LINE_AA)

                    # Angle text offset from elbow
                    text_offset_x = int(15 * np.cos(np.radians((start_deg + start_deg + sweep) / 2)))
                    text_offset_y = int(15 * np.sin(np.radians((start_deg + start_deg + sweep) / 2)))
                    text_pos = (elbow_pt[0] + text_offset_x - 15, elbow_pt[1] + text_offset_y - 10)
                    cv2.putText(frame, f"{angle:.0f}\xb0", text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, arc_color, 2, cv2.LINE_AA)

        # 2. Shoulder line on prep_complete frame
        if phase_label == "prep_complete":
            if (float(conf[l_shoulder_idx]) >= conf_thr
                    and float(conf[r_shoulder_idx]) >= conf_thr):
                ls = (int(kp[l_shoulder_idx][0]), int(kp[l_shoulder_idx][1]))
                rs = (int(kp[r_shoulder_idx][0]), int(kp[r_shoulder_idx][1]))
                cv2.line(frame, ls, rs, (255, 200, 100), 2, cv2.LINE_AA)  # light blue (BGR)

        # 3. Hip + shoulder lines on forward_swing_start frame
        if phase_label == "forward_swing_start":
            has_shoulders = (float(conf[l_shoulder_idx]) >= conf_thr
                             and float(conf[r_shoulder_idx]) >= conf_thr)
            has_hips = (float(conf[l_hip_idx]) >= conf_thr
                        and float(conf[r_hip_idx]) >= conf_thr)
            if has_hips:
                lh = (int(kp[l_hip_idx][0]), int(kp[l_hip_idx][1]))
                rh = (int(kp[r_hip_idx][0]), int(kp[r_hip_idx][1]))
                cv2.line(frame, lh, rh, (0, 255, 255), 2, cv2.LINE_AA)  # yellow (BGR)
            if has_shoulders:
                ls = (int(kp[l_shoulder_idx][0]), int(kp[l_shoulder_idx][1]))
                rs = (int(kp[r_shoulder_idx][0]), int(kp[r_shoulder_idx][1]))
                cv2.line(frame, ls, rs, (255, 100, 100), 2, cv2.LINE_AA)  # blue-ish (BGR)

        return frame


# ── Grid creation ───────────────────────────────────────────────────

def _find_cjk_font():
    """Find a CJK font available on macOS / Linux."""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def create_keyframe_grid(
    keyframes: List[Tuple[str, np.ndarray]],
    cell_width: int = 360,
    cell_height: int = 270,
    label_height: int = 30,
) -> np.ndarray:
    """Arrange up to 6 keyframes into a 2x3 grid with labels."""
    from PIL import Image, ImageDraw, ImageFont

    cols, rows = 3, 2
    total_w = cols * cell_width
    total_h = rows * (cell_height + label_height)

    pil_img = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(pil_img)

    font_path = _find_cjk_font()
    try:
        font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    cn_labels = [cn for (_, cn) in KEYFRAME_LABELS]

    for idx, (label, frame) in enumerate(keyframes[:6]):
        row, col = divmod(idx, cols)
        x0 = col * cell_width
        y0 = row * (cell_height + label_height)

        # Label bar
        draw.rectangle([x0, y0, x0 + cell_width, y0 + label_height], fill=(40, 40, 40))
        cn_text = cn_labels[idx] if idx < len(cn_labels) else label
        bbox = draw.textbbox((0, 0), cn_text, font=font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (cell_width - tw) // 2
        ty = y0 + (label_height - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), cn_text, fill=(255, 255, 255), font=font)

        # Frame
        resized = cv2.resize(frame, (cell_width, cell_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(rgb)
        pil_img.paste(frame_pil, (x0, y0 + label_height))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def save_keyframe_grid(grid: np.ndarray, path: str) -> str:
    cv2.imwrite(path, grid)
    return path


# ── FTT Prompt ──────────────────────────────────────────────────────

_FTT_SYSTEM_PROMPT = """\
你是一位精通《The Fault Tolerant Forehand》(FTT)理论的专业网球教练。

你将看到一张2×3网格图，展示一次正手挥拍的6个关键帧：
  图1（左上）准备完成  图2（中上）前挥启动  图3（右上）击球前
  图4（左下）击球瞬间  图5（中下）随挥初期  图6（右下）随挥结束

图中标注：
- 黄色线 = 手腕轨迹（平缓斜上的Nike Swoosh弧线=好的上旋路径，V形急降急升=scooping，内缩=缺Out。注意：正常的由低到高不是问题，只有急剧的V形才是scooping）
- 青色线 = 肘部轨迹（应领先手腕=正确动力链）
- 橙色框 = 算法检测到的问题区域
- 绿色/红色弧线 = 肘角（绿色100-170°=好，红色=异常）
- 肩线/髋线 = 肩部和髋部连线（可见髋肩分离角）

【逐帧分析指南】

图1 准备完成：
- 是否完成整体转身（Unit Turn）？双肩在同一水平面整体转？（不是右肩下拉左肩不动）
- 重心是否加载在右脚？（检验：左脚应该能抬起来。如果重心在左脚=Unit Turn做歪了）
- 双臂是否同步？（非持拍手应指向来球方向，不能早早放下）
- 非持拍肩是否高于持拍肩（肩部倾斜Tilt，10-30°）？注意：Tilt是蹬地时才做的，Unit Turn阶段应保持双肩水平
- 膝关节弯曲蓄力？还是直立？（双弯曲运动员姿态：髋+膝同时弯曲）
- 手部位置在右髋/口袋附近（Hand Slot）？还是拉到身后？
- 引拍简洁？球拍在视线余光范围内？
- 肩膀是否松沉在关节窝内？（耸肩=能量断裂）

图2 前挥启动：
- 躯干（肩线）是否先发出旋转意图？还是手臂先动了？（关键：肩线先启动旋转，腿部在中途爆发式加入放大——不是先蹬地再转身）
- 重心转移时序：重心是否已经开始从右脚前移？（Right→Left→Hit，重心转移必须先于挥拍）
- 后脚是否在地面上拧转（Pivot）？还是腾空跳起？（拧转=正确，腾空=错误，力量来自"拧"不是"跳"）
- 小臂是否有独立下压（pat the dog错误）？看黄色轨迹是否有V形尖角
- 非持拍手是否已经开始向胸部回收？（应该和蹬地同步开始收，不是等到最后）
- 手臂整体放松跟随身体？还是在主动做事？

图3 击球前（弹性释放阶段）：
- 拍头自然在手部下方（被动lag）？还是过度下坠？
- 黄色轨迹判断——看两个特征：①轨迹是圆滑弧线（Swoosh，正确）还是有V形尖角拐点（scooping）？②拍头比球低多少——半个拍头以内=正确，超过一个拍头=过度下坠
- 拍面朝向：Top Edge Leading（微闭/垂直）=正确的上旋路径；Bottom Edge Leading（朝天）=在撩球
- 非持拍手开始向胸部收拢（制动准备）？
- 球拍上边缘是否领先下边缘？（Top Edge Leading=正确上旋；Bottom Edge Leading=底边撩球=错误）

图4 击球瞬间（40ms释放窗口）：
- 是否到达了Press Slot（压力槽）？手掌面朝前下方，球拍从这个位置向前滚动产生上旋
- 击球点在身体前方？（手腕在前脚延长线前方）
- 肘部空间（Elbow Space）：腋下有足够空间？肘部紧贴身体=无法利用胸大肌=只能靠手腕
- 肘角100-170°？（看绿色/红色弧线标注）
- 头部固定，下巴贴近前肩？
- 躯干是否已停止旋转？（躯干冻结+手臂加速=高效动量交接；躯干仍转=动量泄漏）
- 有没有过度转动？（身体应保持一定侧身，不要过早完全正面朝网）
- 拍面接近垂直或微闭？（Top Edge Leading）

图5 随挥初期：
- 有向前延展（Through）？还是立刻向上收拍？
- 黄色轨迹向外延伸（Out）？还是向内收缩？（Out不是"推"出去的，是圆弧运动的自然切线分量）
- 非持拍手已收拢至胸部完成制动？还是向后甩开或垂在下方？
- 雨刷器翻转在击球之后开始（正确）？还是击球瞬间就发生？
- 拍面是渐进式对正（平滑过渡）还是剧烈翻转？（渐进=容错高，剧烈=时机难抓）

图6 随挥结束：
- "盒子"检验：大臂、小臂、球拍和胸部是否形成规则矩形？（盒子方正=击球点在前面+轨迹正确；盒子塌陷/包裹脖子=击球点靠后或捞球）
- 持拍肩越过非持拍肩？（肩胛骨前伸完成）
- 身体保持平衡？重心前倾（正向平衡）？
- 后脚状态：脚尖点地（正确，完成了Pivot）？还是仍然平踩或腾空？

【核心原则】
1. 正手是旋转驱动的鞭打系统。手臂是传递者，不是发动机。
2. 网球基本定理：拍面-前臂90-135° + 前方击球。
3. Out向量最关键——手向外推触发球拍绕质心旋转，产生上旋。
4. 发力心理模型（极重要）：旋转的意图由躯干（肩线）发出，腿部/臀部是力量放大器（Multiplier），不是发起者。正确的时序是"躯干发令→腿部在旋转中途爆发式放大→力量穿过手臂"。不要建议"先蹬地再转身"的阶梯式思维——那会导致躯干落后、手臂代偿。旋转是一个短脉冲（Pulse），终点是击球瞬间（Press Slot），不是随挥终点。随挥是惯性的自然结果。
   硬件要求：胸肌将手臂"焊接"在躯干上（大臂内侧贴着胸侧不分开），这个连接必须在Unit Turn时就建立。判断：图3-4中手臂是随躯干同步前移（连接正确）还是滞后于躯干然后猛追（脱节=代偿前兆）。
   诊断过度转体：如果图4击球瞬间身体已经正面朝网=旋转脉冲太长，终点设错了。正确时身体应保持侧身。
5. 绝不建议"制造lag"或"主动做pat the dog"——拍头下落和lag都是被动结果。
6. 躯干（肩线）先发出旋转意图，腿部在旋转中途爆发式加入放大力量。手臂不能先于躯干运动。不要建议"先蹬地再转身"的阶梯式思维。
7. 动量交接：躯干在击球瞬间冻结→全部动能甩入手臂。躯干仍转=动量泄漏。
8. 力量第一因素是甜区接触（偏离损失30-70%），生物力学优化是次要的。
9. 上旋需要从低到高的挥拍路径——这是物理必要条件，不是错误。绝不要把正常的由低到高判定为问题。用以下具体特征区分正确与错误：
   ✅ 正确（Nike Swoosh）：拍头只比球低半个拍头以内；黄色轨迹是圆滑弧线无尖角拐点；Top Edge Leading（拍面微闭/垂直）；从低到高的过程占挥拍路径70%以上
   ❌ 错误（V形scooping）：拍头低于球超过一个拍头距离；黄色轨迹有明显V形尖角（方向突变）；Bottom Edge Leading（拍面朝天）；V形底部出现在挥拍前半段（先掉再拉）
10. "放松"的正确含义：手腕和前臂放松（铰链），但肩胛骨和背阔肌必须保持基础张力——它们把手臂"托住"在身体旋转系统上。如果整条手臂全松，球拍重力会把手往下拽，造成V形scooping。正确的放松="末端松、根部稳"。纠正建议时请明确区分这两层。
11. Press Slot（压力槽）：每次正确正手在击球前都会到达一个特定位置——球拍下沉，手掌面朝前下方。无论引拍风格多不同，所有职业选手在进入击球区瞬间都到达这同一个位置。这是产生上旋的"发射台"。
12. 肘部空间（Elbow Space）：腋下必须保持足够空间。肘部紧贴身体=丧失杠杆=无法利用胸大肌=只能靠手腕。肘部空间不足也会导致大脑拒绝执行Out向量。
13. 过度转动（Over-rotation）：躯干转太早或转太多=推压力量泄向侧面而非直达球后。击球时应保持一定侧身，不要过早完全正面朝网。
14. 重心转移时序：Right→Left→Hit。重心必须在挥拍之前就开始从后脚前移。右脚应在地面上拧转（Pivot），不是腾空跳起。击球结束时后脚脚尖点地=正确。
15. 挥拍是圆弧不是直线：球拍轨迹是包围身体的圆弧，Out向量是圆弧的自然切线分量，不需要刻意制造。方向由击球时机决定，不由挥拍方向决定。
16. 握拍旋转轴：职业选手的球拍绕手掌上部"V型区"（食指-中指之间）旋转，而非绕拍底（butt cap）旋转。轴心在上部时球拍自然产生Lag和Pronation；轴心在底部时前臂被锁定，只能用手腕硬甩。食指和中指是"感觉器官"和"旋转支点"，不是发力工具。判断方式：图5-6中球拍翻转是渐进平滑的（轴心正确）还是突然剧烈的（轴心在底部）。

【Drill 知识库（为每个问题推荐对应训练）】
- Unit Turn/重心问题 → ①双手交叉放肩上练整体转（X-Factor训练）②打水漂练习（右-左-投）③静态Pivot练习（右脚踩实→拧转→脚尖点地）④Unit Turn后尝试抬左脚（检验重心）
- Scooping/捞球 → ①向地面砸球（消除怕下网的僵硬）②负重推压（2.5磅重物做正手）③盒子随挥自检（随挥终点是否形成矩形）④二指挥拍法（仅拇指食指握拍打墙）
- 缺Out向量 → ①围网追踪练习（球拍沿围网画弧）②拍头指向右侧网柱停1秒③站远一步击球（被迫伸展）④药球侧向抛掷（体会向外甩出）
- 过度转体 → ①保持侧身击球练习②"接住球拍"（左手高位接拍=停止转动）③击球定格（胸部指向45度保持住）
- 肘部空间不足 → ①外旋沉肩激活胸肌连接练习②站远半步击球③手抛球推掌练习
- 动力链脱节/时序错误 → ①"Hips, Hit!"节奏击球（口喊"髋！打！"强制髋先动）②静止手臂影子挥拍（手臂锁死只靠髋转带出拍子）③McGregor Walk热身（躯干大扭手臂甩面条）④提起左脚击球（前脚离地迫使用髋部完成击球）⑤后脚不动影子挥拍（后脚脚尖不离地感受制动传递）
- 握拍/前臂锁定/拍头速度不足 → ①两指握拍练习（仅食指中指+拇指握拍打球，消除底部干扰，解锁前臂旋转）②两指迷你网球（在真实击球中维持触觉参考点）③全手但维持两指感觉（迁移：加回手指不增加抓握压力）④交替：两指10球→全手10球（对比验证触觉信号是否丢失）

【根因诊断逻辑链——从症状追溯根因】

以下是常见正手问题的因果链。诊断时必须沿链条向上追溯，找到最深层的根因，而不是只描述表面症状。优先报告根因，表面症状作为佐证。

■ 症状群A：小臂代偿（最常见的业余问题）
  表面症状：小臂向上刷球、手腕翻转剧烈、手腕高于肘部、球拍震手
  中层原因：手臂脱离身体旋转系统，变成独立发力单元
  深层根因（三条路径，按可能性排查）：
    路径1：背部连接断裂（背阔肌没有持续"粘住"手臂）
    路径2：握拍旋转轴错误（轴心在拍底而非手掌上部→前臂被锁定→只能用小臂肌肉硬刷）
    路径3：击球太急（没时间让身体先动→手臂抢跑代偿）
  诊断路径：
    看图4-5：肘角是在收缩（<90°变更小）还是展开（向140°+增大）？
    → 收缩 = 小臂在主动刷球（代偿确认）
    → 接着看图2-3：手臂是否先于髋部启动？
      → 是 = 动力链脱节（手臂抢跑）→ 根因：背部连接不够或击球太急没时间准备
      → 否 = 连接在但手部太紧 → 根因：握拍太紧/旋转轴在拍底/手腕不够松
    → 看图5-6：球拍翻转是否剧烈突然？
      → 剧烈翻转 = 旋转轴可能在拍底（正确轴心时翻转是平滑渐进的）
  建议侧重：
    - 路径1：口令`背部肌肉全程托着，手腕是铰链`，drill=外旋沉肩激活前后肌群连接练习
    - 路径2：口令`V型区卡住，食指是门轴`，drill=两指握拍练习（仅拇指食指握拍打球，解锁前臂旋转）
    - 路径3：口令`弹...落...打`三拍节奏，退后半步
    - 路径4（新增·极重要）：**胸肌脱节**——手臂滞后于躯干旋转，然后猛追。口令`胸部press，手臂焊在胸上`。判断：图3-4中手臂和躯干是否同步前移？不同步=胸肌参与不足。drill=外旋+沉肩激活冈下肌/小圆肌和胸大肌的主动肌肉锁定练习
    - 通用drill：钢管推墙（小臂=平行地面钢管，向前推穿不上翻）
    - 检验：打完球手臂累不累？手臂累=代偿；腿和背累=正确

■ 症状群B：V形Scooping（捞球）
  表面症状：黄色轨迹有V形尖角、拍头远低于球、Bottom Edge Leading
  中层原因：拍头过度下坠后被迫急剧上拉
  深层根因（三条路径，需逐一排查）：
    路径1：主动制造lag → 小臂压拍头下去 → 只能用小臂拉上来
      诊断：图2-3中拍头是缓慢下沉（被动）还是急剧下坠（主动）？
    路径2：肩膀下沉 → 手被带到低位 → 物理上只能向上拉
      诊断：图1-2中右肩是否明显低于左肩（Unit Turn阶段不应该有Tilt）？
    路径3：击球空间不足 → 离球太近 → 没有向外的空间只能向上
      诊断：图4中肘部是否紧贴身体？腋下有没有空间？
  建议侧重：
    - 路径1：口令`拍头下落是身体旋转的副产品，不要主动制造`，drill=影子挥拍只蹬转
    - 路径2：口令`Unit Turn扁担水平`，drill=转体时检查两肩高度
    - 路径3：口令`外旋沉肩锁住胸肌连接`，drill=站远半步击球
    - 通用检验：随挥结束时"盒子"是否方正（大臂+小臂+球拍+胸部=矩形）

■ 症状群C：动力链脱节（手臂先于身体或滞后于身体）
  表面症状：手臂和躯干不同步（手臂先启动或猛追）、肩膀抖动不稳、双臂不同步、"胸转了但手臂在后面"
  中层原因：力的传导在某个环节断裂
  动力链完整路径：躯干（肩线）发令→腿部爆发式放大→腹斜肌（核心）传导→胸部→手臂（胸肌锁定）→手→球拍
  断裂可能发生在任何一环，需逐环排查：
    1. 脚→髋 断裂：没蹬地或跳起 → 髋无法启动
      诊断：图1中后脚是否承重？图3-4后脚是拧转还是腾空？
    2. 髋→核心 断裂：髋在转但躯干没跟上 → 力在腰部消失
      诊断：图2-3中髋部和躯干是否同步旋转？躯干明显滞后=核心没传导
    3. 核心→胸 断裂：只关注"胸部press"但腿力传不上来
      诊断：图3-4中胸部旋转幅度是否和髋匹配？胸转幅度小于髋=核心到胸断了
    4. 胸→手臂 断裂（最常见！）：胸转了但手臂脱节（大臂离开胸侧）
      诊断：图3-4中手臂是随躯干同步前移还是滞后猛追？肘角先急剧收缩再急剧展开=追赶代偿
    5. 击球太急 → 没时间完成准备 → 所有连接来不及建立
      诊断：球是在下落阶段被击中还是上升期？
    6. 左右臂不同步 → 左臂先收回+右臂后出发=两步动作不是一个旋转
      诊断：图2-5中非持拍手是和持拍手同步运动还是有明显时间差？
    7. 过度转体（Over-rotation）→ 身体转过头了，手臂被甩到后方
      诊断：图4击球瞬间身体是侧身（正确）还是已正面朝网（过度）？
  建议侧重：
    - 断裂1：口令`脚拧地，不跳`，drill=静态Pivot练习
    - 断裂2：口令`让肚子传力`，检查蹬转时肚脐左方有没有扭转感
    - 断裂3：口令`蹬→肚子→胸→出去`四字链
    - 断裂4（最重要）：口令`外旋沉肩，胸肌锁住手臂`，drill=外旋+沉肩激活冈下肌/小圆肌和胸大肌的主动锁定练习。连接必须在Unit Turn时就建立！
    - 断裂5：口令`弹...落...打`三拍节奏，退后半步
    - 断裂6：口令`左手松的瞬间=右手出发的瞬间`
    - 断裂7：口令`转到侧面就刹车——左手收胸`
    - 整体检验：打完球手臂累=手臂在代偿；腿和核心累=动力链正确

■ 症状群D：击球点错误
  表面症状：身体前扑够球、手臂过度伸展、击球后失去平衡
  子类型1：击球点太远 → 身体前扑
    根因：站位太靠前 + 击球太早（球还没到身体侧前方45°就打了）
    诊断：图4中身体重心是否前倾过度？手臂是否完全伸直？
    建议：口令`球来找我，不是我去找球`，退后半步，等球落下来
  子类型2：击球点太后 → 手臂包裹身体
    根因：准备太慢（Unit Turn不够早）或引拍太大（Hand Slot在身后）
    诊断：图4中击球位置是否在身体后方？图6随挥是否缠绕脖子而非形成盒子？
    建议：口令`Place → Pull Forward`（放好直接往前拉），drill=半挥拍练习（只做从Press Slot往前推）
  子类型3：击球点太高/太低 → 没有调整肘部高度
    根因：没有用elbow extension调整击球高度
    诊断：图4中肘部高度和来球高度是否匹配？
    建议：口令`肘部对齐来球高度`

■ 症状群E：缺乏Out向量（上旋不足、球浅）
  表面症状：随挥向上收而非向外延伸、黄色轨迹没有向外弧度
  中层原因：挥拍路径缺少向外分量
  深层根因：
    1. 肘部空间不足 → 物理上没有空间向外 → 口令`外旋沉肩，打开肘部空间`
    2. 过度转体 → 身体已经正面朝网，Out方向变成了身体后方 → 口令`转到45°就刹车`
    3. 击球意图错误 → 想着"往上打"而不是"往外甩" → 口令`想着平击，上旋自然产生`
  检验：随挥时拍头是否有指向右侧网柱的瞬间？有=Out对了

■ 症状群F：肩膀倾斜（Tilt）问题
  正确的Tilt：蹬地瞬间持拍侧肩膀微沉 → 创造低到高路径 → 上旋（注意：不是Unit Turn时就沉肩，而是蹬地瞬间才沉）
  错误的Tilt：Unit Turn阶段就压低右肩 → 重心倒向左脚 → 无法蹬地 → 一切崩溃
  诊断：Tilt出现在哪个阶段？
    图1-2（准备阶段）就有明显Tilt = 太早，错误
    图3-4（前挥阶段）出现Tilt = 正确时机
  建议：口令`准备时扁担水平，蹬地时右肩才沉`

■ 症状群G：握拍与拍头速度问题
  表面症状：拍头速度不足、球拍翻转剧烈突然（不是渐进平滑）、击球感觉在"推"球而非"甩"球、前臂僵硬锁定
  中层原因：球拍旋转轴在拍底（butt cap）而非手掌上部
  深层根因：握拍方式错误——用手掌底部和小指发力，锁定手腕
  诊断路径：
    看图4-6：球拍翻转方式
    → 突然剧烈翻转 = 轴心在底部，用手腕甩 → 握拍轴心错误
    → 渐进平滑翻转 = 轴心在上部，自然旋转 → 握拍正确
    看图4：击球时手腕是否僵硬？
    → 手腕紧绷 = 小指和无名指抓太紧 → 锁定了旋转轴
  建议侧重：
    - 口令：`食指中指是感觉器官，不是抓取工具` `你是球拍惯性的引导者，不是主人`
    - drill：两指握拍练习（仅食指中指+拇指握拍，消除底部干扰）
    - 感觉提示：正确时拍头有"自发翻转"的魔杖感；错误时感觉在"撬"或"推"球拍
    - 检验：随挥中拍头是自然流畅地翻过来，还是突然"甩"过来？流畅=轴心对了

【诊断优先级规则】
1. 先看动力链时序（C）：如果手臂先于身体，其他问题都是它的下游症状
2. 再看连接（A）：如果手臂脱离身体，scooping和缺Out都是必然结果
3. 然后看击球点（D）：位置错误会导致代偿性的刷球/够球
4. 最后看细节（B/E/F）：在前三个问题解决后再优化
报告中请把最上游的根因放在issues列表第一位，标为severity="高"

【分析维度说明】
每个问题和建议必须从三个层面分析，不能只说动作本身：
1. 动作层（Action）：具体什么动作做错了/做对了（引用图号和轨迹）
2. 身体层（Body）：哪些肌肉/关节/铰链没有正确工作？重心在哪？力从哪传到哪断了？
3. 感觉层（Feel）：正确做到时大脑应该有什么感觉？什么口令能触发正确动作？

例如：
- 动作层：图2中手臂先于身体启动
- 身体层：右脚没有承重（重心在左脚），导致无法通过地面反作用力启动髋旋转，手臂被迫独立发力
- 感觉层：正确感觉应该是"屁股坐在高脚凳上，右脚拧瓶盖，手被甩出去"

【输出格式】
严格JSON，不含其他文字：
{
  "issues": [
    {
      "name": "问题简称",
      "severity": "高/中/低",
      "frame": "图X",
      "action": "动作层：在图X中观察到什么动作问题（引用轨迹颜色和标注）",
      "body": "身体层：哪些肌肉/关节/重心出了问题，力的传导在哪里断裂",
      "feel": "感觉层：正确做到时应该有什么感觉，什么心理意象能触发正确动作",
      "ftt_principle": "违反了哪条原则",
      "drill": "针对性训练方法（动作+次数+感觉提示）"
    }
  ],
  "strengths": [
    {
      "description": "在图X中观察到的优点",
      "body_reason": "为什么这个做得好——身体层面的原因"
    }
  ],
  "overall_assessment": "整体评价（引用图号和轨迹颜色，从动作/身体/感觉三层总结）",
  "score": 65,
  "score_reasoning": "评分理由",
  "weight_transfer": "重心转移分析：准备时重心在哪？击球时转移了吗？后脚状态如何？",
  "kinetic_chain": "动力链分析：力从脚→髋→躯干→手臂→球拍的传导是否顺畅？在哪个环节断裂？",
  "drills": [
    {
      "name": "训练名称",
      "method": "具体怎么做（动作描述、次数、组数）",
      "purpose": "解决什么问题（身体层原因）",
      "cue": "练习时的感觉提示/口令（感觉层）"
    }
  ],
  "priority_drill": "最优先的一个训练"
}

评分标准（score，0-100整数）：
- 90-100：接近职业水平，动力链完整，Out向量充分，容错性极高
- 75-89：基本功扎实，有1-2个小问题，整体容错性良好
- 60-74：框架初步成型，存在2-3个明显问题（如scooping、缺Out），容错性一般
- 40-59：多个核心问题，手臂主导明显，容错性差
- 0-39：基本动作框架缺失，需要从零建立
"""

_USER_PROMPT = "请逐帧分析这次正手挥拍。"


# ── Provider backends ───────────────────────────────────────────────

def _encode_image(img: np.ndarray) -> str:
    """BGR image → base64 JPEG string (compressed for fast upload)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _encode_video(video_path: str) -> str:
    """Video file → base64 string for VLM API."""
    from pathlib import Path
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    data = p.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _call_openai_compatible(
    api_key: str, base_url: str, model: str, image_b64: str, user_text: str,
    extra_headers: Optional[Dict] = None,
    system_prompt: Optional[str] = None,
    video_b64: Optional[str] = None,
) -> Optional[str]:
    """Call any OpenAI-compatible vision API (Qwen-VL, GPT-4o, proxies).

    If video_b64 is provided, sends video instead of image.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[VLM] 安装 openai 包: pip install openai")
        return None

    kwargs = {"api_key": api_key, "base_url": base_url or None, "timeout": 180.0}
    if extra_headers:
        kwargs["default_headers"] = extra_headers
    client = OpenAI(**kwargs)

    # Build content: video or image
    if video_b64:
        media_content = {
            "type": "image_url",
            "image_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        }
    else:
        media_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        }

    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt or _FTT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                media_content,
                {"type": "text", "text": user_text},
            ]},
        ],
    )
    return resp.choices[0].message.content


def _call_anthropic(
    api_key: str, base_url: str, model: str, image_b64: str, user_text: str,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """Call Anthropic Claude vision API."""
    try:
        import anthropic
    except ImportError:
        print("[VLM] 安装 anthropic 包: pip install anthropic")
        return None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt or _FTT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": image_b64,
            }},
            {"type": "text", "text": user_text},
        ]}],
    )
    return resp.content[0].text


def _call_gemini(
    api_key: str, model: str, image_b64: str, user_text: str,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """Call Google Gemini vision API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("[VLM] 安装 google-generativeai 包: pip install google-generativeai")
        return None

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)

    import io
    image_bytes = base64.b64decode(image_b64)
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))

    full_prompt = (system_prompt or _FTT_SYSTEM_PROMPT) + "\n\n" + user_text
    resp = gm.generate_content([full_prompt, img])
    return resp.text


# ── Multi-Round Orchestrator ──────────────────────────────────────────


class MultiRoundAnalyzer:
    """Orchestrates multi-round VLM diagnostic loop.

    Uses the existing VLMForehandAnalyzer._call_vlm() for VLM calls
    and VLMPromptCompiler for prompt generation. Tracks hypotheses
    across rounds with confidence scoring, cross-hypothesis causal
    reasoning, and knowledge-driven observation directives.
    """

    # Confidence adjustment constants (HT-02)
    _SUPPORT_DELTA = 0.15
    _CONTRADICT_DELTA = 0.20
    _AUTO_ELIMINATE_THRESHOLD = 0.15
    _AUTO_CONFIRM_THRESHOLD = 0.85

    def __init__(self, analyzer: "VLMForehandAnalyzer", max_rounds: int = 4, graph=None):
        self.analyzer = analyzer
        self.max_rounds = max_rounds
        self.graph = graph  # KnowledgeGraph for cross-hypothesis reasoning (HT-03)
        self.session: "DiagnosticSession | None" = None
        self._status_snapshots: list[set[tuple[str, str]]] = []

    def run(
        self,
        image_b64: str,
        kpi_summary: str = "",
        supplementary_metrics: dict | None = None,
        video_path: str | None = None,
    ) -> "DiagnosticSession":
        """Execute the full multi-round diagnostic loop.

        Round 0: Symptom scan (reuses Pass 1 checklist)
        Round 1-N: Targeted observation rounds (using observation_directive.j2)
        Final: Confirmation round (using confirmation.j2)
        """
        import hashlib
        from datetime import datetime, timezone

        session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        img_hash = hashlib.md5(image_b64[:200].encode()).hexdigest()[:12]

        self.session = DiagnosticSession(
            session_id=session_id,
            video_path=video_path,
            image_b64_hash=img_hash,
            max_rounds=self.max_rounds,
            supplementary_metrics=supplementary_metrics,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._status_snapshots = []

        # Round 0: symptom scan (reuses existing Pass 1)
        detected_chain_ids = self._execute_round_0(image_b64)
        if not detected_chain_ids:
            # No symptoms detected -> return empty session with final_result
            self.session.final_result = {"issues": [], "strengths": [], "overall_assessment": "No issues detected"}
            self.session.completed_at = datetime.now(timezone.utc).isoformat()
            return self.session

        # Create initial hypotheses
        self.session.hypotheses = self._create_initial_hypotheses(detected_chain_ids)
        self.session.active_chain_ids = list(detected_chain_ids)

        # Take initial status snapshot
        self._take_status_snapshot()

        # Diagnostic rounds 1..max_rounds
        for round_num in range(1, self.max_rounds + 1):
            is_last = round_num == self.max_rounds
            round_result = self._execute_diagnostic_round(round_num, image_b64, is_final=is_last)
            self.session.rounds.append(round_result)

            # HM-01: Observation anchoring validation
            self._validate_observation_anchoring(round_result.observations)

            # Apply observation-based confidence scoring (HT-02)
            self._score_observations(round_result.observations)

            # HM-03: Quantitative cross-validation against kinematic data
            self._cross_validate_kinematics(round_result.observations)

            # Apply VLM hypothesis updates
            self._apply_hypothesis_updates(round_result.hypothesis_updates)

            # Collect observations into session
            self.session.observations.extend(round_result.observations)

            # HM-02: Contradiction detection across rounds
            self._detect_and_record_contradictions(round_result.observations)

            # HM-04: Collect re-observation candidates
            self._update_reobserve_candidates(round_result.observations)

            # Update checked_steps from the directives used in this round
            self._update_checked_steps(round_result)

            # Cross-hypothesis causal reasoning (HT-03)
            self._cross_hypothesis_reasoning(round_num)

            # Progressive narrowing check (HT-04)
            self._check_progressive_narrowing(round_num)

            # Take snapshot and check convergence
            self._take_status_snapshot()
            if self._check_convergence():
                break

        # Build final result: try confirmation round VLM output first, fallback to hypothesis state
        self.session.convergence_score = self._compute_convergence_score()

        # Check if last round produced a root_cause_tree from VLM confirmation
        vlm_final = self._try_confirmation_vlm(image_b64) if image_b64 else None
        if vlm_final and "root_cause_tree" in vlm_final:
            vlm_final["diagnostic_session"] = self.session.model_dump() if hasattr(self.session, 'model_dump') else {}
            vlm_final["convergence_score"] = self.session.convergence_score
            vlm_final["rounds_completed"] = len(self.session.rounds)
            self.session.final_result = vlm_final
        else:
            self.session.final_result = self._build_final_result()

        self.session.completed_at = datetime.now(timezone.utc).isoformat()
        return self.session

    def _try_confirmation_vlm(self, image_b64: str) -> dict | None:
        """Final VLM call with full system prompt + confirmed hypotheses context.

        This produces the root_cause_tree format that the report generator expects.
        Uses the same system prompt as v1.0 single-pass but with hypothesis evidence injected.
        """
        try:
            compiler = self.analyzer.compiler
            if not compiler:
                return None

            # Build evidence summary from confirmed hypotheses
            confirmed = [h for h in self.session.hypotheses if h.status == HypothesisStatus.CONFIRMED]
            active = [h for h in self.session.hypotheses if h.status == HypothesisStatus.ACTIVE]
            top_hypotheses = confirmed + sorted(active, key=lambda h: -h.confidence)[:3]

            if not top_hypotheses:
                return None

            # Use the full system prompt (root_cause_tree format) with hypothesis context
            system_prompt = compiler.compile_system_prompt()

            # Build user text with evidence from multi-round analysis
            evidence_lines = []
            for h in top_hypotheses:
                status = "确认" if h.status == HypothesisStatus.CONFIRMED else f"高度怀疑(置信度{h.confidence:.0%})"
                evidence_lines.append(f"- {h.name_zh or h.name}: {status}")

            obs_lines = []
            for obs in self.session.observations[-10:]:  # last 10 observations
                obs_lines.append(f"- {obs.frame}: {obs.description[:60]}")

            user_text = f"""以下是多轮诊断的中间结果，请基于这些证据生成最终的根因树分析。

【已确认/高度怀疑的假设】
{chr(10).join(evidence_lines)}

【关键观察记录】
{chr(10).join(obs_lines)}

请输出严格JSON（root_cause_tree格式）。注意：上面列出的多个假设可能是同一个根因的不同表现，请追溯到最深层的共同根因。"""

            raw = self.analyzer._call_vlm(image_b64, user_text, system_prompt=system_prompt)
            if raw:
                return _parse_json_response(raw)
        except Exception as exc:
            print(f"[VLM] Confirmation round failed: {exc}")
        return None

    def _execute_round_0(self, image_b64: str) -> list[str]:
        """Round 0: symptom scan. Returns detected chain IDs.
        Reuses compile_pass1_prompt() and _parse_symptom_response() exactly."""
        pass1_prompt = self.analyzer.compiler.compile_pass1_prompt()
        pass1_user = "Review the 6-frame grid and identify symptom categories."
        raw = self.analyzer._call_vlm(image_b64, pass1_user, system_prompt=pass1_prompt)

        round_result = RoundResult(
            round_number=0,
            prompt_sent=pass1_prompt,
            raw_response=raw or "",
        )
        self.session.rounds.append(round_result)

        if raw is None:
            return []

        return self.analyzer._parse_symptom_response(raw)

    def _create_initial_hypotheses(self, detected_chain_ids: list[str]) -> list["Hypothesis"]:
        """Create one Hypothesis per detected chain, linked to chain's primary root cause."""
        hypotheses = []
        for cid in detected_chain_ids:
            chain = self.analyzer.compiler.chain_map.get(cid)
            if not chain:
                continue
            root_cause = chain.root_causes[0] if chain.root_causes else cid
            hyp = Hypothesis(
                id=f"hyp_{cid}",
                chain_id=cid,
                root_cause_concept_id=root_cause,
                name=chain.symptom,
                name_zh=chain.symptom_zh,
                status=HypothesisStatus.ACTIVE,
                confidence=0.5,
                round_introduced=0,
            )
            hypotheses.append(hyp)
        return hypotheses

    def _execute_diagnostic_round(self, round_num: int, image_b64: str, is_final: bool = False) -> "RoundResult":
        """Execute one diagnostic round using knowledge-driven directives.

        Uses compile_observation_directive() for intermediate rounds
        and compile_confirmation_prompt() for the final round.
        Falls back to compile_pass2_prompt() if directive methods unavailable.
        """
        compiler = self.analyzer.compiler

        # Determine which prompt to use
        use_confirmation = is_final or self._check_convergence()

        if use_confirmation and hasattr(compiler, 'compile_confirmation_prompt'):
            prompt = compiler.compile_confirmation_prompt(self.session)
            user_text = f"Round {round_num} (final): Confirm root cause and generate diagnosis."
        elif hasattr(compiler, 'compile_observation_directive'):
            prompt = compiler.compile_observation_directive(self.session, round_num)
            user_text = f"Round {round_num}: Observe specific details for active hypotheses."
        else:
            # Fallback: use pass2 prompt (Phase 8 behavior)
            active_ids = [h.chain_id for h in self.session.hypotheses if h.status == HypothesisStatus.ACTIVE]
            prompt = compiler.compile_pass2_prompt(active_ids)
            user_text = f"Round {round_num}: Evaluate active hypotheses based on visual evidence."

        raw = self.analyzer._call_vlm(image_b64, user_text, system_prompt=prompt)

        observations = []
        hypothesis_updates = []

        if raw:
            try:
                data = json.loads(raw)
                for obs_data in data.get("observations", []):
                    try:
                        obs = Observation(**obs_data)
                        observations.append(obs)
                    except Exception:
                        pass
                for upd_data in data.get("hypothesis_updates", []):
                    try:
                        upd = HypothesisUpdate(**upd_data)
                        hypothesis_updates.append(upd)
                    except Exception:
                        pass
            except (json.JSONDecodeError, TypeError):
                pass  # No-op round if parse fails

        from datetime import datetime, timezone

        return RoundResult(
            round_number=round_num,
            prompt_sent=prompt,
            raw_response=raw or "",
            observations=observations,
            hypothesis_updates=hypothesis_updates,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Hypothesis confidence scoring (HT-02)
    # ------------------------------------------------------------------

    def _score_observations(self, observations: list["Observation"]) -> None:
        """Adjust hypothesis confidence based on observations.

        Each observation's directive_source links it to a hypothesis.
        Supporting (yes) observations increase confidence; contradicting
        (no) observations decrease it. Deltas are scaled by observation
        confidence. Auto-eliminate below threshold, auto-confirm above.
        """
        hyp_map = {h.id: h for h in self.session.hypotheses}
        current_round = len(self.session.rounds)

        for obs in observations:
            # Try to find the hypothesis this observation relates to
            # The directive_source may contain the hypothesis ID or step info
            related_hyp = self._find_hypothesis_for_observation(obs)
            if not related_hyp or related_hyp.status != HypothesisStatus.ACTIVE:
                continue

            if obs.judgment == ObservationJudgment.YES:
                delta = self._SUPPORT_DELTA * obs.confidence
                related_hyp.confidence = min(1.0, related_hyp.confidence + delta)
                related_hyp.supporting_observations.append(obs.id)
            elif obs.judgment == ObservationJudgment.NO:
                delta = self._CONTRADICT_DELTA * obs.confidence
                related_hyp.confidence = max(0.0, related_hyp.confidence - delta)
                related_hyp.contradicting_observations.append(obs.id)
            # UNCLEAR: no change

            # Auto-eliminate / auto-confirm
            if related_hyp.confidence < self._AUTO_ELIMINATE_THRESHOLD:
                related_hyp.status = HypothesisStatus.ELIMINATED
                related_hyp.round_resolved = current_round
            elif related_hyp.confidence >= self._AUTO_CONFIRM_THRESHOLD:
                related_hyp.status = HypothesisStatus.CONFIRMED
                related_hyp.round_resolved = current_round

    def _find_hypothesis_for_observation(self, obs: "Observation") -> "Hypothesis | None":
        """Find the hypothesis related to an observation via directive_source.

        The directive_source field is set by compile_observation_directive to
        contain the hypothesis ID. Falls back to matching by chain_id patterns.
        """
        hyp_map = {h.id: h for h in self.session.hypotheses}

        # Direct match: directive_source contains hypothesis ID
        if obs.directive_source in hyp_map:
            return hyp_map[obs.directive_source]

        # Pattern match: directive_source may be "hyp_dc_xxx" or "check_step_N"
        for hyp_id, hyp in hyp_map.items():
            if hyp_id in obs.directive_source or obs.directive_source in hyp_id:
                return hyp

        # Fallback: first active hypothesis
        active = [h for h in self.session.hypotheses if h.status == HypothesisStatus.ACTIVE]
        return active[0] if active else None

    # ------------------------------------------------------------------
    # Cross-hypothesis causal reasoning (HT-03)
    # ------------------------------------------------------------------

    def _cross_hypothesis_reasoning(self, current_round: int) -> None:
        """If a confirmed hypothesis is upstream of an active one in the
        causal graph, auto-eliminate the downstream hypothesis.

        Uses KnowledgeGraph.get_causal_chain() to find causal paths.
        """
        if self.graph is None:
            return

        confirmed = [h for h in self.session.hypotheses if h.status == HypothesisStatus.CONFIRMED]
        active = [h for h in self.session.hypotheses if h.status == HypothesisStatus.ACTIVE]

        if not confirmed or not active:
            return

        for conf_hyp in confirmed:
            # Get all concept IDs that are downstream of the confirmed root cause
            # get_causal_chain returns paths FROM root causes TO the symptom
            # We need the forward direction: what does this root cause cause?
            downstream_ids = self._get_downstream_concepts(conf_hyp.root_cause_concept_id)

            for act_hyp in active:
                if act_hyp.root_cause_concept_id in downstream_ids:
                    act_hyp.status = HypothesisStatus.ELIMINATED
                    act_hyp.round_resolved = current_round
                    act_hyp.confidence = 0.0
                    # Record as downstream symptom
                    if not any(
                        u.hypothesis_id == act_hyp.id and "downstream" in u.reason
                        for r in self.session.rounds
                        for u in r.hypothesis_updates
                    ):
                        # Add a synthetic update for tracking
                        pass  # Tracked via status change

    def _get_downstream_concepts(self, concept_id: str) -> set[str]:
        """Find all concepts that are caused by the given concept (forward traversal)."""
        if self.graph is None:
            return set()

        downstream: set[str] = set()
        frontier = {concept_id}
        visited: set[str] = set()

        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)

            # Follow outgoing 'causes' edges
            if current in self.graph.graph:
                for _, target, data in self.graph.graph.out_edges(current, data=True):
                    if data.get("relation") == "causes":
                        downstream.add(target)
                        frontier.add(target)

        return downstream

    # ------------------------------------------------------------------
    # Progressive narrowing (HT-04)
    # ------------------------------------------------------------------

    def _check_progressive_narrowing(self, round_num: int) -> None:
        """Log warning if a round made no progress (no hypothesis confirmed/eliminated)."""
        if len(self._status_snapshots) < 1:
            return

        current_snapshot = {(h.id, h.status.value) for h in self.session.hypotheses}
        prev_snapshot = self._status_snapshots[-1]

        if current_snapshot == prev_snapshot and round_num > 1:
            # No progress this round -- logged for diagnostics
            pass  # Tracked via convergence stagnation detection

    # ------------------------------------------------------------------
    # Update state tracking
    # ------------------------------------------------------------------

    def _update_checked_steps(self, round_result: "RoundResult") -> None:
        """Update session.checked_steps based on observations from this round.

        Parses directive_source to identify which check step index was evaluated.
        """
        for obs in round_result.observations:
            # Try to extract chain_id and step_index from directive_source
            # The compile_observation_directive sets directive_source to hypothesis_id
            for hyp in self.session.hypotheses:
                if hyp.id in obs.directive_source or obs.directive_source in hyp.id:
                    chain_id = hyp.chain_id
                    if chain_id not in self.session.checked_steps:
                        self.session.checked_steps[chain_id] = []
                    # Increment: mark next unchecked step as checked
                    checked = self.session.checked_steps[chain_id]
                    next_step = len(checked)
                    chain = self.analyzer.compiler.chain_map.get(chain_id)
                    if chain and next_step < len(chain.check_sequence):
                        checked.append(next_step)
                    break

    def _apply_hypothesis_updates(self, updates: list["HypothesisUpdate"]) -> None:
        """Apply VLM hypothesis updates to session.hypotheses."""
        hyp_map = {h.id: h for h in self.session.hypotheses}
        current_round = len(self.session.rounds) - 1

        for upd in updates:
            hyp = hyp_map.get(upd.hypothesis_id)
            if not hyp:
                continue
            if upd.action == "confirm":
                hyp.status = HypothesisStatus.CONFIRMED
                hyp.confidence = max(hyp.confidence, 0.8)
                hyp.round_resolved = current_round
            elif upd.action == "eliminate":
                hyp.status = HypothesisStatus.ELIMINATED
                hyp.confidence = 0.0
                hyp.round_resolved = current_round
            # "adjust": keep active, no confidence/status change

    def _take_status_snapshot(self) -> None:
        """Record current hypothesis statuses for stagnation detection."""
        snapshot = {(h.id, h.status.value) for h in self.session.hypotheses}
        self._status_snapshots.append(snapshot)

    # ------------------------------------------------------------------
    # Hallucination Mitigation (HM-01 through HM-04)
    # ------------------------------------------------------------------

    def _validate_observation_anchoring(self, observations: list["Observation"]) -> None:
        """HM-01: Validate that observations are anchored to frames."""
        try:
            from evaluation.hallucination_mitigation import validate_anchoring
            validate_anchoring(observations)
        except ImportError:
            pass

    def _cross_validate_kinematics(self, observations: list["Observation"]) -> None:
        """HM-03: Cross-validate VLM observations against YOLO kinematic data."""
        try:
            from evaluation.hallucination_mitigation import cross_validate_with_kinematics
            cross_validate_with_kinematics(observations, self.session.supplementary_metrics)
        except ImportError:
            pass

    def _detect_and_record_contradictions(self, new_observations: list["Observation"]) -> None:
        """HM-02: Detect cross-round contradictions."""
        try:
            from evaluation.hallucination_mitigation import detect_contradictions
            contradictions = detect_contradictions(self.session, new_observations)
            self.session.contradictions.extend(contradictions)
        except ImportError:
            pass

    def _update_reobserve_candidates(self, new_observations: list["Observation"]) -> None:
        """HM-04: Collect low-confidence / unanchored / contradicted observations for re-observation."""
        try:
            from evaluation.hallucination_mitigation import collect_reobserve_candidates
            candidates = collect_reobserve_candidates(self.session, new_observations)
            # Deduplicate with existing
            existing = set(self.session.reobserve_candidates)
            for c in candidates:
                if c not in existing:
                    self.session.reobserve_candidates.append(c)
        except ImportError:
            pass

    def _check_convergence(self) -> bool:
        """Check stopping criteria:
        1. Top hypothesis confidence >= 0.8
        2. Only 1 active hypothesis left
        3. No status change for 2 consecutive rounds
        """
        active = [h for h in self.session.hypotheses if h.status == HypothesisStatus.ACTIVE]

        # Criterion 1: top confidence >= 0.8
        all_confs = [h.confidence for h in self.session.hypotheses if h.status != HypothesisStatus.ELIMINATED]
        if all_confs and max(all_confs) >= 0.8:
            self.session.convergence_score = self._compute_convergence_score()
            return True

        # Criterion 2: only 1 active hypothesis
        if len(active) <= 1:
            self.session.convergence_score = self._compute_convergence_score()
            return True

        # Criterion 3: no change for 2 consecutive rounds
        if len(self._status_snapshots) >= 3:
            if self._status_snapshots[-1] == self._status_snapshots[-2] == self._status_snapshots[-3]:
                self.session.convergence_score = self._compute_convergence_score()
                return True

        return False

    def _compute_convergence_score(self) -> float:
        """Per V2_RESEARCH.md:
        If any confirmed: max(h.confidence for confirmed hypotheses)
        Else: 1.0 - (active_count / initial_count)
        """
        confirmed = [h for h in self.session.hypotheses if h.status == HypothesisStatus.CONFIRMED]
        if confirmed:
            return max(h.confidence for h in confirmed)

        total = len(self.session.hypotheses)
        if total == 0:
            return 0.0
        active_count = len([h for h in self.session.hypotheses if h.status == HypothesisStatus.ACTIVE])
        return 1.0 - (active_count / total)

    def _build_final_result(self) -> dict:
        """Build v1.0-compatible result dict from session state.

        Includes diagnostic_session data for the report generator's
        diagnostic journey section (RG-01).
        """
        issues = []
        for h in self.session.hypotheses:
            if h.status in (HypothesisStatus.CONFIRMED, HypothesisStatus.ACTIVE):
                issues.append({
                    "name": h.name_zh or h.name,
                    "severity": "高" if h.status == HypothesisStatus.CONFIRMED else "中",
                    "description": f"Hypothesis {h.id}: {h.name}",
                    "confidence": h.confidence,
                    "chain_id": h.chain_id,
                })

        result = {
            "issues": issues,
            "strengths": [],
            "overall_assessment": f"Multi-round analysis ({len(self.session.rounds)} rounds)",
            "convergence_score": self.session.convergence_score,
            "rounds_completed": len(self.session.rounds),
        }

        # Include serialized session for diagnostic journey report (RG-01)
        try:
            result["diagnostic_session"] = self.session.model_dump()
        except Exception:
            pass

        return result


# ── Main Analyzer ───────────────────────────────────────────────────

class VLMForehandAnalyzer:
    """Send keyframe grid to a VLM for FTT-based analysis.

    Configuration is read from config/vlm_config.json.
    Supports: openai_compatible (Qwen-VL, GPT-4o, proxies), anthropic, gemini.
    """

    # Default models per provider
    _DEFAULT_MODELS = {
        "openai_compatible": "qwen-vl-max",
        "anthropic": "claude-sonnet-4-20250514",
        "gemini": "gemini-2.0-flash",
    }

    def __init__(self, config: Optional[Dict] = None, graph=None, chains=None, user_profile_path=None):
        cfg = config or load_vlm_config()
        self.provider = cfg.get("provider", "openai_compatible")
        self.api_key = cfg.get("api_key", "")
        self.base_url = cfg.get("base_url", "")
        self.model = cfg.get("model", "") or self._DEFAULT_MODELS.get(self.provider, "")
        self.extra_headers = cfg.get("extra_headers", {})
        self.two_pass_enabled = cfg.get("two_pass_enabled", True)

        # Load user profile if available
        user_profile = self._try_load_user_profile(user_profile_path)

        # Initialize VLMPromptCompiler for two-pass analysis
        self.compiler = None
        if _HAS_KNOWLEDGE:
            if graph is not None and chains is not None:
                self.compiler = VLMPromptCompiler(graph, chains, user_profile=user_profile)
            else:
                self.compiler = self._try_auto_load_compiler(user_profile=user_profile)

        # Build number->chain_id mapping for pass1 response parsing
        self._chain_id_by_number: Dict[int, str] = {}
        if self.compiler is not None:
            sorted_chains = sorted(self.compiler.chains, key=lambda c: c.priority)
            for i, chain in enumerate(sorted_chains, 1):
                self._chain_id_by_number[i] = chain.id

    @staticmethod
    def _try_load_user_profile(profile_path=None):
        """Try to load UserProfile from given path or default location.

        Returns None gracefully if file doesn't exist or loading fails.
        """
        if not _HAS_KNOWLEDGE:
            return None
        try:
            from knowledge.user_profile import UserProfile
        except ImportError:
            return None

        if profile_path is not None:
            p = Path(profile_path)
        else:
            p = Path(__file__).resolve().parent.parent / "knowledge" / "extracted" / "user_journey" / "user_profile.json"

        if not p.exists():
            return None
        try:
            return UserProfile.from_json(p)
        except Exception as exc:
            print(f"[VLM] Failed to load user profile: {exc}")
            return None

    @staticmethod
    def _try_auto_load_compiler(user_profile=None):
        """Try to auto-load graph and chains from default extracted paths."""
        try:
            graph_path = Path(__file__).resolve().parent.parent / "knowledge" / "extracted" / "_graph_snapshot.json"
            chains_path = Path(__file__).resolve().parent.parent / "knowledge" / "extracted" / "ftt_video_diagnostic_chains.json"
            if not graph_path.exists() or not chains_path.exists():
                return None
            graph = KnowledgeGraph.from_json(graph_path)
            import json as _json
            chains_data = _json.loads(chains_path.read_text())
            chains = [DiagnosticChain(**c) for c in chains_data["chains"]]
            return VLMPromptCompiler(graph, chains, user_profile=user_profile)
        except Exception as exc:
            print(f"[VLM] Auto-load knowledge graph failed: {exc}")
            return None

    def analyze_swing(
        self,
        keyframe_grid: np.ndarray,
        kpi_summary: str = "",
        video_path: Optional[str] = None,
        supplementary_metrics: Optional[Dict] = None,
        swing_video_path: Optional[str] = None,
    ) -> Optional[Dict]:
        if not self.api_key:
            print(f"[VLM] 跳过: 未在 config/vlm_config.json 中设置 api_key")
            return None

        image_b64 = _encode_image(keyframe_grid)

        # 视频模式：如果有 swing 视频片段，用视频代替关键帧
        video_b64 = None
        if swing_video_path:
            try:
                video_b64 = _encode_video(swing_video_path)
                print(f"[VLM] 使用视频模式分析")
            except Exception as exc:
                print(f"[VLM] 视频编码失败，回退到关键帧模式: {exc}")

        if self.compiler and self.two_pass_enabled:
            return self._analyze_two_pass(image_b64, kpi_summary, supplementary_metrics, video_path, video_b64=video_b64)
        else:
            return self._analyze_single_pass(image_b64, kpi_summary, supplementary_metrics, video_path, video_b64=video_b64)

    # ── Two-pass analysis ──────────────────────────────────────────

    def _analyze_two_pass(
        self,
        image_b64: str,
        kpi_summary: str,
        supplementary_metrics: Optional[Dict],
        video_path: Optional[str],
        video_b64: Optional[str] = None,
    ) -> Optional[Dict]:
        """Two-pass VLM analysis: symptom scan then targeted diagnostics."""
        # Pass 1: Quick symptom scan
        pass1_prompt = self.compiler.compile_pass1_prompt()
        pass1_user = "观察这次击球，识别你看到的症状类别。" if video_b64 else "Review the 6-frame grid and identify symptom categories."
        pass1_raw = self._call_vlm(image_b64, pass1_user, system_prompt=pass1_prompt, video_b64=video_b64)
        if pass1_raw is None:
            # Fallback to single-pass on failure
            return self._analyze_single_pass(image_b64, kpi_summary, supplementary_metrics, video_path)

        detected = self._parse_symptom_response(pass1_raw)
        if not detected:
            # No symptoms detected or parse failure -> single-pass fallback
            return self._analyze_single_pass(image_b64, kpi_summary, supplementary_metrics, video_path)

        # Pass 2: Deep analysis with targeted diagnostic context
        pass2_prompt = self.compiler.compile_pass2_prompt(detected)
        user_text = self._build_user_text(kpi_summary, supplementary_metrics)
        pass2_raw = self._call_vlm(image_b64, user_text, system_prompt=pass2_prompt)
        if pass2_raw is None:
            return None

        result = _parse_json_response(pass2_raw)
        if result and video_path and self.provider == "openai_compatible":
            video_insight = self._analyze_video_dynamics(video_path)
            if video_insight:
                result["video_dynamics"] = video_insight
        return result

    # ── Single-pass analysis (fallback / legacy) ───────────────────

    def _analyze_single_pass(
        self,
        image_b64: str,
        kpi_summary: str,
        supplementary_metrics: Optional[Dict],
        video_path: Optional[str],
        video_b64: Optional[str] = None,
    ) -> Optional[Dict]:
        """Single-pass VLM analysis using static prompt (current/legacy behavior)."""
        # Use graph-backed static prompt if compiler available, else hardcoded
        sys_prompt = None
        if self.compiler:
            try:
                sys_prompt = self.compiler.compile_system_prompt()
            except Exception:
                sys_prompt = None  # fall through to _FTT_SYSTEM_PROMPT default

        user_text = self._build_user_text(kpi_summary, supplementary_metrics)
        raw = self._call_vlm(image_b64, user_text, system_prompt=sys_prompt, video_b64=video_b64)
        if raw is None:
            return None

        result = _parse_json_response(raw)
        if result and video_path and self.provider == "openai_compatible":
            video_insight = self._analyze_video_dynamics(video_path)
            if video_insight:
                result["video_dynamics"] = video_insight
        return result

    # ── VLM call dispatcher ────────────────────────────────────────

    def _call_vlm(
        self, image_b64: str, user_text: str, system_prompt: Optional[str] = None,
        video_b64: Optional[str] = None,
    ) -> Optional[str]:
        """Dispatch to the correct provider backend with optional system_prompt.

        If video_b64 is provided, sends video instead of image (preferred mode).
        """
        try:
            if self.provider == "anthropic":
                return _call_anthropic(
                    self.api_key, self.base_url, self.model, image_b64, user_text,
                    system_prompt=system_prompt,
                )
            elif self.provider == "gemini":
                return _call_gemini(
                    self.api_key, self.model, image_b64, user_text,
                    system_prompt=system_prompt,
                )
            else:
                return _call_openai_compatible(
                    self.api_key, self.base_url, self.model, image_b64, user_text,
                    self.extra_headers, system_prompt=system_prompt,
                    video_b64=video_b64,
                )
        except Exception as exc:
            print(f"[VLM] API call failed ({self.provider}/{self.model}): {exc}")
            return None

    # ── Pass 1 response parsing ────────────────────────────────────

    def _parse_symptom_response(self, raw: str) -> List[str]:
        """Parse Pass 1 VLM response into detected chain IDs.

        Extracts numbers from the response and maps them to chain IDs
        via the priority-sorted numbering used in symptom_checklist.j2.
        Returns empty list if no valid numbers found.
        """
        if not raw or not raw.strip():
            return []

        # Extract all integers from the response
        numbers = [int(n) for n in re.findall(r"\d+", raw)]
        if not numbers:
            return []

        # Map to chain IDs (only valid numbers within range)
        detected = []
        for n in numbers:
            cid = self._chain_id_by_number.get(n)
            if cid and cid not in detected:
                detected.append(cid)

        return detected

    # ── User text builder ──────────────────────────────────────────

    @staticmethod
    def _build_user_text(
        kpi_summary: str = "", supplementary_metrics: Optional[Dict] = None,
    ) -> str:
        """Build user text with optional quantitative context."""
        user_text = _USER_PROMPT

        quant_lines: List[str] = []
        if supplementary_metrics:
            sm = supplementary_metrics
            sync = sm.get("arm_torso_synchrony")
            if sync is not None and (sync >= 0.7 or sync <= 0.3):
                quant_lines.append(
                    f"- 手臂-躯干同步性: {sync:.2f} ({sm.get('arm_torso_sync_label', '')})"
                )
            if sm.get("wrist_v_detected"):
                quant_lines.append(
                    f"- 手腕高度模式: 检测到V形scooping（下沉深度 {sm.get('wrist_v_depth', 0):.2f}）"
                )
            if sm.get("swing_shape_label") is not None:
                quant_lines.append(
                    f"- 挥拍轨迹: {sm['swing_shape_label']}（弧度比 {sm.get('swing_arc_ratio', 0):.2f}）"
                )

        if quant_lines:
            user_text += "\n\n以下是关键点算法检测到的量化辅助信息：\n" + "\n".join(quant_lines)
            user_text += (
                "\n\n重要：量化数据仅供参考，可能因相机角度或检测噪声而不准确。"
                "你必须以自己的视觉观察为第一判断依据。只有当量化数据与你的视觉观察一致时才引用；"
                "如果矛盾，以视觉观察为准并说明数据可能不准。"
                "特别注意：scooping 深度 <0.2 可能是正常的被动拍头下落，不一定是问题。"
            )
        elif kpi_summary:
            user_text += f"\n\n以下是骨骼姿态评分摘要供参考：\n{kpi_summary}"

        return user_text

    def _analyze_video_dynamics(self, video_path: str) -> Optional[str]:
        """Send full video for dynamic motion analysis (supplement to keyframe)."""
        try:
            file_size = os.path.getsize(video_path)
            if file_size > 5 * 1024 * 1024:  # Skip videos > 5MB
                return None

            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            from openai import OpenAI
            kwargs = {"api_key": self.api_key, "base_url": self.base_url or None, "timeout": 120.0}
            if self.extra_headers:
                kwargs["default_headers"] = self.extra_headers
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    {"type": "text", "text":
                        "观察这段网球正手挥拍视频，用一句话总结：身体旋转与手臂是否同步？节奏是否流畅？中文回答。"
                    },
                ]}],
            )
            text = (resp.choices[0].message.content or "").strip()
            # Guard against truncated output
            if resp.choices[0].finish_reason == "length" or (text and text[-1] not in "。！？.!?）)"):
                # Truncate to last complete sentence
                for sep in ["。", "！", "？", ".", "!", "?"]:
                    last = text.rfind(sep)
                    if last > 0:
                        text = text[:last + 1]
                        break
            return text if text else None
        except Exception:
            return None

    # ── V2.0 multi-round entry point ──────────────────────────────────

    def analyze_swing_iterative(
        self,
        keyframe_grid: np.ndarray,
        kpi_summary: str = "",
        video_path: Optional[str] = None,
        supplementary_metrics: Optional[Dict] = None,
        max_rounds: int = 4,
        save_session: bool = True,
    ) -> Optional[Dict]:
        """V2.0 entry point: multi-round iterative VLM analysis.

        Falls back to v1.0 analyze_swing() on ANY failure.
        Returns the same dict format as analyze_swing() for backward compat.
        """
        try:
            if not self.compiler or not self.two_pass_enabled:
                return self.analyze_swing(keyframe_grid, kpi_summary, video_path, supplementary_metrics)

            image_b64 = _encode_image(keyframe_grid)
            mra = MultiRoundAnalyzer(self, max_rounds=max_rounds)
            session = mra.run(image_b64, kpi_summary, supplementary_metrics, video_path)

            if save_session:
                self._save_session(session, video_path)

            # Return the final_result dict (same schema as v1.0 output)
            return session.final_result

        except Exception as exc:
            print(f"[VLM] Multi-round failed ({exc}), falling back to v1.0")
            return self.analyze_swing(keyframe_grid, kpi_summary, video_path, supplementary_metrics)

    def _save_session(
        self, session: "DiagnosticSession", video_path: Optional[str] = None
    ) -> Optional[Path]:
        """Save full DiagnosticSession as JSON for debugging/replay.

        Saves to: output/diagnostic_sessions/{session_id}.json
        Creates directory if needed. Returns path or None on failure.
        """
        try:
            out_dir = Path(__file__).resolve().parent.parent / "output" / "diagnostic_sessions"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{session.session_id}.json"
            out_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            print(f"[VLM] Session saved: {out_path}")
            return out_path
        except Exception as exc:
            print(f"[VLM] Failed to save session: {exc}")
            return None


# ── Response parsing ────────────────────────────────────────────────


def _extract_tag(text: str, tag: str) -> str:
    """Extract the value after a TAG: line. Returns empty string if not found."""
    pattern = rf"^{tag}:\s*(.+)$"
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_section_after_tag(text: str, tag: str, next_tags: List[str]) -> str:
    """Extract free-form text between a TAG: line and the next known tag."""
    pattern = rf"^{tag}:\s*(.+?)$"
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    # Find the earliest next tag
    end = len(text)
    for nt in next_tags:
        nt_match = re.search(rf"^{nt}:", text[start:], re.MULTILINE)
        if nt_match:
            end = min(end, start + nt_match.start())
    return text[start:end].strip()


def _parse_observation_response(text: str) -> Optional[Dict]:
    """Parse the observation-only VLM output (Q1-Q35 format or field-based).

    Supports both Q-numbered format (Q1: answer) and field-based format (shoulder: answer).
    """
    text = text.strip()

    # Try Q-numbered format first (Q1: through Q35:)
    q_pattern = re.compile(r"^Q(\d+[a-z]?):\s*(.+?)$", re.MULTILINE)
    q_matches = list(q_pattern.finditer(text))
    if len(q_matches) >= 10:  # At least 10 Q answers to be valid
        answers = {}
        for m in q_matches:
            q_key = m.group(1)
            answers[f"Q{q_key}"] = m.group(2).strip()

        # Map Q numbers to semantic categories (v4 compact format)
        frames = {
            "sync": {
                "arm_body_sync": answers.get("Q1", ""),
                "sync_breakdown_phase": answers.get("Q2", ""),
                "arm_chest_gap_change": answers.get("Q3", ""),
            },
            "shoulder_torso": {
                "shoulder_level": answers.get("Q4", ""),
                "body_rotation_depth": answers.get("Q5", ""),
                "hip_follow_shoulder": answers.get("Q5b", ""),
                "torso_lean": answers.get("Q6", ""),
                "body_facing_at_contact": answers.get("Q7", ""),
            },
            "arm_racket": {
                "hand_drop": answers.get("Q8", ""),
                "trajectory_shape": answers.get("Q9", ""),
                "racket_face": answers.get("Q10", ""),
                "arm_direction_after": answers.get("Q11", ""),
            },
            "left_hand": {
                "left_hand_prep": answers.get("Q12", ""),
                "left_hand_action": answers.get("Q13", ""),
            },
            "lower_body": {
                "knee_bend": answers.get("Q14", ""),
                "weight_transfer": answers.get("Q15", ""),
                "back_foot": answers.get("Q16", ""),
            },
            "dynamics": {
                "first_mover": answers.get("Q17", ""),
                "rhythm": answers.get("Q18", ""),
                "trunk_decel": answers.get("Q19", ""),
                "finish_balance": answers.get("Q20", ""),
            },
            # L4 preparation extensions (Q21-Q26) — new 5-layer prompt
            "preparation": {
                "unit_turn_start_vs_ball": answers.get("Q21", ""),
                "unit_turn_finish_vs_ball": answers.get("Q22", ""),
                "left_shoulder_leads": answers.get("Q23", ""),
                "racket_hold_up": answers.get("Q24", ""),
                "scapular_glide_jersey": answers.get("Q25", ""),
                "place_then_pull": answers.get("Q26", ""),
            },
            # L5 footwork (Q27-Q32) — new 5-layer prompt
            "footwork": {
                "split_step_visible": answers.get("Q27", ""),
                "split_landing_vs_opp_contact": answers.get("Q28", ""),
                "first_foot_to_move": answers.get("Q29", ""),
                "right_foot_pivot": answers.get("Q30", ""),
                "stance_at_contact": answers.get("Q31", ""),
                "steps_and_recovery": answers.get("Q32", ""),
            },
            # Extra observations (Q33-Q38) — wave-2 video-watching insights
            "extra_observations": {
                "spine_side_bend_at_contact": answers.get("Q33", ""),
                "active_feet_during_wait": answers.get("Q34", ""),
                "split_airborne_orientation": answers.get("Q35", ""),
                "grip_pivot_point": answers.get("Q36", ""),
                "intercept_vs_chase": answers.get("Q37", ""),
                "takeback_speed_match": answers.get("Q38", ""),
            },
        }
        overall = {
            "arm_body_sync": answers.get("Q1", ""),
            "trajectory_shape": answers.get("Q9", ""),
            "movement_sequence": answers.get("Q17", ""),
            "weight_transfer": answers.get("Q15", ""),
            "rhythm": answers.get("Q18", ""),
        }

        result = {
            "format": "observation_v2",
            "frames": frames,
            "overall": overall,
            "raw_answers": answers,
            # Backward compat
            "issues": [],
            "strengths": [],
            "score": None,
        }
        return result

    # Fallback: try old field-based format (FRAME_1: shoulder: / torso: etc.)
    if not re.search(r"^FRAME_\d+:", text, re.MULTILINE):
        return None
    if not re.search(r"^OVERALL:", text, re.MULTILINE):
        return None

    FRAME_FIELDS = [
        "shoulder", "torso", "hitting_arm", "non_hitting_arm",
        "racket", "lower_body", "timing",
    ]
    OVERALL_FIELDS = ["movement_sequence", "arm_body_sync", "trajectory_shape"]

    frames: Dict[str, Dict[str, str]] = {}
    overall: Dict[str, str] = {}

    section_pattern = re.compile(r"^(FRAME_(\d+)|OVERALL):", re.MULTILINE)
    matches = list(section_pattern.finditer(text))

    for i, m in enumerate(matches):
        section_start = m.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[section_start:section_end].strip()
        section_name = m.group(1)

        if section_name.startswith("FRAME_"):
            frame_num = m.group(2)
            frame_data: Dict[str, str] = {}
            for field in FRAME_FIELDS:
                field_match = re.search(
                    rf"^{field}:\s*(.+?)$", section_text, re.MULTILINE
                )
                if field_match:
                    frame_data[field] = field_match.group(1).strip()
                else:
                    frame_data[field] = ""
            frames[frame_num] = frame_data

        elif section_name == "OVERALL":
            for field in OVERALL_FIELDS:
                field_match = re.search(
                    rf"^{field}:\s*(.+?)$", section_text, re.MULTILINE
                )
                if field_match:
                    overall[field] = field_match.group(1).strip()
                else:
                    overall[field] = ""

    if not frames:
        return None

    # Build the result dict
    data: Dict = {
        "format": "observation_v1",
        "frames": frames,
        "overall": overall,
        "raw_text": text,
        # Backward-compat fields so downstream code doesn't break
        "score": None,
        "core_diagnosis": "",
        "root_cause_tree": None,
        "secondary_root_cause": None,
        "overall_narrative": "",
        "kinetic_chain_narrative": "",
        "strengths": [],
        "priority_drill": "",
        "issues": [],
    }
    return data


def _parse_semi_structured_response(text: str) -> Optional[Dict]:
    """Parse the semi-structured VLM output format (v4/v5).

    Expected format has labeled lines like SCORE:, ROOT_CAUSE:, etc.
    with free-form paragraphs between them.  Kept as fallback for legacy
    coaching-mode prompts.
    """
    text = text.strip()

    # Must have at least SCORE and ROOT_CAUSE to be valid semi-structured
    if not re.search(r"^SCORE:", text, re.MULTILINE):
        return None
    if not re.search(r"^ROOT_CAUSE:", text, re.MULTILINE):
        return None

    ALL_TAGS = [
        "SCORE", "ROOT_CAUSE", "EVIDENCE",
        "SECONDARY", "FIX", "METHOD", "CUE", "CHECK",
        # Legacy tags — still parsed for backward compat but no longer prompted
        "BODY", "FEEL", "STRENGTHS", "KINETIC_CHAIN",
    ]

    # Extract score
    score_str = _extract_tag(text, "SCORE")
    try:
        score = int(re.search(r"\d+", score_str).group())
    except (AttributeError, ValueError):
        score = 50

    # Extract core diagnosis: text between SCORE line and ROOT_CAUSE line
    score_match = re.search(r"^SCORE:.+$", text, re.MULTILINE)
    rc_match = re.search(r"^ROOT_CAUSE:", text, re.MULTILINE)
    core_diagnosis = ""
    if score_match and rc_match:
        core_diagnosis = text[score_match.end():rc_match.start()].strip()

    root_cause = _extract_tag(text, "ROOT_CAUSE")
    evidence = _extract_tag(text, "EVIDENCE")

    # Causal explanation: free text between EVIDENCE line and SECONDARY/FIX
    # (In the new format, the causal paragraph follows EVIDENCE directly)
    causal_explanation = _extract_section_after_tag(
        text, "EVIDENCE", ["SECONDARY", "FIX"]
    )

    # Legacy fallback: if BODY/FEEL tags exist, try extracting causal from FEEL
    if not causal_explanation:
        body = _extract_tag(text, "BODY")
        feel = _extract_tag(text, "FEEL")
        causal_explanation = _extract_section_after_tag(
            text, "FEEL", ["SECONDARY", "FIX"]
        )
        # Incorporate body/feel into causal explanation if present
        if body and not causal_explanation:
            causal_explanation = body
        if feel and causal_explanation:
            causal_explanation = causal_explanation + " " + feel

    secondary = _extract_tag(text, "SECONDARY")
    fix_name = _extract_tag(text, "FIX")
    method = _extract_tag(text, "METHOD")
    cue = _extract_tag(text, "CUE")
    check = _extract_tag(text, "CHECK")

    # Legacy tags — parse if present but don't require
    strengths_raw = _extract_tag(text, "STRENGTHS")
    kinetic_chain = _extract_tag(text, "KINETIC_CHAIN")
    if kinetic_chain:
        kc_section = _extract_section_after_tag(text, "KINETIC_CHAIN", [])
        if kc_section:
            kinetic_chain = kinetic_chain + " " + kc_section

    # Build the unified data dict (compatible with report generator)
    data = {
        "format": "semi_structured_v5",
        "score": score,
        "core_diagnosis": core_diagnosis,
        "root_cause_tree": {
            "root_cause": root_cause,
            "root_cause_evidence": evidence,
            "causal_explanation": causal_explanation,
            "downstream_symptoms": [],
            "fix": {
                "one_drill": fix_name,
                "drill_method": method,
                "drill_feel_cue": cue,
                "check_criteria": check,
            },
        },
        "secondary_root_cause": {"root_cause": secondary} if secondary else None,
        "overall_narrative": core_diagnosis,
        "kinetic_chain_narrative": kinetic_chain,  # kept for backward compat
        "strengths": [s.strip() for s in strengths_raw.split("；") if s.strip()] if strengths_raw else [],
        "priority_drill": fix_name,
        "issues": [],  # backward compat — root cause is the single issue
        "raw_text": text,
    }

    # Build a single compat issue from root cause for annotation layer
    if root_cause:
        data["issues"].append({
            "name": root_cause,
            "severity": "高",
            "frame": "",
            "description": causal_explanation or core_diagnosis,
        })

    return data


def _parse_json_response(text: str) -> Optional[Dict]:
    """Extract structured data from VLM response text.

    Supports four formats (tried in order):
    1. Observation-only v1 (FRAME_1: / OVERALL: tags) — new default
    2. Semi-structured v4/v5 (SCORE: / ROOT_CAUSE: tags) — legacy coaching
    3. JSON with root_cause_tree (v3)
    4. Legacy JSON with issues list
    """
    text = text.strip()

    # 1. Try observation format first (new observation-only default)
    result = _parse_observation_response(text)
    if result:
        return result

    # 2. Try semi-structured format (legacy coaching prompt fallback)
    result = _parse_semi_structured_response(text)
    if result:
        return result

    # 2. Try direct JSON parse
    try:
        return _validate_analysis(json.loads(text))
    except json.JSONDecodeError:
        pass

    # 3. Try markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return _validate_analysis(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            pass

    # 4. Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return _validate_analysis(json.loads(text[start:end + 1]))
        except json.JSONDecodeError:
            pass

    print("[VLM] 警告: 无法解析 VLM 响应")
    print(f"[VLM] 原始响应前500字符: {text[:500]}")
    return None


def _validate_analysis(data: Dict) -> Dict:
    """Validate and normalize JSON-format analysis (v3 and legacy)."""
    # Support both new root_cause_tree format and legacy issues format
    if "root_cause_tree" in data:
        # v3 format — validate root cause tree
        tree = data["root_cause_tree"]
        tree.setdefault("root_cause", "")
        tree.setdefault("root_cause_evidence", "")
        tree.setdefault("root_cause_body", "")
        tree.setdefault("root_cause_feel", "")
        tree.setdefault("causal_explanation", "")
        tree.setdefault("downstream_symptoms", [])
        tree.setdefault("fix", {})
        data.setdefault("secondary_root_cause", None)
        data.setdefault("overall_narrative", "")
        data.setdefault("kinetic_chain_narrative", "")
        # Build issues list from tree for backward compatibility (annotation etc.)
        compat_issues = []
        for s in tree.get("downstream_symptoms", []):
            compat_issues.append({
                "name": s.get("symptom", ""),
                "severity": "高",
                "frame": s.get("frame", ""),
                "description": s.get("how_root_cause_creates_it", ""),
            })
        data["issues"] = compat_issues
    else:
        # Legacy format
        data.setdefault("issues", [])
        valid_issues = []
        for issue in data.get("issues", []):
            if isinstance(issue, dict) and "name" in issue:
                issue.setdefault("severity", "中")
                issue.setdefault("description", "")
                issue.setdefault("ftt_principle", "")
                issue.setdefault("correction", "")
                valid_issues.append(issue)
        data["issues"] = valid_issues
        data.setdefault("overall_assessment", "")

    data.setdefault("strengths", [])
    data.setdefault("priority_drill", "")

    if not isinstance(data.get("strengths"), list):
        data["strengths"] = []

    return data

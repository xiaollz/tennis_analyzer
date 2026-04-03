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

        # Pre-compute joint trajectories for drawing on keyframes
        wrist_trail: List[Optional[Tuple[int, int]]] = []
        elbow_trail: List[Optional[Tuple[int, int]]] = []
        if keypoints_series is not None and confidence_series is not None:
            from config.keypoints import KEYPOINT_NAMES
            side = "right" if is_right_handed else "left"
            wrist_idx = KEYPOINT_NAMES[f"{side}_wrist"]
            elbow_idx = KEYPOINT_NAMES[f"{side}_elbow"]
            for i in range(n):
                if float(confidence_series[i][wrist_idx]) >= 0.3:
                    wrist_trail.append((int(keypoints_series[i][wrist_idx][0]), int(keypoints_series[i][wrist_idx][1])))
                else:
                    wrist_trail.append(None)
                if float(confidence_series[i][elbow_idx]) >= 0.3:
                    elbow_trail.append((int(keypoints_series[i][elbow_idx][0]), int(keypoints_series[i][elbow_idx][1])))
                else:
                    elbow_trail.append(None)

        result: List[Tuple[str, np.ndarray]] = []
        for (label, _cn), pos in zip(KEYFRAME_LABELS, positions):
            pos = max(0, min(pos, n - 1))
            frame = frames_raw[pos].copy()

            # Draw joint trajectories up to this frame
            trail_start = max(0, prep_pos)
            trail_end = min(pos + 1, n)

            # Elbow: cyan trail
            if elbow_trail:
                pts_e = [p for p in elbow_trail[trail_start:trail_end] if p is not None]
                if len(pts_e) >= 2:
                    for i in range(1, len(pts_e)):
                        alpha = 0.4 + 0.6 * (i / len(pts_e))
                        cv2.line(frame, pts_e[i-1], pts_e[i], (int(200*alpha), int(200*alpha), 0), 2, cv2.LINE_AA)

            # Wrist: yellow trail (drawn on top)
            if wrist_trail:
                pts_w = [p for p in wrist_trail[trail_start:trail_end] if p is not None]
                if len(pts_w) >= 2:
                    for i in range(1, len(pts_w)):
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
- 是蹬地转髋启动？还是手臂先动了？（关键：看髋部有无旋转迹象）
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
6. 蹬地转髋必须先于手臂运动。
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
- 肘部空间不足 → ①腋下夹网球练习②站远半步击球③手抛球推掌练习
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
    - 路径1：口令`背部肌肉全程托着，手腕是铰链`，drill=毛巾夹腋下打球
    - 路径2：口令`V型区卡住，食指是门轴`，drill=两指握拍练习（仅拇指食指握拍打球，解锁前臂旋转）
    - 路径3：口令`弹...落...打`三拍节奏，退后半步
    - 路径4（新增·极重要）：**胸肌脱节**——手臂滞后于躯干旋转，然后猛追。口令`胸部press，手臂焊在胸上`。判断：图3-4中手臂和躯干是否同步前移？不同步=胸肌参与不足。drill=腋下夹压力感应器练习（保持腋下压力不让手臂飞出去）
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
    - 路径3：口令`腋下夹大西瓜`，drill=站远半步击球
    - 通用检验：随挥结束时"盒子"是否方正（大臂+小臂+球拍+胸部=矩形）

■ 症状群C：动力链脱节（手臂先于身体或滞后于身体）
  表面症状：手臂和躯干不同步（手臂先启动或猛追）、肩膀抖动不稳、双臂不同步、"胸转了但手臂在后面"
  中层原因：力的传导在某个环节断裂
  动力链完整路径：蹬地→转髋→腹斜肌（核心）→胸部→手臂（腋下夹着）→手→球拍
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
    - 断裂4（最重要）：口令`腋下贴住不分开`，drill=腋下夹毛巾打球。连接必须在Unit Turn时就建立！
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
    1. 肘部空间不足 → 物理上没有空间向外 → 口令`腋下夹大西瓜`
    2. 过度转体 → 身体已经正面朝网，Out方向变成了身体后方 → 口令`转到45°就刹车`
    3. 击球意图错误 → 想着"往上打"而不是"往外甩" → 口令`想着平击，上旋自然产生`
  检验：随挥时拍头是否有指向右侧网柱的瞬间？有=Out对了

■ 症状群F：肩膀倾斜（Tilt）问题
  正确的Tilt：蹬地转髋的同时持拍侧肩膀微沉 → 创造低到高路径 → 上旋
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


def _call_openai_compatible(
    api_key: str, base_url: str, model: str, image_b64: str, user_text: str,
    extra_headers: Optional[Dict] = None,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """Call any OpenAI-compatible vision API (Qwen-VL, GPT-4o, proxies)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[VLM] 安装 openai 包: pip install openai")
        return None

    kwargs = {"api_key": api_key, "base_url": base_url or None, "timeout": 120.0}
    if extra_headers:
        kwargs["default_headers"] = extra_headers
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt or _FTT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                }},
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
    ) -> Optional[Dict]:
        if not self.api_key:
            print(f"[VLM] 跳过: 未在 config/vlm_config.json 中设置 api_key")
            return None

        image_b64 = _encode_image(keyframe_grid)

        if self.compiler and self.two_pass_enabled:
            return self._analyze_two_pass(image_b64, kpi_summary, supplementary_metrics, video_path)
        else:
            return self._analyze_single_pass(image_b64, kpi_summary, supplementary_metrics, video_path)

    # ── Two-pass analysis ──────────────────────────────────────────

    def _analyze_two_pass(
        self,
        image_b64: str,
        kpi_summary: str,
        supplementary_metrics: Optional[Dict],
        video_path: Optional[str],
    ) -> Optional[Dict]:
        """Two-pass VLM analysis: symptom scan then targeted diagnostics."""
        # Pass 1: Quick symptom scan
        pass1_prompt = self.compiler.compile_pass1_prompt()
        pass1_user = "Review the 6-frame grid and identify symptom categories."
        pass1_raw = self._call_vlm(image_b64, pass1_user, system_prompt=pass1_prompt)
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
        raw = self._call_vlm(image_b64, user_text, system_prompt=sys_prompt)
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
    ) -> Optional[str]:
        """Dispatch to the correct provider backend with optional system_prompt."""
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


# ── Response parsing ────────────────────────────────────────────────

def _parse_json_response(text: str) -> Optional[Dict]:
    """Extract structured JSON from VLM response text."""
    text = text.strip()

    # Try direct parse
    try:
        return _validate_analysis(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return _validate_analysis(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return _validate_analysis(json.loads(text[start:end + 1]))
        except json.JSONDecodeError:
            pass

    print("[VLM] 警告: 无法解析返回的 JSON")
    return None


def _validate_analysis(data: Dict) -> Dict:
    data.setdefault("issues", [])
    data.setdefault("strengths", [])
    data.setdefault("overall_assessment", "")
    data.setdefault("priority_drill", "")

    valid_issues = []
    for issue in data.get("issues", []):
        if isinstance(issue, dict) and "name" in issue:
            issue.setdefault("severity", "中")
            issue.setdefault("description", "")
            issue.setdefault("ftt_principle", "")
            issue.setdefault("correction", "")
            valid_issues.append(issue)
    data["issues"] = valid_issues

    if not isinstance(data.get("strengths"), list):
        data["strengths"] = []

    return data

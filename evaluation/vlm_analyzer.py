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

        # 1. Elbow angle arc on contact frame
        if phase_label == "contact":
            if (float(conf[shoulder_idx]) >= conf_thr
                    and float(conf[elbow_idx]) >= conf_thr
                    and float(conf[wrist_idx]) >= conf_thr):
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
                    if 120 <= angle <= 160:
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
- 黄色线 = 手腕轨迹（平滑弧线=好，V形=scooping，内缩=缺Out）
- 青色线 = 肘部轨迹（应领先手腕=正确动力链）
- 橙色框 = 算法检测到的问题区域
- 绿色/红色弧线 = 肘角（绿色120-160°=好，红色=异常）
- 肩线/髋线 = 肩部和髋部连线（可见髋肩分离角）

【逐帧分析指南】

图1 准备完成：
- 是否完成整体转身（Unit Turn）？肩膀充分侧转？
- 非持拍肩是否高于持拍肩（肩部倾斜Tilt，10-30°）？这决定了自然的由下至上挥拍轨迹
- 膝关节弯曲蓄力？还是直立？（双弯曲运动员姿态：髋+膝同时弯曲）
- 手部位置在右髋/口袋附近（Hand Slot）？还是拉到身后？
- 引拍简洁？球拍在视线余光范围内？
- 背部是否有肩胛骨后缩的迹象？（拉伸-缩短循环的弹性储备）

图2 前挥启动：
- 是蹬地转髋启动？还是手臂先动了？（关键：看髋部有无旋转迹象）
- 小臂是否有独立下压（pat the dog错误）？看黄色轨迹是否突然下坠
- 手部是否有水平向外位移？（触发球拍绕质心旋转的前提——不是向前，而是向外）
- 手臂整体放松跟随身体？还是在主动做事？

图3 击球前（弹性释放阶段）：
- 拍头自然在手部下方（被动lag）？还是过度下坠？
- 黄色轨迹如果形成V形底部 = 之前主动压了拍头
- 非持拍手开始向胸部收拢（制动准备）？
- 球拍上边缘是否领先下边缘？（上边缘领先=正确，防止球飞出底线）

图4 击球瞬间（40ms释放窗口）：
- 击球点在身体前方？（手腕在前脚延长线前方）
- 离身体有足够空间？（不被挤压，否则大脑拒绝执行Out向量）
- 肘角120-160°？（看绿色/红色弧线标注）
- 头部固定，下巴贴近前肩？
- 躯干是否已停止旋转？（躯干冻结+手臂加速=高效动量交接；躯干仍转=动量泄漏）
- 拍面接近垂直或微闭？

图5 随挥初期：
- 有向前延展（Through）？还是立刻向上收拍？
- 黄色轨迹向外延伸（Out）？还是向内收缩？（Out是上旋和容错的物理基础）
- 非持拍手收拢至胸部（稳定化制动）？还是向后甩开？
- 雨刷器翻转在击球之后开始（正确）？还是击球瞬间就发生？

图6 随挥结束：
- 持拍肩越过非持拍肩？（肩胛骨前伸完成）
- 身体保持平衡？重心前倾（正向平衡）？
- 随挥自然完成（被动延续）？

【核心原则】
1. 正手是旋转驱动的鞭打系统。手臂是传递者，不是发动机。
2. 网球基本定理：拍面-前臂90-135° + 前方击球。
3. Out向量最关键——手向外推触发球拍绕质心旋转，产生上旋。
4. 主动动作仅3个：蹬地转髋、腹部展开、胸肌按压(Press Slot)。其余全部被动。
5. 绝不建议"制造lag"或"主动做pat the dog"——拍头下落和lag都是被动结果。
6. 蹬地转髋必须先于手臂运动。
7. 动量交接：躯干在击球瞬间冻结→全部动能甩入手臂。躯干仍转=动量泄漏。
8. 力量第一因素是甜区接触（偏离损失30-70%），生物力学优化是次要的。
9. 上旋的感觉像"向下击球"——向上刷的意图导致肌肉紧绷，反而降低效率。
10. "放松"的正确含义：手腕和前臂放松（铰链），但肩胛骨和背阔肌必须保持基础张力——它们把手臂"托住"在身体旋转系统上。如果整条手臂全松，球拍重力会把手往下拽，造成V形scooping。正确的放松="末端松、根部稳"。纠正建议时请明确区分这两层。

【输出格式】
严格JSON，不含其他文字：
{
  "issues": [
    {
      "name": "问题简称",
      "severity": "高/中/低",
      "frame": "图X",
      "description": "在图X中观察到...（引用轨迹颜色和标注作为证据）",
      "ftt_principle": "违反了哪条原则",
      "correction": "具体纠正方法和训练建议"
    }
  ],
  "strengths": ["在图X中观察到优点..."],
  "overall_assessment": "整体评价（2-3句话，引用具体图号和轨迹颜色作为证据）",
  "score": 65,
  "score_reasoning": "评分理由（一句话说明为什么给这个分数）",
  "priority_drill": "最推荐的一个训练方法（具体描述动作、次数、感觉提示）"
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
            {"role": "system", "content": _FTT_SYSTEM_PROMPT},
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
        system=_FTT_SYSTEM_PROMPT,
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

    full_prompt = _FTT_SYSTEM_PROMPT + "\n\n" + user_text
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

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or load_vlm_config()
        self.provider = cfg.get("provider", "openai_compatible")
        self.api_key = cfg.get("api_key", "")
        self.base_url = cfg.get("base_url", "")
        self.model = cfg.get("model", "") or self._DEFAULT_MODELS.get(self.provider, "")
        self.extra_headers = cfg.get("extra_headers", {})

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
        user_text = _USER_PROMPT

        # Build quantitative context from supplementary metrics
        quant_lines = []
        if supplementary_metrics:
            sm = supplementary_metrics
            if sm.get("arm_torso_synchrony") is not None:
                quant_lines.append(f"- 手臂-躯干同步性: {sm['arm_torso_synchrony']:.2f} ({sm.get('arm_torso_sync_label', '')}) — 值越高说明手臂越跟随身体旋转，<0.4表示手臂独立运动")
            if sm.get("wrist_v_detected") is not None:
                if sm["wrist_v_detected"]:
                    quant_lines.append(f"- 手腕高度模式: 检测到V形scooping（下沉深度 {sm.get('wrist_v_depth', 0):.2f} 躯干高度）")
                else:
                    quant_lines.append("- 手腕高度模式: 未检测到明显scooping")
            if sm.get("swing_shape_label") is not None:
                quant_lines.append(f"- 挥拍轨迹: {sm['swing_shape_label']}（弧度比 {sm.get('swing_arc_ratio', 0):.2f}，向外距离 {sm.get('swing_out_distance', 0):.2f}）")

        if quant_lines:
            user_text += "\n\n以下是关键点算法检测到的量化辅助信息：\n" + "\n".join(quant_lines)
            user_text += "\n\n请结合以上量化数据和视觉观察进行分析。如果一致，请在分析中引用；如果矛盾，以视觉观察为准。"
        elif kpi_summary:
            user_text += f"\n\n以下是骨骼姿态评分摘要供参考：\n{kpi_summary}"

        # Primary: keyframe grid analysis (more detailed)
        try:
            if self.provider == "anthropic":
                raw = _call_anthropic(self.api_key, self.base_url, self.model, image_b64, user_text)
            elif self.provider == "gemini":
                raw = _call_gemini(self.api_key, self.model, image_b64, user_text)
            else:
                raw = _call_openai_compatible(self.api_key, self.base_url, self.model, image_b64, user_text, self.extra_headers)
        except Exception as exc:
            print(f"[VLM] API 调用失败 ({self.provider}/{self.model}): {exc}")
            return None

        if raw is None:
            return None

        result = _parse_json_response(raw)

        # Optional: supplement with full video analysis for dynamic insights
        if result and video_path and self.provider == "openai_compatible":
            video_insight = self._analyze_video_dynamics(video_path)
            if video_insight:
                result["video_dynamics"] = video_insight

        return result

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

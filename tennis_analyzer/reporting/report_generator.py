"""Offline analysis report generator.

This module turns per-frame monitor outputs into a shareable report similar to
common consumer coaching apps:
- Overall score + radar chart (6 dimensions)
- Highlights + improvements (actionable deltas, not raw measurements)
- Angle curves (sampled) with basic interpretation notes
- Final narrative report (positive -> root cause -> feel cue -> drill)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_cjk_font(size: int) -> ImageFont.ImageFont:
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_text_pil(img_bgr: np.ndarray, text: str, xy: Tuple[int, int], *, font, color_bgr) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _text_size(text: str, font) -> Tuple[int, int]:
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _status_to_score(status: Optional[str]) -> Optional[float]:
    """Map various status conventions to a 0-100-ish score.

    Returns None for unknown/unavailable values so callers can exclude them from
    averages (important for view-gated metrics).
    """
    if status is None:
        return None
    s = str(status).lower()
    if s in ("good", "status.good"):
        return 90.0
    if s in ("ok", "status.ok"):
        return 78.0
    if s in ("bad", "status.bad", "warning"):
        return 62.0
    if s in ("unknown", "status.unknown"):
        return None
    # Fallback for enums: "Status.GOOD"
    if "good" in s:
        return 90.0
    if "ok" in s:
        return 78.0
    if "bad" in s or "warn" in s:
        return 62.0
    if "unknown" in s:
        return None
    return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@dataclass
class SwingSnapshot:
    swing_idx: int
    impact_frame_idx: int
    impact_time_s: float
    # Big3 messages are already "delta-to-goal" style.
    contact_point: Dict[str, Any]
    weight_transfer: Dict[str, Any]
    contact_zone: Dict[str, Any]
    # Secondary monitors (may be empty).
    extension: Dict[str, Any]
    knee_load: Dict[str, Any]
    spacing: Dict[str, Any]
    xfactor: Dict[str, Any]
    wrist: Dict[str, Any]
    balance: Dict[str, Any]


@dataclass
class ReportData:
    input_path: str
    output_video_path: str
    fps: float
    total_frames: int
    camera_view: Optional[str]
    impacts: List[SwingSnapshot]
    angle_series: Dict[str, List[Tuple[float, float]]]  # metric -> list[(t_s, val)]


class ReportCollector:
    """Collect minimal data during a single-pass video run."""

    def __init__(
        self,
        *,
        fps: float,
        total_frames: int,
        sample_fps: float = 4.0,
        camera_view: Optional[str] = None,
    ):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.total_frames = int(total_frames)
        self.sample_fps = float(sample_fps) if sample_fps and sample_fps > 0 else 0.0
        self.camera_view = str(camera_view) if camera_view is not None else None

        self._sample_every = 0
        if self.sample_fps > 0:
            self._sample_every = max(1, int(round(self.fps / self.sample_fps)))

        self.impacts: List[SwingSnapshot] = []
        self._active_swing_idx: Optional[int] = None

        self.angle_series: Dict[str, List[Tuple[float, float]]] = {}

        self._biomech = None

    def set_biomechanics_analyzer(self, biomech) -> None:
        # Avoid import cycles; caller passes BiomechanicsAnalyzer instance.
        self._biomech = biomech

    def _ensure_series(self, name: str) -> List[Tuple[float, float]]:
        if name not in self.angle_series:
            self.angle_series[name] = []
        return self.angle_series[name]

    def maybe_sample_angles(
        self, *, frame_idx: int, keypoints: np.ndarray, confidence: np.ndarray
    ) -> None:
        if self._sample_every <= 0:
            return
        if (int(frame_idx) % self._sample_every) != 0:
            return
        if self._biomech is None:
            return

        t_s = float(frame_idx) / self.fps

        metrics = self._biomech.analyze(keypoints, confidence) or {}
        # Add a simple "body tilt" metric: hip-center -> nose vs vertical.
        tilt = _body_tilt_deg(keypoints, confidence)
        if tilt is not None:
            metrics["Body Tilt"] = tilt

        for k, v in metrics.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            self._ensure_series(str(k)).append((t_s, float(fv)))

    def on_frame(
        self,
        *,
        frame_idx: int,
        chain_results: Dict[str, Any],
    ) -> None:
        """Call after KineticChainManager.update()."""
        if not chain_results:
            return

        big3 = chain_results.get("big3") or {}
        is_impact = bool(big3.get("is_impact", False))

        # If we already have an active swing awaiting contact_zone, fill it once available.
        if self._active_swing_idx is not None:
            idx = self._active_swing_idx
            if 0 <= idx < len(self.impacts):
                cz = big3.get("contact_zone")
                if _big3_is_known(cz):
                    self.impacts[idx].contact_zone = _big3_to_dict(cz)
                    self._active_swing_idx = None

        if not is_impact:
            return

        impact_frame_idx = int(big3.get("impact_frame_idx") or frame_idx)
        impact_time_s = float(impact_frame_idx) / self.fps

        snap = SwingSnapshot(
            swing_idx=len(self.impacts) + 1,
            impact_frame_idx=int(impact_frame_idx),
            impact_time_s=impact_time_s,
            contact_point=_big3_to_dict(big3.get("contact_point")),
            weight_transfer=_big3_to_dict(big3.get("weight_transfer")),
            contact_zone=_big3_to_dict(big3.get("contact_zone")),
            extension=_module_to_dict(chain_results.get("extension")),
            knee_load=_module_to_dict(chain_results.get("knee_load")),
            spacing=_module_to_dict(chain_results.get("spacing")),
            xfactor=_module_to_dict(chain_results.get("xfactor")),
            wrist=_module_to_dict(chain_results.get("wrist")),
            balance=_module_to_dict(chain_results.get("balance")),
        )
        self.impacts.append(snap)

        # Track as active until contact-zone becomes known (usually a few frames later).
        if not _big3_is_known(big3.get("contact_zone")):
            self._active_swing_idx = len(self.impacts) - 1


def extract_impact_thumbnails(
    *,
    video_path: str,
    impacts: List[SwingSnapshot],
    out_dir: Path,
    prefix: str = "swing",
    max_thumbs: int = 6,
) -> List[Path]:
    """Extract impact-frame thumbnails from the rendered output video.

    This is intentionally done *after* video rendering for reliability, and to
    ensure the thumbnails include whatever overlays the output video contains.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    written: List[Path] = []
    for s in impacts[: max(0, int(max_thumbs))]:
        frame_idx = int(s.impact_frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        p = out_dir / f"{prefix}_{int(s.swing_idx):02d}_impact.jpg"
        try:
            cv2.imwrite(str(p), frame)
            written.append(p)
        except Exception:
            continue

    cap.release()
    return written


def _big3_is_known(obj: Any) -> bool:
    if obj is None:
        return False
    st = getattr(obj, "status", None)
    if st is None:
        return False
    s = str(st).lower()
    return "unknown" not in s


def _big3_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {"status": "unknown", "message": ""}
    status = getattr(obj, "status", "unknown")
    msg = getattr(obj, "message", "") or ""
    return {"status": str(status), "message": str(msg)}


def _module_to_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status": "unknown", "message": ""}
    return {
        "status": str(obj.get("status", "unknown")),
        "message": str(obj.get("message", "") or ""),
    }


def _body_tilt_deg(keypoints: np.ndarray, confidence: np.ndarray) -> Optional[float]:
    # COCO indices are defined in config; but keep it simple here by using common ordering:
    # nose(0), left_hip(11), right_hip(12) in YOLO pose output used by this project.
    try:
        nose = np.asarray(keypoints[0], dtype=np.float32)
        l_hip = np.asarray(keypoints[11], dtype=np.float32)
        r_hip = np.asarray(keypoints[12], dtype=np.float32)
        if float(confidence[0]) < 0.5 or float(confidence[11]) < 0.3 or float(confidence[12]) < 0.3:
            return None
        hip = 0.5 * (l_hip + r_hip)
        v = nose - hip
        if float(np.linalg.norm(v)) < 1e-6:
            return None
        # Angle to vertical (0 = perfectly upright).
        # In image coords, +y is down, so vertical axis is (0, 1).
        v_unit = v / (np.linalg.norm(v) + 1e-6)
        cosang = float(np.clip(np.dot(v_unit, np.asarray([0.0, 1.0], dtype=np.float32)), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosang)))
        return ang
    except Exception:
        return None


def _clean_feedback_message(msg: str) -> str:
    """Normalize monitor messages for the report (remove tags/markers).

    Secondary monitors often prefix messages with tags like "[下肢]" or "[手腕]".
    The report should read like a consumer coaching app: short and clean.
    """
    m = str(msg or "").replace("✓", "").strip()
    if m.startswith("[") and "]" in m:
        m = m.split("]", 1)[1].strip()
    return m


# ---------------------------------------------------------------------------
# Chart rendering (OpenCV + PIL text)
# ---------------------------------------------------------------------------


def render_radar_chart(
    *,
    labels: List[str],
    scores: List[float],
    out_path: Path,
    title: str = "综合维度雷达图",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = 720, 520
    img = np.ones((H, W, 3), dtype=np.uint8) * 245
    cx, cy = W // 2, H // 2 + 20
    r = 180

    # Grid rings.
    for ring in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rr = int(r * ring)
        cv2.circle(img, (cx, cy), rr, (210, 210, 210), 1, lineType=cv2.LINE_AA)

    n = max(3, len(labels))
    angles = [(-np.pi / 2) + (2 * np.pi * i / n) for i in range(n)]

    # Axes.
    for a in angles:
        x2 = int(cx + r * np.cos(a))
        y2 = int(cy + r * np.sin(a))
        cv2.line(img, (cx, cy), (x2, y2), (200, 200, 200), 1, lineType=cv2.LINE_AA)

    # Data polygon.
    pts = []
    for a, sc in zip(angles, scores):
        rr = float(np.clip(sc / 100.0, 0.0, 1.0)) * r
        pts.append([int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))])
    pts_np = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_np], isClosed=True, color=(40, 120, 240), thickness=2, lineType=cv2.LINE_AA)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_np], (120, 180, 255))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # Labels + title (PIL for CJK).
    font_title = _load_cjk_font(26)
    font_lbl = _load_cjk_font(18)
    img = _draw_text_pil(img, title, (24, 18), font=font_title, color_bgr=(30, 30, 30))

    for a, lbl in zip(angles, labels):
        tx = int(cx + (r + 22) * np.cos(a))
        ty = int(cy + (r + 22) * np.sin(a))
        tw, th = _text_size(lbl, font_lbl)
        img = _draw_text_pil(img, lbl, (tx - tw // 2, ty - th // 2), font=font_lbl, color_bgr=(60, 60, 60))

    cv2.imwrite(str(out_path), img)
    return out_path


def render_angle_chart(
    *,
    series: Dict[str, List[Tuple[float, float]]],
    impacts: Optional[List[SwingSnapshot]] = None,
    out_path: Path,
    title: str = "角度分析（采样）",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = 1100, 520
    pad_l, pad_r, pad_t, pad_b = 80, 30, 60, 70
    img = np.ones((H, W, 3), dtype=np.uint8) * 250

    # Select a small set of metrics (similar to the reference app).
    prefer = ["L Elbow", "R Elbow", "L Knee", "R Knee", "Body Tilt"]
    keys = [k for k in prefer if k in series]
    # If missing, take first 5 series.
    if not keys:
        keys = list(series.keys())[:5]

    if not keys:
        cv2.imwrite(str(out_path), img)
        return out_path

    # Determine time range and value range.
    t_min, t_max = 0.0, 1.0
    v_min, v_max = 0.0, 180.0
    all_t = []
    all_v = []
    for k in keys:
        pts = series.get(k, [])
        for t, v in pts:
            all_t.append(float(t))
            all_v.append(float(v))
    if all_t:
        t_min = min(all_t)
        t_max = max(all_t)
    if all_v:
        v_min = max(0.0, min(all_v) - 5.0)
        v_max = min(200.0, max(all_v) + 5.0)
        if v_max - v_min < 10:
            v_max = v_min + 10

    def tx(t: float) -> int:
        if t_max <= t_min + 1e-6:
            return pad_l
        return int(pad_l + (float(t) - t_min) / (t_max - t_min) * (W - pad_l - pad_r))

    def ty(v: float) -> int:
        return int(pad_t + (1.0 - (float(v) - v_min) / (v_max - v_min)) * (H - pad_t - pad_b))

    # Axes.
    cv2.line(img, (pad_l, pad_t), (pad_l, H - pad_b), (180, 180, 180), 2)
    cv2.line(img, (pad_l, H - pad_b), (W - pad_r, H - pad_b), (180, 180, 180), 2)

    # Grid + labels.
    font_lbl = _load_cjk_font(16)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = v_min + frac * (v_max - v_min)
        y = ty(v)
        cv2.line(img, (pad_l, y), (W - pad_r, y), (230, 230, 230), 1)
        img = _draw_text_pil(img, f"{v:.0f}", (10, y - 10), font=font_lbl, color_bgr=(120, 120, 120))

    # Draw series.
    colors = [
        (255, 120, 0),    # blue-ish (BGR)
        (0, 140, 255),    # orange
        (0, 200, 120),    # green
        (120, 60, 230),   # purple
        (80, 80, 80),     # gray
    ]
    for i, k in enumerate(keys):
        pts = series.get(k, [])
        if len(pts) < 2:
            continue
        pts_xy = np.asarray([(tx(t), ty(v)) for (t, v) in pts], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_xy], isClosed=False, color=colors[i % len(colors)], thickness=2, lineType=cv2.LINE_AA)

    # Mark impacts as vertical dashed lines (similar to the reference app).
    if impacts:
        for s in impacts:
            t_hit = float(getattr(s, "impact_time_s", 0.0))
            if not np.isfinite(t_hit):
                continue
            if t_hit < (t_min - 1e-3) or t_hit > (t_max + 1e-3):
                continue
            x_hit = tx(t_hit)
            y_top = pad_t
            y_bot = H - pad_b
            # dashed line segments
            seg = 10
            gap = 8
            y = y_top
            while y < y_bot:
                y2 = min(y_bot, y + seg)
                cv2.line(img, (x_hit, y), (x_hit, y2), (180, 180, 180), 1, lineType=cv2.LINE_AA)
                y = y2 + gap

    # Title + legend.
    font_title = _load_cjk_font(22)
    img = _draw_text_pil(img, title, (24, 18), font=font_title, color_bgr=(30, 30, 30))

    x_leg = pad_l
    y_leg = H - pad_b + 18
    for i, k in enumerate(keys):
        c = colors[i % len(colors)]
        cv2.circle(img, (x_leg + 10, y_leg + 8), 6, c, -1, lineType=cv2.LINE_AA)
        img = _draw_text_pil(img, k, (x_leg + 22, y_leg), font=font_lbl, color_bgr=(80, 80, 80))
        x_leg += 160
        if x_leg > W - 180:
            x_leg = pad_l
            y_leg += 22

    cv2.imwrite(str(out_path), img)
    return out_path


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def build_dimension_scores(report: ReportData) -> Dict[str, float]:
    """Compute 6 dimension scores (0-100) from monitor statuses."""
    dims = {
        "准备启动": [],
        "动力链": [],
        "击球时机": [],
        "随挥收拍": [],
        "拍面控制": [],
        "身体稳定": [],
    }

    def add(dim: str, status: Optional[str]) -> None:
        v = _status_to_score(status)
        if v is None:
            return
        dims[dim].append(float(v))

    for s in report.impacts:
        # 准备启动: 转体时机 + 下肢加载（避免临击球才启动）
        add("准备启动", s.xfactor.get("status"))
        add("准备启动", s.knee_load.get("status"))

        # 动力链: 重心释放 + 伸展（Double Bend）+ 空间（背面更可靠）
        add("动力链", s.weight_transfer.get("status"))
        add("动力链", s.extension.get("status"))
        add("动力链", s.spacing.get("status"))

        # 击球时机: 触球点
        add("击球时机", s.contact_point.get("status"))

        # 随挥收拍: 这里用 Big3 contact_zone
        add("随挥收拍", s.contact_zone.get("status"))

        # 拍面控制: 手腕预设（侧面更可靠）
        add("拍面控制", s.wrist.get("status"))

        # 身体稳定: 平衡
        add("身体稳定", s.balance.get("status"))

    out: Dict[str, float] = {}
    for k, vals in dims.items():
        if not vals:
            out[k] = 70.0
        else:
            out[k] = float(np.mean(np.asarray(vals, dtype=np.float32)))
    return out


def build_dimension_details(report: ReportData, dims: Dict[str, float]) -> List[Dict[str, str]]:
    """Build per-dimension narrative text (for the "维度详情" card list)."""
    swings = report.impacts[-3:] if len(report.impacts) > 3 else report.impacts
    view = str(report.camera_view or "").lower()

    def status_kind(status: Optional[str]) -> str:
        if status is None:
            return "unknown"
        s = str(status).lower()
        if "unknown" in s:
            return "unknown"
        if "good" in s:
            return "good"
        if "ok" in s:
            return "ok"
        if "bad" in s or "warn" in s:
            return "bad"
        return "unknown"

    def latest_known(key: str) -> Optional[Dict[str, Any]]:
        for s in reversed(swings):
            obj = getattr(s, key) or {}
            if status_kind(obj.get("status")) == "unknown":
                continue
            return obj
        return None

    def known_count(*keys: str) -> int:
        n = 0
        for s in swings:
            for k in keys:
                obj = getattr(s, k) or {}
                if status_kind(obj.get("status")) != "unknown":
                    n += 1
        return n

    def score_str(name: str, *keys: str) -> str:
        if known_count(*keys) <= 0:
            return "N/A"
        return f"{dims.get(name, 70.0):.0f}"

    details: List[Dict[str, str]] = []

    # 1) 准备启动
    xfac = latest_known("xfactor")
    knee = latest_known("knee_load")
    prep_parts: List[str] = []
    if xfac is None and knee is None:
        prep_parts.append("当前关键点不足，准备启动暂无法稳定评估。")
    else:
        if xfac is not None:
            k = status_kind(xfac.get("status"))
            if k == "good":
                prep_parts.append("转体到位且提前（避免临击球才补动作）")
            else:
                msg = _clean_feedback_message(str(xfac.get("message", "") or ""))
                if msg:
                    prep_parts.append(f"转体：{msg}")
        if knee is not None:
            k = status_kind(knee.get("status"))
            if k == "good":
                prep_parts.append("下肢有加载（更容易把力量从地面传上来）")
            else:
                msg = _clean_feedback_message(str(knee.get("message", "") or ""))
                if msg:
                    prep_parts.append(f"下肢：{msg}")
        if not prep_parts:
            prep_parts.append("准备启动总体可用：重点看转体是否更早、下肢是否先加载。")

    details.append(
        {
            "name": "准备启动",
            "score": score_str("准备启动", "xfactor", "knee_load"),
            "summary": "；".join(prep_parts),
            "tips": "分腿判断后立刻 unit turn；非持拍手推拍喉把身体先“拧紧”；先转体再上步。",
            "drills": "影子挥拍“两段停顿”（转体到位→再挥）3×8；口令练习“转—步—打”3×8。",
        }
    )

    # 2) 动力链
    wt = latest_known("weight_transfer")
    ext = latest_known("extension")
    sp = latest_known("spacing")
    chain_parts: List[str] = []
    if wt is not None:
        k = status_kind(wt.get("status"))
        if k == "good":
            chain_parts.append("重心释放更充分（下肢把力量送上来）")
        else:
            m = _clean_feedback_message(str(wt.get("message", "") or ""))
            if m:
                chain_parts.append(f"重心：{m}")
    if ext is not None:
        k = status_kind(ext.get("status"))
        if k == "good":
            chain_parts.append("伸展更通透（更容易“打穿”而不是缩手）")
        else:
            m = _clean_feedback_message(str(ext.get("message", "") or ""))
            if m:
                chain_parts.append(f"伸展：{m}")
    if sp is not None and view == "back":
        k = status_kind(sp.get("status"))
        if k == "good":
            chain_parts.append("击球空间更合适（不拥挤）")
        else:
            m = _clean_feedback_message(str(sp.get("message", "") or ""))
            if m:
                chain_parts.append(f"空间：{m}")
    if not chain_parts:
        chain_parts.append("动力链总体可用：核心看“下肢释放→髋带身体质量→手臂延伸”。")

    details.append(
        {
            "name": "动力链",
            "score": score_str("动力链", "weight_transfer", "extension", "spacing"),
            "summary": "；".join(chain_parts),
            "tips": "后脚自然释放（拖地/转出去）让髋先走；手臂保持延伸，把力量送进球里。",
            "drills": "拖后脚影子挥拍 3×10；药球侧抛 3×8（髋先走）。",
        }
    )

    # 3) 击球时机
    cp = latest_known("contact_point")
    timing_summary = ""
    if view == "back":
        timing_summary = "背面视角下触球点深度不可观测，该项自动跳过（建议侧面拍摄）。"
    elif cp is None:
        timing_summary = "触球点关键点不足，暂无法稳定评估。"
    else:
        k = status_kind(cp.get("status"))
        if k == "good":
            timing_summary = "触球点进入舒适区（更容易稳定拍面与方向）"
        else:
            timing_summary = _clean_feedback_message(str(cp.get("message", "") or ""))

    details.append(
        {
            "name": "击球时机",
            "score": score_str("击球时机", "contact_point"),
            "summary": timing_summary,
            "tips": "先转体再上步；非持拍手“抓住来球”在身前；别让头去追球。",
            "drills": "Catch the Ball 3×10；慢速喂球只盯“身前触球区”10 分钟。",
        }
    )

    # 4) 随挥收拍
    cz = latest_known("contact_zone")
    ft_summary = ""
    if view == "back":
        ft_summary = "背面视角下穿透/上刷难以稳定评估，该项自动跳过（建议侧面拍摄）。"
    elif cz is None:
        ft_summary = "随挥关键点不足，暂无法稳定评估。"
    else:
        k = status_kind(cz.get("status"))
        if k == "good":
            ft_summary = "随挥顺畅：先穿透再上刷（更稳、更有上旋）"
        else:
            ft_summary = _clean_feedback_message(str(cz.get("message", "") or ""))

    details.append(
        {
            "name": "随挥收拍",
            "score": score_str("随挥收拍", "contact_zone"),
            "summary": ft_summary,
            "tips": "先沿目标线“送出去”（穿透），再自然上刷；避免一上来就陡刷。",
            "drills": "Service box 控制：先平送 4×8 → 再加上刷 4×8；墙练“先送后刷”每组 60 秒。",
        }
    )

    # 5) 拍面控制
    wrist = latest_known("wrist")
    face_summary = ""
    if view == "back":
        face_summary = "背面视角下手腕结构不稳定，该项自动跳过（建议侧面拍摄）。"
    elif wrist is None:
        face_summary = "手腕关键点不足，暂无法稳定评估。"
    else:
        k = status_kind(wrist.get("status"))
        if k == "good":
            face_summary = "拍面更稳定：手腕结构更稳（更容易控方向与高度）"
        else:
            face_summary = _clean_feedback_message(str(wrist.get("message", "") or ""))

    details.append(
        {
            "name": "拍面控制",
            "score": score_str("拍面控制", "wrist"),
            "summary": face_summary,
            "tips": "引拍时就把手腕角度“锁住”（laid-back）；用身体旋转带拍，不靠主动甩腕控方向。",
            "drills": "Press & Roll 5×5；mini tennis（慢速）只求拍面稳定 5 分钟。",
        }
    )

    # 6) 身体稳定
    bal = latest_known("balance")
    stable_summary = ""
    if bal is None:
        stable_summary = "身体稳定关键点不足，暂无法稳定评估。"
    else:
        k = status_kind(bal.get("status"))
        if k == "good":
            stable_summary = "击球后头部/上身更稳（能控住再还原）"
        else:
            stable_summary = _clean_feedback_message(str(bal.get("message", "") or ""))
    details.append(
        {
            "name": "身体稳定",
            "score": score_str("身体稳定", "balance"),
            "summary": stable_summary,
            "tips": "击球后“定住 1 秒”再还原；上身稳定优先于追求更大动作幅度。",
            "drills": "定点喂球：击球后 hold 1 秒 3×8；单脚支撑转髋（慢）每侧 2×8。",
        }
    )

    return details


def classify_player(dims: Dict[str, float], overall: float) -> str:
    stability = dims.get("身体稳定", 70.0)
    timing = dims.get("击球时机", 70.0)
    chain = dims.get("动力链", 70.0)
    control = dims.get("拍面控制", 70.0)

    if overall >= 80 and stability >= 78 and control >= 75:
        return "稳中求进的节奏型选手"
    if chain >= 82 and timing >= 78 and stability < 70:
        return "爆发型进攻选手（需要更稳）"
    if timing < 70 and chain >= 75:
        return "力量够但偏晚（提早准备就起飞）"
    if stability < 68:
        return "进步型选手（先稳再快）"
    return "可塑型选手（打磨关键细节）"


def _pick_highlights_and_issues(report: ReportData) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (highlights, issues) for the "max speed" feedback card.

    issues are dicts with:
      - title
      - why
      - tips
      - ideal
      - drills
    """
    highlights: List[str] = []
    issues: List[Dict[str, str]] = []

    # Aggregate latest swing first (often what user cares about).
    swings = report.impacts[-3:] if len(report.impacts) > 3 else report.impacts

    def status_kind(status: Optional[str]) -> str:
        if status is None:
            return "unknown"
        s = str(status).lower()
        if "unknown" in s:
            return "unknown"
        if "good" in s:
            return "good"
        if "ok" in s:
            return "ok"
        if "bad" in s or "warn" in s:
            return "bad"
        return "unknown"

    def any_good(key: str) -> bool:
        for s in swings:
            st = status_kind(getattr(s, key).get("status"))
            if st == "unknown":
                continue
            if st == "good":
                return True
        return False

    def any_not_good(key: str) -> bool:
        saw_known = False
        for s in swings:
            st = status_kind(getattr(s, key).get("status"))
            if st == "unknown":
                continue
            saw_known = True
            if st != "good":
                return True
        return False if saw_known else False

    def latest_known_message(key: str, *, prefer_non_good: bool = False) -> str:
        # When we're already in a "not good" branch, prefer grabbing the message
        # from the latest non-good swing (avoid titles like "击球点OK").
        if prefer_non_good:
            for s in reversed(swings):
                obj = getattr(s, key) or {}
                st = status_kind(obj.get("status"))
                if st == "unknown" or st == "good":
                    continue
                msg = str(obj.get("message", "") or "").strip()
                if msg:
                    return _clean_feedback_message(msg)

        for s in reversed(swings):
            obj = getattr(s, key) or {}
            if status_kind(obj.get("status")) == "unknown":
                continue
            msg = str(obj.get("message", "") or "").strip()
            if msg:
                return _clean_feedback_message(msg)
        return ""

    # Highlights (keep it short like the reference app).
    if any_good("xfactor") or any_good("knee_load"):
        highlights.append("准备启动到位：转体更早 + 下肢先加载（更容易打到身前）")
    if any_good("weight_transfer") and any_good("extension"):
        highlights.append("动力链更通透：下肢释放 + 手臂延伸能把力量送进球里")
    if any_good("contact_zone"):
        highlights.append("随挥路径更顺：先穿透再上刷（更稳、更有上旋）")
    if any_good("wrist"):
        highlights.append("拍面控制更稳：手腕结构更稳定（更容易控方向）")
    if any_good("balance"):
        highlights.append("身体更稳：击球后头部/上身能控住再还原")
    if any_good("contact_point"):
        highlights.append("触球点能进舒适区：更容易稳定拍面与落点")

    # Issues (Big3 priority, then supporting dimensions).
    if any_not_good("contact_point"):
        msg = latest_known_message("contact_point", prefer_non_good=True)
        issues.append(
            {
                "title": f"触球点问题：{msg}" if msg else "触球点偏晚/偏前",
                "why": "触球点不在舒适区间时，容易“挤住/够着”导致拍面不稳、出球方向飘。",
                "tips": "先转体再上步；非持拍手去“抓住来球”在身前；击球时头部尽量不抬。",
                "ideal": "理想：触球点在身前舒适区（约 20–40cm），而不是贴着身体旁边。",
                "drills": "Catch the Ball（非持拍手抓住来球）3×10；影子挥拍“两段停顿”（转体到位→再挥）3×8。",
            }
        )

    if any_not_good("weight_transfer"):
        msg = latest_known_message("weight_transfer", prefer_non_good=True)
        issues.append(
            {
                "title": f"重心释放不足：{msg}" if msg else "重心释放不足",
                "why": "下肢不送/髋部不跟，会让你只能用手臂加速，稳定性和速度都上不去。",
                "tips": "后脚“拖地/转出去”让髋先走；顺序是：蹬地→转髋→转肩→手臂跟上。",
                "ideal": "理想：击球后后脚能自然释放，身体质量带进球里，而不是上半身硬甩。",
                "drills": "拖后脚影子挥拍 3×10（击球后让后脚自然转出）；药球侧抛 3×8（髋先走）。",
            }
        )

    if any_not_good("contact_zone"):
        msg = latest_known_message("contact_zone", prefer_non_good=True)
        issues.append(
            {
                "title": f"随挥穿透/上刷不足：{msg}" if msg else "随挥穿透/上刷不足",
                "why": "随挥窗口太短会让你过早“刷上去”，球容易短/飘，容错变低。",
                "tips": "先沿目标线把球“送出去”，再自然上刷；避免一上来就陡刷。",
                "ideal": "理想：击球后先有一段“穿透”，随后自然上刷完成随挥。",
                "drills": "Service box 控制练习：先平送（穿透）→再加上刷 4×8；墙练“先送后刷”每组 60 秒。",
            }
        )

    if any_not_good("wrist"):
        msg = latest_known_message("wrist", prefer_non_good=True)
        issues.append(
            {
                "title": f"拍面控制不稳：{msg}" if msg else "拍面控制不稳（手腕结构）",
                "why": "手腕结构不稳时，拍面角度更容易飘，出球方向与高度不稳定。",
                "tips": "引拍时就把手腕角度“锁住”（laid-back），用身体旋转去带拍，不用手腕去“找感觉”。",
                "ideal": "理想：触球前手腕角度稳定，触球后自然释放，不靠主动甩腕控方向。",
                "drills": "Press & Roll（压住球在拍面上滚过）5×5；慢速 mini tennis 只求拍面稳定 5 分钟。",
            }
        )

    if any_not_good("balance"):
        msg = latest_known_message("balance", prefer_non_good=True)
        issues.append(
            {
                "title": f"身体稳定不足：{msg}" if msg else "身体稳定不足",
                "why": "身体晃动会直接带来拍面不稳：出球更飘、容错更低，还会拖慢还原。",
                "tips": "击球后“定住 1 秒”再还原；想象头顶有一条线，别让头去追球。",
                "ideal": "理想：击球后上身稳定，能稳住再回位，而不是边击球边失衡。",
                "drills": "定点喂球：击球后 hold 1 秒 3×8；单脚支撑转髋（慢）每侧 2×8。",
            }
        )

    if any_not_good("xfactor"):
        msg = latest_known_message("xfactor", prefer_non_good=True)
        issues.append(
            {
                "title": f"转体启动/幅度不足：{msg}" if msg else "转体启动/幅度不足",
                "why": "转体不够或太晚，会把压力推给手臂，导致击球点偏晚、稳定性也会受影响。",
                "tips": "分腿判断后立刻 unit turn；非持拍手把拍喉推回去，把身体先“拧紧”。",
                "ideal": "理想：转体最深点出现在击球前，而不是临击球才补转体。",
                "drills": "影子挥拍：每次先停在 unit turn 位置 1 秒再挥 3×8；对墙慢拉，口令“转—步—打”。",
            }
        )

    if any_not_good("knee_load"):
        msg = latest_known_message("knee_load", prefer_non_good=True)
        issues.append(
            {
                "title": f"下肢加载不足：{msg}" if msg else "下肢加载不足",
                "why": "下肢太直会让你很难从地面拿到力量，只能用上半身补偿。",
                "tips": "准备阶段先“坐下”一点点（不是蹲），再转体；击球时把力量从地面送上来。",
                "ideal": "理想：击球前膝角先变小（加载），击球时再释放伸展。",
                "drills": "无球：分腿→坐下→转体→挥拍 3×10；喂球：每球前做 1 次小的加载再击球 3×8。",
            }
        )

    # Keep top 2 issues like the reference app.
    return highlights[:2], issues[:2]


def write_markdown_report(
    *,
    report: ReportData,
    report_dir: Path,
    assets_rel_dir: str = "assets",
) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = report_dir / assets_rel_dir
    assets_dir.mkdir(parents=True, exist_ok=True)

    thumbs = extract_impact_thumbnails(
        video_path=report.output_video_path,
        impacts=report.impacts,
        out_dir=assets_dir,
        max_thumbs=6,
    )

    dims = build_dimension_scores(report)
    labels = list(dims.keys())
    scores = [dims[k] for k in labels]
    overall = float(np.mean(np.asarray(scores, dtype=np.float32))) if scores else 70.0
    archetype = classify_player(dims, overall)
    dim_cards = build_dimension_details(report, dims)

    # Charts.
    radar_path = render_radar_chart(labels=labels, scores=scores, out_path=assets_dir / "radar.png")

    angle_path = render_angle_chart(
        series=report.angle_series, impacts=report.impacts, out_path=assets_dir / "angles.png"
    )

    highlights, issues = _pick_highlights_and_issues(report)

    # Build markdown ordered to mirror the reference app screenshots 1-6:
    # 1) 综合评分与画像（雷达）
    # 2) 维度详情（六张卡片）
    # 3) 亮点/待改进（训练优先级）
    # 4) 动态叠加选项（说明）
    # 5) 角度分析（曲线 + 读图方法/练习）
    # 6) 最终结论
    md: List[str] = []
    md.append("# 分析报告")
    md.append("")
    md.append(f"- 输入视频: `{report.input_path}`")
    md.append(f"- 输出视频: `{report.output_video_path}`")
    md.append(f"- FPS: {report.fps:.2f} | 总帧数: {report.total_frames}")
    if report.camera_view:
        md.append(f"- 视角: {report.camera_view}")
    md.append(f"- 检测到击球次数: {len(report.impacts)}")
    if report.camera_view:
        cv = str(report.camera_view).lower()
        if cv == "side":
            md.append("- 说明：当前为侧面视角，背面更可靠的指标（转体/空间）会自动跳过。")
        elif cv == "back":
            md.append("- 说明：当前为背面视角，侧面才能判断的指标（触球点/随挥/手腕）会自动跳过。")
    md.append("")

    # 1) Score + archetype + radar (图1)
    md.append("## 1) 综合评分与画像（对应图1）")
    md.append(f"- 综合评分（0-100）: **{overall:.0f}**")
    md.append(f"- 选手画像: **{archetype}**")
    if thumbs:
        md.append("")
        md.append("### 击球帧预览")
        # Keep it simple: list thumbnails in order.
        for p in thumbs:
            md.append(f"![{p.name}]({assets_rel_dir}/{p.name})")
    md.append("")
    md.append(f"![radar]({assets_rel_dir}/{radar_path.name})")
    md.append("")

    # 2) Dimension details (图2)
    md.append("## 2) 维度详情（对应图2）")
    for d in dim_cards:
        name = str(d.get("name", "") or "").strip() or "维度"
        sc = str(d.get("score", "") or "").strip() or "N/A"
        summary = str(d.get("summary", "") or "").strip()
        tips = str(d.get("tips", "") or "").strip()
        drills = str(d.get("drills", "") or "").strip()

        md.append(f"### {name}（{sc}）")
        if summary:
            md.append(f"- {summary}")
        if tips:
            md.append(f"- 技术提示：{tips}")
        if drills:
            md.append(f"- 练习建议：{drills}")
        md.append("")
    md.append("")

    # 3) Highlights/issues (图3)
    md.append("## 3) 核心要点与专项提升（对应图3）")
    md.append("- 已取消挥速指标，当前只聚焦动作质量与稳定性。")
    md.append("")
    md.append("### 亮点表现")
    if highlights:
        for h in highlights:
            md.append(f"- {h}")
    else:
        md.append("- （暂无稳定亮点，建议先把击球点与重心做扎实）")
    md.append("")
    md.append("### 待改进（优先级从高到低）")
    if issues:
        for i, it in enumerate(issues, start=1):
            title = str(it.get("title", "") or "").strip() or "待改进"
            why = str(it.get("why", "") or "").strip()
            tips = str(it.get("tips", "") or "").strip()
            ideal = str(it.get("ideal", "") or "").strip()
            drills = str(it.get("drills", "") or "").strip()

            md.append("<details>")
            md.append(f"<summary>{i}. {title}</summary>")
            md.append("")
            if why:
                md.append(f"- 会带来什么问题：{why}")
            if tips:
                md.append(f"- 技术提示：{tips}")
            if ideal:
                md.append(f"- 理想状态：{ideal}")
            if drills:
                md.append(f"- 练习建议：{drills}")
            md.append("</details>")
            md.append("")
    else:
        md.append("- （暂无足够的有效数据生成稳定的“待改进”项：请先确认击球帧缩略图正确、关键点清晰；侧面视角可解锁触球点/随挥/拍面控制。）")
    md.append("")

    md.append("### 专项提升（建议先做 2 周）")
    if issues:
        for i, it in enumerate(issues, start=1):
            title = str(it.get("title", "") or "").strip() or "待改进"
            drills = str(it.get("drills", "") or "").strip()
            tips = str(it.get("tips", "") or "").strip()
            plan = drills if drills else tips
            if plan:
                md.append(f"{i}. **{title}**：{plan}")
            else:
                md.append(f"{i}. **{title}**：先用慢速定点喂球确认动作顺序，再逐步提速。")
    else:
        md.append("1. 击球帧确认：先校正击球点，再做专项训练。")
        md.append("2. 低速稳定：每次练习先用 5 分钟 mini tennis 建立拍面控制。")
    md.append("")

    # 3) Dynamic analysis options (we can't be interactive, so document what to use)
    md.append("## 4) 动态叠加（对应图4）")
    md.append("- 当前输出视频已包含：骨架 + 手腕轨迹 + Big3 面板（如启用 `--big3-ui`）。")
    md.append("- 如需更清爽：可在生成视频时关闭数值面板 `--no-metrics`。")
    md.append("")

    # 4) Angle curves
    md.append("## 5) 角度分析（对应图5）")
    md.append(f"![angles]({assets_rel_dir}/{angle_path.name})")
    md.append("")
    md.append("### 如何阅读这些角度（参考图5）")
    md.append("- 图中灰色虚线表示每次击球时刻（impact）。")
    md.append("1. 肘角：引拍→加速→击球→随挥。过早变直=容易推球；过小=缩手锁死。")
    md.append("2. 膝角：准备阶段先变小=加载；击球时释放伸展=把力量从地面送上来。")
    md.append("3. 身体倾角：越平稳越好；击球附近大幅波动通常意味着失衡/头部跟球。")
    md.append("")
    md.append("### 小练习（配合角度曲线复盘）")
    md.append("- 影子挥拍：每次先停在 unit turn 1 秒，再挥出去（8 次一组×3 组）。")
    md.append("- 定点喂球：击球后 hold 1 秒再还原（8 次一组×3 组）。")
    md.append("")

    # 6) Final narrative
    md.append("## 6) 最终结论（对应图6）")
    md.append("### 肯定")
    if highlights:
        md.append(f"- {highlights[0]}。这说明你的基础方向是对的。")
    else:
        md.append("- 你愿意做动作分析这一步已经很关键了；接下来把“击球点+重心”打磨出来，提升会很快。")
    md.append("")
    md.append("### 建议（只抓 1-2 个根因）")
    if issues:
        for it in issues:
            title = str(it.get("title", "") or "").strip() or "待改进"
            tips = str(it.get("tips", "") or "").strip()
            drills = str(it.get("drills", "") or "").strip()
            if tips and drills:
                md.append(f"- **{title}**：{tips}（练习：{drills}）")
            elif tips:
                md.append(f"- **{title}**：{tips}")
            elif drills:
                md.append(f"- **{title}**：练习：{drills}")
            else:
                md.append(f"- **{title}**")
    else:
        md.append("- Big3 稳定后，下一步追求“更早转体 + 更大转体 + 更长穿透”。")
    md.append("")
    md.append("### 鼓励与计划")
    md.append("- 你现在缺的不是“更多用力”，而是把关键顺序做对：**先转体→再上步→再释放**。")
    md.append("- 建议 2 周做针对性练习（每次 10-15 分钟），只抓一个主问题，效果会非常明显。")
    md.append("")

    # Per-swing summary at the end (useful for debugging)
    md.append("## 附：每次击球的 Big3 快照")
    for s in report.impacts:
        md.append(
            f"- Swing {s.swing_idx} @ frame {s.impact_frame_idx} ({s.impact_time_s:.2f}s): "
            f"触球点={s.contact_point.get('message','') or 'N/A'}；"
            f"重心={s.weight_transfer.get('message','') or 'N/A'}；"
            f"随挥={s.contact_zone.get('message','') or 'N/A'}"
        )

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path

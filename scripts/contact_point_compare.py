#!/usr/bin/env python3
"""击球点对比工具

输入: 1-3 段视频，每段一个 label (real / shadow / coach)
输出:
  • PNG 对比图（3 帧并排 + 标注 + gap 分析）
  • JSON 数据（精确坐标 + 偏差量）

Usage:
    python3 scripts/contact_point_compare.py \\
        --real videos/my_real_hit.mp4 \\
        --shadow videos/my_shadow_swing.mp4 \\
        --coach videos/sinner_reference.mp4 \\
        --out output/contact_compare.png

可选地只传一两个：缺的那个会显示为"待录制"。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VLM_CONFIG = PROJECT_ROOT / "config" / "youtube_api_config.json"

# ── Reference values (FTT-aligned) ──────────────────────────────────

IDEAL_VALUES = {
    "height_label": "腰胸之间",
    "height_cm_above_ground": (90, 115),
    "lateral_cm": (40, 60),
    "depth_cm": (30, 50),       # in front of front hip
    "angle_deg": 45,
}

# Tones from the Baseline design system
COLOR = {
    "bg":     (247, 243, 236, 255),   # paper
    "card":   (253, 251, 246, 255),
    "ink":    (42, 41, 37, 255),
    "ink2":   (90, 84, 74, 255),
    "ink3":   (138, 130, 118, 255),
    "clay":   (200, 85, 61, 255),
    "amber":  (232, 176, 75, 255),
    "court":  (154, 174, 96, 255),
    "line":   (220, 215, 205, 255),
}

LABEL_TONE = {
    "real":   COLOR["clay"],
    "shadow": COLOR["amber"],
    "coach":  COLOR["court"],
}

LABEL_TEXT = {
    "real":   "实际击球",
    "shadow": "空挥",
    "coach":  "教练 / 参考",
}

# ── Gemini analysis ──────────────────────────────────────────────────

GEMINI_PROMPT_FOR_VIDEO = """你是网球生物力学专家。请精确测量这段视频中**击球瞬间**的击球点位置。

输出严格的 JSON 格式（不要用 markdown 包裹），字段如下：

{
  "impact_timestamp_s": <击球瞬间的视频时间秒，浮点数>,
  "height": {
    "cm_above_ground": <估算的离地高度，整数厘米>,
    "category": "knee" | "thigh" | "hip" | "waist" | "chest" | "shoulder",
    "notes": "<一句话视觉描述>"
  },
  "lateral_distance": {
    "cm": <球离持拍侧身体的水平距离，整数厘米>,
    "category": "cramped" | "close" | "ideal" | "far" | "unreachable",
    "notes": "<一句话视觉描述>"
  },
  "depth": {
    "cm_in_front_of_hip": <相对前胯的纵深，正数=前方，负数=后方>,
    "category": "behind" | "beside" | "front_ideal" | "too_far_front",
    "notes": "<一句话视觉描述>"
  },
  "angle_to_body_midline": {
    "degrees": <0-90 之间的实际角度估算>,
    "notes": "<一句话视觉描述>"
  },
  "verdict": {
    "score_0_to_10": <0-10 评分，10 = 完美在攻击区>,
    "summary": "<一句话整体判定>",
    "main_offset": "<最大偏差的维度名>"
  }
}

测量原则：
- 用拍长 70cm、肩宽约 45cm、身高假设 175cm 作为参照
- 击球瞬间 = 拍头与球距离最小的那一帧
- 无法精确量化时给保守估计，但必须给数字
- 不要输出任何 JSON 之外的文字
"""


def call_gemini_on_video(video_path: str) -> Optional[Dict]:
    """调用 Gemini 分析视频，返回 JSON dict 或 None。"""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[!] google-genai not installed. pip install google-genai")
        return None

    if not os.path.exists(video_path):
        print(f"[!] missing video: {video_path}")
        return None

    cfg = json.load(open(VLM_CONFIG))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    client_kwargs = {"api_key": cfg["api_key"]}
    if cfg.get("base_url"):
        client_kwargs["http_options"] = {
            "api_version": "v1beta",
            "base_url": cfg["base_url"],
        }
    client = genai.Client(**client_kwargs)
    model = cfg.get("model", "gemini-3-flash-preview")

    with open(video_path, "rb") as f:
        data = f.read()

    parts = [
        types.Part.from_bytes(data=data, mime_type="video/mp4"),
        types.Part(text=GEMINI_PROMPT_FOR_VIDEO),
    ]
    contents = [types.Content(role="user", parts=parts)]

    print(f"  → Gemini analyzing {os.path.basename(video_path)} ({len(data)} bytes) ...")
    resp = client.models.generate_content(model=model, contents=contents)
    text = resp.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[!] Gemini returned non-JSON: {text[:200]}")
        # Try to extract JSON object via brace matching
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return None


# ── Frame extraction ─────────────────────────────────────────────────

def extract_impact_frame(video_path: str, timestamp_s: float, out_path: str,
                          max_height: int = 720) -> bool:
    """用 ffmpeg 从视频中截取击球瞬间的一帧。"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, timestamp_s):.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale=-2:{max_height}",
        "-q:v", "2",
        "-loglevel", "error",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=20)
        return r.returncode == 0 and os.path.exists(out_path)
    except Exception as e:
        print(f"[!] ffmpeg failed: {e}")
        return False


# ── Image composition ───────────────────────────────────────────────

def font(size: int) -> ImageFont.ImageFont:
    """Load a system font with fallback."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_metric_row(draw: ImageDraw.ImageDraw, x: int, y: int, width: int,
                    label: str, value: str, ideal_str: str,
                    is_ok: bool) -> int:
    """Draw a single 'metric: value (ideal: X)' row. Returns height used."""
    label_font = font(11)
    value_font = font(15)
    ideal_font = font(10)
    line_h = 22

    # Label
    draw.text((x, y), label.upper(), fill=COLOR["ink3"], font=label_font)
    # Value
    val_color = COLOR["court"] if is_ok else COLOR["clay"]
    draw.text((x, y + line_h), value, fill=val_color, font=value_font)
    # Ideal (small, below)
    draw.text((x, y + line_h + 22), f"理想: {ideal_str}",
              fill=COLOR["ink3"], font=ideal_font)

    return line_h + 22 + 18


def is_within(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def assess_metric(measured: Dict, key: str) -> bool:
    """Decide if a measurement falls in the ideal zone."""
    if not measured:
        return False
    if key == "height":
        cm = measured.get("height", {}).get("cm_above_ground")
        if cm is None:
            return False
        lo, hi = IDEAL_VALUES["height_cm_above_ground"]
        return is_within(cm, lo, hi)
    if key == "lateral":
        cm = measured.get("lateral_distance", {}).get("cm")
        if cm is None:
            return False
        lo, hi = IDEAL_VALUES["lateral_cm"]
        return is_within(cm, lo, hi)
    if key == "depth":
        cm = measured.get("depth", {}).get("cm_in_front_of_hip")
        if cm is None:
            return False
        lo, hi = IDEAL_VALUES["depth_cm"]
        return is_within(cm, lo, hi)
    if key == "angle":
        deg = measured.get("angle_to_body_midline", {}).get("degrees")
        if deg is None:
            return False
        return is_within(deg, 35, 55)
    return False


def render_card(label: str, frame_path: Optional[str], measured: Optional[Dict],
                card_w: int, card_h: int) -> Image.Image:
    """Render a single comparison card (frame on top, metrics below)."""
    card = Image.new("RGBA", (card_w, card_h), COLOR["card"])
    draw = ImageDraw.Draw(card)

    # Header band
    band_h = 50
    band_color = LABEL_TONE.get(label, COLOR["ink"])
    draw.rectangle((0, 0, card_w, band_h), fill=band_color)
    draw.text((16, 14), LABEL_TEXT.get(label, label), fill=(255, 255, 255, 255),
              font=font(20))

    # Frame area
    frame_area_y = band_h + 12
    frame_area_h = int(card_h * 0.45)
    frame_w = card_w - 24
    frame_x = 12

    if frame_path and os.path.exists(frame_path):
        try:
            img = Image.open(frame_path).convert("RGB")
            # Fit into frame area (object-contain)
            img.thumbnail((frame_w, frame_area_h), Image.LANCZOS)
            iw, ih = img.size
            ix = frame_x + (frame_w - iw) // 2
            iy = frame_area_y + (frame_area_h - ih) // 2
            card.paste(img, (ix, iy))
            # Mark contact point with a circle if we have angle/distance
            if measured:
                # Place marker near image center as a visual approximation
                # (real positional overlay would require pose keypoints)
                cx, cy = ix + iw // 2, iy + ih // 2
                r = 8
                draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                             outline=COLOR["clay"], width=3)
        except Exception as e:
            draw.text((frame_x + 10, frame_area_y + 10),
                      f"frame load failed: {e}", fill=COLOR["clay"], font=font(11))
    else:
        # Placeholder
        draw.rectangle((frame_x, frame_area_y, frame_x + frame_w,
                        frame_area_y + frame_area_h),
                       outline=COLOR["line"], width=1)
        msg = "待录制" if label != "coach" else "待找参考视频"
        msg_w = draw.textlength(msg, font=font(14))
        draw.text((frame_x + (frame_w - msg_w) // 2,
                   frame_area_y + frame_area_h // 2 - 10),
                  msg, fill=COLOR["ink3"], font=font(14))

    # Metrics area
    metrics_y = frame_area_y + frame_area_h + 16
    metrics_x = 16
    metrics_w = card_w - 32

    if measured:
        h = measured.get("height", {})
        l = measured.get("lateral_distance", {})
        d = measured.get("depth", {})
        a = measured.get("angle_to_body_midline", {})

        cy = metrics_y
        cy += draw_metric_row(draw, metrics_x, cy, metrics_w, "高度",
                              f"{h.get('cm_above_ground', '?')} cm  ({h.get('category', '?')})",
                              "90-115cm 腰胸",
                              assess_metric(measured, "height"))
        cy += draw_metric_row(draw, metrics_x, cy, metrics_w, "横距",
                              f"{l.get('cm', '?')} cm  ({l.get('category', '?')})",
                              "40-60cm",
                              assess_metric(measured, "lateral"))
        cy += draw_metric_row(draw, metrics_x, cy, metrics_w, "纵深",
                              f"{d.get('cm_in_front_of_hip', '?')} cm  ({d.get('category', '?')})",
                              "前方 30-50cm",
                              assess_metric(measured, "depth"))
        cy += draw_metric_row(draw, metrics_x, cy, metrics_w, "角度",
                              f"{a.get('degrees', '?')}°",
                              "45°",
                              assess_metric(measured, "angle"))

        # Verdict
        v = measured.get("verdict", {}) or {}
        score = v.get("score_0_to_10")
        if score is not None:
            cy += 8
            draw.text((metrics_x, cy), f"评分 {score}/10",
                      fill=band_color, font=font(14))
            cy += 18
            draw.text((metrics_x, cy), v.get("summary", "")[:50],
                      fill=COLOR["ink2"], font=font(11))
    else:
        draw.text((metrics_x, metrics_y), "尚无数据", fill=COLOR["ink3"],
                  font=font(12))

    # Card border
    draw.rectangle((0, 0, card_w - 1, card_h - 1),
                   outline=COLOR["line"], width=1)
    return card


def render_gap_panel(measurements: Dict[str, Dict], width: int) -> Image.Image:
    """Compute and render the 'gap analysis' bottom panel."""
    panel_h = 180
    panel = Image.new("RGBA", (width, panel_h), COLOR["bg"])
    draw = ImageDraw.Draw(panel)

    title_font = font(14)
    head_font = font(11)
    cell_font = font(13)

    draw.text((20, 14), "差距分析（实际 vs 参考）".upper(),
              fill=COLOR["ink3"], font=head_font)
    draw.text((20, 30), "Gap Analysis", fill=COLOR["ink"], font=title_font)

    real = measurements.get("real")
    shadow = measurements.get("shadow")
    coach = measurements.get("coach")

    # Build comparison rows
    rows = []
    def diff(real_dict, ref_dict, dict_key, sub_key, unit):
        if not (real_dict and ref_dict):
            return "—"
        rv = (real_dict.get(dict_key) or {}).get(sub_key)
        cv = (ref_dict.get(dict_key) or {}).get(sub_key)
        if rv is None or cv is None:
            return "—"
        d = rv - cv
        sign = "+" if d > 0 else ""
        return f"{sign}{d}{unit}"

    rows.append(("vs. 空挥", [
        diff(real, shadow, "height", "cm_above_ground", "cm"),
        diff(real, shadow, "lateral_distance", "cm", "cm"),
        diff(real, shadow, "depth", "cm_in_front_of_hip", "cm"),
        diff(real, shadow, "angle_to_body_midline", "degrees", "°"),
    ]))
    rows.append(("vs. 教练", [
        diff(real, coach, "height", "cm_above_ground", "cm"),
        diff(real, coach, "lateral_distance", "cm", "cm"),
        diff(real, coach, "depth", "cm_in_front_of_hip", "cm"),
        diff(real, coach, "angle_to_body_midline", "degrees", "°"),
    ]))

    # Header row
    table_x = 20
    table_y = 70
    col_widths = [110, 120, 120, 140, 90]
    headers = ["对照", "高度", "横距", "纵深", "角度"]
    for i, h in enumerate(headers):
        x = table_x + sum(col_widths[:i])
        draw.text((x, table_y), h, fill=COLOR["ink3"], font=head_font)

    for row_i, (rlabel, vals) in enumerate(rows):
        row_y = table_y + 26 + row_i * 32
        draw.text((table_x, row_y), rlabel, fill=COLOR["ink"], font=cell_font)
        for i, v in enumerate(vals):
            x = table_x + sum(col_widths[: i + 1])
            color = COLOR["clay"] if v != "—" and v != "+0cm" and v != "+0°" else COLOR["ink2"]
            draw.text((x, row_y), v, fill=color, font=cell_font)

    return panel


def compose_final_image(
    measurements: Dict[str, Optional[Dict]],
    frame_paths: Dict[str, Optional[str]],
    output_path: str,
) -> None:
    """Compose the side-by-side comparison + gap panel."""
    card_w = 380
    card_h = 700
    margin = 20
    width = margin * 4 + card_w * 3
    height = margin * 4 + card_h + 180  # cards + gap panel

    final = Image.new("RGBA", (width, height), COLOR["bg"])
    draw = ImageDraw.Draw(final)

    # Title
    draw.text((margin, 20), "击球点对比分析  ·  Contact Point Comparison",
              fill=COLOR["ink"], font=font(20))
    draw.text((margin, 48),
              "FTT 标准 · 击球点应在前胯前方 30-50cm，腰胸高（90-115cm），距身体 40-60cm，角度 45°",
              fill=COLOR["ink3"], font=font(11))

    # Cards
    cards_y = 80
    for i, label in enumerate(["real", "shadow", "coach"]):
        x = margin + i * (card_w + margin)
        card = render_card(label, frame_paths.get(label),
                           measurements.get(label), card_w, card_h)
        final.paste(card, (x, cards_y))

    # Gap panel
    panel = render_gap_panel(measurements, width - margin * 2)
    final.paste(panel, (margin, cards_y + card_h + 20))

    # Footer
    draw.text((margin, height - 24),
              "Generated by Baseline contact_point_compare.py",
              fill=COLOR["ink3"], font=font(10))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final.convert("RGB").save(output_path, "PNG", optimize=True)
    print(f"\n  ✓ Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def analyze_one(video_path: Optional[str], label: str,
                tmp_dir: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Run Gemini + extract impact frame for one video."""
    if not video_path:
        print(f"\n[{label}] 未提供视频")
        return None, None
    if not os.path.exists(video_path):
        print(f"\n[{label}] 文件不存在: {video_path}")
        return None, None

    print(f"\n[{label}] 分析 {video_path}")
    measured = call_gemini_on_video(video_path)
    if not measured:
        print(f"  [!] Gemini analysis failed")
        return None, None

    ts = measured.get("impact_timestamp_s", 0.5)
    print(f"  → 击球瞬间 t={ts}s")
    print(f"  → 高度 {measured.get('height', {}).get('cm_above_ground')}cm "
          f"({measured.get('height', {}).get('category')})")
    print(f"  → 横距 {measured.get('lateral_distance', {}).get('cm')}cm "
          f"({measured.get('lateral_distance', {}).get('category')})")
    print(f"  → 纵深 {measured.get('depth', {}).get('cm_in_front_of_hip')}cm "
          f"({measured.get('depth', {}).get('category')})")
    print(f"  → 角度 {measured.get('angle_to_body_midline', {}).get('degrees')}°")

    frame_out = os.path.join(tmp_dir, f"{label}_impact.jpg")
    if extract_impact_frame(video_path, ts, frame_out):
        return measured, frame_out
    return measured, None


def main():
    ap = argparse.ArgumentParser(description="击球点对比工具")
    ap.add_argument("--real", help="实际击球视频")
    ap.add_argument("--shadow", help="空挥视频（对镜子录的）")
    ap.add_argument("--coach", help="教练 / 参考视频（如 Sinner clip）")
    ap.add_argument("--out", default="output/contact_point_compare.png",
                    help="输出 PNG 路径")
    ap.add_argument("--json", default=None,
                    help="同时保存原始数据 JSON")
    args = ap.parse_args()

    if not (args.real or args.shadow or args.coach):
        ap.error("至少要提供 --real / --shadow / --coach 中的一个")

    measurements: Dict[str, Optional[Dict]] = {}
    frame_paths: Dict[str, Optional[str]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for label, vp in [("real", args.real), ("shadow", args.shadow),
                          ("coach", args.coach)]:
            measured, frame_path = analyze_one(vp, label, tmp)
            measurements[label] = measured
            # Copy frame out of temp dir to a stable path so it survives
            if frame_path:
                stable_path = str(Path(args.out).with_name(f"{label}_impact.jpg"))
                Path(stable_path).parent.mkdir(parents=True, exist_ok=True)
                Path(stable_path).write_bytes(Path(frame_path).read_bytes())
                frame_paths[label] = stable_path
            else:
                frame_paths[label] = None

        compose_final_image(measurements, frame_paths, args.out)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(measurements,
                                              ensure_ascii=False, indent=2))
        print(f"  ✓ JSON saved: {args.json}")

    # Print summary
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for label in ["real", "shadow", "coach"]:
        m = measurements.get(label)
        if m:
            v = m.get("verdict", {}) or {}
            print(f"  {LABEL_TEXT.get(label, label):>10s}: "
                  f"{v.get('score_0_to_10', '?')}/10  "
                  f"{v.get('summary', '')[:40]}")
        else:
            print(f"  {LABEL_TEXT.get(label, label):>10s}: 未提供")


if __name__ == "__main__":
    main()

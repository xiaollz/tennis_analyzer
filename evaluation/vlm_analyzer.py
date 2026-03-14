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
    """Extract 6 key frames from a single swing."""

    def extract(
        self,
        frames_raw: List[np.ndarray],
        frame_indices: List[int],
        swing_event: SwingEvent,
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

        result: List[Tuple[str, np.ndarray]] = []
        for (label, _cn), pos in zip(KEYFRAME_LABELS, positions):
            pos = max(0, min(pos, n - 1))
            result.append((label, frames_raw[pos].copy()))
        return result


# ── Grid creation ───────────────────────────────────────────────────

def create_keyframe_grid(
    keyframes: List[Tuple[str, np.ndarray]],
    cell_width: int = 480,
    cell_height: int = 360,
    label_height: int = 36,
) -> np.ndarray:
    """Arrange up to 6 keyframes into a 2x3 grid with labels."""
    cols, rows = 3, 2
    total_w = cols * cell_width
    total_h = rows * (cell_height + label_height)
    grid = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    cn_labels = [cn for (_, cn) in KEYFRAME_LABELS]

    for idx, (label, frame) in enumerate(keyframes[:6]):
        row, col = divmod(idx, cols)
        x0 = col * cell_width
        y0 = row * (cell_height + label_height)

        cv2.rectangle(grid, (x0, y0), (x0 + cell_width, y0 + label_height), (40, 40, 40), -1)
        cn_text = cn_labels[idx] if idx < len(cn_labels) else label
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts = cv2.getTextSize(cn_text, font, 0.7, 2)[0]
        cv2.putText(grid, cn_text, (x0 + (cell_width - ts[0]) // 2, y0 + label_height - 8),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        resized = cv2.resize(frame, (cell_width, cell_height))
        fy0 = y0 + label_height
        grid[fy0:fy0 + cell_height, x0:x0 + cell_width] = resized

    return grid


def save_keyframe_grid(grid: np.ndarray, path: str) -> str:
    cv2.imwrite(path, grid)
    return path


# ── FTT Prompt ──────────────────────────────────────────────────────

_FTT_SYSTEM_PROMPT = """\
你是一位专业网球教练，精通《The Fault Tolerant Forehand》(FTT) 理论体系。
你将看到一张 2×3 网格图，展示一次正手挥拍的 6 个关键帧（从左到右、从上到下）：
准备完成 → 前挥启动 → 击球前 → 击球瞬间 → 随挥初期 → 随挥结束。

请基于以下 FTT 核心原则进行分析：

【核心原则】
1. 正手是旋转驱动的鞭打系统，手臂是力量的传递者，不是产生者。
2. 网球基本定理：拍面与前臂保持 90-135 度角 + 击球点在身体前方。
3. Out/Up/Through 三向量模型，其中 Out（向外）最关键——它产生 windshield wiper 旋转和上旋。
4. 主动动作 vs 被动结果：髋旋转、腹部展开、胸肌按压是主动的；手腕滞后、球拍旋转、随挥方向是被动结果。
5. 容错性原则：简单准备、放松手腕、大肌群驱动、Out 向量——这些让击球在不完美条件下仍然有效。
6. 重要警告：绝对不要建议"制造 lag"或"刻意 snap 手腕"——这些是身体旋转的被动结果，主动制造会破坏动力链。

【检查清单】
请逐一检查以下 8 个方面：
1. 手臂是否作为整体随身体旋转（vs 前臂独立动作）
2. 击球点是否在身体前方
3. 击球后是否有向外延伸（Out 向量）
4. 是否存在 scooping 模式（手臂先急降再急升）
5. 准备动作是否简洁
6. 头部在击球时是否稳定
7. 随挥是否自然（非刻意）
8. 身体是否旋转驱动（vs 手臂主导）

请以严格 JSON 格式输出，不要包含任何 JSON 之外的文字。JSON 结构如下：
{
  "issues": [
    {
      "name": "问题简称",
      "severity": "高/中/低",
      "description": "描述观察到的问题",
      "ftt_principle": "对应的FTT原则",
      "correction": "纠正建议（不要建议制造lag或刻意snap手腕）"
    }
  ],
  "strengths": ["优点1", "优点2"],
  "overall_assessment": "整体评价（2-3句话）",
  "priority_drill": "最推荐的一个训练方法"
}
"""

_USER_PROMPT = "请分析这次正手挥拍的关键帧。"


# ── Provider backends ───────────────────────────────────────────────

def _encode_image(img: np.ndarray) -> str:
    """BGR image → base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _call_openai_compatible(
    api_key: str, base_url: str, model: str, image_b64: str, user_text: str,
) -> Optional[str]:
    """Call any OpenAI-compatible vision API (Qwen-VL, GPT-4o, proxies)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[VLM] 安装 openai 包: pip install openai")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _FTT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
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
                "type": "base64", "media_type": "image/png", "data": image_b64,
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

    def analyze_swing(
        self,
        keyframe_grid: np.ndarray,
        kpi_summary: str = "",
    ) -> Optional[Dict]:
        if not self.api_key:
            print(f"[VLM] 跳过: 未在 config/vlm_config.json 中设置 api_key")
            return None

        image_b64 = _encode_image(keyframe_grid)
        user_text = _USER_PROMPT
        if kpi_summary:
            user_text += f"\n\n以下是骨骼姿态 KPI 评分摘要供参考：\n{kpi_summary}"

        try:
            if self.provider == "anthropic":
                raw = _call_anthropic(self.api_key, self.base_url, self.model, image_b64, user_text)
            elif self.provider == "gemini":
                raw = _call_gemini(self.api_key, self.model, image_b64, user_text)
            else:
                raw = _call_openai_compatible(self.api_key, self.base_url, self.model, image_b64, user_text)
        except Exception as exc:
            print(f"[VLM] API 调用失败 ({self.provider}/{self.model}): {exc}")
            return None

        if raw is None:
            return None
        return _parse_json_response(raw)


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

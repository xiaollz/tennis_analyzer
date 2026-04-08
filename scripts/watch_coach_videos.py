#!/usr/bin/env python3
"""Watch FTT + Tom Allsopp coach videos via Gemini and extract visual details.

Uses google-genai SDK with packyapi proxy. Saves per-video JSON to
knowledge/extracted/coach_videos_v2/{coach}_{video_id}.json.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.pipeline.video_analyzer import create_client, load_api_config
from google.genai import types

CONFIG_PATH = PROJECT_ROOT / "config" / "youtube_api_config.json"
OUT_DIR = PROJECT_ROOT / "knowledge" / "extracted" / "coach_videos_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 FTT (Johnny / Quentin MF) + 5 Tom Allsopp (TPA Tennis)
VIDEOS = [
    # FTT — Unit Turn / preparation / footwork
    ("ftt", "0m3BMfDDShI", "Build This Foundation – The Rest Will Follow", "Johnny (FTT)"),
    ("ftt", "pWzyP-xfLfU", "Why Your Forehand Doesn't Have Lag (The Pivot Point)", "Johnny (FTT)"),
    ("ftt", "BbGzWTp5pCM", "Why You're Good in Practice but Bad in Matches (Nadal's Secret)", "Johnny (FTT)"),
    ("ftt", "FOmz8Wjv3DQ", "Roger Federer's Footwork Secrets", "Quentin MF (FTT)"),
    ("ftt", "wFIrPMutzRo", "Rotational Power — Side Bending + X Stretch", "Johnny (FTT)"),
    # Tom Allsopp — Unit Turn / preparation
    ("tom", "Vcg_HcHaQ34", "How to get more power on your forehand (Torque & Rotation)", "Tom Allsopp (TPA)"),
    ("tom", "CmXxvX60TOI", "Early Preparation on the Forehand", "Tom Allsopp (TPA)"),
    ("tom", "ubFJi2M3AMM", "Lag Timing — Unit Turn must finish first", "Tom Allsopp (TPA)"),
    ("tom", "utZkaHi9XXM", "Takeback is the result of the turn (not the arm)", "Tom Allsopp (TPA)"),
    ("tom", "M1umUwuPe0w", "Active vs Passive Wrist on the Forehand", "Tom Allsopp (TPA)"),
]

PROMPT_TEMPLATE = """你正在观看由 {coach} 主讲的网球教学视频《{title}》。请仔细观看视频画面与听讲解，
然后用中文按以下结构作答（避免空话，能引用画面细节就引用）：

1. **主题**：本视频具体讲解准备阶段 / Unit Turn / 分裂步 / 步法 的哪个方面？
2. **视觉演示**：列出教练做的每一段示范——身体哪个部位、什么机位（正侧面 / 后斜 45° / 正前方 / 俯拍 / 跟拍）、动作内容。
3. **错 vs 对的对比**：教练是否展示了学生/业余的错误动作 vs 正确动作？描述视觉上的具体差异（关节角度、节奏、停顿等）。
4. **练习/Drill**：视频中是否有明确的训练动作？逐步描述。
5. **使用的机位**：教练拍自己 / 学生时一共用了哪些机位？
6. **关键时间戳**：列出 3-5 个最值得回看的时刻（mm:ss + 发生了什么）。
7. **重复出现的口令 / cue**：教练反复说的英文短语或中文等价说法，原文转写关键的几句。
8. **强调的身体部位**：教练用手指 / 用语言强调了哪些身体部位？
9. **一句话核心收获**：用一句话总结。

回答必须基于你真实看到的画面和听到的话，不要编造。如果某项无法从视频判断，写"视频中未明确"。"""


def analyze_one(client, coach: str, video_id: str, title: str, coach_label: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    prompt = PROMPT_TEMPLATE.format(coach=coach_label, title=title)

    started = time.time()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=url)),
                types.Part(text=prompt),
            ]
        ),
        config=types.GenerateContentConfig(temperature=0),
    )
    elapsed = time.time() - started
    return {
        "coach": coach,
        "coach_label": coach_label,
        "video_id": video_id,
        "title": title,
        "url": url,
        "model": "gemini-3-flash-preview",
        "elapsed_sec": round(elapsed, 1),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "analysis": response.text,
    }


def main():
    config = load_api_config(CONFIG_PATH)
    client = create_client(config)
    print(f"[watch_coach_videos] base_url={config['base_url']} out={OUT_DIR}")

    successes, failures = [], []
    for coach, vid, title, label in VIDEOS:
        out_path = OUT_DIR / f"{coach}_{vid}.json"
        if out_path.exists():
            print(f"[skip] {coach}/{vid} already exists")
            successes.append((coach, vid, title))
            continue
        print(f"[run ] {coach}/{vid} {title[:60]}")
        try:
            result = analyze_one(client, coach, vid, title, label)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"       OK ({result['elapsed_sec']}s, {len(result['analysis'])} chars)")
            successes.append((coach, vid, title))
        except Exception as e:
            print(f"       FAIL: {type(e).__name__}: {e}")
            failures.append((coach, vid, title, str(e)))
        time.sleep(8)

    print(f"\n=== Done. success={len(successes)} fail={len(failures)} ===")
    for f in failures:
        print(f"  FAIL {f[0]}/{f[1]}: {f[3][:200]}")


if __name__ == "__main__":
    main()

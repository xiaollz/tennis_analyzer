#!/usr/bin/env python3
"""Watch Feel Tennis (Tomaz Mencinger) YouTube videos via Gemini and extract
visual feel-to-form details. Output → knowledge/extracted/coach_videos_v2/.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai
from google.genai import types

CONFIG = json.load(open(PROJECT_ROOT / "config" / "youtube_api_config.json"))
OUT_DIR = PROJECT_ROOT / "knowledge" / "extracted" / "coach_videos_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 8 selected Feel Tennis videos covering the requested topics
VIDEOS = [
    ("vcWAEcF6klU", "Tennis Forehand Unit Turn - It's Not A Backswing", "unit_turn"),
    ("guUg4hVI1AE", "Deconstructing A Tennis Forehand Backswing", "backswing_illusion"),
    ("2D7UlPQHce4", "Tennis Forehand Wrist Action: Slap vs Snap Explained", "wrist_lag"),
    ("9KRYA9ZlYmM", "Modern Tennis Forehand Technique In 8 Steps", "modern_forehand_overview"),
    ("5LOKkHpFpFU", "How To Hit A Tennis Forehand - 3 Simple Concepts", "forehand_fundamentals"),
    ("MO01CaN6lFc", "Tennis Forehand Contact Point And How To Find It", "contact_point_timing"),
    ("Auem1-8t3rE", "Why Every Tennis Forehand Starts With An Open Stance", "hip_rotation_stance"),
    ("0Mf8SFX_LuI", "Classic Tennis Forehand vs Modern Forehand Technique", "classic_vs_modern"),
]

PROMPT = """You are watching a tennis instructional video by Tomaz Mencinger from Feel Tennis Instruction. Tomaz is known for "feel-based" teaching. Extract:

1. **Core feel cue**: What body sensation or mental image does he teach?
2. **Visual demonstration**: Describe the demonstrations. Camera angle, body parts shown, what motion.
3. **Wrong vs right**: Does he show a wrong way and a right way? Describe both visually.
4. **Drill**: Any exercises shown? Step by step.
5. **Verbal cues** (transcribe): What words/phrases does he repeat?
6. **Where on body does he point**: Which muscles or body parts does he physically point to?
7. **Key timestamps** (3-5): When does each major teaching moment happen?
8. **Bridge from feel to form**: How does he translate the "feeling" into observable form?
9. **One sentence takeaway**.

Pay special attention to: how he visually demonstrates a sensation (shaking arm, dropping it, exaggerated motions), whether he uses props or slow-motion, and whether he shows a visible pause/wait between preparation and swing.

Answer in Chinese. Return as a JSON object with keys: core_feel_cue, visual_demonstration, wrong_vs_right, drill, verbal_cues, body_points, key_timestamps, feel_to_form_bridge, takeaway."""


def make_client():
    return genai.Client(
        api_key=CONFIG["api_key"],
        http_options=types.HttpOptions(base_url=CONFIG["base_url"], timeout=600000),
    )


def watch_video(client, video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=CONFIG.get("model", "gemini-3-flash-preview"),
                contents=types.Content(parts=[
                    types.Part(file_data=types.FileData(file_uri=url)),
                    types.Part(text=PROMPT),
                ]),
                config=types.GenerateContentConfig(temperature=0),
            )
            return resp.text
        except Exception as e:
            err = str(e)
            if "403" in err:
                raise
            print(f"    attempt {attempt+1} failed: {type(e).__name__}: {err[:120]}", flush=True)
            if attempt < 2:
                time.sleep(min(30 * (2 ** attempt), 90))
    raise RuntimeError(f"all retries failed for {video_id}")


def main():
    client = make_client()
    results = {"ok": [], "failed": []}
    for i, (vid, title, topic) in enumerate(VIDEOS):
        out = OUT_DIR / f"feel_tennis_{vid}.json"
        if out.exists() and out.stat().st_size > 200:
            print(f"[{i+1}/{len(VIDEOS)}] SKIP exists: {vid}", flush=True)
            results["ok"].append((vid, title))
            continue
        print(f"[{i+1}/{len(VIDEOS)}] {vid} {title[:60]}", flush=True)
        try:
            text = watch_video(client, vid)
            # try to parse JSON; otherwise wrap as raw
            payload = {
                "video_id": vid,
                "title": title,
                "topic": topic,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "raw_response": text,
            }
            # Try to extract JSON inside the response
            try:
                t = text.strip()
                if t.startswith("```"):
                    t = t.split("```")[1]
                    if t.startswith("json"):
                        t = t[4:]
                payload["parsed"] = json.loads(t.strip())
            except Exception:
                pass
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            results["ok"].append((vid, title))
            print(f"  OK {len(text)} chars", flush=True)
        except Exception as e:
            results["failed"].append((vid, title, str(e)[:200]))
            print(f"  FAIL: {type(e).__name__}: {str(e)[:160]}", flush=True)
        if i < len(VIDEOS) - 1:
            time.sleep(15)

    print("\n=== DONE ===")
    print(f"OK: {len(results['ok'])}")
    print(f"FAILED: {len(results['failed'])}")
    for vid, title, err in results["failed"]:
        print(f"  {vid}: {err[:120]}")


if __name__ == "__main__":
    main()

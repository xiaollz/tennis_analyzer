#!/usr/bin/env python3
"""Analyze JUL Tennis & Golf 'Federer Forehand Series' via Gemini native video understanding.

Specific videos in scope (6):
  1. kA6PqTN-yls - Federer (5) Chinook pose
  2. mzsNuyJf7IA - Federer (6) passively moved right arm
  3. U0YQUKdGX9I - Federer (10) Do body rotation first
  4. Oiis-Am5FUM - Federer (11) shoulder external rotation
  5. 3MRPqshknlY - Federer (12) shoulder horizontal abduction
  6. 3_2c-kRFLpI - Federer (13) racket head speed

Output: docs/research/jul_tennis_videos/{video_id}.md per video,
plus a synthesis MD assembled by the human/agent afterwards.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "youtube_api_config.json"
OUT_DIR = PROJECT_ROOT / "docs" / "research" / "jul_tennis_videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


VIDEOS = [
    ("kA6PqTN-yls", "Federer (5) Chinook pose"),
    ("mzsNuyJf7IA", "Federer (6) passively moved right arm"),
    ("U0YQUKdGX9I", "Federer (10) Do body rotation first"),
    ("Oiis-Am5FUM", "Federer (11) shoulder external rotation"),
    ("3MRPqshknlY", "Federer (12) shoulder horizontal abduction"),
    ("3_2c-kRFLpI", "Federer (13) racket head speed"),
]


def get_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
    try:
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="replace")
        t = re.search(r"<title>(.*?)(?:\s*-\s*YouTube)?</title>", html)
        c = re.search(r'"ownerChannelName":"([^"]+)"', html)
        length = re.search(r'"lengthSeconds":"(\d+)"', html)
        return {
            "video_id": video_id,
            "url": url,
            "title": t.group(1).strip() if t else "Unknown",
            "channel": c.group(1) if c else "Unknown",
            "length_sec": int(length.group(1)) if length else 0,
        }
    except Exception as e:
        print(f"[meta-warn] {video_id}: {e}")
        return {"video_id": video_id, "url": url, "title": "Unknown",
                "channel": "Unknown", "length_sec": 0}


def _build_client(api_key, base_url):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["http_options"] = types.HttpOptions(
            api_version="v1beta", base_url=base_url, timeout=600000,
        )
    return genai.Client(**kwargs)


_ENDPOINTS: list[tuple] = []


def make_clients():
    global _ENDPOINTS
    if _ENDPOINTS:
        return _ENDPOINTS
    raw = json.loads(CONFIG_PATH.read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    primary_client = _build_client(cfg["api_key"], cfg.get("base_url"))
    primary_model = cfg.get("model", "gemini-3-flash-preview")
    _ENDPOINTS.append((primary_client, primary_model, "primary"))

    bk_key = raw.get("_备用_api_key")
    bk_url = raw.get("_备用_base_url")
    bk_model = raw.get("_备用_model", "gemini-3-flash-preview")
    if bk_key and bk_url:
        _ENDPOINTS.append((_build_client(bk_key, bk_url), bk_model, "backup-packy2"))

    google_key = raw.get("_备用_google_key")
    if google_key:
        _ENDPOINTS.append((_build_client(google_key, None), "gemini-2.5-pro", "backup-google"))

    return _ENDPOINTS


PROMPT = """你是顶级网球生物力学专家 + 知识工程师。下面这段视频来自 JUL Tennis & Golf
频道（中文 + 英文混合教学），是 Federer 正手系列的一集。请你**逐帧 / 逐段实际看完整支视频**
（不是猜内容、不是只读字幕），然后做精确知识提取。

## 已知项目体系（必须做交叉对照）

用户已经建立的网球正手知识体系：
- **HSA**（Horizontal Shoulder Adduction，水平肩内收）= chest fire 物理本体，5/3 命名
- **ESR**（External Shoulder Rotation）= 外旋，准备阶段双外旋（前臂+肱骨）
- **IR 抢跑**（Internal Rotation 抢跑）= 5/16 诊断的根因：内旋过早启动 → 拍头跑前 → 失张力
- **Phase 1 / Phase 2 模型**：5/3 — Phase 1 加载（外旋 + 引拍），Phase 2 释放（HSA + ISR + Pronation）
- **Pat the Dog** = 拍头朝下 + 前臂旋后的引拍 lag 形状
- **Chinook Pose** = ATP 引拍末端的标志姿态（手肘抬高、拍面 vertical、拍头略落）
- **Wrap** = 击球后小臂自然包过身体的收拍
- **双外旋锁定**（5/15 大悟）= 上背 isometric + 前臂旋后 + 肱骨外旋 三件套
- **推肘禁令**（5/6）= 肘前是结果不是动作，禁用主动"推肘"

参考教练对照系：
- FTT (Hugh Clarke) — Fault Tolerant Forehand 体系
- Brian Gordon — Type 3 Forehand
- Tom Allsopp — TPA Tennis 动力链
- Tomaz Mencinger — Feel Tennis 直觉派
- Stephen Bourne — One Minute Tennis

## 你必须输出（中文，markdown，不要省略，6 个章节都要完整）

### §1 视频核心论点（1 句话）
教练在这支视频要说服观众的最核心 ONE thing。

### §2 时间线 + 关键 segment（5-10 个 [hh:mm:ss]）
按时间顺序拆解，每个 segment：
- `[hh:mm:ss]` 起始时间
- 这一段在讲什么 / 演示什么（错误对比？慢动作？分解动作？）
- 教练的核心句子（**英文/中文原话照录**，时间戳精确到秒）

### §3 涉及的肌肉 / 几何 / 时序
- 哪些肌肉被点名？（如：infraspinatus, subscapularis, pec major, lat, serratus...）
- 哪些角度 / 几何关系？（肘高、肩线、躯干 tilt、手臂相对躯干的 H 平面位置）
- 时序：什么先发生、什么后发生？（如果教练讲了顺序）

### §4 教练的核心 cue / 心法 / 比喻
列出教练的所有金句和比喻。每条：
- **原话**（英文 / 中文照录，**< 15 词**）
- 中文翻译
- 时间戳
- 在 FTT / HSA / Pat the Dog / Chinook Pose / IR 抢跑 / ESR 体系中对应什么

### §5 反对的错误模式
教练点名批判了哪些错误做法？每条：
- 错误描述
- 教练为什么说它错
- 在用户体系里这对应哪个 anti-pattern（如 IR 抢跑 / 大臂飘 / 主动推肘 / shoulder dump）

### §6 跟项目体系的精确对接
在以下维度逐项判断（对得上就引用，对不上就说"不涉及"）：
- HSA（水平肩内收 = 胸释放 = chest fire）→ 视频是否触及？哪一段？怎么讲的？
- ESR（外旋）→ 视频怎么讲外旋？是被动还是主动？
- IR 抢跑 → 视频是否警告内旋过早？
- Pat the Dog → 视频里有没有出现这个 lag 形状？
- Chinook Pose → 视频是否讲 prep 末端姿态？
- Phase 1 / Phase 2 → 视频对加载 / 释放的分段是否有自己的版本？
- 推肘禁令 → 视频对"肘"的表述（结果 vs 动作）？

### §7 最关键 3 句话（原话 + 翻译 + 时间戳）
从整支视频里选 **3 句最金的话**（教练的原话），照录 + 时间戳 + 翻译。
< 15 词每句，必须是真的从视频里说出来的，不要编造。

---

⚠ 严格要求：
- **必须基于实际看完视频**输出（你有原生视频理解能力，请用）。
- 时间戳必须真实（不要全部写 [00:00:00]）。
- 原话照录 < 15 词每句，照原文，不要意译。
- 看不清就写"看不清"，不要硬编。
- 中文输出，但英文原话保留。
- 不要 AI 腔，不要泛化。
- 字数：800-1500 字 / 视频。
"""


def analyze_one(video_id: str, hint: str, max_retries: int = 3) -> Path:
    out_path = OUT_DIR / f"{video_id}.md"
    if out_path.exists() and out_path.stat().st_size > 1500:
        print(f"[skip] {video_id} (already analyzed at {out_path})")
        return out_path

    meta = get_metadata(video_id)
    print(f"[analyze] {video_id} | {meta['title']} | {meta['length_sec']}s | hint={hint}")

    contents = [types.Content(role="user", parts=[
        types.Part(file_data=types.FileData(
            file_uri=meta["url"], mime_type="video/mp4")),
        types.Part(text=(
            f"视频 ID：{video_id}\n"
            f"标题：{meta['title']}\n"
            f"系列内容提示：{hint}\n"
            f"时长：{meta['length_sec']} 秒\n"
            f"频道：{meta['channel']}\n\n"
            f"{PROMPT}"
        )),
    ])]

    text = None
    last_err = None
    for ep_idx, (client, model, label) in enumerate(make_clients()):
        for attempt in range(max_retries):
            try:
                print(f"  -> {label} ({model}), attempt {attempt+1}")
                resp = client.models.generate_content(model=model, contents=contents)
                got = (resp.text or "").strip()
                if not got:
                    raise RuntimeError("empty response")
                text = got
                if ep_idx > 0 or attempt > 0:
                    print(f"  [ok] succeeded on {label} attempt {attempt+1}")
                break
            except Exception as e:
                msg = str(e)
                last_err = e
                auth_fatal = any(k in msg for k in ("401", "403", "PERMISSION_DENIED"))
                ep_failed = any(k in msg for k in (
                    "model_not_found", "503", "502", "504", "无可用渠道",
                ))
                if auth_fatal or ep_failed:
                    print(f"  [fail-ep] {label}: {msg[:160]}")
                    break
                if attempt == max_retries - 1:
                    print(f"  [exhausted] {label}: {msg[:160]}")
                    break
                wait = 2 ** attempt
                print(f"  [retry] {label} {attempt+1}/{max_retries} after {wait}s: {msg[:120]}")
                time.sleep(wait)
        if text is not None:
            break

    if text is None:
        raise RuntimeError(f"all endpoints failed for {video_id}: {last_err}")

    header = (
        f"# {video_id} — {meta['title']}\n\n"
        f"- URL: {meta['url']}\n"
        f"- 频道: {meta['channel']} (JUL Tennis & Golf 系列)\n"
        f"- 时长: {meta['length_sec']} 秒\n"
        f"- 系列内容提示: {hint}\n"
        f"- 分析方式: Gemini 原生视频理解 (非字幕爬取)\n\n---\n\n"
    )
    out_path.write_text(header + text, encoding="utf-8")
    print(f"  [saved] {out_path} ({len(text)} chars)")
    return out_path


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for vid, hint in VIDEOS:
        if only and only != vid:
            continue
        try:
            analyze_one(vid, hint)
        except Exception as e:
            print(f"[FAIL] {vid}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()

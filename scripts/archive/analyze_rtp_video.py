#!/usr/bin/env python3
"""Analyze a single Road to Pro Tennis video via Gemini and write markdown.

Usage:
  python3 scripts/analyze_rtp_video.py VIDEO_ID [VIDEO_ID ...]

Output:
  docs/research/road_to_pro_video_analyses/{video_id}.md

Reuses the same prompt structure as analyze_footwork_contact.py — sections
1 through 10, written in Chinese, focused on extracting forehand-relevant
mechanics + integrating with FTT framework.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "youtube_api_config.json"
OUT_DIR = PROJECT_ROOT / "docs" / "research" / "road_to_pro_video_analyses"


PROMPT = """你是顶级网球教练 + 生物力学研究者。仔细看完这段视频每一帧，不要跳。

【背景 — 知识库整合任务】
这个视频来自 Road to Pro Tennis 频道（Sky Kim 教练，前 ATP top-20 教练），是用户新发现的"宝藏频道"，
价值定位与 FTT（The Fault Tolerant Forehand）同等级。任务是把这个视频的内容深度提取出来，
和现有 FTT 知识库结合。

用户当前训练状态（半开放站姿 + 辛纳式 Sinner 正手）：
- 已建立 8 字口令系统：盯/左/架/推/锁/撑/流/撕（再加 Sky Kim 视频带来的"飘"）
- 主要痛点曾经是：击球点 4 坐标全错（偏低/贴身/偏后），Unit Turn 晚，站位太近，重心后倒
- 已识别根因（来自第一个 Sky Kim 视频）：前脚提前落地 → 轴心崩溃 → 击球点被挤
- FTT 框架核心：扭矩(上半身)+位置(下半身)两系统并行，容错性优先

请输出中文分析，目标 3000-5000 字，结构如下：

## 1. 视频元信息
- 讲师 / 时长 / 核心命题（一句话）

## 2. 视频章节拆解（按时间顺序）
分成 4-7 段，每段：
- 起止时间（如 1:30-2:45）
- 讲师说了什么（关键句还原，引用原话尽量保留英文）
- 演示了什么（错误 vs 正确对比）
- 这一段在讲的"真正的点"是什么

## 3. 关键概念清单
列出讲师提到的所有重要术语。每一个：
- 英文原词 + 中文意思
- 讲师在什么情境下用
- 在 FTT 体系里对应的是哪个概念（如有）
- 是新概念还是 FTT 已有概念的更精确表述？

## 4. 对正手击球的具体启发
（如果视频本身就是正手主题——直接展开；如果不是直接的正手主题，
也要写出"这个视角对正手意味着什么"——别敷衍）

## 5. 与用户已有认知的对接
对照 8 字口令系统（盯/左/架/推/锁/撑/流/撕/飘），这个视频：
- 验证 / 加深了哪几个字？
- 引入了什么新动作？
- 有没有和现有口令冲突的地方？

## 6. 训练方法 / drills
讲师演示了哪些具体训练？每个：
- 名字 / 目标 / 步骤 / 验证标准 / 难度（无球/喂球/实战）

## 7. 与 FTT 的对接
- 一致 / 互补 / 冲突？哪一种？
- 视频提了哪些 FTT 没讲过的新角度？
- 视频用了哪些和 FTT 不一样的术语，但说的是同一个东西？
- 如果有冲突，应该信谁？为什么？

## 8. 可执行建议
基于视频内容 + 用户的训练阶段，给出 2-3 个**今天就能在场上做的**具体修正：
- 每条：问题 → 修正 → 验证标准

## 9. 价值评级
- ⭐⭐⭐⭐⭐ 必须收录，作为正手某层的核心参考（指出哪一层）
- ⭐⭐⭐⭐ 强烈推荐
- ⭐⭐⭐ 部分内容值得收录（指出哪些点）
- ⭐⭐ 重申已知，不必单独收录
- ⭐ 内容偏离 / 错误，不要收录

简要说明判断理由。

## 10. 一句话总结
这个视频的唯一核心点是什么？

---

⚠️ 严格要求：
- 必须基于真实观察输出，不要编造
- 不要 AI 腔调、不要套话
- 引用讲师原话尽量还原（哪怕是英文）
- 时间戳要尽量准确
- 如果视频内容偏弱（比如只是营销/泛泛而谈），就老实说，给低评级
"""


def get_metadata(video_id: str) -> dict:
    """Fetch title + channel from YouTube page HTML."""
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
            "title": (t.group(1).strip() if t else "Unknown"),
            "channel": c.group(1) if c else "Road to Pro Tennis",
            "length_sec": int(length.group(1)) if length else 0,
        }
    except Exception:
        return {"video_id": video_id, "url": url, "title": "Unknown",
                "channel": "Road to Pro Tennis", "length_sec": 0}


def make_client() -> tuple[genai.Client, str]:
    """Try primary endpoint, fall back to backup if it fails to reach."""
    raw = json.loads(CONFIG_PATH.read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    backup = {
        "api_key":  raw.get("_备用_api_key"),
        "base_url": raw.get("_备用_base_url"),
        # The backup endpoint hosts a different model lineup — flash isn't
        # available there, but gemini-3.1-pro-preview is, and Pro handles
        # video understanding fine (just costs more tokens).
        "model":    raw.get("_备用_model_override", "gemini-3.1-pro-preview"),
    }

    def try_endpoint(api_key, base_url, model):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["http_options"] = types.HttpOptions(
                api_version="v1beta", base_url=base_url, timeout=600000,
            )
        client = genai.Client(**kwargs)
        # Quick reachability probe — list models is cheap
        try:
            list(client.models.list())
            return client
        except Exception as e:
            print(f"[probe] endpoint {base_url} failed: {e}")
            return None

    primary = try_endpoint(cfg["api_key"], cfg.get("base_url"),
                          cfg.get("model", "gemini-3-flash-preview"))
    if primary is not None:
        print(f"[endpoint] using primary: {cfg.get('base_url')}")
        return primary, cfg.get("model", "gemini-3-flash-preview")

    if backup["api_key"] and backup["base_url"]:
        bk = try_endpoint(backup["api_key"], backup["base_url"], backup["model"])
        if bk is not None:
            print(f"[endpoint] using backup: {backup['base_url']}")
            return bk, backup["model"]

    raise RuntimeError("No reachable Gemini endpoint")


def analyze_one(client: genai.Client, model: str, video_id: str,
                max_retries: int = 4) -> Path:
    out_path = OUT_DIR / f"{video_id}.md"
    if out_path.exists() and out_path.stat().st_size > 500:
        # Skip if already done (lets reruns be idempotent). 500-byte floor
        # guards against half-written files from a previous failure.
        print(f"[skip] {video_id}: already exists at {out_path}")
        return out_path

    meta = get_metadata(video_id)
    print(f"[analyze] {video_id} | {meta['title']} | {meta['length_sec']}s")

    contents = [types.Content(role="user", parts=[
        types.Part(file_data=types.FileData(
            file_uri=meta["url"], mime_type="video/mp4")),
        types.Part(text=(
            f"视频标题：{meta['title']}\n"
            f"频道：{meta['channel']}\n"
            f"时长：{meta['length_sec']} 秒\n\n"
            f"{PROMPT}"
        )),
    ])]

    # Retry on transient errors (SSL handshake fail, 5xx, etc.). Real
    # config errors (401/403/model_not_found) re-raise immediately so
    # we don't waste backoff on hopeless requests.
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            text = resp.text or ""
            if not text.strip():
                raise RuntimeError("empty response from model")
            break
        except Exception as e:
            msg = str(e)
            fatal = any(k in msg for k in ("401", "403", "model_not_found", "PERMISSION_DENIED"))
            if fatal or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt   # 1, 2, 4, 8 seconds
            print(f"[retry] {video_id} attempt {attempt+1}/{max_retries} after {wait}s: {msg[:100]}")
            import time; time.sleep(wait)
            last_err = e
    else:
        raise last_err

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Video Analysis: {meta['title']}\n\n"
        f"- **URL**: {meta['url']}\n"
        f"- **Channel**: {meta['channel']}\n"
        f"- **Length**: {meta['length_sec']}s\n\n"
        f"---\n\n{text}",
        encoding="utf-8",
    )
    print(f"[done] {video_id} → {out_path} ({len(text)} chars)")
    return out_path


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: analyze_rtp_video.py VIDEO_ID [VIDEO_ID ...]", file=sys.stderr)
        return 2
    client, model = make_client()
    print(f"Model: {model}")
    failures = []
    for vid in argv:
        try:
            analyze_one(client, model, vid)
        except Exception as e:
            traceback.print_exc()
            failures.append((vid, str(e)))
    if failures:
        print(f"\n{len(failures)} failures:")
        for vid, err in failures:
            print(f"  {vid}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

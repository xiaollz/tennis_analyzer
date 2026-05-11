#!/usr/bin/env python3
"""JUL Tennis Rubber Arm series - per-video Gemini extraction.

Outputs raw analyses to docs/research/jul_tennis_videos/<id>.md.
Synthesis (rubber_arm_series_synthesis.md) is composed afterward.

Usage:
  python3 scripts/analyze_jul_rubber_arm.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "youtube_api_config.json"
OUT_DIR = PROJECT_ROOT / "docs" / "research" / "jul_tennis_videos"

VIDEOS = [
    ("ejuULcyQbvI", "Rubber arm 1: Completely relaxed dominant hand"),
    ("SITTGP-7jMA", "Rubber arm 4: Seesaw mechanism"),
    ("2N8aXJE-ImY", "Rubber arm 5: Symmetric rotation of both upper arms"),
    ("D5HW7VFwkis", "Rubber arm 6: Stretch reflex & law of inertia"),
    ("F2vynJYGNoI", "Rubber arm 8: Shoulder horizontal abduction"),
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
            "title": (t.group(1).strip() if t else "Unknown"),
            "channel": c.group(1) if c else "Unknown",
            "length_sec": int(length.group(1)) if length else 0,
        }
    except Exception:
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


def make_client() -> tuple[genai.Client, str]:
    global _ENDPOINTS
    if _ENDPOINTS:
        return _ENDPOINTS[0][0], _ENDPOINTS[0][1]
    raw = json.loads(CONFIG_PATH.read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    primary_client = _build_client(cfg["api_key"], cfg.get("base_url"))
    primary_model = cfg.get("model", "gemini-3-flash-preview")
    _ENDPOINTS.append((primary_client, primary_model, "primary"))
    bk_key = raw.get("_备用_api_key")
    bk_url = raw.get("_备用_base_url")
    bk_model = raw.get("_备用_model", "gemini-3-flash-preview")
    if bk_key and bk_url:
        bk_client = _build_client(bk_key, bk_url)
        _ENDPOINTS.append((bk_client, bk_model, "backup"))
    return _ENDPOINTS[0][0], _ENDPOINTS[0][1]


PROMPT = """你是网球生物力学专家 + 反射神经控制研究者。视频出自 JUL Tennis & Golf 频道的 "Rubber Arm" 系列。
完整看完每一帧，不要跳。然后按下面结构输出中文分析（3500-5000 字）。

【项目背景 — 必须对照分析】
用户当前根因：**IR 抢跑（内旋肌过早激活）**。
具体是：肩内旋肌群（三角肌前束 + 肩胛下肌 + 胸大肌 + 背阔肌 + 大圆肌）在 Phase 1（拉拍 + 转体）时
就开始主动收缩，把大臂往前推 + 提早内旋，破坏了"被动手臂"机制。
修正路径：主动 ER（冈下肌 + 小圆肌外旋）→ 反射性抑制 IR 群 → 大臂被躯干带动 → 击球瞬间 ISR 释放。
用户 5/16 受伤教训：试图"用背抬小臂" → 二头肌 + 旋前圆肌 isometric load → 内上髁炎。
用户已有体系：HSA（Horizontal Shoulder Adduction）= chest fire 物理本体，
但"推"是结果不是动作 — 蹬转输入 + 背 isometric 托住 = 肘自动前。

【任务】
1. 提取这支视频独有的"被动手臂"机制描述
2. 找出讲师如何让学员**关闭主动 IR、激活被动机制**
3. 列出所有具体 drill（精确到步骤、计数、感觉描述）
4. 翻译并保留讲师所有金句原话
5. 把这套机制和"IR 抢跑"做精确映射

【输出结构】

## 1. 视频元信息
- 视频编号 / 标题（中英）/ 时长 / 核心论点（一句话不超过 30 字）

## 2. 时间戳分段（4-7 段）
每段：
- `[mm:ss-mm:ss]` 段落主题
- 讲师**原话引用**（保留英文 + 中文翻译）
- 演示了什么动作 / 错误对比
- 这段在讲的物理机制（要写解剖学层面：哪些肌肉、什么反射、什么神经控制）

## 3. 核心机制提取
讲师讲的"rubber arm"在这支视频里指什么？
- 物理定义（关节松紧 / 肌肉激活模式）
- 神经控制原理（被动 vs 主动 / 反射 vs 意志）
- 涉及的反射机制（stretch reflex / Hoffmann reflex / reciprocal inhibition / golgi tendon）

## 4. 反对的错误模式
讲师明确反对的所有错误（要带原话）：
- 是哪一类肌肉过早 / 过度激活？
- 跟用户"IR 抢跑"是不是同一个错误？

## 5. Drill 提取（最重点）
讲师演示 / 推荐的所有训练。每个 drill 必须写齐：
- **名字**（讲师叫法 + 中文）
- **目标**（训练什么神经-肌肉系统）
- **步骤**（精确到每一步、每多少次、休息多久）
- **关键感觉**（讲师说"应该感觉到"什么）
- **判定标准**（怎么知道做对了）
- **难度**（无球 / 镜前 / 喂球 / 实战）
- **是否可在家用 TRX / 肋木架 / 哑铃做**

## 6. 与"IR 抢跑"根因的精确映射
**这是项目级核心问题，不能糊弄**。
- 讲师说的"rubber arm" 失败模式 ≈ 用户的 IR 抢跑？同一现象不同语言？
- 讲师的修正方向（被动 / 反射）能否反向抑制 IR 群？
- 讲师有没有提到 ER（外旋）作为 IR 抑制器？
- 哪些 drill 能直接用来做"主动 ER 训练"？

## 7. 与用户 HSA / 5/6 推肘禁令体系的对接
- 一致 / 互补 / 冲突？
- 讲师对"主动 vs 被动"的边界划在哪？跟用户"蹬+托"原则一致吗？
- 有没有讲师提到的新概念，可以补强 HSA 框架？

## 8. 训练优先级建议
基于这一支视频，给用户：
- 最高优先级 drill 1 个（明天就练）
- 次级 drill 2-3 个（一周内引入）
- 跳过的 drill（如有）+ 理由

## 9. 金句 Top 5
讲师原话 + 翻译 + 出处时间戳。**必须真的是讲师说的**，不要编。

## 10. 价值评级 + 一句话总结
⭐⭐⭐⭐⭐ / ⭐⭐⭐⭐ / ⭐⭐⭐ / ⭐⭐ / ⭐
一句话：这支视频对用户突破 IR 抢跑根因的核心贡献是什么？

---

⚠ 硬性要求：
- 时间戳必须基于真实观察，不要编
- 引用必须是讲师真实说的英文 + 准确中文
- 解剖学描述要精确（肌肉名字、动作方向、关节角度）
- 不要 AI 腔、不要泛泛而谈
- 如果讲师没讲某一项，老实写"视频未涉及"
"""


def analyze_one(client, model, video_id, video_title_hint, max_retries=3):
    out_path = OUT_DIR / f"{video_id}.md"
    if out_path.exists() and out_path.stat().st_size > 1500:
        print(f"[skip] {video_id}: exists ({out_path.stat().st_size} bytes)")
        return out_path

    meta = get_metadata(video_id)
    print(f"[analyze] {video_id} | {meta['title']} | {meta['length_sec']}s")

    contents = [types.Content(role="user", parts=[
        types.Part(file_data=types.FileData(
            file_uri=meta["url"], mime_type="video/mp4")),
        types.Part(text=(
            f"视频标题：{meta['title']}\n"
            f"标题提示：{video_title_hint}\n"
            f"频道：{meta['channel']}\n"
            f"时长：{meta['length_sec']} 秒\n\n"
            f"{PROMPT}"
        )),
    ])]

    text = None
    last_err = None
    for ep_idx, (ep_client, ep_model, ep_label) in enumerate(_ENDPOINTS):
        for attempt in range(max_retries):
            try:
                resp = ep_client.models.generate_content(model=ep_model, contents=contents)
                got = resp.text or ""
                if not got.strip():
                    raise RuntimeError("empty response")
                text = got
                if ep_idx > 0:
                    print(f"[ok] {video_id}: succeeded on {ep_label}")
                break
            except Exception as e:
                msg = str(e)
                last_err = e
                auth_fatal = any(k in msg for k in ("401", "403", "PERMISSION_DENIED"))
                ep_failed = any(k in msg for k in (
                    "model_not_found", "503", "502", "504", "无可用渠道",
                ))
                if auth_fatal or ep_failed:
                    print(f"[fail] {video_id} on {ep_label}: {msg[:120]}")
                    break
                if attempt == max_retries - 1:
                    print(f"[exhausted] {video_id} on {ep_label}: {msg[:120]}")
                    break
                wait = 2 ** attempt
                print(f"[retry] {video_id} {ep_label} attempt {attempt+1}/{max_retries} after {wait}s: {msg[:80]}")
                time.sleep(wait)
        if text is not None:
            break

    if text is None:
        raise last_err if last_err else RuntimeError(f"All endpoints failed for {video_id}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Video Analysis: {meta['title']}\n\n"
        f"- **URL**: {meta['url']}\n"
        f"- **Channel**: {meta['channel']}\n"
        f"- **Length**: {meta['length_sec']}s\n"
        f"- **Series**: JUL Tennis Rubber Arm\n\n"
        f"---\n\n{text}",
        encoding="utf-8",
    )
    print(f"[done] {video_id} → {out_path} ({len(text)} chars)")
    return out_path


def main():
    client, model = make_client()
    print(f"Model: {model}")
    print(f"Output: {OUT_DIR}")
    print()

    failures = []
    for vid, title in VIDEOS:
        try:
            analyze_one(client, model, vid, title)
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
    sys.exit(main())

#!/usr/bin/env python3
"""JUL Tennis & Golf 频道 Djokovic + Nadal + Hip turn 系列视频分析器.

针对用户的"伪 Unit Turn / IR 抢跑"根因诊断 + Federer/Djokovic/Nadal 三球员风格对比，
用 Gemini 原生 file_data 一支一支吃透，再交给主对话做综合。

输出：每支视频独立 markdown → docs/research/jul_tennis_videos/{video_id}.md
最终综合 → docs/research/jul_tennis_videos/djokovic_nadal_concepts_synthesis.md (主对话写)
"""
from __future__ import annotations

import argparse
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

# 5 支必读视频（按用户指定顺序）
VIDEOS = [
    ("kFBqvdtEz0o", "Djokovic forehand (1) Body rotation axis", 176),
    ("vppG84O27Ic", "Djokovic forehand (3) shoulder horizontal abduction", 851),
    ("t8U9hdTthSs", "Nadal forehand (1) body rotation", 159),
    ("Ufw66Jfuu8U", "Nadal forehand (2) racket acceleration", 847),
    ("h7BYkIFKzrA", "Hip turn swing vs Whole body swing Part 1", 1253),
]


def get_metadata(video_id: str, fallback_title: str = "", fallback_length: int = 0) -> dict:
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
            "title": (t.group(1).strip() if t else fallback_title or "Unknown"),
            "channel": c.group(1) if c else "JUL Tennis & Golf",
            "length_sec": int(length.group(1)) if length else fallback_length,
        }
    except Exception:
        return {"video_id": video_id, "url": url,
                "title": fallback_title or "Unknown",
                "channel": "JUL Tennis & Golf",
                "length_sec": fallback_length}


def _build_client(api_key, base_url):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["http_options"] = types.HttpOptions(
            api_version="v1beta", base_url=base_url, timeout=600000,
        )
    return genai.Client(**kwargs)


_ENDPOINTS: list[tuple] = []


def make_endpoints():
    global _ENDPOINTS
    if _ENDPOINTS:
        return
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


def build_prompt(video_id: str, title: str, length_sec: int) -> str:
    return f"""你是顶级网球生物力学研究者。仔细看完这段 JUL Tennis & Golf 频道的视频每一帧。

【视频】{title}（ID: {video_id}, {length_sec}s）

【用户当前根因诊断】
- 主问题：**IR 抢跑 + 伪 Unit Turn**（用腰转代替胯转，导致下肢蹬转无力 / 上身动力链断裂）
- 4/27 圣经：右脚为轴 + 转髋（不是转腰）
- 5/3 HSA 框架：胸肱角主动闭合 = chest fire 物理本体
- 5/15 大悟：双外旋锁定 + Chinook Pose（前臂外旋 + 肩外旋形成"加农炮"姿态）
- 5/16 缺口：髋转幅度不够 → 蹬转无力 → 上身被迫代偿
- 5/6 推肘禁令：肘前是物理结果，禁用主动"推""送"

【三球员风格已知差异】
- Federer：拍头领先 + 身体辅助（小臂为主）
- Djokovic：身体驱动型（urgent 风格）+ 长 takeback + 极致 hip-shoulder separation
- Nadal：极端身体扭转 + 巨大 racket head speed + 上臂内旋猛烈

【任务】
深度提取这支视频的内容。重点关注以下 6 个维度：

## 1. 视频元信息
- 讲师 / 时长 / 核心命题（一句话）

## 2. 视频章节拆解（按时间顺序）
分 4-7 段，**每段必须有时间戳**（如 [0:25-1:10]）：
- 起止时间
- 讲师说了什么（**关键句必须英文原话还原**，不要意译）
- 演示了什么（错误 vs 正确对比 / 球员对比）
- 这一段的"真正的点"

## 3. 关键概念清单
列出讲师所有重要术语：
- 英文原词 + 中文解释
- 是新概念，还是已有概念的精确表述？
- 跟 HSA / 双外旋 / 右脚轴 / 转髋 等用户体系对应关系

## 4. 跟 Federer 对比
讲师有没有**明示或暗示**这位球员（Djokovic 或 Nadal）跟 Federer 有什么风格差异？
特别关注：
- 发力分配（身体 vs 手臂）
- Takeback 路径
- Hip rotation 启动时机
- Shoulder separation 深度
- 击球瞬间的关节驱动顺序

## 5. 跟用户根因的对接（最关键）
- 这个视频是否能解释"伪 Unit Turn"或"IR 抢跑"？
- 讲师讲的"hip turn / body rotation / racket acceleration"是不是 = 用户的"转髋 / 双外旋 / HSA"？
- 有没有讲到"如果 hip 没转够会发生什么"这样的 failure mode？
- 视频里的训练方法 / drill 用户能不能借鉴？

## 6. 教练 cue / 比喻 / 心法
讲师用了哪些**形象的语言 / 类比 / cue**？（拳击/高尔夫/弹弓/...）
**全部列出来**，每条注明时间戳。

## 7. 一句话总结
这支视频的唯一核心点是什么？

---

⚠ 严格要求：
- 必须基于真实观察，不要编造
- 时间戳必须精确（精确到秒，如 [3:42]）
- 引用讲师原话（保留英文）
- 中文输出，目标 2500-4000 字
- 内容偏弱就老实说"无独特价值"
"""


def analyze_one(video_id: str, fallback_title: str, fallback_length: int,
                out_dir: Path, max_retries: int = 4) -> Path:
    out_path = out_dir / f"{video_id}.md"
    if out_path.exists() and out_path.stat().st_size > 1500:
        print(f"[skip] {video_id}: exists at {out_path}")
        return out_path

    meta = get_metadata(video_id, fallback_title, fallback_length)
    print(f"[analyze] {video_id} | {meta['title']} | {meta['length_sec']}s")

    prompt = build_prompt(video_id, meta['title'], meta['length_sec'])
    contents = [types.Content(role="user", parts=[
        types.Part(file_data=types.FileData(
            file_uri=meta["url"], mime_type="video/mp4")),
        types.Part(text=(
            f"视频标题：{meta['title']}\n"
            f"频道：{meta['channel']}\n"
            f"时长：{meta['length_sec']} 秒\n\n"
            f"{prompt}"
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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Video Analysis: {meta['title']}\n\n"
        f"- **URL**: {meta['url']}\n"
        f"- **Channel**: {meta['channel']}\n"
        f"- **Length**: {meta['length_sec']}s\n"
        f"- **Video ID**: {video_id}\n\n"
        f"---\n\n{text}",
        encoding="utf-8",
    )
    print(f"[done] {video_id} → {out_path} ({len(text)} chars)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="single video id, otherwise process all 5")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "docs" / "research" / "jul_tennis_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_endpoints()
    print(f"Endpoints: {[e[2] for e in _ENDPOINTS]}")
    print(f"Output: {out_dir}\n")

    targets = [(v, t, l) for (v, t, l) in VIDEOS
               if (args.video is None or v == args.video)]

    failures = []
    for vid, title, length in targets:
        try:
            analyze_one(vid, title, length, out_dir)
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

#!/usr/bin/env python3
"""Generic per-channel forehand video analyzer.

Takes a channel slug + name + video IDs, analyzes each via Gemini,
writes per-video markdown to `docs/research/{slug}_video_analyses/{video_id}.md`.

Replaces the channel-specific scripts (analyze_rtp_video.py, analyze_tpa_video.py)
which had near-identical logic.

Usage:
  python3 scripts/analyze_channel_video.py \\
      --slug ftcoach --channel-name "Feel Tennis (Tomaz Mencinger)" \\
      VIDEO_ID1 VIDEO_ID2 ...
"""
from __future__ import annotations

import argparse
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


def make_client() -> tuple[genai.Client, str]:
    raw = json.loads(CONFIG_PATH.read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    backup = {
        "api_key":  raw.get("_备用_api_key"),
        "base_url": raw.get("_备用_base_url"),
        "model":    raw.get("_备用_model_override", "gemini-3.1-pro-preview"),
    }

    def try_endpoint(api_key, base_url, model):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["http_options"] = types.HttpOptions(
                api_version="v1beta", base_url=base_url, timeout=600000,
            )
        client = genai.Client(**kwargs)
        try:
            list(client.models.list())
            return client
        except Exception as e:
            print(f"[probe] endpoint {base_url} failed: {e}", file=sys.stderr)
            return None

    primary = try_endpoint(cfg["api_key"], cfg.get("base_url"),
                          cfg.get("model", "gemini-3-flash-preview"))
    if primary is not None:
        return primary, cfg.get("model", "gemini-3-flash-preview")

    if backup["api_key"] and backup["base_url"]:
        bk = try_endpoint(backup["api_key"], backup["base_url"], backup["model"])
        if bk is not None:
            return bk, backup["model"]

    raise RuntimeError("No reachable Gemini endpoint")


def build_prompt(channel_name: str) -> str:
    return f"""你是顶级网球教练 + 生物力学研究者。仔细看完这段视频每一帧，不要跳。

【背景 — 知识库整合任务】
这个视频来自 **{channel_name}** 频道，是用户网球正手知识库扩展的一部分。
用户已经深度建立了三个独立体系：
- FTT (The Fault Tolerant Forehand)：地图——10 层 taxonomy + 容错性公理
- RTP (Road to Pro Tennis / Sky Kim)：GPS——可视觉验证的脚-肘-肩物理标志
- TPA Tennis (Tom Allsopp)：发动机说明书——动力链时序 + 肌肉张力管理

任务是把这个视频的内容深度提取出来，找它的**独特角度**——
它和 FTT/RTP/TPA 比，**最不一样的那 1-2 个观点**是什么。

用户当前训练状态（4/29，半开放站姿 + 辛纳式 Sinner 正手）：

**圣经级顿悟（4/27 晚）**：所有现代正手力学都服务于**让右脚成为旋转轴**。
区分"开放动力链"（左脚为轴=抛手臂）vs"闭合动力链"（右脚为轴=旋转）。

**11 字口令系统**：盯/左/架/推/锁/撑/流/撕（核心 8 字）+ 飘（左脚不踩死）
+ 藏（肘藏到躯干右侧 = 肩部加载）+ 压（右脚承重，被动加载）

**已固化的三条诊断链**：
1. early_front_foot_landing（前脚提前落地→轴心崩溃）
2. wta_takeback_midline_violation（拍头过早倒后→张力丢失）
3. arming_the_shot_false_lag（动作形状对但球软）

请输出中文分析，目标 3000-5000 字，结构如下：

## 1. 视频元信息
- 讲师 / 时长 / 核心命题（一句话）

## 2. 视频章节拆解（按时间顺序）
分成 4-7 段，每段：
- 起止时间
- 讲师说了什么（关键句还原，引用原话尽量保留英文）
- 演示了什么（错误 vs 正确对比）
- 这一段在讲的"真正的点"是什么

## 3. 关键概念清单
列出讲师提到的所有重要术语。每一个：
- 英文原词 + 中文意思
- 在 FTT/RTP/TPA 体系里对应的是哪个概念（如有）
- 是新概念还是已有概念的更精确表述？

## 4. 这位教练的独特切入角度（最重要）
**一定要明确**：这个视频/教练相对 FTT/RTP/TPA，最独特的角度是什么？
比如：
- 是不是从 Pronation/Supination 角度？
- 是不是从 Shoulder Lift 角度？
- 是不是从 Feel / 感觉派？
- 是不是从 Slow Motion frame analysis？
- 是不是从 specific muscle activation？
- 是不是从 timing / 节奏？

如果完全和 FTT/RTP/TPA 重合，就老实说"无独特角度"——给低评级。

## 5. 与用户已有认知的对接
对照 11 字系统（盯/左/架/推/锁/撑/流/撕/飘/藏/压）：
- 验证 / 加深了哪几个字？
- 引入了什么新动作或新认知？
- 与"右脚为轴"圣经原则的关系？
- 有没有冲突？

## 6. 训练方法 / drills
讲师演示了哪些具体训练？每个：
名字 / 目标 / 步骤 / 验证标准 / 难度（无球/喂球/实战）

## 7. 与 FTT/RTP/TPA 的对接
- 一致 / 互补 / 冲突？
- 提了哪些三方都没讲过的新角度？
- 用了哪些和三方不一样的术语，但说的是同一件事？
- 如果有冲突，应该信谁？理由？

## 8. 可执行建议
基于视频 + 用户阶段，给 2-3 个**今天就能做的**修正：
- 每条：问题 → 修正 → 验证标准

## 9. 价值评级
- ⭐⭐⭐⭐⭐ 必须收录，作为正手某层的核心参考
- ⭐⭐⭐⭐ 强烈推荐
- ⭐⭐⭐ 部分内容值得收录
- ⭐⭐ 重申已知，不必单独收录
- ⭐ 内容偏离 / 错误，不要收录

## 10. 一句话总结
这个视频的唯一核心点是什么？

---

⚠ 严格要求：
- 必须基于真实观察输出，不要编造
- 不要 AI 腔
- 引用讲师原话（哪怕英文）
- 时间戳尽量准确
- 内容偏弱就老实说
"""


def analyze_one(client: genai.Client, model: str, video_id: str,
                channel_name: str, out_dir: Path,
                max_retries: int = 4) -> Path:
    out_path = out_dir / f"{video_id}.md"
    if out_path.exists() and out_path.stat().st_size > 500:
        print(f"[skip] {video_id}: exists at {out_path}")
        return out_path

    meta = get_metadata(video_id)
    print(f"[analyze] {video_id} | {meta['title']} | {meta['length_sec']}s")

    prompt = build_prompt(channel_name)
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

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            text = resp.text or ""
            if not text.strip():
                raise RuntimeError("empty response")
            break
        except Exception as e:
            msg = str(e)
            fatal = any(k in msg for k in ("401", "403", "model_not_found", "PERMISSION_DENIED"))
            if fatal or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[retry] {video_id} attempt {attempt+1}/{max_retries} after {wait}s: {msg[:100]}")
            import time; time.sleep(wait)
            last_err = e
    else:
        raise last_err

    out_dir.mkdir(parents=True, exist_ok=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="folder slug, e.g. 'feel_tennis'")
    ap.add_argument("--channel-name", required=True,
                    help="display name, e.g. 'Feel Tennis (Tomaz Mencinger)'")
    ap.add_argument("video_ids", nargs="+", help="YouTube video IDs to analyze")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "docs" / "research" / f"{args.slug}_video_analyses"

    client, model = make_client()
    print(f"Model: {model}")
    print(f"Output: {out_dir}")
    print(f"Channel: {args.channel_name}")
    print()

    failures = []
    for vid in args.video_ids:
        try:
            analyze_one(client, model, vid, args.channel_name, out_dir)
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

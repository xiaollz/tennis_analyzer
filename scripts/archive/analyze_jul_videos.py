#!/usr/bin/env python3
"""JUL Tennis & Golf channel deep mechanism extractor.

5 videos targeted at:
1. Arm swing vs Body swing (zf6dBOucjqg) — long, project-level
2. Hypothenar eminence + ER (zPSEjRrDPQw) — long, anatomical cue
3. Federer balance + left arm (eOqOQWjLf3c)
4. Federer pull back upper arm (9woj7lzAqx4)
5. Federer symmetric upper arm rotation (8FBN_hykgdA)
"""
from __future__ import annotations
import json, sys, re, urllib.request, traceback, time
from pathlib import Path
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "youtube_api_config.json"
OUT = ROOT / "docs" / "research" / "jul_tennis_videos"

VIDEOS = [
    ("zf6dBOucjqg", "Arm swing vs Body swing"),
    ("zPSEjRrDPQw", "How to make deep ER - Hypothenar eminence"),
    ("eOqOQWjLf3c", "Federer (3) Balance and left arm"),
    ("9woj7lzAqx4", "Federer (7) pull back upper arm harshly"),
    ("8FBN_hykgdA", "Federer (14) symmetric rotation of both upper arms"),
]


def get_meta(vid):
    url = f"https://www.youtube.com/watch?v={vid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        h = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
        t = re.search(r"<title>(.*?)(?:\s*-\s*YouTube)?</title>", h)
        c = re.search(r'"ownerChannelName":"([^"]+)"', h)
        L = re.search(r'"lengthSeconds":"(\d+)"', h)
        return {"url": url, "title": t.group(1).strip() if t else "Unknown",
                "channel": c.group(1) if c else "Unknown",
                "length_sec": int(L.group(1)) if L else 0}
    except Exception:
        return {"url": url, "title": "Unknown", "channel": "Unknown", "length_sec": 0}


def build_client():
    raw = json.loads(CFG.read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    eps = []
    pri = genai.Client(api_key=cfg["api_key"], http_options=types.HttpOptions(
        api_version="v1beta", base_url=cfg.get("base_url"), timeout=600000))
    eps.append((pri, cfg.get("model", "gemini-3-flash-preview"), "primary"))
    bk_key = raw.get("_备用_api_key"); bk_url = raw.get("_备用_base_url")
    bk_model = raw.get("_备用_model", "gemini-3-flash-preview")
    if bk_key and bk_url:
        bk = genai.Client(api_key=bk_key, http_options=types.HttpOptions(
            api_version="v1beta", base_url=bk_url, timeout=600000))
        eps.append((bk, bk_model, "backup"))
    return eps


PROMPT_LONG = """你是顶级网球生物力学研究者。这是 JUL Tennis & Golf 频道的核心机制视频，需要极深度提取。

【用户当前阶段】
- 半西方握法 (semi-western grip)
- 已识别根因：IR 抢跑（internal rotation 起跳过早）
- 5/15 大悟：双外旋锁定（肩 ER + 前臂位置稳定）
- HSA 框架：Horizontal Shoulder Adduction = 力量本体
- 5/6 推肘禁令：肘前是物理结果，不是主动动作

【任务】
看完每一帧。这是一支**长视频**（必须深度），我需要你按下面结构输出 4500-6000 字中文。

## 1. 视频基本信息
讲师 / 时长 / 一句话核心命题

## 2. 完整章节拆解（按时间戳，每 1-3 分钟一段）
每段：
- [MM:SS - MM:SS] 时间戳
- 讲师说了什么（关键句尽量保留英文原话）
- 演示了什么（错误 vs 正确对比）
- 这段在讲的"真正物理机制"是什么

## 3. 关键概念清单（详尽）
讲师提到的所有重要术语，每一个：
- 英文原词 + 中文意思
- 解剖位置 / 物理含义
- 在用户已有体系（HSA / 双外旋 / 半西方握法）里对应什么

## 4. 这位教练的独特角度（最重要！）
JUL 教练相对其他人最独特的点是什么？

## 5. 与用户根因（IR 抢跑）的对接
- 这视频如何解释 / 解决 IR 抢跑？
- 引入了什么用户没接触过的精确机制？

## 6. 与"双外旋锁定"的对接
- 肩 ER + 前臂稳定，这视频有什么补充？
- 有没有冲突？

## 7. 训练方法 / drills
所有具体训练，每个：名称 / 目标 / 步骤 / 验证标准

## 8. 关键金句（5-10 句，逐字 + 时间戳）

## 9. 价值评级
⭐⭐⭐⭐⭐ → ⭐ + 理由

## 10. 一句话总结

⚠ 严格要求：必须真实观察、不编造、时间戳精确、英文术语保留原文、不要 AI 腔。
"""

PROMPT_SHORT = """你是顶级网球生物力学研究者。这是 JUL Tennis & Golf 频道的费德勒拆解短视频。

【用户当前阶段】
- 半西方握法
- IR 抢跑根因
- 5/15 双外旋锁定大悟
- HSA = 力量本体
- 5/6 推肘禁令（肘前是结果）

看完每一帧。短视频但必须细致。输出 1500-2500 字中文。

## 1. 视频基本信息
## 2. 章节拆解（按时间戳，30 秒粒度）
## 3. 关键概念
## 4. 独特角度
## 5. 与"双外旋锁定" + IR 抢跑的对接
## 6. 关键金句（3-5 句 + 时间戳）
## 7. 价值评级 + 一句话总结

⚠ 真实观察、时间戳精确、英文术语保留。
"""


def analyze(eps, vid, friendly):
    out = OUT / f"{vid}.md"
    if out.exists() and out.stat().st_size > 800:
        print(f"[skip] {vid}: exists")
        return out
    OUT.mkdir(parents=True, exist_ok=True)
    meta = get_meta(vid)
    is_long = meta["length_sec"] > 600
    prompt = PROMPT_LONG if is_long else PROMPT_SHORT
    print(f"[analyze] {vid} | {friendly} | {meta['length_sec']}s | "
          f"{'LONG' if is_long else 'SHORT'}")
    contents = [types.Content(role="user", parts=[
        types.Part(file_data=types.FileData(file_uri=meta["url"], mime_type="video/mp4")),
        types.Part(text=f"视频标题：{meta['title']}\n时长：{meta['length_sec']}s\n\n{prompt}"),
    ])]
    text = None
    last_err = None
    for ep_client, ep_model, label in eps:
        for attempt in range(4):
            try:
                resp = ep_client.models.generate_content(model=ep_model, contents=contents)
                got = resp.text or ""
                if not got.strip():
                    raise RuntimeError("empty response")
                text = got
                if label != "primary":
                    print(f"[ok] {vid} via {label}")
                break
            except Exception as e:
                msg = str(e); last_err = e
                fatal = any(k in msg for k in ("401","403","PERMISSION_DENIED"))
                ep_fail = any(k in msg for k in ("model_not_found","503","502","504","无可用渠道"))
                if fatal or ep_fail:
                    print(f"[fail-ep] {vid} on {label}: {msg[:120]}")
                    break
                if attempt == 3:
                    print(f"[exhausted] {vid} on {label}: {msg[:120]}")
                    break
                w = 2 ** attempt
                print(f"[retry] {vid} {label} {attempt+1}/4 after {w}s: {msg[:80]}")
                time.sleep(w)
        if text is not None:
            break
    if text is None:
        raise last_err if last_err else RuntimeError(f"all eps failed: {vid}")

    out.write_text(
        f"# {meta['title']}\n\n"
        f"- URL: {meta['url']}\n- Channel: {meta['channel']}\n"
        f"- Length: {meta['length_sec']}s\n- Friendly: {friendly}\n\n---\n\n{text}",
        encoding="utf-8")
    print(f"[done] {vid} → {out} ({len(text)} chars)")
    return out


def main():
    eps = build_client()
    print(f"Endpoints: {[(l) for _, _, l in eps]}")
    fails = []
    for vid, name in VIDEOS:
        try:
            analyze(eps, vid, name)
        except Exception as e:
            traceback.print_exc()
            fails.append((vid, str(e)))
    if fails:
        print(f"\n{len(fails)} failed:")
        for v, e in fails: print(f"  {v}: {e[:120]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

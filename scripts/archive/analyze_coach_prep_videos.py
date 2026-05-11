#!/usr/bin/env python3
"""Analyze coach demonstration videos focused on preparation / unit turn.

Specifically: how do they keep the upper arm connected to the body during prep?
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG = json.load(open(PROJECT_ROOT / "config" / "youtube_api_config.json"))


def get_client():
    key = CFG.get("_备用_google_key") or CFG.get("api_key")
    return genai.Client(api_key=key), "gemini-2.5-pro"


PROMPT = """你是顶级网球生物力学诊断师。这是一段网球教练或职业球员的演示视频，
非常短（4-10 秒）。我需要你**仅**回答以下问题，不要谈论其他事情。

## 用户的具体问题

用户是高级业余学员，他做 unit turn 时**右大臂会自动激活上抬**（三角肌前束代偿）。
他能做到的极简模式：只做 unit turn + 大臂完全被动 + 上臂跟胸贴近不动。
但他想知道：**职业球员/教练是怎么做到的？他们具体做了什么？**

## 你必须回答（按顺序）

### 1. 这个视频里展示的是谁、什么场景
（教练演示 / 球员实战 / 慢动作 / 镜子前自检 等）

### 2. Preparation / Unit Turn 阶段的精确观察
**只看从 ready 位到 forward swing 启动那一帧之间发生了什么。**

具体观察并报告：
a. **右大臂相对躯干的位移**：完全不动？跟着躯干转了 X°？独立上抬了 Y°？
b. **右肘的高度**：相对胸口/嘴的高度？整个 prep 阶段保持还是抬高？
c. **左手（非持拍手）做了什么**：扶拍喉？放胸前？指向某处？随身体转？
d. **躯干转动幅度**（肩转开角度）：估计度数
e. **髋部转动**：跟肩同步还是有时间差？
f. **拍头位置**：从 ready 到 prep 末端，拍头走了什么轨迹？
g. **整个 prep 用时**（如果能估计）

### 3. **关键问题**：他用了什么"机制"让大臂保持被动？
候选机制（你选最像的 1-2 个，或自己描述）：
- A. 左手物理扶拍 → 右手没法独立动
- B. 整个躯干作为刚体转动 → 右手"被带过去"
- C. 神经招募——他根本不发指令给三角肌
- D. 髋部主导 → 上身被动跟随
- E. 已经是肌肉记忆 → 默认不激活上臂
- F. 其他

### 4. **如果用户想模仿他**，你能给的最具体的 cue 是什么？

不要给"放松"、"不要紧张"这种废话——给可观察的体感入口。

## 输出格式

- 简洁，每个问题 1-2 句话
- 不要引言、不要总结
- 中文
- 看不清就写"看不清"
"""


def main():
    client, model = get_client()
    coach_dir = PROJECT_ROOT / "videos" / "coach"
    out_dir = PROJECT_ROOT / "output" / "coach_prep_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(coach_dir.glob("*.mp4"))
    print(f"Found {len(videos)} coach videos")

    for video_path in videos:
        out_path = out_dir / f"{video_path.stem}_prep_analysis.md"
        print(f"\n[upload] {video_path.name} ({video_path.stat().st_size / 1024:.0f} KB)")

        uploaded = client.files.upload(file=str(video_path))
        # Wait for processing
        for i in range(60):
            f = client.files.get(name=uploaded.name)
            state = getattr(f, "state", None)
            state_name = getattr(state, "name", str(state)) if state else "UNKNOWN"
            if state_name == "ACTIVE":
                break
            if state_name == "FAILED":
                print(f"  FAILED")
                break
            time.sleep(2)
        if state_name != "ACTIVE":
            continue

        print(f"[analyze]")
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[uploaded, PROMPT],
            )
            text = resp.text
            out_path.write_text(text, encoding="utf-8")
            print(f"  → {out_path.name} ({len(text)} chars)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n=== Summary ===")
    for video_path in videos:
        out_path = out_dir / f"{video_path.stem}_prep_analysis.md"
        if out_path.exists():
            print(f"✓ {out_path.name}")


if __name__ == "__main__":
    main()

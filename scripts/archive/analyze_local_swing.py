#!/usr/bin/env python3
"""Analyze a local swing video via Gemini File API (native google key)."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG = json.load(open(PROJECT_ROOT / "config" / "youtube_api_config.json"))


def get_client():
    """Use the native google key (备用_google_key) for File API upload."""
    key = CFG.get("_备用_google_key") or CFG.get("api_key")
    return genai.Client(api_key=key), "gemini-2.5-pro"


PROMPT = """你是顶级网球教练 + 生物力学诊断师。这是一段用户今天（2026-04-29）的实战训练录像，
约 25 秒，发球机喂球，半开放站姿，右手持拍。仔细看视频里的**每一拍**，注意先识别有几次挥拍。

## 用户当前认知坐标（必读，回答必须建立在这之上）

**圣经级顿悟（4/27）**：右脚为轴 = 一切。所有现代正手力学服务于让旋转轴落在右脚。

**11 字口令**：盯/左/架/推/锁/撑/流/撕/飘/藏/压

**用户今天的核心困惑（这是诊断主题，不要绕开）**：
1. 空挥（对镜子）时能感觉到**胸部挤压、胸部张力**，能做出"肘前推"，很像 Sinner
2. 实战击球时，按 RTP 的"右脚为轴"做了，确实有用，但**没有脱胎换骨的改变**
3. 用户已经自己怀疑：可能是**髋部没到位 → 没法真正旋转 → 时序错乱**
4. 用户也怀疑：可能有**过度旋转**（Over-rotation）—— 击球点应该在右前 45°，但他转到平行甚至左侧
5. 之前对话已确认：用户的"被身体甩出去"的感觉是对的，"肘往球的方向砸"是错的（那是 Shoulder Flexion 小肌群发力）

## 你必须回答的问题

为什么用户**空挥能做到胸部挤压 + 肘前推（Sinner 感）**，
但**实战做不到弯臂 + 肘先行 + 身体驱动**的 Sinner 式正手？

## 诊断框架（按这个顺序看）

### 1. 先看视频里有几次挥拍，分别在第几秒
列出每一拍的时间戳。

### 2. 从每一拍里挑 1-2 拍最具代表性的，做帧级拆解
每拍 6 阶段：
- Unit Turn / 架拍
- Drop（拍头下沉）
- Forward Swing 启动
- Contact（击球瞬间）
- Follow-through 早段
- Recovery

每个阶段观察：
- 右脚是否撑死（是不是真轴心，还是踮起来/拖动）
- 左手什么时候离开拍喉，离开后去哪了（颈侧锁定？还是垂下来？）
- 髋部领先肩部的角度（"X-factor"）
- 肘部相对躯干的位置（藏在右腰侧？还是已经飘出去了？）
- 肩部是否在击球点停住，还是漏到正左方

### 3. 关键诊断：空挥 vs 实战的差异在哪
基于你看到的实战画面，**直接指出**：
- 空挥那种"胸部挤压"的感觉，在实战哪几个阶段**消失了**？
- 是哪个具体动作环节让胸大肌的张力没建立起来？
- 是时序问题（动作顺序错）还是结构问题（某个关节位置错）还是发力问题（小肌群代偿大肌群）？

候选诊断方向（择优诊断，不要全部套用）：
- A. **轴心未真正落在右脚**：踮脚/拖动/重心横移，导致旋转无支点
- B. **左手没刹车**：颈锁松开太早，导致过度旋转 + 胸大肌张力提前泄掉
- C. **髋肩同步**（缺 X-factor）：髋肩一起转，没有蓄势期，胸前张力建立不起来
- D. **肘部"飘"**：架拍位肘没藏到右腰侧（没"藏"住），实战球速逼迫下肘部后摆走捷径，跳过了"胸前挤压"那一段
- E. **球反应被动**：来球决定动作，时间不够"先架后转"，被迫"边追边挥"
- F. **臂为先 / 身体为后**（典型 arming）：实战节奏紧，手臂抢先启动，身体跟在后面
- G. **过度旋转**：左侧没刹车，肩转过 45°一直漏到左侧，所有蓄势在过度转中泄完

### 4. 一句话根因
**空挥 vs 实战的本质差异**到底是什么？

### 5. 三个可执行修正
基于你看到的实际帧，给三个**今天/明天就能在球场上做的**修正。
每条：观察到什么 → 修正什么 → 验证标准。
不要给空泛建议，要绑定具体帧位/动作环节。

## 风格要求（严格）
- 中文，绝不 AI 腔
- 禁用"全方位/丰富/至关重要/凸显"等词
- 引用具体时间戳（如 "0:08-0:11 那一拍"）
- 教练讲人话，有温度
- 不要列万能清单，要敢站队 —— 多个候选诊断里挑最像的 1-2 个
- 不要开篇说"通过观察..."，直接进诊断
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_local_swing.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1]).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "output" / "local_swing_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_analysis.md"

    client, model = get_client()
    print(f"[upload] {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)")

    uploaded = client.files.upload(file=str(video_path))
    # Wait for processing
    while True:
        f = client.files.get(name=uploaded.name)
        state = getattr(f, "state", None)
        state_name = getattr(state, "name", str(state)) if state else "UNKNOWN"
        print(f"[file state] {state_name}")
        if state_name == "ACTIVE":
            break
        if state_name == "FAILED":
            print("File processing failed")
            sys.exit(1)
        time.sleep(2)

    print(f"[analyze] model={model}")
    resp = client.models.generate_content(
        model=model,
        contents=[uploaded, PROMPT],
    )
    text = resp.text
    out_path.write_text(text, encoding="utf-8")
    print(f"[done] → {out_path} ({len(text)} chars)")
    print()
    print("=" * 80)
    print(text)


if __name__ == "__main__":
    main()

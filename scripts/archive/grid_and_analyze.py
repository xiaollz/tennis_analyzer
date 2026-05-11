#!/usr/bin/env python3
"""Grid frames into batches, send each to Gemini via packyapi."""
from __future__ import annotations
import base64
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG = json.load(open(PROJECT_ROOT / "config" / "youtube_api_config.json"))


def make_client():
    primary_client = genai.Client(
        api_key=CFG["api_key"],
        http_options=types.HttpOptions(
            api_version="v1beta",
            base_url=CFG["base_url"],
            timeout=600000,
        ),
    )
    return primary_client, CFG.get("model", "gemini-3-flash-preview")


def make_grid(frame_paths, cols=5, label_fps=2.0, thumb_w=480):
    imgs = []
    for p in frame_paths:
        img = Image.open(p)
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * ratio)))
        imgs.append((thumb, p.stem))
    rows = (len(imgs) + cols - 1) // cols
    cell_w = thumb_w
    cell_h = imgs[0][0].height
    pad = 6
    label_h = 30
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * (cell_h + label_h) + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = ImageFont.load_default()
    for i, (thumb, stem) in enumerate(imgs):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        # parse frame number from stem like 'f_003' → 3 → time
        try:
            fn = int(stem.split("_")[1])
            t = (fn - 1) / label_fps
            label = f"#{fn}  t={t:.1f}s"
        except Exception:
            label = stem
        draw.text((x + 4, y), label, fill=(255, 255, 0), font=font)
        grid.paste(thumb, (x, y + label_h))
    return grid


PROMPT = """你是顶级网球教练 + 生物力学诊断师。这张大图是用户今天（2026-04-29）一段 25 秒训练视频
的连续抽帧（按 0.5 秒一帧抽出来的）。每个小图左上角的 "#N t=X.Xs" 是帧号 + 视频中的秒数。
半开放站姿、右手持拍、发球机喂球，相机在用户正前方。

## 用户当前认知坐标（必读）

**圣经级顿悟（4/27）**：右脚为轴 = 一切。

**11 字口令**：盯/左/架/推/锁/撑/流/撕/飘/藏/压

**用户今天的核心困惑（这是诊断主题）**：
1. 空挥（对镜子）能感觉到**胸部挤压、胸部张力**，做出"肘前推" Sinner 感
2. 实战时按 RTP 的"右脚为轴"做了，确实有用，但**没有脱胎换骨**
3. 用户自己怀疑：**髋部没到位 → 没法真正旋转 → 时序错乱**
4. 用户也怀疑：**过度旋转**（击球点应在右前 45°，他转到平行甚至左侧）
5. 已确认：用户感到的"被身体甩出去"是对的，"肘往球的方向砸"反而是错的

## 你必须回答的核心问题

**为什么用户空挥能做出 Sinner 式胸部挤压 + 肘前推，实战做不到？**

## 输出格式（严格按这个）

### 1. 视频里识别到几次挥拍
列出每拍的大致时间区间（几秒到几秒）。

### 2. 选最具代表性的 1-2 拍做帧级拆解
用帧号引用证据。每拍按以下阶段看：
- Unit Turn / 架拍：肩转到位了吗？左手在哪？
- Drop：拍头掉下来了吗？还是停在高位？
- Forward Swing 启动：身体先动还是手臂先动？
- Contact：右脚撑死还是踮起？击球点在右前 45° 还是平行/过头？
- Follow-through 早段：肩转到对网就停，还是漏到正左方？
- Recovery：左脚落点

### 3. 关键诊断（站队，不要罗列）
从下面候选里**挑最像的 1-2 个**，引用具体帧号证据：
- A. 轴心未真正落在右脚（踮脚/拖动/重心横移）
- B. 左手没刹车（颈锁松开太早 → 过度旋转 + 胸大肌张力提前泄掉）
- C. 髋肩同步（缺 X-factor → 没有蓄势期 → 胸前张力建立不起来）
- D. 肘部"飘"（架拍位肘没藏到右腰侧 → 实战球速逼迫下肘走捷径）
- E. 球反应被动（来球决定动作 → 时间不够"先架后转" → 边追边挥）
- F. 臂为先 / 身体为后（实战节奏紧 → 手臂抢先启动 → 身体跟在后面）
- G. 过度旋转（左侧没刹车 → 肩漏过 45° → 蓄势在过度转中泄完）

### 4. 一句话根因：空挥 vs 实战的本质差异

### 5. 三个可执行修正
基于看到的实际帧，给三个**今天/明天就能做的**修正。
每条：【看到什么】→【修正什么】→【验证标准】
不要空泛建议，要绑定具体帧位/动作环节。

## 严格要求
- 中文，绝不 AI 腔
- 禁用"全方位/丰富/至关重要/凸显"
- 引用具体帧号（如 "f_023 那一帧"）
- 教练讲人话，有温度
- 多个候选诊断里挑最像的 1-2 个站队，不和稀泥
- 直接进诊断，不要"通过观察..."的开场白
"""


def call_gemini(client, model, image_paths_list):
    """image_paths_list: list of grid PNG paths"""
    contents_parts = [types.Part(text=PROMPT)]
    for p in image_paths_list:
        with open(p, "rb") as f:
            data = f.read()
        contents_parts.append(types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=data,
            )
        ))
    contents = [types.Content(role="user", parts=contents_parts)]
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            if resp.text and resp.text.strip():
                return resp.text
            raise RuntimeError("empty response")
        except Exception as e:
            last_err = e
            print(f"[retry {attempt+1}] {str(e)[:200]}")
            time.sleep(3 * (attempt + 1))
    raise last_err


def main():
    frames_dir = PROJECT_ROOT / "output" / "local_swing_analysis" / "frames"
    out_dir = PROJECT_ROOT / "output" / "local_swing_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("f_*.jpg"))
    if not frame_paths:
        print("No frames found")
        sys.exit(1)
    print(f"Found {len(frame_paths)} frames")

    # Split into 2 batches of 25 frames each. 5 cols × 5 rows.
    batches = []
    n = len(frame_paths)
    half = (n + 1) // 2
    batches.append(frame_paths[:half])
    batches.append(frame_paths[half:])

    grid_paths = []
    for i, bf in enumerate(batches):
        g = make_grid(bf, cols=5, label_fps=2.0, thumb_w=380)
        gp = out_dir / f"grid_batch{i+1}.jpg"
        g.save(gp, quality=85)
        print(f"Saved {gp} ({g.size}, {gp.stat().st_size/1e6:.1f} MB)")
        grid_paths.append(gp)

    client, model = make_client()
    print(f"Calling {model}...")
    text = call_gemini(client, model, grid_paths)
    out_path = out_dir / "swing_diagnosis.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"\n=== Saved to {out_path} ===\n")
    print(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate a markdown study guide from a FeelTennis channel flat playlist dump.

Input:  feeltennis_flat.json (created via youtube-dl --flat-playlist -J ...)
Output: FeelTennis 学习路线（男子 单反）.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Video:
    title: str
    video_id: str

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def _categorize(title: str) -> str:
    # Heuristic single-category classifier. Priority order matters.
    t = title.lower()

    def has(*subs: str) -> bool:
        return any(s in t for s in subs)

    # Return of serve
    if has("return", "returns", "returning"):
        return "接发"

    # Net / transition
    if has("serve & volley", "serve and volley", "serve-and-volley"):
        return "网前/过渡"
    if has("volley", "overhead", "smash", "approach", "at the net", "net", "lob", "drop volley"):
        return "网前/过渡"

    # Serve
    if has("serve", "toss", "pronation", "kick serve", "kick"):
        return "发球"

    # Forehand
    if has("forehand", "inside-out", "inside out"):
        return "正手"

    # Backhand
    if has("backhand", "one-handed", "one handed", "1h", "two-handed", "two handed", "2h"):
        return "反手（单反/切削）"

    # Movement / footwork
    if has(
        "footwork",
        "split step",
        "movement",
        "move ",
        "moving",
        "faster",
        "speed",
        "stance",
        "balance",
        "pivot",
        "recovery",
        "warm up",
        "warming up",
        "warmup",
        "warm-up",
        "coordination",
        "legs",
    ):
        return "移动/脚步"

    # Strategy / match play
    if has(
        "strategy",
        "tactics",
        "pattern",
        "doubles",
        "singles",
        "baseline",
        "match",
        "stats",
        "where to aim",
        "decision",
        "winning",
        "play smarter",
    ):
        return "战术/比赛"

    # Mental / mindset
    if has(
        "mental",
        "confidence",
        "pressure",
        "calm",
        "mind",
        "nerves",
        "fear",
        "mindset",
        "goals",
        "motivate",
        "motivation",
        "competing",
        "choke",
        "anxiety",
    ):
        return "心理/思维"

    # Gear / equipment
    if has("grip", "racket", "racquet", "string", "tension", "ball machine", "review"):
        return "器材/握拍"

    # Practice / drills
    if has("drill", "practice", "lesson", "training", "masterclass"):
        return "训练/Drills"

    # General technique (misc biomechanics/feel concepts)
    if has(
        "technique",
        "swing",
        "contact",
        "timing",
        "spin",
        "topspin",
        "slice",
        "power",
        "control",
        "sweet spot",
        "heavy ball",
        "late",
        "hips",
        "rotation",
        "unit turn",
        "lag",
        "kinetic chain",
        "wrist",
        "arm",
    ):
        return "通用技术"

    return "理念/杂项"


def _escape_md_link_text(text: str) -> str:
    # Basic escaping to prevent titles like "[...]" from breaking markdown link syntax.
    return text.replace("[", r"\[").replace("]", r"\]")


def main() -> None:
    src = Path("feeltennis_flat.json")
    out = Path("FeelTennis 学习路线（男子 单反）.md")

    raw = json.loads(src.read_text(encoding="utf-8"))
    entries = raw.get("entries", [])

    videos: list[Video] = []
    for e in entries:
        title = (e.get("title") or "").strip()
        video_id = (e.get("id") or "").strip()
        if not title or not video_id:
            continue
        videos.append(Video(title=title, video_id=video_id))

    buckets: dict[str, list[Video]] = defaultdict(list)
    for v in videos:
        buckets[_categorize(v.title)].append(v)

    order = [
        "理念/杂项",
        "通用技术",
        "移动/脚步",
        "正手",
        "反手（单反/切削）",
        "发球",
        "接发",
        "网前/过渡",
        "战术/比赛",
        "心理/思维",
        "训练/Drills",
        "器材/握拍",
    ]
    for k in order:
        buckets.setdefault(k, [])

    cnt = Counter({k: len(v) for k, v in buckets.items()})

    lines: list[str] = []
    lines.append("# FeelTennis 学习路线（男子 / 单反）")
    lines.append("")
    lines.append(
        f"- 数据来源：`youtube-dl --flat-playlist -J https://www.youtube.com/@feeltennis/videos`（抓取日期：{date.today().isoformat()}）"
    )
    lines.append(f"- 本次抓取到公开视频数量：**{len(videos)}**")
    lines.append("")

    lines.append("## 这份文档怎么用（偏“科学训练”的逻辑）")
    lines.append("")
    lines.append("1. **先练“稳定产出”，再追求“更漂亮动作”**：击球点（时间/空间）+ 拍面控制 > 动作风格。")
    lines.append("2. **每次只改 1 个变量**：从视频里选 1 个最影响稳定性的点（例如更早 unit turn / 更靠前击球 / 更好的 split step）。")
    lines.append("3. **练习用“进阶阶梯”**：影子挥拍 → 自抛/喂球 → 对拉（合作）→ 半场对抗 → 全场对抗/比赛。")
    lines.append("4. **用外部目标驱动学习**：多用“目标/轨迹/落点/旋转/高度”来约束动作（比“手肘抬高一点”更稳定）。")
    lines.append("5. **录像 + 指标**：每个主题至少录 30 秒；只看 1 个指标（例如：触球是否在身体前方、出球弧线是否稳定、1st serve in%）。")
    lines.append("")

    lines.append("## 推荐学习顺序（适合：成人男性 / 单反）")
    lines.append("")
    lines.append("> 你可以把“本频道”当作一个大课程库：先把主干搭起来，再按需要深入某个分支。")
    lines.append("")
    lines.append("### 第 0 阶段：建立你的“问题优先级”")
    lines.append("")
    lines.append("- 先确定你最常输分的 2 个原因：**（A）被动失误**（晚、飘、下网）还是 **（B）被动挨打**（落点浅、没有压迫）。")
    lines.append("- 单反玩家常见优先级：**早转肩（unit turn）→ 间距（spacing）→ 击球点在身体前方 → 左手/非持拍手控制拍喉到合适时机 → 通过身体转动带动拍头**。")
    lines.append("")

    lines.append("### 第 1 阶段：移动/时间与空间（所有技术的底座）")
    lines.append("")
    lines.append("- 目标：你能更稳定地“到位”，让挥拍在可控的击球区完成。")
    lines.append("- 关键词：split step、第一步方向、调整步、重心稳定、恢复步。")
    lines.append("")

    lines.append("### 第 2 阶段：通用挥拍模型（正反手通用的“发动机”）")
    lines.append("")
    lines.append("- 目标：形成一致的加速模式（身体-手臂-拍头），减少“用手抡”。")
    lines.append("- 关键词：unit turn、圆周/弧线挥拍、甜区命中、击球前后拍面稳定、旋转与控制。")
    lines.append("")

    lines.append("### 第 3 阶段：主攻正手 + 稳住单反")
    lines.append("")
    lines.append("- 正手：优先解决 **击球点偏晚**、**加速路径**、**甜区命中**、**上旋与深度**。")
    lines.append("- 单反：优先解决 **间距**（离身体更远一点）+ **更靠前触球**；高球/快球先用“解决方案”（切削/提前/后撤）再谈美观动作。")
    lines.append("")

    lines.append("### 第 4 阶段：发球/接发（每局必发生的两拍）")
    lines.append("")
    lines.append("- 发球：先把 1 发稳定性做出来，再把 2 发（旋转/高度/安全区）做出来。")
    lines.append("- 接发：先把“判断 + split step + 短小挥拍”做出来，再追求攻击性。")
    lines.append("")

    lines.append("### 第 5 阶段：网前/过渡与战术")
    lines.append("")
    lines.append("- 把“上网/截击/过渡球”当作战术工具：对手回球质量下降时再上。")
    lines.append("- 建议每周至少 1 次：**半场对抗**或**带规则的对抗**（例如只能打斜线、只能打对手反手）。")
    lines.append("")

    lines.append("## 训练模板：把“看视频”变成“上场能力”")
    lines.append("")
    lines.append("把任意一个视频主题，按下面模板练 2–3 次，再换下一个主题。核心是：**少改动 + 多迁移**。")
    lines.append("")
    lines.append("### 45–60 分钟单主题训练（建议）")
    lines.append("")
    lines.append("1. **选 1 个指标**（只看一个）：例如“击球点是否在身体前方 20–40cm”。")
    lines.append("2. **影子挥拍 3–5 分钟**：慢 → 中速；每 5 次停一下检查击球区。")
    lines.append("3. **自抛/喂球 10–15 分钟**：给自己“时间与空间”，把动作做对。")
    lines.append("4. **合作对拉 10–15 分钟**：加入外部约束（只打斜线；必须过网 1m；必须落在后 1/3）。")
    lines.append("5. **半场对抗 10 分钟**：带规则的小比赛（例如：只能打对手反手；拿到短球才能上网）。")
    lines.append("6. **录像 30 秒 + 复盘 2 分钟**：只看你选的那个指标，写一句结论。")
    lines.append("")
    lines.append("### 常用外部约束（更“科学/可迁移”）")
    lines.append("")
    lines.append("- **落点**：深区/边线 1m 内；或固定打到对手反手。")
    lines.append("- **过网高度**：逼自己打出可控弧线（尤其单反、二发、上旋）。")
    lines.append("- **节奏**：规定每拍都要“准备早”，不允许大摆；用来治“来不及”。")
    lines.append("")

    lines.append("## 单反专项：优先把“困难球”处理好")
    lines.append("")
    lines.append("- **高球**：先用“安全方案”赢回中性（切削深区 / 提前截击点 / 后撤留空间），再追求更暴力的上旋。")
    lines.append("- **快球**：准备更早、挥拍更短、击球更靠前；宁可把球打深，也不要强求角度。")
    lines.append("- **跑动反手**：优先练脚步（交叉步+调整步）和击球区，而不是“手怎么摆”。")
    lines.append("")

    lines.append("## 视频索引（按主题归类，勾选式）")
    lines.append("")
    lines.append("说明：以下标题来自频道公开视频列表；每条都附上视频链接。每个分类内按“新→旧”排序。")
    lines.append("")

    for cat_name in order:
        vids = buckets.get(cat_name, [])
        lines.append("<details>")
        lines.append(f"<summary><strong>{cat_name}</strong>（{len(vids)}）</summary>")
        lines.append("")
        for v in vids:
            title = _escape_md_link_text(v.title)
            lines.append(f"- [ ] [{title}]({v.url})")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## （可选）更新抓取/重新生成")
    lines.append("")
    lines.append("如果你想把索引更新到最新：")
    lines.append("")
    lines.append("```bash")
    lines.append("youtube-dl --flat-playlist -J https://www.youtube.com/@feeltennis/videos > feeltennis_flat.json")
    lines.append("python generate_feeltennis_study_guide.py")
    lines.append("```")
    lines.append("")

    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print("wrote", out)
    print("videos", len(videos))
    print("category_counts", dict(cnt))


if __name__ == "__main__":
    main()

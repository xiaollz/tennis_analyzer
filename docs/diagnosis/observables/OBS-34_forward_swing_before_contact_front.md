# OBS-34 During the forward swing before contact, does the front knee move inward toward the other leg, or stay above the front foot?

- **定义**：During the forward swing before contact, does the front knee move inward toward the other leg, or stay above the front foot? 取值：moves_inward / stays_above_front_foot（看不清则记 unclear）。
- **看哪一帧**：题目里写明的时刻 / 阶段。
- **哪个机位**：B（正后方）。
- **谁来测**：B 层：需指定机位 + Pose / 人工抽帧；模型只作参考。
- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。

来源：Claude Code T08 评审提案（跨期 ≥2；涉及 3 期：T5meVwBOKR4, aPcKJDadzek, xLs469ZVMPU）。题目与取值逐字取自提案；升级见 `pipeline/promote_new_obs.py from_review`。不挂接任何故障、不含处方。

# OBS-35 At the start of the forward swing, how far does the racket head trail behind the hitting hand?

- **定义**：At the start of the forward swing, how far does the racket head trail behind the hitting hand? 取值：far_behind_hand / slightly_behind_hand / level_with_or_ahead_of_hand（看不清则记 unclear）。
- **看哪一帧**：题目里写明的时刻 / 阶段。
- **哪个机位**：S（正侧面）。
- **谁来测**：B 层：需指定机位 + Pose / 人工抽帧；模型只作参考。
- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。

来源：Claude Code T08 评审提案（跨期 ≥2；涉及 2 期：FxDmVi3EFnE, LoSrMj-Qxkc）。题目与取值逐字取自提案；升级见 `pipeline/promote_new_obs.py from_review`。不挂接任何故障、不含处方。

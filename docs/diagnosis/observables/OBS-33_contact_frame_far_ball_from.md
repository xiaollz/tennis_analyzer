# OBS-33 At the contact frame, how far is the ball from the hitting-side hip, side to side?

- **定义**：At the contact frame, how far is the ball from the hitting-side hip, side to side? 取值：within_about_one_forearm_length / more_than_one_forearm_length（看不清则记 unclear）。
- **看哪一帧**：题目里写明的时刻 / 阶段。
- **哪个机位**：B（正后方）。
- **谁来测**：B 层：需指定机位 + Pose / 人工抽帧；模型只作参考。
- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。

来源：Claude Code T08 评审提案（跨期 ≥2；涉及 4 期：2Nn9tyMECZU, 9NirSL59an0, PwkGLkRm73o, T5meVwBOKR4）。题目与取值逐字取自提案；升级见 `pipeline/promote_new_obs.py from_review`。不挂接任何故障、不含处方。

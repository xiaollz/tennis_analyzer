# OBS-36 Between the start of the forward swing and contact, does the hitting hand first move outward away from the body, or directly toward the ball?

- **定义**：Between the start of the forward swing and contact, does the hitting hand first move outward away from the body, or directly toward the ball? 取值：outward_first / directly_toward_ball（看不清则记 unclear）。
- **看哪一帧**：题目里写明的时刻 / 阶段。
- **哪个机位**：B（正后方）。
- **谁来测**：B 层：需指定机位 + Pose / 人工抽帧；模型只作参考。
- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。

来源：Claude Code T08 评审提案（跨期 ≥2；涉及 2 期：2Nn9tyMECZU, wzLMv4ASg9s）。题目与取值逐字取自提案；升级见 `pipeline/promote_new_obs.py from_review`。不挂接任何故障、不含处方。

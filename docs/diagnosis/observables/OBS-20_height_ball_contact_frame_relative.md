# OBS-20 Height of the ball at the contact frame relative to the player's body.

- **定义**：Height of the ball at the contact frame relative to the player's body. 取值：below_knee / knee_to_waist / waist_to_chest / above_chest（看不清则记 unclear）。
- **看哪一帧**：题目里写明的时刻 / 阶段。
- **哪个机位**：任意机位。
- **谁来测**：A 层：普通手机视频里模型可答，人工抽帧复核。
- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。

来源：由 NEW-OBS 候选升级（obs_binding 簇 NEW-OBS-02；涉及 2 期：PUN6qIIYU-4, Psidjei5BnI）。升级规则见 `pipeline/promote_new_obs.py`；题目与取值逐字取自候选原文。不挂接任何故障、不含处方。

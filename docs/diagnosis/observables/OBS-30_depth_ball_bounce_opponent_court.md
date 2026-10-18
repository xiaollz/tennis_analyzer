# OBS-30 Depth of ball bounce in the opponent court

- **定义**：Depth of ball bounce in the opponent court 取值：short_near_service_line / deep_near_baseline / past_baseline（看不清则记 unclear）。
- **看哪一帧**：题目里写明的时刻 / 阶段。
- **哪个机位**：B（正后方）。
- **谁来测**：A 层：普通手机视频里模型可答，人工抽帧复核。
- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。

来源：由 NEW-OBS 候选升级（obs_binding 簇 NEW-OBS-04；涉及 3 期：5-Ia7c4XzfU, LoSrMj-Qxkc, T5meVwBOKR4）。升级规则见 `pipeline/promote_new_obs.py`；题目与取值逐字取自候选原文。不挂接任何故障、不含处方。

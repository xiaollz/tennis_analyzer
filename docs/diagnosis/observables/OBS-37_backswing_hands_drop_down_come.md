# OBS-37 During the backswing, do the hands drop down and come back up in a loop before the forward swing, or stay at about the same height?

- **定义**：During the backswing, do the hands drop down and come back up in a loop before the forward swing, or stay at about the same height? 取值：drop_in_loop / stay_at_same_height（看不清则记 unclear）。
- **看哪一帧**：题目里写明的时刻 / 阶段。
- **哪个机位**：S（正侧面）。
- **谁来测**：B 层：需指定机位 + Pose / 人工抽帧；模型只作参考。
- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。

来源：Claude Code T08 评审提案（跨期 ≥2；涉及 2 期：GsHkML2mVEI, lwS7Xxqrrdw）。题目与取值逐字取自提案；升级见 `pipeline/promote_new_obs.py from_review`。不挂接任何故障、不含处方。

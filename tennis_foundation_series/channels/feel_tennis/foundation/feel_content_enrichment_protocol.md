# Feel Tennis：感觉导向内容补齐协议

本协议用于将当前由标题与官方播放列表形成的 **705 条公开视频索引**和 **755 个训练原子候选**，逐步升级为可供教练、教学系统、Agent 与视频分析使用的证据化知识单元。Feel Tennis 的特殊挑战是：同一个“feel”可能是意象、注意焦点、节奏体验或接触反馈；只有保留原始用语、条件和可见代理，才能避免误用。

## 1. 并行内容采集队列

| 优先级 | 队列 | 当前规模 | 采集目标 |
|---|---|---:|---|
| P0 | 官方 `Tennis Drills` 播放列表 | 12条 | 练习设置、步骤、反馈、成功标准、进阶与回退。 |
| P1 | 官方短效技巧播放列表 | 17条 | 原始 cue、动作阶段、作用对象、误用边界。 |
| P2 | 标题含 drill/exercise/progression/practice/training | 265个候选 | 形成可执行 drill 库。 |
| P3 | 放松控制主题 | 55个候选 | 分离“放松”“松散”“稳定”“抢球”等感觉与可见动作。 |
| P4 | 接触反馈主题 | 49个候选 | 提取甜点、声音、触觉、球位与拍面结果。 |
| P5 | 机制解释 | 169个候选 | 区分视频明示机制、画面推断和跨视频综合。 |
| P6 | 错误纠正 | 55个候选 | 构建错误信号→cue/drill→复测的闭环。 |
| P7 | 全量其余视频 | 705条索引 | 补充主题、重复识别与长尾知识。 |

一个视频可落在多个队列。队列优先级用于调度，不是对内容价值的绝对排序。

## 2. 采集输出合约

```json
{
  "video_id": "YouTube ID",
  "source_url": "https://www.youtube.com/watch?v=...",
  "atoms": [
    {
      "atom_type": "drill | cue | mechanism | error_correction | tactical_decision",
      "cue_subtype": "imagery | external_focus | relaxation_control | contact_feedback | null",
      "time_range_sec": [0, 0],
      "evidence_type": "video_explicit | visual_inference | synthesis",
      "problem": "可观察的球员问题",
      "conditions": {"shot_family": "...", "ball_situation": "...", "skill_level": "..."},
      "cue_text_or_imagery": "保留视频原始线索",
      "visible_action_proxy": "教练或VLM可观察的代理变量",
      "mechanism": "仅写视频明示或明确标为推断的解释",
      "drill_steps": ["起始", "执行", "反馈"],
      "success_criterion": "可观察或可计数标准",
      "misuse_or_boundary": "反例、禁忌或适用条件",
      "vlm_observables": ["关键点、球拍、球路、声音、位置"],
      "confidence": "high | medium | low"
    }
  ]
}
```

未在视频中表达的字段必须为空，不能以标题或普通网球常识填补。

## 3. 感觉线索专用质量检查

| 检查 | 合格标准 | 不合格示例 |
|---|---|---|
| 原始性 | 保存原词/原意象，并与时间码绑定 | 将“throw the racket”改写成普适关节指令。 |
| 条件化 | 至少指明技术族和动作阶段；有球况则记录 | 对所有正反手/发球无差别使用同一 cue。 |
| 外显代理 | 给出可见动作、球拍、球路或声音的复测代理 | 只说“感觉更好”。 |
| 误用边界 | 记录视频警告、反例或可能的错误理解 | 将“relax”理解为手臂完全松散。 |
| 练习闭环 | cue 若作为训练建议，必须有练习或复测方法 | 将一句 tip 直接视为完整课程。 |

## 4. 去重、冲突与合成

同一感觉可能在不同球况下有不同含义；同一视频也可能包含多个 cue。系统不应按词面自动合并，而是比较 `problem`、`conditions`、`action_proxy`、`expected_result` 和 `misuse_boundary`。可使用 `variation_of`、`requires`、`corrects`、`contraindicated_when`、`supports` 等关系链接原子。任何跨视频“第一性原理”都必须注明支持视频集合与适用范围。

## 5. 审阅与安全

涉及疼痛、关节、伤病、训练负荷或青少年发展的视频应进入人工专业审阅队列。VLM 的输出须先报告可见事实，再报告链接到知识树的候选假设；单机位视频不得用来诊断伤病、精确量化内部关节力矩或断言唯一技术因果。

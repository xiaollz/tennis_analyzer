# One Minute Tennis：训练原子内容补齐协议

本协议用于将当前的 935 条视频索引与 603 个候选训练原子，升级为可供教练、教学系统和视频分析系统直接使用的证据化知识单元。它专门适配短视频和“一个巧思/一个感觉”式的内容：**每条视频可产生多个原子，但任何原子不得脱离条件与证据单独复用。**

## 1. 并行采集单元

每个内容采集任务以一条视频为输入、以 0–N 个 `TrainingAtom` 为输出。任务之间可以独立并行，最终由 `video_id`、时间码、原子类型和条件合并。

| 输入优先级 | 队列 | 当前规模 | 目标 |
|---|---|---:|---|
| P0 | 官方 `Drills and Exercises` 播放列表 | 17 条 | 提取可执行练习、变式与成功标准。 |
| P1 | 标题明确包含 drill/exercise/practice/training | 46 条 | 建立高价值练习库。 |
| P2 | fix/mistake/myth/stop/trouble/wrong | 51 个错误纠正候选 | 构建错误—线索—复测映射。 |
| P3 | why/explained/kinetic/lag/pronation/swing-path | 358 个机制候选 | 提取解释，严格区分明示与推断。 |
| P4 | feel/hack/tip/secret/simple/key/easy | 124 个感觉线索候选 | 提取线索的目标阶段与误用边界。 |
| P5 | 全量其余视频 | 935 条总索引 | 补充主题与重复内容去重。 |

同一视频可能落入多个队列。优先级用于确定先后顺序，而不是相互排除。

## 2. 内容采集输出合约

```json
{
  "video_id": "YouTube ID",
  "source_url": "https://www.youtube.com/watch?v=...",
  "atoms": [
    {
      "atom_type": "drill | cue | mechanism | error_correction | tactical_decision",
      "time_range_sec": [0, 0],
      "evidence_type": "video_explicit | visual_inference | synthesis",
      "training_problem": "球员要解决的可观察问题",
      "conditions": {
        "shot_family": "serve / forehand / ...",
        "skill_level": "if stated",
        "ball_situation": "if stated",
        "constraint": "if stated"
      },
      "action_or_feel": "仅记录视频实际给出的线索",
      "mechanism": "仅记录视频实际给出的解释；否则为空",
      "drill_steps": ["起始", "动作", "反馈"],
      "success_criterion": "可观察、可计数或可比较",
      "common_misuse": "视频明确警告或可见的误用",
      "vlm_observables": ["关键点/球拍/球路/声音/场地位置"],
      "confidence": "high | medium | low"
    }
  ]
}
```

若视频未表达某个字段，应写 `null` 或空数组，不能用标题或一般网球常识补全。

## 3. 质量门槛

| 检查 | 通过标准 | 拒绝/回退条件 |
|---|---|---|
| 来源 | 每个原子绑定视频 ID 和时间范围 | 只依据标题生成的主张 |
| 原子性 | 一个原子只解决一个主要问题/线索/练习 | 把整条视频压成模糊长摘要 |
| 条件化 | 至少写明技术族；有球况/水平则一并记录 | 将线索写成无条件通用规则 |
| 证据分层 | 明示、视觉推断和跨视频综合分开 | 视觉观察伪装为视频口述或因果事实 |
| 可训练性 | drill 有起始、动作与反馈；cue 有成功/误用边界 | 只给口号，无练习或验证方式 |
| VLM适配 | 只描述可见事件，并说明机位需求 | 从单机位推断精确内部生物力学或伤病 |

## 4. 去重与组合策略

大量短视频会出现同一概念的不同标题、同一 drill 的不同变式或看似矛盾的感觉线索。系统应保留原子级证据，然后根据 `problem`、`conditions`、`action_or_feel`、`mechanism` 与 `success_criterion` 建立关系：`same_as`、`variation_of`、`requires`、`contraindicated_when`、`corrects` 与 `supports`。不得因为标题相似就自动合并。

## 5. 人工审阅重点

涉及疼痛、肘/肩伤病、青少年负荷、训练量或医学机制的原子必须进入 `medical_or_load_review` 队列。涉及发球、手腕、旋前和高速度挥拍的机制性结论，建议结合多机位视频和合格教练审阅后再升级为教学处方。

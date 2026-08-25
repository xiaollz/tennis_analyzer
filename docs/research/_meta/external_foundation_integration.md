# Tennis Foundation Series 接入规则

`tennis_foundation_series/` 作为 VLM 诊断系统的外部补充知识源，不替代项目现有 canonical 知识图谱。

## 数据层级

1. 频道级 curated foundation node：优先使用，保留 confidence、limitations 和视频证据。
2. `video_content_analysis` 的逐视频 keypoint / observable：用于补充具体观察、机制和复测线索。
3. `metadata_title_only`、`metadata_title_and_playlist_only`：只用于资源发现，不进入诊断或训练处方。

当前包中，FTT 和 Tom Allsopp / TPA 有逐视频内容分析；Feel Tennis、One Minute Tennis、Road to Pro 的统一层主要是标题或播放列表候选，因此暂不作为已验证教学证据。

## 系统边界

- 先记录画面事实，再做机制推断；普通视频不能直接证明肌肉激活、握力、疼痛或关节力矩。
- 外部知识与 Tennis Science、项目 FTT 主图谱、ESR/HSA 优先级、Intuition-First 或用户专项规则冲突时，项目规则优先。
- 禁止把肘部向前的可见结果改写为主动“推肘/送肘” cue。
- 每条注入内容必须保留频道、视频链接、证据状态、置信度和限制。
- 提示词只检索当前症状相关的少量条目，不整库注入。

## 接入点

- `knowledge/external_foundation.py`：读取、过滤、分类、检索和来源保留。
- `knowledge/output/vlm_prompt.py`：向定向观察和深度诊断提示注入相关条目。
- `evaluation/diagnosis_engine.py`：在最终结果的 `external_foundation` 字段中附加补充证据。

# Foundation Model / App 接入指南

## 最小加载集

1. `unified/series_manifest.json`：包范围、统计和加载顺序。
2. `unified/channel_registry.json`：各频道的证据等级与保留入口。
3. `unified/canonical_domain_crosswalk.json`：领域标签归一化。
4. `unified/all_videos.jsonl`：视频身份、频道、URL 和证据状态。
5. `unified/all_knowledge_items.jsonl`：逐视频关键点、VLM 观察和元数据候选。

`global_video_id` 和 `knowledge_item_id` 是主键。先按 `domains_canonical` 检索，再按 `evidence_status` 过滤。

## 诊断准入

- `video_content_analysis`：可以进入技术检索，但必须保留来源与边界。
- `metadata_title_only`、`metadata_title_and_playlist_only`：只能用于资源发现，不能生成技术结论。
- `vlm_observable`：只能描述像素中可见的时序、位置、轨迹和结果。
- 频道整理与项目规则冲突时，以项目 canonical 规则为准。

当前运行时只使用 FTT 和 TPA 的深度内容。Feel Tennis、One Minute Tennis、Road to Pro Tennis 的保留文档用于未来内容补齐，不参与当前诊断处方。

## 证据回溯

FTT/TPA 深度记录的 `raw_analysis_path` 指向：

```text
channels/<channel_id>/raw/video_analyses/<video_id>.md
```

先从统一 JSONL 找到候选条目，再回读该 Markdown 核查上下文。统一索引是检索入口，原始视频分析是最高保真的本地证据。

## 输出约束

- 保留频道、视频标题、视频 URL、证据状态和适用条件。
- 相似观点使用 `supports`、`varies_by_condition` 或 `potential_conflict`，不要自动合并成唯一真理。
- 单机位视频不能断言精确肌肉激活、关节负荷、疼痛病因或唯一动力学因果。
- 教学输出采用“一个变量、一个练习、一个复测标准”。

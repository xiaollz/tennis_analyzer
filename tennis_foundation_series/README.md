# 网球频道基础知识库

这是面向网球教学与 VLM 诊断的外部知识包。目录已精简为三层：统一检索数据、频道知识框架、可回溯的深度视频分析。

## 证据范围

| 频道 | 当前证据 | 诊断用途 |
|---|---|---|
| Fault Tolerant Tennis | 93 条视频内容分析 | 可用于检索技术原理、练习和可见动作证据 |
| TPA tennis / Tom Allsopp | 124 条视频内容分析，304 条仅元数据 | 仅深度分析部分可进入诊断 |
| Feel Tennis | 标题/播放列表元数据 | 只作为未来补齐框架，不生成教学结论 |
| One Minute Tennis | 标题/播放列表元数据 | 只作为未来补齐框架，不生成教学结论 |
| Road to Pro Tennis | 标题元数据 | 只作为未来补齐框架，不生成教学结论 |

`evidence_status` 是强制边界。只有 `video_content_analysis` 或后续经人工审阅的内容可以影响技术诊断；元数据候选只能用于资源发现。

## 目录

```text
tennis_foundation_series/
├── README.md
├── docs/
│   └── FOUNDATION_MODEL_INGESTION_GUIDE.md
├── unified/
│   ├── series_manifest.json
│   ├── channel_registry.json
│   ├── universal_series_schema.json
│   ├── canonical_domain_crosswalk.json
│   ├── cross_channel_knowledge_network.json
│   ├── all_videos.jsonl
│   └── all_knowledge_items.jsonl
└── channels/
    ├── fault_tolerant_tennis/
    │   ├── README.md
    │   ├── foundation/
    │   └── raw/video_analyses/
    ├── tom_allsopp_tennis/
    │   ├── README.md
    │   ├── foundation/
    │   └── raw/video_analyses/
    ├── feel_tennis/
    ├── one_minute_tennis/
    └── road_to_pro_tennis/
```

## 加载顺序

1. 读取 `unified/series_manifest.json` 和 `unified/channel_registry.json`。
2. 用 `canonical_domain_crosswalk.json` 归一化领域。
3. 从 `all_videos.jsonl` 和 `all_knowledge_items.jsonl` 检索候选证据。
4. 对 FTT/TPA 的深度证据，按 `raw_analysis_path` 回读原始 Markdown。
5. 频道知识树和种子图谱用于解释与交叉验证，不能覆盖项目自身的 canonical 规则。

## 维护原则

- 不保存代理日志、抓取清单、构建脚本、重复 CSV/Markdown 目录或生成架构图。
- 不把标题推断写成视频明确主张。
- 不跨频道混淆来源；引用时保留频道、视频 ID、URL 和证据状态。
- VLM 只报告可见事实；内部发力、疼痛和唯一因果必须标为假设或交由专业人士判断。

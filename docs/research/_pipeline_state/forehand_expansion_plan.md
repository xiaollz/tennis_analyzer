# 正手教练知识库扩展 - 流水线状态

> 启动: 2026-04-29
> 用户授权: 自动执行，10 小时预算，目标"网上无法复制的正手框架"

## Pipeline 阶段

### Phase 1: 频道发现 (in flight)
- 1 个 agent 调研 r/10s + 论坛
- 输出: docs/research/_meta/channel_discovery.md
- 目标: 5-7 个 P0 频道，每个独特角度

### Phase 2: 频道扫描 (待 P1 完成)
- Python 脚本 yt-dlp 扫每个频道
- 用 filter_tpa_videos.py 类似的脚本过滤正手相关
- 输出: 每个频道的 VIDEOS_TO_ANALYZE.json

### Phase 3: 视频深度分析 (并行 agents)
- 每个频道挑 top 10-15 视频
- 5-7 频道 × ~12 视频 = 60-100 视频
- 分批派 agents 跑，每 agent 5-8 视频
- 共 12-20 agents 并行

### Phase 4: 频道级综合 (并行)
- 每个频道一个 synthesizer agent
- 输出: 每个 docs/research/{slug}_video_analyses/SUMMARY.md

### Phase 5: 主综合 (单 agent)
- 读所有 channel SUMMARY + 现有 FTT/RTP/TPA + 三方对比
- 产出: docs/research/_meta/MASTER_FOREHAND_FRAMEWORK.md
- N 方对比表（替代现有 FTT_RTP_TPA_INTEGRATION）

### Phase 6: 系统更新 (审核后实施)
- taxonomy 新子维度
- 新诊断链候选
- mantra 候选
- OBSERVATION_TO_CONCEPT 候选
- VLM prompt 优化

## 当前阻塞

- 等 Phase 1 channel_discovery.md 输出

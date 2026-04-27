# Road to Pro Tennis 频道知识库扩充任务

> **Goal**：穷尽该频道所有正手相关视频 → Gemini 深度分析 → 与 FTT 知识库结合
> **Channel**：https://www.youtube.com/@RoadtoPro10is
> **Started**：2026-04-27

## Phases

### Phase 1: 视频发现（discovery）
- [ ] 用 yt-dlp 拉全频道视频列表（id + title + description）
- [ ] 过滤"正手相关"——关键词不限于 forehand：
  - forehand / fh / 正手
  - stance / spacing / contact / hitting zone
  - axis / pivot / rotation / kinetic chain
  - load / weight transfer / hip / shoulder turn
  - timing / unit turn / take back
  - 击球感觉 / 发力 / 节奏
- [ ] 已分析过的不重复：`aiwUqHQl-Ec`（"Spacing fix"）
- [ ] 输出 `_VIDEOS_TO_ANALYZE.json`：`[{video_id, title, url, why_relevant, priority}]`

### Phase 2: 并行深度分析（parallel agents）
每个 agent 拿 2-3 个视频：
- 用现有 Gemini script 模板（`scripts/analyze_footwork_contact.py` 风格）
- 输出 `docs/research/road_to_pro_video_analyses/{video_id}.md`
- 结构（沿用 `aiwUqHQl-Ec.md` 的模板）：
  1. 视频元信息（讲师、时长、核心命题）
  2. 视频章节拆解
  3. 关键概念清单（含 FTT 对应关系）
  4. 击球点细节
  5. 脚步细节
  6. 训练方法/drills
  7. 与 FTT 的对接（一致 / 互补 / 冲突）
  8. 给用户的可执行建议
  9. 价值评级（⭐ 1-5）
  10. 一句话总结

### Phase 3: 知识合成（synthesis）
- [ ] 单个 agent 读所有 Phase 2 产物 → `docs/research/road_to_pro_SYNTHESIS.md`
  - 跨视频共识 vs 矛盾
  - 与 FTT 的互补 / 冲突点
  - 新概念清单（FTT 没讲过的）
  - 推荐给用户的优先级队列
- [ ] 更新 `docs/research/diagnostic_chains/` 如果发现新链
- [ ] 更新 `evaluation/diagnosis_engine.py` 的 `OBSERVATION_TO_CONCEPT` 如果有新触发词

## 输出锚点（不丢的中间产物）

```
docs/research/road_to_pro_video_analyses/
  ├── _TASK_PLAN.md            ← 这个文件
  ├── _VIDEOS_TO_ANALYZE.json  ← Phase 1 产出
  ├── aiwUqHQl-Ec.md          ← 已有
  └── {videoId}.md             ← Phase 2 产出（每个视频一份）

docs/research/road_to_pro_SYNTHESIS.md  ← Phase 3 产出
```

## 当前状态

- [x] Phase 0: 任务计划写完
- [ ] Phase 1: 视频发现进行中
- [ ] Phase 2: 待启动
- [ ] Phase 3: 待启动

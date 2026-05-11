# scripts/archive — 历史一次性脚本归档

> **归档日期**：2026-05-11
> **目的**：项目 3 个月演化产生 30 个 scripts，其中 26 个是"一次性 batch 分析"脚本——已产出 KB 文档（在 `docs/research/`）后不再使用。

## 归档分类

### FTT 视频批量分析（4/3）
- `analyze_ftt_batch1.py` (510 行)
- `analyze_ftt_batch2.py` (263 行)
- `analyze_ftt_batch3.py` (325 行)
- `merge_ftt_video_concepts.py` (905 行)
- `extract_existing_ftt_videos.py` (449 行)

→ 产出：`docs/research/09_ftt_videos_*.md` + `12_ftt_videos_synthesis.md`

### TPA / Tom Allsopp 批量（4/3 - 4/28）
- `run_tomallsopp_batch.py` (163 行)
- `run_tomallsopp_extract.py` (124 行)
- `analyze_tpa_video.py` (287 行)
- `filter_tpa_videos.py` (221 行)

→ 产出：`docs/research/14_tpa_videos_*.md` + `15_tpa_synthesis.md`

### Feel Tennis 批量（4/3）
- `batch_feeltennis.py` (230 行)
- `watch_feel_tennis_videos.py` (127 行)

→ 产出：`docs/research/feel_tennis_video_analyses/`

### JUL Tennis & Golf 批量（5/7）
- `analyze_jul_tennis.py` (266 行)
- `analyze_jul_videos.py` (209 行)
- `analyze_jul_federer_series.py` (267 行)
- `analyze_jul_rubber_arm.py` (269 行)

→ 产出：`docs/research/jul_tennis_videos/`

### Channel 扫描 / 视频比较（4/26-29）
- `scrape_and_filter_channel.py` (238 行)
- `analyze_channel_video.py` (302 行)
- `analyze_coach_prep_videos.py` (117 行)
- `compare_split_screen.py` (418 行)
- `contact_point_compare.py` (563 行)

### 早期实验（4/3-4/27）
- `analyze_local_swing.py` (142 行)
- `analyze_rtp_video.py` (259 行)
- `watch_coach_videos.py` (116 行)
- `grid_and_analyze.py` (191 行)
- `reextract_from_markdown.py` (206 行)
- `run_reconciliation.py` (165 行)

---

## 保留在 scripts/ 根的 active utility

| 文件 | 用途 |
|---|---|
| `generate_pwa_icons.py` | PWA 图标生成（可能复用）|
| `retrofit_foundation_check.py` | Foundation 检查集成 |
| `smoke_test_app.py` | 应用 smoke test |
| `test_foundation_report.py` | Foundation 报告测试 |
| `epub_pipeline/` | EPUB 处理管线（子目录）|

---

## 恢复脚本

如果以后某个归档脚本需要复用：
```bash
git mv scripts/archive/{script}.py scripts/
```

git history 完整保留——这次只是 mv，不是 delete。

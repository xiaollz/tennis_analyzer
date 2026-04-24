# Baseline — 网球挥拍分析 App

> 音频切段 + 逐片段诊断 + 简约 Anthropic 风格前端。
> 2026-04-25 首版完成。

---

## 一、运行方式（快速启动）

```bash
# 1. 确保依赖已装
pip install fastapi uvicorn python-multipart

# 2. 启动 API + 前端（同一进程）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

# 3. 浏览器打开
#    http://127.0.0.1:8765/           — 功能原型（可实际跑完整流程）
#    http://127.0.0.1:8765/design.html — Claude Desktop 出的设计稿 canvas
#    http://127.0.0.1:8765/docs        — FastAPI 自动生成的 API 文档
```

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────┐
│  Browser (frontend/dist/index.html, React via CDN)       │
│  Home → Upload → ClipGrid → Analyzing → Results (3 tabs) │
└───────────────────────┬──────────────────────────────────┘
                        │ HTTP JSON + multipart
                        ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI (app/)                                          │
│    routes.py  — HTTP endpoints                           │
│    storage.py — filesystem layout + JSON persistence     │
│    jobs.py    — in-process background job queue          │
│    services.py— wraps segmenter + diagnosis pipeline     │
└───────────────────────┬──────────────────────────────────┘
                        │
            ┌───────────┴────────────┐
            ▼                        ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  segmentation/        │  │  main.py :: TennisAnalysisPipeline │
│  SoundBasedSegmenter  │  │   → core/pose_estimator        │
│   • AudioOnsetDetector│  │   → evaluation/forehand_evaluator│
│   • ffmpeg clip export│  │   → evaluation/vlm_analyzer     │
│   • opencv thumbs     │  │   → evaluation/diagnosis_engine │
└──────────────────────┘  └───────────────────────────────┘
                        │
                        ▼
                  ┌───────────┐
                  │ storage/  │
                  │ videos/   │
                  │ diagnoses/│
                  │ jobs/     │
                  └───────────┘
```

---

## 三、HTTP API 契约（给前端用）

所有接口前缀 `/api`。响应统一 JSON（除视频/图片流）。CORS 全开。

### 健康检查

```
GET /api/health
→ {"status":"ok","service":"baseline","ts":...}
```

### 视频管理

#### 上传视频（触发分段）

```
POST /api/videos
Content-Type: multipart/form-data
Body: file=<video.mp4 | .mov | .m4v | .mkv>

→ 200
{
  "video_id": "abc123...",       # 12-char hex id
  "job_id":   "xyz789..."         # 分段作业 id
}
```

#### 列出所有视频

```
GET /api/videos

→ 200
{
  "videos": [
    {
      "video_id": "...",
      "filename": "Rally_0424.mov",
      "uploaded_at": 1777045000.0,
      "duration_s": 8.9,
      "fps": 30.0,
      "total_onsets": 5,
      "clip_count": 3,
      "has_segments": true
    },
    ...
  ]
}
```

#### 视频详情 + 片段清单

```
GET /api/videos/{video_id}

→ 200
{
  "meta":   {video_id, filename, duration_s, fps, ...},
  "clips":  [{clip_id, start_s, end_s, duration_s, impact_times_s[], onset_strength, thumbnail_path, clip_path}, ...],
  "total_onsets": 5,
  "segmented": true,
  "error": null
}
```

#### 删除视频（连带其所有 clip 和 diagnosis）

```
DELETE /api/videos/{video_id}
→ {"deleted": "abc123"}
```

### 作业状态（用于进度条）

```
GET /api/jobs/{job_id}

→ 200
{
  "job_id": "...",
  "kind": "segment" | "diagnose",
  "status": "queued" | "running" | "done" | "error",
  "progress": 0.0 - 1.0,
  "message": "导出片段 2/3",
  "error": null,                   # 失败时有错误文本
  "result": {...} | null,          # 成功时的摘要
  "payload": {...}
}
```

**前端使用方式**：轮询 `GET /api/jobs/{id}`，每 800-1200ms 一次，直到 `status ∈ {done, error}`。

### Clip 管理

#### 单个 clip 信息

```
GET /api/clips/{clip_id}

→ 200
{
  "clip_id": "...",
  "video_id": "...",
  "clip": {start_s, end_s, impact_times_s, ...},
  "has_diagnosis": true | false,
  "urls": {
    "video":     "/api/clips/{clip_id}/video",
    "thumbnail": "/api/clips/{clip_id}/thumbnail",
    "diagnose":  "/api/clips/{clip_id}/diagnose",      # POST 以启动
    "diagnosis": "/api/diagnoses/{clip_id}"
  }
}
```

#### 片段视频 / 缩略图

```
GET /api/clips/{clip_id}/video      → video/mp4
GET /api/clips/{clip_id}/thumbnail  → image/jpeg
```

#### 启动诊断（异步）

```
POST /api/clips/{clip_id}/diagnose

→ 200
{
  "clip_id": "...",
  "job_id":  "...",
  "status":  "queued" | "already_done"
}
```

**幂等**：已诊断过的 clip 再次 POST 返回 `already_done`，不重跑。

#### 诊断产物（异步生成后可用）

```
GET /api/clips/{clip_id}/annotated   → video/mp4  （骨骼叠加）
GET /api/clips/{clip_id}/keyframes   → image/png  （6 帧 VLM 输入）
GET /api/clips/{clip_id}/report.md   → text/markdown
```

### 诊断结果

```
GET /api/diagnoses/{clip_id}

→ 200 (完成时)
{
  "clip_id": "...",
  "status": "done",
  "ready": true,
  "result": {
    "clip_id": "...",
    "video_id": "...",
    "clip": {...},
    "stroke_type": "forehand" | "one_handed_backhand",
    "overall_score": 72.5,
    "swings": [
      {
        "swing_index": 0,
        "impact_frame": 30,
        "overall_score": 72.5,
        "arm_style": "Double-bend",
        "phase_scores": {
          "unit_turn": {"phase": "unit_turn", "score": 68.0, "kpis": [...]},
          "chain":     {...},
          "contact":   {...},
          "through":   {...},
          "stability": {...}
        },
        "kpi_results": [
          {"name": "shoulder_rotation", "label": "肩转幅度",
           "value": 95.3, "unit": "°", "score": 82, "level": "良好",
           "description": "...", "phase": "unit_turn"},
          ...
        ],
        "raw_metrics": {...}
      }
    ],
    "vlm": {
      "observations": [{phase, severity, observation, evidence}, ...],
      "issues": [...],
      "raw_answers": {...}
    },
    "diagnosis": {
      "root_cause":      {label, description, concept},
      "causal_chain":    [{label, concept}, ...],
      "drill_recommendation": {name, description, cue},
      "narrative":       "...",
      "contradictions":  [...],
      "arbitration":     [{resolution_text}, ...],
      "issues":          [...]
    },
    "artifacts": {
      "clip_video":      "/api/clips/{clip_id}/video",
      "thumbnail":       "/api/clips/{clip_id}/thumbnail",
      "annotated_video": "/api/clips/{clip_id}/annotated" | null,
      "keyframe_grid":   "/api/clips/{clip_id}/keyframes" | null,
      "report_md":       "/api/clips/{clip_id}/report.md" | null
    }
  }
}

→ 404
{"detail": "diagnosis not found (never started)"}
```

**重要字段的 null 约定**：
- 如果 `stroke_type == "one_handed_backhand"` → 目前 **VLM 跳过**（现有代码限制）：`vlm` 和 `diagnosis.root_cause` 会是 null。前端应能优雅降级只显示 biomechanics tab。
- 如果 VLM 无 API 配额 / 网络失败 → 同样 `vlm` 为 null，但 biomechanics 仍然可用。

---

## 四、前端状态机（functional prototype）

```
Home ─── pick existing ─→ ClipGrid
  │                           │
  └── upload new ─→ Uploading │
                      │       │
                      └───────┴── pick clip ─→ Analyzing
                                                  │
                                                  └── done ─→ Results (3 tabs)
                                                                │
                                                                └── back ─→ ClipGrid
```

每步都有 `onBack` 退回的能力。

---

## 五、存储布局

```
storage/
├── videos/{video_id}/
│   ├── meta.json              # 上传元信息 + 分段统计
│   ├── original.mp4           # 用户上传的原视频
│   ├── manifest.json          # 分段结果（clips 清单）
│   └── clips/
│       ├── {clip_id}.mp4      # 每个片段 (libx264 + aac)
│       └── {clip_id}.jpg      # 缩略图（取击球时刻帧，长边 720）
│
├── diagnoses/{clip_id}/
│   ├── status.json            # 状态快照
│   ├── result.json            # 前端可直接吃的 JSON
│   ├── annotated.mp4          # 骨骼叠加视频
│   ├── keyframe_grid.png      # 6 帧网格（VLM 输入）
│   └── report.md              # 长格式 Markdown 报告
│
└── jobs/
    └── {job_id}.json          # 作业状态持久化
```

整个 `storage/` 目录 gitignored，用户数据不入库。

---

## 六、已知约束 & 未来扩展

### 当前约束（设计上）
- **单进程作业队列**：`ThreadPoolExecutor(max_workers=2)`。重启丢失未完成作业。
  适合本地单用户，**不适合多用户并发**。要上生产需换 Celery / Dramatiq。
- **VLM 仅正手**：`main.py::_run_vlm_analysis` 里判断了 `is_backhand`，跳过。
  单反 clip 的 `vlm` 和 `diagnosis` 会是 null。
- **stroke classifier 分类偏差**：短 clip 可能被误分类。如需手动覆盖，可加 `?stroke=forehand` 参数（未实现）。

### 未来扩展（加在哪里很清楚）
- **音频 onset 阈值 UI 调节**：前端给个 slider，传 `threshold_std` 参数给 `/api/videos` 或重分段接口。
- **选多个 clip 一起分析**：批量 POST `/api/clips/{id}/diagnose`，前端合并 job。
- **训练趋势/对比**：`GET /api/videos` 已返回历史，前端做折线即可。
- **视频自动旋转**：管道内已有，暴露为单独接口即可给 UI 预览。

---

## 七、和 Claude Desktop 设计稿的关系

- **`frontend/dist/design.html`** = Claude Desktop 出的全套设计稿（9 屏的 figma-style canvas）。原样保留作为设计基准。
- **`frontend/dist/index.html`** = 我基于这套设计稿实现的**功能版**，用了相同的：
  - 调色板（clay #C8553D / amber #E8B04B / court #9AAE60 / ink #2A2925 / paper #F7F3EC）
  - 字体（Fraunces + Inter + JetBrains Mono）
  - 组件语言（Serif / Sans / Mono / Tag / CTA / AppShell）
  - 信息层次（eyebrow + serif 大标题 + 正文 + Mono 数值）
- 两边差异：设计稿是**静态浏览**（预设数据），功能版**接真接口**（上传/分段/诊断/结果）。

**要进一步打磨 UI**：改 `frontend/dist/index.html` 里的 React 组件即可，无构建步骤。

---

## 八、一键验证（端到端）

```bash
# 终端 A
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

# 终端 B
curl -X POST http://127.0.0.1:8765/api/videos -F "file=@videos/31da695364a2a80e48bb66b6e3ee4827.mp4"
# → {"video_id":"...","job_id":"..."}

# 浏览器
open http://127.0.0.1:8765/
```

**上次测试耗时参考**（3 秒 clip，MacBook M 系列）：
- 上传 + 分段：~0.5s
- 单 clip 诊断（含 YOLO pose + KPI + VLM + 诊断引擎）：~15s
- VLM + 知识推理本身是大头，pose 很快

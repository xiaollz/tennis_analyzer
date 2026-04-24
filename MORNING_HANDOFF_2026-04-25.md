# 早晨交接 · 2026-04-25

> 昨晚你布置的任务：把整套流程封装成 App。逐步汇报完成状态。

---

## ✅ 已交付（可以直接跑）

### 一条命令启动
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

然后浏览器打开 **http://127.0.0.1:8765/** —— 就是功能版 App。

### 端到端流程验证通过

1. ✅ **上传视频** → `POST /api/videos`，返回 video_id + job_id
2. ✅ **音频切段** → 按击球声把视频切成 2-3 秒的片段，每段含 1-2 次击球
3. ✅ **网格展示** → 读 `GET /api/videos/{id}`，每个 clip 有独立缩略图
4. ✅ **用户选一个 clip** → 点击触发 `POST /api/clips/{id}/diagnose`
5. ✅ **诊断**（约 15s）：YOLO pose → KPI → VLM → 网状诊断引擎 → 根因 + drill
6. ✅ **结果三 Tab 展示**：
   - **Biomechanics**：关键数值 + 按阶段评分
   - **Observations**：Gemini VLM 观察 + 时间戳 + 证据
   - **Coach**：根因 + 因果链 + 推荐 drill

### 实测数据（8.9 秒的练习视频）
- 上传 + 分段：**0.5 秒**
- 单 clip 诊断：**15 秒**
- 产出：标注视频、关键帧网格、Markdown 报告、前端可吃的 JSON

---

## 📂 新增代码

```
tennis/
├── app/                        NEW — FastAPI 后端
│   ├── __init__.py
│   ├── main.py                 # FastAPI 启动 + 静态前端挂载
│   ├── routes.py               # 所有 /api/* 端点
│   ├── storage.py              # 存储布局 + JSON 持久化
│   ├── jobs.py                 # 后台作业队列
│   └── services.py             # 段 + 诊断的服务包装
│
├── segmentation/               NEW — 音频驱动分段
│   ├── __init__.py
│   └── segmenter.py            # SoundBasedSegmenter 类
│
├── frontend/dist/              NEW — 前端
│   ├── index.html              # 功能版 App（实际跑流程）
│   └── design.html             # Claude Desktop 的 9 屏设计稿原稿
│
├── BASELINE_APP.md             NEW — 完整 API 契约 + 架构文档
├── MORNING_HANDOFF_2026-04-25.md  NEW — 本文件
└── requirements.txt            UPDATED — 加了 fastapi / uvicorn / multipart
```

---

## 🎨 UI 处理策略

你发给 Claude Desktop 的需求产生了一份完整的 9 屏设计稿 + 设计系统。我：

1. **保留原稿**作为设计基准 → `frontend/dist/design.html`（所有屏摆成 figma canvas）
2. **实现功能版**用同一套设计语言 → `frontend/dist/index.html`
   - 同调色板（clay / amber / court / ink / paper）
   - 同字体（Fraunces + Inter + JetBrains Mono）
   - 同组件（Serif / Sans / Mono / Tag / CTA / AppShell）
   - 差别：功能版接真 API，会跑完整流程

### 下一步 UI 迭代的最佳工作流

你问"应该怎么样工作更好？把咱们的代码发给他，还是把他的前端代码发给你？"——我建议：

**把 `BASELINE_APP.md` 的 API 契约章节发给 Claude Desktop**，让他按这份契约调整他的设计稿（比如把原设计里的假数据改成调我们真接口的占位）。他改完交给你再发给我，我负责把他的 JSX 翻译成 `frontend/dist/index.html` 里的 React 组件。

原因：
- Claude Desktop 不擅长理解后端代码
- 把 API 契约给他做约束，他设计的交互就不会跑偏
- 我这边已经有 Baseline 设计令牌的完整实现，直接套

---

## ⚠️ 已知限制（诚实说）

| 限制 | 影响 | 修复难度 |
|---|---|---|
| **VLM 仅正手**：`main.py::_run_vlm_analysis` 里 `is_backhand` 被跳过 | 单反 clip 的 `vlm` 和 `diagnosis` 字段会是 null，前端已优雅降级（只显示 biomechanics） | 中（要扩展现有正手 VLM 模板到单反）|
| **stroke classifier 对短 clip 偏向 backhand**：测试的两个 3 秒视频都被分类为单反 | 前端看不到 VLM/diagnosis，但 biomechanics + 标注视频能看到 | 低（可加 `?stroke=forehand` 覆盖参数）|
| **作业队列单进程**：重启丢失未完成作业 | 本地单用户没问题，多用户生产要换 Celery | 高 |
| **前端是 React+CDN 一页文件**：没有构建步骤 | 好处是零依赖；坏处是组件拆分粒度不够 | 低（要上线时再拆）|

---

## 🔧 如果发现问题怎么办

1. **API 不通** → 先 `curl http://127.0.0.1:8765/api/health`。如果连这个都 404，说明服务没起。
2. **分段出来 0 个 clip** → 视频没声音或声音太小。看 `GET /api/videos/{id}` 的 `error` 字段。
3. **诊断卡住 / 超时** → 看 `GET /api/jobs/{id}` 的 `error` 字段。多半是 YOLO 模型下载不到或 VLM API 配额。
4. **前端不显示 VLM 内容** → 如果 `stroke_type == one_handed_backhand`，这是预期（见上表第一条）。

---

## 📋 如果要发版，剩下的事

1. [ ] 打开 VLM 的单反支持（或显式提示"单反暂不支持 VLM 分析"）
2. [ ] 在 Upload 页面加一个"手动选择 stroke 类型"的开关（覆盖 auto 分类）
3. [ ] 把 Home 页的 "recent" 列表加删除按钮（后端 `DELETE /api/videos/{id}` 已就绪）
4. [ ] Claude Desktop 设计稿里的 Profile 页（30 天趋势）——需要累积历史数据后接入
5. [ ] 把 `storage/` 的自动清理加个按钮（比如保留最近 30 天）
6. [ ] 测试一下真机手机浏览器的体验（布局已经是 mobile-first）

---

## 🗂 文件入口速查

| 我想... | 去看 |
|---|---|
| 启动服务 | `python -m uvicorn app.main:app` |
| 改前端 | `frontend/dist/index.html`（单文件 React）|
| 改分段逻辑 | `segmentation/segmenter.py` |
| 改分段参数（阈值、片段长度） | `SoundBasedSegmenter.__init__` 的参数 |
| 改诊断流程 | `main.py::TennisAnalysisPipeline`（原有）|
| 改结果 JSON 结构 | `app/services.py::_serialize_result` |
| 加新 API | `app/routes.py` |
| 改存储路径 | `app/storage.py` |
| 看 API 自动文档 | http://localhost:8765/docs |

**一句话版本**：
> 后端能跑通完整流程；前端是功能原型，和设计稿同一套视觉语言；剩下的主要是打磨 UI 细节和补充 VLM 单反支持。

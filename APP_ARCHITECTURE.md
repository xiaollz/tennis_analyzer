# Baseline App — Architecture (v1.0)

> 2026-04-25 · 第一个稳定版。
>
> 这是 Baseline App 的设计与工程档案。和 `ARCHITECTURE.md`（v4.3 算法
> 引擎）的关系是：那一份描述的是**分析能力**，本文件描述的是**包装这
> 些能力成 App 的所有决策**——目录、状态、协议、空间管理、设计权衡。

---

## 目录

1. 视角与边界
2. 三层架构
3. 数据流
4. 文件存储
5. HTTP 协议
6. 前端状态机
7. 后台任务模型
8. 空间管理
9. 关键工程决策（Decisions Log）
10. 已知约束与未来路线

---

## 1. 视角与边界

**Baseline App 解决的是这一件事**：

> 用户拍一段练习视频 → App 自动按击球切片段 → 用户挑一个或多个片段
> → App 跑姿势 + VLM + 诊断引擎 → 返回每一片段的根因 + drill。

**不解决**：
- 实时对话教练（Discord 旁路）
- 多用户云服务（这是个本地优先的工具）
- 同步到云 / 多设备（离线本地）
- 比赛记录、对手追踪、社交分享

---

## 2. 三层架构

```
┌──────────────────────────────────────────────────────────────┐
│                                                                │
│  L3 — 前端                                                      │
│       frontend/dist/index.html  (React via CDN, 单文件 SPA)    │
│       9 屏 iOS 风格 UI（390×844 在桌面，全屏在手机）            │
│                                                                │
│            ↕  HTTP JSON / multipart                           │
│                                                                │
│  L2 — 应用 / 服务层                                              │
│       app/                                                     │
│       ├ main.py     FastAPI 入口 + 静态前端挂载                  │
│       ├ routes.py   /api/* HTTP 端点                            │
│       ├ services.py 包装管道：分段服务 + 诊断服务                  │
│       ├ jobs.py     线程池作业队列 + 进度状态                      │
│       └ storage.py  存储路径 + JSON 持久化 + 用量与清理            │
│                                                                │
│            ↕  内部 Python 调用                                  │
│                                                                │
│  L1 — 分析引擎（已有）                                            │
│       segmentation/  音频驱动分段（onset → 簇内主导峰过滤）        │
│       core/          姿势识别（YOLO 11m-pose）                  │
│       evaluation/    KPI、VLM、诊断引擎 v4.3                    │
│       config/        阈值与提示词                                │
│       knowledge/     FTT 网状知识图                              │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

**层与层之间**：每一层只通过它下面那一层的**显式入口**调用。L3 通过
HTTP 调 L2；L2 通过 Python 函数调 L1。L1 不知道有 App 存在；L2 不
知道前端长什么样。

---

## 3. 数据流（一次完整使用）

```
   ① 用户在 Home 选 "Forehand" + 点 "Upload"
        ↓
   ② 前端 POST /api/videos (multipart: file + stroke)
        ↓
   ③ 后端落盘到 storage/videos/{vid}/original.mp4
        ↓
   ④ 起一个 segment job → SoundBasedSegmenter
        ├ ffmpeg 提音频 → AudioOnsetDetector 检 onset
        ├ 簇内主导峰过滤（去回声）
        ├ 短视频强制单片段
        └ ffmpeg 切片 + OpenCV 缩略图
        ↓
   ⑤ 写 manifest.json，job done
        ↓
   ⑥ 前端轮询 /api/jobs/{job_id} 直到 done
        ↓
   ⑦ 前端 GET /api/videos/{vid} → 拿到片段清单 + 缩略图 URL
        ↓
   ⑧ 用户挑 1 或 N 个片段 (tap / 长按多选)
        ├ 单选 → POST /api/clips/{cid}/diagnose
        └ 多选 → 并行 N 个 POST，每个起一个 diagnose job
        ↓
   ⑨ 每个 diagnose job:
        TennisAnalysisPipeline.run(clip_path, stroke_mode=stroke)
        ├ 旋转检测（content-based，6 帧共识）
        ├ 姿势估计 (YOLO 11m-pose)
        ├ 击球检测 (HybridImpactDetector + dedup)
        ├ 类型分类 (StrokeClassifier，vote-based，conservative)
        ├ 评估 (ForehandEvaluator / BackhandEvaluator)
        ├ VLM 分析 (Gemini，仅 forehand 路径)
        ├ 诊断引擎 (v4.3 仲裁层)
        ├ 标注视频 + 关键帧网格 + Markdown 报告
        └ 落盘到 storage/diagnoses/{cid}/
        ↓
   ⑩ 前端 GET /api/diagnoses/{cid} → result.json
        ↓
   ⑪ 三 Tab 显示：Biomechanics / Observations / Coach
```

---

## 4. 文件存储

```
storage/                          (gitignored)
├── videos/
│   └── {video_id}/
│       ├── original.mp4          原视频（可被"freed"删除）
│       ├── meta.json             {filename, uploaded_at, stroke, ...}
│       ├── manifest.json         分段产物：clips 列表
│       └── clips/
│           ├── {clip_id}.mp4     2-3 秒片段（含音频）
│           └── {clip_id}.jpg     击球时刻缩略图（≤720px 长边）
│
├── diagnoses/
│   └── {clip_id}/
│       ├── status.json           {clip_id, status, started_at, stroke}
│       ├── result.json           前端可直接吃的 JSON（见 §5）
│       ├── annotated.mp4         骨骼叠加视频
│       ├── keyframe_grid.png     6 帧网格（VLM 输入）
│       └── report.md             长格式 Markdown 报告
│
└── jobs/
    └── {job_id}.json             作业状态（轮询用）
```

**ID 约定**：
- `video_id` = uuid4 前 12 位 hex（例：`38e3797e357a`）
- `clip_id` = `{video_id}_c{NNN}`（例：`38e3797e357a_c000`）
  - 这种命名让 `clip_id.split('_c')[0]` 反向找到所属 video
- `job_id` = uuid4 前 12 位 hex

---

## 5. HTTP 协议（前后端契约）

详见 `BASELINE_APP.md` 第三章。这里只列**所有端点**：

```
GET    /api/health                       健康检查
POST   /api/videos                       上传视频 + 触发分段（form: file, stroke）
GET    /api/videos                       视频列表
GET    /api/videos/{video_id}            视频详情 + 片段清单
DELETE /api/videos/{video_id}?keep_clips=bool   删除（可选保留 clips）
GET    /api/jobs/{job_id}                作业进度
GET    /api/clips/{clip_id}              单片段元数据
GET    /api/clips/{clip_id}/video        片段 MP4
GET    /api/clips/{clip_id}/thumbnail    片段缩略图
POST   /api/clips/{clip_id}/diagnose?stroke=    启动诊断
GET    /api/clips/{clip_id}/annotated    骨骼叠加 MP4
GET    /api/clips/{clip_id}/keyframes    关键帧网格 PNG
GET    /api/clips/{clip_id}/report.md    Markdown 报告
GET    /api/diagnoses/{clip_id}          诊断结果 JSON
GET    /api/storage                      磁盘用量摘要
DELETE /api/storage?older_than_days=     清理旧 jobs
POST   /api/storage/wipe                 核选项：清空所有
```

---

## 6. 前端状态机

```
              localStorage.baseline.onboarded?
                    │
        no  ┌───────┴───────┐  yes
            ▼               ▼
       Onboarding         Home
            │               │
            └─→ pickFile ←─ ┘
                   │
                Upload (Import 屏)
                   │ (poll segment job)
                   ▼
                Clips (ClipGrid)
                   │
              ┌────┴───┐
              ▼        ▼
         single tap   long-press → multi-select
              │              │
            Loading      MultiAnalyzing
              │              │
              ▼              ▼
            Result ←──── (jump to first)
              │
              └─→ back to Clips

  Library (从 Home 的 TabBar 进入):
      ├ 显示磁盘用量
      ├ 按 video 分组
      ├ "Free space" (删 original，留 clips/results)
      └ "Delete" (彻底删)
```

**前端持久化**（localStorage）：
- `baseline.onboarded` = "1" 后跳过引导
- `baseline.stroke` = "forehand" / "backhand" / "auto"

---

## 7. 后台任务模型

**在哪里跑**：`app/jobs.py::_executor`，`ThreadPoolExecutor(max_workers=2)`。

**为什么用线程池**：
- 本地单用户场景，不需要分布式
- 阻塞 IO（ffmpeg 等待 + GPU 推理）让线程池有意义
- 重启会丢未完成作业（用户能看到错误并重试）—— 可接受

**作业类型**：
| kind | payload | 时长 |
|---|---|---|
| `segment` | {video_id, video_path} | 0.5-3s |
| `diagnose` | {clip_id, stroke} | 10-30s |

**进度模型**：
- 作业函数接 `progress_cb(pct: float, msg: str)`
- 每次回调持久化到 `storage/jobs/{job_id}.json`
- 前端轮询 800-1500ms 一次

**多并行**：用户多选 N 个片段一起分析时，前端同时 POST N 个 diagnose
请求；后端线程池有 2 个 worker，N 个作业排队执行（前 2 个并行，后续
等待）。

---

## 8. 空间管理

**用户视角**：
```
Library Tab
├── 一个总用量数字（11.8 MB used）
├── 颜色条形图：videos / clips / diagnoses / jobs 占比
├── 按 video 列表：每行可以
│   ├ Open（跳到 ClipGrid）
│   ├ Free space（删 original.mp4，留 clips + results）
│   └ Delete（彻底）
└── 全局：Clean old jobs · Wipe everything
```

**保证**：
1. **物理删除**：所有 delete 操作走 `shutil.rmtree`，不是软删除
2. **关联清理**：删 video 时自动连带删除其所有 clip 的 diagnoses 目录
3. **Free space 模式**：删 original.mp4 但保留 manifest + clips +
   diagnoses，节省 70-90% 空间但保留分析结果
4. **作业回收**：jobs/*.json 默认 7 天过期，可手动触发清理
5. **Wipe**：清空 storage/ 下一切，重建空白结构

**典型空间占用**（10 秒 1080p 视频）：
| 文件 | 大小 |
|---|---|
| original.mp4 | 1.5-3 MB |
| clips/{cid}.mp4 (×3) | 250-300 KB ea |
| clips/{cid}.jpg (×3) | 50-60 KB ea |
| diagnoses/{cid}/annotated.mp4 | 800 KB-1.5 MB |
| diagnoses/{cid}/keyframe_grid.png | 200-500 KB |
| diagnoses/{cid}/result.json | 5-15 KB |
| diagnoses/{cid}/report.md | 1-3 KB |
| **总计** | **5-10 MB / video** |

10 个视频不会爆炸 ≈ 50-100 MB。

---

## 9. 关键工程决策（Decisions Log）

### D-001 · 用户必须显式选 stroke type

**问题**：StrokeClassifier 在 2D 侧拍视频上误判率高（用户的测试视频
被分到 1H Backhand 信心 1.0）。

**根因**：2D 投影下，从不同侧拍角度看，正手和反手的关键点轨迹**会
呈现镜像**——单凭 wrist_x 与 body_center 的关系无法区分。

**结论**：放弃 100% 自动分类。UI 默认 Forehand / Backhand 二选一，
"Auto" 标为 experimental。

**后续可能改进**：
- 让 VLM 介入分类（用 1 帧让 Gemini 判定）
- 引入 3D 姿势估计（成本高）
- 让用户标注几条作为 fine-tune 数据

### D-002 · 单进程线程池而非 Celery

**问题**：本地 App 要不要做分布式作业队列？

**结论**：不做。本地单用户场景。重启丢失未完成作业是可接受的。
如果未来上多用户云服务，再换。

### D-003 · React via CDN，无构建步骤

**问题**：要不要 webpack / vite？

**结论**：不要。前端是单 HTML 文件 + Babel 浏览器内编译。
- ✅ 零依赖、零构建、改完直接刷新
- ✅ Claude Desktop 设计稿用同一套技术，可直接对照
- ❌ 加载时间稍慢（首次 ~2s）—— 本地用 OK
- 上线 / 商用前再上构建系统

### D-004 · 段内"主导峰"过滤回声

**问题**：单击视频被切成多段（onset 检测捕到回声 / 球落地）。

**结论**：在 `SoundBasedSegmenter` 里加：
- 0.7s 窗口内只保留最强 onset（其余视为回声丢弃）
- 短视频（≤4s）强制单片段
- 这是和 `AudioOnsetDetector` **解耦**的过滤层——detector 仍可被
  其他模块原样使用

### D-005 · 击球点 dedup 用 audio_strength 而非 wrist_speed

**问题**：当一个挥拍触发多个 ImpactEvent（contact + 随挥峰值），
原本 `_dedupe_overlapping_impacts` 选 `peak_wrist_speed` 最高的——
但 wrist 在随挥处速度更高，所以选错了"击球帧"，导致 prep window 跑
到错误位置。

**结论**：当 audio_confirmed 存在时，选**最早**的（真正的球-拍接触
瞬间）；audio 没确认时再回退到 wrist_speed。

### D-006 · Stroke override 通过 query / form 参数

**问题**：怎么把用户的 stroke 选择传到 pipeline？

**结论**：
- `POST /api/videos` 接 form 字段 `stroke`，存到 video meta
- `POST /api/clips/{id}/diagnose` 接 query `?stroke=`（可覆盖 video meta）
- pipeline 接 `stroke_mode` constructor 参数

这种"显式参数"路径让前端任何位置都可以覆盖，调试也方便。

### D-007 · 分段缓冲 1.5s pre + 1.8s post

**问题**：原 0.8 / 1.2 缓冲太紧，prep 不全。

**结论**：扩到 1.5 / 1.8 → 单挥拍 ~3.3 秒。够覆盖 Unit Turn → 随挥。
**没用**姿势驱动的精确边界检测——成本太高（要先跑 pose），固定缓冲
对正手节奏稳定。

### D-008 · 多选用长按而非显式按钮

**问题**：多选 UI 怎么做？

**结论**：iOS 习惯——长按 0.45s 进入多选模式，再 tap 是切换选中。
"Cancel" 退出多选。这样**单击仍然是直接打开诊断**，不增加常用路径
的步数。

### D-009 · "Free space" 是软删除原视频，不是软删除全部

**问题**：用户想删除原视频但保留分析结果。

**结论**：`DELETE /api/videos/{id}?keep_clips=true` 只删
`original.mp4`，保留 manifest.json + clips/ + diagnoses/。再分析一次
不需要重传，但**重新分段**做不到（原视频不在了）。

### D-010 · 旋转检测 6 帧共识 + 严阈值

**问题**：旧 `detect_rotation_from_pose` 用 1.5x 阈值 + 2 帧共识，
正常视频里挥拍瞬间躯干倾斜被误判为"侧躺"，触发误旋转。

**结论**：阈值升到 3.0x，要求 |dx| > 80px，且 6/10 帧一致。
代价：极端"侧拍"视频可能漏旋转——但这是更安全的失败模式。

---

## 10. 已知约束与未来路线

### 当前约束（设计上）

| 项 | 现状 | 说明 |
|---|---|---|
| Stroke 自动分类 | 2D 不可靠 | 让用户显式选 |
| VLM only 正手 | 反手路径不调 VLM | 沿用 v4.3 引擎限制；反手也能给 KPI 但没观察/根因 |
| 单进程作业 | ThreadPoolExecutor(2) | 本地用足够 |
| 视频上限 | 没硬限 | 实际看 ffmpeg 处理速度 |
| 移动端 PWA | 没做 | 现在是浏览器加书签可用 |
| 多设备同步 | 不支持 | 本地优先 |
| 离线 VLM | 不支持 | 需要 Gemini API |

### 短期路线（下一周）

1. **反手也走 VLM + 诊断**：扩展 prompts/概念到反手
2. **Stroke 后悔药**：诊断完发现选错 → 一键重跑用对的
3. **批量多选的优化**：当前并发受限于 2 worker，可以打到 4
4. **段内多击球的更细 cut**：1 段 2 击球的话，细分到子段

### 中期路线

1. **PWA 化**：可以"加到主屏幕"，缓存静态资源，离线可看历史
2. **训练趋势**：跨 session 的 KPI 折线 + 分位
3. **对比模式**：选两个 clip 横排对比（设计稿里有）
4. **连接通知**：训练完一键 Discord 推送 + Claude Code 后续聊天

### 长期路线

1. **3D 姿势重建**：解决 stroke 分类不可靠 + 角度依赖
2. **多用户云版本**：Celery + S3 + auth
3. **教练协作**：上传视频 → 教练评论 → 学生看回放

---

## 附录 · 关键文件指针

| 我想... | 去看 |
|---|---|
| 改 UI | `frontend/dist/index.html` |
| 改设计令牌 | `frontend/dist/index.html`（开头 `T = {...}`）|
| 看设计稿原稿 | `frontend/dist/design.html` |
| 改分段算法 | `segmentation/segmenter.py` |
| 改后端 API | `app/routes.py` |
| 改作业逻辑 | `app/jobs.py` |
| 改存储路径 | `app/storage.py` |
| 改诊断流程 | `main.py::TennisAnalysisPipeline` |
| 改分类器 | `evaluation/stroke_classifier.py` |
| 改诊断引擎 | `evaluation/diagnosis_engine.py` (v4.3) |
| 跑端到端测试 | `python scripts/smoke_test_app.py` |
| 启动 | `./start.sh` |

---

**最后更新**：2026-04-25 · v1.0 stable · 三 Bug 修复 + 多选 + 空间
管理 + 双 stroke 默认。

# 项目架构总览

> 框架级文档。改代码或换方向前先读这里，避免重复劳动或走偏。
> 最后更新：2026-04-23（v4.3 · 仲裁层）
> 详细变更见 `docs/research/coach_analysis/integration_changelog.md` 和 `docs/record/learning.md`。

---

## 一、核心理念

**这是给一个学习者（本项目用户）用的网球正手诊断系统，不是给所有人用的产品。**

- 用户：自学者，已读完 FTT 体系，当前在学辛纳（Jannik Sinner）的现代极简正手
- 输入：手机拍摄的训练视频（60fps，960×540 压缩后）
- 输出：根因诊断 + 单一改动建议（一次只改一件事）
- 知识来源：FTT 书 + FTT/Tom Allsopp/Feel Tennis YouTube 频道 + 生物力学教科书 + 用户训练日志（learning.md）

**设计原则：**
1. **VLM 只观察，不推理**。推理交给诊断引擎 + 知识图谱。
2. **Top-down 诊断**：找最早出问题的层 = 根因，下游全是代偿。
3. **一份报告只输出一个根因 + 一个 drill + 一个口令**。
4. **量化数据用来交叉验证 VLM，不用来评分**。VLM 和量化都是"证据"，知识库推理是"医生"。
5. **报告语气是教练，不是 AI 助手**。严禁列条目、严禁 AI 味。
6. **VLM-算法冲突时走仲裁**（v4.3 新增）：量化数据有明确反驳 → 撤销 VLM false positive，不并列矛盾结论。

---

## 二、目录结构

```
tennis/
├── core/                  姿态检测（YOLO COCO17）+ 挥拍切片
│   ├── pose_estimator.py
│   └── video_processor.py
├── analysis/              KPI 计算（基于姿态关键点的几何指标）
│   ├── kinematic_calculator.py
│   └── trajectory.py
├── evaluation/            VLM + 诊断引擎（核心护城河）
│   ├── vlm_analyzer.py          Gemini 调用、视频/关键帧两种模式（2154 行）
│   ├── diagnosis_engine.py      知识图谱推理 + 5 层 top-down + 仲裁层（2550 行）
│   ├── forehand_evaluator.py    正手 KPI 聚合 + 评分入口
│   ├── backhand_evaluator.py    反手 KPI 聚合
│   ├── kpi.py / backhand_kpi.py 单球 KPI 计算
│   ├── event_detector.py        音频+视觉混合击球检测（阈值 600 px/s）
│   ├── stroke_classifier.py     正手/反手自动识别
│   └── hallucination_mitigation.py  VLM 幻觉抑制（稳定态 legacy，VLM 管道在用）
├── knowledge/             知识库
│   ├── extracted/         概念节点 + 因果边 + 肌肉激活指南
│   ├── templates/vlm/     VLM prompt 模板（5 层 38 题）
│   ├── graph/             图谱构建工具
│   └── coach_videos_v2/   真看过的教练视频笔记（FTT/Tom/Feel Tennis）
├── report/                报告生成器（Python 模块）
│   ├── report_generator.py  Markdown 分层显示
│   └── visualizer.py        关键帧 6 帧 2×3 网格 + 轨迹线 + 橙色问题框
├── docs/
│   ├── research/          研究文档（FTT 书提取、教练视频分析、球员对比）
│   ├── record/            用户训练日志（learning.md）+ 录像协议
│   ├── algorithm_design/  算法设计文档
│   ├── metrics/           指标设计
│   ├── knowledge_graph/   图谱设计
│   └── *.md               教学参考
├── reports/               生成的诊断报告（按日期 YYYY-MM-DD 组织）
├── scripts/               一次性脚本（视频观看、EPUB 生成等）
├── tests/                 测试
├── config/                API 密钥 / 配置
├── models/                YOLO 姿态模型权重
├── videos/                训练视频（coach / players / user）
└── main.py                端到端入口
```

**死代码已清理** (2026-04-23)：旧 `tennis_analyzer/` 子包（仅 .DS_Store 空壳）已删除。

---

## 三、视频分析流水线（v4.3）

```
视频输入（60fps / 960×540）
  ↓
[1] 姿态检测       core/pose_estimator.py    YOLO 提 17 关键点序列
  ↓
[2] 挥拍切片       core/video_processor.py   音频+视觉混合检测击球帧
  ↓
[3] KPI 量化       analysis/ + evaluation/kpi.py
                   几何指标：肩转角、重心、击球点、膝角、同步性、
                   scooping_depth、forward_extension 等
  ↓
[4] VLM 观察       evaluation/vlm_analyzer.py
                   每个挥拍单独调用 Gemini，回答 38 个观察题
                   分 5 层：L1 Contact / L2 Rhythm / L3 Kinetic Chain
                            / L4 Preparation / L5 Footwork
  ↓
[5] 诊断引擎       evaluation/diagnosis_engine.py
    ├─ Step 1:   VLM 答案 → 概念 ID（Q-direct + 关键词映射）
    ├─ Step 1.5: KPI 信号注入（v4.2） → L4/L5 量化证据进入候选池
    ├─ Step 1.6: VLM↔量化 仲裁（v4.3） → 撤销 VLM false positive
    ├─ Step 2:   Top-down 根因追溯（L5→L1 earliest layer wins）
    ├─ Step 3:   量化验证（confirm/contradict，跳过已仲裁概念）
    ├─ Step 4:   用户历史对比（learning.md 提取的递归问题）
    └─ Step 5:   拉 drill + 肌肉激活提示 + 口令
  ↓
[6] 报告生成       report/report_generator.py
                   3 段式：问题是什么 → 为什么 → 怎么解决
                   按层级显示根因 + 准备阶段独立板块
                   🔬 VLM-算法仲裁小节（v4.3）
                   VLM 原始观察折叠在 details 块
```

---

## 四、关键文件指针

| 想做什么 | 看哪 |
|---|---|
| 改 VLM 问的问题 | `knowledge/templates/vlm/system_prompt.md.j2` |
| 加新概念到知识库 | `knowledge/extracted/preparation_footwork_concepts.json` 同格式 |
| 改诊断逻辑 | `evaluation/diagnosis_engine.py` |
| 加 KPI 注入规则 | `_KPI_INJECTION_RULES`（L1622） |
| 加 VLM-算法仲裁规则 | `_ARBITRATION_RULES`（L1731） |
| 改报告格式 | `report/report_generator.py` |
| 改录像方式 | `docs/record/recording_protocol.md` |
| 看用户当前状态 | `docs/record/learning.md` |
| 找正手知识 | `docs/research/13_synthesis.md` 最全 |
| 辛纳学习模板 | `docs/research/05_ftt_blog_players.md` §16 |
| 教练视频分析 | `docs/research/ftt_video_analyses/` |

---

## 五、版本演进

| 版本 | 核心变化 | 解决的问题 |
|---|---|---|
| **v1.0** | KPI 评分 + 单 VLM 调用 | 基础诊断能跑通 |
| **v2.0** | 多轮迭代 VLM | 单次 VLM 容易套概念 |
| **v3.0** | 根因树 + 因果叙述 | 输出像列条目，没有逻辑 |
| **v4.0** | VLM 只观察 + 诊断引擎推理 | VLM 编理由、套术语 |
| **v4.1** | 肌肉激活 + 用户历史整合 | 诊断不够个性化 |
| **v4.2** | 5 层 Top-down + 准备/步伐扩展 + KPI 注入 | 只看击球瞬间，看不到准备阶段 |
| **v4.3** | VLM-算法仲裁层 | VLM 视觉误判（如 V 形 scooping）污染诊断 |

### v4.2 扩展细节
- 知识图谱：+39 个准备/步伐概念（带 L1-L5 层级）
- VLM prompt：20 题 → 38 题，按 5 层组织
- 诊断引擎：top-down "earliest layer wins" 推理
- `_inject_kpi_problems()`：shoulder_rotation + min_knee_angle 注入 L4/L5 概念
- 真看了 18 个教练视频补充视觉细节

### v4.3 新增：仲裁层
**动机**：2026-04-22 用户实测发现系统把 V 形轨迹误判为业余 scooping（VLM 视觉判断 vs 算法数值不一致）。

**架构命题**：
> VLM 输出 + 数值指标 = **两类证据**（类似 CT 影像 + 血液检查）
> 知识库推理引擎 = **诊断者**（类似医生）
> 不能简单并列矛盾结论，必须综合论证 → 仲裁

**实现**：
- `_ARBITRATION_RULES` 表：可扩展的 VLM-算法冲突规则
- `_arbitrate_vlm_vs_metrics()`：撤销 VLM false positive（narrow exception catch + warning log）
- `diagnose()` Step 1.6 接入
- `_validate_with_metrics(excluded_concepts=...)`：避免已仲裁概念双重展示
- 报告新增 🔬 VLM-算法仲裁小节

**首条规则（problem_p02 Scooping）**：
- 条件：`scooping_detected=False` 且 `scooping_depth ≤ 0.3`
- 裁决：撤销 VLM，判定为现代正手被动 lag drop（非业余主动捞球）

**实测**（3 视频对比）：
- V 形提及 16 次/报告 → 6 次（-62%）
- Video 1: 53→63 分 / Video 3: 60→81 分

---

## 六、重要的"不要做"清单

- ❌ **不要让 VLM 做推理**。它会编理由。只让它描述看到的东西。
- ❌ **不要把所有问题都报告**。一份报告只有一个根因。
- ❌ **不要给量化指标打分让 VLM 看**。VLM 会被分数带跑。
- ❌ **不要在报告里列条目**（除非真的是清单）。用段落，教练口吻。
- ❌ **不要在没有 layer 标签的情况下加新概念**。新概念必须能进入 top-down 流程。
- ❌ **不要绕过 `_trace_root_causes()` 直接输出**。所有诊断必须走根因追溯。
- ❌ **不要为假设需求加 feature flag**。改就改，不要兼容性壳。
- ❌ **VLM 和算法冲突时不要并列展示**（v4.3）。加仲裁规则，不要改 VLM prompt——prompt 永远抓不完视觉错觉。

---

## 七、已知遗留问题 / 可清理债

### 诊断引擎代码重构（推迟）
> 2026-04-23 code review 发现，不影响功能，但未来扩展时有价值。

1. **三表语义重合**：`_CONCEPT_TO_METRIC_VALIDATION` / `_KPI_INJECTION_RULES` / `_ARBITRATION_RULES` 三张表里同一批概念的阈值各自为政（straight_legs / unit_turn / problem_p02 都同时出现在三张表）。建议合并为一张"概念-指标映射表"，三种用途（validate/inject/arbitrate）是不同字段。
2. **`_METRIC_THRESHOLDS` 存在但未利用**：L356 定义了中心阈值表，但 `_validate_with_metrics`、`_KPI_INJECTION_RULES`、`_ARBITRATION_RULES` 都没从这里读。导致 `scooping_depth > 0.3` 等硬编码散落 7+ 处。
3. **`source` 字段 stringly-typed**：`"kpi" / "kpi+vlm" / "q_direct" / "keyword"` 应改 Enum，新增 source 时有类型保护。
4. **`_generate_narrative` 8 参数**：已到临界，下次加参数前抽 `@dataclass NarrativeContext`。

### 功能未完成
- 17 个 pre-existing 测试失败（test_extraction、test_vlm_prompt、test_v2_modules）—— 历史遗留，与最近改动无关
- 诊断链 JSON 数据质量差（p09 几乎是所有链的 root_cause），目前绕开走 OBSERVATION_TO_CONCEPT
- 球检测和对手击球时刻检测尚未实现（"准备时机相对球"依赖 VLM 自己看）

### v4.3 之后可扩展的仲裁规则
- **Over-rotation**：VLM 看到"身体过转" vs 算法 shoulder_rotation 合理
- **Tilt 时机错误**：VLM 看到"右肩下沉" vs 算法肩高差在合理范围
- **击球晚**：VLM 看到"手臂够球" vs 算法击球点在身前合理位置

---

## 八、下一步可能的方向（不承诺）

1. **仲裁规则扩展**：把上节的 3 个候选实测并加规则
2. **三表合并重构**：消除概念-阈值散落，收敛成单一真理源
3. **球+对手检测**：YOLOv8 加球检测，给"对手击球瞬间"做绝对时间锚
4. **多机位融合**：同一挥拍两个角度同时输入 VLM
5. **训练后追问模式**：让用户文字描述当时的感觉，融合到诊断
6. **历史趋势可视化**：把 learning.md 提取的问题画成时间线

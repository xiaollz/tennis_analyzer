# 项目架构总览

> 框架级文档。改代码或换方向前先读这里，避免重复劳动或走偏。
> 详细变更见 `docs/research/coach_analysis/integration_changelog.md` 和各 phase summary。

---

## 一、核心理念

**这是给一个学习者用的网球正手诊断系统，不是给所有人用的产品。**

- 用户：自学者，已读完 FTT 体系，处于"手臂打 → 身体带"过渡期
- 输入：手机拍摄的训练视频
- 输出：根因诊断 + 单一改动建议（一次只改一件事）
- 知识来源：FTT 书 + FTT/Tom Allsopp/Feel Tennis YouTube 频道 + 生物力学教科书

**设计原则：**
1. VLM 只观察，不推理。推理交给诊断引擎 + 知识图谱。
2. Top-down 诊断：找最早出问题的层 = 根因，下游全是代偿。
3. 一份报告只输出一个根因 + 一个 drill + 一个口令。
4. 量化数据用来交叉验证 VLM，不用来评分。
5. 报告语气是教练，不是 AI 助手。严禁列条目、严禁 AI 味。

---

## 二、目录结构

```
tennis/
├── core/                  姿态检测（YOLO COCO17）+ 挥拍切片
├── analysis/              KPI 计算（基于姿态关键点的几何指标）
├── evaluation/            VLM + 诊断引擎（核心护城河）
│   ├── vlm_analyzer.py    Gemini 调用、视频/关键帧两种模式
│   └── diagnosis_engine.py 知识图谱推理 + 5 层 top-down 根因追溯
├── knowledge/             知识库
│   ├── extracted/         概念节点 + 因果边 + 肌肉激活指南
│   ├── templates/vlm/     VLM prompt 模板（5 层 38 题）
│   └── graph/             图谱构建工具
├── report/                报告生成器（Markdown，分层显示）
├── docs/
│   ├── research/          研究文档（FTT 书提取、教练视频分析）
│   ├── record/            用户训练日志 + 录像协议
│   └── *.md               教学参考文档
├── reports/               生成的诊断报告（按日期）
├── scripts/               一次性脚本（视频观看、EPUB 生成等）
├── tests/                 测试
└── main.py                端到端入口
```

---

## 三、视频分析流水线（v4.2）

```
视频输入
  ↓
[1] 姿态检测       core/      YOLO 提关键点序列
  ↓
[2] 挥拍切片       core/      检测每次挥拍的开始/结束/contact 帧
  ↓
[3] KPI 量化       analysis/  几何指标：肩转角、重心、击球点位置等
  ↓
[4] VLM 观察       evaluation/vlm_analyzer.py
                   每个挥拍单独调用 Gemini，回答 38 个观察题
                   分 5 层：L1 Contact / L2 Rhythm / L3 Kinetic Chain
                            / L4 Preparation / L5 Footwork
  ↓
[5] 诊断引擎       evaluation/diagnosis_engine.py
                   - VLM 答案 → 概念 ID（关键词映射 + Q-direct）
                   - 量化数据交叉验证（标记 VLM 错误观察）
                   - 知识图谱因果链遍历
                   - Top-down：从 L5 往 L1 找最早出问题的层
                   - 拉取该层的肌肉激活提示 + drill
                   - 拉取用户历史（learning.md 提取的递归问题）
  ↓
[6] 报告生成       report/report_generator.py
                   3 段式：问题是什么 → 为什么 → 怎么解决
                   按层级显示根因 + 准备阶段独立板块
                   VLM 原始观察折叠在 details 块
```

---

## 四、关键文件指针

| 想做什么 | 看哪 |
|---|---|
| 改 VLM 问的问题 | `knowledge/templates/vlm/system_prompt.md.j2` |
| 加新概念到知识库 | `knowledge/extracted/preparation_footwork_concepts.json` 同格式 |
| 改诊断逻辑 | `evaluation/diagnosis_engine.py` 的 4 个字典 + `_trace_root_causes()` |
| 改报告格式 | `report/report_generator.py` |
| 改录像方式 | `docs/record/recording_protocol.md` |
| 看用户当前状态 | `docs/record/learning.md` |
| 找正手知识 | `docs/research/13_synthesis.md` 最全 |
| 教练视频分析 | `docs/research/coach_analysis/` |
| 真看过的视频笔记 | `knowledge/extracted/coach_videos_v2/` |

---

## 五、版本演进（高层方向）

| 版本 | 核心变化 | 解决的问题 |
|---|---|---|
| **v1.0** | KPI 评分 + 单 VLM 调用 | 基础诊断能跑通 |
| **v2.0** | 多轮迭代 VLM | 单次 VLM 容易套概念 |
| **v3.0** | 根因树 + 因果叙述 | 输出像列条目，没有逻辑 |
| **v4.0** | VLM 只观察 + 诊断引擎推理 | VLM 编理由、套术语 |
| **v4.1** | 肌肉激活 + 用户历史整合 | 诊断不够个性化 |
| **v4.2** | 5 层结构 + Top-down 根因 + 准备/步伐扩展 | 只看击球瞬间，看不到准备阶段 |

**v4.2 的具体扩展：**
- 知识图谱：+39 个准备/步伐概念（带 L1-L5 层级）
- VLM prompt：20 题 → 38 题，按 5 层组织
- 诊断引擎：top-down "earliest layer wins" 推理
- 真看了 18 个教练视频（10 FTT/Tom + 8 Feel Tennis）补充视觉细节
- 录像协议：斜后 45° 主机位 + 侧面副机位

---

## 六、重要的"不要做"清单

- ❌ **不要让 VLM 做推理**。它会编理由。只让它描述看到的东西。
- ❌ **不要把所有问题都报告**。一份报告只有一个根因。
- ❌ **不要给量化指标打分让 VLM 看**。VLM 会被分数带跑。
- ❌ **不要在报告里列条目**（除非真的是清单）。用段落，教练口吻。
- ❌ **不要在没有 layer 标签的情况下加新概念**。新概念必须能进入 top-down 流程。
- ❌ **不要绕过 `_trace_root_causes()` 直接输出**。所有诊断必须走根因追溯。
- ❌ **不要为假设需求加 feature flag**。改就改，不要兼容性壳。

---

## 七、已知遗留问题

- 17 个 pre-existing 测试失败（test_extraction、test_vlm_prompt、test_v2_modules）—— 与最近改动无关，是历史遗留
- 量化算法对 scooping 检测灵敏度不够（VLM 经常报但量化不报）
- 诊断链 JSON 数据质量差（p09 几乎是所有链的 root_cause），目前直接绕开走 OBSERVATION_TO_CONCEPT
- 球检测和对手击球时刻检测尚未实现（所以"准备时机相对球"的诊断目前依赖 VLM 自己看）

---

## 八、下一步可能的方向（不承诺）

1. **球+对手检测**：YOLOv8 加球检测，给"对手击球瞬间"做绝对时间锚
2. **多机位融合**：同一挥拍两个角度同时输入 VLM
3. **训练后追问模式**：让用户文字描述当时的感觉，融合到诊断
4. **历史趋势可视化**：把 learning.md 提取的问题画成时间线

# V2 Research: Multi-Round VLM Diagnostic System

**Domain:** Iterative VLM reasoning for tennis technique diagnosis
**Researched:** 2026-04-03
**Overall confidence:** HIGH (architecture patterns well-established; Gemini API capabilities verified)

---

## Executive Summary

The v1.0 system performs two-pass VLM analysis: a quick symptom scan (Pass 1) followed by a targeted deep analysis with graph-backed diagnostic chains (Pass 2). This is effective for identifying symptoms and tracing them to root causes in a single shot, but it has a fundamental limitation: the VLM must get everything right in one attempt. Real coaching diagnosis is iterative -- a coach observes, forms hypotheses, looks more closely at specific details, discards wrong hypotheses, and confirms the root cause through targeted re-observation.

The v2.0 upgrade adds a multi-round iterative loop on top of the existing two-pass architecture. The key insight is that each VLM round should ask increasingly specific questions derived from the knowledge graph, not just repeat the same broad analysis. The knowledge graph's 18 diagnostic chains with their `check_sequence` steps are the natural driver for this: each round executes one or more diagnostic checks, and the VLM's observations determine which branch to follow next.

The architecture is a hypothesis-driven state machine: observe -> hypothesize (using the graph) -> re-observe (targeted questions) -> update hypotheses -> converge or loop. This mirrors the ViTAR "think-act-rethink-answer" paradigm from recent medical VLM research, adapted to our domain-specific knowledge graph.

Cost implications are manageable: 3-4 rounds at ~560 tokens per image input on Gemini Flash costs approximately $0.003-0.005 per analysis (assuming ~2K output tokens per round). The existing $10K char prompt budget is sufficient since each round uses a focused subset of the graph, not the full diagnostic context.

---

## 1. Multi-Round VLM Loop Architecture

### Current State (v1.0)

```
Pass 1: Image + symptom_checklist.j2 -> VLM -> detected chain IDs
Pass 2: Image + system_prompt.md.j2 + diagnostic_deep.j2 (for detected chains) -> VLM -> root_cause_tree JSON
```

Both passes send the same 2x3 keyframe grid image. Pass 2 gets all the diagnostic context at once and must reason through everything in a single generation.

### Proposed State (v2.0)

```
Round 0 (Scan):     Image + symptom_checklist -> VLM -> initial hypotheses (same as current Pass 1)
Round 1 (Diagnose): Image + targeted_questions_from_graph -> VLM -> observations + updated hypotheses
Round 2 (Verify):   Image + verification_questions -> VLM -> confirmed/eliminated hypotheses
Round N (Confirm):  Image + confirmation_check -> VLM -> final root cause tree
```

### State That Persists Between Rounds

| State Element | Type | Purpose |
|---|---|---|
| `hypotheses` | `list[Hypothesis]` | Active hypothesis objects with status (active/confirmed/eliminated) and confidence |
| `observations` | `list[Observation]` | Factual VLM observations anchored to specific frames |
| `round_history` | `list[RoundResult]` | Full prompt/response per round for audit trail |
| `active_chains` | `list[str]` | DiagnosticChain IDs still under investigation |
| `checked_steps` | `dict[str, list[int]]` | Which check_sequence steps have been evaluated per chain |
| `convergence_score` | `float` | How close we are to a stable diagnosis (0.0-1.0) |

**Confidence: HIGH** -- This state model maps directly to the existing DiagnosticChain.check_sequence structure. Each `DiagnosticStep` has `check`, `if_true`, `if_false` -- this is already a branching state machine.

### Key Design Decision: Same Image, Different Questions

The same keyframe grid is sent every round. What changes is the prompt -- each round asks the VLM to look at specific details in specific frames. This is critical because:

1. VLMs attend differently based on what they're asked to look for
2. A broad "analyze this" prompt causes the VLM to spread attention thin
3. A targeted "in frame 3, is the wrist above or below the elbow?" focuses attention precisely

This mirrors the Gemini 3 "Agentic Vision" pattern: Think-Act-Observe loop where the same image is re-examined with different focus.

---

## 2. Knowledge-Driven Observation Directives

### From Graph to VLM Questions

The existing `DiagnosticChain.check_sequence` already contains ordered investigation steps. Example from a hypothetical scooping chain:

```python
DiagnosticStep(
    check="Is racket head drop passive (gradual) or active (sudden)?",
    check_zh="拍头下落是被动的(缓慢)还是主动的(急剧)?",
    if_true="active_lag_creation",  # root cause concept ID
    if_false=None  # proceed to next check
)
```

The v2.0 system converts these into targeted VLM observation prompts:

```
Round 1 prompt fragment:
"在图2-3中，观察拍头下落：
 - 黄色手腕轨迹从图2到图3的变化是缓慢平滑的（被动下落）还是有急剧的方向变化（主动下压）？
 - 如果有V形尖角，尖角在轨迹的前半段还是后半段？
 回答格式：{observation: '...', judgment: 'passive|active', confidence: 0-1}"
```

### Concept-to-Observation Mapping

Each `Concept` in the graph has `vlm_features: list[str]` -- these are VLM-observable visual features. The system uses these to generate frame-specific observation requests.

New mapping needed:

| Graph Element | Observation Directive |
|---|---|
| `Concept.vlm_features` | What visual feature to look for |
| `DiagnosticStep.check` | The specific question to answer |
| `DiagnosticChain.vlm_frame` | Which frame(s) to examine |
| `Edge(relation="visible_as")` | How a concept manifests visually |
| `Edge(relation="causes")` | What downstream symptoms to check |

### Template: observation_directive.j2 (new)

A new Jinja2 template that generates targeted observation prompts for a single round:

```jinja2
你正在对一次正手击球进行第{{ round_number }}轮诊断。

【当前假设】
{% for h in active_hypotheses %}
- {{ h.name_zh }}（置信度: {{ h.confidence }}）
  {% if h.supporting_evidence %}支持证据: {{ h.supporting_evidence }}{% endif %}
{% endfor %}

【本轮观察任务】
请仔细观察以下具体细节，不要做全面分析：
{% for directive in directives %}
{{ loop.index }}. 在{{ directive.frame }}中: {{ directive.question_zh }}
   VLM特征: {{ directive.vlm_features | join("; ") }}
{% endfor %}

【回答格式】
严格JSON：
{
  "observations": [
    {"directive_id": 1, "observation": "具体看到了什么", "judgment": "yes|no|unclear", "confidence": 0.0-1.0}
  ],
  "hypothesis_updates": [
    {"hypothesis_id": "...", "action": "confirm|eliminate|adjust", "reason": "基于什么观察"}
  ]
}
```

**Confidence: HIGH** -- The existing graph structure (vlm_features, check_sequence, vlm_frame) already contains the data needed. This is primarily a prompt engineering + orchestration task, not a data modeling task.

---

## 3. Optimal Number of Rounds and Stopping Criteria

### Round Budget

| Rounds | Use Case | Cost (Flash) |
|---|---|---|
| 2 (scan + confirm) | Simple cases: single obvious root cause | ~$0.002 |
| 3 (scan + diagnose + confirm) | Typical cases: 2-3 hypotheses to differentiate | ~$0.003 |
| 4 (scan + diagnose + re-observe + confirm) | Complex cases: contradictory evidence, multiple root causes | ~$0.005 |
| 5+ | Diminishing returns; VLM likely hallucinating or image quality insufficient | N/A |

**Recommendation: Max 4 rounds, with early exit.** Use 2-4 depending on convergence.

### Stopping Criteria (when to exit the loop)

1. **Single dominant hypothesis** with confidence >= 0.8 and at least 2 supporting observations -> STOP, generate report
2. **All hypotheses eliminated** except one -> STOP (last standing = root cause by elimination)
3. **No hypothesis change** for 2 consecutive rounds -> STOP (VLM cannot distinguish further with available evidence)
4. **Max rounds reached** (4) -> STOP, report best hypothesis with uncertainty flag
5. **All check_sequence steps exhausted** for all active chains -> STOP

### Convergence Score

```python
convergence = max(h.confidence for h in active_hypotheses if h.status == "confirmed") 
              if any confirmed 
              else 1.0 - (len(active_hypotheses) / initial_hypothesis_count)
```

When convergence >= 0.8, exit the loop.

**Confidence: MEDIUM** -- The 4-round max is an informed estimate. Needs empirical validation. Medical VLM research (ViTAR) uses 2-3 reasoning steps for typical cases, which aligns.

---

## 4. Hypothesis Tracking Across Rounds

### Hypothesis Model

```python
class Hypothesis(BaseModel):
    """A diagnostic hypothesis being tracked across VLM rounds."""
    id: str                          # e.g., "hyp_scooping_active_lag"
    chain_id: str                    # DiagnosticChain ID
    root_cause_concept_id: str       # From chain.root_causes
    name: str
    name_zh: str
    status: Literal["active", "confirmed", "eliminated"]
    confidence: float                # 0.0 to 1.0
    supporting_observations: list[str]  # observation IDs
    contradicting_observations: list[str]
    check_steps_completed: list[int]   # indices into chain.check_sequence
    round_introduced: int
    round_resolved: int | None = None

class Observation(BaseModel):
    """A factual VLM observation anchored to a specific frame."""
    id: str
    round_number: int
    frame: str                       # e.g., "图3"
    description: str
    judgment: Literal["yes", "no", "unclear"]
    confidence: float
    directive_source: str            # which check_step generated this
```

### Hypothesis Lifecycle

```
Round 0: Symptom scan -> match to chains -> create initial hypotheses (all "active")
Round 1: Execute check_sequence[0] for each active chain -> update confidences
         - if_true hit -> increase hypothesis confidence
         - if_false hit -> decrease or eliminate
Round 2: Execute next check step for surviving hypotheses
         - Also: cross-check between hypotheses (if H1 confirmed, does it explain H2's symptoms?)
Round N: Convergence reached -> final hypothesis is the root cause
```

### Cross-Hypothesis Reasoning

The knowledge graph's causal edges enable a critical optimization: if hypothesis A (e.g., "arm disconnection") is confirmed, check whether it's an upstream cause of hypothesis B (e.g., "scooping"). If A causes B via the graph, then B is a downstream symptom, not an independent root cause. Eliminate B, report it as a downstream symptom in the root cause tree.

This maps directly to `KnowledgeGraph.get_causal_chain()`.

**Confidence: HIGH** -- The hypothesis model is straightforward. The cross-hypothesis reasoning via causal chains is the key differentiator from a simple sequential checklist.

---

## 5. Handling VLM Hallucinations and Wrong Observations

### The Problem

VLMs can:
1. See things that aren't there (false positive observations)
2. Miss things that are there (false negative observations)
3. Be overconfident about ambiguous observations
4. Contradict themselves across rounds

### Mitigation Strategies

**Strategy 1: Confidence-Weighted Observations**
Every observation must include a confidence score. Low-confidence observations (< 0.5) trigger re-observation in the next round with a more specific prompt.

**Strategy 2: Contradiction Detection**
If Round 2 observation contradicts Round 1 on the same visual feature, flag the feature as "ambiguous" and:
- Ask a third time with an even more specific prompt (e.g., zoomed crop description)
- Fall back to quantitative keypoint data if available (e.g., elbow angle from YOLO)

**Strategy 3: Quantitative Cross-Validation**
The system already has YOLO-based kinematic data (elbow angles, wrist trajectories, etc.). Use these as ground truth checks:
- VLM says "elbow angle is wide" but YOLO measures 85 degrees -> VLM is wrong, override
- VLM says "wrist drops sharply" but wrist trajectory shows smooth curve -> VLM is wrong, override

This is the existing `supplementary_metrics` already passed to the VLM. In v2.0, inject specific metrics as verification data in later rounds.

**Strategy 4: Observation Anchoring**
Require every observation to reference a specific frame number and visual feature (trajectory color, annotation). Unanchored observations ("the swing looks rushed") are discarded or flagged.

**Confidence: MEDIUM** -- Strategies 1-2 are straightforward. Strategy 3 requires mapping between kinematic metrics and diagnostic checks (partially exists). Strategy 4 is prompt engineering.

---

## 6. Changes to Existing Code

### evaluation/vlm_analyzer.py

| Change | What | Why |
|---|---|---|
| New class | `MultiRoundAnalyzer` | Orchestrates the multi-round loop; wraps `VLMForehandAnalyzer._call_vlm` |
| New method | `VLMForehandAnalyzer.analyze_swing_iterative()` | Entry point for v2.0 analysis; falls back to `analyze_swing()` on failure |
| Modify | `_analyze_two_pass()` | Add option to continue into multi-round after Pass 1 |
| New | `_execute_diagnostic_round()` | Single round: compile directive prompt -> call VLM -> parse observations -> update hypotheses |
| Keep | `_call_vlm()`, `_parse_symptom_response()` | Unchanged; reused by multi-round |

### knowledge/output/vlm_prompt.py (VLMPromptCompiler)

| Change | What | Why |
|---|---|---|
| New method | `compile_observation_directive()` | Generate a targeted observation prompt for a specific round |
| New method | `compile_confirmation_prompt()` | Generate the final confirmation prompt |
| Modify | `compile_pass2_prompt()` | Keep for single-pass fallback; multi-round replaces it |
| New | `generate_directives_for_hypotheses()` | Convert active hypotheses + unchecked steps into VLM directives |

### knowledge/templates/vlm/ (new templates)

| Template | Purpose |
|---|---|
| `observation_directive.j2` | Round N observation prompt with hypothesis context |
| `confirmation.j2` | Final round: summarize all evidence and generate root_cause_tree |
| `hypothesis_summary.j2` | Compact hypothesis status for injection into prompts |

### knowledge/schemas.py

| Change | What | Why |
|---|---|---|
| New model | `Hypothesis` | Track hypothesis state across rounds |
| New model | `Observation` | Track VLM observations with frame anchoring |
| New model | `DiagnosticSession` | Container for the full multi-round session state |
| New model | `RoundResult` | Single round's prompt/response/observations |

### report/report_generator.py

| Change | What | Why |
|---|---|---|
| New method | `_format_diagnostic_journey()` | Show the iterative reasoning process (not just the result) |
| Modify | `_vlm_section()` | Detect multi-round results and delegate to new formatter |
| New | Reasoning trace rendering | Show which hypotheses were tested, eliminated, and why |

**Confidence: HIGH** -- These changes are well-scoped and additive. No existing functionality is broken.

---

## 7. Token/Cost Analysis

### Current v1.0 Cost Per Swing

| Pass | Input Tokens | Output Tokens | Cost (Gemini 2.0 Flash) |
|---|---|---|---|
| Pass 1 (symptom scan) | ~560 (image) + ~2K (prompt) | ~200 | ~$0.0003 |
| Pass 2 (deep analysis) | ~560 (image) + ~10K (prompt) | ~2K | ~$0.0015 |
| **Total** | | | **~$0.002** |

### Proposed v2.0 Cost Per Swing (3-round typical)

| Round | Input Tokens | Output Tokens | Cost (Gemini 2.0 Flash) |
|---|---|---|---|
| Round 0 (scan) | ~560 + ~2K | ~200 | ~$0.0003 |
| Round 1 (diagnose) | ~560 + ~3K | ~500 | ~$0.0005 |
| Round 2 (verify) | ~560 + ~2K | ~300 | ~$0.0003 |
| Round 3 (confirm) | ~560 + ~4K | ~2K | ~$0.001 |
| **Total** | | | **~$0.002-0.004** |

### Key Insight: Multi-Round is NOT Proportionally More Expensive

Each intermediate round uses a **smaller, focused prompt** (2-3K chars instead of 10K). Only the final confirmation round needs the full diagnostic context. Total token usage is ~1.5-2x the current two-pass, not 3-4x.

The Gemini multi-turn billing note is important: in LiveAPI, tokens accumulate across turns. But since we're using stateless API calls (not LiveAPI sessions), each round is independent -- no token accumulation penalty.

**Confidence: HIGH** -- Based on current Gemini Flash pricing at $0.10/1M input tokens and ~$0.40/1M output tokens.

---

## 8. Presenting Iterative Reasoning in Reports

### The Goal: Show the Journey, Not Just the Destination

A real coach doesn't just say "your root cause is X." They walk you through how they figured it out:

> "First I noticed your wrist dropping sharply in frame 3. That made me suspect scooping. But then I looked more closely and saw your elbow was also tight against your body -- so the real issue isn't the wrist, it's the lack of elbow space forcing the wrist down. When I checked your unit turn, I confirmed the arm-body disconnection started there."

### Report Structure

```markdown
### 诊断推理过程

**第1轮 - 初步观察:**
- 发现: 黄色轨迹在图2-3有V形尖角 -> 怀疑scooping
- 发现: 肘角在图4偏小(~85度) -> 怀疑肘部空间不足
- 初步假设: [scooping] [肘部空间不足] [手臂主导]

**第2轮 - 针对性观察:**
- 检查: 图2-3中拍头下落是被动还是主动?
  -> 观察: 主动下压(V形底部在前半段) -> scooping确认为症状
- 检查: 图1中腋下空间?
  -> 观察: 肘部紧贴身体 -> 肘部空间不足确认
- 假设更新: scooping可能是肘部空间不足的下游

**第3轮 - 根因确认:**
- 检查: 图1 Unit Turn时手臂是否和躯干一体?
  -> 观察: 手臂在身体后方,未贴住 -> 手臂脱离身体旋转系统
- 因果链确认: 手臂脱离 -> 肘部空间不足 -> scooping

### 根因诊断

> **手臂未在Unit Turn时建立与躯干的连接**
[... existing root_cause_tree format ...]
```

### Implementation

The `DiagnosticSession` stores the full round history. The report generator renders it as a narrative "diagnostic journey" section before the final root cause tree. This section is optional (controlled by a flag) for users who just want the result.

**Confidence: HIGH** -- This is the highest-impact UX improvement. It transforms the report from "here's what's wrong" to "here's how I figured out what's wrong" -- which is exactly how coaching works.

---

## Architecture Summary

```
                    +-------------------+
                    |   analyze_swing   |
                    | _iterative()      |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Round 0: Scan    |  <- symptom_checklist.j2 (existing)
                    |  -> hypotheses    |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Round N: Observe  | <- observation_directive.j2 (new)
                    |  -> observations   |
                    |  -> update hyps    |
                    +--------+----------+
                             |
                        converged?
                       /          \
                     no            yes
                      |             |
               loop back      +----v-----------+
                              | Final: Confirm  | <- confirmation.j2 (new)
                              | -> root_cause   |
                              |    _tree JSON   |
                              +----+------------+
                                   |
                              +----v-----------+
                              | Report with    |
                              | reasoning trace|
                              +----------------+
```

---

## Pitfalls

### Critical

1. **Prompt Drift Across Rounds**: If hypothesis context grows too large, later rounds may exceed the prompt budget. Mitigation: only include active hypotheses and their most relevant observations, not full history.

2. **VLM Inconsistency**: The VLM may contradict its own observations across rounds. Mitigation: explicit contradiction detection + quantitative cross-validation from kinematic data.

3. **Infinite Loop Risk**: Poor convergence criteria could cause the loop to never exit. Mitigation: hard max of 4 rounds + convergence score tracking.

### Moderate

4. **Over-Engineering the State Machine**: The hypothesis tracking system could become overly complex. Keep it simple: hypotheses are just tagged with active/confirmed/eliminated + confidence float.

5. **False Convergence**: The system may converge on a wrong root cause if early observations are wrong. Mitigation: the confirmation round re-checks the final hypothesis against ALL available evidence.

### Minor

6. **Report Verbosity**: The diagnostic journey section could be too long. Make it collapsible/optional.

7. **Latency**: 3-4 sequential VLM calls add ~10-15 seconds total. Acceptable for batch analysis, not for real-time.

---

## Sources

- [Think Twice to See More: Iterative Visual Reasoning in Medical VLMs (ViTAR)](https://arxiv.org/html/2510.10052v1)
- [Gemini 3 Flash Agentic Vision Guide](https://gemilab.net/en/articles/gemini-api/gemini-3-flash-agentic-vision-guide)
- [Language Agents for Hypothesis-driven Clinical Decision Making](https://arxiv.org/abs/2506.13474)
- [Gemini API Pricing 2026](https://ai.google.dev/gemini-api/docs/pricing)
- [Build multi-turn conversations using the Gemini API](https://firebase.google.com/docs/ai-logic/chat)
- [Improve Vision Language Model Chain-of-thought Reasoning](https://aclanthology.org/2025.acl-long.82/)
- [Interleaved-Modal Chain-of-Thought (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Interleaved-Modal_Chain-of-Thought_CVPR_2025_paper.pdf)

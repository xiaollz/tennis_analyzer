# Diagnosis Engine HSA Integration Plan

## Executive Summary

The diagnosis engine (evaluation/diagnosis_engine.py) is a causal inference system that maps VLM observations → tennis forehand concepts → root causes → drill prescriptions. HSA (Horizontal Shoulder Adduction) has been integrated at the **VLM prompt level** (Q39, Q40) and **foundation layer** (F7), but it lacks full activation in the diagnosis engine's **concept taxonomy** and **root-cause hierarchy**. 

This plan maps the integration points needed to make HSA findings flow through the diagnosis engine's causal reasoning, enabling the system to "act like a digital FTT coach" that proactively diagnoses HSA failures and prescribes stage-appropriate drills.

---

## Current State of diagnosis_engine.py Architecture

### 1.1 Core Data Structures

The diagnosis engine operates via these key dictionaries:

#### **OBSERVATION_TO_CONCEPT** (lines 35–377)
- Maps VLM keywords (e.g., "大臂没有跨过胸前") → matched concept ID + severity + frame range
- 140+ keyword patterns covering L1–L5 and prep concepts
- **Current HSA-related keywords** (lines 245–251 in foundation_layer.py F7 definition):
  - Fail keywords: "大臂没有跨过胸前", "胸肌未参与", "右臂保持外展", "随挥往侧方", "无内收", "肘卡身侧", "推球", "纯靠手腕翻拍", "纯靠转体", "胸肱角未关闭"
  - Not yet in OBSERVATION_TO_CONCEPT directly — only in foundation_layer.py logic

#### **_CONCEPT_LAYER** (lines 386–470)
- Maps concept ID → diagnostic layer (L1=Contact, L2=Rhythm, L3=Chain, L4=Prep, L5=Footwork)
- **Layer order matters**: L5 → L4 → L3 → L2 → L1 (earliest layer wins as root cause)
- **Critically missing**: HSA concepts have no layer assignment
  - HSA is not a simple layer — it's a **primary engine failure** that sits BELOW F5 (right foot axis) + F6 (scapular slot) in the dependency graph
  - But HSA is ABOVE many forearm/wrist concepts in causality

#### **_CONCEPT_TO_FIX** (lines 624–930)
- Maps concept ID → {drill, method, why, muscle_cue}
- **100+ drills** for existing concepts (e.g., "unit_turn", "problem_p03", "straight_legs")
- **Currently no HSA-specific drills** in this map
  - Should link to stages 0–4 from docs/research/hsa_training_drills_master.md

#### **_CONCEPT_TO_METRIC_VALIDATION** (lines 569–617)
- Maps concept ID → list of quantitative metrics that confirm/contradict the diagnosis
- Example: "problem_p02" (V-shape scooping) checks scooping_depth > 0.3
- **Currently missing**: no validation rules for HSA patterns
  - Should check: hsa_total_closure_deg, hsa_angle_at_contact, hsa_closure_pattern, cross_body_finish

### 1.2 Diagnostic Flow (diagnose() function, line ~1800+)

1. **Observation → Concept Mapping**: VLM text keywords match OBSERVATION_TO_CONCEPT rules
2. **Concept → Layer Ranking**: matched concepts sorted by _LAYER_ORDER (L5–L1)
3. **Earliest Layer Problem**: root-cause winner determined by earliest layer with problems
4. **Metric Validation**: quantitative data confirms/contradicts the root cause
5. **Drill Prescription**: _CONCEPT_TO_FIX lookup retrieves training guidance
6. **Narrative Generation**: diagnosis humanized to Chinese natural language

### 1.3 VLM Prompt Structure

**Q39–Q40** (foundation_layer.py F7, lines 229–277):
- Q39: "比较引拍顶点和随挥末端，持拍大臂角度是否闭合？估算闭合幅度；判断主要发生在击球前？F7 PASS/FAIL"
- Q40: "随挥末端拍头是否越过左肩？描述拍头终点位置；给 F7 跨胸 PASS/FAIL"
- Q39–Q40 are **Foundation-critical** (标 🏛️) but are **observer questions**, not root-cause inference

### 1.4 Foundation Layer Integration (foundation_layer.py F7, lines 220–277)

- F7 is defined as priority=1 (after FTT F1–F4, alongside F5/F6)
- Expects pass_criteria on 4 metrics: hsa_total_closure_deg ≥ 25°, hsa_angle_at_contact 45°–80°, cross_body_finish==True, hsa_health_score ≥ 60
- Downstream cascade documented (chest not firing → ISR disabled → ball soft, no penetration)
- drill property points to FTT Tier S drills, but no code-level linkage to _CONCEPT_TO_FIX

---

## Proposed Concept Additions

### 2.1 HSA Failure Mode Concepts

Add 6 new concept entries to OBSERVATION_TO_CONCEPT (following existing pattern):

```python
# ── HSA (Horizontal Shoulder Adduction) — 5/3 突破 ──
# 数据源：evaluation/hsa_detector.py, foundation_layer.py F7, 
#        docs/research/hsa_master_index.md

{"keywords": ["大臂没有跨过胸前", "胸肌未参与", "右臂保持外展", 
              "无内收", "hsa closure未发生", "大臂始终外展"],
 "concept": "hsa_no_closure", "frame_range": None,
 "severity": 0.95, "label": "HSA无闭合（大臂全程外展）"},

{"keywords": ["闭合过晚", "随挥才闭合", "接触后才内收", "hit后 hsa", 
              "post-contact closure"],
 "concept": "hsa_late_closure", "frame_range": [4, 5],
 "severity": 0.85, "label": "HSA闭合过晚（接触后）"},

{"keywords": ["闭合过早", "肘卡身侧", "接触前角度已小于50", "early closure"],
 "concept": "hsa_early_closure", "frame_range": [3, 4],
 "severity": 0.8, "label": "HSA闭合过早（肘卡身侧推球）"},

{"keywords": ["静态闭合", "整个 swing 角度变化小于10", "几乎没动", 
              "static hsa", "no closure motion"],
 "concept": "hsa_static", "frame_range": None,
 "severity": 0.9, "label": "HSA静态（纯靠转体无内收驱动）"},

{"keywords": ["随挥未越过身体", "拍头未交叉", "随挥停在右侧", 
              "cross body finish missing", "没有越过左肩", "不完整跨胸"],
 "concept": "hsa_insufficient_cross_body", "frame_range": [5, 6],
 "severity": 0.75, "label": "HSA跨胸不足（随挥未完成）"},

{"keywords": ["大臂住肩窝", "胸肌全程参与", "hsa healthy", "闭合幅度充分",
              "角度健康变化", "cross body clean"],
 "concept": "hsa_healthy", "frame_range": None,
 "severity": 0.0, "label": "HSA健康（闭合充分且时序正确）"},
```

### 2.2 HSA Layer Assignment

Add to _CONCEPT_LAYER:

```python
# HSA — 驱动引擎（依赖 F5 F6，影响手臂补偿）
"hsa_no_closure": "L3",        # Primary engine failure in kinetic chain
"hsa_late_closure": "L3",      # Timing error in chain transmission
"hsa_early_closure": "L3",     # Constraint-side failure
"hsa_static": "L3",             # Pure rotation substitution (no HSA component)
"hsa_insufficient_cross_body": "L3",  # Follow-through incompleteness
"hsa_healthy": "L3",            # Diagnostic label (severity 0)
```

**Rationale**: HSA sits in **L3 (Kinetic Chain)** because:
- It's a mechanical failure in power transmission (chest-driven humerus adduction)
- It's a **primary problem** (not compensation) when detected
- But it has **upstream dependencies** (F5: right foot axis must exist first; F6: scapular slot must be set)
- When HSA fails, downstream compensation appears: forearm_arming, elbow_drop, wrist_snap, ball_softness

---

## Proposed Causal Relationships

### 3.1 Upstream Dependencies (HSA requires these to succeed)

**F5 (Right Foot Axis) → HSA**
- Causal: Right foot as rotational pivot generates the torque that drives HSA closure
- Metric check: weight_transfer_timing, pivot_quality must pass F5 PASS before HSA can trigger
- If F5 fails (early weight transfer, no pivot): HSA closure becomes impossible (no torque source)

**F6 (Scapular Slot) → HSA**
- Causal: Scapular stabilization provides the fixed point (fixed point) for pectoralis major to pull
- Metric check: elbow_height_relative_chest, arm_chest_gap_change must pass F6 PASS
- If F6 fails (elbow drops, slot departs): HSA generates no force (pec pulling on unstable base)

### 3.2 Downstream Consequences (when HSA fails)

**HSA Failure → Forearm Compensation**
- Causal chain: "If chest doesn't fire (HSA fails), hand must independently pronate to roll the ball" → forearm_arming (problem_p03)
- Evidence: When hsa_closure_pattern ∈ {no_closure, static, late_closure}, problem_p03 severity increases
- Example keyword collision: "纯靠手腕翻拍" matches both HSA_fail and problem_p03; HSA is upstream cause

**HSA Failure → Ball Softness / Loss of Penetration**
- Causal chain: HSA closure contributes 45–48% forward RHS (Sasaki 2022); when missing, ball velocity drops
- Evidence: hsa_health_score < 40 correlates with forward_extension < 0.2
- Metric linkage: HSA total_closure_deg < 15° → confirm ball softness diagnosis

**HSA Failure → Backward Lean / Falling Away**
- Causal chain: Without chest-driven closure, torso rotates "off the ball" (overrotation); ISR unloads
- Evidence: HSA no_closure + problem_p11 (overrotation) often co-occur
- Counter: Left-hand braking (problem_p11 drill) is compensation; real fix is HSA.

**HSA Failure → Elbow Drop / Space Collapse**
- Causal chain: If HSA doesn't fire, elbow naturally drops (gravity) → arm space compresses
- Evidence: hsa_angle_at_contact > 90° (external rotation) correlates with L1_contact_too_close
- Metric linkage: HSA failure → re-check F6 (slot may be departing)

### 3.3 Proposed Knowledge Graph Edges

In the KnowledgeGraph (knowledge/graph.py), add:

```python
# Upstream dependencies
Edge(F5_right_foot_axis → hsa_no_closure, relation="enables", strength=0.9)
Edge(F6_scapular_slot → hsa_no_closure, relation="enables", strength=0.9)

# Downstream consequences (causal)
Edge(hsa_no_closure → problem_p03, relation="causes", strength=0.7)  # forearm compensation
Edge(hsa_late_closure → problem_p03, relation="causes", strength=0.6)
Edge(hsa_no_closure → forward_softness, relation="causes", strength=0.8)  # new symptom concept
Edge(hsa_no_closure → problem_p11, relation="causes_via_compensation", strength=0.5)  # overrotation as backup

# Visibility
Edge(hsa_no_closure → Q39_answer="no closure", relation="visible_as", strength=0.95)
Edge(hsa_no_closure → Q40_answer="racket stops at right side", relation="visible_as", strength=0.85)
```

---

## Proposed Root-Cause Hierarchy Update

### 4.1 Current Hierarchy (from _LAYER_ORDER)

```
L5 (Footwork: split step, pivot, stance) ← Most upstream
 ↓
L4 (Preparation: unit turn, scapular glide, left hand)
 ↓
L3 (Kinetic Chain: arm-body sync, back tension, hip-shoulder separation)
 ↓
L2 (Rhythm: timing, bounce-hit synchrony)
 ↓
L1 (Contact: strike point, racket face angle) ← Most downstream (symptoms)
```

### 4.2 Refined Hierarchy with HSA

**Insert HSA into the causal chain**:

```
L5 (Footwork)
 ↓
L4 (Preparation)
 ├─→ F5 Right Foot Axis (user's 4/27 breakthrough)
 ├─→ F6 Scapular Slot (user's 4/30 breakthrough)
 └─→ Unit Turn + Hold Up + Place Pull (traditional FTT)
 ↓
L3 (Kinetic Chain)
 ├─→ F7 Horizontal Shoulder Adduction ← **PRIMARY ENGINE** (5/3 breakthrough)
 │    (depends on F5 + F6 being solid)
 ├─→ Arm-Body Sync (problem_p03)
 ├─→ Upper Body Only Turn (prep15)
 └─→ Back Tension (problem_p10)
 ↓
L2 (Rhythm)
 ├─→ Bounce-Hit Timing
 └─→ Acceleration Phase
 ↓
L1 (Contact)
 └─→ Strike Point Geometry (contact too close, too low, etc.)
```

**Key insight**: 
- **HSA is NOT a Layer-2 or Layer-1 issue** (not just timing or contact point)
- **HSA IS a Layer-3 core engine issue** (kinetic chain power transmission)
- **But HSA requires F5 + F6 to function** — if either fails, HSA diagnosis is suspended or marked "CANNOT_ASSESS"

### 4.3 Diagnostic Decision Logic (pseudocode)

```python
def diagnose_forehand(vlm_result, metrics):
    # Check foundations first
    f5_status = check_F5_right_foot_axis(vlm_result, metrics)
    f6_status = check_F6_scapular_slot(vlm_result, metrics)
    
    # If F5 or F6 fail, HSA check returns CANNOT_ASSESS
    if f5_status == FAIL or f6_status == FAIL:
        hsa_diagnosis = {
            "status": "CANNOT_ASSESS",
            "reason": "Right foot axis or scapular slot not established. HSA cannot function.",
            "action": "Fix F5/F6 first (foundation priority)"
        }
    else:
        # F5 + F6 pass — now diagnose HSA
        hsa_closure_deg = metrics.get("hsa_total_closure_deg")
        hsa_pattern = metrics.get("hsa_closure_pattern")
        
        if hsa_pattern == "healthy":
            # HSA is healthy — check downstream for compensation patterns
            root_cause = find_earliest_non_hsa_problem([...matched_concepts...])
        else:
            # HSA fails — this is root cause
            root_cause = hsa_pattern  # (no_closure, late_closure, early_closure, etc.)
            # Suppress downstream forearm concepts as compensations
            suppress_diagnoses(["problem_p03", "problem_p13", "wrist_snap"])
    
    return root_cause, drill_prescription
```

---

## Proposed VLM Prompt Upgrade for "Digital FTT Coach"

### 5.1 Current State (Q39–Q40, lines 134–135 of system_prompt.md.j2)

Q39 and Q40 are **observer questions** asking the VLM to report what it sees, with F7 PASS/FAIL tags. But they are **passive** — they require the human to understand FTT concepts to ask them.

A "digital FTT coach" should **proactively interrogate HSA** from video, even without explicit questions.

### 5.2 Proposed VLM System Prompt Sections

Insert after the F7 definition in system_prompt.md.j2:

```markdown
================================================================
🏛️ F7 DETAILED INTERROGATION PROTOCOL — HSA Detection in Depth
================================================================

When answering Q39–Q40, use the following structured protocol to
identify all 5 HSA failure modes. This protocol runs on every video,
regardless of whether the user asked about HSA.

**Protocol: 3-Frame Comparison Analysis**

Step 1: Identify Three Key Frames
  - Frame A: "Backswing Peak" = moment when unit turn is complete,
    before forward swing starts. Shoulder rotation maximal.
  - Frame B: "Contact" = frame where racket contacts ball (or would,
    in shadow swing).
  - Frame C: "Follow-Through End" = last frame where hitting arm
    still visible and moving.

Step 2: Measure Humerus Angle in Frame A vs. Frame B vs. Frame C
  - Definition: "Humerus angle" = angle between (upper arm vector
    from shoulder→elbow) and (torso horizontal line from left shoulder→right shoulder).
  
  - Typical ranges:
    * Backswing Peak (Frame A): ~90–110° (external rotation/abduction)
    * Contact (Frame B): ~50–80° (adduction in progress)
    * Follow-Through (Frame C): ~30–50° (maximal adduction/cross-body)
  
  - If you cannot measure due to arm occlusion: write "OCCLUDED [description]"

Step 3: Calculate Two Closure Metrics
  - Metric 1: "Total Closure" = Angle(A) - Angle(B)
    * >40° = Strong closure (healthy)
    * 20–40° = Moderate closure (OK)
    * <20° = Weak closure (problem)
  
  - Metric 2: "Post-Contact Closure" = Angle(B) - Angle(C)
    * Should be >0 (arm continues closing after contact)
    * If <0 or close to 0 = arm is opening after hit (wrong)

Step 4: Classify HSA Pattern (Pick ONE)

  | Pattern | Definition | Key Signal |
  |---------|-----------|-----------|
  | **healthy** | Total ≥25°, at contact 45–80°, post-contact >0, cross-body YES | Ball has penetration; no arming visible |
  | **no_closure** | Total <20°, angles A≈B≈C (flat line) | Arm never changes angle. Looks like "winding up" |
  | **late_closure** | Total mostly happens after contact (B→C much bigger than A→B) | Arm doesn't fire until after ball is struck |
  | **early_closure** | At contact, angle <50° (arm too closed) | Elbow is pinned to ribs; pushing motion |
  | **static** | Total <10° across whole swing; pure rotation doing the work | Arm barely moves; hips/shoulders doing all work |
  | **insufficient_cross_body** | Angle C is still >50° (arm hasn't crossed midline) | Racket ends on right side, never reaches left shoulder |

Step 5: Answer Q39–Q40 with Machine-Parseable Format

For Q39 (HSA Closure Visible?):
```
【Frame A humerus angle】= [your estimate]°
【Frame B humerus angle】= [your estimate]°
【Frame C humerus angle】= [your estimate]°
【Total closure (A→B)】= [calculated]°
【Post-contact closure (B→C)】= [calculated]°
【Closure pattern】= [healthy | no_closure | late_closure | early_closure | static | insufficient_cross_body]
【Closure timing】= [90% before contact | evenly split | 90% after contact]
F7-HSA-CLOSURE: [PASS | FAIL | UNCERTAIN: <reason>]
```

For Q40 (Cross-Body Finish?):
```
【Frame C racket head position】= [right hip | center chest | left shoulder | left ear] / height [waist | chest | face | overhead]
【Racket crosses left shoulder?】= [YES / NO / PARTIALLY]
【Is this motion natural (passive from HSA) or active flipping?】= [passive | active | mixed]
F7-HSA-CROSS-BODY: [PASS | FAIL | UNCERTAIN: <reason>]
```

================================================================
ANTI-AMBIGUITY RULES for HSA Detection
================================================================

1. **Arm Occlusion**: If the upper arm disappears behind torso, note the
   last visible angle and mark "OCCLUDED". Do NOT guess the hidden angle.

2. **Optical Illusion Avoidance**:
   - Sometimes the arm looks like it's "staying out" but the torso is
     rotating under it. Measure angle change, not absolute position.
   - Sometimes camera angle makes adduction look smaller than it is.
     Report what you see; measurement > impression.

3. **Pronation ≠ Adduction**:
   - Forearm rotation (pronation/supination) is NOT HSA.
   - HSA = humerus adduction (upper arm closing toward chest).
   - If you see "wrist flip" but no humerus angle change → not HSA,
     it's arming/pronation (problem_p03 territory).

4. **Rotation ≠ HSA**:
   - Torso rotation alone does NOT count as HSA.
   - Example: If torso rotates 60° but arm angle stays 90° the whole time
     → that is "static" HSA (pure rotation substitution).

5. **Contact Timing Clarity**:
   - You must clearly identify the contact frame (or the frame where contact
     WOULD occur in slow motion). Some forehands have ball-on-strings
     contact visible; others are silhouette only.
   - If unsure: "Contact frame unclear; using [description] as proxy."

================================================================
```

### 5.3 Integration with diagnosis_engine.py

After the VLM returns Q39–Q40 answers with these tags, diagnosis_engine.py parses:

```python
def parse_vlm_hsa_findings(vlm_result: Dict) -> Dict:
    """Extract F7-HSA-CLOSURE and F7-HSA-CROSS-BODY tags."""
    
    hsa_findings = {
        "f7_closure_pattern": None,
        "hsa_total_closure_deg": None,
        "hsa_post_contact_closure_deg": None,
        "cross_body_finish": None,
        "vlm_confidence": "uncertain",
    }
    
    # Parse F7-HSA-CLOSURE tag
    if "F7-HSA-CLOSURE: PASS" in vlm_result.get("Q39", ""):
        hsa_findings["vlm_confidence"] = "pass"
    elif "F7-HSA-CLOSURE: FAIL" in vlm_result.get("Q39", ""):
        hsa_findings["vlm_confidence"] = "fail"
    
    # Extract numeric angle estimates
    q39 = vlm_result.get("Q39", "")
    match_closure_total = re.search(r"【Total closure.*?】= (\d+\.?\d*)", q39)
    if match_closure_total:
        hsa_findings["hsa_total_closure_deg"] = float(match_closure_total.group(1))
    
    # Extract pattern classification
    patterns = ["healthy", "no_closure", "late_closure", "early_closure", "static", "insufficient_cross_body"]
    for pattern in patterns:
        if pattern in q39:
            hsa_findings["f7_closure_pattern"] = pattern
            break
    
    # Cross-body finish
    if "F7-HSA-CROSS-BODY: PASS" in vlm_result.get("Q40", ""):
        hsa_findings["cross_body_finish"] = True
    elif "F7-HSA-CROSS-BODY: FAIL" in vlm_result.get("Q40", ""):
        hsa_findings["cross_body_finish"] = False
    
    return hsa_findings
```

Then these findings feed into the diagnosis:

```python
# In diagnose() function
hsa_vlm_findings = parse_vlm_hsa_findings(vlm_result)
hsa_detector_metrics = metrics.get("hsa", {})

# Merge VLM observation + quantitative metrics
hsa_diagnosis = {
    "vlm_pattern": hsa_vlm_findings.get("f7_closure_pattern"),
    "quantitative_pattern": hsa_detector_metrics.get("hsa_closure_pattern"),
    "hsa_health_score": hsa_detector_metrics.get("hsa_health_score"),
    "cross_body_finish_vlm": hsa_vlm_findings.get("cross_body_finish"),
    "cross_body_finish_measured": hsa_detector_metrics.get("cross_body_finish"),
}

# If both VLM and metrics agree → high confidence
# If they disagree → mark UNCERTAIN, report both
```

---

## Proposed Drills Mapping for HSA Failure Modes

### 6.1 Drill Matrix

Based on docs/research/hsa_training_drills_master.md stages and failure modes:

| **HSA Pattern** | **Root Cause** | **Stage** | **Primary Drill** | **Alternative** | **Cue** |
|---|---|---|---|---|---|
| **no_closure** | Arm never internally rotates; pure external rotation or lat motion | 0 → 1 | 0-A: Hand on pec + side pull (FTT Am8j1Zw5KrE) | 1-A: Static no-rotation ball | "Feel pec fire while arm barely moves" |
| **static** | Torso rotates but arm stays planted; HSA component missing | 0 → 1 → 2 | 1-A: Static no-turn hitting (FTT 5KdScDKxVSI [03:40]) | 2-A: Med ball side slam (SSC integration) | "Arm angle closes even without rotating body" |
| **early_closure** | Arm closes before contact; elbow pins to ribs → pushing motion | 1 → 2 | 1-C: Half-court short swing (forces extend) | 3-B: Flip drill (Gordon [11:10]) | "Delay elbow until after contact point" |
| **late_closure** | Arm still opening at contact; HSA happens post-hit | 1 → 2 → 3 | 2-A: Med ball slam (trunk slowdown trigger) | 3-A: Shadow swing with drop (Macci) | "Chest must drive at contact, not after" |
| **insufficient_cross_body** | Follow-through arm doesn't cross midline; incomplete pronation | 2 → 3 → 4 | 3-C: Off-arm pull (left hand braking forces cross) | 3-D: Slow weighted swing (kinematics tracking) | "Let arm momentum carry across; don't stop at center" |
| **healthy** | No drill needed; use HSA as baseline control | 4 | 4-A: Match speed segments (maintain HSA across tempos) | 4-C: Match simulation + video review | "Monitor hsa_health_score ≥ 70 in variety of conditions" |

### 6.2 Drill Prescription Logic

```python
def prescribe_hsa_drills(hsa_closure_pattern: str) -> List[Dict]:
    """Return stage-appropriate drill sequence for each HSA failure mode."""
    
    drills_by_pattern = {
        "no_closure": [
            {"stage": "0", "drill": "0-A: Hand on pec + horizontal pull",
             "source": "FTT Am8j1Zw5KrE [00:00-00:30]",
             "duration": "5 min × 3-5 sets", "metric_target": "Palpate pec contraction"},
            {"stage": "1", "drill": "1-A: Static no-rotation hitting",
             "source": "FTT 5KdScDKxVSI [03:40]",
             "duration": "30-50 balls", "metric_target": "Decent ball without body rotation"},
        ],
        "late_closure": [
            {"stage": "1", "drill": "1-A: Static no-rotation (build HSA isolation)",
             "source": "FTT 5KdScDKxVSI [03:40]"},
            {"stage": "2", "drill": "2-A: Med ball side slam",
             "source": "Kovacs Academy medicine ball drill",
             "duration": "4-6 × 3-4 sets",
             "cue": "Trunk slows → arm accelerates"},
            {"stage": "3", "drill": "3-A: Shadow swing with drop + timing",
             "source": "Macci zac_u3TxxDo [10:20]",
             "cue": "Chest drives AT contact, not after"},
        ],
        # ... other patterns ...
    }
    
    if hsa_closure_pattern not in drills_by_pattern:
        return []  # Unknown pattern
    
    drills = drills_by_pattern[hsa_closure_pattern]
    
    # Filter to stage user is ready for (based on history)
    current_stage = get_user_hsa_stage()
    return [d for d in drills if stage_number(d["stage"]) >= current_stage]
```

### 6.3 Integration with _CONCEPT_TO_FIX

Add HSA drill entries (following existing format):

```python
_CONCEPT_TO_FIX = {
    # ... existing concepts ...
    
    # HSA — 5/3 新概念
    "hsa_no_closure": {
        "drill": "手按胸肌横拉空挥 + 静态无转体击球",
        "method": "第1-2天：左手按住右胸大肌（锁骨头+胸肋交界）→ 右手做横拉空挥"
                  "→ 直到能触摸到胸肌收缩。"
                  "第3-7天：站稳不转体，纯用 HSA 击球 30-50 球，目标是没转体也能打出 decent ball。",
        "why": "大臂从未跨过胸前 = 胸肌完全没参与。HSA 是胸肌的'内收拉动'，不是转体的结果。"
               "先建立胸肌触诊体感，再加击球节奏。",
        "muscle_cue": "手按下去能感觉到胸大肌的充血和硬度。击球时胸部（不是手臂）应该有'往前推'的感觉。"
                      "如果前臂酸，说明胸肌没启动，前臂在代偿。",
        "duration": "1-2 周",
        "progression": ["stage_0_hsa_tactile", "stage_1_hsa_isolated"],
        "video_refs": ["FTT_Am8j1Zw5KrE_00-00-30", "FTT_5KdScDKxVSI_03-40"],
    },
    
    "hsa_late_closure": {
        "drill": "药球侧砸墙 + 翻转 Drill",
        "method": "药球重 2-3 kg，开放站位，胸前举起 → 转身蓄力 → 爆发砸墙（触发躯干减速）"
                  "→ 感受爆发回弹。"
                  "同时做'翻转 Drill'：引拍末端故意停顿 → 启动髋部 → 拍头自动 drop + 翻转"
                  "→ 体感胸部驱动时刻。",
        "why": "闭合过晚 = 接触时胸肌还没启动。药球爆发训练教会'减速触发加速'的 SSC 机制。"
               "正手击球时胸肌必须在接触前的 50-100ms 内达到峰值速度。",
        "muscle_cue": "药球砸墙时躯干应该有'先停才有爆发'的感觉。同样，击球时胸部应该在接触点前"
                      "有一个'压紧'的时刻，然后才释放向前。如果感觉是平缓加速，说明没有减速触发。",
        "duration": "2-3 周",
        "progression": ["stage_2_ssc_loading", "stage_3_hsa_integration"],
        "video_refs": ["Kovacs_medicine_ball_drill", "Macci_zac_u3TxxDo_11-10"],
    },
    
    "hsa_early_closure": {
        "drill": "半场短挥 + 翻转延迟",
        "method": "半场距离击球，引拍极短，强迫胸部把球'推出去'（没空间用大臂）。"
                  "同时意识：不要在接触前关闭肘部，让肘部自由延展到接触点，"
                  "只在接触后才让胸肌内收。",
        "why": "闭合过早 = 肘部在接触前就贴到身侧（卡肘）→ 只能推球。短挥逼迫胸肌启动，"
               "而肘部空间强制拉开。接触后闭合是自然的 follow-through，不是发力位置。",
        "muscle_cue": "击球时肘部应该有'抵抗被拉向身体'的感觉（前锯肌+胸小肌抑制）。"
                      "如果肘部贴胸，说明胸肌没有足够的张力来拉开肘部空间。",
        "duration": "2-3 周",
        "progression": ["stage_1_hsa_isolated", "stage_3_hsa_integration"],
        "video_refs": ["FTT_xf93E0Ja0Lk", "Gordon_o2Cqwa5bxV0_26-30"],
    },
    
    "hsa_static": {
        "drill": "静态无转体击球 → 加入转体",
        "method": "第 1-3 天：完全不转体，纯 HSA 击球 30 球（确保 HSA 机制独立于转体）。"
                  "第 4-7 天：在这个 HSA 基础上加入 Unit Turn，感受 HSA 叠加在转体之上（不是被转体替代）。",
        "why": "静态闭合 = 大臂始终外展，转体在做所有工作。这意味着胸肌完全没启动。"
               "先隔离 HSA，再整合转体。顺序反了就会回到纯转体模式（static 永远改不了）。",
        "muscle_cue": "引拍时胸部应该有'被拉开'的预拉伸感（胸大肌离心）。前挥时胸部应该有'向前压'的感觉（向心）。"
                      "如果全程胸部无感，都是肩膀转，说明 HSA 还没建立。",
        "duration": "2-4 周",
        "progression": ["stage_0_hsa_tactile", "stage_1_hsa_isolated", "stage_3_hsa_integration"],
        "video_refs": ["FTT_5KdScDKxVSI_03-40", "FTT_Am8j1Zw5KrE"],
    },
    
    "hsa_insufficient_cross_body": {
        "drill": "左手主动拉离 + 随挥能量追踪",
        "method": "Unit Turn 时左手主动向后拉（模仿 Fr·éderer + Thiem 动作）"
                  "→ 这个'反向力'强迫右臂（持拍臂）越过身体中线。"
                  "喂球 30 球，每球检查：拍头是否自然越过左肩（不是停在右侧）。",
        "why": "跨胸不足 = 随挥末端 HSA 角度还 > 50°（还没完全闭合跨胸）。"
               "左手是'刹车'，同时也是'对抗力'。左手拉离越强，右臂越容易被反推越过身体。",
        "muscle_cue": "左手拉离时应该感觉到左侧肋下（前锯肌）和腹外斜肌有'对抗'感。"
                      "同时右臂应该被'反推'着越过中线。如果右臂可以停在右侧而不被推过去，"
                      "说明左手拉离没有足够的力。",
        "duration": "2-3 周",
        "progression": ["stage_1_hsa_isolated", "stage_3_hsa_integration", "stage_4_game"],
        "video_refs": ["FTT_standard_offarm_pull", "Federer_follow_through_analysis"],
    },
    
    "hsa_healthy": {
        "drill": "保持基线，变工况测试",
        "method": "不需要修正钻。改为变条件维持训练："
                  "1. 发球机分速段（60-80mph 都保持 HSA）"
                  "2. 变方向喂球（深 / 浅 / 角度都保持 HSA）"
                  "3. 被动回球（救球场景保持 HSA）"
                  "每组后自检：hsa_health_score 是否 ≥70 还是下降。",
        "why": "HSA 一旦建立，需要在变工况下自动化。真实比赛中不是'每次都理想HSA'，"
               "而是'各种情况下都能调出 HSA'。维持 drill 是自动化训练。",
        "muscle_cue": "各种球速 / 来球角度下胸部应该都有相同的'先拉开再压紧'的感觉。"
                      "如果快球时前臂酸、慢球时无感，说明 HSA 还没完全自动化。",
        "duration": "4-8 周（持续）",
        "progression": ["stage_4_game"],
        "video_refs": ["自拍跑 hsa_detector.py 对标"],
    },
}
```

---

## Specific Code Changes Needed (file:line)

### 7.1 evaluation/diagnosis_engine.py

**Addition 1: HSA concepts to OBSERVATION_TO_CONCEPT (after line 377)**

```python
# ── HSA (Horizontal Shoulder Adduction) — 5/3 突破 ──
# See foundation_layer.py F7 + docs/research/hsa_master_index.md
{"keywords": ["大臂没有跨过胸前", "胸肌未参与", "右臂保持外展", "无内收", 
              "hsa closure未发生", "大臂始终外展", "纯靠转体"],
 "concept": "hsa_no_closure", "frame_range": None,
 "severity": 0.95, "label": "HSA无闭合（大臂全程外展）"},

{"keywords": ["闭合过晚", "随挥才闭合", "接触后才内收", "击球后才闭合"],
 "concept": "hsa_late_closure", "frame_range": [4, 5],
 "severity": 0.85, "label": "HSA闭合过晚（接触后才启动）"},

{"keywords": ["闭合过早", "肘卡身侧", "接触前角度已小于50", "推球"],
 "concept": "hsa_early_closure", "frame_range": [3, 4],
 "severity": 0.8, "label": "HSA闭合过早（肘卡身侧推球）"},

{"keywords": ["静态闭合", "整个 swing 角度变化小于10", "几乎没动", "纯靠转体驱动"],
 "concept": "hsa_static", "frame_range": None,
 "severity": 0.9, "label": "HSA静态（转体代替内收）"},

{"keywords": ["随挥未越过身体", "拍头未交叉", "随挥停在右侧", "没有越过左肩"],
 "concept": "hsa_insufficient_cross_body", "frame_range": [5, 6],
 "severity": 0.75, "label": "HSA跨胸不足（随挥末端未越过左肩）"},

{"keywords": ["大臂跨过胸前", "胸肌充分参与", "闭合幅度充分", "随挥越过左肩", "健康闭合"],
 "concept": "hsa_healthy", "frame_range": None,
 "severity": 0.0, "label": "HSA健康（闭合充分时序正确）"},
```

**Addition 2: HSA layer assignments to _CONCEPT_LAYER (after line 470)**

```python
    # HSA — Primary kinetic chain engine (depends on F5 + F6)
    "hsa_no_closure": "L3",
    "hsa_late_closure": "L3",
    "hsa_early_closure": "L3",
    "hsa_static": "L3",
    "hsa_insufficient_cross_body": "L3",
    "hsa_healthy": "L3",
```

**Addition 3: HSA drill mappings to _CONCEPT_TO_FIX (after line 930)**

```python
    # ── HSA Failure Modes (5/3 integration) ──
    "hsa_no_closure": {
        "drill": "手按胸肌横拉空挥 + 静态无转体击球",
        "method": "第 1-2 天: 左手按住右胸大肌锁骨头+胸肋交界处 → 右手横拉空挥 → 直到能触摸胸肌收缩。"
                  "第 3-7 天: 站稳不转体，纯用 HSA 击球 30-50 球，目标没转体也能打 decent ball。",
        "why": "大臂未跨过胸前 = 胸肌完全未参与。HSA 是胸肌'内收拉动'，不是转体结果。"
               "先建立胸肌触诊体感，再加击球节奏。",
        "muscle_cue": "手按下去能感觉胸大肌充血硬度。击球时胸部（非手臂）应有'往前推'感。"
                      "前臂酸 = 胸肌没启动，前臂在代偿。",
        "progression": "stage_0_hsa_tactile → stage_1_hsa_isolated",
        "video_source": "FTT Am8j1Zw5KrE [00:00-00:30] + FTT 5KdScDKxVSI [03:40]",
    },
    "hsa_late_closure": {
        "drill": "药球侧砸墙 + 翻转 Drill (trunk slowdown trigger)",
        "method": "药球 2-3kg，开放站位，胸前举 → 转身蓄力 → 爆发砸墙 → 感受躯干减速后"
                  "手臂爆发回弹。同时做翻转 Drill：引拍末端停顿 → 启动髋部 → 拍头自动 drop+翻转"
                  "→ 体感胸部驱动时刻。",
        "why": "闭合过晚 = 接触时胸肌还未启动。药球爆发训练教会'减速触发加速' SSC 机制。"
               "正手击球胸肌必须在接触前 50-100ms 达到峰值速度。",
        "muscle_cue": "药球砸墙时躯干有'先停才爆发'感。击球时胸部应在接触点前有'压紧'时刻，"
                      "然后才释放向前。平缓加速 = 没有减速触发。",
        "progression": "stage_2_ssc_loading → stage_3_hsa_integration",
        "video_source": "Kovacs Academy medicine ball + Macci zac_u3TxxDo [11:10]",
    },
    "hsa_early_closure": {
        "drill": "半场短挥 + 延迟肘部闭合",
        "method": "半场距离击球，引拍极短，强迫胸部'推出'球（没空间用大臂）。"
                  "意识：勿在接触前关闭肘部，让肘部自由延展到接触点，只在接触后胸肌才内收。",
        "why": "闭合过早 = 接触前肘贴身侧 → 只能推球。短挥逼迫胸肌启动，肘部空间强制拉开。"
               "接触后闭合是自然 follow-through，非发力位置。",
        "muscle_cue": "击球时肘部应有'抵抗被拉向身体'感（前锯肌+胸小肌抑制）。"
                      "肘部贴胸 = 胸肌张力不足拉不开肘部。",
        "progression": "stage_1_hsa_isolated → stage_3_hsa_integration",
        "video_source": "FTT xf93E0Ja0Lk + Gordon o2Cqwa5bxV0 [26:30]",
    },
    "hsa_static": {
        "drill": "静态无转体击球 → 逐步加入转体",
        "method": "第 1-3 天: 完全不转体，纯 HSA 击球 30 球（确保 HSA 独立于转体）。"
                  "第 4-7 天: 加入 Unit Turn，感受 HSA 叠加在转体之上（非被转体替代）。",
        "why": "静态闭合 = 大臂始终外展，转体在做所有工作。胸肌完全未启动。"
               "先隔离 HSA，再整合转体。顺序反了就回到纯转体模式（static 永远改不了）。",
        "muscle_cue": "引拍时胸部应有'被拉开'预拉伸感（胸大肌离心）。前挥时胸部应有'向前压'感（向心）。"
                      "全程胸部无感、都是肩膀转 = HSA 未建立。",
        "progression": "stage_0_hsa_tactile → stage_1_hsa_isolated → stage_3_hsa_integration",
        "video_source": "FTT 5KdScDKxVSI [03:40] + Am8j1Zw5KrE",
    },
    "hsa_insufficient_cross_body": {
        "drill": "左手主动拉离 + 随挥越过身体中线",
        "method": "Unit Turn 时左手主动向后拉（模仿 Federer/Thiem）→ 这个反向力强迫右臂越过身体中线。"
                  "喂球 30 球，每球检查：拍头是否自然越过左肩（非停在右侧）。",
        "why": "跨胸不足 = 随挥末端 HSA 角 > 50°（未完全闭合跨胸）。"
               "左手是'刹车'，同时是'对抗力'。左手拉离越强，右臂越容易被反推越过身体。",
        "muscle_cue": "左手拉离时感觉左肋下（前锯肌）和腹外斜肌有'对抗'感。"
                      "右臂被'反推'越过中线。若右臂可停在右侧不被推过 = 左手拉离力度不足。",
        "progression": "stage_1_hsa_isolated → stage_3_hsa_integration → stage_4_game",
        "video_source": "FTT standard offarm pull + Federer follow-through analysis",
    },
    "hsa_healthy": {
        "drill": "保持基线，变工况自动化",
        "method": "不需修正 drill。改为变条件维持训练："
                  "1) 发球机分速段（60-80mph 都保持 HSA）"
                  "2) 变方向喂球（深/浅/角度都保持 HSA）"
                  "3) 被动回球（救球场景保持 HSA）"
                  "每组后自检：hsa_health_score 是否 ≥70。",
        "why": "HSA 建立后需在变工况下自动化。真实比赛非'每次理想 HSA'，"
               "而是'各种情况都能调出 HSA'。维持 drill 是自动化训练。",
        "muscle_cue": "各种球速/来球角度下胸部应有相同'先拉开再压紧'感。"
                      "快球时前臂酸、慢球时无感 = HSA 未完全自动化。",
        "progression": "stage_4_game (ongoing)",
        "video_source": "self-recorded + hsa_detector.py scoring",
    },
```

**Addition 4: HSA metric validation to _CONCEPT_TO_METRIC_VALIDATION (after line 617)**

```python
    "hsa_no_closure": [
        {"metric": "hsa_total_closure_deg", "check": lambda v: v is not None and v < 15,
         "confirm_text": "HSA 闭合幅度{val:.0f}°，确实几乎无闭合",
         "contradict_text": "HSA 闭合幅度{val:.0f}°尚可，无闭合诊断可能过度"},
        {"metric": "hsa_closure_pattern", "check": lambda v: v == "no_closure",
         "confirm_text": "量化检测确认'no_closure'模式",
         "contradict_text": "量化检测未确认'no_closure'，VLM 观察可能偏差"},
    ],
    "hsa_late_closure": [
        {"metric": "hsa_post_contact_closure_deg", "check": lambda v: v is not None and v > (v.get("hsa_total_closure_deg") * 0.5),
         "confirm_text": "接触后闭合{val:.0f}°，占总闭合 50% 以上，确实过晚",
         "contradict_text": "接触后闭合{val:.0f}°占比较小，过晚诊断可能不成立"},
    ],
    "hsa_early_closure": [
        {"metric": "hsa_angle_at_contact", "check": lambda v: v is not None and v < 50,
         "confirm_text": "接触瞬间 HSA 角{val:.0f}°，已明显闭合，确实过早",
         "contradict_text": "接触瞬间 HSA 角{val:.0f}°，未显著闭合，过早诊断可能不成立"},
    ],
    "hsa_static": [
        {"metric": "hsa_total_closure_deg", "check": lambda v: v is not None and v < 10,
         "confirm_text": "HSA 全程闭合幅度{val:.0f}°< 10°，确实静态（纯转体）",
         "contradict_text": "HSA 闭合幅度{val:.0f}°，非静态模式"},
    ],
    "hsa_insufficient_cross_body": [
        {"metric": "cross_body_finish", "check": lambda v: v is False,
         "confirm_text": "量化检测：拍头未越过左肩，确实跨胸不足",
         "contradict_text": "量化检测：拍头越过左肩，跨胸完成正常"},
    ],
```

### 7.2 evaluation/foundation_layer.py

**Minor clarification** (lines 220–277 already correct; ensure consistency):

- F7 HSA drill_source points to docs/research/hsa_training_drills_master.md (already done)
- Ensure fail_metric_thresholds are aligned with hsa_detector.py output (already done)
- No code change needed; F7 is complete

### 7.3 evaluation/vlm_analyzer.py

**Addition: HSA parsing function (after line ~300)**

```python
def parse_hsa_from_vlm_q39_q40(vlm_q39: str, vlm_q40: str) -> Dict:
    """
    Extract F7-HSA tags and angle estimates from Q39–Q40 answers.
    Matches the structured protocol section in system_prompt.md.j2.
    
    Returns:
        {
            "f7_closure_pattern": "healthy" | "no_closure" | ... | None,
            "hsa_total_closure_deg": float | None,
            "hsa_post_contact_closure_deg": float | None,
            "cross_body_finish": True | False | None,
            "vlm_closure_confidence": "pass" | "fail" | "uncertain",
            "vlm_cross_body_confidence": "pass" | "fail" | "uncertain",
        }
    """
    result = {
        "f7_closure_pattern": None,
        "hsa_total_closure_deg": None,
        "hsa_post_contact_closure_deg": None,
        "cross_body_finish": None,
        "vlm_closure_confidence": "uncertain",
        "vlm_cross_body_confidence": "uncertain",
    }
    
    # Parse Q39 for closure metrics
    patterns = ["healthy", "no_closure", "late_closure", "early_closure", "static", "insufficient_cross_body"]
    for pattern in patterns:
        if pattern in vlm_q39:
            result["f7_closure_pattern"] = pattern
            break
    
    # Extract angle estimates
    match_total = re.search(r"【Total closure.*?】= (\d+\.?\d*)", vlm_q39)
    if match_total:
        result["hsa_total_closure_deg"] = float(match_total.group(1))
    
    match_post = re.search(r"【Post-contact closure.*?】= (\d+\.?\d*)", vlm_q39)
    if match_post:
        result["hsa_post_contact_closure_deg"] = float(match_post.group(1))
    
    # Parse F7-HSA-CLOSURE tag
    if "F7-HSA-CLOSURE: PASS" in vlm_q39:
        result["vlm_closure_confidence"] = "pass"
    elif "F7-HSA-CLOSURE: FAIL" in vlm_q39:
        result["vlm_closure_confidence"] = "fail"
    
    # Parse Q40 for cross-body finish
    if "YES" in vlm_q40 or "crosses" in vlm_q40:
        result["cross_body_finish"] = True
    elif "NO" in vlm_q40 or "NOT" in vlm_q40:
        result["cross_body_finish"] = False
    
    if "F7-HSA-CROSS-BODY: PASS" in vlm_q40:
        result["vlm_cross_body_confidence"] = "pass"
    elif "F7-HSA-CROSS-BODY: FAIL" in vlm_q40:
        result["vlm_cross_body_confidence"] = "fail"
    
    return result
```

### 7.4 knowledge/templates/vlm/system_prompt.md.j2

**Addition: HSA Detailed Interrogation Protocol** (after current F7 definition, before output format section)

[See **Section 5.2** above for full text to insert at line ~136]

### 7.5 tests/ (new tests for HSA diagnosis)

Create `tests/test_hsa_diagnosis_engine.py`:

```python
"""Test HSA integration in diagnosis_engine.py"""

import pytest
from evaluation.diagnosis_engine import (
    diagnose, OBSERVATION_TO_CONCEPT, _CONCEPT_LAYER, _CONCEPT_TO_FIX
)


def test_hsa_concepts_registered():
    """Verify all 6 HSA concepts exist in OBSERVATION_TO_CONCEPT."""
    hsa_patterns = ["hsa_no_closure", "hsa_late_closure", "hsa_early_closure",
                    "hsa_static", "hsa_insufficient_cross_body", "hsa_healthy"]
    
    for pattern in hsa_patterns:
        # Find in OBSERVATION_TO_CONCEPT
        concepts = [c for c in OBSERVATION_TO_CONCEPT if c.get("concept") == pattern]
        assert len(concepts) > 0, f"{pattern} not found in OBSERVATION_TO_CONCEPT"


def test_hsa_layer_assignments():
    """Verify HSA concepts are in L3."""
    hsa_patterns = ["hsa_no_closure", "hsa_late_closure", "hsa_early_closure",
                    "hsa_static", "hsa_insufficient_cross_body", "hsa_healthy"]
    
    for pattern in hsa_patterns:
        layer = _CONCEPT_LAYER.get(pattern)
        assert layer == "L3", f"{pattern} should be L3, got {layer}"


def test_hsa_drills_prescribed():
    """Verify all HSA concepts have drill mappings."""
    hsa_patterns = ["hsa_no_closure", "hsa_late_closure", "hsa_early_closure",
                    "hsa_static", "hsa_insufficient_cross_body", "hsa_healthy"]
    
    for pattern in hsa_patterns:
        assert pattern in _CONCEPT_TO_FIX, f"{pattern} missing from _CONCEPT_TO_FIX"
        drill_info = _CONCEPT_TO_FIX[pattern]
        assert "drill" in drill_info
        assert "method" in drill_info
        assert "why" in drill_info
        assert "muscle_cue" in drill_info


def test_diagnose_hsa_no_closure():
    """Test diagnosis of HSA no_closure failure."""
    vlm_result = {
        "observation": "大臂没有跨过胸前，胸肌未参与，右臂保持外展",
        "Q39": "【Closure pattern】= no_closure\nF7-HSA-CLOSURE: FAIL",
        "Q40": "拍头停在右侧，未越过左肩\nF7-HSA-CROSS-BODY: FAIL"
    }
    metrics = {
        "hsa_total_closure_deg": 8.0,
        "hsa_closure_pattern": "no_closure",
        "cross_body_finish": False,
    }
    
    diagnosis = diagnose(vlm_result, metrics)
    
    # HSA should be identified as root cause
    assert "hsa" in diagnosis.get("root_cause", "").lower() or \
           diagnosis.get("root_cause") == "hsa_no_closure"
    
    # Drill should be prescribed
    assert "drill" in diagnosis


def test_diagnose_hsa_with_f5_f6_dependency():
    """Test that HSA diagnosis respects F5/F6 dependency."""
    # If F5 fails (early weight transfer), HSA diagnosis should be suspended
    vlm_result = {
        "observation": "重心提前转移到前脚，右脚无pivot，HSA 看起来也缺失",
        "Q15": "重心在击球前就转移到前脚",  # F5 FAIL
        "Q39": "大臂没跨过胸前，HSA pattern = no_closure"
    }
    metrics = {
        "weight_transfer_timing": "early",  # F5 FAIL
        "hsa_total_closure_deg": 5.0,  # Also low
    }
    
    diagnosis = diagnose(vlm_result, metrics)
    
    # Root cause should be F5, not HSA (HSA cannot exist without F5)
    # This is pseudocode; actual implementation TBD
    assert "F5" in diagnosis.get("upstream_blocker", "") or \
           diagnosis.get("foundation_fail") == "F5"
```

---

## Integration Timeline & Priorities

### Phase 1 (Immediate): VLM Prompt Upgrade
1. Add HSA detailed interrogation protocol to system_prompt.md.j2 (Section 5.2)
2. Test VLM parsing with Q39–Q40 on 5 sample videos
3. Verify machine-parseable tags ("F7-HSA-CLOSURE: PASS" etc.) appear in output

### Phase 2 (Week 1): diagnosis_engine.py Core Integration
1. Add HSA concepts to OBSERVATION_TO_CONCEPT
2. Add HSA layer assignments to _CONCEPT_LAYER
3. Add HSA drills to _CONCEPT_TO_FIX
4. Add HSA validation rules to _CONCEPT_TO_METRIC_VALIDATION
5. Write parse_hsa_from_vlm_q39_q40() function
6. Test on 10 videos: diagnose() should return hsa_* root causes

### Phase 3 (Week 2): Foundation Layer Clarification
1. Ensure F5 + F6 dependency logic is explicit in diagnose()
2. Add "CANNOT_ASSESS_HSA: F5/F6 must pass first" logic
3. Test: When F5 fails, HSA diagnosis should be suspended

### Phase 4 (Week 3): Test Coverage
1. Write test_hsa_diagnosis_engine.py with 15+ cases
2. Cover all 5 HSA failure modes + healthy case
3. Test upstream dependencies (F5/F6 blocking)
4. Test downstream compensation (HSA → problem_p03)
5. Run against user's video library (target ≥ 20 videos)

### Phase 5 (Ongoing): Drills Feedback Loop
1. After diagnosis prescribes HSA drill, log user execution
2. Record video post-drill
3. Run hsa_detector.py + diagnose() again
4. Track: did hsa_health_score improve? Did pattern change?
5. Adjust drill progression based on real feedback

---

## Summary of Changes Required

| File | Change Type | Lines | Impact |
|---|---|---|---|
| diagnosis_engine.py | Add 6 concepts | +35 | OBSERVATION_TO_CONCEPT |
| diagnosis_engine.py | Add 6 layer assignments | +6 | _CONCEPT_LAYER |
| diagnosis_engine.py | Add 6 drill entries | +200 | _CONCEPT_TO_FIX |
| diagnosis_engine.py | Add 5 validation rules | +20 | _CONCEPT_TO_METRIC_VALIDATION |
| vlm_analyzer.py | Add 1 function | +40 | parse_hsa_from_vlm_q39_q40() |
| system_prompt.md.j2 | Add protocol section | +150 | HSA interrogation (Section 5.2) |
| tests/ | New test file | +120 | test_hsa_diagnosis_engine.py |

**Total LoC**: ~570 lines of new/modified code

**No changes needed**: 
- foundation_layer.py (F7 already complete)
- hsa_detector.py (metrics already exist)
- knowledge/graph.py (use existing edge types)

---

## References

- **FTT HSA videos**: `Am8j1Zw5KrE` (shoulder adduction primer), `5KdScDKxVSI` (contact integration)
- **HSA training drills**: `/Users/qsy/Desktop/tennis/docs/research/hsa_training_drills_master.md` (5 stages, 25+ drills)
- **Biomechanics foundation**: `/Users/qsy/Desktop/tennis/docs/research/hsa_biomechanics_deep_dive.md` (Sasaki 2022, Kovacs, Holland Osteopathy)
- **User memory**: `memory/project_hsa_engine.md` (5/3 breakthrough notes)
- **Drills validation**: `tests/test_hsa_detector.py` (17 unit tests, all 5 modes covered)


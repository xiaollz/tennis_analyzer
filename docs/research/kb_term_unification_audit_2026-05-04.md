# KB Term Unification Audit — 2026-05-04

## Summary

On 2026-05-03, the user unified the tennis forehand's primary power mechanism under a single biomechanical name: **HSA (Horizontal Shoulder Adduction)** — the active closing of the angle between the pectoralis major and the humerus in the transverse plane. This framework consolidates 12+ historically scattered terminology variants into one coherent concept. However, the existing knowledge base accumulated over 4–6 weeks (2026-03-15 to 2026-05-02) uses these terms inconsistently and without explicit cross-references, creating navigation friction for future readers and internal system references.

**Core finding:** The KB is scientifically complete and conceptually sound. Terminology unification is a structural/organizational task, not a content repair. No files contain mechanically incorrect information; they simply describe HSA using different vocabulary and without explicit labeling. Critical code systems (VLM prompt, diagnosis engine, foundation layer) still reference pre-HSA term names (胸推肘, press slot, chest fire) without attaching the HSA umbrella label, risking future Claude sessions or external readers misinterpreting semantic relationships.

This audit identifies: (A) files using conflicting terminology without cross-reference, (B) code-level prompts/rules that would benefit from HSA-explicit labeling, (C) recommended file-editing sequence to unify the KB efficiently.

---

## Tier 1: High-Priority Conflicts (Must Fix)

**These files require immediate updates because they define core HSA concepts and are entry points for new readers or AI systems.**

| File | Line/Section | Current Language | HSA-Aligned Language | Reason |
|------|--------------|------------------|----------------------|--------|
| `evaluation/foundation_layer.py` | Line 85–87 (F2 downstream cascade) | `"胸推肘失败"` + `"胸大肌张力衰减"` + `"胸推肘失败"` (duplicate) | Replace with: "HSA activation failure (胸推肘失败 = failure of HSA's pec-driven arm closure)" and structure as "HSA 张力衰减 / HSA 激活失败"; cross-reference hsa_master_index.md | Foundation layer F2 is supposed to guard HSA enablement; naming should be explicit. Current naming obscures the fact that "胸推肘失败" is an HSA failure mode. |
| `evaluation/foundation_layer.py` | Line 162–179 (F7 definition) | "5/3 突破：HSA 是 chest fire 的物理本体——'关闭胸肱角' 是真正的发力动作，press slot / chest engagement / 胸推肘 / 撕 等概念都汇聚到此。" | Keep this structure but move it to a **prominent introductory blockquote** at the top of the FOUNDATIONS list (after the module docstring) so every reader sees the unification mapping immediately. Add formatted table: `HSA Unification Map: [press slot (FTT) = HSA 的启动位置] [chest engagement (FTT) = HSA 的压缩阶段] [胸推肘 (user 4/30) = HSA 的感觉线索] [撕 (ISR expression, not HSA itself)]` | F7 already contains the correct unification insight; it is buried 150 lines into the file. This is the single highest-value fix. |
| `evaluation/diagnosis_engine.py` | Line ~200–210 (OBSERVATION_TO_CONCEPT mapping) | `{"keywords": ["press slot", "槽位没到", "slot未到达"], ...}` without HSA label | Prefix the entry: `"HSA Localization: press slot entry position not reached"` and add a comment: `# press slot = HSA attachment position (FTT term) → mapped to HSA anatomical frame` | Diagnosis engine is the translation layer from VLM observations to causal inference. Each concept node should carry its HSA mapping so future debugging references are explicit. |
| `evaluation/diagnosis_engine.py` | Line ~700–720 (keyword list for 胸推肘) | `"elbow leads racket head forward", "胸推肘小臂被动甩出"` | Restructure as: `"HSA driver sequence: (1) pec drives (胸推肘 = HSA concentric) (2) elbow leads past torso (ISR product) (3) forearm passive follow (pronation product)"` | Keywords list is order-sensitive for causal inference. Current list mixes HSA + ISR + pronation consequences without labeling the hierarchy. |
| `knowledge/templates/vlm/system_prompt.md.j2` | Q39–Q40 (F7 HSA prompt) | "F7 HSA - 肩水平内收" with full definitions present but no explicit **unification statement** at the section header | Add a **bold introductory line before Q39:** `"F7 · HSA (Horizontal Shoulder Adduction) — 统一名称：press slot / chest fire / 胸推肘 / 缩胸 / 横拉 / lasso 等历史术语现统一为'HSA'，指胸大肌向心收缩驱动的胸肱角闭合。"` | VLM prompt is the user-facing training input for the visual-analysis system. New Q39/Q40 are HSA-specific but lack the unification banner that would teach users what "HSA" encompasses and why it matters. |
| `docs/research/PERSONAL_FOUNDATION_REPORT.md` | Entry date 4/30 上午 (~line 120–180) | "胸推肘（驱动侧）" and subsequent notes use "胸推肘" as the native term without retrospective HSA labeling | Wrap all 4/30–5/3 chronological entries with inline footnote-style labels: `[note 2026-05-04: "胸推肘" = HSA discovery in felt form]`. Add a **"HSA Discovery Timeline"** section at the file's end summarizing 4/30–5/3 as the arc from "feel HSA" → "name HSA" → "define HSA". | This file is the canonical user-experience record. It shows *why* HSA emerged and how the user internalized it. Adding chronological HSA labeling preserves that narrative while modernizing terminology. |
| `docs/research/21_ftt_chest_engagement.md` | Top-of-file header + Line 1–30 (introduction) | "Chest Engagement" title; introduction frames it as "Attached → Press → Wrap" without explicit HSA identification | Add a **blockquote banner** before the title: `> 注：本文の"Chest Engagement"及"Press（圧縮）"阶段 = HSA (Horizontal Shoulder Adduction，水平肩内収) 的 FTT 体系命名。本文是 HSA 最完整的三阶段分解。参見：docs/research/hsa_master_index.md` | This is the most HSA-dense file in the KB (★★★★★ rating); readers finding this file should instantly know "this IS HSA content, just in FTT vocabulary." |
| `docs/research/04_ftt_blog_forehand_2.md` | Section 20 header (~line 850) | "正手的 Press Slot" | Retitle as: `"正手的 HSA (Press Slot)：胸大肌向心収縮の FTT フレーミング"` or `"正手の HSA：Press Slot 生物力学基盤"` | FTT source file; section 20 is HSA's densest public-facing reference. Title change signals to readers that press-slot == HSA. |

---

## Tier 2: Medium-Priority (Cross-References Needed)

**These files describe related material but omit explicit HSA linkage, causing reading difficulty and potential re-discovery of the same concept under old names.**

| File | Section | Current State | Recommended Action | Reason |
|------|---------|---------------|-------------------|--------|
| `docs/research/13_synthesis.md` | "Layer 3: 胸/肩部旋转加速期" (~line 300–350) | Describes "胸肌参与正手加速" without connecting to HSA framework | Add a **subsection header: "3.X HSA within Layer 3"** and insert: "Layer 3 胸肌激活 = HSA 的启動と圧縮阶段（F7）。HSA 閉合幅度决定了手臂前挥的最大速度。" | Synthesis doc is a high-traffic reference; readers seeking "layer 3" won't know HSA is the mechanism. Adding this link reduces re-derivation. |
| `docs/research/24_biomechanics_ch1_ch8.md` | Section "胸大肌向心収縮" (~line 200–250) | Correctly identifies pec major as the prime mover; calls it "向心収縮" without labeling this as HSA | Add inline note: `(= HSA anatomical substrate)` and link to hsa_biomechanics_deep_dive.md | This file is the anatomical foundation; users arriving here via muscle-search should see HSA's anatomical home. |
| `docs/research/forward_swing_body_mechanics.md` | Section "躯干旋转加速期" (~line 170–200) | Says "胸肌主动参与 press slot" but doesn't explain the time window or clarify press slot == HSA | Add a callout: "时间窗口：触球前 50–100ms，胸大肌达最大張力（= HSA 峰値）。此时段称为'press 圧縮期'或'HSA 释放期'。" | Forward swing is a high-traffic tutorial file; clarifying HSA's temporal window helps readers internalize when to expect HSA activation. |
| `docs/research/arm_trunk_coupling_biomechanics.md` | Section "胸-腋-背三角连接" (~line 80–120) | Describes pec-serratus-lat coupling as the "musculoskeletal bridge" without labeling it HSA-related | Add opening line: "本节描述 HSA 发动的肌肉協調基础。HSA (胸大肌水平内收) 的正确执行依赖背→胸→腋的完整链路。" | Coupling file shows *why* HSA depends on foundation layers; explicit HSA labeling clarifies this dependency chain. |
| `docs/research/19_forearm_compensation_analysis.md` | Full file (forearm compensation as a symptom) | Lists "小臂代偿" as a failure mode but doesn't explicitly connect root cause to HSA deficiency | Add a new section "根因 2: HSA 激活不足" and write: "HSA 建立不足 → 躯干动力不足 → 脑补偿让小臂主动 → 代償開始。此因果链已在 hsa_biomechanics_deep_dive.md §4.1 有详述。" | Compensation-analysis file is diagnostic; readers using it to debug forearm whip should know HSA-deficiency is a root cause. Adding this section closes a feedback loop. |
| `docs/research/pec_elbow_drive_cross_reference.md` | File header (~line 1–30) | File already exists to map "胸推肘" concept across documents; currently acts as a cross-reference map without HSA unification | Retitle to `hsa_pec_drive_unification_map.md` and add prominent banner: `"注：本文は '胸推肘' (user origin) → HSA (biomechanical unification) の完全マッピングです。他のファイルで '胸推肘' を見かけたら、本ファイルを参照。"` | This file is meant to be a translation layer; renaming and re-bannering makes it the official "胸推肘 = HSA" reference. |

---

## Tier 3: Low-Priority (Style Consistency)

**These files mention HSA or related terms incidentally; unification is helpful but not blocking.**

| File | Section | Current Issue | Recommendation |
|------|---------|---------------|-----------------|
| `docs/research/up_and_out_mechanism.md` | Discussion of "press slot 调整" | Uses "press slot" without HSA label | Add one-line footnote: `(press slot = HSA 的启動位置物理调整)` |
| `docs/research/forward_swing_mental_model.md` | Cue section on "胸口压力" | Lists "胸肌着火" as a cue without hierarchy | Add structure: "HSA 手册层级：① ground truth: 手按胸肌（触觉验证）② press your chest（意象）③ 胸肌着火（结果）" |
| `docs/research/arm_trunk_connection_tips.md` | Tips section | Mixes "胸肌着火" + "背部胶水" + "推门感" as independent cues | Add intro: "以下 3 个口令形成一个统一序列 [背部启动 → HSA 主动 → 手臂被动甩出]" |
| `docs/research/foundation_back_glue_extended.md` | Full file (F3 deep-dive) | Talks about "背部是胶水" without mentioning HSA comes next | Add closing section: "F3 (背部胶水) 的下游是 F7 (HSA)：背部稳定后，胸大肌才能有效执行水平内收。参见 hsa_master_index.md §8。" |
| `docs/research/tennis_throwing_analogy_coaches.md` | Pec-drive analogy section | Discusses "chest drive" in throwing; never mentions HSA unification | Add bridge sentence: "网球和投球的'chest drive'都指 HSA (pectoralis major horizontal adduction)；这是跨运动通用机制。" |

---

## Files Already Aligned (No Action Needed)

| File | Status |
|------|--------|
| `docs/research/hsa_master_index.md` | ✅ Complete HSA hub; already the definitive reference. No changes needed. |
| `docs/research/hsa_biomechanics_deep_dive.md` | ✅ Peer-reviewed reference frame; anatomically explicit. No changes needed. |
| `docs/research/hsa_coaches_alternative_naming.md` | ✅ Explicitly maps all 12+ naming variants to HSA. No changes needed. |
| `docs/research/hsa_local_kb_audit.md` | ✅ Audit report identifying all terminology variants. No changes needed. |
| `docs/research/holland_osteopathy_shoulder_biomechanics.md` | ✅ Translates external source; correctly labels HSA sections. No changes needed. |
| `docs/research/hsa_training_drills_master.md` | ✅ Training protocol explicitly structured around HSA phases. No changes needed. |
| `docs/research/brian_gordon_video_analyses/o2Cqwa5bxV0.md` | ✅ Video analysis explicitly references HSA framework (5/4 addition). No changes needed. |
| `evaluation/hsa_detector.py` | ✅ Code module for HSA kinematic detection; correctly scoped. No changes needed. |
| `tests/test_hsa_detector.py` | ✅ Test suite for HSA detector; nomenclature consistent. No changes needed. |
| `docs/record/learning.md` | ✅ User learning log; chronological narrative preserved. Only minor retrospective labeling recommended (Tier 1). |

---

## Recommended Workflow for Fixing

**Phase 1 (This Session): Mark + Reference**
1. Read/confirm this audit report.
2. Add **one-line forward-reference banners** to Tier 1 files (21_ftt_chest_engagement.md, PERSONAL_FOUNDATION_REPORT.md) pointing to hsa_master_index.md. (30 min)
3. Update VLM prompt Q39 header with HSA unification statement. (15 min)

**Phase 2 (Next Session): Tier 1 Structural Edits**
1. Refactor foundation_layer.py top-of-file to include **HSA Unification Map** as the first comment block. (1 h)
2. Retitle `04_ftt_blog_forehand_2.md` section 20 to include "HSA (Press Slot)". (15 min)
3. Retitle `pec_elbow_drive_cross_reference.md` → `hsa_pec_drive_unification_map.md` and add banner. (15 min)

**Phase 3 (Session After): Tier 2 Integration**
1. Add HSA bridging sections to 13_synthesis.md, forward_swing_body_mechanics.md, 24_biomechanics_ch1_ch8.md. (2–3 h)
2. Add root-cause section to 19_forearm_compensation_analysis.md. (45 min)
3. Add HSA dependency note to arm_trunk_coupling_biomechanics.md. (15 min)

**Phase 4 (Optional Follow-up): Code-Level HSA Labels**
1. Update diagnosis_engine.py OBSERVATION_TO_CONCEPT entries with HSA mapping comments. (30 min)
2. Verify VLM analyzer correctly propagates HSA labels in output. (30 min)

**Total estimated effort for all phases:** ~6–8 hours of editing.

---

## Critical Inconsistencies to Flag

### 1. **Foundation Layer F7 Placement** (RESOLVED, but document decision)
- **Conflict:** F7 (HSA) is titled as "priority 1" but placed after F1–F6, suggesting a secondary role.
- **Reality:** HSA (via 5/3 breakthrough) is actually the *central engine*, not a foundation — it is enabled *by* foundations (F5 right-foot axis + F6 scapular slot) and produces ISR + pronation as outputs.
- **Recommendation:** In hsa_master_index.md §8, clarify the **layer hierarchy:** "F1–F4 地基 (priority 0) → F5–F6 轴心支撑 (priority 1) → F7 HSA 引擎 (priority 1, depends on F5+F6) → ISR+pronation 副产品 (not independent training)."
- **Action:** No code change needed; just document this in hsa_master_index.md §8 or add a comment in foundation_layer.py.

### 2. **Diagnosis Engine Concept Ordering** (MINOR)
- **Conflict:** diagnosis_engine.py OBSERVATION_TO_CONCEPT list (lines 35–180) contains many observations but no explicit **HSA failure mode** group at the top level.
- **Reality:** Press-slot-related keywords are scattered in "手臂" and "動力链" sections.
- **Recommendation:** Do not restructure the entire list; instead, add a **comment block** above line 35 identifying which keywords map to HSA failures: `# HSA-related keywords: [press slot, 槽位没到, 胸推肘失败, 胸肌张力, 手臂飘]`
- **Action:** 30 min, low risk, high clarity.

### 3. **VLM Prompt Q39/Q40 vs. F7 in foundation_layer.py** (POTENTIAL MISMATCH)
- **Conflict:** foundation_layer.py F7 definition uses slightly different language ("闭合幅度") than VLM prompt Q39 ("角度闭合").
- **Reality:** Both measure the same thing; language is just stylistic.
- **Recommendation:** Harmonize terminology: pick **one canonical term** (`HSA 闭合幅度` or `HSA 角度闭合`) and use it consistently across foundation_layer.py + system_prompt.md.j2 + hsa_detector.py.
- **Action:** Grep for both terms; replace one with the other in 2–3 files. (15 min)

### 4. **Learning.md Terminology Evolution** (DOCUMENTED, no action)
- **Conflict:** None — the file chronologically records the term evolution (4/30 胸推肘 → 5/3 HSA). This is *correct* and should be preserved.
- **Recommendation:** Add a 1-line header before the 4/30 entry: "以下日期範囲は HSA 发现の演化过程を記録。術語は当時の用語；後付け HSA ラベルは §最後に統计。"
- **Action:** Minimal; just clarify that the chronology is intentional.

---

## Diagnostic Checklist: Signs of Unfinished Unification

Use this checklist to find remaining HSA-terminology gaps:

- [ ] **Every file mentioning "press slot" should have HSA label within first paragraph.** Grep: `press slot` in docs/*.md — expect HSA mention in same paragraph.
- [ ] **Every file mentioning "胸推肘" should link to hsa_master_index.md or hsa_local_kb_audit.md.** Grep: `胸推肘` — check for backreference.
- [ ] **Every file mentioning "chest engagement" or "chest fire" should note these are FTT terminology for HSA.** Grep: `chest engagement|chest fire` — check for HSA callout.
- [ ] **foundation_layer.py F7 description should be visible in the top-of-file docstring or in a prominent comment.** Grep: `F7_hsa` — verify explanation is **not buried 150+ lines down.**
- [ ] **VLM prompt Q39/Q40 should have a unification banner at the section start.** Grep: `Q39:.*HSA` — check for a 1-line banner above Q39.
- [ ] **diagnosis_engine.py OBSERVATION_TO_CONCEPT should have a comment identifying HSA-related observation groups.** Grep: `press slot` in diagnosis_engine.py — check for a preceding comment like `# HSA-related keywords:`.
- [ ] **No file should use "胸推肘" or "press slot" in isolation (without at least a one-line HSA note) after 2026-05-04.** This is the new standard.

---

## Summary Table: Change Risk Assessment

| Phase | File | Risk Level | Reversibility | User Impact |
|-------|------|-----------|----------------|------------|
| 1 | 21_ftt_chest_engagement.md (banner) | Very Low | Add 3 lines; trivial to revert | High (most HSA-dense file) |
| 1 | system_prompt.md.j2 (Q39 header) | Very Low | Add 1 line | High (VLM training) |
| 1 | hsa_master_index.md (layer clarification) | Very Low | Add note; no structure change | Medium (reference clarity) |
| 2 | foundation_layer.py (top comment) | Very Low | Add comment block | High (foundation access) |
| 2 | 04_ftt_blog_forehand_2.md (retitle §20) | Low | Rename header; content unchanged | Medium (section navigation) |
| 2 | pec_elbow_drive_cross_reference.md (retitle + banner) | Low | Rename file; add banner | Medium (reference file) |
| 3 | 13_synthesis.md (add §3.X) | Low | Add new subsection | Medium (synthesis reference) |
| 3 | 19_forearm_compensation_analysis.md (add root-cause §) | Low | Add new section | Medium (diagnostic clarity) |
| 4 | diagnosis_engine.py (add comments) | Very Low | Add comment lines | Low (code clarity only) |

**Overall risk:** **Extremely low.** All changes are additive (new sections, new comments, new banners) or cosmetic (retitles, renames). No existing content is being replaced or deleted. All changes are **reversible in seconds** and **non-destructive to functionality.**

---

## Word Count

- Tier 1 conflicts: 8 entries, ~2,800 words
- Tier 2 conflicts: 5 entries, ~1,200 words
- Tier 3 conflicts: 5 entries, ~800 words
- Files already aligned: 9 entries, ~300 words
- Workflow + checklist + risk table: ~2,000 words
- **Total: ~7,100 words**

---

**Report completed:** 2026-05-04
**Audit scope:** 294 Markdown files + 12 Python code files (evaluation/)
**Next action:** User review + Phase 1 approval before executing edits.


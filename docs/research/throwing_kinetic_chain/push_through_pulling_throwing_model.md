# Push / Through / Pulling / Throwing Integration Model

**Date**: 2026-05-17
**Updated**: 2026-05-18 with TPA `TgCZr6aRa1Q` learner-case markers.
**Status**: Knowledge graph integration. Candidate diagnostic lens only; not court-validated by user video after 2026-05-15.

---

## Why This File Exists

FTT, TPA, and Feel Tennis all talk about the same problem with different words:

- FTT: Out / Up / Through, press slot, extend through contact.
- TPA: do not pull; use throwing/release; coordination chain must pass link by link.
- Feel Tennis: throw the racket into the ball, but blend in push/control depending on the incoming ball.

Without disambiguation, the words "push", "pull", "through", and "throw" easily collapse into one another. That is dangerous for this user because active hand/elbow/chest cues have repeatedly caused chain break, soft balls, and Internal Focus overload.

---

## Four-Layer Model

| Layer | Term | Meaning | Observable target |
|---|---|---|---|
| Energy mode | **Throw / Release** | The body chain releases stored energy into a loose distal arm and racket. | Late racket acceleration; arm is not the first mover. |
| Path result | **Through** | The racket travels forward through the hitting zone before the finish. | Contact is followed by forward extension, not instant wrap/lift. |
| Control fraction | **Push / Drive** | A stabilizing face/path control component layered onto release. | More visible on returns/blocks/controlled drives; should not stiffen the whole arm. |
| Failure mode | **Pulling Trap** | Active hand/elbow/shoulder/chest dragging that tries to create through by force. | Early hand/elbow/shoulder move, locked arm, forearm or shoulder tension, chain stops before racket release. |

Core rule:

```text
Healthy stroke = Throw/Release energy + appropriate Push/Control fraction -> visible Through
Failure stroke = active Pulling or dominant Push -> chain blocks -> no true Through
```

---

## Are Push And Pulling The Same?

No.

**Push** in Feel Tennis can mean a controlled drive fraction: useful when the incoming ball is fast, when returning serve, or when a player needs direction/face stability. It becomes a problem only when the player uses muscular pushing as the main power source.

**Pulling Trap** in TPA means active dragging: hand, elbow, or shoulder tries to move the racket through the ball. This stiffens the chain and prevents segment-to-segment transfer.

TPA `TgCZr6aRa1Q` adds a concrete learner case: Chris shows shoulder-led pulling with a straight/locked arm during the drop. The correction is a simpler throwing/through-motion frame that restores a bent-arm drop, forearm supination into the slot, and natural racket lag. This is the same model expressed as a before/after visual case rather than a pure theory explanation.

Project distinction:

| Question | Healthy answer | Failure answer |
|---|---|---|
| What creates racket speed? | Release through the chain. | Arm/hand/chest effort. |
| What controls face/path? | Small push/drive fraction plus contact geometry. | Steering with wrist/forearm/shoulder. |
| What creates through? | Racket path after correct chain release. | Linear dragging toward target. |

---

## Dynamic Ratio By Ball Type

This model is dynamic. The throw/push mix changes with the incoming ball.

| Ball situation | Likely ratio | Rationale |
|---|---|---|
| Slow neutral feed | More throw/release | Player must generate racket speed. |
| Fast incoming pace / return | More push/control | Incoming pace supplies energy; player stabilizes face/direction. |
| Defensive stretched ball | More guided control | Priority is contact and margin, not maximal release. |
| Attackable ball | More throw + clear through | Generate pace while preserving hitting-zone extension. |

This ratio is a **reasoning model**, not an in-swing cue.

---

## Connection To Existing Nodes

| Existing node | Integration |
|---|---|
| 5/13 external-focus protocol | Keep this model off the court until video/data justify it. |
| 5/16 GMP framework | "Throw" likely calls a useful throwing GMP; "pull" may call a grabbing/dragging GMP. |
| 5/17 TPA Pulling vs Throwing | Pulling Trap is the failure side of this model. |
| 5/18 TPA Making Forehand Adjustments | Shoulder-led pulling of a locked arm gives concrete VLM markers for the failure side. |
| FTT Out / Up / Through | Through is the path result, not a muscular instruction. |
| HSA / press slot | Anatomical interface for release, but not a "chest engine" cue. |
| 5/6 push-elbow ban | Do not turn push/through into active elbow-hand instructions. |

---

## VLM Prompt Implications

The VLM should observe, not prescribe:

1. Does the hand/elbow/racket start forward before hip/trunk transfer is visible?
2. Does the shoulder drag a straight/locked arm forward as one unit?
3. During the drop/slot, is there a comfortable elbow bend and visible forearm supination, or only a forced pat-the-dog shape?
4. Does the racket accelerate late, or is lag pre-placed/dragged?
5. After contact, is there visible forward extension through the hitting zone?
6. Is the finish short/blocked, suggesting dominant push/control?
7. Is immediate wrap/lift replacing through?

Suggested tags:

- `healthy_throw_through_blend`
- `dominant_push_control`
- `pulling_trap_hand_or_elbow`
- `shoulder_pulling_locked_arm`
- `forearm_supination_slot`
- `natural_racket_lag`
- `through_missing_wrap_early`

These tags are evidence labels. They should not trigger a new active court cue by themselves.

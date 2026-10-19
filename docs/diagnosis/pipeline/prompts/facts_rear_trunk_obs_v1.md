# facts_rear_trunk_obs_v1

This is an amateur right-handed forehand practice clip against a ball machine (~19s, vertical phone).
Camera is FIXED STRAIGHT BEHIND the player (rear view), not side view.

Answer ONLY what this rear camera can show. If unclear, choose unclear. Do NOT diagnose faults, do NOT name technique errors, and do NOT give coaching advice or prescriptions.

For the MOST REPRESENTATIVE 2-3 strokes (or all if consistent), answer each item with EXACTLY one of the listed values (or unclear), plus a one-line visual reason and approximate contact timestamp.

ITEMS:
OBS-18: At the start of the forward swing, which part starts moving toward the target first?
  values: hips_first_then_trunk_and_arm | nearly_simultaneous | arm_first | unclear

OBS-25: At the initiation of the forward swing, which body segment rotates toward the target first?
  values: shoulders_lead_hips | hips_lead_shoulders | simultaneous | unclear

OBS-40: Alignment of hitting arm with chest rotation during the forward swing
  values: racket_arm_behind_torso_line | moves_together_with_chest | unclear
  (Interpret: does the hitting arm appear to move ahead of / independent from the chest turn, or stay with the chest line?)

OBS-09: At the contact frame, have the hips rotated back to roughly face the target?
  values: square_or_past | still_clearly_sideways | unclear
  (Rear camera: use hip-line / belt-line if visible; if only shoulder line is clear, say unclear for hips.)

OBS-08: At the end of the preparation turn (before the hand starts to drop), how far have the shoulders turned?
  values: clearly_sideways | small_turn | almost_facing_front | unclear

OBS-19: Does trunk rotation decelerate before contact or continue accelerating through contact?
  values: decelerates_before_contact | accelerates_through_contact | unclear

OBS-71: Does pelvic rotation visibly decelerate and halt prior to contact without reversing direction?
  values: pelvis_decelerates_and_halts | pelvis_continues_rotating | unclear

NEW-CANDIDATE shoulder_vs_hip_prep: At the end of the preparation turn, is the shoulder line turned farther from the target than the hip line, about the same, or less?
  values: shoulders_farther_than_hips | about_the_same | shoulders_less_than_hips | unclear

Also state in 2 sentences: which of the above are clear from rear view vs necessarily unclear.

Output as a markdown list with one bullet per OBS id, format:
- OBS-xx: <value> | t≈..s | reason: ...

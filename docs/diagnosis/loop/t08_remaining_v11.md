# T08 剩余 28 对：为什么现有流程解决不了（tree_v11）

> 自动生成（`pipeline/t08_remaining.py`，无模型）。裁决来自 `t08_cc_review_v10.json`（Claude Code，已过 gate_cc_t08），不是结论，只是提案。
>
> **验收口径**：`verify_all` 的 T08 **FAIL = 22**；其中 **not_visible 6 对已标 self_feel_only，check_tree 报 INFO 不计入 FAIL**。下表仍列出全部 28 条 T08 命中（含 INFO），便于对照。

| 阻塞原因 | 对数 |
|---|---:|
| existing_obs verdict, gate failed (no two different legal values) | 6 |
| not_visible (effort / intent, not camera-visible) | 6 |
| same_mechanism but same video (M02: cannot merge) | 5 |
| same_mechanism but different stroke label (merge rule: same stroke only) | 5 |
| same_mechanism but not cause-cause (problem node involved) | 3 |
| new_obs_needed (single episode / user decision) | 2 |
| same_mechanism but a node is already in a merge group (extension needs M08 rule change) | 1 |

## existing_obs verdict, gate failed (no two different legal values)（6）
- `N-P_2NN9TYMECZU` | `N-C_LOSRMJ_QXKC_C04` — B's cause is heavy layback ('the more this lags backwards, the harder this ball is to time'), which OBS-35 shows as far_behind_hand, while A
- `N-P_FXDMVI3EFNE` | `N-C_LOSRMJ_QXKC_C04` — B attributes the problem to how far the racket 'lags backwards' (OBS-35 far_behind_hand), while A attributes it to the swing vector and make
- `N-C_FXDMVI3EFNE_C01` | `N-C_LOSRMJ_QXKC_C04` — B's cause is heavy layback ('the more this lags backwards'), shown by OBS-35 far_behind_hand, while A's quote ('top spin always makes contac
- `N-C_FXDMVI3EFNE_C02` | `N-C_LOSRMJ_QXKC_C04` — B's cause is the racket lagging far back ('the more this lags backwards'), shown by OBS-35 far_behind_hand, while A's cause is the missing f
- `N-C_FXDMVI3EFNE_C03` | `N-C_LOSRMJ_QXKC_C04` — B locates the cause in heavy shoulder stretch/layback ('the more this lags backwards'), shown by OBS-35 far_behind_hand, while A locates it 
- `N-C_9NIRSL59AN0_C02` | `N-C_PQ793MBQE50_C03` — Both are late contacts, but B's lateness comes from a pull slot 'almost to your stomach' instead of letting the hand 'pass well in front of 

## not_visible (effort / intent, not camera-visible)（6）
- `N-P_DX8AGSIO24W` | `N-C_PUN6QIIYU_4_C04` — A's only quote announces 'four low backand tips' and states no mechanism, so the material gives nothing visible that would differ from B's l
- `N-P_LBDQPIHVXYA` | `N-C_PUN6QIIYU_4_C04` — A's only quote refers vaguely to 'this kind of a preparation' without describing anything visible, so the material gives no camera-visible f
- `N-P_WZLMV4ASG9S` | `N-C_RVFMDK_IEWO_C03` — A's listed quotes are about grip ('handle is seated in the top of your hand', 'grip tighter ... at the bottom of the handle'), which a camer
- `N-C_LWS7XXQRRDW_C02` | `N-C_PQ793MBQE50_C01` — Both show the same visible wrist break; they differ only in which muscle effort is missing ('the right hand providing the driving vector' vs
- `N-C_LWS7XXQRRDW_C02` | `N-C_PUN6QIIYU_4_C03` — Both show the same wristy swing and differ only in which source of effort is missing ('the right hand providing the driving vector' vs 'driv
- `N-C_PQ793MBQE50_C01` | `N-C_PUN6QIIYU_4_C03` — Both show the wrist breaking at contact and differ only in muscle use ('a little bit of forearm tension' vs relying on 'the small muscles of

## same_mechanism but same video (M02: cannot merge)（5）
- `N-C_FXDMVI3EFNE_C01` | `N-C_FXDMVI3EFNE_C02` — An 'upward brushing stroke' and a swing that 'just create[s] the upward vector and not ... the forward vector' are the same upward-only swin
- `N-C_FXDMVI3EFNE_C01` | `N-C_FXDMVI3EFNE_C03` — 'Upward brushing stroke' and 'excessive low-to-high swing path' are two wordings of the same steep upward swing.
- `N-C_FXDMVI3EFNE_C02` | `N-C_FXDMVI3EFNE_C03` — 'Not create the forward vector' and 'rather than shoulder drive' ('the shoulder is much better used for that back to front driving') name th
- `N-P_WFOY0RKWBTG` | `N-C_WZLMV4ASG9S_C01` — 'If the racket does go towards the ball ... not going to have velocity' / 'that first Vector is out' and 'swing it out first, not straight t
- `N-C_HNVBBPEOB3G_C01` | `N-C_HNVBBPEOB3G_C02` — 'Let your chest go first, let the arm lag here, and then let it whip through' and 'if I let this go back, I have to make sure I time this st

## same_mechanism but different stroke label (merge rule: same stroke only)（5）
- `N-C_APCKJDADZEK_C06` | `N-C_DX8AGSIO24W_C02` — Both describe the missing hip bend ('instead of hinging at the hips' vs 'the hips aren't bent'); A's text also has the knee sliding forward,
- `N-C_DX8AGSIO24W_C03` | `N-C_FOMZ8WJV3DQ_C01` — 'The stance is too narrow' ('you need a base to twist off of') and 'an overly narrow stance width' ('you're here as far as stance width') ar
- `N-C_DX8AGSIO24W_C04` | `N-C_GSHKML2MVEI_C01` — Both texts describe a static pause before the swing that removes the dynamic load ('Starting the racket statically ... removes swing momentu
- `N-C_NEPD7FIM7HI_C02` | `N-C_XXLNDJNRA4E_C03` — 'Adjust to low contact using the hand alone rather than bending at the knee and hip' and 'far below their eye level without bending their kn
- `N-C_VG8LBXOHM3E_C02` | `N-C_WFIRPMUTZRO_C04` — Both describe the same missing base ('hip tilt, knee bend, or wide-stance stability' vs 'this nice wide base and that little bit of hip flex

## same_mechanism but not cause-cause (problem node involved)（3）
- `N-P_2NN9TYMECZU` | `N-P_FXDMVI3EFNE` — Both treat the low-to-high/brushing swing as the fault and point to the same missing forward component ('press the racket the direction I'm 
- `N-P_2NN9TYMECZU` | `N-C_FXDMVI3EFNE_C01` — A's fault is a 'low-to-high' 'linear swing' and C01's is an 'upward brushing stroke'; both describe the same steep upward swing, and C01 onl
- `N-P_2NN9TYMECZU` | `N-C_FXDMVI3EFNE_C02` — 'Get out of the linear swing and into the rotational swing where I press the racket the direction I'm facing' and 'just create the upward ve

## new_obs_needed (single episode / user decision)（2）
- `N-P_2NN9TYMECZU` | `N-C_FXDMVI3EFNE_C03` — Both name a low-to-high path, but A's faulty players roll 'just the forearm, not the whole arm', while C03's low-to-high path comes 'rather 
- `N-P_DX8AGSIO24W` | `N-P_LBDQPIHVXYA` — A concerns low two-handed backhands ('four low backand tips') and B concerns forehands hit during a sprint; stroke side is visible but not i

## same_mechanism but a node is already in a merge group (extension needs M08 rule change)（1）
- `N-C_2NN9TYMECZU_C03` | `N-C_UJPUDUX0JIA_C06` — By text both are the same spacing error ('Spacing too close to the body' vs 'Positioning too close laterally to the ball'), but none of B's 

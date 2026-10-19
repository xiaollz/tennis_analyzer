# T08 剩余 37 对：为什么现有流程解决不了（tree_v10）

> 自动生成（`pipeline/t08_remaining.py`，无模型）。裁决来自 `t08_cc_review_v10.json`（Claude Code，已过 gate_cc_t08），不是结论，只是提案。

| 阻塞原因 | 对数 |
|---|---:|
| same_mechanism but same video (M02: cannot merge) | 8 |
| same_mechanism but different stroke label (merge rule: same stroke only) | 8 |
| existing_obs verdict, gate failed (no two different legal values) | 6 |
| not_visible (effort / intent, not camera-visible) | 6 |
| same_mechanism but not cause-cause (problem node involved) | 3 |
| same_mechanism but a node is already in a merge group (extension needs M08 rule change) | 3 |
| new_obs_needed (single episode / user decision) | 2 |
| same_mechanism, confirmed by 2 Gemini runs, but check_mech_merge M08 rejects it (members were not grouped together by both original merge runs) | 1 |

## same_mechanism but same video (M02: cannot merge)（8）
- `N-P_FXDMVI3EFNE` | `N-C_FXDMVI3EFNE_C01` — 'Pure brushing ... thin contact' and 'upward brushing stroke ... making contact extremely thin' are the same thing, and both rest on the sam
- `N-P_FXDMVI3EFNE` | `N-C_FXDMVI3EFNE_C02` — The problem is resolved by 'combining the upward vector ... with a backtofront driving vector', and C02 is that driving vector missing ('not
- `N-P_FXDMVI3EFNE` | `N-C_FXDMVI3EFNE_C03` — 'Pure brushing' and 'excessive low-to-high swing path rather than shoulder drive' both describe an upward path without the back-to-front sho
- `N-C_FXDMVI3EFNE_C01` | `N-C_FXDMVI3EFNE_C02` — An 'upward brushing stroke' and a swing that 'just create[s] the upward vector and not ... the forward vector' are the same upward-only swin
- `N-C_FXDMVI3EFNE_C01` | `N-C_FXDMVI3EFNE_C03` — 'Upward brushing stroke' and 'excessive low-to-high swing path' are two wordings of the same steep upward swing.
- `N-C_FXDMVI3EFNE_C02` | `N-C_FXDMVI3EFNE_C03` — 'Not create the forward vector' and 'rather than shoulder drive' ('the shoulder is much better used for that back to front driving') name th
- `N-P_WFOY0RKWBTG` | `N-C_WZLMV4ASG9S_C01` — 'If the racket does go towards the ball ... not going to have velocity' / 'that first Vector is out' and 'swing it out first, not straight t
- `N-C_HNVBBPEOB3G_C01` | `N-C_HNVBBPEOB3G_C02` — 'Let your chest go first, let the arm lag here, and then let it whip through' and 'if I let this go back, I have to make sure I time this st

## same_mechanism but different stroke label (merge rule: same stroke only)（8）
- `N-C_APCKJDADZEK_C06` | `N-C_DX8AGSIO24W_C02` — Both describe the missing hip bend ('instead of hinging at the hips' vs 'the hips aren't bent'); A's text also has the knee sliding forward,
- `N-C_APCKJDADZEK_C06` | `N-C_QXATDSEUKFY_C02` — 'Instead of hinging at the hips' and 'not tilted forward from a hip hinge' ('her hips are hinged') are the same missing hinge; B adds a swin
- `N-C_DX8AGSIO24W_C02` | `N-C_DZADHZVESDS_C04` — 'Without this double Bend in the in the hips and the knees' and 'you can also hinge at the hips ... his whole hitting apparatus is lower' de
- `N-C_DZADHZVESDS_C04` | `N-C_QXATDSEUKFY_C02` — 'Fails to hinge at the hips with a flat back' and 'not tilted forward from a hip hinge with a straight back' ('Her back is erect ... her hip
- `N-C_DX8AGSIO24W_C03` | `N-C_FOMZ8WJV3DQ_C01` — 'The stance is too narrow' ('you need a base to twist off of') and 'an overly narrow stance width' ('you're here as far as stance width') ar
- `N-C_DX8AGSIO24W_C04` | `N-C_GSHKML2MVEI_C01` — Both texts describe a static pause before the swing that removes the dynamic load ('Starting the racket statically ... removes swing momentu
- `N-C_NEPD7FIM7HI_C02` | `N-C_XXLNDJNRA4E_C03` — 'Adjust to low contact using the hand alone rather than bending at the knee and hip' and 'far below their eye level without bending their kn
- `N-C_VG8LBXOHM3E_C02` | `N-C_WFIRPMUTZRO_C04` — Both describe the same missing base ('hip tilt, knee bend, or wide-stance stability' vs 'this nice wide base and that little bit of hip flex

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

## same_mechanism but not cause-cause (problem node involved)（3）
- `N-P_2NN9TYMECZU` | `N-P_FXDMVI3EFNE` — Both treat the low-to-high/brushing swing as the fault and point to the same missing forward component ('press the racket the direction I'm 
- `N-P_2NN9TYMECZU` | `N-C_FXDMVI3EFNE_C01` — A's fault is a 'low-to-high' 'linear swing' and C01's is an 'upward brushing stroke'; both describe the same steep upward swing, and C01 onl
- `N-P_2NN9TYMECZU` | `N-C_FXDMVI3EFNE_C02` — 'Get out of the linear swing and into the rotational swing where I press the racket the direction I'm facing' and 'just create the upward ve

## same_mechanism but a node is already in a merge group (extension needs M08 rule change)（3）
- `N-C_2NN9TYMECZU_C03` | `N-C_UJPUDUX0JIA_C06` — By text both are the same spacing error ('Spacing too close to the body' vs 'Positioning too close laterally to the ball'), but none of B's 
- `N-C_APCKJDADZEK_C06` | `N-C_DZADHZVESDS_C04` — 'Instead of hinging at the hips and keeping the hips behind' and 'fails to hinge at the hips' ('you can also hinge at the hips') are the sam
- `N-C_T5MEVWBOKR4_C01` | `N-C_XLS469ZVMPU_C01` — 'Fails to freeze or decelerate the trunk right before contact' ('how slow it's moving right before contact') and 'it starts and stops you do

## new_obs_needed (single episode / user decision)（2）
- `N-P_2NN9TYMECZU` | `N-C_FXDMVI3EFNE_C03` — Both name a low-to-high path, but A's faulty players roll 'just the forearm, not the whole arm', while C03's low-to-high path comes 'rather 
- `N-P_DX8AGSIO24W` | `N-P_LBDQPIHVXYA` — A concerns low two-handed backhands ('four low backand tips') and B concerns forehands hit during a sprint; stroke side is visible but not i

## same_mechanism, confirmed by 2 Gemini runs, but check_mech_merge M08 rejects it (members were not grouped together by both original merge runs)（1）
- `N-C_DX8AGSIO24W_C02` | `N-C_QXATDSEUKFY_C02` — 'The hips aren't bent' and 'her hips are hinged, and that allows her racket path to go more low to high' describe the same hip hinge, presen

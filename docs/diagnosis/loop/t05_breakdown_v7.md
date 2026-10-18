# T05 结构拆解（tree_v7，2026-10-18）

> 范围：`tree_v7.json` 里 `must_fix=true` 且 `signature.required` 为空的节点（check_tree T05 FAIL = **178**）。
> must_fix 总数 241；有 required 的 63。
> 不改规则、不放宽阈值；本文件只分析与可选路径。

## 1. 四档归类

| 档 | 数量 | 含义 |
|---|---:|---|
| **problem_node** | 47 | 问题节点：标题改写的「本期要解决什么」。模型几乎从不给 required（问题是抽象提问，不是可见动作）。 |
| **has_NEW_OBS_candidate** | 34 | 绑定跑出了 NEW-OBS 候选，但尚未升级进目录（多数只出现在 1 期，过不了跨期规则）。 |
| **feel_or_tactics_unobservable** | 22 | 体感 / 意图 / 战术用语（feel、tension、conscious、tactic…）或绑定层全是 C：相机看不到。 |
| **unobservable_per_model** | 75 | 其余：模型两次都没给出任何 required，也没有 NEW-OBS 候选。 |

## 2. 问题节点为何几乎永远 T05

- 构建规则：每期一张卡 → 一个 problem 节点，`must_fix=true`，签名来自对该「问题」文本的 OBS 绑定。
- 问题文本是「Why … / How …」类提问，不是「触球时肘是否弯曲」类可见动作 → 模型合理判「不可观察」。
- 结果：51/51 个问题节点里绝大多数落在 T05（有 required 的问题节点在 v7 只有 4 个）。
- **这不是绑定失败，是问题节点与 T05 规则的结构性冲突。**

## 3. 不放宽阈值的可选路径（只写方案，本轮不改规则）

### 路径 A — 问题节点判定下沉到子根因（规则变更，需用户拍板）
- 改 `check_tree` T05：对 `kind=problem` 的节点，若其所有出边指向的 cause 子节点里**至少有一个**有 required，则问题节点本身不报 T05。
- 语义：确认问题 = 确认至少一个根因；问题不再需要自己的可见签名。
- 影响：v7 大约能消掉 ~47 个 T05（所有无 required 的问题节点），剩下的是真正缺签名的根因。
- 风险：一个「空问题」下面挂着无关但有签名的根因时会被放过 → 需要加「子根因必须通过该问题的 chain/edge 相连」约束。
- **本轮不改规则**；若批准，下一轮改 `check_tree` + 加变异样例。

### 路径 B — 继续补跨期 OBS（现行合法路径）
- 对 `has_NEW_OBS_candidate` 档：凑齐 ≥2 期再升（`promote_new_obs.py`）。
- 对 T08 评审提出的跨期观察点：`from_review`（已做 OBS-33…37）。
- 预期：每升一批，部分根因拿到 required，T05 绝对数缓降；问题节点仍会 T05（见路径 A）。

### 路径 C — 体感/战术期降权或不入 must_fix（规则变更）
- 对 `feel_or_tactics_unobservable`：卡片层标 `must_fix=false` 或树构建时对 stroke=general 且无可见症状的根因默认 `must_fix=false`。
- 与「不改 must_fix 绕过 T05」的硬约束冲突 → **必须用户明确批准改构建规则**，不能在绑定阶段偷偷改。

### 路径 D — 维持现状，接受 VERIFY FAIL
- T05 作为「还没绑定完」的仪表盘，不挡继续抽卡 / 补 OBS。
- PLAN_v2 停止条件仍要求最终 VERIFY PASS → 最终仍需 A 或 B+C。

## 4. 建议优先级

1. 用户拍板路径 A（问题节点 T05 下沉）——性价比最高，一次消掉约 1/4 的 T05。
2. 继续路径 B（跨期 OBS）——已在做。
3. 路径 C 仅在用户批准「战术/心理期根因可 must_fix=false」后动。

## 5. 明细（节点 id）

### problem_node（47）
- `N-P_1W1ROPE5L0K` (problem) Why should players serve in-to-out rather than low-to-high?
- `N-P_2NN9TYMECZU` (problem) Why do pros swing around rather than low-to-high?
- `N-P_42BFBKSTGB4` (problem) Why can't players use their good practice swings in matches, and how can they ma
- `N-P_5_IA7C4XZFU` (problem) How does string bed tension/stiffness expose swing speed mismatches and cause sh
- `N-P_7_JPQHG4G64` (problem) Why do recreational players stop moving or get caught flat-footed between shots?
- `N-P_8R09TLIP_AK` (problem) Why do players take high risks without getting rewards, and what patterns create
- `N-P_9NIRSL59AN0` (problem) Why Does the Modern Topspin Forehand Paradoxically Feel Like Hitting Down?
- `N-P_APCKJDADZEK` (problem) How do pros step into their shots differently from amateur players?
- `N-P_APPDZQXH_2K` (problem) How to Master Tennis's Most Important Skill?
- `N-P_CIIEYN42O4G` (problem) What is the #1 backscratch mistake on the serve, and how do you fix it?
- `N-P_DDYKUNZTDYU` (problem) How do you find and utilize the 'pull slot' on a one-handed backhand to generate
- `N-P_FBJNFBTZOZE` (problem) How can players prevent choking and freezing up during high-pressure moments in 
- `N-P_FJPR9PS02TM` (problem) Why don't 99% of tennis grips work?
- `N-P_FOMZ8WJV3DQ` (problem) Why do players struggle to move explosively and stay balanced on the tennis cour
- `N-P_GSHKML2MVEI` (problem) How should you time your forearm rolling back relative to trunk rotation on the 
- `N-P_HNVBBPEOB3G` (problem) How does engaging the chest improve racket face control and timing on the foreha
- `N-P_JZCA_KU7YHK` (problem) Why does a good practice forehand fall apart during matches, leading to unexpect
- `N-P_LCKDP2GKR_Y` (problem) Why does your wrist hurt on the forehand, and how can you fix it?
- `N-P_LOSRMJ_QXKC` (problem) How do contrasting forehand mechanics—shoulder whip versus chest press—affect sp
- `N-P_LWS7XXQRRDW` (problem) How do you build a simple, controllable foundation for the two-handed backhand a
- `N-P_MCCB_RFYD0W` (problem) How does the non-dominant side function to stabilize the forehand?
- `N-P_NEPD7FIM7HI` (problem) How can players improve their backhand slice using fundamentals from Federer's t
- `N-P_P41GGC_Y5XK` (problem) How does your hand control every shot without getting tight or injured?
- `N-P_PFDQNFF_VYA` (problem) Why is your overhead and serve so slow and weak?
- `N-P_PJVOQKQH9DA` (problem) Why do players struggle to dominate and close out matches against opponents they
- `N-P_PQ793MBQE50` (problem) How does scapular retraction power the one-handed backhand?
- `N-P_PSIDJEI5BNI` (problem) How can players consistently attack and crush slow balls?
- `N-P_PUN6QIIYU_4` (problem) How can you improve consistency and aggression on the two-handed backhand?
- `N-P_PWKGLKRM73O` (problem) Why are recreational players unable to drive high balls or harness body rotation
- `N-P_QSZZ0N4FRB4` (problem) Why do you get tight and hesitate on your forehand, and how can you fix it to sw
- `N-P_QXATDSEUKFY` (problem) How can you rip low backhands effectively?
- `N-P_RVFMDK_IEWO` (problem) How can players find their hand slot so that contact feels effortless, solid, an
- `N-P_T5MEVWBOKR4` (problem) How does Carlos Alcaraz hit harder than everyone else on the forehand?
- `N-P_TOARB3WKJCS` (problem) What is the most common ball-machine mistake and how can you fix it?
- `N-P_TZGAWR3QGV0` (problem) How can players emulate Jannik Sinner's explosive yet controlled high-speed fore
- `N-P_UB6SBA_KX9E` (problem) How can tennis players properly sequence trunk and hip rotation to generate line
- `N-P_UJPUDUX0JIA` (problem) Why does your forehand keep disappearing, and how do you build a repeatable, con
- `N-P_VG8LBXOHM3E` (problem) Why do you get tight and struggle in matches despite hitting well in practice ra
- `N-P_V_QKILD4V_W` (problem) Which visual tracking strategy allows a player to see the serve best and time th
- `N-P_WFIRPMUTZRO` (problem) How do you unlock effortless rotational power and trunk control on the forehand?
- `N-P_WOVEWH7FFFC` (problem) How can an adult player optimize elastic loading and timing on the forehand and 
- `N-P_WVA4XQPCAQS` (problem) How can you find your perfect contact point on any stroke?
- `N-P_WZLMV4ASG9S` (problem) Why doesn't your racket flip properly on the forehand, and how do you achieve an
- `N-P_XLS469ZVMPU` (problem) How can you fix forehand over-rotation and control your body through the hitting
- `N-P_XXLNDJNRA4E` (problem) How can you use peripheral vision and proper mechanics to hit clean reaction vol
- `N-P_YBLIT9PNG2U` (problem) How should you direct your racket trajectory to hit an effective kick serve?
- `N-P__QU1LOWKLAW` (problem) How should singles players balance patience and ruthlessness to avoid overplayin

### has_NEW_OBS_candidate（34）
- `N-C_1W1ROPE5L0K_C01` (cause) Approaching the ball from the top or outside rather than from the inside prevent
- `N-C_1W1ROPE5L0K_C02` (cause) Failing to roll the shoulder back and forward leaves the arm stationary and prev
- `N-C_2NN9TYMECZU_C04` (cause) Hand choked up too far on the handle rather than resting pinky on the flare of t
- `N-C_APCKJDADZEK_C03` (cause) The player drives force through the heel and inside edge of the front foot where
- `N-C_APCKJDADZEK_C05` (cause) The player rushes immediately onto the front foot without first balancing and co
- `N-C_APPDZQXH_2K_C02` (cause) Catching or striking the ball far away from eye level makes it much harder to ju
- `N-C_DDYKUNZTDYU_C02` (cause) Failing to throw and deliver the racket accurately into the pull slot before pul
- `N-C_DX8AGSIO24W_C03` (cause) The stance is too narrow, failing to provide a stable wide base to twist the tru
- `N-C_DZADHZVESDS_C01` (cause) The player initiates forward rotational acceleration too early while the hand an
- `N-C_FJPR9PS02TM_C01` (cause) Players do not use the off-hand to seat the racket back into the palm or V befor
- `N-C_FOMZ8WJV3DQ_C01` (cause) Players maintain an overly narrow stance width, restricting leg push vectors to 
- `N-C_FOMZ8WJV3DQ_C03` (cause) Lack of a triple bend at the hips, knees, and ankles disengages the lower-body m
- `N-C_GSHKML2MVEI_C02` (cause) Treating the backswing as an explosive swing and actively throwing the hand back
- `N-C_HNVBBPEOB3G_C01` (cause) Allowing the chest to turn first while the arm lags behind forces the arm throug
- `N-C_LBDQPIHVXYA_C02` (cause) The player runs without pre-setting the racket and attempts to prepare and swing
- `N-C_LCKDP2GKR_Y_C03` (cause) Forward back-to-front racket head speed dissipates immediately after contact rel
- `N-C_NEPD7FIM7HI_C03` (cause) Rounding the back instead of maintaining a neutral spine while bending at the hi
- `N-C_PFDQNFF_VYA_C01` (cause) The player attempts to throw the racket up or down-and-up directly at the ball r
- `N-C_PFDQNFF_VYA_C03` (cause) The racket is not nestled deep into the V of the hand and index palm, preventing
- `N-C_PQ793MBQE50_C02` (cause) Holding the racket with knuckles slanted offside at a ~135 degree volley angle r
- `N-C_PSIDJEI5BNI_C01` (cause) Slow balls do not penetrate through the court and drop straight down, requiring 
- `N-C_PSIDJEI5BNI_C03` (cause) Players prepare their hand way below the incoming ball to brush topspin instead 
- `N-C_PWKGLKRM73O_C01` (cause) Rotational energy is not harnessed because energy cannot pass through the should
- `N-C_T5MEVWBOKR4_C02` (cause) The arm-to-racket momentum handoff fails because the racket does not overtake th
- `N-C_TOARB3WKJCS_C02` (cause) The prolonged interval between feeds makes continuous foot movement unnatural, c
- `N-C_TOARB3WKJCS_C03` (cause) Practicing with a pause between shots prevents the player from linking the landi
- `N-C_TZGAWR3QGV0_C02` (cause) Players attempt a large, gravity-driven loop up to shoulder height, overcomplica
- `N-C_UJPUDUX0JIA_C05` (cause) Upper trapezius shrugging and lack of isometric engagement in the mid-traps and 
- `N-C_WOVEWH7FFFC_C01` (cause) The racket tip does not have enough inward momentum during preparation to spring
- `N-C_WOVEWH7FFFC_C04` (cause) The racket fails to load into-out on the serve because there is insufficient str
- `N-C_XLS469ZVMPU_C03` (cause) The player fails to let the free non-dominant foot kick upward or across the bod
- `N-C_XLS469ZVMPU_C04` (cause) The player fails to produce static pressing force into the court through the lea
- `N-C_XXLNDJNRA4E_C01` (cause) Players try to take a backswing on reaction volleys instead of hovering the hand
- `N-C_YBLIT9PNG2U_C03` (cause) Players do not utilize the high degree of orthogonality required between the arm

### feel_or_tactics_unobservable（22）
- `N-C_2NN9TYMECZU_C01` (cause) Brain is consciously focused on swinging upward or lifting the entire racket sys
- `N-C_5_IA7C4XZFU_C03` (cause) Swinging faster than the playable threshold of a soft or lost-tension string bed
- `N-C_7_JPQHG4G64_C01` (cause) Players post-process their own shot and critique technique rather than locking i
- `N-C_9NIRSL59AN0_C01` (cause) Trying consciously to hit the ball up over the net causes the brain to alter ori
- `N-C_DZADHZVESDS_C02` (cause) The player fails to mentally let go during the backswing, reflexively gripping t
- `N-C_FJPR9PS02TM_C03` (cause) Players fall into the false binary of either death-gripping the handle (locking 
- `N-C_FOMZ8WJV3DQ_C04` (cause) The brain subconsciously forces a narrow stance to protect unprepared soft tissu
- `N-C_HNVBBPEOB3G_C03` (cause) Simultaneously loosening the upper chest/arm and the wrist deprives the stroke o
- `N-C_HNVBBPEOB3G_C04` (cause) The brain subconsciously throttles swing speed when attempting to make the court
- `N-C_P41GGC_Y5XK_C03` (cause) Players attempt to consciously micromanage individual finger mechanics instead o
- `N-C_PSIDJEI5BNI_C05` (cause) Players fail to watch the ball hit the court because its slow pace feels unthrea
- `N-C_QSZZ0N4FRB4_C01` (cause) The forward flick and roll through contact occurs in under half a second, making
- `N-C_QSZZ0N4FRB4_C02` (cause) The player lacks trust in the release because they have not identified a single 
- `N-C_QSZZ0N4FRB4_C03` (cause) The player fails to establish the precise comfortable strike distance relative t
- `N-C_RVFMDK_IEWO_C02` (cause) Players focus mentally on manipulating the racket head rather than feeling and d
- `N-C_RVFMDK_IEWO_C04` (cause) Players fail to coordinate timing by neither waiting patiently for the ball to a
- `N-C_TOARB3WKJCS_C01` (cause) Feeding balls too slowly leaves several seconds of idle time, prompting rational
- `N-C_UB6SBA_KX9E_C03` (cause) Players mentally visualize rotating their shoulder line continuously all the way
- `N-C_UJPUDUX0JIA_C07` (cause) Gripping too loosely (0/10) or with gaps between the handle and palm, causing th
- `N-C_WOVEWH7FFFC_C03` (cause) The player attempts to consciously control and steer the back of the swing rathe
- `N-C_YBLIT9PNG2U_C01` (cause) Players fundamentally fail to create proper kick serve racket motion because the
- `N-C__QU1LOWKLAW_C01` (cause) Players overplay and press because they feel they should never lose points to an

### unobservable_per_model（75）
- `N-C_1W1ROPE5L0K_C04` (cause) Intuitive panic when hitting long leads players to deliberately yank the entire 
- `N-C_2NN9TYMECZU_C06` (cause) Failing to coil with an open hip and ground force, leading player to search for 
- `N-C_42BFBKSTGB4_C01` (cause) Players naturally balance with their arms and compensate with their upper body r
- `N-C_42BFBKSTGB4_C02` (cause) Players lack an intuitive kinesthetic sense of their racket contact point in spa
- `N-C_42BFBKSTGB4_C03` (cause) Players hyperfocus on the sending piece of the equation (the swing) and neglect 
- `N-C_5_IA7C4XZFU_C01` (cause) At higher swing speeds on a soft string bed, the ball sinks deep into the pocket
- `N-C_7_JPQHG4G64_C02` (cause) Players assume the opponent cannot retrieve the ball and guess the point will en
- `N-C_7_JPQHG4G64_C03` (cause) The attention pattern ends abruptly before the ball is confirmed dead, causing p
- `N-C_8R09TLIP_AK_C02` (cause) Players misinterpret playing with margin and patience as an instruction to slow 
- `N-C_8R09TLIP_AK_C03` (cause) Shot selection suffers from poor risk-to-reward ratios, opting for high-difficul
- `N-C_8R09TLIP_AK_C04` (cause) Players grant opponents too much respect on defense and attempt wild bailout sho
- `N-C_9NIRSL59AN0_C04` (cause) Over-rotating the torso early in the swing prevents pressing across the body and
- `N-C_APPDZQXH_2K_C01` (cause) Over-focusing on swing mechanics or silently verbalizing 'watch the ball' detrac
- `N-C_APPDZQXH_2K_C03` (cause) Over-focusing attention on the hand and handle mechanics causes the player to dr
- `N-C_APPDZQXH_2K_C04` (cause) Visual attention becomes overly anchored on the target, leaving the player unabl
- `N-C_CIIEYN42O4G_C01` (cause) Players mistakenly treat the backscratch position as the maximum power launch po
- `N-C_CIIEYN42O4G_C02` (cause) Players focus on the shoulder as the final link of the kinetic chain rather than
- `N-C_CIIEYN42O4G_C03` (cause) Players lack awareness that the forearm must first roll out and stretch into sup
- `N-C_DDYKUNZTDYU_C01` (cause) Failing to initiate an active two-handed pull driven by squeezing the upper back
- `N-C_DX8AGSIO24W_C02` (cause) Players stand too upright on low balls without activating a double bend at both 
- `N-C_FBJNFBTZOZE_C02` (cause) A psychological flight response leads to sideways recovery and waiting passively
- `N-C_FBJNFBTZOZE_C03` (cause) Remaining at the baseline allows athletic opponents time to recover and counter,
- `N-C_FBJNFBTZOZE_C04` (cause) Fixation on match outcome rather than acceptance creates anxiety that blocks agg
- `N-C_FJPR9PS02TM_C02` (cause) The racket moves independently inside the hand rather than moving as a coupled u
- `N-C_FXDMVI3EFNE_C04` (cause) Training devices like the Topspin Pro have physical wings that disincentivize pl
- `N-C_HNVBBPEOB3G_C02` (cause) Disconnecting the arm from the chest creates an extra degree of freedom that int
- `N-C_JZCA_KU7YHK_C01` (cause) Stroke mechanics break down the moment the incoming ball is not where expected o
- `N-C_JZCA_KU7YHK_C02` (cause) The player lacks the adaptable skill set to coordinate their body and produce th
- `N-C_JZCA_KU7YHK_C03` (cause) Practice sessions consist of sterile, repetitive drills that lack chaotic unpred
- `N-C_JZCA_KU7YHK_C04` (cause) The player operates under the misconception that athletic skill means repeating 
- `N-C_LBDQPIHVXYA_C01` (cause) The hitting arm apparatus is not stabilized because the player fails to engage t
- `N-C_LCKDP2GKR_Y_C02` (cause) Small muscles in the wrist and hand are not strong enough to actively drive the 
- `N-C_LWS7XXQRRDW_C02` (cause) Failing to engage the right hand and shoulder to pull the driving vector leaves 
- `N-C_LWS7XXQRRDW_C04` (cause) Hitting at improper contact depth jams rotation or exhausts body release before 
- `N-C_MCCB_RFYD0W_C01` (cause) The left side of the chest must actively press back across the body to oppose ou
- `N-C_MCCB_RFYD0W_C02` (cause) The initial rotational explosion that starts the swing sends the left side outwa
- `N-C_MCCB_RFYD0W_C03` (cause) Failing to actively press inward leaves outward momentum unchecked, destroying s
- `N-C_NEPD7FIM7HI_C04` (cause) Attempting to copy a pro's exact swing angle instead of calibrating to one's own
- `N-C_NEPD7FIM7HI_C05` (cause) Attempting to stretch and release the shoulder for extra velocity before establi
- `N-C_P41GGC_Y5XK_C01` (cause) Players attempt to generate power and force actively from the hand rather than u
- `N-C_P41GGC_Y5XK_C02` (cause) Players fail to sense racket head orientation through hand pressure sensors, lea
- `N-C_PFDQNFF_VYA_C02` (cause) The brain is psychologically anchored on clearing the net, preventing a committe
- `N-C_PFDQNFF_VYA_C04` (cause) The player fails to recognize that the overhead/serve uses the exact same outwar
- `N-C_PJVOQKQH9DA_C01` (cause) Players believe they must execute miraculous, razor-thin knifing or driving shot
- `N-C_PJVOQKQH9DA_C02` (cause) Players' egos prevent them from accepting that an inferior opponent hit a good s
- `N-C_PJVOQKQH9DA_C03` (cause) Players become emotionally perturbed and distracted when an opponent attacks the
- `N-C_PSIDJEI5BNI_C04` (cause) Players react late to short balls, allowing them to drop low below net height wh
- `N-C_PWKGLKRM73O_C02` (cause) The brain fears hitting into the ground and forces the arm to swing underhand an
- `N-C_PWKGLKRM73O_C04` (cause) The brain matches arm movement to the large, visible low-to-high motion of the p
- `N-C_RVFMDK_IEWO_C01` (cause) Players hit the ball using active slow-twitch muscular contraction in the forear
- `N-C_TZGAWR3QGV0_C01` (cause) The chest pectoral muscles fail to engage sufficiently to overcome inertia, allo
- `N-C_UB6SBA_KX9E_C01` (cause) Players incorrectly assume the rotational command originates from the legs or th
- `N-C_UB6SBA_KX9E_C02` (cause) When attempting to imitate compact swings, players disrupt the acceleration rhyt
- `N-C_VG8LBXOHM3E_C01` (cause) Low-variety cooperative rallying allows successful ball outcomes without demandi
- `N-C_VG8LBXOHM3E_C03` (cause) The player treats the swing as a single monolithic motion calibrated to one exac
- `N-C_VG8LBXOHM3E_C04` (cause) Common match-specific awkward patterns, such as low short approach shots, are ha
- `N-C_V_QKILD4V_W_C02` (cause) The returner fails to see the ball early during the initial phase of the serve t
- `N-C_V_QKILD4V_W_C03` (cause) Fixating on the server's racket face or body parts fails because the racket move
- `N-C_V_QKILD4V_W_C04` (cause) Human serve motions vary significantly between players and speeds, unlike the ba
- `N-C_WFIRPMUTZRO_C01` (cause) The player treats the shoulder and hip as isolated parts rather than diagonal co
- `N-C_WFIRPMUTZRO_C04` (cause) Lacking a wide base and hip flexion disables glute support and shifts rotational
- `N-C_WOVEWH7FFFC_C02` (cause) The hitting hand and wrist are held too tight, actively pressing the racket and 
- `N-C_WVA4XQPCAQS_C01` (cause) The player's contact point changes depending on their degree of rotation and bod
- `N-C_WVA4XQPCAQS_C02` (cause) The player does not visually register and memorize the exact spatial spot where 
- `N-C_WVA4XQPCAQS_C03` (cause) Catching the ball in a position where the musculoskeletal alignment cannot gener
- `N-C_WZLMV4ASG9S_C02` (cause) Handle is not firmly seated in the top of the hand, losing visceral tactile conn
- `N-C_WZLMV4ASG9S_C03` (cause) Squeezing tightly with the bottom of the hand too early before the natural hinge
- `N-C_WZLMV4ASG9S_C04` (cause) Confusing the rolling follow-through / windshield wiper action with the back-to-
- `N-C_WZLMV4ASG9S_C05` (cause) Failure to stabilize the scapula with back and side muscles (lats, mid/lower tra
- `N-C_WZLMV4ASG9S_C06` (cause) Trying to initiate the racket flip directly out of a high backswing loop rather 
- `N-C_XLS469ZVMPU_C02` (cause) The player fails to kick the hitting leg backward behind the body to arrest rota
- `N-C_XXLNDJNRA4E_C02` (cause) Players attempt to track the ball immediately off the strings with foveal focus 
- `N-C__QU1LOWKLAW_C02` (cause) Players fail to calibrate anxiety properly, either over-hyping and blasting neut
- `N-C__QU1LOWKLAW_C03` (cause) Aiming too close to lines and corners removes the safety margin, turning small m
- `N-C__QU1LOWKLAW_C04` (cause) Under big-moment pressure, players alter their game by overhitting or becoming t

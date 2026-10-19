# T05 结构拆解（tree_v9）

> 自动生成（`pipeline/t05_breakdown.py`，无模型）。T05 FAIL = **154**（路径 A 已生效）；must_fix 节点 241，有 required 的节点 62。

| 档 | 数量 | 含义 |
|---|---:|---|
| problem_node | 22 | 问题节点，其下没有任何带 required 的根因 |
| has_NEW_OBS_candidate | 37 | 绑定产生了 NEW-OBS 候选但尚未升级为观察点 |
| feel_or_tactics | 58 | 体感/意图/战术/心理类描述，相机看不到 |
| unobservable_per_model | 37 | 其余：模型两次都没给 required，也没有候选 |

has_NEW_OBS_candidate 中，候选簇跨 ≥2 期的节点：**2**；仅单期：**35**。

## problem_node（22）
- `N-P_1W1ROPE5L0K` (problem) Why should players serve in-to-out rather than low-to-high?
- `N-P_42BFBKSTGB4` (problem) Why can't players use their good practice swings in matches, and how can they master recei
- `N-P_7_JPQHG4G64` (problem) Why do recreational players stop moving or get caught flat-footed between shots? · 簇 NEW-OBS-06,NEW-OBS-73
- `N-P_APPDZQXH_2K` (problem) How to Master Tennis's Most Important Skill?
- `N-P_BOMJDZUFHM4` (problem) How should players manage anxiety during high-pressure moments in tennis?
- `N-P_CIIEYN42O4G` (problem) What is the #1 backscratch mistake on the serve, and how do you fix it?
- `N-P_DDYKUNZTDYU` (problem) How do you find and utilize the 'pull slot' on a one-handed backhand to generate power and
- `N-P_FBJNFBTZOZE` (problem) How can players prevent choking and freezing up during high-pressure moments in tennis?
- `N-P_FJPR9PS02TM` (problem) Why don't 99% of tennis grips work?
- `N-P_HNVBBPEOB3G` (problem) How does engaging the chest improve racket face control and timing on the forehand?
- `N-P_JZCA_KU7YHK` (problem) Why does a good practice forehand fall apart during matches, leading to unexpected losses?
- `N-P_MCCB_RFYD0W` (problem) How does the non-dominant side function to stabilize the forehand? · 簇 NEW-OBS-07
- `N-P_P41GGC_Y5XK` (problem) How does your hand control every shot without getting tight or injured?
- `N-P_PFDQNFF_VYA` (problem) Why is your overhead and serve so slow and weak? · 簇 NEW-OBS-14
- `N-P_PJVOQKQH9DA` (problem) Why do players struggle to dominate and close out matches against opponents they are bette
- `N-P_RVFMDK_IEWO` (problem) How can players find their hand slot so that contact feels effortless, solid, and controll
- `N-P_TOARB3WKJCS` (problem) What is the most common ball-machine mistake and how can you fix it?
- `N-P_V_QKILD4V_W` (problem) Which visual tracking strategy allows a player to see the serve best and time the return e
- `N-P_WOVEWH7FFFC` (problem) How can an adult player optimize elastic loading and timing on the forehand and serve?
- `N-P_WVA4XQPCAQS` (problem) How can you find your perfect contact point on any stroke?
- `N-P_YBLIT9PNG2U` (problem) How should you direct your racket trajectory to hit an effective kick serve? · 簇 NEW-OBS-08
- `N-P__QU1LOWKLAW` (problem) How should singles players balance patience and ruthlessness to avoid overplaying?

## has_NEW_OBS_candidate（37）
- `N-C_1W1ROPE5L0K_C01` (cause) Approaching the ball from the top or outside rather than from the inside prevents forearm  · 簇 NEW-OBS-09
- `N-C_1W1ROPE5L0K_C02` (cause) Failing to roll the shoulder back and forward leaves the arm stationary and prevents stret · 簇 NEW-OBS-10
- `N-C_2NN9TYMECZU_C04` (cause) Hand choked up too far on the handle rather than resting pinky on the flare of the butt ca · 簇 NEW-OBS-11
- `N-C_42BFBKSTGB4_C01` (cause) Players naturally balance with their arms and compensate with their upper body rather than · 簇 NEW-OBS-12
- `N-C_9NIRSL59AN0_C04` (cause) Over-rotating the torso early in the swing prevents pressing across the body and leads to  · 簇 NEW-OBS-69
- `N-C_APCKJDADZEK_C03` (cause) The player drives force through the heel and inside edge of the front foot where it first  · 簇 NEW-OBS-65
- `N-C_APCKJDADZEK_C05` (cause) The player rushes immediately onto the front foot without first balancing and coiling thei · 簇 NEW-OBS-55
- `N-C_APPDZQXH_2K_C02` (cause) Catching or striking the ball far away from eye level makes it much harder to judge depth  · 簇 NEW-OBS-70
- `N-C_DX8AGSIO24W_C03` (cause) The stance is too narrow, failing to provide a stable wide base to twist the trunk against · 簇 NEW-OBS-03 (跨期:NEW-OBS-03)
- `N-C_DZADHZVESDS_C01` (cause) The player initiates forward rotational acceleration too early while the hand and racket a · 簇 NEW-OBS-56
- `N-C_FJPR9PS02TM_C01` (cause) Players do not use the off-hand to seat the racket back into the palm or V before each swi · 簇 NEW-OBS-57
- `N-C_FOMZ8WJV3DQ_C01` (cause) Players maintain an overly narrow stance width, restricting leg push vectors to vertical r · 簇 NEW-OBS-05
- `N-C_FOMZ8WJV3DQ_C03` (cause) Lack of a triple bend at the hips, knees, and ankles disengages the lower-body musculature · 簇 NEW-OBS-13
- `N-C_GSHKML2MVEI_C02` (cause) Treating the backswing as an explosive swing and actively throwing the hand back with exce · 簇 NEW-OBS-01
- `N-C_HNVBBPEOB3G_C01` (cause) Allowing the chest to turn first while the arm lags behind forces the arm through an exten · 簇 NEW-OBS-58
- `N-C_LBDQPIHVXYA_C02` (cause) The player runs without pre-setting the racket and attempts to prepare and swing right nex · 簇 NEW-OBS-60
- `N-C_LBDQPIHVXYA_C03` (cause) The player prepares the hand high at chest level on low balls, requiring an extreme and er · 簇 NEW-OBS-61
- `N-C_LCKDP2GKR_Y_C03` (cause) Forward back-to-front racket head speed dissipates immediately after contact release, leav · 簇 NEW-OBS-59
- `N-C_MCCB_RFYD0W_C02` (cause) The initial rotational explosion that starts the swing sends the left side outward with st · 簇 NEW-OBS-72
- `N-C_NEPD7FIM7HI_C03` (cause) Rounding the back instead of maintaining a neutral spine while bending at the hips and kne · 簇 NEW-OBS-15,NEW-OBS-74 (跨期:NEW-OBS-15)
- `N-C_PFDQNFF_VYA_C01` (cause) The player attempts to throw the racket up or down-and-up directly at the ball rather than · 簇 NEW-OBS-54
- `N-C_PFDQNFF_VYA_C03` (cause) The racket is not nestled deep into the V of the hand and index palm, preventing relaxed c · 簇 NEW-OBS-75
- `N-C_PQ793MBQE50_C02` (cause) Holding the racket with knuckles slanted offside at a ~135 degree volley angle rather than · 簇 NEW-OBS-62
- `N-C_PQ793MBQE50_C04` (cause) Initiating the stroke solely with one hand instead of executing a two-handed forward movem · 簇 NEW-OBS-98,NEW-OBS-99
- `N-C_PSIDJEI5BNI_C03` (cause) Players prepare their hand way below the incoming ball to brush topspin instead of matchin · 簇 NEW-OBS-76,NEW-OBS-77
- `N-C_QXATDSEUKFY_C01` (cause) The player does not sit into a deep back knee bend, preventing them from maintaining balan · 簇 NEW-OBS-16
- `N-C_QXATDSEUKFY_C02` (cause) The torso is not tilted forward from a hip hinge with a straight back, preventing the swin · 簇 NEW-OBS-78
- `N-C_TOARB3WKJCS_C01` (cause) Feeding balls too slowly leaves several seconds of idle time, prompting rational verbal th · 簇 NEW-OBS-100,NEW-OBS-101
- `N-C_TOARB3WKJCS_C02` (cause) The prolonged interval between feeds makes continuous foot movement unnatural, conditionin · 簇 NEW-OBS-63
- `N-C_TOARB3WKJCS_C03` (cause) Practicing with a pause between shots prevents the player from linking the landing recover · 簇 NEW-OBS-64
- `N-C_UJPUDUX0JIA_C05` (cause) Upper trapezius shrugging and lack of isometric engagement in the mid-traps and serratus a · 簇 NEW-OBS-66
- `N-C_WOVEWH7FFFC_C01` (cause) The racket tip does not have enough inward momentum during preparation to spring-load the  · 簇 NEW-OBS-52
- `N-C_WOVEWH7FFFC_C04` (cause) The racket fails to load into-out on the serve because there is insufficient stretch in th · 簇 NEW-OBS-53
- `N-C_WZLMV4ASG9S_C06` (cause) Trying to initiate the racket flip directly out of a high backswing loop rather than letti · 簇 NEW-OBS-102
- `N-C_XLS469ZVMPU_C03` (cause) The player fails to let the free non-dominant foot kick upward or across the body to slow  · 簇 NEW-OBS-67
- `N-C_XLS469ZVMPU_C04` (cause) The player fails to produce static pressing force into the court through the lead foot, al · 簇 NEW-OBS-68
- `N-C_XXLNDJNRA4E_C01` (cause) Players try to take a backswing on reaction volleys instead of hovering the hand behind th · 簇 NEW-OBS-04

## feel_or_tactics（58）
- `N-C_1W1ROPE5L0K_C04` (cause) Intuitive panic when hitting long leads players to deliberately yank the entire arm down i
- `N-C_2NN9TYMECZU_C01` (cause) Brain is consciously focused on swinging upward or lifting the entire racket system for he
- `N-C_42BFBKSTGB4_C02` (cause) Players lack an intuitive kinesthetic sense of their racket contact point in space, so the
- `N-C_42BFBKSTGB4_C03` (cause) Players hyperfocus on the sending piece of the equation (the swing) and neglect deliberate
- `N-C_5_IA7C4XZFU_C03` (cause) Swinging faster than the playable threshold of a soft or lost-tension string bed.
- `N-C_7_JPQHG4G64_C01` (cause) Players post-process their own shot and critique technique rather than locking into the op
- `N-C_7_JPQHG4G64_C02` (cause) Players assume the opponent cannot retrieve the ball and guess the point will end, causing
- `N-C_7_JPQHG4G64_C03` (cause) The attention pattern ends abruptly before the ball is confirmed dead, causing players to 
- `N-C_8R09TLIP_AK_C02` (cause) Players misinterpret playing with margin and patience as an instruction to slow down their
- `N-C_8R09TLIP_AK_C03` (cause) Shot selection suffers from poor risk-to-reward ratios, opting for high-difficulty flat sh
- `N-C_8R09TLIP_AK_C04` (cause) Players grant opponents too much respect on defense and attempt wild bailout shots rather 
- `N-C_9NIRSL59AN0_C01` (cause) Trying consciously to hit the ball up over the net causes the brain to alter orientation a
- `N-C_APPDZQXH_2K_C03` (cause) Over-focusing attention on the hand and handle mechanics causes the player to drop the lat
- `N-C_CIIEYN42O4G_C01` (cause) Players mistakenly treat the backscratch position as the maximum power launch point rather
- `N-C_CIIEYN42O4G_C02` (cause) Players focus on the shoulder as the final link of the kinetic chain rather than recognizi
- `N-C_CIIEYN42O4G_C03` (cause) Players lack awareness that the forearm must first roll out and stretch into supination to
- `N-C_DZADHZVESDS_C02` (cause) The player fails to mentally let go during the backswing, reflexively gripping tight to se
- `N-C_FBJNFBTZOZE_C01` (cause) Anxiety triggers a physical freeze response with excess forearm and upper body tension, sl
- `N-C_FBJNFBTZOZE_C03` (cause) Remaining at the baseline allows athletic opponents time to recover and counter, requiring
- `N-C_FBJNFBTZOZE_C04` (cause) Fixation on match outcome rather than acceptance creates anxiety that blocks aggressive co
- `N-C_FJPR9PS02TM_C03` (cause) Players fall into the false binary of either death-gripping the handle (locking the wrist 
- `N-C_HNVBBPEOB3G_C03` (cause) Simultaneously loosening the upper chest/arm and the wrist deprives the stroke of upper-bo
- `N-C_JZCA_KU7YHK_C01` (cause) Stroke mechanics break down the moment the incoming ball is not where expected or when the
- `N-C_JZCA_KU7YHK_C04` (cause) The player operates under the misconception that athletic skill means repeating the identi
- `N-C_MCCB_RFYD0W_C01` (cause) The left side of the chest must actively press back across the body to oppose outward rota
- `N-C_NEPD7FIM7HI_C04` (cause) Attempting to copy a pro's exact swing angle instead of calibrating to one's own string be
- `N-C_NEPD7FIM7HI_C05` (cause) Attempting to stretch and release the shoulder for extra velocity before establishing rack
- `N-C_P41GGC_Y5XK_C02` (cause) Players fail to sense racket head orientation through hand pressure sensors, leaving them 
- `N-C_P41GGC_Y5XK_C03` (cause) Players attempt to consciously micromanage individual finger mechanics instead of hitting 
- `N-C_PFDQNFF_VYA_C02` (cause) The brain is psychologically anchored on clearing the net, preventing a committed and free
- `N-C_PJVOQKQH9DA_C01` (cause) Players believe they must execute miraculous, razor-thin knifing or driving shots under pr
- `N-C_PJVOQKQH9DA_C02` (cause) Players' egos prevent them from accepting that an inferior opponent hit a good shot, provo
- `N-C_PJVOQKQH9DA_C03` (cause) Players become emotionally perturbed and distracted when an opponent attacks their weaker 
- `N-C_PSIDJEI5BNI_C01` (cause) Slow balls do not penetrate through the court and drop straight down, requiring the hitter
- `N-C_PSIDJEI5BNI_C05` (cause) Players fail to watch the ball hit the court because its slow pace feels unthreatening, lo
- `N-C_PWKGLKRM73O_C02` (cause) The brain fears hitting into the ground and forces the arm to swing underhand and low-to-h
- `N-C_PWKGLKRM73O_C04` (cause) The brain matches arm movement to the large, visible low-to-high motion of the pro's racke
- `N-C_QSZZ0N4FRB4_C01` (cause) The forward flick and roll through contact occurs in under half a second, making it far to
- `N-C_QSZZ0N4FRB4_C02` (cause) The player lacks trust in the release because they have not identified a single comfortabl
- `N-C_QSZZ0N4FRB4_C03` (cause) The player fails to establish the precise comfortable strike distance relative to the body
- `N-C_RVFMDK_IEWO_C02` (cause) Players focus mentally on manipulating the racket head rather than feeling and directing t
- `N-C_RVFMDK_IEWO_C04` (cause) Players fail to coordinate timing by neither waiting patiently for the ball to arrive in t
- `N-C_UB6SBA_KX9E_C01` (cause) Players incorrectly assume the rotational command originates from the legs or the ground r
- `N-C_UB6SBA_KX9E_C02` (cause) When attempting to imitate compact swings, players disrupt the acceleration rhythm by losi
- `N-C_UB6SBA_KX9E_C03` (cause) Players mentally visualize rotating their shoulder line continuously all the way across fr
- `N-C_UJPUDUX0JIA_C07` (cause) Gripping too loosely (0/10) or with gaps between the handle and palm, causing the racket t
- `N-C_VG8LBXOHM3E_C01` (cause) Low-variety cooperative rallying allows successful ball outcomes without demanding match-e
- `N-C_VG8LBXOHM3E_C03` (cause) The player treats the swing as a single monolithic motion calibrated to one exact ball ins
- `N-C_VG8LBXOHM3E_C04` (cause) Common match-specific awkward patterns, such as low short approach shots, are habitually a
- `N-C_WFIRPMUTZRO_C01` (cause) The player treats the shoulder and hip as isolated parts rather than diagonal connections 
- `N-C_WOVEWH7FFFC_C02` (cause) The hitting hand and wrist are held too tight, actively pressing the racket and killing it
- `N-C_WOVEWH7FFFC_C03` (cause) The player attempts to consciously control and steer the back of the swing rather than tru
- `N-C_WVA4XQPCAQS_C01` (cause) The player's contact point changes depending on their degree of rotation and body orientat
- `N-C_WZLMV4ASG9S_C03` (cause) Squeezing tightly with the bottom of the hand too early before the natural hinge occurs.
- `N-C_WZLMV4ASG9S_C04` (cause) Confusing the rolling follow-through / windshield wiper action with the back-to-front flip
- `N-C__QU1LOWKLAW_C01` (cause) Players overplay and press because they feel they should never lose points to an opponent'
- `N-C__QU1LOWKLAW_C03` (cause) Aiming too close to lines and corners removes the safety margin, turning small mishits int
- `N-C__QU1LOWKLAW_C04` (cause) Under big-moment pressure, players alter their game by overhitting or becoming tentative i

## unobservable_per_model（37）
- `N-C_5_IA7C4XZFU_C01` (cause) At higher swing speeds on a soft string bed, the ball sinks deep into the pocket and is fl
- `N-C_APPDZQXH_2K_C01` (cause) Over-focusing on swing mechanics or silently verbalizing 'watch the ball' detracts from ac
- `N-C_APPDZQXH_2K_C04` (cause) Visual attention becomes overly anchored on the target, leaving the player unable to proce
- `N-C_DDYKUNZTDYU_C01` (cause) Failing to initiate an active two-handed pull driven by squeezing the upper back shoulder 
- `N-C_DDYKUNZTDYU_C02` (cause) Failing to throw and deliver the racket accurately into the pull slot before pulling throu
- `N-C_FBJNFBTZOZE_C02` (cause) A psychological flight response leads to sideways recovery and waiting passively on the ba
- `N-C_FJPR9PS02TM_C02` (cause) The racket moves independently inside the hand rather than moving as a coupled unit, so th
- `N-C_FOMZ8WJV3DQ_C04` (cause) The brain subconsciously forces a narrow stance to protect unprepared soft tissues from hi
- `N-C_FXDMVI3EFNE_C04` (cause) Training devices like the Topspin Pro have physical wings that disincentivize players from
- `N-C_HNVBBPEOB3G_C02` (cause) Disconnecting the arm from the chest creates an extra degree of freedom that introduces an
- `N-C_HNVBBPEOB3G_C04` (cause) The brain subconsciously throttles swing speed when attempting to make the court because i
- `N-C_JZCA_KU7YHK_C02` (cause) The player lacks the adaptable skill set to coordinate their body and produce the intended
- `N-C_JZCA_KU7YHK_C03` (cause) Practice sessions consist of sterile, repetitive drills that lack chaotic unpredictability
- `N-C_LBDQPIHVXYA_C01` (cause) The hitting arm apparatus is not stabilized because the player fails to engage the upper b
- `N-C_LCKDP2GKR_Y_C02` (cause) Small muscles in the wrist and hand are not strong enough to actively drive the heavy rack
- `N-C_LWS7XXQRRDW_C04` (cause) Hitting at improper contact depth jams rotation or exhausts body release before the ball i
- `N-C_MCCB_RFYD0W_C03` (cause) Failing to actively press inward leaves outward momentum unchecked, destroying stroke bala
- `N-C_P41GGC_Y5XK_C01` (cause) Players attempt to generate power and force actively from the hand rather than using it pu
- `N-C_PFDQNFF_VYA_C04` (cause) The player fails to recognize that the overhead/serve uses the exact same outward release 
- `N-C_PUN6QIIYU_4_C03` (cause) Relying on the small hand and wrist muscles to guide the swing rather than driving weight 
- `N-C_PWKGLKRM73O_C01` (cause) Rotational energy is not harnessed because energy cannot pass through the shoulder into an
- `N-C_RVFMDK_IEWO_C01` (cause) Players hit the ball using active slow-twitch muscular contraction in the forearm and wris
- `N-C_TZGAWR3QGV0_C01` (cause) The chest pectoral muscles fail to engage sufficiently to overcome inertia, allowing the h
- `N-C_V_QKILD4V_W_C01` (cause) The returner cannot synchronize the timing of their split step and return rhythm with the 
- `N-C_V_QKILD4V_W_C02` (cause) The returner fails to see the ball early during the initial phase of the serve to accurate
- `N-C_V_QKILD4V_W_C03` (cause) Fixating on the server's racket face or body parts fails because the racket moves too rapi
- `N-C_V_QKILD4V_W_C04` (cause) Human serve motions vary significantly between players and speeds, unlike the ball toss wh
- `N-C_WFIRPMUTZRO_C04` (cause) Lacking a wide base and hip flexion disables glute support and shifts rotational load to l
- `N-C_WVA4XQPCAQS_C02` (cause) The player does not visually register and memorize the exact spatial spot where their body
- `N-C_WVA4XQPCAQS_C03` (cause) Catching the ball in a position where the musculoskeletal alignment cannot generate force 
- `N-C_WZLMV4ASG9S_C02` (cause) Handle is not firmly seated in the top of the hand, losing visceral tactile connection to 
- `N-C_WZLMV4ASG9S_C05` (cause) Failure to stabilize the scapula with back and side muscles (lats, mid/lower traps) during
- `N-C_XLS469ZVMPU_C02` (cause) The player fails to kick the hitting leg backward behind the body to arrest rotational mom
- `N-C_XXLNDJNRA4E_C02` (cause) Players attempt to track the ball immediately off the strings with foveal focus rather tha
- `N-C_YBLIT9PNG2U_C01` (cause) Players fundamentally fail to create proper kick serve racket motion because they do not u
- `N-C_YBLIT9PNG2U_C03` (cause) Players do not utilize the high degree of orthogonality required between the arm's throwin
- `N-C__QU1LOWKLAW_C02` (cause) Players fail to calibrate anxiety properly, either over-hyping and blasting neutral balls 

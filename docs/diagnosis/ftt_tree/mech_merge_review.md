# 机械合并抽检（自动生成，供人工复核）

- 合并前 97 个根因节点 → 合并后 93；最终分组 3 组（只有一次运行同意的候选对 2 个，未合并）。
- 规则：两次独立运行都把它们分在一起才合并；组代表文字 = 组内某成员原文，不改写；同一视频的两条根因不合并。
- 出处列的是该节点证据引用按编号排序的前 3 条（原文逐字，来自卡片），不一定是最贴近该根因的那句；完整证据见 tree_v2.json / mech_merge.json。

## MG01（forehand，代表 `N-C_WZLMV4ASG9S_C07`）
- 模型理由 run1：Continuously rotating the trunk or hips instead of braking rotation to transfer momentum into the arm and racket.
- 模型理由 run2：Spinning the trunk continuously through contact instead of decelerating to transfer momentum to the arm.
- 词面余弦（对照，阈值 0.40 未改）：{'N-C_2NN9TYMECZU_C05|N-C_WZLMV4ASG9S_C07': 0.046, 'N-C_2NN9TYMECZU_C05|N-C_XLS469ZVMPU_C01': 0.051, 'N-C_WZLMV4ASG9S_C07|N-C_XLS469ZVMPU_C01': 0.098}
- **N-C_2NN9TYMECZU_C05**（2Nn9tyMECZU；卡片根因 2Nn9tyMECZU:C05）Accelerating trunk rotation all the way through to the target instead of decelerating short of the target to release the shoulder
  - 出处 2Nn9tyMECZU:Q01 @ 13.52s：“In order to get this elbow up, racket down finish to happen, your shoulder needs to load into external rotation with the scap serratus, the back stabilized here, and then fly through into protraction and internal rotation.”
  - 出处 2Nn9tyMECZU:Q19 @ 691.56s：“your goal is like if visualizing that target, I'm trying to accelerate my trunk short of the target and then let my arm shoulder action finish to the target. So,”
  - 出处 2Nn9tyMECZU:Q24 @ 970.32s：“See the path of the elbow. See how the elbow is is is only going up a little bit, maybe 6 in of net upward elbow movement through the swing, and that's it. But then the racket is rotating like”
- **N-C_WZLMV4ASG9S_C07**（wzLMv4ASg9s；卡片根因 wzLMv4ASg9s:C07）Trunk spins continuously without an anchor to decelerate, preventing momentum transfer to the arm and racket.
  - 出处 wzLMv4ASg9s:Q01 @ 0.4s：“The secret to the racket flip is that you want to take your racket and swing it out first, not straight towards the ball. Now, it doesn't mean that you have”
  - 出处 wzLMv4ASg9s:Q05 @ 78.24s：“the grip. And the key to the grip is to make sure that the handle is seated in the top of your hand. So, if you're an”
  - 出处 wzLMv4ASg9s:Q10 @ 298.56s：“it is very, very important that the bottom of your hand is not tight and trying to pull the racket head around early. So, if any coach, including”
- **N-C_XLS469ZVMPU_C01**（xLs469ZVMPU；卡片根因 xLs469ZVMPU:C01）The player twists continuously through contact instead of allowing hip rotation to be transient and brake to create a whip effect.
  - 出处 xLs469ZVMPU:Q03 @ 57.0s：“to power your forehand swing your hip rotation on the beforehand has to be transient which means it starts and stops you don't just twist all the way through contact”

## MG02（backhand，代表 `N-C_PQ793MBQE50_C03`）
- 模型理由 run1：Setting the pull slot and contact point too close to the body rather than out in front.
- 模型理由 run2：Setting the pull slot or pulling too close to the body rather than well out in front.
- 词面余弦（对照，阈值 0.40 未改）：{'N-C_DDYKUNZTDYU_C04|N-C_PQ793MBQE50_C03': 0.315}
- **N-C_DDYKUNZTDYU_C04**（dDYKuNZtdyU；卡片根因 dDYKuNZtdyU:C04）Pulling too close to the body or relying excessively on rotational body mechanics rather than pulling well out in front.
  - 出处 dDYKuNZtdyU:Q14 @ 273.0s：“out in front of the body feeling where you kind of throw the racket almost to your stomach maybe a little farther back than that let it drift out to here and then pull from here”
  - 出处 dDYKuNZtdyU:Q15 @ 284.0s：“when you pull farther away from your body like Roger feder does and in my opinion working with people the the more rotational swing is more difficult”
  - 出处 dDYKuNZtdyU:Q16 @ 310.0s：“allow either hand to pass well in front of your torso and then pull when it's out here and pull on a line through the ball you're going to get a fall tolerant Hitting Zone”
- **N-C_PQ793MBQE50_C03**（pQ793MBQE50；卡片根因 pQ793MBQE50:C03）Catching the ball too late with the pull slot set too close to the stomach rather than out in front.
  - 出处 pQ793MBQE50:Q03 @ 43.0s：“you have to find the contact point that allows you to pull and if you're in the wrong spot you're not going to be able to pull”
  - 出处 pQ793MBQE50:Q04 @ 63.0s：“from this body position you really can't pull the pull slot on the one hander is out here”
  - 出处 pQ793MBQE50:Q26 @ 964.0s：“you just have to find the right pull slot if you have to pull from here you can't do anything I I would have to pull with these super weak muscles it's impossible”

## MG03（forehand，代表 `N-C_UJPUDUX0JIA_C01`）
- 模型理由 run1：Allowing the hips to extend forward and knees drift over the toes instead of maintaining a hip hinge.
- 模型理由 run2：Allowing hips and knees to drift forward instead of maintaining a hip hinge.
- 词面余弦（对照，阈值 0.40 未改）：{'N-C_APCKJDADZEK_C06|N-C_UJPUDUX0JIA_C01': 0.313}
- **N-C_APCKJDADZEK_C06**（aPcKJDadzek；卡片根因 aPcKJDadzek:C06）The player lets the hips drift forward and slides the knee forward over the toes instead of hinging at the hips and keeping the hips behind.
  - 出处 aPcKJDadzek:Q01 @ 12.44s：“is that whether you are stepping in or not, you are always preparing, turning over your outside foot. So,”
  - 出处 aPcKJDadzek:Q05 @ 88.88s：“If this knee caves in early, you're going to lose your anchor and are going to over-rotate. So, I'm going to feed a”
  - 出处 aPcKJDadzek:Q23 @ 788.12s：“more of this kind of a swing where not only is it less powerful because the kinetic chain doesn't snap, but also, like I can't like change the vector like that. Like I'm trying to go down the line, I”
- **N-C_UJPUDUX0JIA_C01**（uJpuDUx0jIA；卡片根因 uJpuDUx0jIA:C01）Hips extending forward in front of the shoulders with knees forward and chest up (quad dominant posture) rather than maintaining a hip hinge.
  - 出处 uJpuDUx0jIA:Q01 @ 0.48s：“The forehand that we're going to be building today, it's not the fastest version of your forehand. It's not the heaviest version of your forehand, but it is stable, it is controllable, it is”
  - 出处 uJpuDUx0jIA:Q05 @ 102.96000000000001s：“harder, the problem is it's just going to send your torso up. So if coaches are telling you, you know, don't pop up or don't over rotate, driving through that unstable hicker is sending her body up and around instead”
  - 出处 uJpuDUx0jIA:Q09 @ 344.479s：“And so we're kind of hitting almost as if we are doing a Romanian deadlift. And it's going to feel ridiculous. And you're going to notice, huh, even though it feels ridiculous, because it is forcing me to activate my”

## 仅一次运行同意、未合并的候选对
- 仅 run2 同意（backhand）：
  - N-C_DDYKUNZTDYU_C03：Letting go of the racket with the non-dominant hand too early instead of supporting it deep into the forward swing.
  - N-C_PQ793MBQE50_C04：Initiating the stroke solely with one hand instead of executing a two-handed forward movement into the pull slot.
- 仅 run2 同意（forehand）：
  - N-C_UJPUDUX0JIA_C05：Upper trapezius shrugging and lack of isometric engagement in the mid-traps and serratus anterior, detaching the hitting arm from the trunk.
  - N-C_WZLMV4ASG9S_C05：Failure to stabilize the scapula with back and side muscles (lats, mid/lower traps) during forward drive.

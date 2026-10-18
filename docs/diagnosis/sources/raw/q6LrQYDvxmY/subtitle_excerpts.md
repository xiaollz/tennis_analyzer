# q6LrQYDvxmY 字幕摘取片段（仅存核对用到的短片段）

- **来源不是 YouTube 官方字幕**：yt-dlp 取字幕在 box 与用户 Mac 上均 HTTP 429（见 calls.md）。改用 faster-whisper 转写本地已有的 720p 文件（时长 1035.6 s，与 yt-dlp 的 1036 s 一致）。
- 转写模型：`small.en`，每 60 s 分块、VAD 关闭（`whisper_tr.py` 同目录脚本为 medium.en+VAD 版，其结果有 5 处空洞，故改用分块版做主参照）。
- **转写可能有听错**（例如 C03 里 small.en 听成 "left arm"，medium.en 听成 "non-hitting arm"，Gemini 也写 "non-hitting arm"）。核对结论以「一致/部分一致」表述，不等于逐字相同。
- 只摘各主张对应的几行；完整转写不入库（版权）。时间为视频播放时间 mm:ss.s。

### C01 默认前提（00:13.0–00:28.0）
- [00:16.2] you're getting your chest down, your hips stabilized, and your shoulder attached and
- [00:21.9] swinging through this gentle semicircle, you need to fall away from you farther out to
- [00:26.7] your right.
- [00:28.0] hitting lane here is going to feel comically far away to your right compared to this hitting lane

### C02 回吸旧被挤点（00:50.0–01:00.0）
- [00:52.0] everything right, you're going to intuitively go back to your old jammed contact point,
- [00:56.2] And that's going to make things break down, even though you're a lot closer than you think.
- [00:59.2] And so this video is all about how can we learn.
- [01:00.0] is all about how can we learn to push the ball away from us

### C03 非持拍手推出（01:06.0–01:16.0）
- [01:06.2] First trick for finding the spacing is use your left arm
- [01:12.4] and push it across your body in your preparation.
- [01:15.4] So if you watch, you can see her left hand, her left elbow.

### C04 平衡/间距提示（01:23.0–01:32.0）
- [01:23.6] And this this is like it is a balanced cue
- [01:26.9] and a spacing queue to keep that ball off to your right
- [01:30.6] where that better hitting slot is.

### C05a 故意收手（01:34.0–01:46.0）
- [01:34.2] Okay, so now what we're gonna do is
- [01:35.8] she's gonna keep it in on purpose.
- [01:37.5] So this is not what we want to do.
- [01:38.9] We want you to keep that out to find the spacing,.
- [01:41.0] And it's much more difficult to find the spacing.
- [01:43.0] Now you can still find it, right?
- [01:44.0] Like that one she spaced pretty well.
- [01:45.5] And so this is not physically necessary

### C05b 速度/协调（01:54.0–02:12.0）
- [01:54.5] You'll notice her velocity is a little lower here.
- [01:56.8] The other thing that this left arm does is it helps you coordinate that trunk activation in
- [02:00.0] that trunk activation into the ball.
- [02:02.0] So now she's gonna try to hit hard,
- [02:03.6] but with that left arm kind of dangling in here
- [02:05.2] instead of out and stabilizing her,
- [02:06.9] and like she's obviously gonna make a few,
- [02:08.9] but you're gonna notice that the whole thing
- [02:11.0] is not gonna look, nearly as coordinated there.

### C06 握锥桶（02:41.0–02:55.0）
- [02:41.7] something will help you do it. Because the act of holding it
- [02:45.8] the act of kind of seeing it in your periphery will all of a
- [02:48.0] sudden make your brain kind of pay attention to that left hand
- [02:50.1] pay attention to that lever that you have there. And so if you
- [02:52.5] hold just like a cone in your hand, that can help you cue

### C07 2.5磅（03:07.0–03:20.0）
- [03:07.7] you can upgrade it to a weight.
- [03:08.9] This is two and a half pounds.
- [03:10.1] This is more of like an adult male
- [03:12.3] already kind of reasonably strong with this.
- [03:14.4] And this is the same kind of cue,
- [03:16.9] except the difference here is now it's loaded.
- [03:19.6] So there's like a lot of demand on me

### C08 远得夸张（04:09.0–04:22.0）
- [04:10.0] The correct hitting slot is going to feel comically far off to your right.
- [04:13.9] What I want you to notice here, this is too far from Alexa, like I am making her stretch
- [04:18.1] like crazy, and yet, actually, it is still pretty intuitive to create this topspin contact

### C09 弓步/远探（04:48.0–05:02.0）
- [04:51.9] don't let yourself move and just lunge or like actually
- [04:55.9] probe the ball comically far away and notice that even if you have to lunge at the end of the drill
- [05:00.0] the lunge at the end of the drill,
- [05:01.9] you can actually create your top spin contact feeling.

### C10 重心进整只脚（05:42.0–06:06.0）
- [05:42.1] is if you watch the balance here,
- [05:44.1] see how Alexa's balance is kinda centered,
- [05:45.7] it's more over the right heel.
- [05:47.3] When you probe the ball comically.far to the right,
- [05:49.0] that is gonna force you to shift your balance
- [05:51.7] onto your whole foot or the ball of your foot
- [05:54.1] when you load.
- [05:54.9] And a lot of people just kinda stay here
- [05:56.1] between their heels and then the swing falls off to the left,
- [05:58.8] because instead of the, you know,
- [06:00.0] you know, entire right leg driving you through the court,
- [06:02.6] you drive through the heel of the leg and you drive back.

### C11 pressing vector（06:12.0–06:30.0）
- [06:17.3] see how the elbow is being driven
- [06:19.0] in the same direction she's facing.
- [06:20.9] So, if the ball is probed here,
- [06:22.8] you have to kind of under-hand yank the elbow
- [06:25.4] across your body.
- [06:26.3] But, because the ball is probed here,
- [06:27.9] she's facing this direction,
- [06:29.1] the elbow is driving that direction.

### C12 弓步检验（07:14.0–07:22.0）
- [07:14.7] And so this is an amazing drill if you're like,
- [07:16.6] hmm, where does my shoulder work the best?
- [07:18.7] Well, don't let yourself use anything else
- [07:20.6] except that kind of quick upper body rotation

### C13 纵向斜线（08:25.0–08:36.0）
- [08:25.5] And the key here is you want a sense,
- [08:29.3] especially on these waist or even knee height balls,
- [08:32.7] that your swing is going to go
- [08:35.0] on a gentle upward diagonal through the ball.

### C14 高肘≠俯冲（09:09.0–09:14.0）
- [09:09.4] So even if you're using a high elbow, it doesn't mean you're dive bombing the contact zone
- [09:13.2] from there.

### C15 低球/转动平面（09:49.0–10:20.0）
- [09:49.1] Let's do a couple of low ones here.
- [09:51.0] So here, I want to get into that almost kind of
- [09:54.4] Romanian deadlift, imagining that hand
- [09:58.7] like so far, power so far below.
- [10:00.0] power so far below the ball if I want free margin, you have to hit it every time.
- [10:06.0] And so what that's doing is, you know, even though my relative arm position isn't changing too,
- [10:10.0] I can lower the racket with this lever and then rotate in a more vertical plane of rotation,
- [10:18.0] but that doesn't mean I have to independently lift with the arm here.

### C16 低球平面（10:30.0–10:56.0）
- [10:30.7] Whoops, well, you can see why I'm getting nervous
- [10:32.5] because my plane of rotation here
- [10:35.0] is just not, is giving me very little margin.
- [10:39.5] Very little margin.
- [10:40.4] Like I have to kind of catch it perfectly with my hand
- [10:42.4] versus if because that ball is so low
- [10:44.6] if I just use a slightly more slanted plane of rotation.
- [10:47.7] See I didn't catch that perfectly with my hand
- [10:49.2] but it still cleared the net.
- [10:50.8] And so now with this slightly slanted
- [10:52.4] more slanted plane of rotation
- [10:54.4] I have access to all of the height

### C17a 胸高球（11:50.0–12:18.0）
- [11:52.2] Now these, I probably want a basically,
- [11:56.1] Basically back to front.
- [12:00.0] Rack is starting, like, in my mind, at least,
- [12:03.3] I guess I don't know where it physically is,
- [12:04.8] but in my mind, a little bit higher, do me a favor.
- [12:08.7] In my mind, like, basically, like I'm imagining,
- [12:12.2] ball here, hand here, right?
- [12:13.7] And so that is basically, maybe ever so slightly
- [12:16.9] low to high slanted, but basically a horizontal

### C17b 吊高球（12:47.0–12:56.0）
- [12:50.8] there, then I'm hitting a lob. So if I accelerate here, I'm going to miss, if I start below the

### C18 前后间距（13:28.0–13:34.0）
- [13:28.8] if you get the back front spacing wrong,
- [13:30.9] that can also kind of paralyze your shoulder,
- [13:33.0] paralyze your natural hitting action.

### C19 鸡翅/卡在躯干后（14:05.0–14:15.0）
- [14:05.5] So see how now, you know,
- [14:06.6] she's got to either get the hitting arm way in
- [14:08.1] like a chicken wing or it gets stuck behind the trunk
- [14:09.8] or whatever, her prep was great.
- [14:12.7] It's just, just that it's all that's here, here.
- [14:15.0] That's where the arm can connect.

### C20 前倾+6英寸（14:36.0–14:42.0）
- [14:39.2] that is another, you know,
- [14:40.6] six inches of depth that you have to set up for.

### C21 跑向球把自己挤住（16:02.0–16:10.0）
- [16:02.1] Now, you know, sometimes we see this,
- [16:04.3] where the desire to be aggressive,
- [16:07.8] you run into the ball and jam yourself like this.

### C22 butt back（16:40.0–16:46.0）
- [16:40.6] Now, the other failure mode is, okay, oh, I got to keep it in front. I got to leave enough space.
- [16:44.2] Sometimes you get this butt-back racket forward position if you reach too much.

### C23 耐心（17:03.0–17:16.0）
- [17:05.4] absolutely in front of you and you need that space for the racket to fly through but that doesn't
- [17:10.6] mean you have to exaggerate it if you find yourself in the butt back racket forward be a little more
- [17:14.6] patient.

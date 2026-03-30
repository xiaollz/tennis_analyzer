"""Tennis terminology glossary for bilingual translation."""

GLOSSARY = {
    # Core FTT concepts
    "fault tolerant": "容错",
    "fault tolerance": "容错性",
    "margin for error": "容错空间",
    "fundamental theorem": "基本定理",
    "fundamental theorem of tennis": "网球基本定理",
    # Contact & racket
    "string angle": "拍面角度",
    "racket-forearm angle": "球拍-前臂夹角",
    "contact point": "击球点",
    "contact": "击球接触",
    "racket head": "拍头",
    "butt cap": "拍柄底盖",
    "swing path": "挥拍路径",
    "racket face": "拍面",
    "rolling the racket": "翻转球拍",
    "roll over it": "翻转压拍",
    # Grips
    "semi-western grip": "半西方式握拍",
    "eastern grip": "东方式握拍",
    "western grip": "西方式握拍",
    "continental grip": "大陆式握拍",
    "grip": "握拍",
    # Spin
    "topspin": "上旋",
    "backspin": "下旋",
    "slice": "切削",
    "windshield wiper": "雨刷式",
    # Preparation
    "unit turn": "整体转体",
    "backswing": "引拍",
    "hand preparation": "手部准备",
    "preparing the hand": "手部准备",
    "wrist lag": "手腕滞后",
    "elbow extension": "肘部伸展",
    "visual tracking": "视觉追踪",
    "inhale and wait": "吸气等待",
    # Stances
    "closed stance": "关闭式站位",
    "open stance": "开放式站位",
    "neutral stance": "中性站位",
    # Forward swing
    "forward swing": "前挥",
    "follow-through": "随挥",
    "rotational kinetic chain": "旋转动力链",
    "kinetic chain": "动力链",
    "abdominal coil": "腹部蓄力",
    "out vector": "Out向量",
    "up vector": "Up向量",
    "through vector": "Through向量",
    "out, up, and through": "Out、Up、Through三向量",
    "hip rotation": "髋旋转",
    "shoulder rotation": "肩旋转",
    "hip-shoulder separation": "肩髋分离角",
    "lead with the hips": "以髋部引导",
    # Anatomy
    "pronation": "旋前",
    "supination": "旋外",
    "non-hitting arm": "非持拍臂",
    "non-hitting hand": "非持拍手",
    "hitting arm": "持拍臂",
    "wrist tension": "手腕张力",
    # Training
    "drop feed": "抛球喂球",
    "live-ball": "活球对打",
    "mini-tennis": "迷你网球（短距离对打）",
    "shadow swing": "空挥（影子挥拍）",
    "offense-defense": "攻防练习",
    "X-drill": "X训练法",
    # General tennis
    "stroke production": "击球动作",
    "movement pattern": "运动模式",
    "coaching cue": "教学提示",
    "power limit": "力量上限",
    "rally": "回合",
    "baseline": "底线",
    "net": "球网",
    "cross-court": "斜线球",
    "down-the-line": "直线球",
    "passing shot": "穿越球",
    "pusher": "磨控型选手",
}

def build_glossary_prompt() -> str:
    """Build glossary section for translation prompt."""
    lines = ["TENNIS TERMINOLOGY GLOSSARY (use these translations exactly):"]
    for en, zh in GLOSSARY.items():
        lines.append(f"  {en} = {zh}")
    return "\n".join(lines)

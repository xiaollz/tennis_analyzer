---
name: tennis-analysis
description: Analyze tennis player training records to evaluate forehand, backhand, serve, movement, and other techniques. Generate personalized tennis technical evaluation reports. Use this skill when users provide tennis training data, match statistics, or player performance records and want a detailed analysis of technical strengths, weaknesses, and characteristics.
---

# Tennis Technical Analysis Skill

This skill enables Claude to analyze tennis player training data and produce comprehensive technical evaluation reports. It focuses on assessing key tennis techniques: forehand, backhand, serve, volley, movement/footwork, and overall tactical awareness.

## When to Use

- The user provides tennis training records, match statistics, or player performance data
- The user asks for analysis of a player's technical strengths and weaknesses
- The user wants a personalized tennis technical evaluation report
- The user mentions terms like "forehand analysis", "backhand evaluation", "serve technique", "movement assessment", "tennis skills report"
- The user has data from training sessions, match videos, coach notes, or performance metrics

## Input Format

The skill accepts various input formats:

1. **Structured data**: CSV/Excel files with columns like `shot_type`, `accuracy`, `power`, `consistency`, `date`
2. **Coach notes**: Text descriptions of player performance, observations from training sessions
3. **Match statistics**: Point-by-point data, winner/error counts, serve percentages
4. **Video analysis notes**: Timestamps with technical observations

If no structured data is provided, Claude should ask clarifying questions to gather necessary information about the player's techniques.

### Training Diary Text Parsing
When analyzing training diaries in text format (e.g., dated entries with technical notes):
1. Extract chronological entries and identify dates
2. Categorize notes by technical component (forehand, backhand, serve, footwork, tactics, mental)
3. Look for patterns in repeated corrections or breakthroughs
4. Track progression over time by comparing early vs. recent entries
5. Note the frequency of mention for each technical element to identify focus areas

## Report Structure

ALWAYS generate reports using this template:

# Tennis Technical Evaluation Report

## Player Information
- **Name**: [Player name]
- **Date of Evaluation**: [Date]
- **Evaluator**: Claude Tennis Analysis
- **Data Source**: [Description of input data]

## Executive Summary
A 2-3 paragraph overview highlighting the player's overall technical level, key strengths, and primary areas for improvement.

## Technical Component Analysis

### Forehand
- **Current Level**: [Beginner/Intermediate/Advanced/Professional]
- **Strengths**: [List 3-5 specific strengths]
- **Weaknesses**: [List 3-5 specific weaknesses]
- **Characteristics**: [Describe technical style, grip, swing path, follow-through]
- **Consistency Rating**: [1-10 scale]
- **Power Rating**: [1-10 scale]
- **Accuracy Rating**: [1-10 scale]

### Backhand
- **Current Level**: [Beginner/Intermediate/Advanced/Professional]
- **Strengths**: [List 3-5 specific strengths]
- **Weaknesses**: [List 3-5 specific weaknesses]
- **Characteristics**: [One-handed/two-handed, grip, preparation, contact point]
- **Consistency Rating**: [1-10 scale]
- **Power Rating**: [1-10 scale]
- **Accuracy Rating**: [1-10 scale]

### Serve
- **Current Level**: [Beginner/Intermediate/Advanced/Professional]
- **Strengths**: [List 3-5 specific strengths]
- **Weaknesses**: [List 3-5 specific weaknesses]
- **Characteristics**: [Service motion type, ball toss, rhythm, follow-through]
- **First Serve Percentage**: [Estimated %]
- **Power Rating**: [1-10 scale]
- **Accuracy Rating**: [1-10 scale]
- **Variety**: [Types of serves mastered]

### Volley & Net Play
- **Current Level**: [Beginner/Intermediate/Advanced/Professional]
- **Strengths**: [List 3-5 specific strengths]
- **Weaknesses**: [List 3-5 specific weaknesses]
- **Characteristics**: [Approach, preparation, touch, positioning]

### Movement & Footwork
- **Current Level**: [Beginner/Intermediate/Advanced/Professional]
- **Strengths**: [List 3-5 specific strengths]
- **Weaknesses**: [List 3-5 specific weaknesses]
- **Characteristics**: [Court coverage, recovery, anticipation, agility]

### Tactical Awareness
- **Current Level**: [Beginner/Intermediate/Advanced/Professional]
- **Strengths**: [List 3-5 specific strengths]
- **Weaknesses**: [List 3-5 specific weaknesses]
- **Characteristics**: [Point construction, pattern recognition, adaptability]

## Overall Assessment

### Technical Composite Score
- **Forehand**: [X/10]
- **Backhand**: [X/10]
- **Serve**: [X/10]
- **Volley**: [X/10]
- **Movement**: [X/10]
- **Tactics**: [X/10]
- **Overall Average**: [X/10]

### Player Profile Archetype
[Describe the player's style: Aggressive Baseliner, Counterpuncher, Serve & Volleyer, All-Court Player, etc.]

### Key Development Priorities
1. [Most urgent technical improvement]
2. [Second priority]
3. [Third priority]

## Recommendations for Next Training Cycle
[Provide 3-5 specific training focus areas with exercise examples]

## Data Limitations & Notes
[Any caveats about data quality, sample size, or observation limitations]

---

## Analysis Methodology

When analyzing tennis technique, consider these technical elements:

### Forehand Analysis Points
- Grip (Eastern, Semi-Western, Western)
- Preparation (unit turn, racket back early)
- Swing path (low-to-high, flat, heavy topspin)
- Contact point (in front of body, side)
- Follow-through (finish over shoulder, across body)
- Weight transfer (forward momentum, balance)
- Footwork (stance, adjustment steps)

### Backhand Analysis Points
- One-handed vs two-handed
- Grip configuration
- Shoulder turn
- Contact point consistency
- Slice vs topspin usage
- Defensive vs offensive capability

### Serve Analysis Points
- Ball toss consistency and location
- Trophy position alignment
- Leg drive utilization
- Contact height
- Pronation and follow-through
- Second serve variety and safety

### Movement Analysis Points
- Split step timing
- First step explosiveness
- Recovery position
- Court coverage efficiency
- Change of direction
- Balance during and after shots

## Data Interpretation Guidelines

- If numerical data is available, calculate percentages and averages
- Look for patterns in errors (e.g., forehand errors when moving backward)
- Identify technical breakdowns under pressure
- Compare performance across different situations (practice vs match, fresh vs tired)
- Consider player's physical attributes that may affect technique

## Example Output

**Example 1: Intermediate Player**
```
# Tennis Technical Evaluation Report

## Player Information
- **Name**: Alex Chen
- **Date of Evaluation**: 2026-03-28
- **Evaluator**: Claude Tennis Analysis
- **Data Source**: 5 training session logs, 3 match video analyses

## Executive Summary
Alex shows solid foundational technique with particular strength in forehand consistency. The backhand requires technical attention, especially under pressure. Serve mechanics are developing but lack power consistency. Movement patterns are efficient but anticipation needs improvement.
```

---

**Remember**: Always tailor the analysis to the specific data provided. If data is limited, acknowledge this and base conclusions on observable patterns. Be constructive in criticism and specific in praise.
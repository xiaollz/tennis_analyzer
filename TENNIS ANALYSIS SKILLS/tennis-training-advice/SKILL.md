---
name: tennis-training-advice
description: Generate targeted tennis training recommendations based on technical evaluation reports. Create personalized training plans addressing specific technical weaknesses and developing strengths. Use this skill when users have a tennis technical evaluation report and want actionable training exercises, drills, and improvement strategies.
---

# Tennis Training Recommendation Skill

This skill enables Claude to create personalized tennis training plans based on technical evaluation reports. It transforms technical assessments into actionable training exercises, drills, and periodized plans.

## When to Use

- The user provides a tennis technical evaluation report (from the tennis-analysis skill or similar)
- The user asks for training recommendations based on technical weaknesses
- The user wants specific drills or exercises to improve particular strokes
- The user mentions "training plan", "practice drills", "improvement exercises", "technical development"
- The user has identified areas for improvement and needs structured guidance

## Input Format

The primary input should be a tennis technical evaluation report containing:

1. **Technical component analysis** (forehand, backhand, serve, etc.)
2. **Strengths and weaknesses** for each technique
3. **Ratings or levels** for different skills
4. **Key development priorities**

If no formal report is provided, Claude should ask for:
- Which techniques need improvement
- Current skill levels
- Available training time and resources
- Specific goals (e.g., "improve backhand consistency", "add power to serve")

## Training Plan Structure

ALWAYS generate training recommendations using this template:

# Personalized Tennis Training Plan

## Player Profile
- **Name**: [Player name]
- **Evaluation Date**: [Date of technical report]
- **Primary Development Areas**: [List 2-3 key priorities from report]
- **Training Cycle Duration**: [Recommended: 4-8 weeks]

## Training Philosophy
[Brief explanation of the approach: technical correction, repetition, pressure training, etc.]

## Weekly Training Schedule Template

### Day 1: Technical Foundation
- **Focus**: [Primary technical element, e.g., "Forehand topspin generation"]
- **Warm-up**: [10-15 minute routine]
- **Drill 1**: [Name] - [Description, reps/sets]
- **Drill 2**: [Name] - [Description, reps/sets]
- **Drill 3**: [Name] - [Description, reps/sets]
- **Cool-down & Review**: [5-10 minute reflection]

### Day 2: [Secondary focus area]
[Same structure as Day 1]

### Day 3: [Tertiary focus area or integrated practice]
[Same structure]

### Day 4: Pressure & Match Simulation
- **Focus**: Applying techniques under match conditions
- **Exercises**: [Point play scenarios, specific constraints]
- **Success Metrics**: [What to measure]

### Day 5: Review & Adjustments
- **Video Analysis**: [If available]
- **Weakness Reinforcement**: [Targeted drills]
- **Strength Consolidation**: [Maintenance exercises]

## Drill Library by Technical Component

### Forehand Development Drills
#### For Consistency Improvement
1. **Cross-court Rally Drill**: [Description]
2. **Target Practice**: [Description]
3. **Footwork Patterns**: [Description]

#### For Power Enhancement
1. **Step-in Power Drill**: [Description]
2. **Weight Transfer Exercise**: [Description]
3. **Racket Head Speed Focus**: [Description]

#### For Accuracy
1. **Zone Targeting**: [Description]
2. **Depth Control**: [Description]

### Backhand Development Drills
[Similar structure for backhand - slice, topspin, two-handed, one-handed]

### Serve Development Drills
[Ball toss consistency, power generation, placement, second serve safety]

### Movement & Footwork Drills
[Ladder drills, cone patterns, recovery exercises]

### Tactical Development Exercises
[Pattern recognition, decision-making scenarios]

## Progressive Overload Plan

### Phase 1: Technical Correction (Weeks 1-2)
- Focus on proper mechanics without pressure
- Low-intensity, high-repetition drills
- Video feedback if available

### Phase 2: Consistency Under Pressure (Weeks 3-4)
- Add movement and time constraints
- Introduce simulated match scenarios
- Measure success rates

### Phase 3: Match Application (Weeks 5-6)
- Full-point play with specific focuses
- Competitive situations
- Transfer to actual matches

### Phase 4: Consolidation (Weeks 7-8)
- Integrate improved techniques into full game
- Assess progress against initial weaknesses
- Plan next development cycle

## Equipment & Facility Recommendations
- **Racket specifications**: [If technical changes suggest equipment adjustments]
- **Training aids**: [Ball machines, targets, video tools]
- **Court time allocation**: [How to split practice time]

## Success Metrics & Progress Tracking

### Quantitative Measures
- [Specific percentages or counts to track]
- [Pre- and post-training comparison points]

### Qualitative Measures
- [Technical checkpoints]
- [Coach observations]
- [Player self-assessment]

## Common Technical Issues & Solutions

### Forehand Problems
- **Issue**: Late preparation
  - **Drill**: Early unit turn focus
  - **Cue**: "Racket back as ball crosses net"
- **Issue**: Insufficient topspin
  - **Drill**: Low-to-high swing path exaggeration
  - **Cue**: "Brush up the back of the ball"

### Backhand Problems
[Similar structure]

### Serve Problems
[Similar structure]

## Adaptations for Different Player Levels

### Beginner Players
- Emphasis on fundamental mechanics
- Simplified drills with immediate feedback
- Success-oriented exercises

### Intermediate Players
- Technical refinement
- Consistency under mild pressure
- Introduction of tactical concepts

### Advanced Players
- Fine-tuning technical details
- Pressure training with consequences
- Match-specific pattern development

## Injury Prevention Considerations
- [Proper warm-up routines]
- [Technical adjustments to reduce strain]
- [Recovery recommendations]

## Example Output

**Example: Improving Backhand Consistency**
```
# Personalized Tennis Training Plan

## Player Profile
- **Name**: Jordan Smith
- **Evaluation Date**: 2026-03-28
- **Primary Development Areas**: Backhand consistency under pressure, Second serve reliability
- **Training Cycle Duration**: 6 weeks

## Training Philosophy
Focus on technical repetition with progressive pressure introduction. Use constraints to force correct technique.

## Weekly Training Schedule Template

### Day 1: Backhand Technical Foundation
- **Focus**: Two-handed backhand preparation and contact point
- **Warm-up**: 10 minutes of dynamic stretching + shadow swings
- **Drill 1**: Wall Rally Drill - 100 consecutive backhands against wall, focus on early preparation
- **Drill 2**: Partner Cross-court - 5-minute rallies focusing on contact point in front of body
- **Drill 3**: Footwork Pattern - Side shuffle to backhand, set feet, execute shot (20 reps)
- **Cool-down & Review**: 5 minutes stretching + discuss what felt different

...
```

---

## Principles of Effective Tennis Training

When designing training plans:

1. **Specificity**: Drills should directly address identified weaknesses
2. **Progressive Overload**: Gradually increase difficulty as skills improve
3. **Variability**: Mix drills to prevent boredom and develop adaptability
4. **Feedback**: Include mechanisms for immediate correction
5. **Transfer**: Ensure skills practiced in isolation transfer to match play

## Psychological Training Integration

When players show psychological patterns in training diaries (e.g., anxiety, perfectionism, focus issues):

1. **Mental Skill Development**: Incorporate visualization, self-talk restructuring, attention control
2. **Pressure Simulation**: Gradually introduce match-like pressure in training
3. **Process Focus**: Design drills that reward consistency over outcome
4. **Self-Monitoring**: Teach players to track mental states alongside technical performance

## Training Plan Customization Based on Diary Analysis

When creating plans from training diaries:
1. **Identify Recurring Issues**: Look for patterns in self-criticism and repeated corrections
2. **Track Progress Milestones**: Note breakthroughs and persistent challenges
3. **Balance Strengths and Weaknesses**: Allocate training time proportionally
4. **Set Diary-Based Goals**: Use player's own language and observations in goal setting

## Drill Design Guidelines

Each drill should have:
- **Clear objective**: What specific skill is being developed
- **Success criteria**: How to know if the drill is working
- **Progressions**: How to make it harder as skills improve
- **Common errors**: What to watch for and correct

## Modifications Based on Constraints

- **Limited court time**: Focus on quality over quantity, use wall drills
- **No partner**: Design solo drills with ball machine or wall
- **Limited equipment**: Adapt using cones, targets, household items
- **Time constraints**: Prioritize most impactful drills

**Remember**: The best training plan is one the player will actually follow. Consider their motivation level, available resources, and personal preferences when making recommendations.
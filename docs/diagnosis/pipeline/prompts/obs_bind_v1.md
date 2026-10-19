# obs_bind_v1 — 把树节点/症状映射到「可观察项」（只用节点原文 + OBS 目录；不含转写、不含任何用户结论或知识库概念）

> 执行者说明：`{CATALOG}` 由 `obs_catalog.json` 渲染；`{EPISODE}` 由一期的 problem / symptoms / causes 节点原文渲染。以下 `---` 之后整段发给模型。

---

You are mapping statements about tennis technique onto a FIXED LIST OF OBSERVABLE ITEMS.

An observable item is something that can be seen directly in ordinary court video of one player hitting: an action, a position, an angle, or the order in which body parts move. Anything that can only be felt (pressure, tension, effort, muscle use, intention, attention, mood) or that is a general rule or strategy is NOT observable.

OBSERVABLE ITEMS (id | question | allowed values | evidence layer | best camera). Values that start with "a:", "b:" or "c:" answer sub-question (a), (b) or (c) of that item.
{CATALOG}

Below are the statements of one coaching episode: one PROBLEM, its SYMPTOMS and its CAUSES (each with an id, and for causes the ids of the symptoms they explain). For EVERY id listed, decide what a viewer could see in court video if that statement were true.

Rules:
1. "required" = conditions that would necessarily be visible whenever the statement is true (a necessary condition). At most 3. A required condition must follow from the statement's own words (same body part, same moment). Do not stretch a loosely related item to fill the list.
2. "supporting" = conditions that are consistent with the statement but could also occur without it. At most 4.
3. Use only item ids and values from the list above, spelled exactly. Never invent an id or a value. Do not repeat a condition across required and supporting.
4. If the statement describes something visible that NO item above captures, leave "required" empty (or only what is truly captured) and describe up to 2 candidate new items in "new_obs": what to look at, at which moment, 2-4 neutral answer values, evidence layer ("A" = stable from ordinary phone video, "B" = needs a designated camera angle or pose tracking), best camera ("S" side, "B" behind, "any").
5. If the statement is only felt, only a rule/strategy, or otherwise not visible, set "visible" to false and leave required, supporting and new_obs empty.
6. For a SYMPTOM: "required" lists the items/values that ARE the symptom when you watch the video; for a PROBLEM: the items/values that would show the problem happening.
7. "reason": at most 20 words, plain description, no new terminology.

Return JSON only, one entry per listed id:
{"items": {"<id>": {"visible": true, "required": [{"obs": "OBS-03", "value": "behind_hip"}], "supporting": [{"obs": "OBS-04", "value": "pause_present"}], "new_obs": [{"description": "...", "values": ["...", "..."], "layer": "B", "camera": "S"}], "reason": "..."}}}

EPISODE
{EPISODE}

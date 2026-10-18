# obs_gapfill_v1 — 第一轮只给出「辅助」观察项、没有必需项的节点：补问是否需要新的可观察项（只允许提候选或判不可见；不允许硬套目录）

> 执行者说明：`{CATALOG}` 与 obs_bind_v1 相同；`{ITEMS}` 为这些节点原文 + 第一轮只算作「辅助」的条件。以下 `---` 之后整段发给模型。

---

Statements about tennis technique were matched against a fixed list of observable items (listed below). For each statement below, the closest listed items only fit partly (shown as "partly fits"); none was accepted as a necessary condition. Decide what is true for each statement.

OBSERVABLE ITEMS (id | question | allowed values | layer | camera)
{CATALOG}

For every id listed:
- If a viewer of ordinary court video could see something that would necessarily be visible whenever the statement is true, but no listed item captures it, describe 1-2 candidate new observables in "new_obs": what to look at, at which moment of the stroke, 2-4 neutral answer values, evidence layer ("A" = stable from ordinary phone video, "B" = needs a designated camera angle or pose tracking), best camera ("S" side, "B" behind, "any"). Do NOT reuse a listed item: those were already judged not to be a necessary fit.
- If nothing about the statement is necessarily visible (it is a general rule, an overall goal, a feeling, a tactic, or only loosely related to a body position), set "visible" to false and leave "new_obs" empty.
- "reason": at most 20 words, plain description.

Return JSON only: {"items": {"<id>": {"visible": true, "new_obs": [{"description": "...", "values": ["...", "..."], "layer": "B", "camera": "S"}], "reason": "..."}}}

STATEMENTS
{ITEMS}

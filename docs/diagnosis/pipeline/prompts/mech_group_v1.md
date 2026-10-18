# mech_group_v1 — 同击球类型内按机制给根因分组（模型只分组；不改写文字；不含任何用户结论或知识库概念）

> 执行者说明：`{ITEMS}` 由同一击球类型的根因节点原文渲染。以下 `---` 之后整段发给模型。

---

Below are root-cause statements extracted from different tennis coaching videos about the SAME stroke type. Each line is: id | video | layer | statement. A line may be followed by "also worded:" lines, which are other statements already known to be the same cause.

Task: put together the statements that describe the SAME mechanism: the same body action, position or timing, with the same reason it hurts the stroke, so that a coach would say "these two say the same thing". Only group; never rewrite anything.

Rules:
- A group needs at least 2 ids; an id may be in at most one group.
- Two ids from the SAME video must never be in one group.
- The same body part or the same general topic is NOT enough. Different phases of the stroke, opposite directions, or a cause versus its consequence are different mechanisms: keep them apart.
- When unsure, leave the statements ungrouped. Ungrouped ids are simply left out of the output.
- "rep" must be one of the group's own ids: the member whose own wording states the mechanism most clearly and generally.
- "why": at most 25 words, say what the shared mechanism is.

Return JSON only: {"groups": [{"members": ["id1", "id2"], "rep": "id1", "why": "..."}]}

STATEMENTS
{ITEMS}

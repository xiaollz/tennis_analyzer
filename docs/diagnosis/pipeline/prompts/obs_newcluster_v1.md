# obs_newcluster_v1 — 把「新观察项候选」按含义分组（只分组；组代表取组内原文之一；不含用户结论或知识库概念）

> 执行者说明：`{ITEMS}` 由 obs_bind 产出的 new_obs 候选渲染。以下 `---` 之后整段发给模型。

---

Below are candidate new observables for tennis court video (things a viewer could see that a fixed list of observables does not cover). Each line is: cid | description | answer values | layer | camera.

Task: group the candidates that ask about the SAME visible thing at the same moment of the stroke. Only group; never rewrite.

Rules:
- A group needs at least 2 cids; a cid may be in at most one group; ungrouped cids are left out.
- Different body parts, different moments of the stroke, or different directions are different observables.
- "rep" must be one of the group's own cids: the clearest, most general description.

Return JSON only: {"groups": [{"members": ["c1", "c2"], "rep": "c1", "why": "..."}]}

CANDIDATES
{ITEMS}

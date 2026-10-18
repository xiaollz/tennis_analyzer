# diagnosis prompts

版本规则：

- 文件名带版本号：`fact_extract_v1.md`、`fact_extract_v2.md`…
- **改提示词正文 = 升版本**（新建 `vN+1`，旧版保留只读，便于对账复现）。
- 只改本 README 或对账规则说明、不改发给模型的正文 → 可不升版本，但须在对应文件顶栏记一行变更说明。
- 跑数时在输出文件名或元数据里写明所用版本（例如 `fact_extract_v1`），与 `observables/00_runs_*.md` 对齐。

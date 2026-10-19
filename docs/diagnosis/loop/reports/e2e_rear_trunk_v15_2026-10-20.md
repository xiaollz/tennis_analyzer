# E2E 验收：rear 2026-10-19 × tree_v15 × kinetic_chain_trunk（Asia/Shanghai 2026-10-20）

## VERIFY 树
- `verify_2026-10-20_tree_v15.json` → **pass=true**（未挂 V5 gate；与 v14 同策略）
- tree：`ftt_tree/tree_v15.json`（56 期问题节点；OBS 72；T05/T08 check_tree 0 FAIL）
- 新挂树非 holdout：`6ak4ZScH_gc`、`KU7FHy1qQOI`
- holdout 已 Stage1 但不进树：`ui-RFRdQKRk`、`5jHCDc44SQM`（知识进 trunk 文档）

## 主干一句 → 断点 → 触发细节（对照已有 rear 事实，不编造）

| 层 | 内容 | 证据 |
|---|---|---|
| **主干一句** | 髋扭转不足 → 链断 → 手臂独立发力（FLT-02） | `kinetic_chain_trunk.md` §0；`rear_20261019_trunk.md` §1 |
| **断点步骤** | A 脚钉死/腿未加载（上游）→ B 髋扭转/胸未打开（主断点，待验证）→ C 臂独立（表现）→ 随挥缠（下游） | trunk §2；rear §2–3 |
| **触发细节（已有事实）** | 宽开放钉原地；膝无明显深弯；触球肩线半开/背朝镜头；准备 unit turn OK；随挥缠左肩；横向间距 OK | `/workspace/today_swing/rear_20261019_facts.md` 等；rear §3 表 |
| **能证明的能力** | 叙事可从主干一句拆到 A/B/C 与每条事实的挂点；OBS 题号已落到 trunk（OBS-05/08/09/18/19/25/40/71） | tree_v15 catalog 含上述 OBS |

## 待补 OBS 候选（正后方可见题，未测过 → 不编造取值）
1. **肩线 vs 髋线分离**（准备末）：区分「扭」vs「转」——见 trunk NEW-OBS 候选
2. **OBS-40 / OBS-18**：臂是否先于胸（正后方最合适机位；本轮 facts 未问）
3. **OBS-25 / OBS-71**：髋先 / 骨盆是否减速（前挥）
4. **OBS-09**：触球时髋是否转回

## V5
- 本轮 **skip**（如实；与 v14 甜区代决一致）。recall@3 历史为 0。

## 硬约束
- tree.json v1 未覆盖 ✓；阈值未放宽 ✓；holdout 未入树 ✓；禁令口令未写入 ✓

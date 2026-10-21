# e2e book_root_v1 — 2026-10-21

> 书根骨架 + 视频挂接 + 视觉层 + 确定性门闩。不覆盖 tree.json v1 / tree_vN；holdout 仅知识挂接。

## PASS/FAIL 总表

| 检查 | 结果 |
|---|---|
| `check_book_root.py --selftest`（原话/页码、attach↔orphans 覆盖、悬空 id、禁词、8 坏样例） | **PASS** |
| `check_trunk_fixture.py --selftest`（rear 固定样本 + 10 变异） | **PASS** |
| `verify_all.py`（tree_v18 + binding/merge/manifest/neutrality） | **PASS** |

## 书根骨架

- **树干（BR-TRUNK）**：正手的根是容错的击球（身前、离开身体、保持拍臂夹角与拍面）；旋转动力链只是为了让同一个容错击球更快（书 p.13 / 22 / 23 / 108）。
- **粗枝（章节，页码）**：见 `book_root_v1.json` 31 章；诊断粗枝 LB-1 / LB-2 挂接见下。
- **现有粗枝 1（髋肩 coil 不足 → 手臂代偿）**：主挂 **CH18**（Lead With The Hips, p.83–88）、**CH19**（Abdominal Coil, p.89–91）、**CH24**（POWER Checklist, p.115–121）；旁挂 CH04 / CH10 / CH23 / CH08。**建议改说法**：书 p.119 明说肩髋分离角不是必需，核心应写「髋没有先启动 / 髋胸没转回」；「手臂往上兜」来自视频，书对应的是「手臂被卡、用小肌肉」（p.30）。
- **现有粗枝 2（启动过早）**：主挂 **CH13**（Inhale and Wait, p.74–75）；旁挂 CH19 / CH23 / CH04。**建议改说法**：书的「早」分两种——相对来球的等待（CH13）vs 链内过早开胸（CH19/23，应作 LB-1 叶子）；具体「弹跳最高点 / 下落 80%」非书出处。

## 挂接统计

| 项 | 数 |
|---|---|
| tree_v18 节点 | 282 |
| episode 卡 | 86（磁盘实际；任务写 87） |
| 合计 | 368 |
| attach | **182**（tree 134 + episode 48；holdout 标 true 8） |
| orphans | **186**（A 书没讲 163 / B 书视频冲突 10 / C 拿不准 13） |

各章挂接（一条可挂多章，按章计）：CH20 25、CH27 22、CH04/CH18 各 21、CH24 20、CH23 19、CH11/CH17 各 16、CH19 14、CH12 11、CH28 10、CH21 8、CH01/05/10/14/22 各 7、CH03 6、其余 ≤4；CH06/08/15/16/29/31 无挂接。

### 3 个最值得用户看的孤儿 / 冲突

1. **髋用法书视频正面冲突**（B：`N-C_UJPUDUX0JIA_C02`、`N-C_6AK4ZSCH_GC_C03`）— 书 p.83/117 有意识蹬后腿、「twisting the hips back towards the net」；视频反对有意把持拍侧髋往前推。直接关系到粗枝 1 说法与处方。
2. **Unit Turn 要不要含髋**（B：`EP:ui-RFRdQKRk` holdout）— 书 p.59 髋也转离目标；视频「The unit doesn't include your hips」。
3. **启动时机：等待 vs 不停顿**（C：`GsHkML2mVEI` / `wd4YRQW3TOc` 相关）— 书 p.75 准备后等待是正常最后一步；两期视频说引拍后停顿毁掉拉伸。关系到粗枝 2，未硬挂。

## 正后方固定样本可用 OBS（`book_visual_v1.rear_fixture_obs`）

已答（fixture 实测）：OBS-18、OBS-25、OBS-09、OBS-08。
可答未测：OBS-10、OBS-04、OBS-05、OBS-15、OBS-38、OBS-33、OBS-36、OBS-32、OBS-02、OBS-07、OBS-16。

## 产物

- `docs/diagnosis/ftt_tree/book_root_v1.json` + `book_root_v1.md`
- `docs/diagnosis/ftt_tree/book_attach_v1.json`
- `docs/diagnosis/ftt_tree/book_orphans_v1.md`
- `docs/diagnosis/ftt_tree/book_visual_v1.json`
- `docs/diagnosis/pipeline/check_book_root.py`
- `books/ftt_fault_tolerant_forehand/original.txt`（pdftotext -layout，供门闩检索）

## 下一步缺口（≤5）

1. 用户裁决 B 类 10 条书视频冲突（尤其髋 / Unit Turn / 非持拍手）。
2. 按书证改粗枝 1 说法（coil → 髋先启动；「往上兜」标视频源）。
3. 粗枝 2 拆成「相对来球等待」vs「链内过早开胸」叶子。
4. 正后方可答未测 OBS（OBS-10/04/05/15/38/33…）补进 fixture。
5. CH06/08/15/16/29/31 无视频挂接——后续 episode 或持 orphan。

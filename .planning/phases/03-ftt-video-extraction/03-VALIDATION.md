# Phase 3: FTT Video Extraction - Validation

**Created:** 2026-04-03
**Source:** 03-RESEARCH.md Validation Architecture section

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | existing pytest config |
| Quick run command | `python -m pytest tests/ -x -q --tb=short` |
| Full suite command | `python -m pytest tests/ -v` |

## Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | Created By |
|--------|----------|-----------|-------------------|------------|
| FTT-01 | Channel enumeration produces 73 videos (33 analyzed + 40 pending) | unit | `pytest tests/test_ftt_video_pipeline.py::test_inventory_count -x` | Plan 03-01 |
| FTT-02 | Existing analyses parsed into per-video sections with real concepts | unit | `pytest tests/test_ftt_video_pipeline.py::test_parse_existing -x` | Plan 03-01 |
| FTT-03 | Gemini API returns valid analysis for a video (proxy base_url threaded) | integration | `pytest tests/test_ftt_video_pipeline.py::test_gemini_video -x` | Plan 03-01 |
| FTT-04 | Concepts merge into registry without explosion (300-500 total) | unit | `pytest tests/test_ftt_merge.py::test_registry_merge -x` | Plan 03-06 |
| FTT-05 | Diagnostic chains extracted from video content (>= 3 chains) | unit | `pytest tests/test_ftt_merge.py::test_diagnostic_chains -x` | Plan 03-06 |

## Test Files

| File | Covers | Created By |
|------|--------|------------|
| `tests/test_ftt_video_pipeline.py` | FTT-01, FTT-02, FTT-03: video state, analysis, extraction | Plan 03-01 |
| `tests/test_ftt_merge.py` | FTT-04, FTT-05: registry merge, diagnostic chains | Plan 03-06 |

## Integration Validation Points

### State Consistency (post-merge, Plan 03-06)
```bash
python3 -c "
import json
state = json.load(open('knowledge/state/ftt_video_state.json'))
total = len(state['videos'])
by_status = {}
for v in state['videos'].values():
    s = v['status']
    by_status[s] = by_status.get(s, 0) + 1
print(f'Total: {total} (expect 73)')
print(f'Status: {by_status}')
assert total == 73
assert by_status.get('pending', 0) == 0
print('OK')
"
```

### Registry Size (post-merge, Plan 03-06)
```bash
python3 -c "
import json
reg = json.load(open('knowledge/extracted/_registry_snapshot.json'))
print(f'Registry: {len(reg)} concepts (expect 300-500)')
assert 300 <= len(reg) <= 500
print('OK')
"
```

### Per-Video Extraction Quality (post Plans 02-05)
```bash
python3 -c "
import json, os
d = 'knowledge/extracted/ftt_videos'
files = [f for f in os.listdir(d) if f.endswith('.json') and not f.startswith('_')]
total_concepts = 0
empty_files = []
for f in files:
    data = json.load(open(os.path.join(d, f)))
    n = len(data.get('concepts', []))
    total_concepts += n
    if n < 2:
        empty_files.append(f)
print(f'Files: {len(files)}, Total concepts: {total_concepts}')
print(f'Files with < 2 concepts: {empty_files}')
assert total_concepts > 100
print('OK')
"
```

### Diagnostic Chains (post Plan 03-06)
```bash
python3 -c "
import json
chains = json.load(open('knowledge/extracted/ftt_video_diagnostic_chains.json'))
print(f'Chains: {chains[\"count\"]} (expect >= 3)')
assert chains['count'] >= 3
print('OK')
"
```

## Race Condition Mitigation

Plans 03-02 through 03-05 run in Wave 2 (parallel). To prevent race conditions on shared state:
- Each plan writes its own state slice: `batch0_state.json`, `batch1_state.json`, `batch2_state.json`, `batch3_state.json`
- Plan 03-06 (Wave 3) merges all slices into canonical `ftt_video_state.json`
- Per-video extraction JSONs are safe (each plan writes different video IDs, no overlap)

## API Rate Limit Mitigation

Three parallel API batch plans (03-03, 03-04, 03-05) use staggered starts:
- Batch 1 (03-03): starts immediately, 20s between calls
- Batch 2 (03-04): 30s initial delay, 20s between calls
- Batch 3 (03-05): 60s initial delay, 20s between calls

## Video Count Reconciliation

RESEARCH.md narrative mentions 70 total / 37 pending, but the actual tables contain:
- 33 unique already-analyzed videos (table rows 1-33)
- 40 pending unanalyzed videos (table rows 1-40)
- **73 total unique videos**

All plans use the table data (73/33/40) as source of truth, not the narrative counts.

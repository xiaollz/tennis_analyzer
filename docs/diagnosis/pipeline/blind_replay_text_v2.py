#!/usr/bin/env python3
"""Text blind replay, variant 2 (NEW script; blind_replay_text.py and its results.json are untouched).

Variant = cause retrieval is independent of problem matching: prompts/tree_retrieve_v2.md asks for problems (TASK A) and causes (TASK B)
as two separate tasks and has NO global "none_match" abstain flag, so an unmatched problem never silences the cause list.
Same holdout, same gold (copied from tree_blind_v1/gold), same scoring function (blind_replay_text.score), same judge prompt.

Configs (each run exactly ONCE; prompts were fixed and committed BEFORE any holdout call; no prompt was tuned on this holdout):
  main : tree v2 (tree_v2.json) + tree_retrieve_v2.md; library menu shows "also worded:" lines for merged groups;
         answer key = tree_judge_v1.md over the same menu (18 judge calls on the v2 tree).
  ref  : tree v1 (tree.json) + tree_retrieve_v2.md with the plain v1 menu; answer key = the v1 key already on disk (same tree) => isolates
         the retrieval-variant effect on the old tree.
usage: blind_replay_text_v2.py <main|ref|all> [--workers N]
Key only from env MICU_GEMINI_KEY (never written). Raw prompts/responses -> /workspace/s1_work/tree_blind_v2 (outside the repo).
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blind_replay_text as brt
from common import DIAG_DIR, load_json

W1 = brt.WORK  # tree_blind_v1 (gold + v1 answer keys live here)
W2 = Path("/workspace/s1_work/tree_blind_v2")
HERE = Path(__file__).resolve().parent
OUTDIR = DIAG_DIR / "replay" / "text_blind"
V2_TREE = DIAG_DIR / "ftt_tree" / "tree_v2.json"
V1_TREE = DIAG_DIR / "ftt_tree" / "tree.json"


def menu_plain(tree):
    return brt.tree_menu(tree)


def menu_variants(tree, merge_groups):
    """Same as the plain menu, plus 'also worded:' lines (original member statements) under each merged cause node."""
    also = {}
    for g in merge_groups:
        also[g["rep"]] = [m["text"] for m in g["members"] if m["node"] != g["rep"]] + [a for m in g["members"] for a in m["also_worded"]]
    probs = [f"{n['id']} | {n['name']}" for n in tree["nodes"] if n["kind"] == "problem"]
    cause = []
    for n in tree["nodes"]:
        if n["kind"] == "cause":
            cause.append(f"{n['id']} | {n['name']}")
            for a in dict.fromkeys(also.get(n["id"], [])):
                if a != n["name"]:
                    cause.append(f"    also worded: {a}")
    return "PROBLEM NODES\n" + "\n".join(probs) + "\n\nCAUSE NODES\n" + "\n".join(cause)


def load_golds(holdout):
    have = [v for v in holdout if (brt.S1.SUBS / f"{v}.norm.tsv").exists() and (brt.S1.SUBS / f"{v}.en.vtt").exists()]
    golds = {}
    for v in have:
        src = W1 / "gold" / v / "gold.json"
        dst = W2 / "gold" / v / "gold.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst)
        golds[v] = load_json(dst)
    return have, golds


def retrieve(args):
    cfg, vid, n, menu = args
    d = W2 / f"retrieve_{cfg}" / vid
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run{n}.json"
    if p.exists():
        return load_json(p)
    tpl = (HERE / "prompts" / "tree_retrieve_v2.md").read_text(encoding="utf-8")
    prompt = tpl.replace("{TREE}", menu).replace("{TRANSCRIPT}", brt.transcript_block(vid))
    assert "屈书毅" not in prompt and "用户" not in prompt
    if n == 1:
        (d / "prompt_run1.txt").write_text(prompt, encoding="utf-8")
    out, u, txt = brt.gcall(prompt, brt.RET_TEMPS[n], f"ret{n}")
    (d / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
    res = {"id": vid, "run": n, "ok": out is not None, "usage": u, "out": out}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def judge(args):
    vid, gold, menu = args
    d = W2 / "judge_main" / vid
    d.mkdir(parents=True, exist_ok=True)
    p = d / "key.json"
    if p.exists():
        return load_json(p)
    lines = [f"G-P | {g}" for g in gold["problems"][:1]] + [f"G-C{i} | {t}" for i, t in enumerate(gold["causes"], 1)]
    tpl = (HERE / "prompts" / "tree_judge_v1.md").read_text(encoding="utf-8")
    prompt = tpl.replace("{GOLD}", "\n".join(lines)).replace("{TREE}", menu)
    (d / "prompt.txt").write_text(prompt, encoding="utf-8")
    out, u, txt = brt.gcall(prompt, 0.0, "judge")
    (d / "raw.txt").write_text(txt, encoding="utf-8")
    res = {"id": vid, "ok": out is not None, "usage": u, "matches": (out or {}).get("matches", {}) if isinstance(out, dict) else {}}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def run_cfg(cfg, workers):
    mm = load_json(DIAG_DIR / "ftt_tree" / "mech_merge.json")
    tree = load_json(V2_TREE if cfg == "main" else V1_TREE)
    menu = menu_variants(tree, mm["groups"]) if cfg == "main" else menu_plain(tree)
    holdout = sorted(tree["holdout_episodes"])
    have, golds = load_golds(holdout)
    usable = [v for v in have if golds[v]["usable"]]
    ex = ThreadPoolExecutor(max_workers=workers)
    jobs = [(cfg, v, n, menu) for v in usable for n in (1, 2, 3)]
    rets = {(v, n): r for (_, v, n, _), r in zip(jobs, ex.map(retrieve, jobs))}
    if cfg == "main":
        jj = [(v, golds[v], menu) for v in usable if golds[v]["is_teaching"] != "non_teaching"]
        keys = {v: r for (v, _, _), r in zip(jj, ex.map(judge, jj))}
    else:
        keys = {v: load_json(W1 / "judge" / v / "key.json") for v in usable if (W1 / "judge" / v / "key.json").exists()}
    m, gate, rows = brt.score(tree, holdout, golds, rets, keys)
    usage = Counter()
    for r in list(rets.values()) + (list(keys.values()) if cfg == "main" else []):
        usage.update(r.get("usage") or {})
    res = {"config": cfg, "tree": "tree_v2.json" if cfg == "main" else "tree.json (v1)", "retrieval_prompt": "prompts/tree_retrieve_v2.md",
           "menu": "variants (also worded)" if cfg == "main" else "plain (v1)",
           "answer_key": "tree_judge_v1 over the same menu, run now" if cfg == "main" else "v1 key reused (same tree)",
           "metrics": m, "gate_reference": gate, "gemini_usage": dict(usage), "model": brt.S1.MODEL,
           "retrieve_calls": len(jobs), "judge_calls": len(jj) if cfg == "main" else 0,
           "holdout_without_transcript": [v for v in holdout if v not in have], "rows": rows}
    s = json.dumps(res, ensure_ascii=False, indent=1)
    assert brt.key() not in s
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"results_v2_indep_{cfg}.json").write_text(s + "\n", encoding="utf-8")
    # compact artifacts (retrieved ids per run) for review
    art = {}
    for v in usable:
        art[v] = {"title": golds[v].get("title"), "answer_key": keys.get(v, {}).get("matches", {}),
                  "retrieval": {f"run{n}": {"video_is_teaching": (rets[(v, n)]["out"] or {}).get("video_is_teaching"), "problems": (rets[(v, n)]["out"] or {}).get("problems"),
                                            "causes": (rets[(v, n)]["out"] or {}).get("causes")} for n in (1, 2, 3) if isinstance(rets[(v, n)]["out"], dict)}}
    (OUTDIR / f"artifacts_v2_indep_{cfg}.json").write_text(json.dumps(art, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(cfg, json.dumps(m, ensure_ascii=False, indent=1))
    print("usage", dict(usage))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", choices=["main", "ref", "all"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for c in (["main", "ref"] if a.cfg == "all" else [a.cfg]):
        run_cfg(c, a.workers)

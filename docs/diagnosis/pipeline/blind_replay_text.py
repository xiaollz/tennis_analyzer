#!/usr/bin/env python3
"""Transcript blind replay (supplement to V5): can the tree find a HOLDOUT video's problem + causes from its subtitles alone?

NOT the PLAN §4.4 muted-clip replay (that needs audio-stripped clips and facts; see blind_replay.py). This one:
  gold     : holdout episode -> gold problem/causes. Verified card if one exists, otherwise 2 raw Stage-1 runs
             (ep_s1_v6, same extractor as the cards) kept only if every run passes validate_cards + check_quotes.
  retrieve : Gemini sees ONLY the holdout transcript + the tree's problem/cause node texts (prompts/tree_retrieve_v1.md),
             3 runs; the gold is never shown. Output = node ids (+confidence).
  judge    : one temperature-0 call per case maps each gold item to the tree nodes that state the same mechanism
             (prompts/tree_judge_v1.md) = the answer key (independent of retrieval). Empty key = the tree has a gap.
  score    : pure script. hit = retrieved id in answer key (one hop upstream counts for causes, as in PLAN §4.4).
usage: blind_replay_text.py <gold|retrieve|judge|score|all> [--tree T] [--workers N] [--out results.json]
Key only from env MICU_GEMINI_KEY (never written). Raw prompts/responses go to /workspace/s1_work/tree_blind_v1 (outside the repo).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import check_quotes
import check_rerun
import s1_extract as S1
import validate_cards
from common import CONFIG, DIAG_DIR, fails, load_json
from textsim import Sim

WORK = Path("/workspace/s1_work/tree_blind_v1")
EP = DIAG_DIR / "episodes"
HERE = Path(__file__).resolve().parent
RET_TEMPS = {1: 0.2, 2: 0.4, 3: 0.4}
SIM_THRESHOLD = 0.40  # same constant as build_tree.MERGE_COSINE (lexical secondary metric only)


def key():
    return os.environ["MICU_GEMINI_KEY"]


def gcall(prompt, temp, tag):
    """One Gemini call with up to 2 retries; returns (parsed_json|None, usage dict, log)."""
    last = None
    tot = Counter()
    for k in range(3):
        text, log = S1.call_gemini(prompt, temp, key())
        for kk in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount", "thoughtsTokenCount"):
            tot[kk] += (log.get("usage") or {}).get(kk, 0)
        tot["calls"] += 1
        last = log
        if log["status"] == "ok" and text:
            try:
                return S1.parse_json(text), dict(tot), text
            except Exception as e:  # noqa: BLE001
                last = {"status": f"parse: {e}"}
        time.sleep(3)
    return None, dict(tot), json.dumps(last)


def transcript_block(vid):
    return S1.transcript_block(vid)


def tree_menu(tree):
    probs = [f"{n['id']} | {n['name']}" for n in tree["nodes"] if n["kind"] == "problem"]
    cause = [f"{n['id']} | {n['name']}" for n in tree["nodes"] if n["kind"] == "cause"]
    return "PROBLEM NODES\n" + "\n".join(probs) + "\n\nCAUSE NODES\n" + "\n".join(cause)


# ------------------------------------------------------------------ gold
def gold_one(vid):
    d = WORK / "gold" / vid
    d.mkdir(parents=True, exist_ok=True)
    card_p = EP / f"{vid}.card.json"
    if card_p.exists() and load_json(card_p)["status"] == "stage1_verified":
        c = load_json(card_p)
        g = {"id": vid, "source": "verified_card", "usage": {}, "usable": True, "why": "",
             "is_teaching": c["classification"]["is_teaching"], "title": c["video"]["title"],
             "problems": [c["problem"]["title_as_question"]],
             "causes": [x["statement"] for x in c["causes"]]}
        (d / "gold.json").write_text(json.dumps(g, ensure_ascii=False, indent=1))
        return g
    usage = Counter()
    old = d / "gold.json"
    if old.exists():
        usage.update(load_json(old).get("usage") or {})
    runs = {}
    prompt = S1.build_prompt(vid)
    for n in (1, 2):
        p = d / f"run{n}.card.json"
        if p.exists():
            runs[n] = load_json(p)
            continue
        (d / f"run{n}.prompt.txt").write_text(prompt, encoding="utf-8")
        raw, u, txt = gcall(prompt, S1.TEMPS[n], f"gold{n}")
        usage.update(u)
        (d / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
        if raw is not None:
            try:
                ev = S1.Ev()
                runs[n] = S1.normalize_card(raw, vid, [f"gold_r{n}_{vid[:6]}", "gold"], ev)
                p.write_text(json.dumps(runs[n], ensure_ascii=False, indent=1))
            except Exception as e:  # noqa: BLE001
                (d / f"run{n}.error.txt").write_text(f"{type(e).__name__}: {e}")
    g = {"id": vid, "source": "stage1_raw_2runs", "usage": dict(usage), "usable": False, "why": "", "problems": [], "causes": []}
    if len(runs) < 2:
        g["why"] = f"only {len(runs)} run(s) produced a card"
    else:
        bad = []
        for n, c in runs.items():
            f = validate_cards.check_card(c, f"{vid}.card.json") + check_quotes.check_card(c, f"{vid}.card.json", S1.SUBS)[0]
            if fails(f):
                bad.append(f"run{n}: {len(fails(f))} FAIL e.g. {fails(f)[0].msg[:100]}")
        rr = check_rerun.cards(runs[1], runs[2])
        g["rerun_agree"] = not fails(rr)
        teach = {c["classification"]["is_teaching"] for c in runs.values()}
        g["is_teaching"] = "non_teaching" if teach == {"non_teaching"} else ("teaching" if "non_teaching" not in teach else "ambiguous")
        g["title"] = runs[1]["video"]["title"]
        if bad:
            g["why"] = "gold runs failed verify: " + "; ".join(bad)
        else:
            g["usable"] = True
            g["problems"] = [c["problem"]["title_as_question"] for c in runs.values() if c["problem"]["title_as_question"]]
            g["causes"] = [x["statement"] for c in runs.values() for x in c["causes"]]
    (d / "gold.json").write_text(json.dumps(g, ensure_ascii=False, indent=1))
    return g


# ------------------------------------------------------------------ retrieve
def retrieve_one(args):
    vid, n, tree = args
    d = WORK / "retrieve" / vid
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run{n}.json"
    if p.exists():
        return load_json(p)
    tpl = (HERE / "prompts" / "tree_retrieve_v1.md").read_text(encoding="utf-8")
    prompt = tpl.replace("{TREE}", tree_menu(tree)).replace("{TRANSCRIPT}", transcript_block(vid))
    assert "屈书毅" not in prompt and "用户" not in prompt
    if n == 1:
        (d / "prompt_run1.txt").write_text(prompt, encoding="utf-8")
    out, u, txt = gcall(prompt, RET_TEMPS[n], f"ret{n}")
    (d / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
    res = {"id": vid, "run": n, "ok": out is not None, "usage": u, "out": out}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


# ------------------------------------------------------------------ judge
def judge_one(args):
    vid, gold, tree = args
    d = WORK / "judge" / vid
    d.mkdir(parents=True, exist_ok=True)
    p = d / "key.json"
    if p.exists():
        return load_json(p)
    lines = [f"G-P | {g}" for g in gold["problems"][:1]]  # first problem statement is the gold problem
    lines += [f"G-C{i} | {t}" for i, t in enumerate(gold["causes"], 1)]
    tpl = (HERE / "prompts" / "tree_judge_v1.md").read_text(encoding="utf-8")
    prompt = tpl.replace("{GOLD}", "\n".join(lines)).replace("{TREE}", tree_menu(tree))
    (d / "prompt.txt").write_text(prompt, encoding="utf-8")
    out, u, txt = gcall(prompt, 0.0, "judge")
    (d / "raw.txt").write_text(txt, encoding="utf-8")
    res = {"id": vid, "ok": out is not None, "usage": u, "matches": (out or {}).get("matches", {}) if isinstance(out, dict) else {}}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


# ------------------------------------------------------------------ score
def score(tree, holdout_ids, golds, rets, keys):
    ids = {n["id"] for n in tree["nodes"]}
    kind = {n["id"]: n["kind"] for n in tree["nodes"]}
    up = {}
    for e in tree["edges"]:
        up.setdefault(e["to"], set()).add(e["from"])
    cause_txt = [n["name"] for n in tree["nodes"] if n["kind"] == "cause"]
    sim = Sim(cause_txt)
    rows = []
    for vid in holdout_ids:
        g = golds.get(vid)
        row = {"episode": vid, "title": (g or {}).get("title", ""), "gold_source": (g or {}).get("source"), "excluded": None}
        if not g or not g["usable"]:
            row["excluded"] = (g or {}).get("why") or "no gold (no transcript)"
            rows.append(row); continue
        row["is_teaching"] = g["is_teaching"]
        k = keys.get(vid, {}).get("matches", {})
        kp = {x for x in k.get("G-P", []) if x in ids and kind[x] == "problem"}
        kc = {f"G-C{i}": {x for x in k.get(f"G-C{i}", []) if x in ids and kind[x] == "cause"} for i in range(1, len(g["causes"]) + 1)}
        row["gold_problem"] = g["problems"][:1]
        row["gold_causes"] = len(g["causes"])
        row["key_problem"] = sorted(kp)
        row["key_cause_items_covered"] = sum(1 for v in kc.values() if v)
        row["lex_best_cos"] = [round(max((sim.cos(c, t) for t in cause_txt), default=0.0), 3) for c in g["causes"]]
        runs = []
        for n in (1, 2, 3):
            r = rets.get((vid, n))
            if not r or not r["ok"] or not isinstance(r["out"], dict):
                runs.append(None); continue
            o = r["out"]
            probs = [x for x in (o.get("problems") or []) if isinstance(x, dict) and x.get("id") in ids and kind.get(x["id"]) == "problem"][:3]
            caus = [x for x in (o.get("causes") or []) if isinstance(x, dict) and x.get("id") in ids and kind.get(x["id"]) == "cause"][:3]
            pid = [x["id"] for x in probs]
            cid = [x["id"] for x in caus]
            cexp = set(cid)
            for c in cid:
                cexp |= up.get(c, set())
            union_key_c = set().union(*kc.values()) if kc else set()
            runs.append({
                "problems": pid, "causes": cid, "none_match": bool(o.get("none_match")),
                "high": any(x.get("confidence") == "high" for x in probs + caus),
                "p_hit3": bool(set(pid) & kp), "p_hit1": bool(pid and pid[0] in kp),
                "c_hit3_strict": bool(set(cid) & union_key_c), "c_hit3_exp": bool(cexp & union_key_c),
                "c_hit1": bool(cid and cid[0] in union_key_c),
                "gold_cause_recall": sum(1 for v in kc.values() if v & cexp) / len(kc) if kc else 0.0,
            })
        row["runs"] = runs
        valid = [r for r in runs if r]
        row["valid_runs"] = len(valid)
        row["problem_covered"] = bool(kp)
        row["cause_covered"] = any(kc.values())
        row["negative"] = g["is_teaching"] == "non_teaching"
        row["top1_problem_stable"] = (Counter(r["problems"][0] if r["problems"] else "NONE" for r in valid).most_common(1)[0][1] / len(valid)) if valid else None
        rows.append(row)
    used = [r for r in rows if not r["excluded"]]
    pos = [r for r in used if not r["negative"] and r["is_teaching"] != "ambiguous"]
    neg = [r for r in used if r["negative"]]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None

    def per_run(rs, f):  # mean over cases of the mean over valid runs
        return mean([mean([f(x) for x in r["runs"] if x]) for r in rs if r["valid_runs"]])
    covered_any = [r for r in pos if r["problem_covered"] or r["cause_covered"]]
    m = {
        "cases_scored": len(used), "positives": len(pos), "negatives": len(neg), "episodes": len({r["episode"] for r in used}),
        "excluded": [{"episode": r["episode"], "why": r["excluded"]} for r in rows if r["excluded"]],
        "ambiguous_gold": [r["episode"] for r in used if r["is_teaching"] == "ambiguous"],
        "coverage_problem": mean([r["problem_covered"] for r in pos]),
        "coverage_cause_any": mean([r["cause_covered"] for r in pos]),
        "coverage_any": mean([bool(r["problem_covered"] or r["cause_covered"]) for r in pos]),
        "p_hit3": per_run(pos, lambda x: x["p_hit3"]), "p_hit1": per_run(pos, lambda x: x["p_hit1"]),
        "c_hit3_strict": per_run(pos, lambda x: x["c_hit3_strict"]), "c_hit3_with_upstream": per_run(pos, lambda x: x["c_hit3_exp"]),
        "c_hit1": per_run(pos, lambda x: x["c_hit1"]),
        "case_hit3 (problem OR cause-with-upstream)": per_run(pos, lambda x: x["p_hit3"] or x["c_hit3_exp"]),
        "case_hit1 (problem OR cause top-1)": per_run(pos, lambda x: x["p_hit1"] or x["c_hit1"]),
        "gold_cause_recall_with_upstream": per_run(pos, lambda x: x["gold_cause_recall"]),
        "conditional_on_tree_coverage": {
            "n": len(covered_any),
            "case_hit3": per_run(covered_any, lambda x: x["p_hit3"] or x["c_hit3_exp"]),
            "p_hit3": per_run(covered_any, lambda x: x["p_hit3"]),
        },
        "uncovered_positives_abstain_rate": mean([mean([x["none_match"] or not (x["problems"] or x["causes"]) for x in r["runs"] if x]) for r in pos if not (r["problem_covered"] or r["cause_covered"]) and r["valid_runs"]]),
        "negative_high_conf_fp": per_run(neg, lambda x: x["high"]),
        "top1_problem_stable_ge_2of3": mean([r["top1_problem_stable"] >= 2 / 3 for r in pos if r["top1_problem_stable"] is not None]),
        "lexical_gold_cause_cos_ge_0.40_any_tree_node": mean([c >= SIM_THRESHOLD for r in pos for c in r["lex_best_cos"]]),
    }
    g1 = CONFIG["replay_gate"]["G1"]
    gate = {
        "reference_only_note": "G1 numbers applied to the TEXT replay for reference; the official V5 (muted clips) was not run",
        "cases >= %d" % g1["min_cases"]: m["positives"] + m["negatives"] >= g1["min_cases"],
        "episodes >= %d" % g1["min_episodes"]: m["episodes"] >= g1["min_episodes"],
        "recall@3 >= %s" % g1["recall_at_3"]: (m["case_hit3 (problem OR cause-with-upstream)"] or 0) >= g1["recall_at_3"],
        "negatives > 0 and FP <= %s" % g1["max_negative_high_conf"]: m["negatives"] > 0 and (m["negative_high_conf_fp"] or 0) <= g1["max_negative_high_conf"],
    }
    return m, gate, rows


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gold", "retrieve", "judge", "score", "all"])
    ap.add_argument("--tree", default=str(DIAG_DIR / "ftt_tree" / "tree.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(DIAG_DIR / "replay" / "text_blind" / "results.json"))
    a = ap.parse_args(argv)
    tree = load_json(a.tree)
    holdout = sorted(tree["holdout_episodes"])
    have = [v for v in holdout if (S1.SUBS / f"{v}.norm.tsv").exists() and (S1.SUBS / f"{v}.en.vtt").exists()]
    st = a.stage
    golds, rets, keys = {}, {}, {}
    ex = ThreadPoolExecutor(max_workers=a.workers)
    if st in ("gold", "all", "score", "judge", "retrieve"):
        for v, g in zip(have, ex.map(gold_one, have)):
            golds[v] = g
        print("gold usable:", [v for v in have if golds[v]["usable"]], "unusable:", {v: golds[v]["why"][:80] for v in have if not golds[v]["usable"]})
    usable = [v for v in have if golds[v]["usable"]]
    if st in ("retrieve", "all", "score"):
        jobs = [(v, n, tree) for v in usable for n in (1, 2, 3)]
        for (v, n, _), r in zip(jobs, ex.map(retrieve_one, jobs)):
            rets[(v, n)] = r
    if st in ("judge", "all", "score"):
        jobs = [(v, golds[v], tree) for v in usable if golds[v]["is_teaching"] != "non_teaching"]
        for (v, _, _), r in zip(jobs, ex.map(judge_one, jobs)):
            keys[v] = r
    if st in ("score", "all"):
        m, gate, rows = score(tree, holdout, golds, rets, keys)
        usage = Counter()
        for g in golds.values():
            usage.update(g.get("usage") or {})
        for r in list(rets.values()) + list(keys.values()):
            usage.update(r.get("usage") or {})
        res = {"metrics": m, "gate_reference": gate, "gemini_usage": dict(usage), "model": S1.MODEL,
               "holdout_without_transcript": [v for v in holdout if v not in have], "rows": rows}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        s = json.dumps(res, ensure_ascii=False, indent=1)
        assert key() not in s
        Path(a.out).write_text(s + "\n", encoding="utf-8")
        print(json.dumps(m, ensure_ascii=False, indent=1))
        print(json.dumps(gate, ensure_ascii=False, indent=1))
        print("usage:", dict(usage))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

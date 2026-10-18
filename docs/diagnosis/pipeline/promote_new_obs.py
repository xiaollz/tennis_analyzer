#!/usr/bin/env python3
"""Promote NEW-OBS candidates to catalog observables (OBS-19...). Model only groups; selection + writing are script rules.

usage: promote_new_obs.py group   # 2 model runs: group the NEW-OBS cluster representatives into observable 'families' (prompts/obs_newcluster_v1.md)
       promote_new_obs.py apply   # select by rule, append to obs_catalog.json, write observables/OBS-NN_*.md, write ftt_tree/new_obs_promotion.json
Input: ftt_tree/obs_binding.json (v2 binding; its NEW-OBS clusters). Family = two clusters grouped together by BOTH runs (partition intersection).
Selection rule (fixed here, recorded in new_obs_promotion.json):
  eligible = family with >= 2 distinct source episodes, representative layer in {A, B}, camera in {S, B} (or layer A with camera 'any'),
             no config.neutral_prompt_forbidden term in question/values, not a duplicate of an existing catalog question;
  rank     = distinct episodes desc, then distinct bound items desc, then cluster id; take the first MAX_PROMOTE (8); fewer than 5 -> stop and report.
Question and values are copied verbatim from the family representative (a model-written candidate); nothing is hand-edited and no
discriminator is invented. Ids are assigned OBS-19.. in rank order.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import blind_replay_text as brt
from common import CONFIG, DIAG_DIR, load_json
from obs_bind import partition
from tv2_common import PIPE, RUN_TEMPS, WORK

BIND = DIAG_DIR / "ftt_tree" / "obs_binding.json"
OUT = DIAG_DIR / "ftt_tree" / "new_obs_promotion.json"
CAT = PIPE / "obs_catalog.json"
OBS_DIR = DIAG_DIR / "observables"
MAX_PROMOTE, MIN_PROMOTE = 8, 5
FW = WORK / "families"


def clusters_view(b):
    cand = {c["cid"]: c for c in b["new_obs_candidates"]}
    out = {}
    for c in b["new_obs_clusters"]:
        rep = cand[c["rep"]]
        items = sorted({cand[m]["item"] for m in c["members"]})
        out[c["id"]] = {"id": c["id"], "rep_cid": c["rep"], "description": rep["description"], "values": rep["values"], "layer": rep["layer"],
                        "camera": rep["camera"], "items": items, "episodes": sorted({b["items"][i]["episode"] for i in items})}
    return out


def cmd_group():
    cl = clusters_view(load_json(BIND))
    lines = [f"{c['id']} | {c['description']} | {', '.join(c['values'])} | {c['layer']} | {c['camera']}" for c in cl.values()]
    tpl = (PIPE / "prompts" / "obs_newcluster_v1.md").read_text(encoding="utf-8").split("\n---\n", 1)[1]
    prompt = tpl.replace("{ITEMS}", "\n".join(lines))
    FW.mkdir(parents=True, exist_ok=True)
    (FW / "prompt.txt").write_text(prompt, encoding="utf-8")
    for n in (1, 2):
        p = FW / f"run{n}.json"
        if p.exists():
            continue
        out, u, txt = brt.gcall(prompt, RUN_TEMPS[n], f"fam{n}")
        (FW / f"run{n}.raw.txt").write_text(txt, encoding="utf-8")
        p.write_text(json.dumps({"ok": out is not None, "usage": u, "out": out}, ensure_ascii=False, indent=1))
    print("family grouping runs saved for", len(cl), "clusters")


def slug(q):
    w = re.findall(r"[a-z]+", q.lower())
    stop = {"the", "a", "an", "is", "are", "does", "do", "at", "of", "to", "during", "in", "on", "or", "and", "how", "what", "where", "which"}
    return "_".join([x for x in w if x not in stop][:5])


def forbidden_hit(text):
    for t in CONFIG["neutral_prompt_forbidden"]:
        pat = re.escape(t) if not t.isascii() else r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])"
        if re.search(pat, text, re.I):
            return t
    return None


def cmd_apply():
    b = load_json(BIND)
    cl = clusters_view(b)
    runs = [load_json(FW / f"run{n}.json") for n in (1, 2)]
    ids = list(cl)
    parts = []
    for r in runs:
        groups = ((r["out"] or {}).get("groups") if r["ok"] and isinstance(r["out"], dict) else []) or []
        parts.append(partition(groups, ids))
    by = defaultdict(list)
    for i in ids:
        if i in parts[0] and i in parts[1]:
            by[(parts[0][i], parts[1][i])].append(i)
    fams, used = [], set()
    for ms in sorted((sorted(m) for m in by.values() if len(m) >= 2), key=lambda m: m[0]):
        fams.append(ms)
        used |= set(ms)
    fams += [[i] for i in ids if i not in used]
    cat = load_json(CAT)
    exist_q = {o["question"].strip().lower() for o in cat["observables"]}
    rows = []
    for ms in fams:
        # family representative: the member with the most episodes (ties: lowest id); description/values copied verbatim
        rep = max((cl[m] for m in ms), key=lambda c: (len(c["episodes"]), -int(c["id"].split("-")[-1])))
        eps = sorted({e for m in ms for e in cl[m]["episodes"]})
        its = sorted({i for m in ms for i in cl[m]["items"]})
        why = []
        if len(eps) < 2:
            why.append("fewer than 2 source episodes")
        if rep["layer"] not in ("A", "B"):
            why.append("layer not A/B")
        if not (rep["camera"] in ("S", "B") or (rep["layer"] == "A" and rep["camera"] == "any")):
            why.append("camera not explicit")
        h = forbidden_hit(rep["description"] + " " + " ".join(rep["values"]))
        if h:
            why.append(f"forbidden term {h!r}")
        if rep["description"].strip().lower() in exist_q:
            why.append("duplicate of existing observable")
        rows.append({"family": ms, "rep_cluster": rep["id"], "description": rep["description"], "values": rep["values"], "layer": rep["layer"], "camera": rep["camera"],
                     "episodes": eps, "items": its, "eligible": not why, "ineligible_because": why})
    elig = sorted([r for r in rows if r["eligible"]], key=lambda r: (-len(r["episodes"]), -len(r["items"]), r["rep_cluster"]))
    picked = elig[:MAX_PROMOTE]
    if len(picked) < MIN_PROMOTE:
        print(f"only {len(picked)} eligible families (< {MIN_PROMOTE}); nothing written")
        return 1
    if any(o["id"] == "OBS-19" for o in cat["observables"]):
        raise SystemExit("OBS-19 already in catalog (apply is not idempotent on purpose)")
    nxt = 19
    promoted = []
    for r in picked:
        oid = f"OBS-{nxt:02d}"
        nxt += 1
        fname = f"{oid}_{slug(r['description'])}.md"
        entry = {"id": oid, "src": fname, "layer": r["layer"], "camera": r["camera"], "reliability": "untested", "question": r["description"], "values": r["values"],
                 "origin": {"new_obs_clusters": r["family"], "episodes": r["episodes"], "promoted_by": "pipeline/promote_new_obs.py"}}
        cat["observables"].append(entry)
        cam = {"S": "S（正侧面）", "B": "B（正后方）", "any": "任意机位"}[r["camera"]]
        who = "A 层：普通手机视频里模型可答，人工抽帧复核" if r["layer"] == "A" else "B 层：需指定机位 + Pose / 人工抽帧；模型只作参考"
        md = [f"# {oid} {r['description']}", "",
              f"- **定义**：{r['description']} 取值：{' / '.join(r['values'])}（看不清则记 unclear）。",
              "- **看哪一帧**：题目里写明的时刻 / 阶段。", f"- **哪个机位**：{cam}。", f"- **谁来测**：{who}。",
              "- **可靠度**：**未测**（尚未对任何视频跑过事实抽取）。", "",
              f"来源：由 NEW-OBS 候选升级（obs_binding 簇 {', '.join(r['family'])}；涉及 {len(r['episodes'])} 期：{', '.join(r['episodes'])}）。升级规则见 `pipeline/promote_new_obs.py`；题目与取值逐字取自候选原文。不挂接任何故障、不含处方。", ""]
        (OBS_DIR / fname).write_text("\n".join(md), encoding="utf-8")
        promoted.append({"obs": oid, "file": f"observables/{fname}", **{k: r[k] for k in ("rep_cluster", "family", "description", "values", "layer", "camera", "episodes", "items")}})
    CAT.write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    doc = {"rule": __doc__.split("Selection rule")[1].split("Question and values")[0].strip(), "model": brt.S1.MODEL, "prompt": "pipeline/prompts/obs_newcluster_v1.md (reused on cluster representatives)",
           "families_total": len(rows), "families_multi_cluster": sum(len(r["family"]) > 1 for r in rows), "eligible": len(elig), "promoted": promoted,
           "not_promoted_eligible": [{k: r[k] for k in ("rep_cluster", "description", "episodes")} for r in elig[MAX_PROMOTE:]],
           "ineligible_summary": {w: sum(w in r["ineligible_because"] for r in rows) for w in sorted({x for r in rows for x in r["ineligible_because"]})},
           "families": rows, "usage": {k: sum((r["usage"] or {}).get(k, 0) for r in runs) for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount", "calls")}}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for p in promoted:
        print(p["obs"], p["layer"], p["camera"], len(p["episodes"]), p["episodes"], "|", p["description"], p["values"])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["group", "apply"])
    a = ap.parse_args()
    if a.cmd == "group":
        cmd_group()
    else:
        sys.exit(cmd_apply())

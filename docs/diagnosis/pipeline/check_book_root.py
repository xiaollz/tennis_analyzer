#!/usr/bin/env python3
"""Deterministic gate for the FTT book root skeleton (book_root_v1).
Checks:
 Q  every book-node quote is found verbatim (whitespace/quote-normalized) on its stated page of pdftotext output
 C  every tree_v18 node + every episode card is in attach XOR orphans (no omission, no double listing)
 R  every attach book_node id exists in book_root
 V  every visual entry references an existing book node, camera in {B,S,any}, OBS id in catalog or NEW-*
 F  prescription/drill text contains no banned cue words
 H  holdout episodes attached are marked holdout and never used as evidence
--selftest builds >=5 corrupted variants; each must be caught.
"""
import argparse, copy, json, os, re, subprocess, sys

BANNED = ["推肘", "降拍", "旋后", "伸肘", "送髋", "把髋送进球里", "髋往前顶"]
PRESC_KEYS = ("prescription_zh", "drill_zh", "prescription", "drill")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = os.path.join(ROOT, "docs/diagnosis")

def norm(s):
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\u00ad", "")
    s = re.sub(r"[\u2018\u2019\u201b`´]", "'", s)
    s = re.sub(r"[\u201c\u201d\u201e]", '"', s)
    s = re.sub(r"[\u2013\u2014\u2012]", "-", s)
    s = s.replace("\u2026", "...")
    s = re.sub(r"-\s*\n\s*", "-", s)  # keep hyphen, join line
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"-\s+", "-", s)  # hyphen split by line wrap, applied to both sides
    return s.strip().lower()

def load_pages(pdf, txt):
    if not os.path.exists(txt):
        subprocess.run(["pdftotext", "-layout", pdf, txt], check=True)
    raw = open(txt, encoding="utf-8").read().split("\f")
    return {i + 1: norm(p) for i, p in enumerate(raw)}

def book_nodes(br):
    out = []
    t = br["trunk"]; out.append(t)
    for ch in br["chapters"]:
        out.append(ch)
        out.extend(ch.get("symptoms", []))
    return out

def walk_presc(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in PRESC_KEYS and isinstance(v, str):
                yield path + "." + k, v
            else:
                yield from walk_presc(v, path + "." + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_presc(v, f"{path}[{i}]")

def parse_orphans(md):
    cats, cur = {}, None
    for line in md.splitlines():
        m = re.match(r"^##\s+([ABC])\b", line)
        if m: cur = m.group(1); continue
        if line.lstrip().startswith(("-", "|")) and cur:
            m = re.search(r"`([^`]+)`", line)
            if m: cats.setdefault(m.group(1), []).append(cur)
    return cats

def check(br, attach, orphans_md, visual, pages, tree_ids, ep_ids, holdout, catalog):
    errs = []
    nodes = book_nodes(br)
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)): errs.append("R: duplicate book node ids")
    idset = set(ids)
    for n in nodes:
        qs = n.get("quotes") or []
        if not qs: errs.append(f"Q: {n['id']} has no quote")
        for q in qs:
            t, p = q.get("text", ""), q.get("page")
            if len(t.split()) > 40: errs.append(f"Q: {n['id']} quote >40 words")
            if not isinstance(p, int) or p not in pages:
                errs.append(f"Q: {n['id']} bad page {p}"); continue
            if norm(t) not in pages[p]:
                where = [k for k, v in pages.items() if norm(t) in v]
                errs.append(f"Q: {n['id']} quote not on p.{p}" + (f" (found p.{where})" if where else " (not in book)") + f": {t[:60]}")
    for lb in br.get("legacy_branches", []):
        for a in lb.get("attached_to", []):
            if a not in idset: errs.append(f"R: legacy {lb['id']} -> missing {a}")
    att = attach["attachments"]
    seen = {}
    for a in att:
        nid = a["node_id"]; seen[nid] = seen.get(nid, 0) + 1
        bns = a["book_node"] if isinstance(a["book_node"], list) else [a["book_node"]]
        for b in bns:
            if b not in idset: errs.append(f"R: attach {nid} -> dangling {b}")
        if not a.get("reason"): errs.append(f"R: attach {nid} no reason")
        if not (a.get("obs") or a.get("visual_cue")): errs.append(f"R: attach {nid} no obs/visual_cue")
        ep = nid[3:] if nid.startswith("EP:") else None
        if ep in holdout and not a.get("holdout"): errs.append(f"H: {nid} holdout not marked")
        if a.get("holdout") and a.get("evidence"): errs.append(f"H: {nid} holdout used as evidence")
    orph = parse_orphans(orphans_md)
    want = set(tree_ids) | {"EP:" + e for e in ep_ids}
    for w in sorted(want):
        n = seen.get(w, 0) + len(orph.get(w, []))
        if n == 0: errs.append(f"C: {w} missing from attach and orphans")
        elif n > 1: errs.append(f"C: {w} listed {n} times")
    for k in list(seen) + list(orph):
        if k not in want: errs.append(f"C: unknown node {k}")
    for v in visual.get("book_nodes", []):
        if v["book_node"] not in idset: errs.append(f"V: visual -> dangling {v['book_node']}")
        for o in v.get("obs", []):
            if o.get("camera") not in ("B", "S", "any"): errs.append(f"V: {v['book_node']} {o.get('id')} bad camera")
            oid = o.get("id", "")
            if not (oid in catalog or oid.startswith("NEW")): errs.append(f"V: unknown OBS {oid}")
    for obj, name in ((br, "book_root"), (attach, "attach"), (visual, "visual")):
        for path, txt in walk_presc(obj):
            for w in BANNED:
                if w in txt: errs.append(f"F: banned '{w}' in {name}{path}")
    return errs

def main():
    ap = argparse.ArgumentParser()
    ft = os.path.join(D, "ftt_tree")
    ap.add_argument("--root", default=os.path.join(ft, "book_root_v1.json"))
    ap.add_argument("--attach", default=os.path.join(ft, "book_attach_v1.json"))
    ap.add_argument("--orphans", default=os.path.join(ft, "book_orphans_v1.md"))
    ap.add_argument("--visual", default=os.path.join(ft, "book_visual_v1.json"))
    ap.add_argument("--tree", default=os.path.join(ft, "tree_v18.json"))
    ap.add_argument("--episodes", default=os.path.join(D, "episodes"))
    ap.add_argument("--catalog", default=os.path.join(D, "pipeline/obs_catalog.json"))
    ap.add_argument("--pdf", default=os.path.join(ROOT, "books/ftt_fault_tolerant_forehand/original.pdf"))
    ap.add_argument("--txt", default="/tmp/ftt_original_layout.txt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    pages = load_pages(a.pdf, a.txt)
    tree = json.load(open(a.tree))
    tree_ids = [n["id"] for n in tree["nodes"]]
    holdout = set(tree.get("holdout_episodes", []))
    ep_ids = sorted(f[:-10] for f in os.listdir(a.episodes) if f.endswith(".card.json"))
    catalog = {o["id"] for o in json.load(open(a.catalog))["observables"]}
    br, att, vis = json.load(open(a.root)), json.load(open(a.attach)), json.load(open(a.visual))
    om = open(a.orphans, encoding="utf-8").read()
    args = (pages, tree_ids, ep_ids, holdout, catalog)
    errs = check(br, att, om, vis, *args)
    for e in errs: print("  ", e)
    ok = not errs
    print(f"[{'PASS' if ok else 'FAIL'}] real files: {len(errs)} errors; book nodes={len(book_nodes(br))} attach={len(att['attachments'])} orphans={len(parse_orphans(om))}")
    if a.selftest:
        cases = []
        b = copy.deepcopy(br); b["chapters"][0]["quotes"][0]["text"] = "The forehand is mostly about pushing the elbow toward the net at all times"; cases.append(("fabricated quote", "Q", b, att, om, vis))
        b = copy.deepcopy(br); q = b["chapters"][0]["quotes"][0]; q["page"] = q["page"] + 3 if q["page"] < 140 else q["page"] - 3; cases.append(("wrong page", "Q", b, att, om, vis))
        t = copy.deepcopy(att); t["attachments"] = t["attachments"][1:]; cases.append(("missing node", "C", br, t, om, vis))
        t = copy.deepcopy(att); t["attachments"][0]["book_node"] = "BR-CH99-NOPE"; cases.append(("dangling id", "R", br, t, om, vis))
        b = copy.deepcopy(br); b["chapters"][0].setdefault("symptoms", [{}]); b["chapters"][0]["symptoms"][0]["prescription_zh"] = "练习时主动送髋并推肘"; cases.append(("banned word", "F", b, att, om, vis))
        t = copy.deepcopy(att); t["attachments"].append(copy.deepcopy(t["attachments"][0])); cases.append(("double listed", "C", br, t, om, vis))
        b = copy.deepcopy(br); b["chapters"][0]["quotes"][0]["text"] = " ".join(["word"] * 41); cases.append(("quote >40 words", "Q", b, att, om, vis))
        v = copy.deepcopy(vis); v["book_nodes"][0]["obs"][0]["camera"] = "top"; cases.append(("bad camera", "V", br, att, om, v))
        st_ok = True
        for name, code, b, t, o, v in cases:
            e = check(b, t, o, v, *args)
            hit = any(x.startswith(code + ":") for x in e)
            st_ok &= hit
            print(f"  selftest {'CAUGHT' if hit else 'MISSED'}: {name} ({code})")
        print(f"[{'PASS' if st_ok else 'FAIL'}] selftest {len(cases)} bad cases")
        ok &= st_ok
    print("OVERALL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

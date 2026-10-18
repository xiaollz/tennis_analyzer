#!/usr/bin/env python3
"""Stage-1 (subtitle layer) extraction runner + deterministic normalizer.  stdlib only.

usage:
  s1_extract.py run   <VIDEO_ID> <RUN_N>   # call Gemini once (prompt = S1_PROMPT env, default ep_s1_v3), save raw run card + log
  s1_extract.py final <VIDEO_ID>           # pick passing run pair via check_rerun, write final card
  s1_extract.py consensus <VIDEO_ID>       # only when 3 runs exist and no raw pair passes: 3-run majority adjudication
Key comes ONLY from env MICU_GEMINI_KEY (never written/printed). Rendered prompts and raw
model responses go to /workspace/s1_work/<S1_WORK, default round3> (outside the repo; transcripts are not committed).

Normalization is deterministic and never rewrites content: it maps field aliases, pads ids,
snaps quotes to the verbatim subtitle excerpt, drops items without surviving evidence, and
records every action in <run>.norm_events.json so alias/repair counts can be reported.
"""
from __future__ import annotations
import copy, json, os, re, sys, time, urllib.error, urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_rerun  # noqa: E402
from common import CONFIG, ban_hits, load_json, load_transcript, locate_quote, sha256_file, tokens  # noqa: E402

ROOT = HERE.parent.parent.parent
SUBS = Path("/workspace/ftt_subs")
EP = ROOT / "docs/diagnosis/episodes"
WORK = Path("/workspace/s1_work") / os.environ.get("S1_WORK", "round3")
MODEL = "gemini-3.8-flash"
BASE = "https://www.micuapi.ai"
PROMPT_VERSION = os.environ.get("S1_PROMPT", "ep_s1_v3")
TEMPS = {1: 0.25, 2: 0.45, 3: 0.35}
BANS = ("推肘", "降拍", "旋后", "伸肘")

ENUM = {
    "is_teaching": ("teaching", "partial", "non_teaching"),
    "stroke": ("forehand", "backhand", "serve", "volley", "movement", "general", "other"),
    "format": ("lecture", "demo_with_player", "lesson_review", "pro_analysis", "short", "vlog_other"),
    "about": ("general", "demonstrator_natural", "demonstrator_induced", "pro_footage"),
    "speaker": ("coach", "player", "unknown"),
    "sym_kind": ("visible", "ball_outcome", "feel"),
    "phase": ("ready", "unit_turn", "placement", "forward_swing", "contact", "follow_through", "recovery", "whole"),
    "aop": ("active", "passive", "unclear"),
    "strength": ("coach_stated", "coach_implied", "inferred"),
    "angle": ("mechanics", "feel", "visual_model", "analogy", "contrast_demo", "misconception", "pro_example"),
    "sol_kind": ("drill", "cue", "check", "progression"),
    "scenario": ("machine_low", "mini_tennis", "shadow", "needs_feeder", "match_only"),
}


# ------------------------------------------------------------------ meta / prompt
def meta_for(vid):
    m = json.loads((SUBS / f"{vid}.meta.json").read_text())
    ud = m["upload_date"]
    if re.fullmatch(r"\d{8}", ud):
        ud = f"{ud[:4]}-{ud[4:6]}-{ud[6:8]}"
    return {"id": vid, "url": m.get("url") or f"https://www.youtube.com/watch?v={vid}", "title": m["title"],
            "channel": "Fault Tolerant Tennis", "duration_s": float(m["duration"]), "upload_date": ud,
            "metadata_source": "yt-dlp"}


def transcript_block(vid):
    out = []
    for row in (SUBS / f"{vid}.norm.tsv").read_text(encoding="utf-8").splitlines():
        if row.strip():
            a, _, t = row.split("\t", 2)
            out.append(f"[{int(float(a))}] {t}")
    return "\n".join(out)


def build_prompt(vid):
    tpl = (HERE / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")
    if "\n---\n" in tpl:  # only the part after the executor header goes to the model
        tpl = tpl.split("\n---\n", 1)[1].lstrip("\n")
    return tpl.replace("{META}", json.dumps(meta_for(vid), ensure_ascii=False, indent=2)).replace("{TRANSCRIPT}", transcript_block(vid))


# ------------------------------------------------------------------ gemini
def call_gemini(prompt, temperature, key):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 65536, "responseMimeType": "application/json"}}
    req = urllib.request.Request(f"{BASE}/v1beta/models/{MODEL}:generateContent", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key,
                                          "Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})
    t0 = time.time(); status, text, usage = "ok", "", None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            payload = json.loads(r.read().decode())
        text = "".join(p.get("text", "") for p in payload["candidates"][0]["content"]["parts"])
        usage = payload.get("usageMetadata")
    except urllib.error.HTTPError as e:
        status = f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")[:300]
    except Exception as e:  # noqa: BLE001
        status = f"{type(e).__name__}: {e}"
    return text, {"status": status, "elapsed_s": round(time.time() - t0, 1), "usage": usage}


def parse_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            raise
        return json.loads(m.group(0))


# ------------------------------------------------------------------ normalizer
class Ev:
    def __init__(self):
        self.c = Counter()
        self.detail = []

    def add(self, kind, detail=""):
        self.c[kind] += 1
        if detail and len(self.detail) < 200:
            self.detail.append(f"{kind}: {detail}")


def pick(obj, canon, aliases, ev, ctx, default=None):
    if canon in obj and obj[canon] not in (None, ""):
        return obj[canon]
    for a in aliases:
        if a in obj and obj[a] not in (None, ""):
            ev.add("alias", f"{ctx}.{canon}<-{a}")
            return obj[a]
    return default


def as_list(x, ev, ctx):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    ev.add("scalar_to_list", ctx)
    if isinstance(x, str):
        return [p.strip() for p in re.split(r"[;\n]", x) if p.strip()]
    return [x]


def norm_id(x, prefix):
    m = re.fullmatch(rf"\s*{prefix}[\s_-]*0*(\d+)\s*", str(x), re.I)
    return f"{prefix}{int(m.group(1)):02d}" if m else None


def norm_enum(val, key, ev, ctx, substr=False):
    allowed = ENUM[key]
    if val in allowed:
        return val
    s = str(val).strip().lower().replace(" ", "_").replace("-", "_")
    if s in allowed:
        ev.add("enum_case", f"{ctx}:{val}")
        return s
    if substr:
        hits = [a for a in allowed if a in s]
        if len(hits) == 1:
            ev.add("enum_substr", f"{ctx}:{val}->{hits[0]}")
            return hits[0]
    return val  # left as is: validate_cards must fail loudly rather than us inventing a value


def split_ids(ev_list, ev, ctx):
    out = []
    for e in as_list(ev_list, ev, ctx + ".evidence"):
        out += [p for p in re.split(r"[,;\s]+", str(e)) if p]
    return out


def snap_quote(text, start, segs, ev, qid):
    """Return (text, start) that check_quotes will accept, or None."""
    need, tol, maxw = CONFIG["quote_match_min"]["auto"], CONFIG["timestamp_tolerance_s"], CONFIG["quote_max_words"]
    words = len(re.findall(r"\S+", text))
    score, t = locate_quote(text, segs, around=start, search_window=CONFIG["quote_search_window_s"])
    if score >= need and t is not None and abs(t - start) <= tol and words <= maxw and len(text) <= 320:
        return text, start
    score2, t2 = locate_quote(text, segs)
    if score2 > score:
        score, t = score2, t2
    if t is None or score < 0.5:
        ev.add("quote_dropped_nomatch", f"{qid} score={score:.2f}")
        return None
    idx = min(range(len(segs)), key=lambda i: abs(segs[i][0] - t))
    n = min(maxw, max(8, len(tokens(text))))
    pieces, got = [], 0
    for j in range(idx, len(segs)):
        p = segs[j][2].strip()
        if p:
            pieces.append(p); got += len(tokens(p))
        if got >= n:
            break
    snapped = " ".join(" ".join(pieces).split()[:maxw])
    while len(snapped) > 320:
        snapped = snapped.rsplit(" ", 1)[0]
    s_score, s_t = locate_quote(snapped, segs, around=segs[idx][0], search_window=CONFIG["quote_search_window_s"])
    if len(tokens(snapped)) < 3 or s_score < need or abs(s_t - segs[idx][0]) > tol:
        ev.add("quote_dropped_snapfail", qid)
        return None
    ev.add("quote_snapped", f"{qid} score={score:.2f}")
    return snapped, float(segs[idx][0])


def _anchor_stats(anchors, prefix, idmap, evid_by_new, quotes, ev, label):
    """v3 diagnostics only: does each item's evidence sit in its declared 30 s anchor bucket? Never edits the card."""
    qs = {q["id"]: q["start"] for q in quotes}
    anchors = anchors if isinstance(anchors, list) else []
    if not anchors:
        ev.add(f"{label}_anchors_missing"); return
    for a in anchors:
        if not isinstance(a, dict):
            continue
        new = idmap.get(norm_id(a.get("id"), prefix))
        try:
            ab = int(float(a.get("anchor_s")) // check_rerun.BUCKET)
        except (TypeError, ValueError):
            ev.add(f"{label}_anchor_bad", str(a)); continue
        if new is None or new not in evid_by_new:
            continue
        bks = {int(qs[e] // check_rerun.BUCKET) for e in evid_by_new[new] if e in qs}
        if bks == {ab}:
            ev.add(f"{label}_anchor_exact")
        elif ab in bks:
            ev.add(f"{label}_anchor_evidence_spills", f"{new} anchor_bucket={ab} evidence_buckets={sorted(bks)}")
        else:
            ev.add(f"{label}_anchor_miss", f"{new} anchor_bucket={ab} evidence_buckets={sorted(bks)}")


def normalize_card(raw, vid, run_ids, ev):
    raw = raw if isinstance(raw, dict) else {}
    meta = meta_for(vid)
    vtt = SUBS / f"{vid}.en.vtt"
    segs = load_transcript(vtt)
    ends = [float(r.split("\t")[1]) for r in (SUBS / f"{vid}.norm.tsv").read_text().splitlines() if r.strip()]

    cl = raw.get("classification") if isinstance(raw.get("classification"), dict) else {}
    classification = {
        "is_teaching": norm_enum(cl.get("is_teaching"), "is_teaching", ev, "classification.is_teaching", True),
        "stroke": norm_enum(pick(cl, "stroke", ["primary_stroke"], ev, "classification", ""), "stroke", ev, "classification.stroke", True),
        "format": norm_enum(pick(cl, "format", ["style"], ev, "classification", ""), "format", ev, "classification.format", True),
        "reason": str(pick(cl, "reason", ["summary", "rationale"], ev, "classification", "")),
    }
    pr = raw.get("problem") if isinstance(raw.get("problem"), dict) else {}
    problem = {
        "title_as_question": str(pick(pr, "title_as_question", ["question", "title"], ev, "problem", "")),
        "definition": str(pick(pr, "definition", ["summary", "statement", "description"], ev, "problem", "")),
        "node_id": "PENDING",
        "prerequisites": [str(x) for x in as_list(pr.get("prerequisites"), ev, "problem.prerequisites")],
        "target_player": str(pr.get("target_player") or ""),
    }
    if pr.get("node_id") not in (None, "PENDING"):
        ev.add("node_id_forced_pending", "problem")

    # ---- quotes
    qmap, quotes, seen_text = {}, [], {}
    for i, q in enumerate(as_list(raw.get("quotes"), ev, "quotes"), 1):
        if not isinstance(q, dict):
            ev.add("quote_nondict"); continue
        old = norm_id(q.get("id", f"Q{i}"), "Q") or f"Q{i:02d}"
        text = str(pick(q, "text", ["quote", "content"], ev, "quotes", "")).strip()
        st = pick(q, "start", ["start_s", "timestamp_s", "timestamp", "t"], ev, "quotes", 0)
        try:
            st = float(st)
        except (TypeError, ValueError):
            ev.add("quote_bad_start", old); st = 0.0
        if not text:
            ev.add("quote_empty", old); continue
        res = snap_quote(text, st, segs, ev, old)
        if not res:
            continue
        text, st = res
        key = (round(st), " ".join(tokens(text)[:6]))
        if key in seen_text:
            qmap[old] = seen_text[key]; ev.add("quote_dedup", old); continue
        new = f"Q{len(quotes) + 1:02d}"
        seen_text[key] = new; qmap[old] = new
        quotes.append({"id": new, "text": text, "start": st,
                       "speaker": norm_enum(q.get("speaker", "coach"), "speaker", ev, f"quotes.{old}.speaker"),
                       "about": norm_enum(q.get("about", "general"), "about", ev, f"quotes.{old}.about"),
                       "source": "subtitle"})
    for q in quotes:  # leave illegal enum values for the validator, except obvious fallbacks documented here
        if q["speaker"] not in ENUM["speaker"]:
            ev.add("enum_fallback", f"{q['id']}.speaker={q['speaker']}->coach"); q["speaker"] = "coach"
        if q["about"] not in ENUM["about"]:
            ev.add("enum_fallback", f"{q['id']}.about={q['about']}->general"); q["about"] = "general"

    def ev_map(lst, ctx):
        out = []
        for e in split_ids(lst, ev, ctx):
            if e.upper().startswith("BOOK:"):
                out.append(e); continue
            nid = norm_id(e, "Q")
            if nid and nid in qmap:
                if qmap[nid] not in out:
                    out.append(qmap[nid])
                if e != nid:
                    ev.add("evidence_id_padded", ctx)
            else:
                ev.add("evidence_dropped_unknown", f"{ctx}:{e}")
        return out

    # ---- symptoms
    smap, symptoms = {}, []
    for i, s in enumerate(as_list(raw.get("symptoms"), ev, "symptoms"), 1):
        if not isinstance(s, dict):
            continue
        old = norm_id(s.get("id", f"S{i}"), "S") or f"S{i:02d}"
        evid = ev_map(s.get("evidence"), f"symptoms.{old}")
        if not evid:
            ev.add("item_dropped_no_evidence", f"symptom {old}"); continue
        kind = norm_enum(s.get("kind", "visible"), "sym_kind", ev, f"symptoms.{old}.kind")
        layer = str(s.get("layer", "B")).upper()
        camera = s.get("camera", "any")
        obs = str(s.get("obs_ref", "NEW-OBS"))
        phase = s.get("phase", "whole")
        if layer not in ("A", "B", "C"):
            ev.add("enum_fallback", f"symptoms.{old}.layer={layer}->B"); layer = "B"
        if kind == "feel":
            if (layer, camera, obs) != ("C", "none", "NONE"):
                ev.add("layer_rule_fix", f"symptoms.{old} feel -> C/none/NONE")
            layer, camera, obs = "C", "none", "NONE"
        else:
            if kind == "visible" and layer == "C":
                ev.add("layer_rule_fix", f"symptoms.{old} visible layer C->B"); layer = "B"
            if camera not in ("S", "B", "R", "any", "none") or (kind == "visible" and camera == "none"):
                ev.add("layer_rule_fix", f"symptoms.{old} camera {camera}->any"); camera = "any"
            if not re.fullmatch(r"OBS-\d{2}|NEW-OBS|NONE", obs) or (kind == "visible" and obs == "NONE"):
                ev.add("layer_rule_fix", f"symptoms.{old} obs_ref {obs}->NEW-OBS"); obs = "NEW-OBS"
        if phase not in ENUM["phase"]:
            ev.add("enum_fallback", f"symptoms.{old}.phase={phase}->whole"); phase = "whole"
        new = f"S{len(symptoms) + 1:02d}"; smap[old] = new
        symptoms.append({"id": new, "description": str(pick(s, "description", ["summary", "statement", "text"], ev, "symptoms", "")),
                         "kind": kind, "layer": layer, "camera": camera, "phase": phase, "obs_ref": obs, "evidence": evid})

    # ---- causes
    cmap, causes = {}, []
    for i, c in enumerate(as_list(raw.get("causes"), ev, "causes"), 1):
        if not isinstance(c, dict):
            continue
        old = norm_id(c.get("id", f"C{i}"), "C") or f"C{i:02d}"
        evid = ev_map(c.get("evidence"), f"causes.{old}")
        if not evid:
            ev.add("item_dropped_no_evidence", f"cause {old}"); continue
        try:
            depth = max(1, min(5, int(c.get("depth", 1))))
        except (TypeError, ValueError):
            ev.add("enum_fallback", f"causes.{old}.depth->1"); depth = 1
        layer = str(c.get("layer", "B")).upper()
        if layer not in ("A", "B", "C"):
            ev.add("enum_fallback", f"causes.{old}.layer={layer}->B"); layer = "B"
        aop = norm_enum(c.get("active_or_passive", "unclear"), "aop", ev, f"causes.{old}.aop")
        if aop not in ENUM["aop"]:
            ev.add("enum_fallback", f"causes.{old}.aop={aop}->unclear"); aop = "unclear"
        if c.get("node_id") not in (None, "PENDING"):
            ev.add("node_id_forced_pending", f"causes.{old}")
        new = f"C{len(causes) + 1:02d}"; cmap[old] = new
        causes.append({"id": new, "statement": str(pick(c, "statement", ["description", "summary", "text"], ev, "causes", "")),
                       "depth": depth, "layer": layer, "node_id": "PENDING", "active_or_passive": aop, "evidence": evid})
    depth_of = {c["id"]: c["depth"] for c in causes}
    cev = {c["id"]: c["evidence"] for c in causes}
    _anchor_stats(raw.get("cause_anchors"), "C", cmap, {c["id"]: c["evidence"] for c in causes}, quotes, ev, "cause")

    def end_id(x):
        if str(x).strip().upper() == "PROBLEM":
            return "PROBLEM"
        return cmap.get(norm_id(x, "C")) or smap.get(norm_id(x, "S"))

    # ---- chain
    chain = []
    for e in as_list(raw.get("chain"), ev, "chain"):
        if not isinstance(e, dict):
            continue
        fr, to = cmap.get(norm_id(e.get("from"), "C")), end_id(e.get("to"))
        if not fr or not to:
            ev.add("chain_dropped_unknown_endpoint", f"{e.get('from')}->{e.get('to')}"); continue
        if to in depth_of and depth_of[fr] <= depth_of[to]:
            ev.add("chain_depth_retarget_to_PROBLEM", f"{fr}->{to}"); to = "PROBLEM"
        evid = ev_map(e.get("evidence"), "chain")
        if not evid:
            ev.add("chain_evidence_from_cause", f"{fr}->{to}"); evid = cev[fr][:1]
        st = norm_enum(e.get("strength", "coach_implied"), "strength", ev, "chain.strength")
        if st not in ENUM["strength"]:
            ev.add("enum_fallback", f"chain.strength={st}->coach_implied"); st = "coach_implied"
        chain.append({"from": fr, "to": to, "strength": st, "evidence": evid})
    used = {e["from"] for e in chain}
    for c in causes:
        if c["id"] not in used:
            ev.add("chain_added_missing_cause_inferred", c["id"])
            chain.append({"from": c["id"], "to": "PROBLEM", "strength": "inferred", "evidence": c["evidence"][:1]})

    # ---- angles
    angles = []
    for i, a in enumerate(as_list(raw.get("angles"), ev, "angles"), 1):
        if isinstance(a, str):
            ev.add("angle_bare_string"); continue  # no evidence -> cannot keep
        if not isinstance(a, dict):
            continue
        evid = ev_map(a.get("evidence"), f"angles.{i}")
        if not evid:
            ev.add("item_dropped_no_evidence", f"angle {i}"); continue
        typ = norm_enum(a.get("type", "mechanics"), "angle", ev, f"angles.{i}.type")
        if typ not in ENUM["angle"]:
            ev.add("enum_fallback", f"angles.{i}.type={typ}->mechanics"); typ = "mechanics"
        angles.append({"id": f"A{len(angles) + 1:02d}", "type": typ,
                       "summary": str(pick(a, "summary", ["statement", "description", "text"], ev, "angles", "")), "evidence": evid})

    # ---- solutions
    sols, dmap = [], {}
    for i, d in enumerate(as_list(raw.get("solutions"), ev, "solutions"), 1):
        if not isinstance(d, dict):
            continue
        evid = ev_map(d.get("evidence"), f"solutions.{i}")
        if not evid:
            ev.add("item_dropped_no_evidence", f"solution {i}"); continue
        kind = pick(d, "kind", ["type"], ev, "solutions", "drill")
        kind = norm_enum(kind, "sol_kind", ev, f"solutions.{i}.kind")
        if kind not in ENUM["sol_kind"]:
            ev.add("enum_fallback", f"solutions.{i}.kind={kind}->drill"); kind = "drill"
        targets = []
        for t in split_ids(d.get("targets"), ev, f"solutions.{i}.targets"):
            tt = "PROBLEM" if t.strip().upper() == "PROBLEM" else (cmap.get(norm_id(t, "C")) or smap.get(norm_id(t, "S")))
            if tt and tt not in targets:
                targets.append(tt)
            elif not tt:
                ev.add("target_dropped_unknown", f"solutions.{i}:{t}")
        if not targets:
            ev.add("target_default_PROBLEM", f"solutions.{i}"); targets = ["PROBLEM"]
        summary = str(pick(d, "summary", ["description", "name", "text"], ev, "solutions", ""))
        goal = str(d.get("external_goal") or "")
        sc = norm_enum(d.get("scenario_fit", "machine_low"), "scenario", ev, f"solutions.{i}.scenario_fit")
        if sc not in ENUM["scenario"]:
            ev.add("enum_fallback", f"solutions.{i}.scenario_fit={sc}->machine_low"); sc = "machine_low"
        bc = d.get("ban_conflict") if isinstance(d.get("ban_conflict"), dict) else {}
        terms = [t for t in as_list(bc.get("terms"), ev, "ban.terms") if t in BANS]
        for h in ban_hits(summary + " " + goal):
            if h not in terms:
                ev.add("ban_rescan_added", f"solutions.{i}:{h}"); terms.append(h)
        flag = bool(bc.get("flag")) or bool(terms)
        ifok = bool(d.get("intuition_first_ok", False))
        if flag and ifok:
            ev.add("intuition_first_forced_false", f"solutions.{i} ban-flagged"); ifok = False
        if ifok and not goal:
            ev.add("intuition_first_forced_false", f"solutions.{i} no external_goal"); ifok = False
        dmap[norm_id(d.get("id", f"D{i}"), "D") or f"D{i:02d}"] = f"D{len(sols) + 1:02d}"
        sols.append({"id": f"D{len(sols) + 1:02d}", "kind": kind, "targets": targets, "summary": summary, "external_goal": goal,
                     "scenario_fit": sc, "ban_conflict": {"flag": flag, "terms": terms, "note": str(bc.get("note") or "")},
                     "intuition_first_ok": ifok, "evidence": evid})

    _anchor_stats(raw.get("solution_anchors"), "D", dmap,
                  {d["id"]: d["evidence"] for d in sols}, quotes, ev, "solution")
    pev = ev_map(pr.get("evidence"), "problem")
    if not pev and quotes:
        ev.add("problem_evidence_fallback_Q01"); pev = [quotes[0]["id"]]
    problem["evidence"] = pev

    return {
        "schema_version": "2.0", "video": meta, "classification": classification,
        "transcript": {"path": str(vtt), "kind": "auto", "sha256": sha256_file(vtt),
                       "coverage_ratio": round(min(1.0, max(ends) / meta["duration_s"]), 4) if ends else 0.0},
        "extraction": {"stage1_model": MODEL, "stage1_prompt_version": PROMPT_VERSION, "stage1_run_ids": run_ids,
                       "user_context_in_prompt": False},
        "problem": problem, "quotes": quotes, "symptoms": symptoms, "causes": causes, "chain": chain, "angles": angles,
        "solutions": sols, "video_layer": [], "conflicts": [], "status": "stage1_draft",
    }


# ------------------------------------------------------------------ commands
def cmd_run(vid, n):
    key = os.environ["MICU_GEMINI_KEY"]
    raw_dir = EP / "raw" / vid
    wdir = WORK / vid
    raw_dir.mkdir(parents=True, exist_ok=True); wdir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(vid)
    assert "屈书毅" not in prompt and "用户" not in prompt
    (wdir / f"s1_run{n}.prompt.txt").write_text(prompt, encoding="utf-8")
    attempts, card, ev = [], None, Ev()
    for k in range(3):  # 1 call + up to 2 retries on transport/parse failure
        text, log = call_gemini(prompt, TEMPS[n], key)
        log["response_chars"] = len(text)
        (wdir / f"s1_run{n}.attempt{k + 1}.raw.txt").write_text(text, encoding="utf-8")
        if log["status"] == "ok" and text:
            try:
                raw = parse_json(text)
                card = normalize_card(raw, vid, [f"s1_r{n}_{vid[:6]}", "s1_pair_pending"], ev)
                attempts.append(log); break
            except Exception as e:  # noqa: BLE001
                log["status"] = f"parse/normalize: {type(e).__name__}: {e}"
        attempts.append(log)
        time.sleep(3)
    tot = Counter()
    for a in attempts:
        for kk in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount", "thoughtsTokenCount", "cachedContentTokenCount"):
            tot[kk] += (a.get("usage") or {}).get(kk, 0)
    out = {"id": vid, "run": n, "model": MODEL, "prompt_version": PROMPT_VERSION, "temperature": TEMPS[n],
           "ts": time.strftime("%Y-%m-%d %H:%M:%S %z"), "calls": len(attempts), "attempts": attempts, "usage_total": dict(tot),
           "ok": card is not None}
    s = json.dumps(out, ensure_ascii=False, indent=2)
    assert key not in s
    (raw_dir / f"s1_run{n}.log.json").write_text(s, encoding="utf-8")
    if card:
        (raw_dir / f"s1_run{n}.card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        (raw_dir / f"s1_run{n}.norm_events.json").write_text(json.dumps({"counts": dict(ev.c), "detail": ev.detail}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{vid} run{n}] ok={out['ok']} calls={len(attempts)} quotes={len(card['quotes']) if card else '-'} events={dict(ev.c)}")
    return 0 if card else 1


def cmd_final(vid):
    raw_dir = EP / "raw" / vid
    runs = sorted(int(p.name.split(".")[0][6:]) for p in raw_dir.glob("s1_run*.card.json"))
    cards = {n: load_json(raw_dir / f"s1_run{n}.card.json") for n in runs}
    res = {"id": vid, "runs": runs, "pairs": {}}
    chosen = None
    for a, b in ((1, 2), (1, 3), (2, 3)):
        if a in cards and b in cards:
            f = check_rerun.cards(cards[a], cards[b])
            ok = not any(x.level == "FAIL" for x in f)
            res["pairs"][f"{a}-{b}"] = {"pass": ok, "findings": [str(x) for x in f]}
            if ok and chosen is None:
                chosen = (a, b)
        if chosen:
            break
        if (a, b) == (1, 2) and 3 not in cards:
            break  # 3rd run not executed yet; caller decides
    res["chosen_pair"] = list(chosen) if chosen else None
    if chosen:
        final = copy.deepcopy(cards[chosen[0]])
        final["extraction"]["stage1_run_ids"] = [f"s1_r{n}_{vid[:6]}" for n in runs]
        final["status"] = "stage1_verified"
        (EP / f"{vid}.card.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "rerun_adjudication.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0 if chosen else 1


# ------------------------------------------------------------------ 3-run consensus adjudication
def _bk(card, quote_ids):
    qt = {q["id"]: q["start"] for q in card["quotes"]}
    return {int(qt[e] // check_rerun.BUCKET) for e in quote_ids if e in qt}


def _scores(a, b):
    ja = check_rerun._jac(check_rerun._buckets(a, a["causes"]), check_rerun._buckets(b, b["causes"]))
    jd = check_rerun._jac({(d["kind"], k) for d in a["solutions"] for k in check_rerun._buckets(a, [d])},
                          {(d["kind"], k) for d in b["solutions"] for k in check_rerun._buckets(b, [d])})
    ea = {(min(check_rerun._buckets(a, [c])) if check_rerun._buckets(a, [c]) else -1) for c in a["causes"]}
    eb = {(min(check_rerun._buckets(b, [c])) if check_rerun._buckets(b, [c]) else -1) for c in b["causes"]}
    return ja, check_rerun._jac(ea, eb), jd


def consensus_card(cards, base_n):
    """Majority adjudication over >=3 runs. Only REMOVES unsupported evidence/items and votes the enum `kind`;
    never writes new text. Base = run closest to the others."""
    base = copy.deepcopy(cards[base_n])
    others = [c for n, c in cards.items() if n != base_n]
    log = {"base_run": base_n, "dropped_causes": [], "evidence_pruned": 0, "dropped_solutions": [], "kind_votes_changed": []}
    # other runs' cause-bucket union
    cause_sup = set().union(*[check_rerun._buckets(o, o["causes"]) for o in others])
    keep, cmap = [], {}
    for c in base["causes"]:
        ev = [e for e in c["evidence"] if _bk(base, [e]) and next(iter(_bk(base, [e]))) in cause_sup or e.startswith("BOOK:")]
        if len(ev) != len(c["evidence"]):
            log["evidence_pruned"] += len(c["evidence"]) - len(ev)
        if not ev:
            log["dropped_causes"].append(c["id"]); continue
        c["evidence"] = ev
        keep.append(c)
    cmap = {c["id"]: f"C{i:02d}" for i, c in enumerate(keep, 1)}
    for c in keep:
        c["id"] = cmap[c["id"]]
    base["causes"] = keep
    depth = {c["id"]: c["depth"] for c in keep}
    smap_ok = {s["id"] for s in base["symptoms"]}
    chain = []
    for e in base["chain"]:
        if e["from"] not in cmap:
            continue
        to = e["to"]
        if to.startswith("C"):
            to = cmap.get(to, "PROBLEM")
        e = dict(e, **{"from": cmap[e["from"]], "to": to})
        if to in depth and depth[e["from"]] <= depth[to]:
            e["to"] = "PROBLEM"
        chain.append(e)
    used = {e["from"] for e in chain}
    for c in keep:
        if c["id"] not in used:
            chain.append({"from": c["id"], "to": "PROBLEM", "strength": "inferred", "evidence": c["evidence"][:1]})
    base["chain"] = chain
    # solutions
    sols = []
    for d in base["solutions"]:
        d = copy.deepcopy(d)
        d["targets"] = [cmap.get(t, t) if t.startswith("C") else t for t in d["targets"]]
        d["targets"] = [t for t in d["targets"] if not t.startswith("C") or t in depth] or ["PROBLEM"]
        sup = set()
        votes = Counter()
        for o in others:
            for od in o["solutions"]:
                ob = check_rerun._buckets(o, [od])
                if ob & check_rerun._buckets(base, [d]):
                    sup |= ob & check_rerun._buckets(base, [d])
                    votes[od["kind"]] += 1
        ev = [e for e in d["evidence"] if next(iter(_bk(base, [e])), None) in sup]
        if not ev:
            log["dropped_solutions"].append(d["id"]); continue
        d["evidence"] = ev
        votes[d["kind"]] += 1
        top = max(votes.values())
        winners = [k for k, v in votes.items() if v == top]
        if d["kind"] not in winners:
            log["kind_votes_changed"].append(f"{d['id']}:{d['kind']}->{winners[0]}")
            d["kind"] = winners[0]
        sols.append(d)
    for i, d in enumerate(sols, 1):
        d["id"] = f"D{i:02d}"
    base["solutions"] = sols
    base["conflicts"] = [{"id": "X01", "between": [f"s1_r{n}_{base['video']['id'][:6]}" for n in sorted(cards)],
                          "description": "No raw run pair met check_rerun thresholds after 3 runs; final card = closest run adjudicated by 3-run majority (unsupported cause/solution evidence removed, solution kind voted); no text rewritten. See raw/<id>/rerun_adjudication.json.",
                          "resolution": "open_user_decides"}]
    return base, log


def cmd_consensus(vid):
    raw_dir = EP / "raw" / vid
    cards = {n: load_json(raw_dir / f"s1_run{n}.card.json") for n in (1, 2, 3)}
    sc = {n: sum(sum(_scores(cards[n], cards[m])) for m in cards if m != n) for n in cards}
    base_n = max(sc, key=lambda n: (sc[n], -n))
    final, log = consensus_card(cards, base_n)
    res = load_json(raw_dir / "rerun_adjudication.json")
    res["consensus"] = {"closeness": {str(k): round(v, 3) for k, v in sc.items()}, **log, "pairs_vs_final": {}}
    ok_any = False
    for n, c in cards.items():
        if n == base_n:
            continue
        f = check_rerun.cards(final, c)
        ok = not any(x.level == "FAIL" for x in f)
        ok_any |= ok
        res["consensus"]["pairs_vs_final"][f"final-r{n}"] = {"pass": ok, "findings": [str(x) for x in f]}
    res["consensus"]["pass"] = ok_any
    final["extraction"]["stage1_run_ids"] = [f"s1_r{n}_{vid[:6]}" for n in sorted(cards)]
    if ok_any:
        final["status"] = "stage1_verified"
        (EP / f"{vid}.card.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        res["chosen_pair"] = ["consensus", f"r{base_n}"]
    else:  # keep the best-effort card as a draft; it is NOT verified and must not enter the tree
        final["status"] = "stage1_draft"
        final["conflicts"][0]["description"] = ("check_rerun FAILED on every run pair and on the 3-run consensus card; kept as stage1_draft, "
                                                "not verified. See raw/<id>/rerun_adjudication.json.")
        (EP / f"{vid}.card.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "rerun_adjudication.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res["consensus"], ensure_ascii=False))
    return 0 if ok_any else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "run":
        sys.exit(cmd_run(a[1], int(a[2])))
    if len(a) == 2 and a[0] == "final":
        sys.exit(cmd_final(a[1]))
    if len(a) == 2 and a[0] == "consensus":
        sys.exit(cmd_consensus(a[1]))
    print(__doc__); sys.exit(2)

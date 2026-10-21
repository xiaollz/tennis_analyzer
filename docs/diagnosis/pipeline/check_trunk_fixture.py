#!/usr/bin/env python3
"""Loop gate: compare compose_trunk_report output against golden expectations.

usage:
  python3 check_trunk_fixture.py --report <trunk_report.json> \\
    --golden ../user_case/rear_20261019_golden.json \\
    --facts  ../user_case/rear_20261019_facts_frozen.json \\
    [--live <new_facts.json>] \\
    [--selftest] \\
    --out ../loop/reports/trunk_gate_rear_20261019_<tag>.json

Exit 1 on any FAIL. See TRUNK_REPORT_CONTRACT.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

from common import CONFIG, load_json

BAN = CONFIG["ban_patterns"]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_path(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


def _scan_ban(text: str) -> list[str]:
    hits = []
    for label, pats in BAN.items():
        for p in pats:
            if re.search(p, text, flags=re.IGNORECASE):
                hits.append(label)
                break
    return hits


def _branch_map(report: dict) -> dict:
    return {b["id"]: b for b in report.get("branches", [])}


def check(report: dict, golden: dict, facts_doc: dict, facts_path: Path | None,
          live_facts: dict | None = None) -> dict:
    checks = []
    drifts = []

    def add(kid, ok, detail=""):
        checks.append({"id": kid, "result": "PASS" if ok else "FAIL", "detail": detail})

    # K01 trunk
    add("K01", report.get("trunk", {}).get("id") == golden["trunk_id"],
        f"got={report.get('trunk',{}).get('id')} want={golden['trunk_id']}")

    # K02 branch set exact
    got_ids = [b["id"] for b in report.get("branches", [])]
    want_ids = golden["branch_ids_exact"]
    add("K02", got_ids == want_ids, f"got={got_ids} want={want_ids}")

    bm = _branch_map(report)
    g_branches = golden["branches"]

    # K03 B1 has_facts + evidence
    b1_id = "B1_HIP_TRANSFER_ARM_COMPENSATION"
    b1 = bm.get(b1_id)
    g1 = g_branches[b1_id]
    if not b1:
        add("K03", False, "B1 missing")
    else:
        evid = []
        for L in b1.get("leaves", []):
            if L.get("state") == "fact" and L.get("obs") and L.get("value"):
                evid.append(f"{L['obs']}={L['value']}")
        need = g1["evidence_must_include"]
        ok = b1.get("status") == g1["status"] and all(e in evid for e in need)
        add("K03", ok, f"status={b1.get('status')} evid={evid} need={need}")

    # K04 B2 present + pending_self_report
    b2_id = "B2_SWING_START_TOO_EARLY"
    b2 = bm.get(b2_id)
    g2 = g_branches[b2_id]
    if not b2:
        add("K04", False, "B2 silently missing")
    else:
        add("K04", b2.get("status") == g2["status"],
            f"status={b2.get('status')} want={g2['status']}")

    # K05 no forged timing facts
    if not b2:
        add("K05", False, "B2 missing")
    else:
        forged = []
        ban_obs = set(g2["must_not_have_fact_values_for"])
        for L in b2.get("leaves", []):
            if L.get("role") == "evidence" and L.get("state") == "fact" and L.get("obs") in ban_obs:
                forged.append(L["obs"])
        add("K05", not forged, f"forged={forged}")

    # K06 missing_facts listed with camera S
    if not b2:
        add("K06", False, "B2 missing")
    else:
        mf_obs = {m.get("obs") for m in b2.get("missing_facts", [])}
        need = set(g2["missing_facts_must_include"])
        cams = {m.get("camera") for m in b2.get("missing_facts", []) if m.get("obs") in need}
        ok = need.issubset(mf_obs) and (not cams or cams == {g2["missing_camera"]})
        add("K06", ok, f"mf={sorted(mf_obs)} cams={cams}")

    # K07 hierarchy: never_top_level; feet upstream; finish downstream
    top_bad = []
    for nid in golden["never_top_level"]:
        if nid in got_ids:
            top_bad.append(nid)
        # also fail if a leaf id somehow appears as branch id (already covered)
    feet_ok = False
    finish_ok = False
    if b1:
        for L in b1.get("leaves", []):
            if L.get("id") == "L1_feet_planted" and L.get("role") == "upstream_condition" and L.get("state") in ("legacy", "fact", "self_report"):
                feet_ok = True
            if L.get("id") == "L1_finish_wraps" and L.get("role") == "downstream" and L.get("state") in ("legacy", "fact", "self_report"):
                finish_ok = True
    add("K07", not top_bad and feet_ok and finish_ok,
        f"top_bad={top_bad} feet_ok={feet_ok} finish_ok={finish_ok}")

    # K08 retracted
    ret_keys = {r.get("key") for r in report.get("retracted", [])}
    need_ret = set(golden["retracted_must_include"])
    # keys must not appear in any leaf/evidence
    leaked = []
    for b in report.get("branches", []):
        for L in b.get("leaves", []):
            blob = json.dumps(L, ensure_ascii=False)
            for k in need_ret:
                if k in blob and L.get("id") != k:
                    # only flag if the retracted key is used as evidence content improperly
                    if L.get("state") == "fact" and k in (L.get("value"), L.get("obs"), L.get("id")):
                        leaked.append(k)
    # also: contact_late must not be a leaf id with fact state
    for b in report.get("branches", []):
        for L in b.get("leaves", []):
            if L.get("id") in need_ret or L.get("legacy_key") in need_ret:
                leaked.append(L.get("id") or L.get("legacy_key"))
    add("K08", need_ret.issubset(ret_keys) and not leaked,
        f"ret={sorted(ret_keys)} leaked={leaked}")

    # K09 not_visible OBS-40/19/71
    if not b1:
        add("K09", False, "B1 missing")
    else:
        bad = []
        for L in b1.get("leaves", []):
            if L.get("obs") in ("OBS-40", "OBS-19", "OBS-71"):
                if L.get("state") != "not_visible":
                    bad.append(f"{L['obs']}:{L.get('state')}")
                if L.get("state") == "fact":
                    bad.append(f"{L['obs']}:used_as_fact")
        # golden requires them in not_visible_must_include
        have = {L.get("obs") for L in b1.get("leaves", []) if L.get("state") == "not_visible"}
        need_nv = set(g1["not_visible_must_include"])
        add("K09", not bad and need_nv.issubset(have), f"bad={bad} have_nv={sorted(have)}")

    # K10 self_report leaves not fact without qualifying input
    if not b1:
        add("K10", False, "B1 missing")
    else:
        bad = []
        qobs = {f["obs"]: f for f in facts_doc.get("facts", [])
                if f.get("source") in ("model", "pose", "manual")
                and f.get("value") != "unclear"}
        for lid in g1.get("self_report_leaves", []):
            for L in b1.get("leaves", []):
                if L.get("id") == lid and L.get("state") == "fact":
                    # only OK if there is a real qualifying fact for that obs
                    obs = L.get("obs")
                    if not obs or obs not in qobs or str(obs).startswith("NEW-"):
                        bad.append(lid)
        add("K10", not bad, f"bad={bad}")

    # K11 no invented facts
    invented = []
    fact_index = {(f["obs"], f["value"], f.get("agree")): f for f in facts_doc.get("facts", [])}
    for b in report.get("branches", []):
        for L in b.get("leaves", []):
            if L.get("state") != "fact":
                continue
            key = (L.get("obs"), L.get("value"), L.get("agree"))
            if key not in fact_index:
                # also allow match without agree strictness if agree missing? contract says same agree
                invented.append(f"{L.get('obs')}={L.get('value')}/{L.get('agree')}")
    add("K11", not invented, f"invented={invented}")

    # K12 confidence / validation
    bad_c = []
    for b in report.get("branches", []):
        if b.get("confidence") == "high":
            bad_c.append(f"{b['id']}:high")
        if golden["global"].get("v5_passed") is False and b.get("validation_label") != golden["global"]["validation_label"]:
            bad_c.append(f"{b['id']}:label={b.get('validation_label')}")
    add("K12", not bad_c, f"bad={bad_c}")

    # K13 next_one_thing
    not_ = report.get("next_one_thing") or {}
    zh = not_.get("zh") or ""
    breath = not_.get("breath") or ""
    count_ok = bool(zh) and golden["global"]["next_one_thing_count"] == 1
    breath_ok = (not golden["global"]["breath_reminder_required"]) or bool(breath.strip())
    add("K13", count_ok and breath_ok, f"zh={bool(zh)} breath={bool(breath)}")

    # K14 ban terms in prescription
    ban_blob = " ".join([zh, breath, not_.get("protocol", ""), " ".join(not_.get("verify", []))])
    hits = _scan_ban(ban_blob)
    add("K14", not hits, f"hits={hits}")

    # K15 holdout: if tree writeback fields exist they must not cite holdout; knowledge OK with label
    holdout_ids = {"ui-RFRdQKRk", "5jHCDc44SQM"}
    bad_h = []
    for b in report.get("branches", []):
        for field in ("tree_evidence", "tree_writeback", "evidence_for_tree"):
            if field in b:
                blob = json.dumps(b[field], ensure_ascii=False)
                for h in holdout_ids:
                    if h in blob:
                        bad_h.append(f"{b['id']}.{field}:{h}")
        for hk in b.get("holdout_knowledge", []):
            if isinstance(hk, dict) and hk.get("label") != "holdout":
                bad_h.append(f"{b['id']}:holdout_unlabeled:{hk.get('id')}")
    add("K15", not bad_h, f"bad={bad_h}")

    # K16 determinism — checked externally by re-running compose; here we mark PASS if meta has hashes
    # The selftest / caller should verify byte identity. We record PASS placeholder when hashes present.
    meta = report.get("meta") or {}
    add("K16", bool(meta.get("branches_file_sha256")) and bool(meta.get("facts_file_sha256")),
        "determinism hashes present (caller may re-run compose for byte check)")

    # K17 facts file not mutated vs meta
    if facts_path is not None:
        cur = _sha256_path(facts_path)
        want = meta.get("facts_file_sha256")
        add("K17", cur == want, f"cur={cur[:12]}… meta={str(want)[:12]}…")
    else:
        add("K17", False, "facts_path missing")

    # live drift (not FAIL)
    if live_facts is not None:
        old = {f["obs"]: f for f in facts_doc.get("facts", [])}
        new = {f["obs"]: f for f in live_facts.get("facts", [])}
        for obs, nf in new.items():
            of = old.get(obs)
            if of and (of.get("value") != nf.get("value") or of.get("agree") != nf.get("agree")):
                drifts.append({
                    "obs": obs,
                    "from": f"{of.get('value')} ({of.get('agree')})",
                    "to": f"{nf.get('value')} ({nf.get('agree')})",
                })

    failed = [c for c in checks if c["result"] == "FAIL"]
    return {
        "pass": not failed,
        "checks": checks,
        "fail_count": len(failed),
        "drifts": drifts,
    }


def run_selftest(compose_mod, branches_path: Path, facts_path: Path, golden: dict) -> dict:
    """10 mutation cases that must all be caught."""
    import compose_trunk_report as ctr

    facts = load_json(facts_path)
    branches = load_json(branches_path)
    base = ctr.compose(branches, facts)
    base["meta"]["branches_file_sha256"] = _sha256_path(branches_path)
    base["meta"]["facts_file_sha256"] = _sha256_path(facts_path)

    results = []

    def expect_fail(name, mutated_report, mutated_facts=None, expect_ids=None):
        r = check(mutated_report, golden, mutated_facts or facts, facts_path)
        failed_ids = {c["id"] for c in r["checks"] if c["result"] == "FAIL"}
        ok = bool(failed_ids) and (expect_ids is None or bool(set(expect_ids) & failed_ids))
        results.append({"case": name, "caught": ok, "failed_ids": sorted(failed_ids)})

    # 1 drop B2
    m = copy.deepcopy(base)
    m["branches"] = [b for b in m["branches"] if b["id"] != "B2_SWING_START_TOO_EARLY"]
    expect_fail("drop_B2", m, expect_ids=["K02", "K04"])

    # 2 forge B2 timing fact
    m = copy.deepcopy(base)
    for b in m["branches"]:
        if b["id"] == "B2_SWING_START_TOO_EARLY":
            for L in b["leaves"]:
                if L.get("obs") == "NEW-SWING_START_VS_BOUNCE":
                    L["state"] = "fact"
                    L["value"] = "still_rising"
                    L["agree"] = "3/3"
            b["status"] = "has_facts"
            b["missing_facts"] = []
    expect_fail("forge_B2_timing", m, expect_ids=["K05", "K11"])

    # 3 feet as top-level branch
    m = copy.deepcopy(base)
    m["branches"].append({"id": "L1_feet_planted", "status": "has_facts", "confidence": "medium",
                          "validation_label": "待验证", "leaves": []})
    expect_fail("feet_top_level", m, expect_ids=["K02", "K07"])

    # 4 drop contact_late from retracted
    m = copy.deepcopy(base)
    m["retracted"] = [r for r in m["retracted"] if r["key"] != "contact_late"]
    expect_fail("drop_retract_contact_late", m, expect_ids=["K08"])

    # 5 hang contact_late as B2 leaf
    m = copy.deepcopy(base)
    for b in m["branches"]:
        if b["id"] == "B2_SWING_START_TOO_EARLY":
            b["leaves"].append({"id": "contact_late", "role": "evidence", "state": "fact",
                                "obs": "OBS-03", "value": "late", "agree": "3/3"})
    expect_fail("contact_late_as_leaf", m, expect_ids=["K08"])

    # 6 OBS-40 as evidence fact
    m = copy.deepcopy(base)
    for b in m["branches"]:
        if b["id"] == "B1_HIP_TRANSFER_ARM_COMPENSATION":
            for L in b["leaves"]:
                if L.get("obs") == "OBS-40":
                    L["state"] = "fact"
                    L["value"] = "racket_arm_behind_torso_line"
                    L["agree"] = "3/3"
                    L["role"] = "evidence"
    expect_fail("OBS40_as_fact", m, expect_ids=["K09"])

    # 7 forearm_lift as fact
    m = copy.deepcopy(base)
    for b in m["branches"]:
        if b["id"] == "B1_HIP_TRANSFER_ARM_COMPENSATION":
            for L in b["leaves"]:
                if L.get("id") == "L1_forearm_lift":
                    L["state"] = "fact"
                    L["value"] = "hand_rises_above_elbow"
                    L["agree"] = "3/3"
    expect_fail("forearm_as_fact", m, expect_ids=["K10"])

    # 8 B1 confidence high
    m = copy.deepcopy(base)
    for b in m["branches"]:
        if b["id"] == "B1_HIP_TRANSFER_ARM_COMPENSATION":
            b["confidence"] = "high"
    expect_fail("B1_high_conf", m, expect_ids=["K12"])

    # 9 ban term in next_one_thing
    m = copy.deepcopy(base)
    m["next_one_thing"]["zh"] = "触球时伸肘把拍送出去"
    expect_fail("ban_伸肘", m, expect_ids=["K14"])

    # 10 facts sha mismatch
    m = copy.deepcopy(base)
    m["meta"]["facts_file_sha256"] = "0" * 64
    expect_fail("facts_sha_mismatch", m, expect_ids=["K17"])

    all_caught = all(r["caught"] for r in results)
    return {"pass": all_caught, "cases": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--live", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--branches", default="")  # for selftest compose
    a = ap.parse_args()

    golden = load_json(a.golden)
    facts_path = Path(a.facts)
    facts = load_json(facts_path)
    live = load_json(a.live) if a.live else None

    out = {"selftest": None, "gate": None}

    if a.selftest:
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        branches = a.branches or str(here.parent / "ftt_tree" / "trunk_branches.json")
        out["selftest"] = run_selftest(None, Path(branches), facts_path, golden)
        print(f"selftest: {'PASS' if out['selftest']['pass'] else 'FAIL'}")
        for c in out["selftest"]["cases"]:
            print(f"  {'OK' if c['caught'] else 'MISS'} {c['case']} -> {c['failed_ids']}")

    if a.report:
        report = load_json(a.report)
        out["gate"] = check(report, golden, facts, facts_path, live)
        print(f"gate: {'PASS' if out['gate']['pass'] else 'FAIL'} ({out['gate']['fail_count']} fails)")
        for c in out["gate"]["checks"]:
            print(f"  {c['result']} {c['id']}: {c['detail']}")
        if out["gate"]["drifts"]:
            print("DRIFT:")
            for d in out["gate"]["drifts"]:
                print(f"  {d['obs']}: {d['from']} → {d['to']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")

    failed = False
    if out.get("selftest") is not None and not out["selftest"]["pass"]:
        failed = True
    if out.get("gate") is not None and not out["gate"]["pass"]:
        failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""Shared helpers for the tree v2 scripts (OBS binding, mechanism merge, v2 build). stdlib only; no secrets.

Model calls go through blind_replay_text.gcall (same endpoint/model as the other pipeline scripts). The key is read
from env MICU_GEMINI_KEY by s1_extract and is never written. Raw prompts/responses go to WORK (outside the repo).
"""
import hashlib
import json
import re
from pathlib import Path

from common import DIAG_DIR, load_json

import os
WORK = Path(os.environ.get("TV_WORK", "/workspace/s1_work/tree_v2"))  # default = v2 work dir; v3 uses TV_WORK=/workspace/s1_work/tree_v3
TREE_V1 = Path(os.environ["TV_TREE_V1"]) if os.environ.get("TV_TREE_V1") else DIAG_DIR / "ftt_tree" / "tree.json"  # skeleton the binding/merge run on (v5: tree_base_v5.json)
EP_DIR = DIAG_DIR / "episodes"
PIPE = Path(__file__).resolve().parent
RUN_TEMPS = {1: 0.2, 2: 0.4}  # two independent runs per task; only what both runs agree on is used as 'required'/'grouped'


def up(ep):
    return re.sub(r"[^A-Z0-9]", "_", ep.upper())


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_catalog():
    return load_json(PIPE / "obs_catalog.json")["observables"]


def load_cards(episodes):
    return {ep: load_json(EP_DIR / f"{ep}.card.json") for ep in episodes}


def cause_item_map(tree, cards):
    """(ep, cause id) -> v1 node id, rebuilt from node id naming (N-C_<UP(ep)>_<Cnn> is the cluster representative) and
    the build log's merged clusters. Returns also node -> [(ep, cid)] members."""
    log = load_json(DIAG_DIR / "ftt_tree" / "build_log.json")
    members = {}
    for ep, c in cards.items():
        for x in c["causes"]:
            members.setdefault(f"N-C_{up(ep)}_{x['id']}", []).append((ep, x["id"]))
    for cl in log["merged_clusters"]:
        for m in cl["members"]:
            ep, cid = m.split(":")
            members.pop(f"N-C_{up(ep)}_{cid}", None)  # non-representative members are not nodes of their own
        members[cl["node"]] = [tuple(m.split(":")) for m in cl["members"]]
    node_of = {}
    for nid, ms in members.items():
        for m in ms:
            node_of[m] = nid
    return node_of, members

"""Shared helpers for the FTT diagnosis verify pipeline (stdlib only).

Every check script returns a list of Finding(level, where, msg); level is
"FAIL" or "WARN". A check passes when it has zero FAIL findings.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_DIR = HERE / "schema"
DIAG_DIR = HERE.parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


@dataclass
class Finding:
    level: str
    where: str
    msg: str

    def __str__(self):
        return f"[{self.level}] {self.where}: {self.msg}"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_schema(name):
    return load_json(SCHEMA_DIR / name)


def fails(findings):
    return [f for f in findings if f.level == "FAIL"]


# ---------------------------------------------------------------- schema
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "integer": int, "number": (int, float)}


def validate_schema(data, schema, root=None, path="$"):
    """Subset of JSON Schema draft-07 used by our schema files."""
    root = root or schema
    out = []
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return validate_schema(data, root["definitions"][ref], root, path)
    t = schema.get("type")
    if t:
        py = _TYPES[t]
        ok = isinstance(data, py) and not (t in ("integer", "number") and isinstance(data, bool))
        if not ok:
            return [Finding("FAIL", path, f"expected {t}, got {type(data).__name__}")]
    if "enum" in schema and data not in schema["enum"]:
        out.append(Finding("FAIL", path, f"{data!r} not in {schema['enum']}"))
    if isinstance(data, str):
        if "pattern" in schema and not re.search(schema["pattern"], data):
            out.append(Finding("FAIL", path, f"{data!r} does not match {schema['pattern']}"))
        if len(data) < schema.get("minLength", 0):
            out.append(Finding("FAIL", path, "string too short"))
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            out.append(Finding("FAIL", path, f"string longer than {schema['maxLength']}"))
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            out.append(Finding("FAIL", path, f"{data} < {schema['minimum']}"))
        if "maximum" in schema and data > schema["maximum"]:
            out.append(Finding("FAIL", path, f"{data} > {schema['maximum']}"))
    if isinstance(data, list):
        if len(data) < schema.get("minItems", 0):
            out.append(Finding("FAIL", path, f"needs >= {schema['minItems']} items"))
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            out.append(Finding("FAIL", path, f"needs <= {schema['maxItems']} items"))
        for i, item in enumerate(data):
            if "items" in schema:
                out += validate_schema(item, schema["items"], root, f"{path}[{i}]")
    if isinstance(data, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in data:
                out.append(Finding("FAIL", path, f"missing required '{k}'"))
        for k, v in data.items():
            if k in props:
                out += validate_schema(v, props[k], root, f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                out.append(Finding("FAIL", path, f"unknown field '{k}'"))
    return out


# ---------------------------------------------------------------- transcripts
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ts(s):
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


_TAG = re.compile(r"<[^>]+>")


def load_transcript(path):
    """Return [(start, end, text)] from .vtt / .srt / .json (whisper or list) / .json3."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    segs = []
    if p.suffix in (".vtt", ".srt"):
        block_t = None
        buf = []
        for line in raw.splitlines() + [""]:
            if "-->" in line:
                a, b = line.split("-->")[:2]
                block_t = (_ts(a), _ts(b.split()[0]))
                buf = []
            elif line.strip() == "":
                if block_t and buf:
                    segs.append((block_t[0], block_t[1], " ".join(buf)))
                block_t, buf = None, []
            elif block_t:
                buf.append(_TAG.sub("", line).strip())
    else:
        data = json.loads(raw)
        if isinstance(data, dict) and "events" in data:  # YouTube json3
            for ev in data["events"]:
                txt = "".join(s.get("utf8", "") for s in ev.get("segs", []) or [])
                if txt.strip():
                    st = ev.get("tStartMs", 0) / 1000
                    segs.append((st, st + ev.get("dDurationMs", 0) / 1000, txt))
        else:
            items = data["segments"] if isinstance(data, dict) else data
            for s in items:
                segs.append((float(s["start"]), float(s.get("end", s["start"])), s["text"]))
    # auto-subs repeat rolling lines; drop exact consecutive duplicates
    dedup = []
    for s in segs:
        if not dedup or s[2].strip() != dedup[-1][2].strip():
            dedup.append(s)
    return dedup


def tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def locate_quote(quote, segs, around=None, search_window=None):
    """Best (score, seg_start) for quote in transcript. score = LCS(tokens)/len(quote tokens).

    Windows are built from consecutive segments spanning ~ (quote words * 0.5 s + 12 s).
    If `around` is given, only anchors within +-search_window seconds are tried.
    """
    q = tokens(quote)
    if not q:
        return 0.0, None
    span = len(q) * 0.5 + 12
    best = (0.0, None)
    for i, (st, _, _) in enumerate(segs):
        if around is not None and abs(st - around) > search_window:
            continue
        words = []
        j = i
        while j < len(segs) and segs[j][0] <= st + span:
            words += tokens(segs[j][2])
            j += 1
        score = _lcs(q, words) / len(q)
        if score >= best[0] and score > 0:  # ties -> latest anchor = where the quote starts
            best = (score, st)
    return best


# ---------------------------------------------------------------- bans
def ban_hits(text):
    """Return banned active-cue terms found in text (zh + en patterns from config)."""
    hits = []
    for term, pats in CONFIG["ban_patterns"].items():
        if any(re.search(p, text, re.I) for p in pats):
            hits.append(term)
    return hits


def card_paths(arg_paths):
    out = []
    for a in arg_paths:
        p = Path(a)
        out += sorted(p.glob("*.card.json")) if p.is_dir() else [p]
    return out

"""Deterministic lexical similarity (stdlib only) shared by build_tree.py and blind_replay_text.py.

Same-mechanism rule = TF-IDF cosine over stemmed content tokens. No model, no hand-written text.
"""
import math
import re
from collections import Counter

STOP = set("""a an the and or but if then of to in on at by for with from as is are was were be been being it its this that
these those their there they them he she his her you your we our i me my not no do does did done doing can could will would
should may might must have has had having so than too very just also into over under up down out off about more most
other another such only own same both each few some any all one two three way ways often even still when while which who
whom whose what where why how because instead rather itself themselves yourself through during before after between
above below again further once here let get gets got going go goes make makes making made use used using like thing things
lot lots really actually player players body""".split())


def stem(w):
    for suf in ("ingly", "edly", "ations", "ation", "ments", "ment", "ness", "ing", "ies", "ied", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            if suf in ("ies", "ied"):
                w += "y"
            break
    return w


def toks(text):
    return [stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2]


class Sim:
    """Fit IDF on a corpus of texts; cos(a, b) between any two texts."""

    def __init__(self, corpus):
        df = Counter()
        for t in corpus:
            df.update(set(toks(t)))
        n = max(len(corpus), 1)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.default = math.log(1 + n) + 1.0

    def vec(self, text):
        c = Counter(toks(text))
        v = {w: tf * self.idf.get(w, self.default) for w, tf in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def cos(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        return sum(x * vb.get(w, 0.0) for w, x in va.items())

#!/usr/bin/env python3
"""V6 prompt neutrality: fact-extraction prompts must not contain project / FTT terms.

usage: check_neutrality.py <prompt files...>
Only applies to prompts that look at the PLAYER's video (facts_*). Episode-extraction
prompts (ep_s1/ep_s2) may name FTT but must not contain the user's problems: they are
checked against the user-terms subset.
"""
import re
import sys
from pathlib import Path

from common import CONFIG, Finding, fails

USER_TERMS = ["用户", "屈书毅", "学员问题", "我的问题", "learning.md", "problem_register", "P-0"]


def check(path):
    text = Path(path).read_text(encoding="utf-8")
    name = Path(path).name
    if "\n---\n" in text:  # template: only the part after the executor header is sent to the model
        head, text = text.split("\n---\n", 1)
        text = "\n" * (head.count("\n") + 2) + text  # keep line numbers
    terms = CONFIG["neutral_prompt_forbidden"] if name.startswith("facts_") else USER_TERMS
    out = []
    for t in terms:
        pat = re.escape(t) if not t.isascii() else r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])"
        for m in re.finditer(pat, text, re.I):
            line = text[: m.start()].count("\n") + 1
            out.append(Finding("FAIL", f"{name}:{line}", f"N01 forbidden term {t!r} in prompt"))
    return out


def main(args):
    findings = [f for p in args for f in check(p)]
    for f in findings:
        print(f)
    print(f"check_neutrality: {len(args)} prompts, {len(fails(findings))} FAIL")
    return 1 if fails(findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

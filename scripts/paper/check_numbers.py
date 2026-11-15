"""Trace every decimal number in paper/main.tex to a source.

A number is FOUND if its absolute value equals some value in results/analysis/*.json rounded to 1, 2 or 3 decimals
(JSON numbers are walked recursively; values on a 0-1 scale are also tried as percentages), or if the literal string
appears in notebook/*.md, content/*.md, GOAL.md or data_partd_c/README.md. Everything else is printed as MISSING with
its line, for a human to justify (derived arithmetic) or fix.

usage: python3 scripts/paper/check_numbers.py [paper/main.tex]   exit 0 = all found, 1 = some missing
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NUM = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})*\.\d+|(?<![\w.])[-+]?\d+\.\d+")


def json_values():
    vals = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            v = abs(float(x))
            for d in (1, 2, 3):
                vals.add(round(v, d))
                if v <= 1.0:
                    vals.add(round(100 * v, d))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for p in glob.glob(os.path.join(ROOT, "results", "analysis", "*.json")):
        with open(p) as f:
            walk(json.load(f))
    return vals


def text_corpus():
    parts = []
    for pat in ("notebook/*.md", "content/*.md", "GOAL.md", "data_partd_c/README.md"):
        for p in glob.glob(os.path.join(ROOT, pat)):
            with open(p, encoding="utf-8") as f:
                parts.append(f.read())
    return "\n".join(parts)


def main():
    tex = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper", "main.tex")
    vals, corpus = json_values(), text_corpus()
    missing, total = [], 0
    with open(tex, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.lstrip().startswith("%"):
                continue
            for m in NUM.finditer(line):
                s = m.group(0).lstrip("+-").replace(",", "")
                total += 1
                v = float(s)
                d = len(s.split(".")[1])
                if round(v, d) in vals or s in corpus or m.group(0).lstrip("+-") in corpus:
                    continue
                missing.append((ln, m.group(0), line.strip()[:110]))
    for ln, n, ctx in missing:
        print(f"MISSING line {ln}: {n}    | {ctx}")
    print(f"{total} decimal numbers, {total - len(missing)} found, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

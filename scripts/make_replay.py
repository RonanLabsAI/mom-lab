"""Replay mixtures for Step 6: specialist data + a replay slice drawn from the OTHER five domains' specialist data,
sized at FRAC of the specialist's own examples, split evenly across those domains (translate included, so replay also
exposes the specialist to non-English text). Output: data/mixtures/N14M/replay/<d>-r<pct>.jsonl (shuffled, seeded).
usage: python3 scripts/make_replay.py <frac> <domain>...     e.g. 0.2 code sql tools
"""
import json, random, sys
DOMAINS = ["math", "code", "sql", "tools", "translate", "general"]
frac = float(sys.argv[1]); targets = sys.argv[2:]
S = "data/mixtures/N14M/spec"
lines = {d: open(f"{S}/{d}.jsonl").readlines() for d in DOMAINS}
import os; os.makedirs("data/mixtures/N14M/replay", exist_ok=True)
for d in targets:
    rng = random.Random(20261029 + DOMAINS.index(d))
    others = [o for o in DOMAINS if o != d]
    per = round(frac * len(lines[d]) / len(others))
    mix = list(lines[d])
    for o in others:
        mix += rng.sample(lines[o], per)
    rng.shuffle(mix)
    out = f"data/mixtures/N14M/replay/{d}-r{int(frac * 100)}.jsonl"
    open(out, "w").writelines(mix)
    print(out, len(lines[d]), "own +", per * len(others), "replay")

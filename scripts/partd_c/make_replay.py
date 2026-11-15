"""Replay mixtures for the Part D / Experiment C skill tasks, built the same way as scripts/make_replay.py:
specialist data + a replay slice sized at FRAC of the specialist's own examples, split evenly across the replay
domains, drawn with random.sample from data/mixtures/N14M/spec/<domain>.jsonl, then shuffled (seeded).
Difference: the new tasks are not among the six N14M domains, so ALL six (math, code, sql, tools, translate,
general) are replay domains. Seed = 20261029 + 6 + index of the task in TASKS (distinct from make_replay.py's).
Output: data/mixtures/partd_c/replay/<task>-r<pct>.jsonl
usage: .venv/bin/python -P scripts/partd_c/make_replay.py 0.2 metar medcalc
"""
import os
import sys

DOMAINS = ["math", "code", "sql", "tools", "translate", "general"]
TASKS = ["metar", "medcalc", "notam"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
S = os.path.join(ROOT, "data/mixtures/N14M/spec")
P = os.path.join(ROOT, "data/mixtures/partd_c")


def main():
    import random
    frac = float(sys.argv[1]); targets = sys.argv[2:]
    lines = {d: open(f"{S}/{d}.jsonl").readlines() for d in DOMAINS}
    os.makedirs(f"{P}/replay", exist_ok=True)
    for t in targets:
        rng = random.Random(20261029 + len(DOMAINS) + TASKS.index(t))
        own = open(f"{P}/{t}.jsonl").readlines()
        per = round(frac * len(own) / len(DOMAINS))
        mix = list(own)
        for o in DOMAINS:
            mix += rng.sample(lines[o], per)
        rng.shuffle(mix)
        out = f"{P}/replay/{t}-r{int(frac * 100)}.jsonl"
        open(out, "w").writelines(mix)
        print(out, len(own), "own +", per * len(DOMAINS), "replay")


if __name__ == "__main__":
    main()

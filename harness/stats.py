"""Bootstrap statistics over scores.json.

python -m harness.stats --a RUN_A [--b RUN_B] [--step STEP] [--step-b STEP_B]
RUN may be a run id (with --step) or a path to a scores.json.
CIs: resample items with replacement within each benchmark (within each direction for translate),
2,000 reps, seed 0; s_d = mean of its benchmarks, A = mean of the 6 s_d (or of the domains present).
Paired difference (B - A): identical resampled indices on the items common to both runs;
p = two-sided bootstrap p-value 2*min(P(d<=0), P(d>=0)), floored at 1/(reps+1).
Translate uses mean sentence-level chrF++ in the bootstrap (reported scores are corpus-level), so
translate point estimates here differ slightly from scores.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import common as C

REPS = 2000


def load_scores(run, step=None, results_root=C.RESULTS) -> dict:
    p = Path(run)
    if p.suffix != ".json":
        if step is None:
            raise SystemExit("--step required when RUN is a run id")
        p = C.run_dir(step, run, Path(results_root)) / "scores.json"
    return json.loads(p.read_text())


def _groups(entry, idx):
    """list of index arrays (positions into the common item list) to resample within."""
    g = entry.get("groups")
    if not g:
        return [np.arange(len(idx))]
    g = np.array(g)[idx]
    return [np.where(g == u)[0] for u in sorted(set(g))]


def _prepare(runs, benches):
    """Per benchmark: (aligned value matrices per run, resample groups)."""
    prepped = {}
    for b in benches:
        ents = [r["benchmarks"][b] for r in runs]
        common = [i for i in ents[0]["ids"] if all(i in set(e["ids"]) for e in ents[1:])] if len(ents) > 1 \
            else list(ents[0]["ids"])
        vals = []
        for e in ents:
            pos = {i: k for k, i in enumerate(e["ids"])}
            vals.append(np.array([e["correct"][pos[i]] for i in common], dtype=float))
        pos0 = {i: k for k, i in enumerate(ents[0]["ids"])}
        idx0 = [pos0[i] for i in common]
        prepped[b] = (vals, _groups(ents[0], idx0), ents[0]["domain"], bool(ents[0].get("groups")))
    return prepped


def _boot(prepped, domains, reps, seed):
    """Returns dict run_index -> {name: array(reps)} for benchmarks, domains and A."""
    rng = np.random.default_rng(seed)
    nruns = len(next(iter(prepped.values()))[0])
    out = [{} for _ in range(nruns)]
    for b, (vals, groups, dom, is_t) in prepped.items():
        per_group = [[] for _ in range(nruns)]
        for gi in groups:
            if len(gi) == 0:
                continue
            draw = rng.integers(0, len(gi), size=(reps, len(gi)))
            for r in range(nruns):
                v = vals[r][gi]
                per_group[r].append(v[draw].mean(axis=1))
        for r in range(nruns):
            m = np.mean(per_group[r], axis=0)
            out[r][b] = m if is_t else 100.0 * m
    for r in range(nruns):
        dsc = {}
        for d in domains:
            arrs = [out[r][b] for b, p in prepped.items() if p[2] == d]
            if arrs:
                dsc[d] = np.mean(arrs, axis=0)
        out[r].update({f"domain:{d}": v for d, v in dsc.items()})
        if dsc:
            out[r]["A"] = np.mean(list(dsc.values()), axis=0)
    return out


def _ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci(scores: dict, reps=REPS, seed=0) -> dict:
    benches = list(scores["benchmarks"])
    prepped = _prepare([scores], benches)
    boot = _boot(prepped, list(C.DOMAINS), reps, seed)[0]
    res = {}
    for name, arr in boot.items():
        lo, hi = _ci(arr)
        res[name] = {"mean": float(arr.mean()), "ci_lo": lo, "ci_hi": hi}
    # point estimates from the actual data
    for b, e in scores["benchmarks"].items():
        res[b]["point"] = e["score"]
    for d, v in scores["domains"].items():
        res[f"domain:{d}"]["point"] = v
    if "A" in res:
        res["A"]["point"] = scores["A"] if scores.get("A") is not None else scores["A_partial"]
    return res


def paired_diff(sa: dict, sb: dict, reps=REPS, seed=0) -> dict:
    """B - A on common benchmarks and common item ids."""
    benches = [b for b in sa["benchmarks"] if b in sb["benchmarks"]]
    if not benches:
        raise SystemExit("no common benchmarks")
    prepped = _prepare([sa, sb], benches)
    doms = [d for d in C.DOMAINS if any(p[2] == d for p in prepped.values())]
    ba, bb = _boot(prepped, doms, reps, seed)
    res = {}
    for name in ba:
        d = bb[name] - ba[name]
        lo, hi = _ci(d)
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        res[name] = {"a": float(ba[name].mean()), "b": float(bb[name].mean()),
                     "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                     "p": float(min(1.0, max(p, 1.0 / (reps + 1))))}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harness.stats")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b")
    ap.add_argument("--step")
    ap.add_argument("--step-b")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-root", default=str(C.RESULTS))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sa = load_scores(a.a, a.step, a.results_root)
    if a.b:
        sb = load_scores(a.b, a.step_b or a.step, a.results_root)
        res = paired_diff(sa, sb, a.reps, a.seed)
        if a.json:
            print(json.dumps(res, indent=2))
            return
        print(f"{'':22}{'A':>8}{'B':>8}{'B-A':>8}   95% CI          p")
        for k, v in res.items():
            print(f"{k:22}{v['a']:8.2f}{v['b']:8.2f}{v['diff']:+8.2f}   [{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}]  {v['p']:.4f}")
    else:
        res = bootstrap_ci(sa, a.reps, a.seed)
        if a.json:
            print(json.dumps(res, indent=2))
            return
        print(f"{'':22}{'score':>8}   95% CI")
        for k, v in res.items():
            print(f"{k:22}{v['point']:8.2f}   [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")


if __name__ == "__main__":
    main()

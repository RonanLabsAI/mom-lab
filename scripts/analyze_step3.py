"""Step 3 analysis: specialists vs pooled, oracle-domain routing, base+specialists, transfer matrix.

usage: python3 scripts/analyze_step3.py <prefix> <lr> [--base base-qwen3-1.7b]
  prefix: pt17 or b17;  lr: e.g. 2e-5  (specialists <prefix>-spec-<d>-lr<lr>, pooled <prefix>-pooled-lr<lr>)
Composite = mean of six domain scores (0-100). Item-level bootstrap (2,000 resamples, within benchmark)
gives 95% CIs and paired differences. Oracle-domain library: each domain answered by its own specialist.
Base+specialists: per domain, the better of base and specialist (chosen on these same dev items, so this is
an optimistic upper bound for that design; Part B makes the choice on dev and scores on test).
"""
import json, os, sys, random
import numpy as np
DOMAINS = ["math", "code", "sql", "tools", "translate", "general"]
pre, lr = sys.argv[1], sys.argv[2]
base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "base-qwen3-1.7b"
R = "results/step3"


def load(run):
    p = f"{R}/{run}/scores.json"
    return json.load(open(p)) if os.path.exists(p) else None


def bench_items(sc):
    """benchmark -> (domain, ids, correct array)"""
    return {b: (v["domain"], v["ids"], np.array(v["correct"], dtype=float)) for b, v in sc["benchmarks"].items() if v["domain"] in DOMAINS}  # mixed set scored in Step 4


def domain_scores_from(bmap):
    """composite given benchmark->correct arrays (translate uses mean sentence chrF as proxy)."""
    dom = {}
    for b, (d, ids, c) in bmap.items():
        dom.setdefault(d, []).append(100 * c.mean() if b != "flores200" else c.mean())
    return {d: float(np.mean(v)) for d, v in dom.items()}


def assemble(source_by_domain, runs):
    """build a benchmark map where each domain's benchmarks come from the run chosen for that domain."""
    out = {}
    for d, run in source_by_domain.items():
        for b, (dd, ids, c) in bench_items(runs[run]).items():
            if dd == d:
                out[b] = (dd, ids, c)
    return out


def boot(bmaps, n=2000, seed=0):
    """bootstrap composite for several systems on identical resampled items; returns array [n, systems]."""
    rng = np.random.default_rng(seed)
    benches = list(bmaps[0].keys())
    res = np.zeros((n, len(bmaps)))
    for i in range(n):
        idx = {b: rng.integers(0, len(bmaps[0][b][2]), len(bmaps[0][b][2])) for b in benches}
        for k, bm in enumerate(bmaps):
            dom = {}
            for b in benches:
                d, ids, c = bm[b]
                v = c[idx[b]].mean()
                dom.setdefault(d, []).append(100 * v if b != "flores200" else v)
            res[i, k] = np.mean([np.mean(v) for v in dom.values()])
    return res


runs = {base: load(base), "pooled": load(f"{pre}-pooled-lr{lr}")}
specs = {d: load(f"{pre}-spec-{d}-lr{lr}") for d in DOMAINS}
missing = [k for k, v in list(runs.items()) + [(f"spec-{d}", s) for d, s in specs.items()] if v is None]
print(f"== {pre} lr {lr}   missing: {missing or 'none'}")
for d, s in specs.items():
    if s:
        runs[f"spec-{d}"] = s

# transfer matrix: rows = model, cols = domain score
print("\nTransfer matrix (domain scores):")
print(f"{'model':16s}" + "".join(f"{d:>10s}" for d in DOMAINS) + f"{'A':>8s}")
for name, sc in runs.items():
    ds = sc["domains"]
    print(f"{name:16s}" + "".join(f"{ds.get(d, float('nan')):10.1f}" for d in DOMAINS) + f"{np.mean([ds[d] for d in DOMAINS]):8.2f}")

if not missing:
    oracle = assemble({d: f"spec-{d}" for d in DOMAINS}, runs)
    best_of = {d: (f"spec-{d}" if runs[f"spec-{d}"]["domains"][d] >= runs[base]["domains"][d] else base) for d in DOMAINS}
    basespec = assemble(best_of, runs)
    pooled = bench_items(runs["pooled"]); basem = bench_items(runs[base])
    systems = {"base": basem, "pooled": pooled, "oracle-library": oracle, "base+specialists": basespec}
    B = boot(list(systems.values()))
    point = {k: domain_scores_from(v) for k, v in systems.items()}
    print("\nSystems (composite A, 95% bootstrap CI over items):")
    for k, (name, bm) in enumerate(systems.items()):
        A = np.mean([point[name][d] for d in DOMAINS])
        lo, hi = np.percentile(B[:, k], [2.5, 97.5])
        print(f"  {name:18s} A={A:6.2f}  [{lo:6.2f}, {hi:6.2f}]   " + " ".join(f"{d}={point[name][d]:.1f}" for d in DOMAINS))
    print("  base+specialists routes:", best_of)
    names = list(systems)
    for a, b in [("oracle-library", "pooled"), ("base+specialists", "pooled"), ("pooled", "base"), ("base+specialists", "base")]:
        dlt = B[:, names.index(a)] - B[:, names.index(b)]
        print(f"  {a} - {b}: {dlt.mean():+.2f}  [{np.percentile(dlt, 2.5):+.2f}, {np.percentile(dlt, 97.5):+.2f}]  P(>0)={np.mean(dlt > 0):.3f}")
    out = {"prefix": pre, "lr": lr, "point": point, "routes": best_of,
           "boot_ci": {n: [float(np.percentile(B[:, i], 2.5)), float(np.percentile(B[:, i], 97.5))] for i, n in enumerate(names)}}
    os.makedirs("results/analysis", exist_ok=True)
    json.dump(out, open(f"results/analysis/step3_{pre}_lr{lr}.json", "w"), indent=1)
    print("note: translate composite in bootstrap uses mean sentence chrF++ (point values from scores.json use corpus chrF++).")

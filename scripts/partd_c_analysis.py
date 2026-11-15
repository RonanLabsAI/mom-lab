"""Part D addendum C analysis (content/PART_D_C_ADDENDUM.md): METAR decoding, MedCalc-Bench, Pre-Flight.

Rules as PART_D_PREREG.md: seed-averaged per-item scores (0/100); paired item bootstrap (2,000 resamples, numpy seed
0), 95% percentile CI; beats = CI low > 0; NI = CI low > -2; FAIL = CI high < bar; else INCONCLUSIVE.
Results from results/partdC/<run>/scores.json (split=test, n_missing=0, item ids must match the untouched 4B's).
Each adapter is scored as always-on (single-domain test files = oracle routing on its own task; on the other two
benchmarks it shows out-of-lane behaviour), as the addendum states.
Tests per task t in {metar, medcalc}: C1 4B+t > 4B; C2 4B+t > 8B; C3 1.7B+t > 4B and > 8B (two tests);
C4 replay NI to plain on t, and replay > plain on each of the other two benchmarks (two tests).
C5 (descriptive): Pre-Flight 8B - 4B and the change from each 4B / 1.7B adapter, with CIs.
C6 (descriptive): (untouched base score, adapter gain) per size and task, for the gain-vs-base-weakness plot.
Extra (descriptive): AWQ-8B comparisons; METAR mean field accuracy.

usage: .venv/bin/python -P scripts/partd_c_analysis.py   -> results/analysis/partd_c.json
"""
import json, os
import numpy as np

R = "results/partdC"; NB = 2000; NI = -2.0
S3 = ["", "-s1", "-s2"]
BENCH = ["metar", "medcalc", "preflight"]
SYS = {"1.7B": ["base-qwen3-1.7b"], "4B": ["base-qwen3-4b"], "8B": ["base-qwen3-8b"], "AWQ-8B": ["base-qwen3-8b-awq"]}
for t in ("metar", "medcalc"):
    SYS[f"4B+{t}"] = [f"pt4-lora16-{t}-lr2e-4{s}" for s in S3]
    SYS[f"1.7B+{t}"] = [f"pt17-lora16-{t}-lr2e-4{s}" for s in S3]
    SYS[f"4B+{t} replay"] = [f"pt4-lora16-replay-{t}-r20-lr2e-4{s}" for s in S3]


def load(run):
    p = f"{R}/{run}/scores.json"
    if not os.path.exists(p):
        return None
    sc = json.load(open(p))
    if sc.get("split") != "test":
        return None
    b = sc["benchmarks"]
    return b if all(k in b and b[k].get("n_missing", 0) == 0 for k in BENCH) else None


def main():
    ref = load("base-qwen3-4b"); ids = {b: ref[b]["ids"] for b in BENCH}
    arr, missing, mfa = {}, [], {}
    for name, runs in SYS.items():
        cols = {b: [] for b in BENCH}; ok = True
        for r in runs:
            s = load(r)
            if s is None or any(s[b]["ids"] != ids[b] for b in BENCH):
                missing.append(r); ok = False; break
            for b in BENCH:
                cols[b].append(np.array(s[b]["correct"], float) * 100)
            mfa.setdefault(name, []).append(s["metar"].get("extra", {}).get("mean_field_accuracy"))
        if ok:
            arr[name] = {b: np.mean(cols[b], axis=0) for b in BENCH}
    rng = np.random.default_rng(0)
    idx = {b: [rng.integers(0, len(ids[b]), len(ids[b])) for _ in range(NB)] for b in BENCH}

    def test(hyp, b, x, y, kind):
        if x not in arr or y not in arr:
            return {"hyp": hyp, "bench": b, "x": x, "y": y, "verdict": "MISSING"}
        X, Y = arr[x][b], arr[y][b]
        diff = float(X.mean() - Y.mean()); bs = np.array([X[i].mean() - Y[i].mean() for i in idx[b]])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        if kind == "desc":
            return {"hyp": hyp, "bench": b, "x": x, "y": y, "kind": kind, "diff": diff, "lo": lo, "hi": hi, "verdict": "descriptive"}
        bar = 0.0 if kind == "beats" else NI
        v = "PASS" if lo > bar else "FAIL" if hi < bar else "INCONCLUSIVE"
        return {"hyp": hyp, "bench": b, "x": x, "y": y, "kind": kind, "diff": diff, "lo": lo, "hi": hi, "verdict": v}

    T = []
    for t in ("metar", "medcalc"):
        other = [b for b in BENCH if b != t]
        T += [test("C1", t, f"4B+{t}", "4B", "beats"), test("C2", t, f"4B+{t}", "8B", "beats"),
              test("C3", t, f"1.7B+{t}", "4B", "beats"), test("C3", t, f"1.7B+{t}", "8B", "beats"),
              test("C4", t, f"4B+{t} replay", f"4B+{t}", "ni")]
        T += [test("C4-damage", b, f"4B+{t} replay", f"4B+{t}", "beats") for b in other]
        T += [test("extra", t, f"4B+{t}", "AWQ-8B", "desc"), test("extra", t, f"1.7B+{t}", "AWQ-8B", "desc")]
    T.append(test("C5", "preflight", "8B", "4B", "desc"))
    for t in ("metar", "medcalc"):
        T += [test("C5", "preflight", f"4B+{t}", "4B", "desc"), test("C5", "preflight", f"1.7B+{t}", "1.7B", "desc"),
              test("C5", "preflight", f"4B+{t} replay", "4B", "desc")]
    c6 = []
    for size in ("1.7B", "4B"):
        for t in ("metar", "medcalc"):
            if size in arr and f"{size}+{t}" in arr:
                c6.append({"size": size, "task": t, "base": float(arr[size][t].mean()),
                           "gain": float(arr[f"{size}+{t}"][t].mean() - arr[size][t].mean())})
    out = {"n": {b: len(ids[b]) for b in BENCH}, "missing": missing,
           "scores": {n: {b: float(v[b].mean()) for b in BENCH} for n, v in arr.items()},
           "metar_mean_field_accuracy": {n: float(np.mean(v)) for n, v in mfa.items() if n in arr and None not in v},
           "tests": T, "c6": c6}
    os.makedirs("results/analysis", exist_ok=True)
    p = "results/analysis/partd_c.json"; json.dump(out, open(p, "w"), indent=1)
    print(f"== Part D C (METAR / MedCalc / Pre-Flight) n={out['n']}")
    for n, v in out["scores"].items():
        print(f"  {n:20s} " + "  ".join(f"{b} {v[b]:5.1f}" for b in BENCH) + f"  metar-fields {out['metar_mean_field_accuracy'].get(n, float('nan')):5.1f}")
    for t in T:
        print(f"  {t['hyp']:9s} {t['bench']:9s} {t['x']:18s} vs {t['y']:12s} " +
              (f"{t['diff']:+6.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] {t['verdict']}" if "diff" in t else t["verdict"]))
    print("  C6:", [(c["size"], c["task"], round(c["base"], 1), round(c["gain"], 1)) for c in c6])
    if missing:
        print("  missing:", missing)
    print("wrote", p)


if __name__ == "__main__":
    main()

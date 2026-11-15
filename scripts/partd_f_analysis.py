"""Part D addendum F analysis (content/PART_D_F_ADDENDUM.md): do the C adapters still win against a prompted /
thinking 8B?

Rules as PART_D_PREREG.md: per-item scores 0/100; seed-averaged per item where a system has several runs; paired
item bootstrap (2,000 resamples, numpy seed 0), 95% percentile CI; beats = CI low > 0; FAIL = CI high < 0; else
INCONCLUSIVE. Runs must have split=test, n_missing=0 and the item ids of the untouched 4B (results/partdC).
Adapters and zero-shot bases come from results/partdC (Part D C runs, unchanged); prompted runs from results/partdF.
Sampled (thinking) systems average the seeds present among s0-s2; s0 is required. Seeds used are reported.

Tests per task t in {metar, medcalc} (x = the 4B + t adapter, 3 training seeds):
  F1 x beats 8B few-shot;  F2 x beats 8B thinking;  F3 x beats 8B few-shot + thinking;
  F4 (headline) x beats the best 8B condition (highest mean on t among zero-shot, few-shot, thinking, few-shot +
     thinking; all four required);  F5 1.7B + t adapter beats the best 8B condition.
Descriptive: each prompting condition minus zero-shot for 1.7B / 4B / 8B; x vs the best 4B condition; x vs 14B,
32B and 32B thinking (if run); share of the 8B-zero-shot-to-adapter gap that the best 8B condition closes;
Pre-Flight thinking minus zero-shot; MedCalc F1-F5 without the items whose note contains an exemplar note
(data_partd_f/build.json any_exemplar); METAR all-fields score with "HHMM" accepted for time_utc; mean output tokens
and latency per item; truncation rate (finish_reason length) per run.

usage: .venv/bin/python -P scripts/partd_f_analysis.py      -> results/analysis/partd_f.json
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from harness_d.score import metar as M  # noqa: E402

RC, RF = "results/partdC", "results/partdF"
NB = 2000
BENCH = ["metar", "medcalc", "preflight"]
GEN = {"metar": "aviation/metar", "medcalc": "medcalc/medcalc", "preflight": "avknow/preflight"}
S3 = ["", "-s1", "-s2"]
TS = ["-s0", "-s1", "-s2"]
SYS = {}  # name -> (root, [runs], benches, sampled)
for sz in ("1.7b", "4b", "8b"):
    SZ = sz.upper()
    SYS[f"{SZ} zero-shot"] = (RC, [f"base-qwen3-{sz}"], BENCH, False)
    SYS[f"{SZ} few-shot"] = (RF, [f"base-qwen3-{sz}-fs"], ["metar", "medcalc"], False)
for sz in ("4b", "8b"):
    SZ = sz.upper()
    SYS[f"{SZ} thinking"] = (RF, [f"base-qwen3-{sz}-think{s}" for s in TS], BENCH, True)
    SYS[f"{SZ} few-shot+thinking"] = (RF, [f"base-qwen3-{sz}-fsthink{s}" for s in TS], ["metar", "medcalc"], True)
SYS["14B zero-shot"] = (RF, ["base-qwen3-14b"], BENCH, False)
SYS["32B zero-shot"] = (RF, ["base-qwen3-32b"], BENCH, False)
SYS["32B thinking"] = (RF, ["base-qwen3-32b-think-s0"], BENCH, True)
for t in ("metar", "medcalc"):
    SYS[f"4B+{t}"] = (RC, [f"pt4-lora16-{t}-lr2e-4{s}" for s in S3], BENCH, False)
    SYS[f"1.7B+{t}"] = (RC, [f"pt17-lora16-{t}-lr2e-4{s}" for s in S3], BENCH, False)
COND8 = ["8B zero-shot", "8B few-shot", "8B thinking", "8B few-shot+thinking"]
COND4 = ["4B zero-shot", "4B few-shot", "4B thinking", "4B few-shot+thinking"]


def load(root, run, benches, ids):
    p = f"{root}/{run}/scores.json"
    if not os.path.exists(p):
        return None
    sc = json.load(open(p))
    if sc.get("split") != "test":
        return None
    b = sc["benchmarks"]
    if not all(k in b and b[k].get("n_missing", 0) == 0 and b[k]["ids"] == ids[k] for k in benches):
        return None
    return b


def gens(root, run, bench):
    p = f"{root}/{run}/gen/{GEN[bench]}.test.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()} if os.path.exists(p) else None


def lenient_metar(items, g):
    out = []
    for it in items:
        o = g[it["id"]]["output"] or ""
        if "</think>" in o:
            o = o.split("</think>", 1)[1]
        elif "<think>" in o:
            o = ""
        obj = M.parse_output(o) or {}
        tu = obj.get("time_utc")
        if isinstance(tu, str) and ":" not in tu and len(tu.strip()) == 4 and tu.strip().isdigit():
            obj["time_utc"] = tu.strip()[:2] + ":" + tu.strip()[2:]
        out.append(100.0 * all(M.field_ok(f, obj.get(f, M._MISSING), it["reference"][f]) for f in M.FIELDS))
    return np.array(out)


def main():
    sc0 = json.load(open(f"{RC}/base-qwen3-4b/scores.json"))["benchmarks"]
    ids = {b: sc0[b]["ids"] for b in BENCH}
    metar_items = [json.loads(l) for l in open("data_partd_c/eval/aviation/metar/test.jsonl")]
    assert [it["id"] for it in metar_items] == ids["metar"]
    arr, len_arr, used, missing, cost, trunc = {}, {}, {}, [], {}, {}
    for name, (root, runs, benches, sampled) in SYS.items():
        cols = {b: [] for b in benches}; lcol = []; got = []
        for r in runs:
            s = load(root, r, benches, ids)
            if s is None:
                if sampled and r != runs[0]:
                    continue  # extra sampling seed not run: average the seeds present
                missing.append(f"{root}/{r}"); got = None; break
            got.append(r)
            for b in benches:
                cols[b].append(np.array(s[b]["correct"], float) * 100)
                g = gens(root, r, b)
                if g is not None:
                    cost.setdefault(name, {}).setdefault(b, []).append(
                        (np.mean([x.get("n_out_tokens") or 0 for x in g.values()]), np.mean([x.get("latency_s") or 0 for x in g.values()])))
                    trunc[f"{r}/{b}"] = 100.0 * float(np.mean([x.get("finish_reason") == "length" for x in g.values()]))
                    if b == "metar":
                        lcol.append(lenient_metar(metar_items, g))
        if not got:
            continue
        arr[name] = {b: np.mean(cols[b], axis=0) for b in benches}
        if lcol:
            len_arr[name] = np.mean(lcol, axis=0)
        used[name] = got
    rng = np.random.default_rng(0)
    idx = {b: [rng.integers(0, len(ids[b]), len(ids[b])) for _ in range(NB)] for b in BENCH}
    ov = set(json.load(open("data_partd_f/build.json"))["overlap"]["test"]["any_exemplar"])
    keep = np.array([i not in ov for i in ids["medcalc"]])
    rng2 = np.random.default_rng(0); nk = int(keep.sum())
    idx_k = [rng2.integers(0, nk, nk) for _ in range(NB)]

    def test(hyp, b, x, y, kind, sub=False):
        if x not in arr or y not in arr or b not in arr[x] or b not in arr[y]:
            return {"hyp": hyp, "bench": b, "x": x, "y": y, "verdict": "MISSING"}
        X, Y = arr[x][b], arr[y][b]
        ii = idx[b]
        if sub:
            X, Y, ii = X[keep], Y[keep], idx_k
        diff = float(X.mean() - Y.mean()); bs = np.array([X[i].mean() - Y[i].mean() for i in ii])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        v = "descriptive" if kind == "desc" else "PASS" if lo > 0 else "FAIL" if hi < 0 else "INCONCLUSIVE"
        return {"hyp": hyp, "bench": b, "x": x, "y": y, "kind": kind, "diff": diff, "lo": lo, "hi": hi, "verdict": v,
                **({"subset": "medcalc without exemplar-overlap items", "n": nk} if sub else {})}

    def best(conds, t):
        if not all(c in arr for c in conds):
            return None
        return max(conds, key=lambda c: (arr[c][t].mean(), -conds.index(c)))

    T, bests = [], {}
    for t in ("metar", "medcalc"):
        b8 = best(COND8, t); b4 = best(COND4, t); bests[t] = {"8B": b8, "4B": b4}
        T += [test("F1", t, f"4B+{t}", "8B few-shot", "beats"), test("F2", t, f"4B+{t}", "8B thinking", "beats"),
              test("F3", t, f"4B+{t}", "8B few-shot+thinking", "beats")]
        T += [test("F4", t, f"4B+{t}", b8, "beats") if b8 else {"hyp": "F4", "bench": t, "verdict": "MISSING"},
              test("F5", t, f"1.7B+{t}", b8, "beats") if b8 else {"hyp": "F5", "bench": t, "verdict": "MISSING"}]
        for c in ("few-shot", "thinking", "few-shot+thinking"):
            for sz in ("1.7B", "4B", "8B"):
                if f"{sz} {c}" in SYS:
                    T.append(test("prompt-gain", t, f"{sz} {c}", f"{sz} zero-shot", "desc"))
        if b4:
            T.append(test("vs-best-4B", t, f"4B+{t}", b4, "desc"))
        for big in ("14B zero-shot", "32B zero-shot", "32B thinking"):
            T.append(test("vs-large", t, f"4B+{t}", big, "desc"))
        if t == "medcalc":
            T += [test("F1-sens", t, f"4B+{t}", "8B few-shot", "beats", True), test("F2-sens", t, f"4B+{t}", "8B thinking", "beats", True),
                  test("F3-sens", t, f"4B+{t}", "8B few-shot+thinking", "beats", True)]
            if b8:
                T += [test("F4-sens", t, f"4B+{t}", b8, "beats", True), test("F5-sens", t, f"1.7B+{t}", b8, "beats", True)]
    for x, y in (("4B thinking", "4B zero-shot"), ("8B thinking", "8B zero-shot"), ("32B thinking", "32B zero-shot"),
                 ("8B zero-shot", "4B zero-shot"), ("32B zero-shot", "8B zero-shot")):
        T.append(test("preflight", "preflight", x, y, "desc"))
    gap = {}
    for t in ("metar", "medcalc"):
        b8 = bests[t]["8B"]
        if b8 and f"4B+{t}" in arr:
            z, a, bb = arr["8B zero-shot"][t].mean(), arr[f"4B+{t}"][t].mean(), arr[b8][t].mean()
            gap[t] = {"best_8B": b8, "share_closed": float((bb - z) / (a - z)) if a != z else None}
    out = {"n": {b: len(ids[b]) for b in BENCH}, "missing": missing, "seeds_used": used, "best": bests,
           "scores": {n: {b: float(v[b].mean()) for b in v} for n, v in arr.items()},
           "metar_lenient_time": {n: float(v.mean()) for n, v in len_arr.items()},
           "cost_per_item": {n: {b: {"out_tokens": float(np.mean([c[0] for c in v])), "latency_s": float(np.mean([c[1] for c in v]))}
                                 for b, v in d.items()} for n, d in cost.items()},
           "truncation_pct": trunc, "gap_closed": gap, "tests": T}
    os.makedirs("results/analysis", exist_ok=True)
    p = "results/analysis/partd_f.json"; json.dump(out, open(p, "w"), indent=1)
    print(f"== Part D F (prompted / thinking baselines) n={out['n']}")
    for n, v in out["scores"].items():
        c = out["cost_per_item"].get(n, {})
        print(f"  {n:22s} " + "  ".join(f"{b} {v[b]:5.1f}" if b in v else f"{b}   -- " for b in BENCH)
              + f"  metar-lenient {out['metar_lenient_time'].get(n, float('nan')):5.1f}"
              + "  tokens " + " ".join(f"{b[:3]} {c[b]['out_tokens']:.0f}" for b in c) + f"  seeds {len(used[n])}")
    print("  best:", bests, " gap closed:", gap)
    for t in T:
        print(f"  {t['hyp']:11s} {t['bench']:9s} {t.get('x', ''):22s} vs {t.get('y', '') or '':22s} " +
              (f"{t['diff']:+6.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] {t['verdict']}" if "diff" in t else t["verdict"]))
    hi_tr = {k: round(float(v), 1) for k, v in trunc.items() if v > 0}
    print("  truncation % (nonzero):", hi_tr)
    if missing:
        print("  missing:", missing)
    print("wrote", p)


if __name__ == "__main__":
    main()

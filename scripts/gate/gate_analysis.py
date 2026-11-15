"""METAR schema gate analysis (preregs/2026-11-14-metar-schema-gate.md, "Tests" and "Decision rule").

Statistics as content/PART_D_PREREG.md and scripts/partd_f_analysis.py: per-item 0/100, per-item mean over the runs
of a system, paired item bootstrap over the 1,000 METAR test items, 2,000 resamples drawn once from numpy
default_rng(0) and shared by every test, 95% percentile CI. PASS = CI low > 0, FAIL = CI high < 0, else INCONCLUSIVE.

x = 4B+metar (Part D-C runs pt4-lora16-metar-lr2e-4{,-s1,-s2}). Schema arms (all primary):
  8B: zero-shot, 5-shot, 5-shot+thinking s0;  32B: zero-shot, 5-shot, thinking s0.
G1: x vs the best 8B schema arm. G2: x vs the best 32B schema arm. "Best" = highest test mean; ties go to the arm
    listed first above. Selection after the fact makes G1/G2 harder for x and makes G3 "equivalent" easier.
G3: d = x minus the best schema arm overall (TOST, margin 2.0): "schema ahead" if CI high < 0; "equivalent" if
    CI low > -2 and CI high < 2; otherwise "not equivalent".
G4 (descriptive): each schema arm minus its unconstrained control (gate-server reruns; thinking arms vs Part F runs,
    labelled "different serving path, no reasoning parser").
S-cluster (sensitivity): G1-G3 with a station-cluster bootstrap (resample stations, then items within a station).
Headline:
  VOID      parity not 1,000/1,000, or any primary arm missing, or a schema arm fails the canary (a non-truncated
            output that is unparseable, schema-invalid, or fenced; or lenient-time score above strict).
  Rewritten G3 "schema ahead" or "equivalent".
  Stands    G1 PASS, G2 PASS and CI low of d > 2.0.
  Narrowed  anything else; reported as "d = adapter minus best schema arm = D [lo, hi]" with no direction word.

usage: .venv-gate/bin/python -P scripts/gate/gate_analysis.py [--results-root results] [--cf-root results] [--out P]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jsonschema
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from harness_d.score import metar as M  # noqa: E402
from partd_f_analysis import lenient_metar  # noqa: E402  (frozen helper; importing does not run its main)

NB, MARGIN = 2000, 2.0
X = "4B+metar"
G8 = ["8B zero-shot + schema", "8B few-shot + schema", "8B few-shot+thinking + schema s0"]
G32 = ["32B zero-shot + schema", "32B few-shot + schema", "32B thinking + schema s0"]
F_BEST_8B, F_32B_THINK = "8B few-shot+thinking (F)", "32B thinking (F)"


def systems(g, cf):
    S = {X: [(cf, "partdC", f"pt4-lora16-metar-lr2e-4{s}") for s in ("", "-s1", "-s2")],
         "8B zero-shot (C)": [(cf, "partdC", "base-qwen3-8b")],
         "8B few-shot (F)": [(cf, "partdF", "base-qwen3-8b-fs")],
         "8B few-shot+thinking s0 (F)": [(cf, "partdF", "base-qwen3-8b-fsthink-s0")],
         F_BEST_8B: [(cf, "partdF", f"base-qwen3-8b-fsthink{s}") for s in ("-s0", "-s1", "-s2")],
         F_32B_THINK: [(cf, "partdF", "base-qwen3-32b-think-s0")],
         "32B zero-shot (F)": [(cf, "partdF", "base-qwen3-32b")]}
    for sz in ("8b", "32b"):
        SZ = sz.upper()
        S[f"{SZ} zero-shot (gate, no schema)"] = [(g, "partdG", f"gate-qwen3-{sz}-zs-free")]
        S[f"{SZ} few-shot (gate, no schema)"] = [(g, "partdG", f"gate-qwen3-{sz}-fs-free")]
        S[f"{SZ} zero-shot + schema"] = [(g, "partdG", f"gate-qwen3-{sz}-zs-schema")]
        S[f"{SZ} few-shot + schema"] = [(g, "partdG", f"gate-qwen3-{sz}-fs-schema")]
    S["8B few-shot+thinking + schema s0"] = [(g, "partdG", "gate-qwen3-8b-fsthink-schema-s0")]
    S["32B thinking + schema s0"] = [(g, "partdG", "gate-qwen3-32b-think-schema-s0")]
    return S


G4_PAIRS = [("8B zero-shot + schema", "8B zero-shot (gate, no schema)", "same server"),
            ("8B few-shot + schema", "8B few-shot (gate, no schema)", "same server"),
            ("32B zero-shot + schema", "32B zero-shot (gate, no schema)", "same server"),
            ("32B few-shot + schema", "32B few-shot (gate, no schema)", "same server"),
            ("8B few-shot+thinking + schema s0", "8B few-shot+thinking s0 (F)",
             "different serving path, no reasoning parser"),
            ("32B thinking + schema s0", F_32B_THINK, "different serving path, no reasoning parser")]


def load_metar(root, step, run, ids):
    p = os.path.join(root, step, run, "scores.json")
    if not os.path.exists(p):
        return None
    sc = json.load(open(p))
    b = sc["benchmarks"].get("metar")
    if sc.get("split") != "test" or b is None or b.get("n_missing", 0) != 0:
        return None
    if b["ids"] != ids:
        raise ValueError(f"{p}: item ids differ from the reference order")
    return b


def gens(root, step, run):
    p = os.path.join(root, step, run, "gen", "aviation", "metar.test.jsonl")
    if not os.path.exists(p):
        return None
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}


def canary(g, validator):
    """Counts for a schema arm's outputs; any bad non-truncated output means the grammar was not applied."""
    c = {"truncated": 0, "unparseable": 0, "schema_invalid": 0, "fenced": 0}
    for r in g.values():
        if r.get("finish_reason") == "length":
            c["truncated"] += 1
            continue
        out = r.get("output") or ""
        obj = M.parse_output(out)
        c["fenced"] += "```" in out.split("</think>")[-1]
        if obj is None:
            c["unparseable"] += 1
        elif not validator.is_valid(obj):
            c["schema_invalid"] += 1
    return c


def boot_indices(n, station, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    item_idx = [rng.integers(0, n, n) for _ in range(NB)]
    groups = {}
    for i, s in enumerate(station):
        groups.setdefault(s, []).append(i)
    keys = sorted(groups)
    rng2 = np.random.default_rng(rng_seed)
    clus = []
    for _ in range(NB):
        pick = rng2.integers(0, len(keys), len(keys))
        idx = []
        for k in pick:
            members = groups[keys[k]]
            idx.extend(np.asarray(members)[rng2.integers(0, len(members), len(members))])
        clus.append(np.asarray(idx))
    return item_idx, clus


def analyse(g_root, cf_root, data_file, require_gens=True):
    ids = json.load(open(os.path.join(cf_root, "partdC", "base-qwen3-8b", "scores.json")))["benchmarks"]["metar"]["ids"]
    items = [json.loads(l) for l in open(data_file)]
    assert [it["id"] for it in items] == ids
    station = [it["meta"]["raw"].split()[0] for it in items]
    validator = jsonschema.Draft202012Validator(json.load(open(os.path.join(ROOT, "configs/gate/metar_schema.json"))))
    arr, field, parse_fail, cost, trunc, lenient, canaries, missing = {}, {}, {}, {}, {}, {}, {}, []
    for name, runs in systems(g_root, cf_root).items():
        cols, fa, pf, got = [], [], [], True
        for root, step, run in runs:
            b = load_metar(root, step, run, ids)
            if b is None:
                missing.append(f"{step}/{run}"); got = False; break
            cols.append(np.array(b["correct"], float) * 100)
            fa.append(b["extra"]["field_accuracy"]); pf.append(b["extra"]["parse_fail_rate"])
            gg = gens(root, step, run)
            if step == "partdG" and gg is not None:
                cost[name] = {"out_tokens": float(np.mean([x.get("n_out_tokens") or 0 for x in gg.values()])),
                              "latency_s": float(np.mean([x.get("latency_s") or 0 for x in gg.values()]))}
                trunc[name] = 100.0 * float(np.mean([x.get("finish_reason") == "length" for x in gg.values()]))
                lenient[name] = float(np.mean(lenient_metar(items, gg)))
                if "schema" in run:
                    canaries[name] = canary(gg, validator)
            elif step == "partdG" and "schema" in run:
                canaries[name] = None  # outputs not available: canary cannot be checked
        if got:
            arr[name] = np.mean(cols, axis=0)
            field[name] = {k: float(np.mean([d[k] for d in fa])) for k in fa[0]}
            parse_fail[name] = float(np.mean(pf))
    item_idx, clus_idx = boot_indices(len(ids), station)

    def test(hyp, x, y, kind, idx=item_idx, note=None):
        if x not in arr or y not in arr:
            return {"hyp": hyp, "x": x, "y": y, "verdict": "MISSING"}
        A, B = arr[x], arr[y]
        bs = np.array([A[i].mean() - B[i].mean() for i in idx])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        v = "descriptive" if kind == "desc" else "PASS" if lo > 0 else "FAIL" if hi < 0 else "INCONCLUSIVE"
        out = {"hyp": hyp, "x": x, "y": y, "kind": kind, "diff": float(A.mean() - B.mean()), "lo": lo, "hi": hi,
               "verdict": v}
        if note:
            out["note"] = note
        return out

    def best(conds):
        if not all(c in arr for c in conds):
            return None
        return max(conds, key=lambda c: (arr[c].mean(), -conds.index(c)))

    def g3(t):
        if t["verdict"] == "MISSING":
            return t
        t["verdict"] = ("schema ahead" if t["hi"] < 0 else "equivalent" if (t["lo"] > -MARGIN and t["hi"] < MARGIN)
                        else "not equivalent")
        return t

    b8, b32, ball = best(G8), best(G32), best(G8 + G32)
    T = []
    for idx, tag in ((item_idx, ""), (clus_idx, "-cluster")):
        T += [test("G1" + tag, X, b8, "beats", idx) if b8 else {"hyp": "G1" + tag, "verdict": "MISSING"},
              test("G2" + tag, X, b32, "beats", idx) if b32 else {"hyp": "G2" + tag, "verdict": "MISSING"},
              g3(test("G3" + tag, X, ball, "desc", idx)) if ball else {"hyp": "G3" + tag, "verdict": "MISSING"}]
    T += [test("G4", a, b, "desc", note=n) for a, b, n in G4_PAIRS]
    T += [test("context", X, F_BEST_8B, "desc"), test("context", X, F_32B_THINK, "desc")]
    if ball:
        T += [test("context", ball, F_BEST_8B, "desc"), test("context", ball, F_32B_THINK, "desc")]

    void = []
    parity_p = os.path.join(g_root, "partdG", "parity.json")
    parity = json.load(open(parity_p)) if os.path.exists(parity_p) else None
    if not (parity and parity.get("ok")):
        void.append("parity not 1,000/1,000")
    miss_primary = [a for a in G8 + G32 if a not in arr]
    if miss_primary or X not in arr:
        void.append("missing arms: " + ", ".join(miss_primary or [X]))
    for name, c in canaries.items():
        if c is None:
            if require_gens:
                void.append(f"{name}: outputs missing, canary unchecked")
        elif c["unparseable"] or c["schema_invalid"] or c["fenced"]:
            void.append(f"{name}: canary failed {c}")
        if name in lenient and name in arr and lenient[name] > float(arr[name].mean()) + 1e-9:
            void.append(f"{name}: lenient-time score above strict (time pattern not enforced)")
    by = {t["hyp"]: t for t in T if t["hyp"] in ("G1", "G2", "G3")}
    if void:
        headline = "VOID"
    elif by["G3"]["verdict"] in ("schema ahead", "equivalent"):
        headline = "Rewritten"
    elif by["G1"]["verdict"] == "PASS" and by["G2"]["verdict"] == "PASS" and by["G3"]["lo"] > MARGIN:
        headline = "Stands"
    else:
        headline = "Narrowed"
    x_tok = 143.22666666666666  # Part F cost_per_item for 4B+metar (results/analysis/partd_f.json)
    gap = None
    if ball and X in arr:
        a, bb = float(arr[X].mean()), float(arr[ball].mean())
        gap = {"best_schema_arm": ball}
        for ref in ("8B zero-shot (C)", F_BEST_8B, F_32B_THINK):
            if ref in arr:
                z = float(arr[ref].mean())
                gap[f"share_of_gap_from_{ref}_closed"] = (bb - z) / (a - z) if a != z else None
    return {"n": len(ids), "n_stations": len(set(station)), "missing": missing,
            "best": {"8B": b8, "32B": b32, "overall": ball},
            "scores": {k: float(v.mean()) for k, v in arr.items()}, "field_accuracy": field,
            "parse_fail_rate": parse_fail, "lenient_time_all_fields": lenient, "cost_per_item": cost,
            "token_ratio_to_adapter": {k: v["out_tokens"] / x_tok for k, v in cost.items()},
            # prereg: thinking arms averaging under 300 tokens suggest usage excludes reasoning tokens
            "token_counts_unreliable": [k for k, v in cost.items() if "thinking" in k and v["out_tokens"] < 300],
            "truncation_pct": trunc, "canary": canaries, "gap": gap, "tests": T, "headline": headline,
            "void_reasons": void, "parity": parity,
            "rules": {"bootstrap": NB, "rng": "numpy default_rng(0)", "ci": "95% percentile", "margin": MARGIN}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cf-root", default=os.path.join(ROOT, "results"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data_partd_c/eval/aviation/metar/test.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/analysis/gate_metar_schema.json"))
    a = ap.parse_args(argv)
    res = analyse(a.results_root, a.cf_root, a.data)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["scores"].items():
        print(f"{k:40s} {v:6.2f}")
    for t in res["tests"]:
        if "diff" in t:
            print(f"{t['hyp']:10s} {t['x']} - {t['y']}: {t['diff']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] {t['verdict']}")
        else:
            print(f"{t['hyp']:10s} {t['verdict']}")
    print("headline:", res["headline"], "| void:", res["void_reasons"])


if __name__ == "__main__":
    main()

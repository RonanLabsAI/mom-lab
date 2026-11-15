"""Part B confirmatory analysis (frozen with PART_B_PREREG.md before the test pass).

Systems are scored per item on three prompt sets (core six domains, paraphrase, mixed-skill). A library system
routes each item with the frozen router (artifacts/router_e5lr.pkl; --router oracle for the secondary analysis):
a predicted domain in the size's ROUTES goes to that domain's adapter, everything else to the untouched base.
Trained arms have 3 seeds; a system's per-item score is the mean over its seeds (seed k of the library = the seed-k
adapters). Composite per set = mean over domains of the mean over that domain's benchmarks (translate = mean
sentence chrF++), as in scripts/system_compare.py. Differences: paired item bootstrap (2,000 resamples within
benchmark, numpy seed 0), 95% percentile CI.
Verdicts: "beats" PASS if CI low > 0; "non-inferior" PASS if CI low > -2.0; FAIL if CI high < bar; else INCONCLUSIVE.
A hypothesis passes only if every component passes.

usage: .venv/bin/python -P scripts/partb_analysis.py --split dev|test [--router embed-lr|oracle] [--strict]
  dev: results from results/{partb,step6,step4,step3}; test: results/partbtest only (--strict: every seed required).
Writes results/analysis/partb_<split>_<router>.json.
"""
import argparse, json, os, pickle
import numpy as np

D = ["math", "code", "sql", "tools", "translate", "general"]
CORE = {"gsm8k": "math", "math500": "math", "humaneval_plus": "code", "mbpp_plus": "code", "livecodebench": "code",
        "bird_dev": "sql", "bfcl": "tools", "flores200": "translate", "mmlu_pro": "general", "ifeval": "general"}
PARA = {"gsm8k_pp": "math", "bird_pp": "sql", "mbpp_pp": "code", "bfcl_pp": "tools", "mmlupro_pp": "general", "flores_pp": "translate"}
MIX = {"gsm8k_ml": "math", "bird_ml": "sql", "mbpp_ml": "code", "bfcl_ml": "tools"}
SETS = {"core": CORE, "para": PARA, "mixed": MIX}
DIR = {**{b: d for b, d in CORE.items()}, **{b: "para" for b in PARA}, **{b: "mixed" for b in MIX}}
SEEDS = ["", "-s1", "-s2"]
NI, NB = -2.0, 2000

# ---- frozen arms (PART_B_PREREG.md) ----
ROUTES = {"1.7B": ["code", "sql", "tools"], "4B": ["sql", "tools"]}
ARMS = {
    "1.7B": {"base": ["base-qwen3-1.7b"], "pooled": [f"pt17-pooled-lr2e-5{s}" for s in SEEDS],
             "lora": [f"pt17-lora16-{{d}}-lr2e-4{s}" for s in SEEDS], "replay": [f"pt17-replay-{{d}}-r20-lr2e-5{s}" for s in SEEDS],
             "distill": [f"pt17-distill-lr2e-5{s}" for s in SEEDS]},
    "4B": {"base": ["base-qwen3-4b"], "pooled": [f"pt4-pooled-lr2e-5{s}" for s in SEEDS],
           "lora": [f"pt4-lora16-{{d}}-lr2e-4{s}" for s in SEEDS], "distill": [f"pt4-distill-lr2e-5{s}" for s in SEEDS],
           "pooled-Base-init": ["b4-pooled-lr2e-5"], "8B": ["base-qwen3-8b"],
           "awq4": ["base-qwen3-4b-awq"], "awq4+lora": [f"awq4-lora16-{{d}}-lr2e-4{s}" for s in SEEDS], "awq8": ["base-qwen3-8b-awq"],
           "fp8-4": ["base-qwen3-4b-fp8"], "fp8-8": ["base-qwen3-8b-fp8"]},
}
LIB_BASE = {"lora": "base", "replay": "base", "awq4+lora": "awq4"}  # library -> the base it falls back to


def tests(size):
    """(hypothesis, component, system x, system y, set, kind). H1 and H3 get one verdict per size (prereg B, 2026-10-31)."""
    t = []
    for s in SETS:
        t.append((f"H1-{size}", f"{size} lora > base [{s}]", "lora", "base", s, "beats"))
    t.append((f"H1-{size}", f"{size} lora > pooled [core]", "lora", "pooled", "core", "beats"))
    for s in ("para", "mixed"):
        t.append((f"H1-{size}", f"{size} lora >= pooled - 2 [{s}]", "lora", "pooled", s, "ni"))
    t.append((f"H3-{size}", f"{size} lora > distill [core]", "lora", "distill", "core", "beats"))
    for s in ("para", "mixed"):
        t.append((f"H3-{size}", f"{size} lora >= distill - 2 [{s}]", "lora", "distill", s, "ni"))
    if size == "4B":
        for s in SETS:
            t.append(("H2", f"4B lora >= 8B - 2 [{s}]", "lora", "8B", s, "ni"))
            t.append(("H2-4bit", f"AWQ-4B+lora >= AWQ-8B - 2 [{s}]", "awq4+lora", "awq8", s, "ni"))
    if size == "1.7B":  # secondary: full-FT replay library
        for s in SETS:
            t.append(("H1-replay (secondary)", f"1.7B replay > base [{s}]", "replay", "base", s, "beats"))
        t.append(("H1-replay (secondary)", "1.7B replay > pooled [core]", "replay", "pooled", "core", "beats"))
        for s in ("para", "mixed"):
            t.append(("H1-replay (secondary)", f"1.7B replay >= pooled - 2 [{s}]", "replay", "pooled", s, "ni"))
    return t


def load(run, split, need):
    roots = ["partbtest"] if split == "test" else ["partb", "step6", "step4", "step3"]
    for st in roots:
        p = f"results/{st}/{run}/scores.json"
        if os.path.exists(p):
            sc = json.load(open(p))
            if sc.get("split", "dev") != split:
                continue
            if all(b in sc["benchmarks"] for b in SETS[need]):
                return sc["benchmarks"]
    return None


def item_texts(split, bench, ids):
    rows = {}
    sub = DIR[bench]
    path = f"data/eval/{sub}/{bench}/{split}.jsonl" if sub in ("para", "mixed") else \
        next(f"data/eval/{d}/{bench}/{split}.jsonl" for d in D if os.path.exists(f"data/eval/{d}/{bench}/{split}.jsonl"))
    for l in open(path):
        r = json.loads(l); rows[r["id"]] = r
    return [next(m["content"] for m in rows[i]["messages"] if m["role"] == "user") for i in ids], \
        [rows[i].get("meta", {}).get("source_domain", SETS["core"].get(bench) or SETS["para"].get(bench)) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"]); ap.add_argument("--router", default="embed-lr")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    rt = pickle.load(open("artifacts/router_e5lr.pkl", "rb"))
    emb = None; route_memo = {}
    out = {"split": a.split, "router": a.router, "systems": {}, "tests": [], "missing": [], "router_acc": {}}
    rng = np.random.default_rng(0)
    for size, arms in ARMS.items():
        ref = load(arms["base"][0], a.split, "core")
        for sname, bset in SETS.items():
            ref = load(arms["base"][0], a.split, sname)
            if ref is None:
                out["missing"].append(f"{arms['base'][0]} {sname}"); continue
            ids = {b: ref[b]["ids"] for b in bset}
            # route
            route = {}
            for b in bset:
                texts, truth = item_texts(a.split, b, ids[b])
                if a.router == "oracle":
                    route[b] = truth
                elif b in route_memo:
                    route[b] = route_memo[b]
                else:
                    if emb is None:
                        from sentence_transformers import SentenceTransformer
                        emb = SentenceTransformer(rt["emb_model"], device="cpu", revision=rt["emb_revision"]); emb.max_seq_length = 512
                    E = emb.encode(["query: " + t[:4000] for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
                    route[b] = route_memo[b] = list(rt["clf"].predict(E))
                out["router_acc"][f"{size}/{b}"] = float(np.mean([p == t for p, t in zip(route[b], truth)]))

            def per_item(run_names, lib_routes, fallback):
                """Seed-averaged per-item score arrays (bench -> array) or None if nothing available."""
                seeds = []
                for rn in run_names:
                    arrs = {}
                    for b in bset:
                        col = []
                        for i, dom in enumerate(route[b]):
                            r = rn.format(d=dom) if (lib_routes and dom in lib_routes) else (fallback if lib_routes else rn)
                            sc = cache.get(r) if r in cache else cache.setdefault(r, load(r, a.split, sname))
                            if sc is None or sc[b]["ids"] != ids[b]:
                                return_missing.add(r); arrs = None; break
                            col.append(sc[b]["correct"][i])
                        if arrs is None:
                            break
                        arrs[b] = np.array(col, float) * (1 if b.startswith("flores") else 100)
                    if arrs is not None:
                        seeds.append(arrs)
                if not seeds:
                    return None, 0
                return {b: np.mean([s[b] for s in seeds], axis=0) for b in bset}, len(seeds)

            cache = {}; return_missing = set(); sysarr = {}
            for arm, names in arms.items():
                if arm in LIB_BASE:
                    fb = arms[LIB_BASE[arm]][0]
                    arr, n = per_item(names, ROUTES[size], fb)
                else:
                    arr, n = per_item(names, None, None)
                if arr is None:
                    continue
                if a.strict and n != len(names):
                    raise SystemExit(f"--strict: {size} {arm} {sname} has {n}/{len(names)} seeds")
                sysarr[arm] = arr
                out["systems"].setdefault(f"{size} {arm}", {})[sname] = {"score": comp(arr, bset), "seeds": n}
            out["missing"] += sorted(f"{r} ({sname})" for r in return_missing)
            idxs = [{b: rng.integers(0, len(ids[b]), len(ids[b])) for b in bset} for _ in range(NB)]
            for hyp, label, x, y, s, kind in tests(size):
                if s != sname:
                    continue
                if x not in sysarr or y not in sysarr:
                    out["tests"].append({"hyp": hyp, "test": label, "verdict": "MISSING"}); continue
                diff = comp(sysarr[x], bset) - comp(sysarr[y], bset)
                bs = np.array([comp(sysarr[x], bset, i) - comp(sysarr[y], bset, i) for i in idxs])
                lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
                bar = 0.0 if kind == "beats" else NI
                v = "PASS" if lo > bar else "FAIL" if hi < bar else "INCONCLUSIVE"
                out["tests"].append({"hyp": hyp, "test": label, "diff": diff, "lo": lo, "hi": hi, "bar": bar, "verdict": v})
    verdicts = {}
    for t in out["tests"]:
        verdicts.setdefault(t["hyp"], []).append(t["verdict"])
    out["hypotheses"] = {h: ("PASS" if all(v == "PASS" for v in vs) else "MISSING" if "MISSING" in vs else
                             "FAIL" if "FAIL" in vs else "INCONCLUSIVE") for h, vs in verdicts.items()}
    os.makedirs("results/analysis", exist_ok=True)
    p = f"results/analysis/partb_{a.split}_{a.router}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"== Part B analysis: split={a.split} router={a.router}")
    for n, v in out["systems"].items():
        print(f"  {n:22s} " + "  ".join(f"{s} {v[s]['score']:6.2f} (n={v[s]['seeds']})" if s in v else f"{s}   --  " for s in SETS))
    for t in out["tests"]:
        print(f"  {t['hyp']:22s} {t['test']:34s} " + (f"{t['diff']:+6.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] {t['verdict']}" if "diff" in t else t["verdict"]))
    print("  HYPOTHESES:", out["hypotheses"])
    if out["missing"]:
        print("  missing:", sorted(set(out["missing"]))[:40])
    print("wrote", p)


def comp(arr, bset, idx=None):
    ds = {}
    for b, d in bset.items():
        x = arr[b] if idx is None else arr[b][idx[b]]
        ds.setdefault(d, []).append(x.mean())
    return float(np.mean([np.mean(v) for v in ds.values()]))


if __name__ == "__main__":
    main()

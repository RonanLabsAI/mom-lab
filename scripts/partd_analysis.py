"""Part D analysis for sections A, D, E of content/PART_D_PREREG.md (B and C have their own scripts).

Reuses scripts/partb_analysis.py unchanged (imported): benchmark sets, item texts, composite (mean over domains of
the mean over that domain's benchmarks), bars. Same rules as the prereg: seed-averaged per-item scores; paired item
bootstrap (2,000 resamples within benchmark, numpy seed 0), 95% percentile CI; beats = CI low > 0; NI = CI low > -2;
FAIL = CI high < bar; else INCONCLUSIVE. Each test reported on its own.

Sets: core / para / mixed = Part B test sets (data/), para2 = Part C fresh paraphrases (data_partc/eval/para).
Results: new systems from results/partd (core, para, mixed) and results/partd2 (para2); earlier systems from
results/partbtest and results/partc. A run must have split=test and every benchmark of the set.
Library (routed): the frozen router artifacts/router_e5lr.pkl decides per item; 4B routes sql, tools; any other
prediction goes to the system's untouched base. --router oracle uses each item's true domain (secondary).
Always-on (D1): the single adapter answers every item.
Own-domain tests (D2, E1) are single-benchmark comparisons of the routed libraries (router misses fall back to the
base, as everywhere else in Part B/D).

usage: .venv/bin/python -P scripts/partd_analysis.py [--router embed-lr|oracle]  -> results/analysis/partd_<router>.json
"""
import argparse, importlib.util, json, os, pickle
import numpy as np

spec = importlib.util.spec_from_file_location("pb", "scripts/partb_analysis.py")
pb = importlib.util.module_from_spec(spec); spec.loader.exec_module(pb)
SEEDS = pb.SEEDS
SETS = {"core": pb.CORE, "para": pb.PARA, "mixed": pb.MIX, "para2": pb.PARA}
ROOTS = {"core": ["partd", "partbtest"], "para": ["partd", "partbtest"], "mixed": ["partd", "partbtest"],
         "para2": ["partd2", "partc"]}
ROUTES = ["sql", "tools"]

# name -> (run name templates, fallback base or None for single/always-on systems)
SYS = {
    "4B base": (["base-qwen3-4b"], None),
    "FP8-4B base": (["base-qwen3-4b-fp8"], None),
    "AWQ-8B": (["base-qwen3-8b-awq"], None),
    "4B lib r16": ([f"pt4-lora16-{{d}}-lr2e-4{s}" for s in SEEDS], "base-qwen3-4b"),
    "FP8-4B lib r16": ([f"fp84-lora16-{{d}}-lr2e-4{s}" for s in SEEDS], "base-qwen3-4b-fp8"),
    "4B lib r64": ([f"pt4-lora64-{{d}}-lr2e-4{s}" for s in SEEDS], "base-qwen3-4b"),
    "4B lib r128": ([f"pt4-lora128-{{d}}-lr2e-4{s}" for s in SEEDS], "base-qwen3-4b"),
    "4B lib sdreplay": ([f"pt4-lora16-sdreplay-{{d}}-r20-lr2e-4{s}" for s in SEEDS], "base-qwen3-4b"),
}
for d in ROUTES:
    SYS[f"always-on r16 {d}"] = ([f"pt4-lora16-{d}-lr2e-4{s}" for s in SEEDS], None)
    SYS[f"always-on sdreplay {d}"] = ([f"pt4-lora16-sdreplay-{d}-r20-lr2e-4{s}" for s in SEEDS], None)

# (hyp, label, x, y, set, kind, single benchmark or None)
T = []
for s in ("core", "para", "para2", "mixed"):
    T.append(("A1", f"FP8-4B lib >= AWQ-8B - 2 [{s}]", "FP8-4B lib r16", "AWQ-8B", s, "ni", None))
for s in ("core", "para", "para2", "mixed"):
    T.append(("A2", f"FP8-4B lib >= 4B lib - 2 [{s}]", "FP8-4B lib r16", "4B lib r16", s, "ni", None))
for s in ("core", "para2"):
    T.append(("A3", f"FP8-4B lib > FP8-4B base [{s}]", "FP8-4B lib r16", "FP8-4B base", s, "beats", None))
for d in ROUTES:
    for s in ("core", "para2"):
        T.append(("D1", f"always-on sdreplay {d} > always-on r16 {d} [{s}]", f"always-on sdreplay {d}", f"always-on r16 {d}", s, "beats", None))
T.append(("D2", "sdreplay lib >= r16 lib - 2 [core bird_dev]", "4B lib sdreplay", "4B lib r16", "core", "ni", "bird_dev"))
T.append(("D2", "sdreplay lib >= r16 lib - 2 [core bfcl]", "4B lib sdreplay", "4B lib r16", "core", "ni", "bfcl"))
for s in ("core", "para2"):
    T.append(("D3", f"sdreplay lib > r16 lib [{s}]", "4B lib sdreplay", "4B lib r16", s, "beats", None))
for r in ("r64", "r128"):
    for s, b in (("core", "bird_dev"), ("para2", "bird_pp"), ("core", "bfcl"), ("mixed", "bfcl_ml")):
        T.append((f"E1-{r}", f"{r} lib > r16 lib [{s} {b}]", f"4B lib {r}", "4B lib r16", s, "beats", b))
    for s in ("core", "para2"):
        T.append((f"E2-{r}", f"{r} lib > r16 lib [{s}]", f"4B lib {r}", "4B lib r16", s, "beats", None))


def load(run, sname):
    for st in ROOTS[sname]:
        p = f"results/{st}/{run}/scores.json"
        if os.path.exists(p):
            sc = json.load(open(p))
            if sc.get("split") == "test" and all(b in sc["benchmarks"] for b in SETS[sname]):
                return sc["benchmarks"]
    return None


def texts(sname, bench, ids):
    if sname != "para2":
        return pb.item_texts("test", bench, ids)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"data_partc/eval/para/{bench}/test.jsonl")}
    return ([next(m["content"] for m in rows[i]["messages"] if m["role"] == "user") for i in ids],
            [rows[i]["meta"]["source_domain"] for i in ids])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--router", default="embed-lr"); a = ap.parse_args()
    rt = pickle.load(open("artifacts/router_e5lr.pkl", "rb")); emb = None
    out = {"router": a.router, "systems": {}, "tests": [], "missing": [], "router_acc": {}}
    for sname, bset in SETS.items():
        rng = np.random.default_rng(0)
        ref = load("base-qwen3-4b", sname)
        if ref is None:
            raise SystemExit(f"missing base-qwen3-4b {sname}")
        ids = {b: ref[b]["ids"] for b in bset}
        route = {}
        for b in bset:
            t, truth = texts(sname, b, ids[b])
            if a.router == "oracle":
                route[b] = truth
            else:
                if emb is None:
                    from sentence_transformers import SentenceTransformer
                    emb = SentenceTransformer(rt["emb_model"], device="cpu", revision=rt["emb_revision"]); emb.max_seq_length = 512
                E = emb.encode(["query: " + x[:4000] for x in t], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
                route[b] = list(rt["clf"].predict(E))
            out["router_acc"][f"{sname}/{b}"] = float(np.mean([p == q for p, q in zip(route[b], truth)]))
        cache, missing, sysarr = {}, set(), {}

        def per_item(names, fb):
            seeds = []
            for rn in names:
                arrs = {}
                for b in bset:
                    col = []
                    for i, dom in enumerate(route[b]):
                        r = (rn.format(d=dom) if dom in ROUTES else fb) if fb else rn
                        sc = cache[r] if r in cache else cache.setdefault(r, load(r, sname))
                        if sc is None or sc[b]["ids"] != ids[b]:
                            missing.add(r); arrs = None; break
                        col.append(sc[b]["correct"][i])
                    if arrs is None:
                        break
                    arrs[b] = np.array(col, float) * (1 if b.startswith("flores") else 100)
                if arrs is not None:
                    seeds.append(arrs)
            if len(seeds) != len(names):
                return None, len(seeds)
            return {b: np.mean([s[b] for s in seeds], axis=0) for b in bset}, len(seeds)

        for name, (names, fb) in SYS.items():
            arr, n = per_item(names, fb)
            if arr is None:
                continue
            sysarr[name] = arr
            out["systems"].setdefault(name, {})[sname] = {"composite": pb.comp(arr, bset), "seeds": n,
                                                          "bench": {b: float(arr[b].mean()) for b in bset}}
        out["missing"] += sorted(f"{r} ({sname})" for r in missing)
        idxs = [{b: rng.integers(0, len(ids[b]), len(ids[b])) for b in bset} for _ in range(pb.NB)]
        for hyp, label, x, y, s, kind, bench in T:
            if s != sname:
                continue
            if x not in sysarr or y not in sysarr:
                out["tests"].append({"hyp": hyp, "test": label, "verdict": "MISSING"}); continue
            sub = {bench: bset[bench]} if bench else bset
            diff = pb.comp(sysarr[x], sub) - pb.comp(sysarr[y], sub)
            bs = np.array([pb.comp(sysarr[x], sub, i) - pb.comp(sysarr[y], sub, i) for i in idxs])
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            bar = 0.0 if kind == "beats" else pb.NI
            v = "PASS" if lo > bar else "FAIL" if hi < bar else "INCONCLUSIVE"
            out["tests"].append({"hyp": hyp, "test": label, "diff": diff, "lo": float(lo), "hi": float(hi), "bar": bar, "verdict": v})
    os.makedirs("results/analysis", exist_ok=True)
    p = f"results/analysis/partd_{a.router}.json"; json.dump(out, open(p, "w"), indent=1)
    print(f"== Part D A/D/E analysis router={a.router}")
    for n, v in out["systems"].items():
        print(f"  {n:26s} " + "  ".join(f"{s} {v[s]['composite']:6.2f}" if s in v else f"{s}   --  " for s in SETS))
    for t in out["tests"]:
        print(f"  {t['hyp']:8s} {t['test']:52s} " + (f"{t['diff']:+6.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] {t['verdict']}" if "diff" in t else t["verdict"]))
    if out["missing"]:
        print("  missing:", sorted(set(out["missing"]))[:40])
    print("wrote", p)


if __name__ == "__main__":
    main()

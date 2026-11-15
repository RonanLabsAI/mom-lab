"""Part C analysis (PART_C_PREREG.md): every paraphrase component of the Part B hypotheses, re-tested on the fresh
paraphrase set data_partc/eval/para (scripts/build_para2.py), plus the Part C combined verdicts.

Reuses scripts/partb_analysis.py unchanged (imported): same arms, routes, frozen router, seed averaging, composite,
paired item bootstrap (2,000 resamples, numpy seed 0), bars and verdict rule. Only the data root (data_partc), the
results root (results/partc) and the set (para only) differ.
Combined verdict per Part B hypothesis: its non-paraphrase components keep their Part B test verdicts
(results/analysis/partb_test_<router>.json); its paraphrase components take the fresh-set verdicts. PASS only if all
pass; FAIL if any fails; else INCONCLUSIVE.
Secondary (exploratory): 4B LoRA+replay library (Part A adapters, seed 0 only) vs pooled and vs the plain library.

usage: .venv/bin/python -P scripts/partc_analysis.py [--router embed-lr|oracle] [--strict]
Writes results/analysis/partc_<router>.json.
"""
import argparse, importlib.util, json, os, pickle
import numpy as np

spec = importlib.util.spec_from_file_location("pb", "scripts/partb_analysis.py")
pb = importlib.util.module_from_spec(spec); spec.loader.exec_module(pb)
BSET = pb.PARA
ARMS = {k: dict(v) for k, v in pb.ARMS.items()}
ARMS["4B"]["lora-replay"] = ["pt4-lora16-replay-{d}-r20-lr2e-4"]  # secondary, seed 0 only
LIB_BASE = {**pb.LIB_BASE, "lora-replay": "base"}
SECONDARY = [("C-sec 4B lora-replay", "4B lora-replay >= pooled - 2 [para2]", "lora-replay", "pooled", "ni"),
             ("C-sec 4B lora-replay", "4B lora-replay vs lora [para2]", "lora-replay", "lora", "beats")]


def load(run):
    p = f"results/partc/{run}/scores.json"
    if not os.path.exists(p):
        return None
    sc = json.load(open(p))
    return sc["benchmarks"] if sc.get("split") == "test" and all(b in sc["benchmarks"] for b in BSET) else None


def texts(bench, ids):
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"data_partc/eval/para/{bench}/test.jsonl")}
    return ([next(m["content"] for m in rows[i]["messages"] if m["role"] == "user") for i in ids],
            [rows[i]["meta"]["source_domain"] for i in ids])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--router", default="embed-lr"); ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    rt = pickle.load(open("artifacts/router_e5lr.pkl", "rb")); emb = None
    rng = np.random.default_rng(0)
    out = {"router": a.router, "systems": {}, "tests": [], "missing": [], "router_acc": {}}
    route = {}
    for size, arms in ARMS.items():
        ref = load(arms["base"][0])
        if ref is None:
            raise SystemExit(f"missing {arms['base'][0]}")
        ids = {b: ref[b]["ids"] for b in BSET}
        assert all(ids[b] == out.setdefault("_ids", ids)[b] for b in BSET), "item lists differ between sizes"
        for b in BSET:
            t, truth = texts(b, ids[b])
            if b not in route:
                if a.router == "oracle":
                    route[b] = truth
                else:
                    if emb is None:
                        from sentence_transformers import SentenceTransformer
                        emb = SentenceTransformer(rt["emb_model"], device="cpu", revision=rt["emb_revision"]); emb.max_seq_length = 512
                    E = emb.encode(["query: " + x[:4000] for x in t], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
                    route[b] = list(rt["clf"].predict(E))
            out["router_acc"][b] = float(np.mean([p == q for p, q in zip(route[b], truth)]))
        cache = {}; missing = set(); sysarr = {}

        def per_item(names, routes, fb):
            seeds = []
            for rn in names:
                arrs = {}
                for b in BSET:
                    col = []
                    for i, dom in enumerate(route[b]):
                        r = rn.format(d=dom) if (routes and dom in routes) else (fb if routes else rn)
                        sc = cache[r] if r in cache else cache.setdefault(r, load(r))
                        if sc is None or sc[b]["ids"] != ids[b]:
                            missing.add(r); arrs = None; break
                        col.append(sc[b]["correct"][i])
                    if arrs is None:
                        break
                    arrs[b] = np.array(col, float) * (1 if b.startswith("flores") else 100)
                if arrs is not None:
                    seeds.append(arrs)
            return ({b: np.mean([s[b] for s in seeds], axis=0) for b in BSET}, len(seeds)) if seeds else (None, 0)

        for arm, names in arms.items():
            arr, n = per_item(names, pb.ROUTES[size], arms[LIB_BASE[arm]][0]) if arm in LIB_BASE else per_item(names, None, None)
            if arr is None:
                continue
            if a.strict and n != len(names):
                raise SystemExit(f"--strict: {size} {arm} has {n}/{len(names)} seeds")
            sysarr[arm] = arr
            out["systems"][f"{size} {arm}"] = {"para2": pb.comp(arr, BSET), "seeds": n}
        out["missing"] += sorted(missing)
        idxs = [{b: rng.integers(0, len(ids[b]), len(ids[b])) for b in BSET} for _ in range(pb.NB)]
        comps = [(h, l.replace("[para]", "[para2]"), x, y, k) for h, l, x, y, s, k in pb.tests(size) if s == "para"]
        if size == "4B":
            comps += SECONDARY
        for hyp, label, x, y, kind in comps:
            if x not in sysarr or y not in sysarr:
                out["tests"].append({"hyp": hyp, "test": label, "verdict": "MISSING"}); continue
            diff = pb.comp(sysarr[x], BSET) - pb.comp(sysarr[y], BSET)
            bs = np.array([pb.comp(sysarr[x], BSET, i) - pb.comp(sysarr[y], BSET, i) for i in idxs])
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            bar = 0.0 if kind == "beats" else pb.NI
            v = "PASS" if lo > bar else "FAIL" if hi < bar else "INCONCLUSIVE"
            out["tests"].append({"hyp": hyp, "test": label, "diff": diff, "lo": lo, "hi": hi, "bar": bar, "verdict": v})
    # combined verdicts
    out.pop("_ids", None)
    b = json.load(open(f"results/analysis/partb_test_{a.router}.json"))
    comb = {}
    for t in b["tests"]:
        if "[para]" not in t["test"]:
            comb.setdefault(t["hyp"], []).append(t["verdict"])
    for t in out["tests"]:
        if not t["hyp"].startswith("C-sec"):
            comb.setdefault(t["hyp"], []).append(t["verdict"])
    out["combined"] = {h: ("PASS" if all(v == "PASS" for v in vs) else "MISSING" if "MISSING" in vs else
                           "FAIL" if "FAIL" in vs else "INCONCLUSIVE") for h, vs in comb.items()}
    os.makedirs("results/analysis", exist_ok=True)
    p = f"results/analysis/partc_{a.router}.json"; json.dump(out, open(p, "w"), indent=1)
    print(f"== Part C analysis (fresh paraphrase set): router={a.router}")
    print("  n items:", {b: len(ids[b]) for b in BSET}, " router acc:", {k: round(v, 3) for k, v in out["router_acc"].items()})
    for n, v in out["systems"].items():
        print(f"  {n:22s} para2 {v['para2']:6.2f} (n={v['seeds']})")
    for t in out["tests"]:
        print(f"  {t['hyp']:22s} {t['test']:36s} " + (f"{t['diff']:+6.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] {t['verdict']}" if "diff" in t else t["verdict"]))
    print("  COMBINED (Part B non-para components + fresh para):", out["combined"])
    if out["missing"]:
        print("  missing:", out["missing"][:40])
    print("wrote", p)


if __name__ == "__main__":
    main()

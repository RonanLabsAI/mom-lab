"""Part D addendum B analysis: fresh SQL (Spider, scored as bird_dev) and fresh tools (BFCL live, scored as bfcl).
Rules from content/PART_D_PREREG.md + PART_D_B_ADDENDUM.md: seed-averaged per-item scores; paired item bootstrap
(2,000 resamples, numpy seed 0), 95% percentile CI; beats = CI low > 0; NI = CI low > -2; FAIL = CI high < bar.
Libraries: the frozen router (artifacts/router_e5lr.pkl) decides per item; items it does not send to the adapter's
domain are answered by that system's untouched base. --router oracle sends every item to the adapter.
usage: .venv/bin/python -P scripts/partd_b_analysis.py [--router embed-lr|oracle]   -> results/analysis/partd_b_<router>.json
"""
import argparse, json, os, pickle
import numpy as np

R = "results/partdB"; NB = 2000; NI = -2.0
S3 = ["", "-s1", "-s2"]
BENCH = {"sql": "bird_dev", "tools": "bfcl"}


def load(run, bench):
    p = f"{R}/{run}/scores.json"
    if not os.path.exists(p):
        return None
    b = json.load(open(p))["benchmarks"].get(bench)
    return b if b and b.get("n_missing", 0) == 0 else None


def single(runs, bench, ids):
    cols = []
    for r in runs:
        s = load(r, bench)
        if s is None or s["ids"] != ids:
            return None
        cols.append(np.array(s["correct"], float) * 100)
    return np.mean(cols, axis=0)


def library(adapter_runs, base_run, bench, ids, routed):
    base = single([base_run], bench, ids)
    cols = []
    for r in adapter_runs:
        a = single([r], bench, ids)
        if a is None or base is None:
            return None
        cols.append(np.where(routed, a, base))
    return np.mean(cols, axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--router", default="embed-lr"); a = ap.parse_args()
    ref = {d: load("base-qwen3-4b", b) for d, b in BENCH.items()}
    ids = {d: ref[d]["ids"] for d in BENCH}
    rows = {d: {json.loads(l)["id"]: json.loads(l) for l in open(f"data_partd/eval/{d}/{BENCH[d]}/test.jsonl")} for d in BENCH}
    routed, acc = {}, {}
    if a.router == "oracle":
        routed = {d: np.ones(len(ids[d]), bool) for d in BENCH}
    else:
        rt = pickle.load(open("artifacts/router_e5lr.pkl", "rb"))
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(rt["emb_model"], device="cpu", revision=rt["emb_revision"]); emb.max_seq_length = 512
        for d in BENCH:
            t = [next(m["content"] for m in rows[d][i]["messages"] if m["role"] == "user") for i in ids[d]]
            E = emb.encode(["query: " + x[:4000] for x in t], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            routed[d] = np.array([p == d for p in rt["clf"].predict(E)])
    acc = {d: float(routed[d].mean()) for d in BENCH}
    sysd = {}
    for d, b in BENCH.items():
        sy = {}
        for n in ["base-qwen3-1.7b", "base-qwen3-4b", "base-qwen3-8b", "base-qwen3-8b-awq", "base-qwen3-8b-fp8", "base-qwen3-4b-fp8"]:
            sy[n] = single([n], b, ids[d])
        sy["4B lib"] = library([f"pt4-lora16-{d}-lr2e-4{s}" for s in S3], "base-qwen3-4b", b, ids[d], routed[d])
        sy["FP8-4B lib"] = library([f"fp84-lora16-{d}-lr2e-4{s}" for s in S3], "base-qwen3-4b-fp8", b, ids[d], routed[d])
        sy["1.7B lib (LoRA)"] = library([f"pt17-lora16-{d}-lr2e-4{s}" for s in S3], "base-qwen3-1.7b", b, ids[d], routed[d])
        if d == "sql":
            sy["1.7B replay specialist"] = library([f"pt17-replay-sql-r20-lr2e-5{s}" for s in S3], "base-qwen3-1.7b", b, ids[d], routed[d])
        sy["4B distill"] = single([f"pt4-distill-lr2e-5{s}" for s in S3], b, ids[d])
        sy["4B pooled"] = single([f"pt4-pooled-lr2e-5{s}" for s in S3], b, ids[d])
        for tag in ["lora64", "lora128", "lora16-sdreplay"]:  # secondary, when present
            suf = "-r20" if "sdreplay" in tag else ""
            sy[f"4B lib {tag}"] = library([f"pt4-{tag}-{d}{suf}-lr2e-4{s}" for s in S3], "base-qwen3-4b", b, ids[d], routed[d])
        sysd[d] = sy
    rng = np.random.default_rng(0)
    idx = {d: [rng.integers(0, len(ids[d]), len(ids[d])) for _ in range(NB)] for d in BENCH}
    T = [("B1", "sql", "4B lib", "base-qwen3-8b", "beats"), ("B2", "tools", "4B lib", "base-qwen3-8b", "ni"),
         ("B3", "sql", "4B lib", "base-qwen3-8b-awq", "beats"), ("B4", "sql", "1.7B replay specialist", "base-qwen3-4b", "ni"),
         ("B5", "sql", "4B lib", "base-qwen3-4b", "beats"), ("B5", "tools", "4B lib", "base-qwen3-4b", "beats"),
         ("B6", "sql", "4B distill", "base-qwen3-8b", "beats"), ("B7", "sql", "FP8-4B lib", "base-qwen3-8b-awq", "beats"),
         ("sec", "sql", "4B lib lora64", "4B lib", "beats"), ("sec", "sql", "4B lib lora128", "4B lib", "beats"),
         ("sec", "sql", "4B lib lora16-sdreplay", "4B lib", "ni"), ("sec", "tools", "4B lib lora16-sdreplay", "4B lib", "ni"),
         ("sec", "tools", "FP8-4B lib", "base-qwen3-8b-awq", "ni"), ("sec", "sql", "1.7B lib (LoRA)", "base-qwen3-4b", "ni")]
    out = {"router": a.router, "router_acc": acc, "n": {d: len(ids[d]) for d in BENCH},
           "scores": {d: {n: (float(v.mean()) if v is not None else None) for n, v in sy.items()} for d, sy in sysd.items()}, "tests": []}
    for h, d, x, y, k in T:
        X, Y = sysd[d].get(x), sysd[d].get(y)
        if X is None or Y is None:
            out["tests"].append({"hyp": h, "set": d, "x": x, "y": y, "verdict": "MISSING"}); continue
        diff = float(X.mean() - Y.mean()); bs = np.array([X[i].mean() - Y[i].mean() for i in idx[d]])
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5); bar = 0.0 if k == "beats" else NI
        v = "PASS" if lo > bar else "FAIL" if hi < bar else "INCONCLUSIVE"
        out["tests"].append({"hyp": h, "set": d, "x": x, "y": y, "kind": k, "diff": diff, "lo": float(lo), "hi": float(hi), "verdict": v})
    os.makedirs("results/analysis", exist_ok=True)
    p = f"results/analysis/partd_b_{a.router}.json"; json.dump(out, open(p, "w"), indent=1)
    print(f"== Part D B (fresh SQL/tools) router={a.router} n={out['n']} router_acc={ {k: round(v, 3) for k, v in acc.items()} }")
    for d in BENCH:
        print(f" {d}: " + ", ".join(f"{n} {v:.1f}" for n, v in out["scores"][d].items() if v is not None))
    for t in out["tests"]:
        print(f"  {t['hyp']:4s} {t['set']:5s} {t['x']:24s} vs {t['y']:20s} " + (f"{t['diff']:+6.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] {t['verdict']}" if "diff" in t else t["verdict"]))
    print("wrote", p)


if __name__ == "__main__":
    main()

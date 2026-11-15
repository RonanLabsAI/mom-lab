"""Honest estimate for 'base + specialists': choose each domain's route on one random half of the dev items,
score on the other half; swap; repeat over 200 random splits. Reports mean composite and the share of splits
where each domain was routed to its specialist.
usage: python3 scripts/cv_routes.py <prefix> <lr> [--base base-qwen3-1.7b] [--pooled NAME] [--spec TEMPLATE]
  --spec: specialist run-name template with {pre}, {d}, {lr} (default {pre}-spec-{d}-lr{lr}), e.g. {pre}-lora16-{d}-lr2e-4.
  Results are looked up in results/*/<run>/scores.json; a domain without a specialist result always routes to the base.
"""
import glob, json, os, sys, numpy as np
DOMAINS = ["math", "code", "sql", "tools", "translate", "general"]
pre, lr = sys.argv[1], sys.argv[2]
opt = lambda k, v: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else v
base = opt("--base", "base-qwen3-1.7b"); spec = opt("--spec", "{pre}-spec-{d}-lr{lr}")


def find(r):
    hits = sorted(glob.glob(f"results/*/{r}/scores.json"))
    return hits[0] if hits else None


L = lambda r: json.load(open(find(r)))
B = L(base); P = L(opt("--pooled", f"{pre}-pooled-lr{lr}"))
S = {d: L(spec.format(pre=pre, d=d, lr=lr)) for d in DOMAINS if find(spec.format(pre=pre, d=d, lr=lr))}
print("specialists found:", {d: spec.format(pre=pre, d=d, lr=lr) for d in S}, "; others route to base")


def arr(sc, b):
    return np.array(sc["benchmarks"][b]["correct"], dtype=float)


def dscore(sc, d, idx):
    vals = []
    for b, v in sc["benchmarks"].items():
        if v["domain"] == d:
            c = arr(sc, b)[idx[b]]
            vals.append(c.mean() if b == "flores200" else 100 * c.mean())
    return float(np.mean(vals))


rng = np.random.default_rng(0)
res, picks = [], {d: 0 for d in DOMAINS}
bench = {b: len(v["correct"]) for b, v in B["benchmarks"].items()}
for _ in range(200):
    half = {b: rng.permutation(n) for b, n in bench.items()}
    folds = [{b: p[: n // 2] for b, (p, n) in zip(half, zip(half.values(), bench.values()))},
             {b: p[n // 2:] for b, (p, n) in zip(half, zip(half.values(), bench.values()))}]
    for sel, ev in [(0, 1), (1, 0)]:
        A = []
        for d in DOMAINS:
            use_spec = d in S and dscore(S[d], d, folds[sel]) > dscore(B, d, folds[sel])
            picks[d] += use_spec
            A.append(dscore(S[d] if use_spec else B, d, folds[ev]))
        res.append((np.mean(A), np.mean([dscore(B, d, folds[ev]) for d in DOMAINS]),
                    np.mean([dscore(P, d, folds[ev]) for d in DOMAINS])))
res = np.array(res)
print(f"{pre} lr {lr}: cross-validated base+specialists A={res[:,0].mean():.2f}  base={res[:,1].mean():.2f}  pooled={res[:,2].mean():.2f}")
print(f"  gain vs base {np.mean(res[:,0]-res[:,1]):+.2f} (5-95%: {np.percentile(res[:,0]-res[:,1],5):+.2f}, {np.percentile(res[:,0]-res[:,1],95):+.2f});"
      f"  vs pooled {np.mean(res[:,0]-res[:,2]):+.2f}")
print("  share of splits routing domain to its specialist:", {d: round(v / len(res), 2) for d, v in picks.items()})

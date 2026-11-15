"""Build the Part A training files with equal token budgets.

For each domain: a seeded random sample of the decontaminated pool (stratified by source, proportional)
until the token budget N is reached. Then:
- spec/<domain>.jsonl      : the domain's sample (specialist training set)
- pooled.jsonl             : exact union of the six samples, shuffled (pooled / tagged-pooled)
- randpart/part<k>.jsonl   : the same union split into 6 random parts of ~equal tokens (random-partition control)
Writes mixtures/<name>/manifest.json with counts, tokens and per-source shares.
"""
import json, random, os, sys, collections
BUDGET = int(float(sys.argv[1])) if len(sys.argv) > 1 else 14_000_000
NAME = sys.argv[2] if len(sys.argv) > 2 else f"N{BUDGET // 1_000_000}M"
SEED = 20261029
DOMAINS = ["math", "code", "sql", "tools", "translate", "general"]
root = os.path.join("data", "mixtures", NAME)
os.makedirs(os.path.join(root, "spec"), exist_ok=True)
os.makedirs(os.path.join(root, "randpart"), exist_ok=True)
rng = random.Random(SEED)
man = {"budget_tokens": BUDGET, "seed": SEED, "domains": {}}
union = []
for d in DOMAINS:
    pool = [json.loads(l) for l in open(f"data/train/{d}/pool.jsonl")]
    by_src = collections.defaultdict(list)
    for ex in pool:
        by_src[ex.get("source", "?")].append(ex)
    total = sum(ex["n_tokens"] for ex in pool)
    picked, tok = [], 0
    if total <= BUDGET:
        picked, tok = pool[:], total
    else:
        # proportional per-source quotas, then fill in random order within each source
        for s, exs in by_src.items():
            rng.shuffle(exs)
            quota = BUDGET * sum(e["n_tokens"] for e in exs) / total
            t = 0
            for e in exs:
                if t >= quota:
                    break
                picked.append(e); t += e["n_tokens"]
            tok += t
    for e in picked:
        e["domain"] = d
    rng.shuffle(picked)
    with open(os.path.join(root, "spec", f"{d}.jsonl"), "w") as f:
        for e in picked:
            f.write(json.dumps(e) + "\n")
    srcs = collections.Counter(e.get("source", "?") for e in picked)
    man["domains"][d] = {"examples": len(picked), "tokens": tok, "pool_tokens": total,
                         "sources": dict(srcs), "under_budget": total < BUDGET}
    union += picked
    print(d, len(picked), tok, "(pool", total, ")", flush=True)
rng.shuffle(union)
with open(os.path.join(root, "pooled.jsonl"), "w") as f:
    for e in union:
        f.write(json.dumps(e) + "\n")
# random partitions: 6 parts of ~equal tokens, domain-mixed
parts = [[] for _ in range(6)]; ptok = [0] * 6
for e in union:
    k = min(range(6), key=lambda i: ptok[i]); parts[k].append(e); ptok[k] += e["n_tokens"]
for k, p in enumerate(parts):
    with open(os.path.join(root, "randpart", f"part{k}.jsonl"), "w") as f:
        for e in p:
            f.write(json.dumps(e) + "\n")
man["pooled"] = {"examples": len(union), "tokens": sum(e["n_tokens"] for e in union)}
man["randpart_tokens"] = ptok
json.dump(man, open(os.path.join(root, "manifest.json"), "w"), indent=1)
print("pooled", man["pooled"], "randpart", ptok)

"""Step 4: does routing keep the mixture's gain? Offline: route every dev item (six domains + mixed-skill set),
score with the routed model's saved per-item correctness from results/step3/<run>/scores.json.

Routers (all predict one of six domains; a library maps domain -> model):
  oracle    true domain (mixed items: source_domain)
  rules     hand-written format rules on the prompt text (no training)
  embed-lr  multilingual-e5-small embeddings + logistic regression trained ONLY on training-mixture prompts
            (data/mixtures/N14M/spec/<d>.jsonl, N_TRAIN per domain); never sees eval prompts or their templates
  random    uniform over six domains (mean of 20 draws)
Libraries:  specialists only  /  base + specialists (domain -> base unless listed in ROUTE_SPEC)
usage: python -m scripts.route_step4 <prefix> <lr> [--spec-domains code,sql,tools]   (run from mom-lab/)
"""
import json, os, re, sys, random
import numpy as np
DOMAINS = ["math", "code", "sql", "tools", "translate", "general"]
pre, lr = sys.argv[1], sys.argv[2]
ROUTE_SPEC = (sys.argv[sys.argv.index("--spec-domains") + 1].split(",") if "--spec-domains" in sys.argv
              else ["code", "sql", "tools"])
BASE = "base-qwen3-1.7b"; N_TRAIN = 2000; SEED = 20261029
PARA = "--para" in sys.argv
R = "results/step4" if PARA else "results/step3"; CACHE = "logs/step4"; os.makedirs(CACHE, exist_ok=True)
EMB = "intfloat/multilingual-e5-small"


def user_text(msgs):
    return next((m["content"] for m in msgs if m["role"] == "user"), "")


# ---- dev items, in scores.json order ----
def dev_items():
    items = {}  # bench -> list of (id, text, true_domain)
    for d in (["para"] if PARA else DOMAINS + ["mixed"]):
        for b in sorted(os.listdir(f"data/eval/{d}")):
            p = f"data/eval/{d}/{b}/dev.jsonl"
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p)]
            items[b] = {r["id"]: (user_text(r["messages"]), r.get("meta", {}).get("source_domain", d)) for r in rows}
    return items


def load(run):
    bm = json.load(open(f"{R}/{run}/scores.json"))["benchmarks"]
    if PARA:  # score each paraphrase bench under its source domain; no mixed set here
        src = {"gsm8k_pp": "math", "bird_pp": "sql", "mbpp_pp": "code", "bfcl_pp": "tools", "mmlupro_pp": "general", "flores_pp": "translate"}
        bm = {b: dict(v, domain=src[b]) for b, v in bm.items() if b in src}
    return bm


# ---- routers ----
RULES = [  # first match wins; ordered from most to least specific surface cue
    ("sql", re.compile(r"CREATE TABLE|SQLite|\bSELECT\b.*\bFROM\b|```sql", re.S | re.I)),
    ("tools", re.compile(r'"parameters"\s*:|function[s]? (call|calling)|"name"\s*:\s*"[\w.]+"\s*,\s*"description"', re.I)),
    ("translate", re.compile(r"\btranslate\b|traduc|übersetz|翻译", re.I)),
    ("code", re.compile(r"```python|\bdef \w+\(|\bpython (function|program|script)|assert \w+\(", re.I)),
    ("math", re.compile(r"\\boxed|\\frac|\$[^$]+\$|how many|how much|calculate|compute", re.I)),
]


def route_rules(t):
    for d, rx in RULES:
        if rx.search(t):
            return d
    return "general"


def train_texts():
    rng = random.Random(SEED); X, y = [], []
    for d in DOMAINS:
        lines = open(f"data/mixtures/N14M/spec/{d}.jsonl").readlines()
        for l in rng.sample(lines, min(N_TRAIN, len(lines))):
            X.append(user_text(json.loads(l)["messages"])); y.append(d)
    return X, y


def embed(texts, tag):
    p = f"{CACHE}/emb-{tag}.npy"
    if os.path.exists(p):
        e = np.load(p)
        if len(e) == len(texts):
            return e
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB, device="cpu"); m.max_seq_length = 512
    e = m.encode(["query: " + t[:4000] for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(p, e); return e


# ---- scoring ----
def score(route, models, bm, items, lib):
    """route: bench -> list of domains (aligned to scores ids); lib: domain -> run. Returns (A, per-domain, mixed)."""
    dom = {d: [] for d in DOMAINS}; mixed = []
    for b, v in bm.items():
        corr = np.array([np.array(models[lib[r]][b]["correct"], dtype=float)[i] for i, r in enumerate(route[b])])
        val = corr.mean() if b in ("flores200", "flores_pp") else 100 * corr.mean()
        (mixed if v["domain"] == "mixed" else dom[v["domain"]]).append(val)
    ds = {d: float(np.mean(x)) for d, x in dom.items()}
    return float(np.mean(list(ds.values()))), ds, float(np.mean(mixed)) if mixed else float("nan")


items = dev_items()
spec = {d: f"{pre}-spec-{d}-lr{lr}" for d in DOMAINS}
runs = set(spec.values()) | {BASE, f"{pre}-pooled-lr{lr}"}
models = {r: load(r) for r in runs}
bm = models[BASE]
ids = {b: v["ids"] for b, v in bm.items()}
for r in runs:  # all models must share item order
    assert all(models[r][b]["ids"] == ids[b] for b in ids), r
texts = {b: [items[b][i][0] for i in ids[b]] for b in ids}
truth = {b: [items[b][i][1] for i in ids[b]] for b in ids}

routers = {"oracle": truth, "rules": {b: [route_rules(t) for t in texts[b]] for b in ids}}
Xtr, ytr = train_texts()
Etr = embed(Xtr, f"train{N_TRAIN}")
flat = [t for b in ids for t in texts[b]]
Edev = embed(flat, "para" if PARA else "dev")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=3000, C=4.0).fit(Etr, ytr)
pred = list(clf.predict(Edev)); k = 0; routers["embed-lr"] = {}
for b in ids:
    routers["embed-lr"][b] = pred[k:k + len(ids[b])]; k += len(ids[b])

libs = {"specialists": spec, "base+specialists": {d: (spec[d] if d in ROUTE_SPEC else BASE) for d in DOMAINS}}
print(f"== Step 4 routing, {pre} lr{lr}; base+specialists routes {ROUTE_SPEC} to specialists")
print("Router accuracy vs true domain (core six / mixed-skill set):")
for name, rt in list(routers.items())[1:]:
    core = [p == t for b in ids if bm[b]["domain"] != "mixed" for p, t in zip(rt[b], truth[b])]
    mix = [p == t for b in ids if bm[b]["domain"] == "mixed" for p, t in zip(rt[b], truth[b])]
    print(f"  {name:9s} core {100*np.mean(core):5.1f}%   mixed {100*np.mean(mix):5.1f}%")
for b in ids:
    if bm[b]["domain"] == "mixed":
        for name in ("rules", "embed-lr"):
            from collections import Counter
            print(f"    {name:8s} {b:9s} -> {dict(Counter(routers[name][b]).most_common(3))}")
print("\nSystem scores (A = six-domain composite; mixed = mixed-skill set accuracy):")
for r, lab in ((BASE, "base"), (f"{pre}-pooled-lr{lr}", "pooled")):
    A, ds, mx = score({b: [DOMAINS[0]] * len(ids[b]) for b in ids}, models, bm, items, {d: r for d in DOMAINS})
    print(f"  {lab:34s} A={A:6.2f}  mixed={mx:5.1f}")
rng = np.random.default_rng(SEED)
for ln, lib in libs.items():
    for name, rt in routers.items():
        A, ds, mx = score(rt, models, bm, items, lib)
        print(f"  {ln + ' / ' + name:34s} A={A:6.2f}  mixed={mx:5.1f}  " + " ".join(f"{d}={ds[d]:.1f}" for d in DOMAINS))
    res = [score({b: list(rng.choice(DOMAINS, len(ids[b]))) for b in ids}, models, bm, items, lib) for _ in range(20)]
    print(f"  {ln + ' / random (20 draws)':34s} A={np.mean([x[0] for x in res]):6.2f}  mixed={np.mean([x[2] for x in res]):5.1f}")

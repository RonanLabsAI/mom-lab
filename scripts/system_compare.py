"""Step 6 system comparison on three prompt sets (core dev, paraphrase, mixed-skill), oracle domain routing.
A system = per-domain choice of model; base+X routes ROUTED domains to X's specialist and the rest to the untouched
post-trained base. Paired item bootstrap (2,000 resamples within benchmark) for differences vs a reference system.
Composite per set = mean over its domains (translate = mean sentence chrF++). Core reads results/step3 (step6 for
Step 6 runs); para reads results/step4 (step6); mixed reads the mixed benches of the core results.
usage: python -P scripts/system_compare.py
"""
import json, os
import numpy as np
D = ["math", "code", "sql", "tools", "translate", "general"]
CORE = {"gsm8k": "math", "math500": "math", "humaneval_plus": "code", "mbpp_plus": "code", "livecodebench": "code",
        "bird_dev": "sql", "bfcl": "tools", "flores200": "translate", "mmlu_pro": "general", "ifeval": "general"}
PARA = {"gsm8k_pp": "math", "bird_pp": "sql", "mbpp_pp": "code", "bfcl_pp": "tools", "mmlupro_pp": "general", "flores_pp": "translate"}
MIX = {"gsm8k_ml": "math", "bird_ml": "sql", "mbpp_ml": "code", "bfcl_ml": "tools"}
ROUTED = ["code", "sql", "tools"]; BASE = "base-qwen3-1.7b"


def find(run, need):
    for st in ("step6", "step4", "step3") if need == "para" else ("step6", "step3"):
        p = f"results/{st}/{run}/scores.json"
        if os.path.exists(p):
            b = json.load(open(p))["benchmarks"]
            if all(k in b for k in (PARA if need == "para" else CORE if need == "core" else MIX)):
                return b
    raise FileNotFoundError(f"{run} {need}")


def arrays(system, bset, need):
    """system: domain -> run. Returns bench -> per-item array."""
    cache = {}
    out = {}
    for b, d in bset.items():
        r = system[d]
        if r not in cache:
            cache[r] = find(r, need)
        out[b] = np.array(cache[r][b]["correct"], float) * (1 if b.startswith("flores") else 100)
    return out


def comp(arr, bset, idx=None):
    ds = {}
    for b, d in bset.items():
        a = arr[b] if idx is None else arr[b][idx[b]]
        ds.setdefault(d, []).append(a.mean())
    return float(np.mean([np.mean(v) for v in ds.values()]))


def system(kind):
    if kind == "base":
        return {d: BASE for d in D}
    if kind.startswith("pooled:"):
        return {d: kind.split(":", 1)[1] for d in D}
    fmt = kind.split(":", 1)[1]  # e.g. b17-replay-{d}-r20-lr2e-5
    return {d: (fmt.format(d=d) if d in ROUTED else BASE) for d in D}


SYSTEMS = {
    "base": "base",
    "pooled (Base init)": "pooled:b17-pooled-lr2e-5",
    "pooled (post-trained init)": "pooled:pt17-pooled-lr2e-5",
    "b+s plain (Base init)": "bs:b17-spec-{d}-lr2e-5",
    "b+s replay (Base init)": "bs:b17-replay-{d}-r20-lr2e-5",
    "b+s plain (post-trained init)": "bs:pt17-spec-{d}-lr2e-5",
    "b+s replay (post-trained init)": "bs:pt17-replay-{d}-r20-lr2e-5",
    "b+s LoRA r16 (post-trained init)": "bs:pt17-lora16-{d}-lr2e-4",
}
rng = np.random.default_rng(0)
sets = {"core": CORE, "para": PARA, "mixed": MIX}
A = {s: {n: arrays(system(k), bs, s) for n, k in SYSTEMS.items()} for s, bs in sets.items()}
print(f"{'system':32s}" + "".join(f"{s:>10s}" for s in sets))
for n in SYSTEMS:
    print(f"{n:32s}" + "".join(f"{comp(A[s][n], bs):10.2f}" for s, bs in sets.items()))
print("\nPaired differences, 95% CI (2,000 item resamples):")
pairs = [("b+s replay (post-trained init)", "pooled (post-trained init)"), ("b+s replay (post-trained init)", "base"),
         ("b+s replay (Base init)", "pooled (Base init)"), ("b+s replay (Base init)", "base"),
         ("b+s replay (post-trained init)", "b+s plain (post-trained init)"), ("b+s replay (Base init)", "b+s plain (Base init)"),
         ("b+s LoRA r16 (post-trained init)", "base"), ("b+s LoRA r16 (post-trained init)", "pooled (post-trained init)"),
         ("b+s LoRA r16 (post-trained init)", "b+s plain (post-trained init)")]
for s, bs in sets.items():
    idxs = [{b: rng.integers(0, len(A[s]["base"][b]), len(A[s]["base"][b])) for b in bs} for _ in range(2000)]
    for x, y in pairs:
        d = np.array([comp(A[s][x], bs, i) - comp(A[s][y], bs, i) for i in idxs])
        print(f"  {s:6s} {x:32s} - {y:30s} {comp(A[s][x], bs) - comp(A[s][y], bs):+6.2f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")

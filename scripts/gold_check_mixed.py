"""Run as: .venv/bin/python -P scripts/gold_check_mixed.py  (-P: scripts/queue.py shadows stdlib queue otherwise).
Feed gold references of data/eval/mixed/* through the registered scorers (all items). Expect ~100."""
import json, os, sys
sys.path.insert(0, '.')
from collections import defaultdict
from harness.score import get_scorer

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def gold_output(bench, it):
    r = it["reference"]
    if bench == "gsm8k_ml": return f"The answer is \\boxed{{{r}}}."
    if bench == "bird_ml": return f"```sql\n{r}\n```"
    if bench == "mbpp_ml": return f"```python\n{r}\n```" if isinstance(r, str) and r.strip() else None
    if bench == "bfcl_ml":
        calls = []
        for c in r:
            for fn, args in c.items():
                calls.append({"name": fn, "arguments": {k: v[0] for k, v in args.items() if v and v[0] != ""}})
        return json.dumps(calls)

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
for bench in ["gsm8k_ml", "bird_ml", "mbpp_ml", "bfcl_ml"]:
    items = load(f"data/eval/mixed/{bench}/{SPLIT}.jsonl")
    gens = {}
    for it in items:
        o = gold_output(bench, it)
        if o is not None: gens[it["id"]] = {"output": o}
    items = [it for it in items if it["id"] in gens]
    res = get_scorer(bench)(items, gens)
    per = defaultdict(list)
    for it, c in zip(items, res["correct"]): per[it["meta"]["language"]].append(c)
    print(f"mixed {bench:10s} n={len(items):4d} gold-score={res['score']:.1f}  " +
          " ".join(f"{l}={100*sum(v)/len(v):.1f}(n={len(v)})" for l, v in per.items()))
    bad = [it["id"] for it, c in zip(items, res["correct"]) if not c]
    if bad: print("  failing:", bad[:10], len(bad))

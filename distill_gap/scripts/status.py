"""Summarise gap-check results: per model solved %, errors, cap hits; per task table with --tasks."""
import json, glob, os, sys, collections
R = os.path.join(os.path.dirname(__file__), "..", "results")
order = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-32b", "glm-5.3-flash"]
data = {}
for m in order:
    f = os.path.join(R, m + ".jsonl")
    if os.path.exists(f):
        data[m] = [json.loads(l) for l in open(f)]
for m, r in data.items():
    n = len(r); s = sum(x["score"] >= 0.999 for x in r)
    toks = [x["completion_tokens"] for x in r if x.get("completion_tokens")]
    print(f"{m:15s} n={n:4d} solved={s:4d} ({100*s/max(n,1):5.1f}%) errors={sum('error' in x for x in r)} "
          f"cap={sum(x.get('finish')=='length' for x in r)} mean_tok={sum(toks)/max(len(toks),1):.0f}")
if "--tasks" in sys.argv:
    tasks = sorted({x["task"] for r in data.values() for x in r})
    print("\n" + "task".ljust(20) + "".join(m.replace("qwen3-", "")[:9].rjust(10) for m in data))
    for t in tasks:
        row = t.ljust(20)
        for m, r in data.items():
            xs = [x for x in r if x["task"] == t]
            row += (f"{100*sum(x['score']>=0.999 for x in xs)/len(xs):.0f}" if xs else "-").rjust(10)
        print(row)

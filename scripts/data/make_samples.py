import sys, os, glob, json; sys.path.insert(0, os.path.dirname(__file__))
from common import *
out = open(os.path.join(ROOT, "logs/data/samples.txt"), "w")
def trunc(x, n=1200):
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    return s if len(s) <= n else s[:n] + f"...[+{len(s)-n} chars]"
for p in sorted(glob.glob(os.path.join(DATA, "eval/*/*/dev.jsonl"))):
    rows = read_jsonl(p); name = "/".join(p.split("/")[-3:-1])
    for r in rows[:1] + rows[len(rows)//2:len(rows)//2+1]:
        out.write(f"=== EVAL {name} id={r['id']} max_tokens={r['max_tokens']}\nPROMPT:\n{trunc(r['messages'][0]['content'])}\nREFERENCE ({type(r['reference']).__name__}): {trunc(r['reference'], 500)}\nMETA keys: {list(r['meta'])}\n\n")
for d in ["math","code","sql","tools","translate","general"]:
    rows = read_jsonl(os.path.join(DATA, f"train/{d}/pool.jsonl"))
    for r in rows[:2]:
        out.write(f"=== TRAIN {d} id={r['id']} source={r['source']} n_tokens={r['n_tokens']}\nUSER:\n{trunc(r['messages'][0]['content'], 900)}\nASSISTANT:\n{trunc(r['messages'][1]['content'], 700)}\n\n")

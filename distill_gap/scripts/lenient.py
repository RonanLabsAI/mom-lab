"""Format-insensitive re-score.
solved = official grader gives 1.0, OR
  - grid tasks with one unique answer (GRID): the numbers (and < > signs) in the prediction, in order, equal the gold's;
  - jugs: the prediction's JSON is flattened to a plain list of move strings and re-scored by the official grader.
Every other task stays strict (official grader only)."""
import json, re, os
import reasoning_gym as rg
D = os.path.join(os.path.dirname(__file__), "..")
GRID = {"futoshiki", "mini_sudoku", "binary_matrix", "kakurasu", "survo", "arc_1d"}
gold = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/data/gap_v1.jsonl")}
tok = lambda s: re.findall(r"-?\d+|[<>]", s or "")
_ds = {}
def entry(g):
    k = (g["task"], g["seed"])
    if k not in _ds: _ds[k] = rg.create_dataset(g["task"], size=50, seed=g["seed"])
    return _ds[k], _ds[k][g["idx"]]
def flat(v):
    if isinstance(v, str): return [v]
    if isinstance(v, dict): return [s for x in v.values() for s in flat(x)]
    if isinstance(v, list): return [s for x in v for s in flat(x)]
    return []
def solved(x):
    if x["score"] >= 0.999: return True
    g = gold[x["id"]]; p = x.get("pred")
    if not p: return False
    if x["task"] in GRID and g["answer"]:
        return tok(p) == tok(g["answer"])
    if x["task"] == "jugs":
        try: moves = flat(json.loads(p))
        except Exception: return False
        ds, e = entry(g)
        try: return ds.score_answer(json.dumps(moves), e) >= 0.999
        except Exception: return False
    return False
models = [m for m in ["qwen3-0.6b","qwen3-1.7b","qwen3-4b","qwen3-8b","qwen3-32b","glm-5.3-flash"] if os.path.exists(f"{D}/results/{m}.jsonl")]
res = {}
for m in models:
    for l in open(f"{D}/results/{m}.jsonl"):
        x = json.loads(l); res.setdefault(m, {}).setdefault(x["task"], []).append(solved(x))
out = {m: {t: 100*sum(v)/len(v) for t, v in res[m].items()} for m in models}
json.dump(out, open(f"{D}/results/lenient_table.json", "w"), indent=1)
tasks = sorted(next(iter(res.values())))
print("task".ljust(20) + "".join(m.replace("qwen3-","")[:9].rjust(10) for m in models))
for t in tasks:
    print(t.ljust(20) + "".join((f"{out[m][t]:.0f}" if t in out[m] else "-").rjust(10) for m in models))
print("ALL".ljust(20) + "".join(f"{100*sum(sum(v) for v in res[m].values())/sum(len(v) for v in res[m].values()):.1f}".rjust(10) for m in models))

"""Validate data/eval/para/<bench>/<split>.jsonl: protected literals byte-identical, kept values present, no banned
template phrase, reference/max_tokens/meta copied from the source item. Run: .venv/bin/python -P scripts/validate_para.py [dev|test]"""
import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("build_para", "scripts/build_para.py"); bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
tot = 0
for b, cfg in bp.BENCH.items():
    src = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/eval/{cfg['src']}/{SPLIT}.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open(f"data/eval/para/{b}/{SPLIT}.jsonl", encoding="utf-8")]
    bad = []
    for r in rows:
        s = src[r["meta"]["source_id"]]; c = r["messages"][0]["content"]
        sk, lits, keep, _ = cfg["sk"](s["messages"][0]["content"], s["meta"]) if b == "flores_pp" else cfg["sk"](s["messages"][0]["content"])
        if r["reference"] != s["reference"] or r.get("max_tokens") != s.get("max_tokens"): bad.append((r["id"], "ref"))
        if {k: v for k, v in r["meta"].items() if k not in ("source_domain", "source_id", "source_benchmark")} != s["meta"]: bad.append((r["id"], "meta"))
        bad += [(r["id"], "literal") for l in lits if l not in c]
        bad += [(r["id"], f"value {v!r}") for v in set(keep) if v and v not in c]
        bad += [(r["id"], "banned") for x in bp.BANNED if x.lower() in c.lower()]
        if "{{" in c and "{{" not in s["messages"][0]["content"]: bad.append((r["id"], "placeholder-left"))
    print(f"{b}: n={len(rows)} problems={len(bad)}", bad[:5]); tot += len(bad)
print("TOTAL problems", tot)

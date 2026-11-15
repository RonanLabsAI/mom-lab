"""Validation (b): back-translate 5 random translated NL spans per language (same server). Run with -P."""
import json, re, random, sys, importlib.util
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("bm", "scripts/build_mixed.py"); bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
ep = sys.argv[1]; rng = random.Random(7); out = []
for lang in bm.LANGS:
    picks = []
    for b, cfg in bm.BENCH.items():
        rows = [json.loads(l) for l in open(f"data/eval/mixed/{b}/dev.jsonl", encoding="utf-8")]
        rows = [r for r in rows if r["meta"]["language"] == lang]
        picks += [(b, r) for r in rng.sample(rows, 2)]
    for b, r in rng.sample(picks, 5):
        cfg = bm.BENCH[b]; sid = r["meta"]["source_id"]
        src = next(json.loads(l) for l in open(f"data/eval/{cfg['src']}/dev.jsonl", encoding="utf-8") if json.loads(l)["id"] == sid)
        segs = cfg["seg"](src["messages"][0]["content"])
        pat = "".join(re.escape(x[1]) if x[0] == "lit" else "(.*?)" for x in segs)
        g = re.fullmatch(pat, r["messages"][0]["content"], re.S).groups()
        trs = [x for x in segs if x[0] == "tr" and x[2] in ("math", "sql", "code", "tools")]
        idx = [i for i, x in enumerate([x for x in segs if x[0] == "tr"]) if x[2] in ("math", "sql", "code", "tools")]
        for x, i in zip(trs, idx):
            sysp = "Translate the user's text into English. Output only the translation."
            back = bm.chat(ep, sysp, g[i])
            print(f"[{lang}] {r['id']}\n  SRC : {x[1]}\n  TRN : {g[i]}\n  BACK: {back}\n")

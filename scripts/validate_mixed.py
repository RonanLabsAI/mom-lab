"""Validation (a): non-translated parts byte-identical to source; ids/meta/reference unchanged. Run: .venv/bin/python -P scripts/validate_mixed.py"""
import json, os, re, sys
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location("build_mixed", "scripts/build_mixed.py"); bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
tot_bad = 0
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
for b, cfg in bm.BENCH.items():
    src = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/eval/{cfg['src']}/{SPLIT}.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open(f"data/eval/mixed/{b}/{SPLIT}.jsonl", encoding="utf-8")]
    bad = []; langs = {}; srcids = {}
    for r in rows:
        m = r["meta"]; s = src[m["source_id"]]
        langs[m["language"]] = langs.get(m["language"], 0) + 1
        srcids.setdefault(m["language"], set()).add(m["source_id"])
        if r["id"] != f'{m["language"]}-{s["id"]}': bad.append((r["id"], "id"))
        if r["reference"] != s["reference"] or r["max_tokens"] != s["max_tokens"] or r["messages"][0]["role"] != "user" or len(r["messages"]) != 1:
            bad.append((r["id"], "ref/max_tokens/messages"))
        mm = {k: v for k, v in m.items() if k not in ("source_domain", "language", "source_id", "source_benchmark")}
        if mm != s["meta"]: bad.append((r["id"], "meta"))
        if m["source_domain"] != cfg["domain"]: bad.append((r["id"], "domain"))
        segs = cfg["seg"](s["messages"][0]["content"])
        pat = "".join(re.escape(x[1]) if x[0] == "lit" else "(.*?)" for x in segs)
        c = r["messages"][0]["content"]
        mt = re.fullmatch(pat, c, re.S)
        if not mt: bad.append((r["id"], "literal-parts-differ")); continue
        trs = [x for x in segs if x[0] == "tr"]
        for g, x in zip(mt.groups(), trs):
            if bm.digits(x[1]) - bm.digits(g) and x[2] in ("math", "tools", "code", "sql"):
                bad.append((r["id"], f"digits-missing:{x[2]}"))
            if x[2] == "instr":
                for tok in re.findall(r"```\w*|\\boxed\{\}", x[1]):
                    if tok not in g: bad.append((r["id"], f"token-missing:{tok}"))
            if g.strip() == x[1].strip() and x[2] in ("math", "code", "tools") and len(x[1]) > 25:
                bad.append((r["id"], "untranslated"))
    ok = len(set(map(len, srcids.values()))) == 1 and len({frozenset(v) for v in srcids.values()}) == 1
    print(f"{b}: n={len(rows)} per-lang={langs} same-source-items-across-langs={ok} problems={len(bad)}")
    for x in bad[:15]: print("   ", x)
    tot_bad += len(bad)
print("TOTAL problems", tot_bad)

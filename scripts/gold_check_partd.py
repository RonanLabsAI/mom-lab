"""Part D experiment B: validate data_partd/eval/{sql/bird_dev,tools/bfcl}/test.jsonl. No model is run.

Checks: (1) frozen harness loads the files via harness.common.eval_path(data_root=data_partd); (2) gold check:
reference rendered as the model output, scored by harness.score.get_scorer('bird_dev' / 'bfcl'), expect 100;
(3) every SQL db file resolves, opens read-only with sqlite3 and passes PRAGMA quick_check; (4) format: same top-level
keys as data/eval/<d>/<b>/test.jsonl, meta keys a superset, prompt rebuilt with the same prompt function matches;
(5) contamination re-check against the same index the builder uses (expect 0 hits); (6) unique ids.

Usage: .venv/bin/python -P scripts/gold_check_partd.py
"""
import json, sqlite3, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts/data"))
import harness.common as C  # noqa: E402
from harness.score import get_scorer  # noqa: E402
from harness.score.sql import resolve_db, execute  # noqa: E402
import build_partd_b as B  # noqa: E402  (main() is guarded; importing does not rebuild)
from sqlfmt import sqlite_schema, sql_prompt  # noqa: E402
from toolfmt import tools_prompt  # noqa: E402

DR = ROOT / "data_partd"
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


sets = {("sql", "bird_dev"): None, ("tools", "bfcl"): None}
for d, b in sets:
    p = C.eval_path(d, b, "test", DR)
    check(p.exists(), f"eval_path({d},{b},test,data_partd) -> {p.relative_to(ROOT)} exists")
    sets[(d, b)] = C.read_jsonl(p)
    items = sets[(d, b)]
    ref = C.read_jsonl(C.eval_path(d, b, "test"))[0]
    check(all(set(it) == set(ref) for it in items), f"{b}: top-level keys == {sorted(ref)} (n={len(items)})")
    check(all(set(ref["meta"]) <= set(it["meta"]) for it in items), f"{b}: meta keys superset of {sorted(ref['meta'])}")
    check(len({it['id'] for it in items}) == len(items), f"{b}: ids unique")
    check(all(it["max_tokens"] == ref["max_tokens"] for it in items), f"{b}: max_tokens == {ref['max_tokens']}")

# gold
for (d, b), items in sets.items():
    gens = {it["id"]: {"output": B.gold_output(d, it)} for it in items}
    res = get_scorer(b)(items, gens)
    check(res["score"] == 100.0, f"{b}: gold-check via get_scorer('{b}') = {res['score']:.3f} on n={len(items)}")
    if d == "tools":
        gn = {it["id"]: {"output": B.gold_output(d, it, naive=True)} for it in items}
        print(f"INFO {b}: gold-check with the old naive generator (scripts/gold_check.py style) = {get_scorer(b)(items, gn)['score']:.3f}")
        print("INFO bfcl categories:", dict(Counter(it["meta"]["bfcl_category"] for it in items)))

# SQL dbs + prompts
sql_items = sets[("sql", "bird_dev")]
dbs = sorted({it["meta"]["db_path"] for it in sql_items})
bad = []
for p in dbs:
    f = resolve_db(p)
    try:
        assert f.exists() and str(f).startswith(str(DR))
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        r = con.execute("PRAGMA quick_check").fetchone()[0]
        n = con.execute("select count(*) from sqlite_master where type='table'").fetchone()[0]
        con.close()
        assert r == "ok" and n > 0
    except Exception as e:  # noqa
        bad.append((p, repr(e)))
check(not bad, f"bird_dev: {len(dbs)} db files resolve under data_partd/, open read-only, quick_check ok ({bad[:3]})")
check(all(Path(it["meta"]["db_path"]).parts[0] == "data_partd" for it in sql_items), "bird_dev: every meta.db_path points into data_partd/")
schemas = {p: sqlite_schema(str(resolve_db(p))) for p in dbs}
check(all(it["messages"][0]["content"] == sql_prompt(schemas[it["meta"]["db_path"]], it["meta"]["question"], it["meta"]["evidence"])
          for it in sql_items), "bird_dev: prompts == sqlfmt.sql_prompt(schema, question, evidence)")
empty = sum(1 for it in sql_items if not execute(resolve_db(it["meta"]["db_path"]), it["reference"]))
print(f"INFO bird_dev: gold query returns 0 rows for {empty}/{len(sql_items)} items (any 0-row prediction scores 1 on these)")
print("INFO bird_dev sources:", dict(Counter(it["meta"]["source"] for it in sql_items)),
      "difficulty:", dict(Counter(it["meta"]["difficulty"] for it in sql_items)), "dbs:", len(dbs))
tl = sets[("tools", "bfcl")]
check(all(it["messages"][0]["content"] == tools_prompt(it["meta"]["function"], B.question_of(it)) for it in tl),
      "bfcl: prompts == toolfmt.tools_prompt(function, question)")

# contamination re-check
idx = B.contam_index()
for (d, b), items in sets.items():
    hits = [(it["id"], B.contam_hits(it, idx)) for it in items]
    hits = [h for h in hits if h[1]]
    check(not hits, f"{b}: contamination hits in final file = {len(hits)} {hits[:3]}")

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

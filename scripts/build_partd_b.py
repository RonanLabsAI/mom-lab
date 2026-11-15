"""Part D, experiment B: build a fresh held-out SQL + tool-calling test set under a separate data root.

Output (read by the frozen harness with --data-root data_partd, reusing the existing benchmark names/scorers):
  data_partd/eval/sql/bird_dev/test.jsonl     Spider 1.0 dev (all) + seeded sample of Spider 1.0 test
  data_partd/eval/sql/bird_dev/databases/<db>/<db>.sqlite
  data_partd/eval/tools/bfcl/test.jsonl       BFCL v3 live_{simple,multiple,parallel,parallel_multiple}
  data_partd/build_b_log.json                 counts, drops by reason, sources, hashes

Item format is the same as scripts/data/eval_sql.py / eval_tools.py (same sql_prompt / tools_prompt). Filters, in
order: (1) structural (BFCL items with a system turn or without ground truth; the frozen prompt format has one user
turn), (2) exact duplicate within the new set, (3) contamination: question text or full prompt equal, after lowercase +
whitespace collapse (and, stricter, also ignoring punctuation), to any user prompt / embedded question in the SQL+tools
training data, or (SQL) gold query equal ignoring case/punctuation/whitespace to any training answer or eval reference (N14M spec, plus the full
train pools and heldout) or any existing eval item (data/eval/{sql,tools,mixed,para}, data_partc), (4) gold check:
the reference, fed as the model output through the frozen harness scorer, must score 1. Then SQL keeps all surviving
dev items and draws a seeded sample of surviving test items so the total is TARGET_SQL.

No model is run. Usage: .venv/bin/python -P scripts/build_partd_b.py
"""
import hashlib, json, random, re, shutil, sqlite3, sys, urllib.request, zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/data"))
import os; os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from sqlfmt import sqlite_schema, sql_prompt  # noqa: E402
from toolfmt import tools_prompt  # noqa: E402

OUT = ROOT / "data_partd"
RAW = OUT / "raw/spider"
SPIDER_URL = "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
SPIDER_SHA = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
BFCL = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REV = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"  # same revision as data/eval/tools/bfcl (data/manifest.json)
LIVE = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
SEED = 20261101
TARGET_SQL = 2000
REF_FMT = "BFCL ground_truth: list of {func_name: {param: [allowed values...]}}; model output is a JSON list of {name, arguments}"


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def loose(s):
    return re.sub(r"[\W_]+", " ", (s or "").lower()).strip()


_Q = [re.compile(r"\n\nQuestion: (.*?)\n\nAnswer with only the SQL", re.S),
      re.compile(r"\n\nUser request: (.*?)\n\nAnswer with only the function", re.S)]


def texts_of_prompt(content):
    """Normalised full prompt + the embedded question/request if the prompt uses our templates."""
    out = {norm(content)}
    for rx in _Q:
        m = rx.search(content)
        if m: out.add(norm(m.group(1)))
    return out


# ---------------- Spider hardness (official spider/evaluation.py eval_hardness, quirks kept) ----------------
_LIKE = 9  # WHERE_OPS.index('like')


def _has_agg(u): return u[0] != 0  # AGG_OPS.index('none') == 0


def _count_agg(units): return len([u for u in units if _has_agg(u)])


def _nested(sql):
    n = []
    for cu in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if type(cu[3]) is dict: n.append(cu[3])
        if type(cu[4]) is dict: n.append(cu[4])
    for k in ("intersect", "except", "union"):
        if sql[k] is not None: n.append(sql[k])
    return n


def _c1(sql):
    c = sum(1 for k in ("where", "groupBy", "orderBy") if len(sql[k]) > 0) + (sql["limit"] is not None)
    if len(sql["from"]["table_units"]) > 0: c += len(sql["from"]["table_units"]) - 1
    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    c += len([t for t in ao if t == "or"])
    cus = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    c += len([cu for cu in cus if cu[1] == _LIKE])
    return c


def _others(sql):
    c = 0
    a = _count_agg(sql["select"][1]) + _count_agg(sql["where"][::2]) + _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        a += _count_agg([u[1] for u in sql["orderBy"][1] if u[1]] + [u[2] for u in sql["orderBy"][1] if u[2]])
    a += _count_agg(sql["having"])
    c += a > 1
    c += len(sql["select"][1]) > 1
    c += len(sql["where"]) > 1
    c += len(sql["groupBy"]) > 1
    return c


def hardness(sql):
    c1, c2, o = _c1(sql), len(_nested(sql)), _others(sql)
    if c1 <= 1 and o == 0 and c2 == 0: return "easy"
    if (o <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and o < 2 and c2 == 0): return "medium"
    if (o > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and o <= 2 and c2 == 0) or (c1 <= 1 and o == 0 and c2 <= 1):
        return "hard"
    return "extra"


# ---------------- sources ----------------
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def spider_root():
    z = RAW / "spider_data.zip"
    if not z.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print("downloading Spider ...", flush=True)
        urllib.request.urlretrieve(SPIDER_URL, z)
    assert sha256(z) == SPIDER_SHA, "spider_data.zip sha256 mismatch"
    d = RAW / "spider_data"
    if not (d / "dev.json").exists():
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():  # refuse path traversal
                assert not n.startswith("/") and ".." not in Path(n).parts, n
            zf.extractall(RAW)
    return d


def sql_candidates():
    src = spider_root()
    dbdst = OUT / "eval/sql/bird_dev/databases"
    items, schemas = [], {}
    for split, jf, dbdir in (("dev", "dev.json", "database"), ("test", "test.json", "test_database")):
        rows = json.load(open(src / jf, encoding="utf-8"))
        for i, r in enumerate(rows):
            db = r["db_id"]
            if db not in schemas:
                f = src / dbdir / db / f"{db}.sqlite"
                (dbdst / db).mkdir(parents=True, exist_ok=True)
                g = dbdst / db / f"{db}.sqlite"
                if not g.exists(): shutil.copy2(f, g)
                schemas[db] = sqlite_schema(str(g))
            q = r["question"].strip()
            items.append(dict(id=f"spider-{split}-{i}", messages=[{"role": "user", "content": sql_prompt(schemas[db], q, "")}],
                              reference=r["query"].strip().rstrip(";").strip(),
                              meta={"db_id": db, "db_path": f"data_partd/eval/sql/bird_dev/databases/{db}/{db}.sqlite",
                                    "difficulty": hardness(r["sql"]), "evidence": "", "question": q,
                                    "source": f"spider1.0-{split}", "source_index": i}, max_tokens=512))
    return items


def tools_candidates(drops):
    from huggingface_hub import hf_hub_download
    items = []
    for cat in LIVE:
        q = [json.loads(l) for l in open(hf_hub_download(BFCL, f"BFCL_v3_{cat}.json", repo_type="dataset", revision=BFCL_REV))]
        a = {}
        for l in open(hf_hub_download(BFCL, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset", revision=BFCL_REV)):
            x = json.loads(l); a.setdefault(x["id"], []).append(x["ground_truth"])
        for r in q:
            turns = r["question"]
            if len(turns) != 1 or [m["role"] for m in turns[0]] != ["user"]:
                drops["tools"]["system_or_multiturn"].append(r["id"]); continue
            if len(a.get(r["id"], [])) != 1:
                drops["tools"]["no_unique_ground_truth"].append(r["id"]); continue
            base = cat[len("live_"):]
            items.append(dict(id=f"bfcl-{r['id']}", messages=[{"role": "user", "content": tools_prompt(r["function"], turns[0][0]["content"])}],
                              reference=a[r["id"]][0], meta={"category": base, "bfcl_category": cat, "function": r["function"],
                                                             "bfcl_id": r["id"], "reference_format": REF_FMT}, max_tokens=512))
    return items


def question_of(it):
    m = it.get("meta") or {}
    if m.get("question"): return m["question"]
    c = it["messages"][0]["content"]
    mm = _Q[1].search(c)
    return mm.group(1) if mm else c


# ---------------- contamination index ----------------
def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip(): yield json.loads(l)


def contam_index():
    """-> {normalised text: source label}. Training user prompts + embedded questions; existing eval items."""
    idx = {}
    def add(texts, label):
        for t in texts:
            if t:
                idx.setdefault("x:" + t, label)        # exact after lowercase + whitespace collapse
                if loose(t): idx.setdefault("p:" + loose(t), label)  # also ignoring punctuation
    def add_sql(sql, label):
        if sql and sql.strip(): idx.setdefault("s:" + loose(sql.replace("```sql", "").replace("```", "")), label)
    train = [ROOT / f"data/mixtures/N14M/spec/{d}.jsonl" for d in ("sql", "tools")]
    train += [ROOT / f"data/train/{d}/{s}.jsonl" for d in ("sql", "tools") for s in ("pool", "heldout")]
    for p in train:
        lab = str(p.relative_to(ROOT))
        for r in iter_jsonl(p):
            for m in r["messages"]:
                if m["role"] == "user": add(texts_of_prompt(m["content"]), lab)
                elif m["role"] == "assistant" and ("/sql/" in lab or lab.endswith("/sql.jsonl")): add_sql(m["content"], lab)
    evals = sorted((ROOT / "data/eval/sql").rglob("*.jsonl")) + sorted((ROOT / "data/eval/tools").rglob("*.jsonl"))
    evals += [p for p in sorted((ROOT / "data/eval").glob("*/*/*.jsonl")) if p.parent.name in ("bird_ml", "bfcl_ml", "bird_pp", "bfcl_pp")]
    evals += sorted((ROOT / "data_partc").rglob("*.jsonl"))
    for p in evals:
        lab = str(p.relative_to(ROOT))
        for r in iter_jsonl(p):
            add(texts_of_prompt(r["messages"][0]["content"]) | {norm((r.get("meta") or {}).get("question"))}, lab)
            if isinstance(r.get("reference"), str): add_sql(r["reference"], lab)
    return idx


def contam_hits(it, idx):
    """-> (kind, source) or None. kind: exact_text | punct_insensitive_text | gold_sql_equal (SQL only)."""
    ts = (norm(question_of(it)), norm(it["messages"][0]["content"]))
    for t in ts:
        if "x:" + t in idx: return "exact_text", idx["x:" + t]
    for t in ts:
        if loose(t) and "p:" + loose(t) in idx: return "punct_insensitive_text", idx["p:" + loose(t)]
    if isinstance(it["reference"], str) and "s:" + loose(it["reference"]) in idx:
        return "gold_sql_equal", idx["s:" + loose(it["reference"])]
    return None


# ---------------- gold ----------------
def _concrete(x):
    """BFCL nested ground truth -> one concrete value. Dicts in ground truth are {key: [allowed values]} (BFCL
    possible-answer format, recursively); take the first allowed value, omit keys whose first value is "" (optional)."""
    if isinstance(x, dict) and x and all(isinstance(v, list) for v in x.values()):
        return {k: _concrete(v[0]) for k, v in x.items() if v and v[0] != ""}
    if isinstance(x, list):
        return [_concrete(e) for e in x]
    return x


def gold_output(dom, it, naive=False):
    """Reference rendered as a model output. naive=True reproduces scripts/gold_check.py (top-level first value only)."""
    r = it["reference"]
    if dom == "sql": return f"```sql\n{r}\n```"
    calls = []
    for c in r:
        for fn, args in c.items():
            calls.append({"name": fn, "arguments": {k: (v[0] if naive else _concrete(v[0])) for k, v in args.items() if v and v[0] != ""}})
    return json.dumps(calls, ensure_ascii=False)


def gold_scores(dom, items, naive=False):
    from harness.score import get_scorer
    gens = {it["id"]: {"output": gold_output(dom, it, naive)} for it in items}
    res = get_scorer("bird_dev" if dom == "sql" else "bfcl")(items, gens)
    return dict(zip(res["ids"], res["correct"]))


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    drops = {d: {k: [] for k in ("system_or_multiturn", "no_unique_ground_truth", "duplicate_in_set", "contamination", "gold_fail")}
             for d in ("sql", "tools")}
    contam_detail = {"sql": Counter(), "tools": Counter()}
    print("indexing training + existing eval prompts ...", flush=True)
    idx = contam_index()
    print("index size", len(idx), flush=True)
    log = {"sources": {
        "sql": {"dataset": "Spider 1.0 (Yu et al., EMNLP 2018), official release spider_data.zip", "url": SPIDER_URL,
                "page": "https://yale-lily.github.io/spider", "zip_sha256": SPIDER_SHA, "license": "CC BY-SA 4.0",
                "splits": "dev.json (1,034; 20 dbs) + test.json (2,147; 40 dbs); db_ids disjoint from Spider train"},
        "tools": {"hf": BFCL, "revision": BFCL_REV, "license": "Apache-2.0",
                  "files": [f"BFCL_v3_{c}.json + possible_answer/BFCL_v3_{c}.json" for c in LIVE]}}}
    for dom in ("sql", "tools"):
        cands = sql_candidates() if dom == "sql" else tools_candidates(drops)
        n0 = len(cands) + sum(len(v) for v in drops[dom].values())
        raw_cands = list(cands)
        seen, keep = set(), []
        for it in cands:  # within-set duplicates (same db + question for SQL; same full prompt for tools)
            k = (it["meta"]["db_id"], norm(it["meta"]["question"])) if dom == "sql" else norm(it["messages"][0]["content"])
            if k in seen: drops[dom]["duplicate_in_set"].append(it["id"]); continue
            seen.add(k); keep.append(it)
        cands, keep = keep, []
        for it in cands:
            h = contam_hits(it, idx)
            if h: drops[dom]["contamination"].append(it["id"]); contam_detail[dom][h[0] + " | " + h[1]] += 1; continue
            keep.append(it)
        print(dom, "gold-checking", len(keep), flush=True)
        g = gold_scores(dom, keep)
        pre_gold = 100.0 * sum(g.values()) / max(1, len(g))
        gn = gold_scores(dom, keep, naive=True) if dom == "tools" else g
        pre_gold_naive = 100.0 * sum(gn.values()) / max(1, len(gn))
        for it in keep:
            if g[it["id"]] != 1.0: drops[dom]["gold_fail"].append(it["id"])
        keep = [it for it in keep if g[it["id"]] == 1.0]
        if dom == "sql":
            dev = [it for it in keep if it["meta"]["source"].endswith("dev")]
            test = sorted([it for it in keep if it["meta"]["source"].endswith("test")], key=lambda x: x["id"])
            rng = random.Random(SEED)
            pick = {it["id"] for it in rng.sample(test, min(len(test), max(0, TARGET_SQL - len(dev))))}
            order = {it["id"]: i for i, it in enumerate(raw_cands)}
            final = sorted(dev + [it for it in test if it["id"] in pick], key=lambda x: order[x["id"]])
            # db files actually used; remove copies of dbs no kept item uses
            used = {it["meta"]["db_id"] for it in final}
            for d in (OUT / "eval/sql/bird_dev/databases").iterdir():
                if d.name not in used: shutil.rmtree(d)
            extra = {"pool_after_filters": {"dev": len(dev), "test": len(test)}, "test_sampled": len(pick), "dbs": len(used),
                     "by_source": Counter(it["meta"]["source"] for it in final),
                     "by_difficulty": Counter(it["meta"]["difficulty"] for it in final),
                     "dev_pool_difficulty_all1034": Counter(it["meta"]["difficulty"] for it in raw_cands if it["meta"]["source"].endswith("dev"))}
            write_jsonl(OUT / "eval/sql/bird_dev/test.jsonl", final)
        else:
            final = keep
            extra = {"by_category": Counter(it["meta"]["bfcl_category"] for it in final)}
            write_jsonl(OUT / "eval/tools/bfcl/test.jsonl", final)
        log[dom] = {"candidates_raw": n0, "final": len(final), "gold_checked": len(g), "gold_score_before_gold_drop": round(pre_gold, 3),
                    "gold_score_before_gold_drop_naive_generator": round(pre_gold_naive, 3),
                    "drops": {k: len(v) for k, v in drops[dom].items()}, "drop_ids": drops[dom],
                    "contamination_by_source": dict(contam_detail[dom]), **extra}
        print(dom, json.dumps({k: v for k, v in log[dom].items() if k != "drop_ids"}), flush=True)
    log["files_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in (OUT / "eval/sql/bird_dev/test.jsonl", OUT / "eval/tools/bfcl/test.jsonl")}
    json.dump(log, open(OUT / "build_b_log.json", "w"), indent=1, default=dict)


if __name__ == "__main__":
    main()

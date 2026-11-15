import sys, os, re; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sqlfmt import sql_prompt
from datasets import load_dataset
def norm_schema(s):
    s = re.sub(r"[ \t]+", " ", s).strip()
    return re.sub(r"\n\s*", "\n", s)
def ans(q): return "```sql\n" + q.strip().rstrip(";") + "\n```"
def spider_schema(r):
    tabs = [t.strip() for t in r["Schema (values (type))"].split(" | ")]
    pks = {}
    for p in r["Primary Keys"].split(" | "):
        if " : " in p:
            t, c = p.split(" : ", 1); pks.setdefault(t.strip().lower(), []).extend(x.strip() for x in c.split(","))
    fks = {}
    for f in r["Foreign Keys"].split(" | "):
        m = re.match(r"(.+?) : (.+?) equals (.+?) : (.+)", f)
        if m: fks.setdefault(m[1].strip().lower(), []).append((m[2].strip(), m[3].strip(), m[4].strip()))
    out = []
    for t in tabs:
        name, cols = t.split(" : ", 1)
        name = name.strip(); parts = []
        for c in cols.split(" , "):
            m = re.match(r"(.+) \((\w+)\)$", c.strip())
            cn, ty = (m[1], m[2]) if m else (c.strip(), "text")
            parts.append(f"{cn} {'INTEGER' if ty=='number' else 'TEXT'}")
        if name.lower() in pks: parts.append(f"PRIMARY KEY ({', '.join(pks[name.lower()])})")
        for c, rt, rc in fks.get(name.lower(), []): parts.append(f"FOREIGN KEY ({c}) REFERENCES {rt}({rc})")
        out.append(f"CREATE TABLE {name} ({', '.join(parts)});")
    return "\n".join(out)
def main():
    rows = []; seen = set()
    def add(src, i, schema, q, ev, sql):
        k = (q.strip().lower(), sql.strip().lower())
        if k in seen or not sql.strip() or not q.strip(): return
        seen.add(k)
        rows.append(dict(id=f"{src}-{i}", messages=[{"role":"user","content":sql_prompt(schema, q, ev)},{"role":"assistant","content":ans(sql)}], domain="sql", source=src))
    b = load_dataset("xu3kev/BIRD-SQL-data-train", split="train")
    for i, r in enumerate(b): add("bird_train", i, norm_schema(r["schema"]), r["question"], r["evidence"], r["SQL"])
    nb = len(rows)
    sc = {r["db_id"]: spider_schema(r) for r in load_dataset("richardr1126/spider-schema", split="train")}
    for i, r in enumerate(load_dataset("xlangai/spider", split="train")): add("spider_train", i, sc[r["db_id"]], r["question"], None, r["query"])
    ns = len(rows) - nb
    for i, r in enumerate(load_dataset("b-mc2/sql-create-context", split="train")): add("sql_create_context", i, r["context"], r["question"], None, r["answer"])
    print(nb, ns, len(rows) - nb - ns)
    finalize_train("sql", rows, {"sources":{
      "bird_train":{"hf":"xu3kev/BIRD-SQL-data-train","revision":hf_sha("xu3kev/BIRD-SQL-data-train"),"license":"BIRD train: CC BY-SA 4.0 (share-alike); card has no license","n":nb},
      "spider_train":{"hf":"xlangai/spider","revision":hf_sha("xlangai/spider"),"license":"CC BY-SA 4.0 (share-alike)","n":ns,"schema_source":"richardr1126/spider-schema (CC-BY-4.0) converted to CREATE TABLE"},
      "sql_create_context":{"hf":"b-mc2/sql-create-context","revision":hf_sha("b-mc2/sql-create-context"),"license":"CC BY 4.0 (derived from WikiSQL (BSD-3) + Spider (CC BY-SA 4.0))","n":len(rows)-nb-ns}},
      "substitution":"BIRD train.zip (8.9 GB) download too slow (~1 MB/s after 2.9 GB); used xu3kev/BIRD-SQL-data-train (schema-in-row) + Spider with schemas from richardr1126/spider-schema; sql-create-context added for volume. Tables with spider/sql-create-context have no evidence field."})
main()

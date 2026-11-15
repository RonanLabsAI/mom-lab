import sys, os, json, shutil, glob; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sqlfmt import *
SRC = os.path.join(DATA, "raw/bird/dev_x/dev_20240627")
def main():
    dbroot = os.path.join(SRC, "db_x/dev_databases")
    dst = os.path.join(DATA, "eval/sql/bird_dev/databases")
    schemas = {}
    for db in sorted(os.listdir(dbroot)):
        f = os.path.join(dbroot, db, f"{db}.sqlite")
        if not os.path.isfile(f): continue
        os.makedirs(os.path.join(dst, db), exist_ok=True)
        if not os.path.exists(os.path.join(dst, db, f"{db}.sqlite")):
            shutil.copy2(f, os.path.join(dst, db, f"{db}.sqlite"))
        schemas[db] = sqlite_schema(os.path.join(dst, db, f"{db}.sqlite"))
    dev = json.load(open(os.path.join(SRC, "dev.json")))
    items = []
    for r in dev:
        db = r["db_id"]
        items.append(dict(id=f"bird-dev-{r['question_id']}", messages=[{"role":"user","content":sql_prompt(schemas[db], r["question"], r["evidence"])}],
            reference=r["SQL"], meta={"db_id":db,"db_path":f"data/eval/sql/bird_dev/databases/{db}/{db}.sqlite","difficulty":r["difficulty"],"evidence":r["evidence"],"question":r["question"]}, max_tokens=512))
    o = write_eval("sql", "bird_dev", items, strat_key=lambda x: x["meta"]["db_id"] + "|" + x["meta"]["difficulty"])
    update_manifest("eval", "sql", {"bird_dev":{"source":"https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip (dev_20240627)","revision":"dev_20240627 (zip Last-Modified 2024-06-29)","license":"CC BY-SA 4.0 (BIRD)","metric":"execution accuracy", **o}})
main()

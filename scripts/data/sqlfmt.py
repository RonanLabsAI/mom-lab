import sqlite3, re
def sqlite_schema(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute("select sql from sqlite_master where sql is not null and type in ('table','view') and name not like 'sqlite_%' order by rowid").fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        t = re.sub(r"[ \t]+", " ", r[0]).strip().rstrip(";")
        t = re.sub(r"\n\s*", "\n", t)
        out.append(t + ";")
    return "\n".join(out)
def sql_prompt(schema, question, evidence=None):
    s = "Write a SQLite query that answers the question, given the database schema.\n\n"
    s += f"Database schema:\n{schema}\n\n"
    if evidence and evidence.strip():
        s += f"External knowledge: {evidence.strip()}\n\n"
    s += f"Question: {question.strip()}\n\nAnswer with only the SQL query in a ```sql block."
    return s

"""sql: BIRD dev execution accuracy.

Item schema: reference = gold SQL; meta.db_path = sqlite file (absolute, or relative to the repo root
or to data/eval/sql/bird_dev/databases/). Official BIRD script compares set(predicted_rows) ==
set(gold_rows) (order and duplicates ignored); that is the default here. Pass strict_multiset=True
(env HARNESS_SQL_MULTISET=1) to compare as multisets instead.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from ..common import DATA, ROOT, strip_think
from . import output_of, pmap

TIMEOUT_S = 30.0
_FENCE = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    text = strip_think(text)
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip().rstrip(";").strip()
    m = re.search(r"```[ \t]*(?:sql)?[ \t]*\n(.*)$", text, re.S | re.I)  # unterminated
    if m:
        return m.group(1).strip().rstrip(";").strip()
    m = re.search(r"\b(with|select)\b.*", text, re.S | re.I)
    return (m.group(0) if m else text).strip().rstrip(";").strip()


def resolve_db(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    for base in (ROOT, DATA / "eval" / "sql" / "bird_dev" / "databases", DATA / "eval" / "sql" / "bird_dev"):
        if (base / p).exists():
            return base / p
    return p


def execute(db: Path, sql: str, timeout: float = TIMEOUT_S):
    """Run on a read-only connection with a wall-clock limit. Returns list of rows; raises on error."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=timeout)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        con.execute("PRAGMA query_only = ON")
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _rows_equal(a, b, multiset: bool) -> bool:
    if multiset:
        return Counter(a) == Counter(b)
    return set(a) == set(b)


def _one(args) -> float:
    out, gold, db_path, multiset = args
    if out is None:
        return 0.0
    try:
        db = resolve_db(db_path)
        pred = extract_sql(out)
        if not pred:
            return 0.0
        gold_rows = execute(db, gold)
        pred_rows = execute(db, pred)
        return float(_rows_equal(pred_rows, gold_rows, multiset))
    except BaseException:  # noqa
        return 0.0


def score_sql(items, gens, workers=None, strict_multiset: bool | None = None):
    if strict_multiset is None:
        strict_multiset = os.environ.get("HARNESS_SQL_MULTISET") == "1"
    args = [(output_of(gens, it["id"]), it["reference"], (it.get("meta") or {}).get("db_path", ""),
             strict_multiset) for it in items]
    vals = pmap(_one, args, workers)
    return {"ids": [it["id"] for it in items], "correct": vals,
            "score": 100.0 * sum(vals) / max(1, len(vals))}

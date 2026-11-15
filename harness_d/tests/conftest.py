import sqlite3
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_db():
    FIX.mkdir(exist_ok=True)
    p = FIX / "tiny.sqlite"
    if p.exists():
        p.unlink()
    con = sqlite3.connect(p)
    con.executescript("""
    CREATE TABLE emp(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER);
    INSERT INTO emp VALUES (1,'ann','eng',100),(2,'bob','eng',90),(3,'cy','ops',70),(4,'di','ops',70);
    """)
    con.commit()
    con.close()
    return p

"""Smoke rules (prereg "Smoke test")."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke_check", ROOT / "scripts/gate/smoke_check.py")
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)

OK = {"truncated": 0, "unparseable": 0, "schema_invalid": 0, "fenced": 0,
      "nulls": {f: {"gold": 5, "out": 5} for f in SC.NULLABLE}}


def test_pass():
    assert SC.verdict("gate-qwen3-8b-zs-schema", OK) == 0


def test_whitespace_rule():
    assert SC.verdict("gate-qwen3-8b-zs-schema", dict(OK, truncated=3)) == 3
    assert SC.verdict("gate-qwen3-8b-zs-schema", dict(OK, truncated=2)) == 0


def test_canary_rules():
    assert SC.verdict("gate-qwen3-8b-zs-schema", dict(OK, fenced=1)) == 4
    assert SC.verdict("gate-qwen3-8b-zs-schema", dict(OK, schema_invalid=1)) == 4
    lost = dict(OK, nulls=dict(OK["nulls"], wind_gust_kt={"gold": 20, "out": 0}))
    assert SC.verdict("gate-qwen3-8b-zs-schema", lost) == 4


def test_free_runs_never_gate():
    assert SC.verdict("gate-qwen3-8b-zs-free", dict(OK, fenced=30)) == 0


def test_run_name_from_path(tmp_path, capsys):
    import json
    d = tmp_path / "partdG-smoke" / "gate-qwen3-8b-zs-schema" / "gen" / "aviation"
    d.mkdir(parents=True)
    (d / "metar.dev.jsonl").write_text(json.dumps({"id": "metar-dev-0000", "output": "```json\n{}\n```",
                                                   "finish_reason": "stop"}) + "\n")
    rc = SC.main(["--results-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.startswith("gate-qwen3-8b-zs-schema ")
    assert rc == 4  # fenced output in a schema run must trip the canary

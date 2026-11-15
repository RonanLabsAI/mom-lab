"""Gate schema tests (preregs/2026-11-14-metar-schema-gate.md).

The schema must accept every gold reference: a schema stricter than the gold would cap the constrained arms'
accuracy below 100 and bias the gate against constrained decoding.
"""
import json
import os
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_d.score import metar as M  # noqa: E402

SCHEMA = ROOT / "configs/gate/metar_schema.json"
# eval files are tracked in git; fall back to the main checkout if a worktree lacks them
DATA = [ROOT / "data_partd_c/eval/aviation/metar",
        Path("/opt/release-user/projects/RonanLabs/mom-lab/data_partd_c/eval/aviation/metar")]


def _items(split):
    for d in DATA:
        p = d / f"{split}.jsonl"
        if p.exists():
            return [json.loads(l) for l in p.open() if l.strip()]
    pytest.skip(f"METAR {split} file not found")


def _validator():
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize("split,n", [("dev", 300), ("test", 1000)])
def test_all_gold_validate(split, n):
    v = _validator()
    items = _items(split)
    assert len(items) == n
    bad = [(it["id"], e.message) for it in items for e in v.iter_errors(it["reference"])]
    assert bad == []


def test_keys_match_scorer():
    schema = json.loads(SCHEMA.read_text())
    assert schema["required"] == M.FIELDS
    assert list(schema["properties"]) == M.FIELDS
    assert schema["additionalProperties"] is False


def test_schema_rejects_hhmm():
    v = _validator()
    ref = dict(_items("test")[0]["reference"])
    ref["time_utc"] = "0355"
    assert not v.is_valid(ref)


def test_schema_rejects_unknown_category():
    v = _validator()
    ref = dict(_items("test")[0]["reference"])
    ref["flight_category"] = "VMC"
    assert not v.is_valid(ref)

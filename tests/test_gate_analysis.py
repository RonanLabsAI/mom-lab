"""Gate analysis on stand-in data: Part D-C/F runs copied under gate run names must reproduce Part F's own tests."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("gate_analysis", ROOT / "scripts/gate/gate_analysis.py")
GA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GA)

DATA = ROOT / "data_partd_c/eval/aviation/metar/test.jsonl"
F = json.loads((ROOT / "results/analysis/partd_f.json").read_text())
BASE = {"gate-qwen3-8b-zs-schema": ("partdC", "base-qwen3-8b"),
        "gate-qwen3-8b-fs-schema": ("partdF", "base-qwen3-8b-fs"),
        "gate-qwen3-8b-fsthink-schema-s0": ("partdC", "base-qwen3-4b"),
        "gate-qwen3-32b-zs-schema": ("partdF", "base-qwen3-32b"),
        "gate-qwen3-32b-fs-schema": ("partdF", "base-qwen3-14b"),
        "gate-qwen3-32b-think-schema-s0": ("partdF", "base-qwen3-32b-think-s0")}


def _standin(tmp_path, mapping, parity=True):
    for gate_run, (step, run) in mapping.items():
        d = tmp_path / "partdG" / gate_run
        d.mkdir(parents=True)
        shutil.copy(ROOT / "results" / step / run / "scores.json", d / "scores.json")
    (tmp_path / "partdG").mkdir(exist_ok=True)
    if parity:
        (tmp_path / "partdG" / "parity.json").write_text(json.dumps({"ok": True, "identical": 1000}))
    return tmp_path


def _t(res, hyp):
    return next(t for t in res["tests"] if t["hyp"] == hyp)


def test_standin_reproduces_f(tmp_path):
    res = GA.analyse(str(_standin(tmp_path, BASE)), str(ROOT / "results"), str(DATA), require_gens=False)
    f1 = next(t for t in F["tests"] if t["hyp"] == "F1" and t["bench"] == "metar")
    g1 = _t(res, "G1")
    assert res["best"]["8B"] == "8B few-shot + schema"
    assert g1["diff"] == pytest.approx(f1["diff"], abs=1e-9)
    assert g1["lo"] == pytest.approx(f1["lo"], abs=1e-9) and g1["hi"] == pytest.approx(f1["hi"], abs=1e-9)
    assert res["best"]["32B"] == "32B thinking + schema s0"
    g2 = _t(res, "G2")
    assert g2["diff"] == pytest.approx(99.8333333 - 88.9, abs=1e-5)
    assert res["headline"] == "Stands" and res["void_reasons"] == []
    assert _t(res, "G1-cluster")["verdict"] == "PASS"
    assert res["n_stations"] > 50


def test_equivalent_when_arm_equals_adapter(tmp_path):
    m = dict(BASE, **{"gate-qwen3-8b-fs-schema": ("partdC", "pt4-lora16-metar-lr2e-4")})
    res = GA.analyse(str(_standin(tmp_path, m)), str(ROOT / "results"), str(DATA), require_gens=False)
    assert _t(res, "G3")["verdict"] in ("equivalent", "schema ahead")
    assert res["headline"] == "Rewritten"


def test_void_without_parity(tmp_path):
    res = GA.analyse(str(_standin(tmp_path, BASE, parity=False)), str(ROOT / "results"), str(DATA),
                     require_gens=False)
    assert res["headline"] == "VOID"
    assert "parity not 1,000/1,000" in res["void_reasons"]


def test_void_when_outputs_missing(tmp_path):
    res = GA.analyse(str(_standin(tmp_path, BASE)), str(ROOT / "results"), str(DATA), require_gens=True)
    assert res["headline"] == "VOID"


def test_void_when_arms_missing(tmp_path):
    (tmp_path / "partdG").mkdir()
    res = GA.analyse(str(tmp_path), str(ROOT / "results"), str(DATA), require_gens=False)
    assert res["headline"] == "VOID"
    assert "partdG/gate-qwen3-8b-zs-schema" in res["missing"]


def test_canary_flags_fenced_and_invalid():
    import jsonschema
    v = jsonschema.Draft202012Validator(json.loads((ROOT / "configs/gate/metar_schema.json").read_text()))
    ref = json.loads(DATA.open().readline())["reference"]
    g = {"a": {"output": json.dumps(ref), "finish_reason": "stop"},
         "b": {"output": "```json\n" + json.dumps(ref) + "\n```", "finish_reason": "stop"},
         "c": {"output": json.dumps(dict(ref, time_utc="0355")), "finish_reason": "stop"},
         "d": {"output": "{", "finish_reason": "length"}}
    c = GA.canary(g, v)
    assert c == {"truncated": 1, "unparseable": 0, "schema_invalid": 1, "fenced": 1}

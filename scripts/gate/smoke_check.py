"""METAR schema gate smoke check (preregs/2026-11-14-metar-schema-gate.md, "Smoke test"; dev split, first 30 items).

Reads results/partdG-smoke/<run>/gen/aviation/metar.dev.jsonl. Prints per run: items, truncations
(finish_reason == "length"), unparseable, schema-invalid, fenced outputs, mean output tokens, mean latency, and for
each nullable field how often gold and output are null. No scores are computed.
Fixed rules (schema runs only):
  exit 4  canary: any non-truncated output that is unparseable, fails schema v1, or contains a ``` fence (the grammar
          was not applied, e.g. a router stripped response_format); or a nullable field that is null in >= 3 of the
          30 gold items is never null in the output (a lost null branch). The run stops; nothing is changed by rule.
  exit 3  whitespace rule: any schema run truncates more than 2 of 30 items -> disable_any_whitespace (deviation).
  exit 0  pass.
usage: .venv-gate/bin/python -P scripts/gate/smoke_check.py [--results-root results]
"""
import glob
import json
import os
import sys

import jsonschema

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from harness_d.score import metar as M  # noqa: E402

NULLABLE = ["wind_dir_deg", "wind_gust_kt", "ceiling_ft", "dewpoint_c"]
GOLD = os.path.join(ROOT, "data_partd_c/eval/aviation/metar/dev.jsonl")


def summarize(path, validator, gold):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    s = {"n": len(rows), "truncated": 0, "unparseable": 0, "schema_invalid": 0, "fenced": 0,
         "mean_out_tokens": round(sum(r.get("n_out_tokens") or 0 for r in rows) / max(1, len(rows)), 1),
         "mean_latency_s": round(sum(r.get("latency_s") or 0 for r in rows) / max(1, len(rows)), 2),
         "nulls": {f: {"gold": 0, "out": 0} for f in NULLABLE}}
    for r in rows:
        out = r.get("output") or ""
        if r.get("finish_reason") == "length":
            s["truncated"] += 1
            continue
        obj = M.parse_output(out)
        s["fenced"] += "```" in M.strip_think(out) if hasattr(M, "strip_think") else "```" in out
        if obj is None:
            s["unparseable"] += 1
            continue
        if not validator.is_valid(obj):
            s["schema_invalid"] += 1
        g = gold.get(r["id"])
        for f in NULLABLE:
            s["nulls"][f]["out"] += obj.get(f, 0) is None
            s["nulls"][f]["gold"] += g is not None and g[f] is None
    return s


def verdict(run, s):
    if "schema" not in run:
        return 0
    if s["unparseable"] or s["schema_invalid"] or s["fenced"]:
        return 4
    if any(v["gold"] >= 3 and v["out"] == 0 for v in s["nulls"].values()):
        return 4
    if s["truncated"] > 2:
        return 3
    return 0


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    root = argv[argv.index("--results-root") + 1] if "--results-root" in argv else os.path.join(ROOT, "results")
    schema = json.load(open(os.path.join(ROOT, "configs/gate/metar_schema.json")))
    validator = jsonschema.Draft202012Validator(schema)
    gold = {json.loads(l)["id"]: json.loads(l)["reference"] for l in open(GOLD) if l.strip()}
    paths = sorted(glob.glob(f"{root}/partdG-smoke/*/gen/aviation/metar.dev.jsonl"))
    if not paths:
        print("no smoke outputs found")
        return 1
    worst = 0
    for p in paths:
        run = p.split(os.sep)[-4]  # results/partdG-smoke/<run>/gen/aviation/metar.dev.jsonl (deviation 2: was -5)
        s = summarize(p, validator, gold)
        v = verdict(run, s)
        print(run, json.dumps(s), "->", v)
        worst = 4 if 4 in (worst, v) else max(worst, v)
    print("GATE:", {0: "PASS", 3: "RAISE (disable_any_whitespace)", 4: "CANARY FAILED (stop)"}[worst])
    return worst


if __name__ == "__main__":
    sys.exit(main())

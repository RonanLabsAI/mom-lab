"""Check that xgrammar (the gate's backend) accepts every METAR gold reference under schema v1.

Runs inside the serving image (CPU only, no model): the grammar is compiled from configs/gate/metar_schema.json
exactly as vLLM's xgrammar backend does (Grammar.from_json_schema, any_whitespace as the server setting), then
every gold object is serialised compactly and indented and tested for membership. Exit 0 only if all are accepted
and two known-bad objects ("HHMM" time, unknown flight category) are rejected.

usage (from run_gate.py check):
  docker run --rm -v <schema>:/schema.json:ro -v <golds.jsonl>:/golds.jsonl:ro -v <this>:/c.py:ro \
      --entrypoint python3 <image> /c.py [--no-whitespace]
"""
import json
import sys

import xgrammar as xg
from xgrammar.testing import _is_grammar_accept_string as accepts

any_ws = "--no-whitespace" not in sys.argv
schema = open("/schema.json").read()
g = xg.Grammar.from_json_schema(schema, any_whitespace=any_ws)
golds = [json.loads(l) for l in open("/golds.jsonl") if l.strip()]
bad = []
for i, r in enumerate(golds):
    compact = json.dumps(r, separators=(",", ":"))
    # fixed-separator mode (vLLM disable_any_whitespace) emits xgrammar default separators ", " and ": "
    forms = [compact, json.dumps(r), json.dumps(r, indent=2)] if any_ws else [json.dumps(r)]
    if not all(accepts(g, s) for s in forms):
        bad.append(i)
neg = dict(golds[0]); neg["time_utc"] = "0355"
neg2 = dict(golds[0]); neg2["flight_category"] = "VMC"
cj = json.dumps
rejects = (not accepts(g, cj(neg))) and (not accepts(g, cj(neg2)))
print(json.dumps({"n": len(golds), "accepted": len(golds) - len(bad), "rejected_idx": bad[:20],
                  "negatives_rejected": rejects, "any_whitespace": any_ws}))
sys.exit(0 if not bad and rejects else 1)

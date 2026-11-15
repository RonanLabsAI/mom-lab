"""Part D addendum F smoke gate (dev split, first 30 items; content/PART_D_F_ADDENDUM.md "Smoke test").

Reads results/partdF-smoke/<run>/gen/*/*.dev.jsonl and prints, per run and benchmark: items, thinking truncations
(finish_reason == "length" or an unclosed <think>), unparseable answers, mean output tokens and mean latency.
Gate (fixed in the addendum): if any thinking run truncates more than 2 of 30 items on any benchmark, every thinking
job's maxtok goes from 16384 to 26000 before the test runs (logged as a deviation). Exit 0 = gate passed, 3 = raise.
Scores are NOT printed (dev only, and not used for anything).

usage: .venv/bin/python -P scripts/partd_f/smoke_check.py
"""
import glob, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from harness_d.score import metar as M, medcalc as MC  # noqa: E402

R = "results/partdF-smoke"


def answer_found(bench, out):
    if "</think>" in out:
        out = out.split("</think>", 1)[1]
    elif "<think>" in out:
        return False
    if bench == "metar":
        return M.parse_output(out) is not None
    if bench == "medcalc":
        return MC.locate_answer(out) != "Not Found"
    return "ANSWER" in out.upper()


def main():
    raise_tok = False
    for gp in sorted(glob.glob(f"{R}/*/gen/*/*.dev.jsonl")):
        run = gp.split("/")[2]; bench = os.path.basename(gp).split(".")[0]
        g = [json.loads(l) for l in open(gp) if l.strip()]
        trunc = sum(1 for x in g if x.get("finish_reason") == "length" or ("<think>" in (x["output"] or "") and "</think>" not in x["output"]))
        noans = sum(1 for x in g if not answer_found(bench, x["output"] or ""))
        tok = sum(x.get("n_out_tokens") or 0 for x in g) / max(1, len(g))
        lat = sum(x.get("latency_s") or 0 for x in g) / max(1, len(g))
        flag = ""
        if "think" in run and trunc > 2:
            raise_tok = True; flag = "  <-- truncation gate"
        print(f"{run:28s} {bench:9s} n={len(g):3d} truncated={trunc:2d} no_answer={noans:2d} mean_tokens={tok:7.0f} mean_latency_s={lat:6.1f}{flag}")
    print("GATE:", "RAISE thinking maxtok to 26000" if raise_tok else "pass (keep 16384)")
    sys.exit(3 if raise_tok else 0)


if __name__ == "__main__":
    main()

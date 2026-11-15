"""Score a run: python -m harness.aggregate --run-id ID --step STEP [--split dev]

Writes results/<step>/<run_id>/scores.json:
 {run_id, step, split, metadata, A, A_partial, complete, domains: {d: s_d}, domain_complete: {d: bool},
  benchmarks: {b: {domain, score, n, n_missing, ids, correct, groups?, extra?}}}
A = mean of the 6 domain scores (null unless all six domains are fully present; A_partial = mean over
domains present). s_d = mean of the domain's benchmark scores.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import common as C
from .score import get_scorer


def load_gens(path: Path) -> dict:
    gens = {}
    if path.exists():
        for g in C.read_jsonl(path):
            out = g.get("output") or ""
            if "</think>" in out:  # thinking-mode runs: score only the final answer
                g = {**g, "output": out.split("</think>", 1)[1], "had_thinking": True}
            elif "<think>" in out:  # thinking never closed (truncated): no final answer
                g = {**g, "output": "", "had_thinking": True, "think_truncated": True}
            gens[g["id"]] = g
    return gens


def aggregate(step, run_id, split="dev", benchmarks=None, workers=None,
              data_root=C.DATA, results_root=C.RESULTS, allow_test=False, write=True) -> dict:
    C.check_split(split, allow_test)
    rdir = C.run_dir(step, run_id, Path(results_root))
    meta_path = rdir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metadata["scored_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    metadata["scorer_git_hash"] = C.git_hash()
    res = {"run_id": run_id, "step": step, "split": split, "metadata": metadata,
           "benchmarks": {}, "domains": {}, "domain_complete": {}}
    for d, bs in C.ALL_DOMAINS.items():
        scores = []
        complete = True
        for b in bs:
            if benchmarks and b not in benchmarks:
                complete = False
                continue
            gp = C.gen_path(rdir, d, b, split)
            ep = C.eval_path(d, b, split, Path(data_root))
            if not gp.exists() or not ep.exists():
                complete = False
                continue
            items = C.read_jsonl(ep)
            gens = load_gens(gp)
            r = get_scorer(b)(items, gens, workers=workers)
            entry = {"domain": d, "score": r["score"], "n": len(items),
                     "n_missing": sum(1 for it in items if it["id"] not in gens),
                     "ids": r["ids"], "correct": r["correct"]}
            for k in ("groups", "extra"):
                if k in r:
                    entry[k] = r[k]
            res["benchmarks"][b] = entry
            if entry["n_missing"]:
                complete = False
            scores.append(r["score"])
            print(f"  {d}/{b}: {r['score']:.2f}  (n={len(items)}, missing={entry['n_missing']})", file=sys.stderr)
        if scores:
            res["domains"][d] = sum(scores) / len(scores)
        if d in C.EXTRA_DOMAINS and not scores:
            continue  # optional extra domain (e.g. mixed) not run: leave it out entirely
        res["domain_complete"][d] = complete and len(scores) == len(bs)
    ds = [v for d, v in res["domains"].items() if d in C.DOMAINS]  # core domains only (mixed excluded from A)
    res["A_partial"] = sum(ds) / len(ds) if ds else None
    res["complete"] = all(res["domain_complete"].get(d, False) for d in C.DOMAINS)
    res["A"] = sum(res["domains"][d] for d in C.DOMAINS) / len(C.DOMAINS) if res["complete"] else None
    if write:
        (rdir).mkdir(parents=True, exist_ok=True)
        (rdir / "scores.json").write_text(json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harness.aggregate")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--benchmarks")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--data-root", default=str(C.DATA))
    ap.add_argument("--results-root", default=str(C.RESULTS))
    a = ap.parse_args(argv)
    r = aggregate(a.step, a.run_id, a.split, set(a.benchmarks.split(",")) if a.benchmarks else None,
                  a.workers, a.data_root, a.results_root, a.allow_test)
    print(json.dumps({"A": r["A"], "A_partial": r["A_partial"], "domains": r["domains"],
                      "complete": r["complete"]}, indent=2))


if __name__ == "__main__":
    main()

"""Per-benchmark scorers.

Every scorer has signature  score(items, gens, workers=None) -> dict  with keys
  ids:     list[str]   item ids (order of `items`)
  correct: list[float] per-item value; 0/1 for accuracy benchmarks, sentence chrF++ (0-100) for translate
  score:   float       benchmark score on 0-100
  (optional) groups: list[str] per-item group label (translate direction), extra: dict
`gens` maps id -> generation record ({"output": ...}). A missing generation scores 0.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor


def pmap(fn, args, workers=None):
    """Process-pool map (fork) so signal-based timeouts work; serial for tiny inputs."""
    args = list(args)
    workers = workers or min(32, os.cpu_count() or 1)
    if len(args) <= 2 or workers <= 1:
        return [fn(a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, args, chunksize=max(1, len(args) // (workers * 4))))


def output_of(gens, item_id) -> str | None:
    g = gens.get(item_id)
    if g is None:
        return None
    return g.get("output") or ""


def get_scorer(benchmark: str):
    from . import code, general, math_, sql, tools, translate
    table = {
        "gsm8k": math_.score_gsm8k, "math500": math_.score_math500,
        "humaneval_plus": code.score_code, "mbpp_plus": code.score_code,
        "livecodebench": code.score_code,
        "bird_dev": sql.score_sql, "bfcl": tools.score_bfcl,
        "flores200": translate.score_flores,
        "mmlu_pro": general.score_mmlu_pro, "ifeval": general.score_ifeval,
        # mixed-skill (translated) sets reuse the source scorers
        "gsm8k_ml": math_.score_gsm8k, "bird_ml": sql.score_sql,
        "mbpp_ml": code.score_code, "bfcl_ml": tools.score_bfcl,
        # paraphrase sets (Step 4) reuse the source scorers
        "gsm8k_pp": math_.score_gsm8k, "bird_pp": sql.score_sql, "mbpp_pp": code.score_code,
        "bfcl_pp": tools.score_bfcl, "mmlupro_pp": general.score_mmlu_pro, "flores_pp": translate.score_flores,
    }
    return table[benchmark]

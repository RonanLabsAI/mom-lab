"""general: MMLU-Pro (letter extraction) and IFEval (official checkers, prompt-level strict).

IFEval item schema: meta.instruction_id_list, meta.kwargs (list of dicts), prompt = messages[-1].content
(or meta.prompt). Checkers are Google's instruction_following_eval as vendored in lm-evaluation-harness
(lm_eval.tasks.ifeval); they are loaded by path so the rest of lm_eval need not be installed.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

from ..common import strip_think
from . import output_of

try:  # langdetect is random unless seeded; IFEval language checks must be deterministic
    from langdetect import DetectorFactory as _DF
    _DF.seed = 0
except Exception:
    pass


LETTERS = "ABCDEFGHIJ"
_PATS = [
    re.compile(r"answer\s+is\s*:?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])", re.I),
    re.compile(r"answer\s*:\s*\**\s*\(?([A-J])\)?(?![A-Za-z])", re.I),
]


def extract_letter(text: str) -> str | None:
    text = strip_think(text)
    for i, pat in enumerate(_PATS):
        flags = pat.flags
        m = pat.findall(text)
        if m:
            return m[-1].upper()
    m = re.findall(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", text[-200:])
    return m[-1] if m else None


def score_mmlu_pro(items, gens, workers=None):
    vals = []
    for it in items:
        o = output_of(gens, it["id"])
        pred = extract_letter(o) if o else None
        vals.append(float(pred is not None and pred == str(it["reference"]).strip().upper()[:1]))
    return {"ids": [it["id"] for it in items], "correct": vals,
            "score": 100.0 * sum(vals) / max(1, len(vals))}


_IF = None


def _load_ifeval():
    global _IF
    if _IF is not None:
        return _IF
    import importlib.util
    spec = importlib.util.find_spec("lm_eval")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("lm_eval not installed (pip install --no-deps lm_eval==0.4.13)")
    base = Path(list(spec.submodule_search_locations)[0]) / "tasks" / "ifeval"
    for name, path in [("lm_eval", None), ("lm_eval.tasks", None), ("lm_eval.tasks.ifeval", base)]:
        if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
            m = types.ModuleType(name)
            m.__path__ = [str(path)] if path else []
            sys.modules[name] = m
    from lm_eval.tasks.ifeval import utils  # noqa
    _IF = utils
    return utils


def _if_one(it, out) -> float:
    try:
        u = _load_ifeval()
        meta = it.get("meta") or {}
        prompt = meta.get("prompt") or it["messages"][-1]["content"]
        inp = u.InputExample(key=0, instruction_id_list=meta["instruction_id_list"], prompt=prompt,
                             kwargs=meta.get("kwargs") or [{} for _ in meta["instruction_id_list"]])
        # The official checkers draw missing instruction args from Python's global `random`
        # (e.g. when a kwarg is falsy and gets dropped), which made scores depend on call history.
        import random as _r
        _r.seed(0)
        return float(u.test_instruction_following_strict(inp, strip_think(out or "")).follow_all_instructions)
    except Exception:
        return 0.0


def score_ifeval(items, gens, workers=None):
    vals = [_if_one(it, output_of(gens, it["id"])) if output_of(gens, it["id"]) is not None else 0.0
            for it in items]
    return {"ids": [it["id"] for it in items], "correct": vals,
            "score": 100.0 * sum(vals) / max(1, len(vals))}

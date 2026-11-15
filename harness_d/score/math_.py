"""math: gsm8k (numeric) and math500 (math-verify)."""
from __future__ import annotations

import re

from ..common import strip_think
from . import output_of, pmap

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+")


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces), also handles \\fbox."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def last_number(text: str) -> str | None:
    m = _NUM.findall(text)
    return m[-1] if m else None


def extract_answer(text: str) -> tuple[str | None, bool]:
    """(answer, from_boxed)."""
    text = strip_think(text)
    b = last_boxed(text)
    if b is not None:
        return b, True
    m = re.findall(r"answer is[:\s]*\$?\\?\(?\s*(.+?)\s*\\?\)?\$?\s*(?:\.|$)", text, re.I | re.M)
    if m:
        n = last_number(m[-1])
        if n:
            return n, False
    m = re.findall(r"####\s*(.+)", text)
    if m:
        return m[-1].strip(), False
    return last_number(text), False


def _to_float(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").replace("\\$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(\d+)", s)
    try:
        if m:
            return float(m.group(1)) / float(m.group(2))
        return float(s)
    except ValueError:
        m = _NUM.findall(s)
        if len(m) == 1 and len(s) < len(m[0]) + 12:
            return _to_float(m[0]) if m[0] != s else None
        return None


def _mv_equal(gold: str, pred: str) -> bool:
    from math_verify import parse, verify
    g = parse("\\boxed{" + gold + "}", parsing_timeout=5)
    p = parse("\\boxed{" + pred + "}", parsing_timeout=5)
    if not g or not p:
        return False
    return bool(verify(g, p, timeout_seconds=5))


def _score_one(args) -> float:
    kind, gold, out = args
    try:
        if out is None:
            return 0.0
        ans, _ = extract_answer(out)
        if ans is None or not ans.strip():
            return 0.0
        gold = str(gold).strip()
        if "####" in gold:
            gold = gold.split("####")[-1].strip()
        if kind == "gsm8k":
            g, p = _to_float(gold), _to_float(ans)
            if g is not None and p is not None:
                return float(abs(g - p) < 1e-4 * max(1.0, abs(g)))
            return float(_mv_equal(gold, ans))
        if gold.replace(" ", "") == ans.replace(" ", ""):
            return 1.0
        return float(_mv_equal(gold, ans))
    except BaseException:  # noqa - parser timeouts / bad latex must never crash a run
        return 0.0


def _score(kind, items, gens, workers):
    args = [(kind, it["reference"], output_of(gens, it["id"])) for it in items]
    vals = pmap(_score_one, args, workers)
    return {"ids": [it["id"] for it in items], "correct": vals,
            "score": 100.0 * sum(vals) / max(1, len(vals))}


def score_gsm8k(items, gens, workers=None):
    return _score("gsm8k", items, gens, workers)


def score_math500(items, gens, workers=None):
    return _score("math500", items, gens, workers)

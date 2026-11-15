"""MedCalc-Bench scorer: a port of the official evaluation (github.com/ncbi-nlp/MedCalc-Bench @ 20b10f9,
evaluation/run.py extract_answer + evaluation/evaluate.py check_correctness).

Item schema: reference = Ground Truth Answer (str); meta.calid (int), meta.lower_limit, meta.upper_limit (str).
Only deviation from the official code is how the answer string is located in the output: the official prompt
asks for a JSON dict and takes the value after the last `answer":`; our prompt asks for a final line
"Answer: <value>", so we take the text after the LAST "Answer:" (case-insensitive, markdown asterisks
stripped). If there is none, the official JSON pattern is tried as a fallback. The extracted string is then
passed through the official per-calculator parsing and correctness rules unchanged:
  dates (calc 13, 68): exact date match;  gestational age (69): (weeks, days) exact;
  integer calculators: round(answer) == ground truth;  decimal calculators: lower_limit <= answer <= upper_limit.
Any parse error counts as incorrect (the official run.py catches exceptions the same way).
The official code eval()s model text; here a small AST evaluator accepts only arithmetic, numbers and the
functions the official eval namespace exposes (min, pow, round, abs, int, float, math.*), so results are
identical for legitimate expressions but arbitrary code cannot run. Plain numeric strings are parsed with float()
first (the official eval() would raise on a leading zero such as "05" and score it wrong).
"""
from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime

from ..common import strip_think
from . import output_of

DATE_IDS = {13, 68}
WEEKS_DAYS_IDS = {69}
INT_IDS = {4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69}
DEC_IDS = {2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61,
           62, 63, 64, 65, 66, 67}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"min": min, "max": max, "pow": pow, "round": round, "abs": abs, "int": int, "float": float}


def safe_eval(expr: str):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Tuple):
            return tuple(ev(e) for e in n.elts)
        if isinstance(n, ast.Call) and not n.keywords:
            f = n.func
            if isinstance(f, ast.Name) and f.id in _FUNCS:
                return _FUNCS[f.id](*[ev(a) for a in n.args])
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("math", "np", "numpy") \
                    and f.attr in ("sqrt", "log", "log10", "exp", "ceil", "floor", "pow", "fabs"):
                return getattr(math, f.attr)(*[ev(a) for a in n.args])
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "math" and n.attr in ("pi", "e"):
            return getattr(math, n.attr)
        raise ValueError(f"disallowed expression: {ast.dump(n)[:80]}")
    if len(expr) > 500:
        raise ValueError("expression too long")
    return ev(ast.parse(expr.strip(), mode="eval"))


_ANS = re.compile(r"answer\s*:", re.I)


def locate_answer(text: str) -> str:
    """Our format: text after the last 'Answer:' on that line. Fallback: official JSON pattern. Else 'Not Found'."""
    text = strip_think(text or "")
    ms = list(_ANS.finditer(text))
    if ms:
        rest = text[ms[-1].end():].strip()
        line = rest.split("\n")[0].strip() if rest else ""
        line = line.strip("*").strip().strip("`").strip().rstrip(".").strip()
        if line:
            return line
    ex = re.findall(r'[Aa]nswer":\s*(.*?)\}', text)  # official extraction (run.py)
    if ex:
        a = ex[-1].strip().strip('"')
        if a not in ("str(short_and_direct_answer_of_the_question)", "str(value which is the answer to the question)", "X.XX"):
            return a
    return "Not Found"


def extract_answer(extracted_answer: str, calid: int) -> str:
    """Official run.py extract_answer, minus the JSON lookup (done in locate_answer)."""
    calid = int(calid)
    if calid in DATE_IDS:
        match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
        if match:
            return f"{int(match.group(1)):02}/{int(match.group(2)):02}/{match.group(3)}"
        return "N/A"
    if calid in WEEKS_DAYS_IDS:
        extracted_answer = extracted_answer.replace("[", "(").replace("]", ")").replace("'", "").replace('"', "")
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        return f"({match.group(1)}, {match.group(3)})" if match else "N/A"
    if calid in INT_IDS:
        match = re.search(r"(\d+) out of", extracted_answer)
        if match:
            return match.group(1)
        match = re.search(r"-?\d+(, ?-?\d+)+", extracted_answer)
        if match:
            return str(len(match.group(0).split(",")))
        match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
        return match[-1][0] if match else "N/A"
    if calid in DEC_IDS:
        match = re.search(r"str\((.*)\)", extracted_answer)
        if match:
            expression = match.group(1).replace("^", "**").replace("is odd", "% 2 == 1").replace("is even", "% 2 == 0") \
                .replace("sqrt", "math.sqrt").replace(".math", "").replace("weight", "").replace("height", "") \
                .replace("mg/dl", "").replace("g/dl", "").replace("mmol/L", "").replace("kg", "").replace("g", "") \
                .replace("mEq/L", "")
            expression = expression.split('#')[0]
            if expression.count('(') > expression.count(')'):
                expression += ')' * (expression.count('(') - expression.count(')'))
            elif expression.count(')') > expression.count('('):
                expression = '(' * (expression.count(')') - expression.count('(')) + expression
            try:
                answer = safe_eval(expression)
            except Exception:
                return "N/A"
        else:
            match = re.search(r"(-?\d+(\.\d+)?)\s*mL/min/1.73", extracted_answer)
            if match:
                answer = float(match.group(1))
            else:
                match = re.findall(r"(-?\d+(\.\d+)?)\%", extracted_answer)
                if match:
                    answer = float(match[-1][0]) / 100
                else:
                    match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                    if not match:
                        return "N/A"
                    answer = float(match[-1][0])
        return str(answer)
    raise ValueError(f"Unknown calculator ID: {calid}")


def _num(s):
    s = str(s).strip()
    try:  # plain numbers first: the official eval() raises on e.g. "05" (leading zero); float() does not
        return float(s)
    except ValueError:
        return safe_eval(s)


def check_correctness(answer: str, ground_truth, calid, upper_limit, lower_limit) -> int:
    """Official evaluate.py check_correctness (eval -> safe_eval)."""
    calid = int(calid)
    if calid in DATE_IDS:
        return int(datetime.strptime(answer, "%m/%d/%Y").date() == datetime.strptime(ground_truth, "%m/%d/%Y").date())
    if calid in WEEKS_DAYS_IDS:
        pat = r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?"
        g = re.search(pat, ground_truth)
        a = re.search(pat, answer)
        if not a:
            return 0
        return int((int(a.group(1)), int(a.group(3))) == (int(g.group(1)), int(g.group(3))))
    if calid in INT_IDS:
        return int(round(_num(answer)) == _num(ground_truth))
    if calid in DEC_IDS:
        v = _num(answer)
        return int(_num(lower_limit) <= v <= _num(upper_limit))
    raise ValueError(f"Unknown calculator ID: {calid}")


def score_one(item: dict, output: str | None) -> float:
    if output is None:
        return 0.0
    m = item["meta"]
    try:
        ans = extract_answer(locate_answer(output), m["calid"])
        return float(check_correctness(ans, str(item["reference"]), m["calid"], m["upper_limit"], m["lower_limit"]))
    except Exception:
        return 0.0


def score_medcalc(items, gens, workers=None):
    vals = [score_one(it, output_of(gens, it["id"])) for it in items]
    by_type = {}
    for it, v in zip(items, vals):
        by_type.setdefault(it["meta"].get("category", "?"), []).append(v)
    return {"ids": [it["id"] for it in items], "correct": vals,
            "score": 100.0 * sum(vals) / max(1, len(vals)),
            "extra": {"by_category": {k: 100.0 * sum(v) / len(v) for k, v in sorted(by_type.items())}}}

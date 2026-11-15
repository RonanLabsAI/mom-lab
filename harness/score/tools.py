"""tools: BFCL non-live single-turn AST categories via BFCL's own ast_checker.

Item schema: meta.function = list of BFCL function specs, reference = BFCL ground truth
([{"fname": {"param": [allowed values...]}}, ...]), meta.category in simple|multiple|parallel|
parallel_multiple (default 'simple'). Model output may be JSON (a list of {"name","arguments"}, a single
object, or {fname: {args}} dicts), <tool_call>{...}</tool_call> blocks, or python-style `[f(a=1), g(b=2)]`.
ast_checker.py is loaded from the installed bfcl-eval package by path (stubbing its unused model-config
and Java/JS converter imports) so the heavy bfcl-eval dependency tree is not required.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import types
from pathlib import Path

from ..common import strip_think
from . import output_of

_CHK = None


def _load_checker():
    global _CHK
    if _CHK is not None:
        return _CHK
    import importlib.util
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None:
        raise ImportError("bfcl_eval not installed (pip install --no-deps bfcl-eval==2026.3.23)")
    base = Path(list(spec.submodule_search_locations)[0])

    def pkg(name, path=None):
        m = types.ModuleType(name)
        m.__path__ = [str(path)] if path else []
        sys.modules.setdefault(name, m)
        return sys.modules[name]

    pkg("bfcl_eval", base)
    pkg("bfcl_eval.constants", base / "constants")
    pkg("bfcl_eval.eval_checker")
    pkg("bfcl_eval.eval_checker.ast_eval")
    pkg("bfcl_eval.eval_checker.ast_eval.type_convertor")
    cfg = types.ModuleType("bfcl_eval.constants.model_config")

    class _Cfg:
        underscore_to_dot = False

    class _Map(dict):
        def __missing__(self, k):
            return _Cfg()

    cfg.MODEL_CONFIG_MAPPING = _Map()
    sys.modules["bfcl_eval.constants.model_config"] = cfg
    for n, attr in (("java_type_converter", "java_type_converter"), ("js_type_converter", "js_type_converter")):
        m = types.ModuleType(f"bfcl_eval.eval_checker.ast_eval.type_convertor.{n}")
        setattr(m, attr, lambda *a, **k: None)
        sys.modules[m.__name__] = m
    for name in ("enums", "type_mappings"):
        s = importlib.util.spec_from_file_location(f"bfcl_eval.constants.{name}", base / "constants" / f"{name}.py")
        m = importlib.util.module_from_spec(s)
        sys.modules[s.name] = m
        s.loader.exec_module(m)
    s = importlib.util.spec_from_file_location("bfcl_eval.eval_checker.ast_eval.ast_checker",
                                               base / "eval_checker" / "ast_eval" / "ast_checker.py")
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    _CHK = m
    return m


def _norm_calls(obj):
    """-> list[{name: {args}}] or None."""
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or not obj:
        return None
    calls = []
    for c in obj:
        if not isinstance(c, dict):
            return None
        if "name" in c and isinstance(c.get("name"), str):
            args = c.get("arguments", c.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    return None
            if not isinstance(args, dict):
                return None
            calls.append({c["name"]: args})
        elif len(c) == 1 and isinstance(next(iter(c.values())), dict):
            calls.append(c)
        else:
            return None
    return calls


def _py_value(node):
    return ast.literal_eval(node)


def _parse_python_calls(text: str):
    t = text.strip()
    if not t.startswith("["):
        t = "[" + t + "]"
    tree = ast.parse(t, mode="eval")
    if not isinstance(tree.body, ast.List):
        return None
    calls = []
    for el in tree.body.elts:
        if not isinstance(el, ast.Call):
            return None
        name = ast.unparse(el.func)
        if el.args:
            return None
        calls.append({name: {kw.arg: _py_value(kw.value) for kw in el.keywords}})
    return calls or None


def parse_calls(text: str):
    text = strip_think(text)
    cands = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if cands:
        out = []
        for c in cands:
            try:
                n = _norm_calls(json.loads(c))
            except ValueError:
                return None
            if not n:
                return None
            out += n
        return out
    fences = re.findall(r"```[ \t]*\w*\n(.*?)```", text, re.S)
    for s in ([fences[-1]] if fences else []) + [text]:
        s = s.strip()
        try:
            n = _norm_calls(json.loads(s))
            if n:
                return n
        except ValueError:
            pass
        m = re.search(r"[\[{].*[\]}]", s, re.S)
        if m:
            try:
                n = _norm_calls(json.loads(m.group(0)))
                if n:
                    return n
            except ValueError:
                pass
        try:
            n = _parse_python_calls(s)
            if n:
                return n
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            pass
    return None


def check_one(it, out) -> float:
    if not out:
        return 0.0
    try:
        calls = parse_calls(out)
        if not calls:
            return 0.0
        chk = _load_checker()
        meta = it.get("meta") or {}
        cat = meta.get("category", "simple")
        Language = sys.modules["bfcl_eval.constants.enums"].Language
        res = chk.ast_checker(meta["function"], calls, it["reference"], Language.PYTHON, cat, "harness")
        return float(bool(res.get("valid")))
    except Exception:
        return 0.0


def score_bfcl(items, gens, workers=None):
    vals = [check_one(it, output_of(gens, it["id"])) for it in items]
    return {"ids": [it["id"] for it in items], "correct": vals,
            "score": 100.0 * sum(vals) / max(1, len(vals))}

"""METAR decoding scorer (Part D / Experiment C, data built by scripts/partd_c/build_metar.py).

Item: reference = gold dict with the 14 schema fields (scripts/partd_c/metar_decode.py FIELDS).
Output parsing: drop <think>; take the last ```json fenced block, else the span from the first "{" to the
last "}"; json.loads must give an object. Unparseable -> every field wrong.
Field comparison (a missing key is wrong):
  station, time_utc, flight_category: case-insensitive string equality (surrounding whitespace ignored)
  wind_dir_deg: "VRB" (case-insensitive) or an integer; null must be null
  day, wind_speed_kt, wind_gust_kt, ceiling_ft, temperature_c, dewpoint_c: integer equality (numeric strings
      and integral floats accepted; booleans rejected); null must be null
  visibility_sm: |pred - gold| < 1e-6;  altimeter_inhg: |pred - gold| < 0.005
  weather: list of strings, same length and order, case-insensitive equality
  clouds: list of {cover, base_ft}, same length and order, cover case-insensitive, base_ft integer equality
Item correct = all 14 fields correct. extra: mean field accuracy, per-field accuracy, parse-failure rate.
"""
from __future__ import annotations

import json
import re

from ..common import strip_think
from . import output_of

FIELDS = ["station", "day", "time_utc", "wind_dir_deg", "wind_speed_kt", "wind_gust_kt", "visibility_sm",
          "weather", "clouds", "ceiling_ft", "temperature_c", "dewpoint_c", "altimeter_inhg", "flight_category"]
INT_FIELDS = {"day", "wind_speed_kt", "wind_gust_kt", "ceiling_ft", "temperature_c", "dewpoint_c"}
STR_FIELDS = {"station", "time_utc", "flight_category"}
_MISSING = object()


def parse_output(text: str):
    text = strip_think(text or "")
    blocks = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.S)
    cands = [b for b in blocks if "{" in b][-1:] or []
    if "{" in text and "}" in text:
        cands.append(text[text.index("{"): text.rindex("}") + 1])
    for c in cands:
        try:
            obj = json.loads(c.strip())
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def _int_eq(p, g):
    if g is None:
        return p is None
    v = _num(p)
    return v is not None and v == int(v) and int(v) == g


def field_ok(name, p, g) -> bool:
    if p is _MISSING:
        return False
    if name in STR_FIELDS:
        return isinstance(p, str) and p.strip().upper() == str(g).upper()
    if name == "wind_dir_deg":
        if g is None:
            return p is None
        if g == "VRB":
            return isinstance(p, str) and p.strip().upper() == "VRB"
        return _int_eq(p, g)
    if name in INT_FIELDS:
        return _int_eq(p, g)
    if name == "visibility_sm":
        v = _num(p)
        return v is not None and abs(v - g) < 1e-6
    if name == "altimeter_inhg":
        v = _num(p)
        return v is not None and abs(v - g) < 0.005
    if name == "weather":
        return isinstance(p, list) and len(p) == len(g) and all(
            isinstance(a, str) and a.strip().upper() == b.upper() for a, b in zip(p, g))
    if name == "clouds":
        if not (isinstance(p, list) and len(p) == len(g)):
            return False
        for a, b in zip(p, g):
            if not isinstance(a, dict) or not isinstance(a.get("cover"), str) \
                    or a["cover"].strip().upper() != b["cover"] or not _int_eq(a.get("base_ft"), b["base_ft"]):
                return False
        return True
    raise KeyError(name)


def score_one(item, output):
    gold = item["reference"]
    obj = parse_output(output) if output is not None else None
    if obj is None:
        return 0.0, {f: 0.0 for f in FIELDS}, True
    fr = {f: float(field_ok(f, obj.get(f, _MISSING), gold[f])) for f in FIELDS}
    return float(all(fr.values())), fr, False


def score_metar(items, gens, workers=None):
    vals, per_field, fails = [], {f: [] for f in FIELDS}, 0
    field_mean = []
    for it in items:
        v, fr, failed = score_one(it, output_of(gens, it["id"]))
        vals.append(v)
        fails += failed
        field_mean.append(sum(fr.values()) / len(FIELDS))
        for f in FIELDS:
            per_field[f].append(fr[f])
    n = max(1, len(items))
    return {"ids": [it["id"] for it in items], "correct": vals, "score": 100.0 * sum(vals) / n,
            "extra": {"mean_field_accuracy": 100.0 * sum(field_mean) / n,
                      "field_accuracy": {f: 100.0 * sum(v) / n for f, v in per_field.items()},
                      "parse_fail_rate": 100.0 * fails / n}}

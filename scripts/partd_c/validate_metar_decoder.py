"""Independent validation of scripts/partd_c/metar_decode.py (ground truth of the metar task).

A. Field cross-check against the python-metar package (PyPI `metar`, independent parser) on every report in the
   metar train/dev/test files: station, day/hour/minute, wind dir/speed/gust, visibility, sky layers,
   temperature, dewpoint, altimeter.
B. Flight-category cross-check against NOAA AWC's own computed `fltCat` (aviationweather.gov data API, JSON) on
   current METARs (live data; reported, not reproducible bit-for-bit). Skip with --no-awc.
Run with an interpreter that has `metar` installed, e.g.:
  python3 -m venv /tmp/metarvenv && /tmp/metarvenv/bin/pip install metar==1.11.0
  /tmp/metarvenv/bin/python -I scripts/partd_c/validate_metar_decoder.py
"""
import collections
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metar_decode import decode  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def raws():
    out = []
    for s in ("dev", "test"):
        for l in open(os.path.join(ROOT, f"data_partd_c/eval/aviation/metar/{s}.jsonl")):
            out.append(json.loads(l)["meta"]["raw"])
    for l in open(os.path.join(ROOT, "data/mixtures/partd_c/metar.jsonl")):
        out.append(json.loads(l)["messages"][0]["content"].split("METAR: ", 1)[1].split("\n", 1)[0])
    return out


def pm_fields(raw):
    from metar.Metar import Metar
    body = raw.split(" RMK")[0]
    if not body.startswith(("METAR", "SPECI")):
        body = "METAR " + body
    m = Metar(body, strict=False)
    vis = m.vis.value("SM") if m.vis else None
    return {"station": m.station_id, "day": m.time.day if m.time else None,
            "time_utc": f"{m.time.hour:02d}:{m.time.minute:02d}" if m.time else None,
            "wind_speed_kt": round(m.wind_speed.value("KT")) if m.wind_speed else None,
            "wind_gust_kt": round(m.wind_gust.value("KT")) if m.wind_gust else None,
            "wind_dir_deg": (None if (m.wind_speed and m.wind_speed.value("KT") == 0) else
                             ("VRB" if m.wind_dir is None and m.wind_speed else round(m.wind_dir.value()) if m.wind_dir else None)),
            "visibility_sm": round(vis, 4) if vis is not None else None,
            "clouds": [{"cover": c[0], "base_ft": round(c[1].value("FT"))} for c in m.sky if c[1] is not None],
            "temperature_c": round(m.temp.value("C")) if m.temp else None,
            "dewpoint_c": round(m.dewpt.value("C")) if m.dewpt else None,
            "altimeter_inhg": round(m.press.value("IN"), 2) if m.press else None}


def check_pm():
    mism = collections.Counter(); ex = collections.defaultdict(list); n = 0; err = 0
    for raw in raws():
        g = decode(raw)
        try:
            p = pm_fields(raw)
        except Exception as e:  # noqa: BLE001
            err += 1; ex["parse_error"].append(f"{raw} :: {e}"[:200]); continue
        n += 1
        for k, v in p.items():
            gv = g[k]
            if k == "visibility_sm" and v is not None and gv is not None:
                same = abs(v - gv) < 1e-3
            elif k == "altimeter_inhg":
                same = v is not None and abs(v - gv) < 0.006
            else:
                same = v == gv
            if not same:
                mism[k] += 1
                if len(ex[k]) < 4:
                    ex[k].append(f"{raw} | ours={gv} python-metar={v}")
    print(f"A. python-metar cross-check: {n} reports compared, {err} python-metar parse errors")
    for k in ["station", "day", "time_utc", "wind_dir_deg", "wind_speed_kt", "wind_gust_kt", "visibility_sm", "clouds",
              "temperature_c", "dewpoint_c", "altimeter_inhg"]:
        print(f"   {k:15s} mismatches: {mism[k]}")
    for k, v in ex.items():
        for e in v[:4]:
            print(f"   e.g. [{k}] {e}")
    return {"compared": n, "pm_errors": err, "mismatches": dict(mism)}


def check_awc():
    data = []
    for st in "CA TX FL NY CO AK WA IL MN GA AZ MT".split():  # @XX = all stations in state XX (AWC API syntax)
        url = f"https://aviationweather.gov/api/data/metar?ids=@{st}&format=json&hours=6"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0 (mom-lab research dataset check)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data += json.loads(r.read())
    n = agree = 0; dis = []
    for o in data:
        raw = o.get("rawOb") or ""
        g = decode(raw)
        if g is None or not o.get("fltCat"):
            continue
        n += 1
        if g["flight_category"] == o["fltCat"]:
            agree += 1
        elif len(dis) < 8:
            dis.append(f"{raw} | ours={g['flight_category']} AWC={o['fltCat']}")
    print(f"B. AWC fltCat cross-check: {agree}/{n} agree" + (f" ({100 * agree / n:.2f}%)" if n else ""))
    for d in dis:
        print("   disagree:", d)
    return {"awc_compared": n, "awc_agree": agree}


if __name__ == "__main__":
    res = {"python_metar": check_pm()}
    if "--no-awc" not in sys.argv:
        res["awc"] = check_awc()
    json.dump(res, open(os.path.join(ROOT, "data_partd_c/metar_decoder_validation.json"), "w"), indent=1)

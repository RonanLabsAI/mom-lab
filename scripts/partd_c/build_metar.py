"""METAR decoding task: raw US METAR/SPECI -> fixed JSON schema (scripts/partd_c/metar_decode.py).

Inputs: data_partd_c/raw/metar/iem/*.csv (scripts/partd_c/fetch_metar.py; 377 stations x 24 days of 2024).
Steps: dedupe report texts -> strict decode (reports with any unrecognised body group are dropped) ->
split STATIONS (not reports) into train / dev / test (no station shared) -> stratified sample per split by
flight category (VFR 40%, MVFR 20%, IFR 20%, LIFR 20%; shortfalls filled from the remaining pool) with a
per-station cap so no station dominates.
Outputs: data_partd_c/eval/aviation/metar/{dev,test}.jsonl, data/mixtures/partd_c/metar.jsonl
usage: .venv/bin/python -P scripts/partd_c/build_metar.py
"""
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from metar_decode import FIELDS, decode  # noqa: E402

RAWD = os.path.join(C.RAW, "metar", "iem")
OUT_EVAL = os.path.join(C.DROOT, "eval", "aviation", "metar")
SIZES = {"train": 8000, "dev": 300, "test": 1000}
STATION_FRAC = {"test": 0.20, "dev": 0.08}
CAT_SHARE = {"VFR": 0.4, "MVFR": 0.2, "IFR": 0.2, "LIFR": 0.2}

SPEC = """Decode the METAR below into a JSON object. Decode only the report body (ignore everything from RMK on).

METAR: {raw}

Return only the JSON object (in a ```json block) with exactly these keys, in this order:
- "station": ICAO station identifier (string)
- "day": day of month of the observation (integer)
- "time_utc": observation time as "HH:MM" (string)
- "wind_dir_deg": wind direction in degrees (integer), "VRB" if variable, null if calm (00000KT)
- "wind_speed_kt": wind speed in knots (integer, 0 if calm)
- "wind_gust_kt": gust speed in knots (integer) or null
- "visibility_sm": prevailing visibility in statute miles (number; 1/2 -> 0.5, 1 1/4 -> 1.25; M1/4 -> 0.25; P6 -> 6)
- "weather": list of present-weather groups exactly as written, in report order (e.g. ["-RA", "BR"]); [] if none
- "clouds": list of layers in report order, each {{"cover": "FEW"|"SCT"|"BKN"|"OVC"|"VV", "base_ft": height in feet}} (ignore CB/TCU suffixes); [] if CLR/SKC
- "ceiling_ft": base in feet of the lowest BKN, OVC or VV layer (integer), or null if there is none
- "temperature_c": temperature in deg C from the body temperature group (integer; M = minus)
- "dewpoint_c": dewpoint in deg C (integer) or null if missing
- "altimeter_inhg": altimeter setting in inches of mercury (number; A2992 -> 29.92)
- "flight_category": FAA flight category, the worse of the ceiling and visibility categories: "LIFR" (ceiling below 500 ft or visibility below 1 SM), "IFR" (ceiling 500 to below 1,000 ft or visibility 1 to below 3 SM), "MVFR" (ceiling 1,000 to 3,000 ft or visibility 3 to 5 SM), otherwise "VFR" (no ceiling or ceiling above 3,000 ft, and visibility above 5 SM)"""


def load_reports():
    seen, rows, n_raw, n_drop = set(), [], 0, 0
    files = sorted(glob.glob(os.path.join(RAWD, "*.csv")))
    for p in files:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            n_raw += 1
            raw = " ".join(r["metar"].split())
            if raw in seen:
                continue
            seen.add(raw)
            d = decode(raw)
            if d is None:
                n_drop += 1
                continue
            rows.append({"raw": raw, "gold": d, "iem_station": r["station"], "valid": r["valid"], "file": os.path.basename(p)})
    return rows, {"raw_rows": n_raw, "unique_reports": len(seen), "dropped_undecodable": n_drop, "decoded": len(rows),
                  "files": {os.path.basename(p): C.sha256(p) for p in files}}


def sample(pool, n, rng, cap):
    by_cat = collections.defaultdict(list)
    for r in pool:
        by_cat[r["gold"]["flight_category"]].append(r)
    for k in by_cat:
        by_cat[k].sort(key=lambda r: r["raw"])
        rng.shuffle(by_cat[k])
    per_station, out, chosen = collections.Counter(), [], set()

    def take(cands, k):
        got = 0
        for r in cands:
            if got >= k:
                break
            if id(r) in chosen or per_station[r["gold"]["station"]] >= cap:
                continue
            chosen.add(id(r)); per_station[r["gold"]["station"]] += 1; out.append(r); got += 1
        return got
    for cat, share in CAT_SHARE.items():
        take(by_cat.get(cat, []), round(n * share))
    if len(out) < n:  # fill shortfall from the whole pool
        rest = [r for cat in CAT_SHARE for r in by_cat.get(cat, [])]
        rng.shuffle(rest)
        take(rest, n - len(out))
    return out


def main():
    rows, info = load_reports()
    stations = sorted({r["gold"]["station"] for r in rows})
    rng = C.rng("metar-stations")
    st = stations[:]
    rng.shuffle(st)
    n_te, n_dv = round(len(st) * STATION_FRAC["test"]), round(len(st) * STATION_FRAC["dev"])
    split_of = {s: ("test" if i < n_te else "dev" if i < n_te + n_dv else "train") for i, s in enumerate(st)}
    pools = collections.defaultdict(list)
    for r in rows:
        pools[split_of[r["gold"]["station"]]].append(r)
    out = {}
    for split, n in SIZES.items():
        nst = sum(1 for s in split_of.values() if s == split)
        cap = max(1, -(-3 * n // nst))  # at most ~3x the uniform per-station share
        out[split] = sample(pools[split], n, C.rng(f"metar-sample-{split}"), cap)
    for split in ("dev", "test"):
        items = []
        for r in sorted(out[split], key=lambda r: (r["gold"]["station"], r["raw"])):
            items.append({"id": f"metar-{split}-{len(items):04d}",
                          "messages": [{"role": "user", "content": SPEC.format(raw=r["raw"])}],
                          "reference": r["gold"], "meta": {"raw": r["raw"], "valid_utc": r["valid"], "fields": FIELDS},
                          "max_tokens": 1024})
        C.write_jsonl(os.path.join(OUT_EVAL, f"{split}.jsonl"), items)
    train = []
    for i, r in enumerate(out["train"]):
        train.append({"id": f"metar-train-{i:05d}",
                      "messages": [{"role": "user", "content": SPEC.format(raw=r["raw"])},
                                   {"role": "assistant", "content": "```json\n" + json.dumps(r["gold"]) + "\n```"}],
                      "domain": "aviation", "source": "noaa_metar_iem_2024"})
    C.add_n_tokens(train)
    C.write_jsonl(os.path.join(C.TRAIN, "metar.jsonl"), train)
    info.update({
        "source": "NWS/FAA METAR via Iowa Environmental Mesonet ASOS archive (mesonet.agron.iastate.edu)",
        "license": "public domain (US government observations); IEM archive freely available",
        "stations": {k: sum(1 for s in split_of.values() if s == k) for k in ("train", "dev", "test")},
        "pool": {k: len(v) for k, v in pools.items()},
        "counts": {k: len(v) for k, v in out.items()},
        "train_tokens": sum(r["n_tokens"] for r in train),
        "flight_category": {k: dict(collections.Counter(r["gold"]["flight_category"] for r in v)) for k, v in out.items()},
        "with_weather": {k: sum(1 for r in v if r["gold"]["weather"]) for k, v in out.items()},
        "with_gust": {k: sum(1 for r in v if r["gold"]["wind_gust_kt"]) for k, v in out.items()},
        "station_split": split_of,
    })
    C.update_manifest("metar", info)
    print(json.dumps({k: v for k, v in info.items() if k not in ("station_split", "files")}, indent=1))


if __name__ == "__main__":
    main()

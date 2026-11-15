"""Download raw METARs (routine + SPECI) from the Iowa Environmental Mesonet ASOS archive.

The METAR reports themselves are NWS/FAA observations (US government works, public domain); IEM
(mesonet.agron.iastate.edu) is the archive that serves them. Station lists come from the IEM network
geojson for the 50 US states. Selection is deterministic (seeded), so rerunning reproduces the same
requests; the archive is static for past dates.

Writes data_partd_c/raw/metar/{networks/<NET>.json, stations.json, iem/<YYYY-MM-DD>.csv, fetch_log.json}
usage: .venv/bin/python -P scripts/partd_c/fetch_metar.py
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

OUT = os.path.join(C.RAW, "metar")
STATES = ("AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY "
          "OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY").split()
PER_STATE = 8
DATES = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in (5, 20)]
UA = {"User-Agent": "mom-lab dataset builder (research; contact via github)"}


def get(url, path, tries=5):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                data = r.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data)
            time.sleep(2)  # be polite to IEM (they ask for no parallel hammering)
            return data
        except Exception as e:  # noqa: BLE001
            print(f"  retry {i + 1}: {e}", file=sys.stderr)
            time.sleep(10 * (i + 1))
    raise SystemExit(f"failed: {url}")


def main():
    stations = []
    for st in STATES:
        net = f"{st}_ASOS"
        url = f"https://mesonet.agron.iastate.edu/geojson/network.py?network={net}"
        feats = json.loads(get(url, os.path.join(OUT, "networks", f"{net}.json")))["features"]
        ok = sorted(f["id"] for f in feats
                    if f["properties"].get("online") and (f["properties"].get("archive_begin") or "9999") <= "2015-01-01"
                    and len(f["id"]) == 3 and f["id"].isalnum())
        pick = sorted(C.rng(f"stations-{net}").sample(ok, min(PER_STATE, len(ok))))
        stations += [{"id": s, "state": st} for s in pick]
    json.dump(stations, open(os.path.join(OUT, "stations.json"), "w"), indent=0)
    print(f"{len(stations)} stations")
    log = []
    for d in DATES:
        y, m, dd = map(int, d.split("-"))
        q = [("station", s["id"]) for s in stations] + [
            ("data", "metar"), ("year1", y), ("month1", m), ("day1", dd), ("year2", y), ("month2", m), ("day2", dd),
            ("hour1", 0), ("hour2", 23), ("minute2", 59), ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
            ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", 3), ("report_type", 4)]
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(q)
        path = os.path.join(OUT, "iem", f"{d}.csv")
        data = get(url, path)
        log.append({"date": d, "bytes": len(data), "rows": data.count(b"\n") - 1, "sha256": C.sha256(path)})
        print(log[-1], flush=True)
    json.dump({"stations": len(stations), "dates": DATES, "files": log,
               "url_template": "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=...&data=metar&..."},
              open(os.path.join(OUT, "fetch_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Strict deterministic decoder for US METAR/SPECI reports -> the fixed JSON schema of the metar task.

Only the report body (before RMK) is decoded. Any body token the decoder does not recognise makes it
return None (the report is dropped), so every kept ground truth comes from fully-understood reports.

Schema (key order fixed):
  station          str   ICAO id, e.g. "KSFO"
  day              int   day of month of the observation
  time_utc         str   "HH:MM"
  wind_dir_deg     int | "VRB" | null   direction in degrees true; "VRB" for variable; null when calm (00000KT)
  wind_speed_kt    int   (0 when calm)
  wind_gust_kt     int | null
  visibility_sm    float statute miles (1/4 -> 0.25, 1 1/2 -> 1.5; "M1/4SM" -> 0.25, "P6SM" -> 6)
  weather          list[str] present-weather groups in report order, verbatim (e.g. ["-RA", "BR"]); [] if none
  clouds           list[{"cover": FEW|SCT|BKN|OVC|VV, "base_ft": int}] in report order (CB/TCU suffix dropped);
                   [] for CLR/SKC
  ceiling_ft       int | null  lowest BKN/OVC/VV base; null if none
  temperature_c    int
  dewpoint_c       int | null
  altimeter_inhg   float (A2987 -> 29.87)
  flight_category  "VFR" | "MVFR" | "IFR" | "LIFR"  (FAA / NWS AWC definition, worst of ceiling and visibility):
                   LIFR: ceiling < 500 ft or visibility < 1 SM
                   IFR:  ceiling 500 to < 1,000 ft or visibility 1 to < 3 SM
                   MVFR: ceiling 1,000 to 3,000 ft or visibility 3 to 5 SM
                   VFR:  ceiling > 3,000 ft (or none) and visibility > 5 SM
"""
from __future__ import annotations

import re
from fractions import Fraction

FIELDS = ["station", "day", "time_utc", "wind_dir_deg", "wind_speed_kt", "wind_gust_kt", "visibility_sm",
          "weather", "clouds", "ceiling_ft", "temperature_c", "dewpoint_c", "altimeter_inhg", "flight_category"]

_WIND = re.compile(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT$")
_WVAR = re.compile(r"^\d{3}V\d{3}$")
_VIS = re.compile(r"^([MP])?(\d+|\d/\d{1,2})SM$")
_RVR = re.compile(r"^R\d{2}[LRC]?/[MP]?\d{4}(V[MP]?\d{4})?FT(/[UDN])?$")
_WX = re.compile(r"^(?:[+-]|VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)*$")
_SKY = re.compile(r"^(FEW|SCT|BKN|OVC|VV)(\d{3})(CB|TCU)?$")
_TEMP = re.compile(r"^(M?\d{2})/(M?\d{2})?$")
_ALT = re.compile(r"^A(\d{4})$")
_TIME = re.compile(r"^(\d{2})(\d{2})(\d{2})Z$")


def _t(s):
    return -int(s[1:]) if s.startswith("M") else int(s)


def flight_category(ceiling_ft, vis_sm):
    c = ceiling_ft if ceiling_ft is not None else 10 ** 9
    if c < 500 or vis_sm < 1:
        return "LIFR"
    if c < 1000 or vis_sm < 3:
        return "IFR"
    if c <= 3000 or vis_sm <= 5:
        return "MVFR"
    return "VFR"


def decode(raw: str) -> dict | None:
    body = raw.strip().rstrip("=").split(" RMK")[0]
    tok = body.split()
    i = 0
    if tok and tok[0] in ("METAR", "SPECI"):
        i += 1
    if len(tok) < i + 2 or not re.fullmatch(r"[A-Z][A-Z0-9]{3}", tok[i]):
        return None
    out = {"station": tok[i]}
    i += 1
    m = _TIME.match(tok[i]) if i < len(tok) else None
    if not m:
        return None
    out["day"], out["time_utc"] = int(m[1]), f"{m[2]}:{m[3]}"
    i += 1
    while i < len(tok) and tok[i] in ("AUTO", "COR"):
        i += 1
    m = _WIND.match(tok[i]) if i < len(tok) else None
    if not m:
        return None
    spd = int(m[2])
    gust = int(m[3]) if m[3] else None
    if m[1] == "000" and spd == 0:
        wdir = None
        if gust:
            return None
    else:
        wdir = "VRB" if m[1] == "VRB" else int(m[1])
        if wdir != "VRB" and not (0 < wdir <= 360):
            return None
    out.update(wind_dir_deg=wdir, wind_speed_kt=spd, wind_gust_kt=gust)
    i += 1
    if i < len(tok) and _WVAR.match(tok[i]):
        i += 1
    # visibility: "1 1/2SM" spans two tokens
    vis = None
    if i + 1 < len(tok) and re.fullmatch(r"\d", tok[i]) and _VIS.match(tok[i + 1]) and "/" in tok[i + 1] \
            and not _VIS.match(tok[i + 1])[1]:
        vis = int(tok[i]) + Fraction(_VIS.match(tok[i + 1])[2])
        i += 2
    elif i < len(tok) and _VIS.match(tok[i]):
        mv = _VIS.match(tok[i])
        vis = Fraction(mv[2])
        i += 1
    if vis is None:
        return None
    out["visibility_sm"] = round(float(vis), 4)
    while i < len(tok) and _RVR.match(tok[i]):
        i += 1
    wx = []
    while i < len(tok) and _WX.match(tok[i]) and not _SKY.match(tok[i]) and tok[i] not in ("CLR", "SKC") \
            and not _TEMP.match(tok[i]):
        t = tok[i]
        core = t.lstrip("+-")
        if core.startswith("VC"):
            core = core[2:]
        if not core:
            return None
        wx.append(t)
        i += 1
    out["weather"] = wx
    clouds = []
    if i < len(tok) and tok[i] in ("CLR", "SKC"):
        i += 1
    else:
        while i < len(tok) and _SKY.match(tok[i]):
            ms = _SKY.match(tok[i])
            clouds.append({"cover": ms[1], "base_ft": int(ms[2]) * 100})
            i += 1
        if not clouds:
            return None
    out["clouds"] = clouds
    ceil = [c["base_ft"] for c in clouds if c["cover"] in ("BKN", "OVC", "VV")]
    out["ceiling_ft"] = min(ceil) if ceil else None
    m = _TEMP.match(tok[i]) if i < len(tok) else None
    if not m:
        return None
    out["temperature_c"] = _t(m[1])
    out["dewpoint_c"] = _t(m[2]) if m[2] else None
    i += 1
    m = _ALT.match(tok[i]) if i < len(tok) else None
    if not m:
        return None
    out["altimeter_inhg"] = int(m[1]) / 100
    i += 1
    if i != len(tok):  # trailing unknown body tokens
        return None
    if not (25.0 <= out["altimeter_inhg"] <= 32.5):
        return None
    out["flight_category"] = flight_category(out["ceiling_ft"], out["visibility_sm"])
    return {k: out[k] for k in FIELDS}

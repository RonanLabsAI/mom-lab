"""Unit tests for the Part D / Experiment C scorers (hand-made right / wrong outputs)."""
import json

from harness_d.score import get_scorer, medcalc, metar


def G(**kw):
    return {k: {"id": k, "output": v} for k, v in kw.items()}


def item(i, ref=None, meta=None):
    return {"id": i, "messages": [{"role": "user", "content": "q"}], "reference": ref, "meta": meta or {}}


# ---------------- MedCalc ----------------
DEC = {"calid": 2, "lower_limit": "48.65105", "upper_limit": "53.77221", "category": "lab test"}
INT = {"calid": 4, "lower_limit": "5", "upper_limit": "5", "category": "risk"}
DATE = {"calid": 13, "lower_limit": "06/16/2012", "upper_limit": "06/16/2012", "category": "date"}
GA = {"calid": 69, "lower_limit": "('16 weeks', '1 days')", "upper_limit": "('16 weeks', '1 days')", "category": "date"}
PCT = {"calid": 46, "lower_limit": "0.1", "upper_limit": "0.2", "category": "x"}


def test_medcalc_decimal_limits():
    its = [item(c, "51.21163", DEC) for c in "abcdefgh"]
    r = medcalc.score_medcalc(its, G(
        a="... so CrCl = 51.2 mL/min.\nAnswer: 51.21",       # inside limits
        b="Answer: 48.6",                                  # just below lower limit
        c="Answer: **53.7** mL/min",                       # markdown, inside
        d="The answer is 51.2",                            # no 'Answer:' line -> Not Found
        e='{"step_by_step_thinking": "x", "answer": "50.0"}',  # official JSON fallback
        f="Answer: str(round(140 - 51) * 0.85 * 79 / (1.35 * 72))",  # official str(expr) case -> 61.0 (wrong)
        g="<think>Answer: 51</think>Answer: 10",           # think block dropped
        # h missing
    ))
    assert r["correct"] == [1, 0, 1, 0, 1, 0, 0, 0]


def test_medcalc_integer_date_ga():
    its = [item("a", "5", INT), item("b", "5", INT), item("c", "5", INT), item("d", "06/16/2012", DATE),
           item("e", "06/16/2012", DATE), item("f", "06/16/2012", DATE), item("g", "('16 weeks', '1 days')", GA),
           item("h", "('16 weeks', '1 days')", GA), item("i", "('16 weeks', '1 days')", GA), item("j", "5", INT)]
    r = medcalc.score_medcalc(its, G(
        a="Answer: 5", b="Answer: 4.6", c="Answer: 6 points",
        d="Answer: 6/16/2012", e="Answer: 06/17/2012", f="Answer: June 16, 2012",
        g="Answer: (16 weeks, 1 days)", h="Answer: 16 weeks and 1 day", i="Answer: (16, 2)", j="Answer: 05"))
    # h: the official (weeks, days) regex misreads "16 weeks and 1 day" as (1, 6) -> wrong (faithful to the official code)
    assert r["correct"] == [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]


def test_medcalc_percent_and_unsafe_expr():
    its = [item("a", "0.15", PCT), item("b", "0.15", PCT), item("c", "51.2", DEC)]
    r = medcalc.score_medcalc(its, G(a="Answer: 15%", b="Answer: 15", c="Answer: str(__import__('os').getcwd())"))
    assert r["correct"] == [1, 0, 0]


# ---------------- METAR ----------------
GOLD = {"station": "KORD", "day": 12, "time_utc": "16:51", "wind_dir_deg": 270, "wind_speed_kt": 15, "wind_gust_kt": 25,
        "visibility_sm": 1.5, "weather": ["-SN", "BR"], "clouds": [{"cover": "BKN", "base_ft": 800}, {"cover": "OVC", "base_ft": 1500}],
        "ceiling_ft": 800, "temperature_c": -2, "dewpoint_c": -4, "altimeter_inhg": 29.9, "flight_category": "IFR"}


def _mut(**kw):
    d = json.loads(json.dumps(GOLD)); d.update(kw); return "```json\n" + json.dumps(d) + "\n```"


def test_metar_scorer():
    its = [item(c, GOLD) for c in "abcdefghij"]
    no_gust = json.loads(json.dumps(GOLD)); del no_gust["wind_gust_kt"]
    r = metar.score_metar(its, G(
        a=_mut(),                                              # exact
        b="Here you go:\n" + json.dumps({**GOLD, "station": "kord", "altimeter_inhg": "29.90", "day": 12.0}),  # lenient types, no fence
        c=_mut(flight_category="MVFR"),                        # one field wrong
        d=_mut(weather=["BR", "-SN"]),                         # order matters
        e=_mut(clouds=[{"cover": "BKN", "base_ft": 8}, {"cover": "OVC", "base_ft": 15}]),  # hundreds of feet, wrong
        f="```json\n" + json.dumps(no_gust) + "\n```",         # missing key
        g="not json at all",
        h=_mut(temperature_c=2),                               # sign error
        i=_mut(wind_gust_kt=None),
        # j missing generation
    ))
    assert r["correct"] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert abs(r["extra"]["field_accuracy"]["station"] - 80.0) < 1e-9   # a,b,c,d,e,f,h,i right; g,j wrong
    assert r["extra"]["parse_fail_rate"] == 20.0


def test_metar_null_and_vrb():
    g = {**GOLD, "wind_dir_deg": None, "wind_speed_kt": 0, "wind_gust_kt": None, "ceiling_ft": None, "clouds": [], "weather": []}
    v = {**g, "wind_dir_deg": "VRB", "wind_speed_kt": 3}
    its = [item("a", g), item("b", g), item("c", v), item("d", v)]
    r = metar.score_metar(its, G(a=json.dumps(g), b=json.dumps({**g, "wind_dir_deg": 0}),
                                 c=json.dumps({**v, "wind_dir_deg": "vrb"}), d=json.dumps({**v, "ceiling_ft": 0})))
    assert r["correct"] == [1, 0, 1, 0]


# ---------------- Pre-Flight ----------------
def test_preflight_letter():
    its = [item(c, "B") for c in "abcd"]
    r = get_scorer("preflight")(its, G(a="ANSWER: B", b="ANSWER: C", c="The answer is (B).", d="I am not sure"))
    assert r["correct"] == [1, 0, 1, 0]


def test_registry():
    from harness_d import common as C
    assert C.BENCH_DOMAIN["metar"] == "aviation" and C.BENCH_DOMAIN["medcalc"] == "medcalc"
    assert C.BENCH_DOMAIN["preflight"] == "avknow"
    for b in ("metar", "medcalc", "preflight", "gsm8k", "bird_dev"):
        get_scorer(b)

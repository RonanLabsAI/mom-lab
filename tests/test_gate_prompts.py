"""Schema-arm prompt variant (prereg: the fence instruction conflicts with a grammar that cannot emit backticks)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bsp", ROOT / "scripts/gate/build_schema_prompts.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

FENCED = "Return only the JSON object (in a ```json block) with exactly these keys"
RAW = "Return only the JSON object (raw JSON, no code fence) with exactly these keys"


def test_user_turn_rewritten():
    it = {"id": "x", "messages": [{"role": "user", "content": "METAR: K 0Z\n\n" + FENCED + ", in order"}]}
    out = B.transform(it)
    assert out["messages"][0]["content"] == "METAR: K 0Z\n\n" + RAW + ", in order"
    assert it["messages"][0]["content"].startswith("METAR")  # input not mutated
    assert FENCED in it["messages"][0]["content"]


def test_assistant_unfenced_and_still_json():
    body = '{"station": "KEFK", "day": 20}'
    it = {"id": "x", "messages": [{"role": "user", "content": FENCED}, {"role": "assistant",
                                  "content": "```json\n" + body + "\n```"}, {"role": "user", "content": FENCED}]}
    out = B.transform(it)
    assert out["messages"][1]["content"] == body
    json.loads(out["messages"][1]["content"])
    assert out["messages"][2]["content"] == RAW


def test_missing_phrase_is_an_error():
    it = {"id": "x", "messages": [{"role": "user", "content": "no instruction here"}]}
    try:
        B.transform(it)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

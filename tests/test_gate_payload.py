"""The gate wrapper must add exactly one request field and otherwise send what harness_d.generate sends."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_d import generate as G  # noqa: E402

SCHEMA = ROOT / "configs/gate/metar_schema.json"
spec = importlib.util.spec_from_file_location("schema_generate", ROOT / "scripts/gate/schema_generate.py")
SG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SG)

ITEM = {"id": "metar-test-0000", "messages": [{"role": "user", "content": "Decode ..."}], "max_tokens": 1024}
BASE_ARGS = ["--endpoint", "http://127.0.0.1:4000/v1", "--model", "research-mom-gate-qwen3-8b", "--run-id", "r",
             "--step", "partdG", "--max-tokens-override", "4096"]


def test_payload_unchanged_without_schema():
    args = SG.build_parser().parse_args(BASE_ARGS)
    assert SG.make_payload_fn(None)(ITEM, args.model, args) == G.build_payload(ITEM, args.model, args)


def test_payload_has_strict_schema():
    args = SG.build_parser().parse_args(BASE_ARGS + ["--schema", str(SCHEMA)])
    p = SG.make_payload_fn(SG.load_schema(args.schema))(ITEM, args.model, args)
    base = G.build_payload(ITEM, args.model, args)
    rf = p.pop("response_format")
    assert p == base
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "metar"
    assert rf["json_schema"]["schema"] == json.loads(SCHEMA.read_text())


def test_auth_client_sends_bearer():
    shim = SG.auth_httpx("k-test")
    c = shim.AsyncClient()
    assert c.headers["Authorization"] == "Bearer k-test"
    assert shim.HTTPError is not None and shim.Limits is not None


def test_thinking_args_pass_through():
    args = SG.build_parser().parse_args(BASE_ARGS + ["--schema", str(SCHEMA), "--thinking", "--temperature", "0.6",
                                                     "--top-p", "0.95", "--top-k", "20", "--seed", "0"])
    p = SG.make_payload_fn(SG.load_schema(args.schema))(ITEM, args.model, args)
    assert p["chat_template_kwargs"] == {"enable_thinking": True}
    assert p["top_k"] == 20 and p["temperature"] == 0.6 and p["seed"] == 0

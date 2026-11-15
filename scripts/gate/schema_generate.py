"""Gate generation wrapper (preregs/2026-11-14-metar-schema-gate.md).

harness_d.generate with one optional addition: --schema PATH adds
    response_format = {"type": "json_schema", "json_schema": {"name": "metar", "schema": <file>, "strict": true}}
to every request. Without --schema the request is exactly what harness_d.generate sends. harness_d/ is frozen and is
not edited; this file swaps its build_payload at run time.

usage (from the repo root):
  MOM_DATA_ROOT=<root> python -P scripts/gate/schema_generate.py <harness_d.generate args> [--schema PATH]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from harness_d import generate as G  # noqa: E402

_BASE_BUILD = G.build_payload


def load_schema(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def make_payload_fn(schema: dict | None):
    def build(item, model, args):
        p = _BASE_BUILD(item, model, args)
        if schema is not None:
            p["response_format"] = {"type": "json_schema",
                                    "json_schema": {"name": "metar", "schema": schema, "strict": True}}
        return p
    return build


def auth_httpx(key: str | None):
    """A stand-in for the httpx module inside harness_d.generate whose AsyncClient sends the router key.
    The key is never logged or written to metadata."""
    import types
    import httpx

    class AuthClient(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            if key:
                kw.setdefault("headers", {})["Authorization"] = f"Bearer {key}"
            super().__init__(*a, **kw)

    return types.SimpleNamespace(AsyncClient=AuthClient, Limits=httpx.Limits, Timeout=httpx.Timeout,
                                 HTTPError=httpx.HTTPError)


def build_parser():
    ap = G.build_parser()
    ap.prog = "python -P scripts/gate/schema_generate.py"
    ap.add_argument("--schema", help="JSON schema file; adds response_format json_schema (strict)")
    ap.add_argument("--api-key-file", help="file holding the LiteLLM router key (sent as a Bearer header)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    G.build_payload = make_payload_fn(load_schema(args.schema))
    if args.api_key_file:
        G.httpx = auth_httpx(Path(args.api_key_file).expanduser().read_text().strip())
    rc = asyncio.run(G.amain(args))
    # record the schema in the run metadata next to harness_d's own fields
    rdir = G.C.run_dir(args.step, args.run_id, Path(args.results_root))
    mp = rdir / "metadata.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    meta["gate"] = {"schema_path": args.schema, "schema": load_schema(args.schema),
                    "wrapper": "scripts/gate/schema_generate.py"}
    mp.write_text(json.dumps(meta, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()

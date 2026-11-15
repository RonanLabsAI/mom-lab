"""Generate model outputs for eval items via an OpenAI-compatible endpoint.

python -m harness.generate --endpoint http://HOST:PORT/v1 --model NAME --run-id ID \
    --step STEP --split dev [--domains math,code] [--benchmarks gsm8k] [--concurrency 64] [--limit N]

Output: results/<step>/<run_id>/gen/<domain>/<bench>.<split>.jsonl (resumable: ids already present
are skipped) and results/<step>/<run_id>/metadata.json.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import httpx

from . import common as C

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


async def call_model(client: httpx.AsyncClient, url: str, payload: dict, retries: int,
                     base_delay: float) -> tuple[dict, float]:
    """POST with exponential backoff. Returns (json, latency_s of the successful attempt)."""
    last = None
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                return r.json(), time.monotonic() - t0
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in RETRY_STATUS:
                break
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            await asyncio.sleep(min(60.0, base_delay * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(last)


def build_payload(item: dict, model: str, args) -> dict:
    msgs = item["messages"]
    if getattr(args, "domain_tag", None):
        # tagged-pooled arm: same "[domain: X]" prefix as in training (train/train.py --domain-tag)
        msgs = [dict(m) for m in msgs]
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"[domain: {args.domain_tag_value or item.get('_domain')}]\n" + m["content"]
                break
    p = {
        "model": model,
        "messages": msgs,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens_override or int(item.get("max_tokens") or args.default_max_tokens),
        # what openai-python's extra_body does: vLLM reads this as a top-level JSON field
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    if args.top_k:
        p["top_k"] = args.top_k
    if args.presence_penalty:
        p["presence_penalty"] = args.presence_penalty
    return p


async def run_benchmark(client, args, domain, bench, items, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for g in C.read_jsonl(out_path):
            done.add(g["id"])
    todo = [dict(it, _domain=domain) for it in items if it["id"] not in done]
    print(f"[{domain}/{bench}] {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return 0, 0
    sem = asyncio.Semaphore(args.concurrency)
    url = args.endpoint.rstrip("/") + "/chat/completions"
    n_ok = n_fail = 0
    fout = open(out_path, "a", encoding="utf-8")

    async def one(it):
        nonlocal n_ok, n_fail
        async with sem:
            try:
                data, lat = await call_model(client, url, build_payload(it, args.model, args),
                                             args.retries, args.backoff)
                ch = data["choices"][0]
                usage = data.get("usage") or {}
                rec = {
                    "id": it["id"],
                    "output": (ch.get("message") or {}).get("content") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "n_out_tokens": usage.get("completion_tokens"),
                    "latency_s": round(lat, 4),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
            except Exception as e:  # not written -> retried on resume
                n_fail += 1
                print(f"  FAIL {it['id']}: {e}", file=sys.stderr, flush=True)
            tot = n_ok + n_fail
            if tot % 50 == 0:
                print(f"  [{domain}/{bench}] {tot}/{len(todo)}", flush=True)

    try:
        await asyncio.gather(*(one(it) for it in todo))
    finally:
        fout.close()
    return n_ok, n_fail


def write_metadata(rdir: Path, args):
    path = rdir / "metadata.json"
    meta = {}
    if path.exists():
        meta = json.loads(path.read_text())
    meta.update({
        "model": args.model,
        "revision": args.revision,
        "endpoint": args.endpoint,
        "hardware": args.hardware,
        "run_id": args.run_id,
        "step": args.step,
        "settings": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                     "presence_penalty": args.presence_penalty, "seed": args.seed,
                     "max_tokens_override": args.max_tokens_override,
                     "chat_template_kwargs": {"enable_thinking": args.thinking},
                     "default_max_tokens": args.default_max_tokens,
                     "limit": args.limit},
        "git_hash": C.git_hash(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })
    path.write_text(json.dumps(meta, indent=2))


async def amain(args) -> int:
    C.check_split(args.split, args.allow_test)
    if args.router_file:
        raise NotImplementedError("--router-file is not implemented yet (stub).")
    domains = args.domains.split(",") if args.domains else list(C.DOMAINS)
    benches = set(args.benchmarks.split(",")) if args.benchmarks else None
    data_root = Path(args.data_root)
    rdir = C.run_dir(args.step, args.run_id, Path(args.results_root))
    rdir.mkdir(parents=True, exist_ok=True)
    write_metadata(rdir, args)
    total_fail = 0
    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits) as client:
        for d in domains:
            for b in C.ALL_DOMAINS[d]:
                if benches and b not in benches:
                    continue
                p = C.eval_path(d, b, args.split, data_root)
                if not p.exists():
                    print(f"[{d}/{b}] missing {p}, skipped", file=sys.stderr)
                    continue
                items = C.read_jsonl(p)
                if args.limit:
                    items = items[: args.limit]
                _, nf = await run_benchmark(client, args, d, b, items,
                                            C.gen_path(rdir, d, b, args.split))
                total_fail += nf
    if total_fail:
        print(f"{total_fail} items failed after retries; rerun the same command to resume.", file=sys.stderr)
    return 1 if total_fail else 0


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m harness.generate")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--domains")
    ap.add_argument("--benchmarks")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--backoff", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--default-max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode")
    ap.add_argument("--max-tokens-override", type=int, default=0)
    ap.add_argument("--domain-tag", action="store_true", help="prefix '[domain: <item domain>]' to the user message")
    ap.add_argument("--domain-tag-value", default="", help="force a tag value (default: the benchmark's domain)")
    ap.add_argument("--revision", default="")
    ap.add_argument("--hardware", default="")
    ap.add_argument("--router-file", help="(stub) routed generation, not implemented yet")
    ap.add_argument("--data-root", default=str(C.DATA))
    ap.add_argument("--results-root", default=str(C.RESULTS))
    return ap


def main(argv=None):
    sys.exit(asyncio.run(amain(build_parser().parse_args(argv))))


if __name__ == "__main__":
    main()

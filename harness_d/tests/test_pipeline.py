import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from harness import aggregate, generate, stats
from harness import common as C

REQS = []
FAIL_ONCE = {"n": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQS.append(body)
        if FAIL_ONCE["n"] > 0:
            FAIL_ONCE["n"] -= 1
            self.send_response(503)
            self.end_headers()
            return
        q = body["messages"][-1]["content"]
        ans = {"2+2": "\\boxed{4}", "3+3": "\\boxed{6}"}.get(q, "\\boxed{0}")
        resp = {"choices": [{"message": {"content": ans}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    REQS.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def write_eval(root, domain, bench, split, rows):
    p = root / "eval" / domain / bench / f"{split}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def mk(i, q, ref):
    return {"id": i, "messages": [{"role": "user", "content": q}], "reference": ref, "meta": {}, "max_tokens": 64}


def gen_args(server, tmp_path, run="r1", **kw):
    argv = ["--endpoint", server, "--model", "fake", "--run-id", run, "--step", "s0", "--split", "dev",
            "--data-root", str(tmp_path / "data"), "--results-root", str(tmp_path / "results"),
            "--backoff", "0.01", "--hardware", "cpu-test"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return generate.build_parser().parse_args(argv)


def test_generate_aggregate_stats(server, tmp_path):
    data = tmp_path / "data"
    write_eval(data, "math", "gsm8k", "dev", [mk("1", "2+2", "4"), mk("2", "3+3", "6"), mk("3", "9+9", "18")])
    FAIL_ONCE["n"] = 2  # first two calls get 503 -> retried
    rc = asyncio.run(generate.amain(gen_args(server, tmp_path, domains="math", concurrency=4)))
    assert rc == 0
    p = tmp_path / "results/s0/r1/gen/math/gsm8k.dev.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == ["1", "2", "3"]
    assert all(r["finish_reason"] == "stop" and r["n_out_tokens"] == 5 and r["latency_s"] >= 0 for r in rows)
    body = REQS[-1]
    assert body["temperature"] == 0 and body["top_p"] == 1 and body["seed"] == 0
    assert body["max_tokens"] == 64 and body["chat_template_kwargs"] == {"enable_thinking": False}
    meta = json.loads((tmp_path / "results/s0/r1/metadata.json").read_text())
    assert meta["model"] == "fake" and meta["hardware"] == "cpu-test" and "git_hash" in meta

    # resume: nothing is re-requested
    n = len(REQS)
    asyncio.run(generate.amain(gen_args(server, tmp_path, domains="math")))
    assert len(REQS) == n

    res = aggregate.aggregate("s0", "r1", "dev", data_root=data, results_root=tmp_path / "results", workers=1)
    assert res["benchmarks"]["gsm8k"]["score"] == pytest.approx(200 / 3)
    assert res["benchmarks"]["gsm8k"]["correct"] == [1, 1, 0]
    assert res["A"] is None and res["complete"] is False  # only one domain present
    assert res["metadata"]["model"] == "fake"
    assert (tmp_path / "results/s0/r1/scores.json").exists()


def test_limit_and_test_split_guard(server, tmp_path):
    data = tmp_path / "data"
    write_eval(data, "math", "gsm8k", "dev", [mk(str(i), "2+2", "4") for i in range(10)])
    asyncio.run(generate.amain(gen_args(server, tmp_path, domains="math", limit=3)))
    assert len(REQS) == 3
    with pytest.raises(SystemExit):
        asyncio.run(generate.amain(generate.build_parser().parse_args(
            ["--endpoint", server, "--model", "m", "--run-id", "x", "--step", "s", "--split", "test"])))


def test_router_stub(server, tmp_path):
    with pytest.raises(NotImplementedError):
        asyncio.run(generate.amain(gen_args(server, tmp_path, router_file="r.json")))


def synth_scores(rng, p, n=200, dirs=False):
    ids = [str(i) for i in range(n)]
    return ids, (rng.random(n) < p).astype(float).tolist()


def fake_scores(p_math, p_code, seed):
    rng = np.random.default_rng(seed)
    out = {"benchmarks": {}, "domains": {}}
    for b, d, p in [("gsm8k", "math", p_math), ("math500", "math", p_math), ("humaneval_plus", "code", p_code)]:
        ids, c = synth_scores(rng, p)
        out["benchmarks"][b] = {"domain": d, "score": 100 * float(np.mean(c)), "ids": ids, "correct": c}
    for d in ("math", "code"):
        out["domains"][d] = float(np.mean([e["score"] for e in out["benchmarks"].values() if e["domain"] == d]))
    out["A"] = None
    out["A_partial"] = float(np.mean(list(out["domains"].values())))
    return out


def test_bootstrap_ci_and_paired():
    a = fake_scores(0.5, 0.5, 1)
    ci = stats.bootstrap_ci(a)
    for k in ("gsm8k", "domain:math", "A"):
        assert ci[k]["ci_lo"] < ci[k]["point"] < ci[k]["ci_hi"]
        assert ci[k]["ci_hi"] - ci[k]["ci_lo"] < 15
    assert stats.bootstrap_ci(a) == ci  # deterministic (seed 0)

    # same items, B clearly better on math (+0.3) but equal on code
    b = json.loads(json.dumps(a))
    rng = np.random.default_rng(7)
    for bn in ("gsm8k", "math500"):
        ent = b["benchmarks"][bn]
        ent["correct"] = [1.0 if (c == 1 or rng.random() < 0.6) else 0.0 for c in ent["correct"]]
        ent["score"] = 100 * float(np.mean(ent["correct"]))
    d = stats.paired_diff(a, b)
    assert d["domain:math"]["diff"] > 15 and d["domain:math"]["p"] < 0.01
    assert d["domain:math"]["ci_lo"] > 0
    assert d["humaneval_plus"]["diff"] == 0 and d["humaneval_plus"]["p"] == 1.0
    same = stats.paired_diff(a, a)
    assert same["A"]["diff"] == 0 and same["A"]["p"] == 1.0

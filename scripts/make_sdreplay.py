"""Part D (D): self-distilled replay. Same mixture as data/mixtures/N14M/replay/<d>-r20.jsonl, but every replay
example (a line not in the domain's own spec file) gets untouched Qwen3-4B's own greedy, non-thinking answer
(max 2,048 tokens) in place of the public reference. Truncated or failed answers are dropped. Own-domain lines unchanged.
Output: data/mixtures/N14M/sdreplay/<d>-r20.jsonl (same order as the replay file). Counts printed.
usage: .venv/bin/python -P scripts/make_sdreplay.py --endpoint http://host-b693d1b8.invalid:8698 --model base sql tools
"""
import argparse, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "data/mixtures/N14M"


def chat(ep, model, msgs):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": msgs}).encode()
    err = None
    for _ in range(3):
        try:
            req = urllib.request.Request(ep + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                c = json.loads(r.read())["choices"][0]
                return c["message"]["content"].strip(), c.get("finish_reason")
        except Exception as e:
            err = e
    return None, f"error: {err}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--endpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=256); ap.add_argument("domains", nargs="+")
    a = ap.parse_args()
    (M / "sdreplay").mkdir(exist_ok=True)
    for d in a.domains:
        own = set(open(M / f"spec/{d}.jsonl").readlines())
        lines = open(M / f"replay/{d}-r20.jsonl").readlines()
        rows = [json.loads(l) for l in lines]
        idx = [i for i, l in enumerate(lines) if l not in own]

        def run(i):
            msgs = [m for m in rows[i]["messages"] if m["role"] != "assistant"]
            return chat(a.endpoint, a.model, msgs)
        with ThreadPoolExecutor(a.workers) as ex:
            outs = dict(zip(idx, ex.map(run, idx)))
        out, kept, dropped = [], 0, 0
        for i, r in enumerate(rows):
            if i in outs:
                txt, fin = outs[i]
                if txt is None or fin != "stop" or not txt:
                    dropped += 1; continue
                r = json.loads(json.dumps(r))
                r["messages"] = [m for m in r["messages"] if m["role"] != "assistant"] + [{"role": "assistant", "content": txt}]
                kept += 1
            out.append(json.dumps(r, ensure_ascii=False) + "\n")
        open(M / f"sdreplay/{d}-r20.jsonl", "w").writelines(out)
        print(d, "own", len(rows) - len(idx), "replay kept", kept, "dropped", dropped, flush=True)


if __name__ == "__main__":
    main()

"""Self-distillation data (SDFT-style) for Step 6: the post-trained base model rewrites each training answer in its
own words, conditioned on the reference answer, so the specialist trains on targets close to its own distribution
(expected: less forgetting, less phrasing brittleness). Output: data/mixtures/N14M/sdft/<domain>.jsonl.

Acceptance (else the reference answer is kept, and the example is counted as 'fallback'):
  tools: the ```json call list must parse and equal the reference call list
  sql:   a ```sql block must be present
  code/other: must contain a code fence if the reference does; non-empty
usage: .venv/bin/python -P scripts/make_sdft.py --endpoint http://<station>:<port> --model <served name> <domain>...
"""
import argparse, json, re, threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request

HINT = ("\n\n(For reference, here is a correct answer:\n{ref}\n\nNow write your own complete answer to the request "
        "above. Follow its format instructions exactly. Do not mention the reference.)")


def chat(ep, model, user):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": user}]}).encode()
    for _ in range(3):
        try:
            req = urllib.request.Request(ep + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                c = json.loads(r.read())["choices"][0]
                return c["message"]["content"].strip(), c.get("finish_reason")
        except Exception as e:
            err = e
    raise err


def calls(t):
    m = re.search(r"```json\s*(.*?)```", t, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except Exception:
        return None


def accept(domain, ref, out, fin):
    if fin != "stop" or not out:
        return False
    if domain == "tools":
        a = calls(out)
        return a is not None and a == calls(ref)
    if domain == "sql":
        return "```sql" in out
    return ("```" in out) or ("```" not in ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=128); ap.add_argument("--limit", type=int)
    ap.add_argument("domains", nargs="+")
    a = ap.parse_args()
    import os; os.makedirs("data/mixtures/N14M/sdft", exist_ok=True)
    for d in a.domains:
        rows = [json.loads(l) for l in open(f"data/mixtures/N14M/spec/{d}.jsonl")][: a.limit or None]
        stats = {"accepted": 0, "fallback": 0}; lock = threading.Lock()

        def run(r):
            user = r["messages"][-2]["content"]; ref = r["messages"][-1]["content"]
            out, fin = chat(a.endpoint, a.model, user + HINT.format(ref=ref))
            ok = accept(d, ref, out, fin)
            with lock:
                stats["accepted" if ok else "fallback"] += 1
            n = json.loads(json.dumps(r)); n["messages"][-1]["content"] = out if ok else ref; n["sdft"] = ok
            return n
        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(run, rows))
        with open(f"data/mixtures/N14M/sdft/{d}.jsonl", "w") as f:
            for n in outs:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        print(d, len(outs), stats, flush=True)


if __name__ == "__main__":
    main()

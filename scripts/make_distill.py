"""Distillation data for the H3 baseline (Part B): the adapter library (s0) answers every pooled training prompt.
Prompts whose training domain is routed (1.7B: code, sql, tools; 4B: sql, tools) go to that domain's LoRA adapter;
all other prompts go to the untouched base. Greedy, non-thinking, max 4096 new tokens; truncated answers
(finish_reason != stop) and empty answers are dropped. Output: data/mixtures/N14M/distill/<size>.jsonl.

Serve the base with LoRA modules named after the domains, e.g. (station 6000):
  bash serve.sh 0 8690 /models/hf/Qwen__Qwen3-1.7B base --enable-lora --max-lora-rank 16 --max-loras 3 \
    --lora-modules code=/lab/adapters/pt17-lora16-code-lr2e-4 sql=/lab/adapters/pt17-lora16-sql-lr2e-4 ...
usage: .venv/bin/python -P scripts/make_distill.py --endpoint http://<station>:<port> --size 17|4 [--limit N]
         [--only base|lora] [--tag NAME]
Resumable: answers are appended to <out>.partial (or <out>.<tag>.partial); a rerun skips ids present in ANY
<out>*.partial, so several servers can work disjoint shards (--only base: base-routed prompts, served without LoRA;
--only lora: adapter-routed prompts). Assembly merges every partial file.
"""
import argparse, glob, json, os, threading
from concurrent.futures import ThreadPoolExecutor
import urllib.error, urllib.request

ROUTES = {"17": ["code", "sql", "tools"], "4": ["sql", "tools"]}


def chat(ep, model, user):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": user}]}).encode()
    for _ in range(4):
        try:
            req = urllib.request.Request(ep + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                c = json.loads(r.read())["choices"][0]
                return c["message"]["content"], c.get("finish_reason")
        except urllib.error.HTTPError as e:
            if e.code == 400:  # prompt + 4096 new tokens exceed max-model-len 8192: dropped like a truncation
                return "", "error"
            err = e
        except Exception as e:
            err = e
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--size", required=True, choices=list(ROUTES))
    ap.add_argument("--workers", type=int, default=256); ap.add_argument("--limit", type=int)
    ap.add_argument("--only", choices=["base", "lora"]); ap.add_argument("--tag")
    a = ap.parse_args()
    os.makedirs("data/mixtures/N14M/distill", exist_ok=True)
    out = f"data/mixtures/N14M/distill/{a.size}.jsonl"; part = out + (f".{a.tag}" if a.tag else "") + ".partial"
    rows = [json.loads(l) for l in open("data/mixtures/N14M/pooled.jsonl")][: a.limit or None]
    done = set()
    for pf in glob.glob(out + "*.partial"):
        for l in open(pf):
            done.add(json.loads(l)["id"])
    lock = threading.Lock(); f = open(part, "a")

    def run(r):
        model = r["domain"] if r["domain"] in ROUTES[a.size] else "base"
        text, fin = chat(a.endpoint, model, r["messages"][0]["content"])
        rec = {"id": r["id"], "teacher": model, "finish": fin, "output": text}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    routed = lambda r: r["domain"] in ROUTES[a.size]
    todo = [r for r in rows if r["id"] not in done and (not a.only or (a.only == "lora") == routed(r))]
    print(f"todo {len(todo)} of {len(rows)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, todo))
    f.close()
    ans = {}
    for pf in sorted(glob.glob(out + "*.partial")):
        for l in open(pf):
            d = json.loads(l); ans[d["id"]] = d
    if any(r["id"] not in ans for r in rows):
        print(f"shard done; {sum(r['id'] not in ans for r in rows)} prompts still unanswered: not assembling", flush=True)
        return
    kept = 0; dropped = {}
    with open(out, "w") as g:
        for r in rows:
            d = ans.get(r["id"])
            if not d or d["finish"] != "stop" or not d["output"].strip():
                why = "missing" if not d else d["finish"] if d["finish"] != "stop" else "empty"
                dropped[why] = dropped.get(why, 0) + 1; continue
            n = {k: v for k, v in r.items() if k != "n_tokens"}
            n["messages"] = [r["messages"][0], {"role": "assistant", "content": d["output"]}]
            n["teacher"] = d["teacher"]; kept += 1
            g.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"size {a.size}: kept {kept}, dropped {dropped}", flush=True)


if __name__ == "__main__":
    main()

"""Gap check: send the Reasoning Gym puzzle set to one OpenAI-compatible endpoint, score with Reasoning Gym's graders.

Resumable by item id. Usage:
  run_gap.py --endpoint http://host-3cc546d0.invalid:8710/v1 --model qwen3-4b --out results/qwen3-4b.jsonl [--no-think]
"""
import argparse, asyncio, json, os, time
import httpx
import reasoning_gym as rg
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

p = argparse.ArgumentParser()
p.add_argument("--endpoint", required=True)
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "gap_v1.jsonl"))
p.add_argument("--no-think", action="store_true")
p.add_argument("--max-tokens", type=int, default=16384)
p.add_argument("--concurrency", type=int, default=128)
p.add_argument("--api-key", default="none")
p.add_argument("--extra", default="{}", help="extra JSON merged into the request body")
a = p.parse_args()

items = [json.loads(l) for l in open(a.data)]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
todo = [it for it in items if it["id"] not in done]

# Rebuild each task's dataset from its seed so graders get the full entry (metadata included).
_ds = {}
def entry(it):
    key = (it["task"], it["seed"])
    if key not in _ds:
        _ds[key] = rg.create_dataset(it["task"], size=50, seed=it["seed"])
    return _ds[key], _ds[key][it["idx"]]

async def one(client, sem, it, fout):
    body = {"model": a.model, "max_tokens": a.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM_PROMPTS["default"]},
                         {"role": "user", "content": it["question"]}]}
    if a.no_think:
        body.update(temperature=0.7, top_p=0.8, top_k=20, chat_template_kwargs={"enable_thinking": False})
    else:
        body.update(temperature=0.6, top_p=0.95, top_k=20, chat_template_kwargs={"enable_thinking": True})
    body.update(json.loads(a.extra))
    async with sem:
        t0 = time.time()
        for attempt in range(3):
            try:
                r = await client.post(f"{a.endpoint}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {a.api_key}"})
                r.raise_for_status()
                j = r.json(); break
            except Exception as e:
                if attempt == 2:
                    j = {"error": str(e)[:300]}
                await asyncio.sleep(5)
    if "error" in j:
        rec = {"id": it["id"], "task": it["task"], "error": j["error"], "score": 0.0}
    else:
        msg = j["choices"][0]["message"]
        text = msg.get("content") or ""
        pred = extract_answer(text)
        ds, e = entry(it)
        try:
            score = float(ds.score_answer(pred, e))
        except Exception:
            score = 0.0
        rec = {"id": it["id"], "task": it["task"], "pred": pred, "score": score,
               "finish": j["choices"][0].get("finish_reason"),
               "completion_tokens": j.get("usage", {}).get("completion_tokens"),
               "secs": round(time.time() - t0, 1), "content": text[-2000:],
               "reasoning_len": len(msg.get("reasoning_content") or msg.get("reasoning") or "")}
    fout.write(json.dumps(rec) + "\n"); fout.flush()

async def main():
    sem = asyncio.Semaphore(a.concurrency)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    async with httpx.AsyncClient(timeout=7200) as client:
        with open(a.out, "a") as fout:
            await asyncio.gather(*(one(client, sem, it, fout) for it in todo))
    print(f"{a.model}: wrote {len(todo)} new, {len(done)} already done")

asyncio.run(main())

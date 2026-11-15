"""Part C: build a fresh paraphrase set from TEST-split source items that are in neither paraphrase set built so far.

Same rewriter, styles, protected literals and checks as scripts/build_para.py (imported, not modified; that file is
frozen in the Part B manifest). Differences: items already drawn for data/eval/para/<bench>/test.jsonl (including
ones dropped during that build) are excluded, the draw uses seed SEED+2, ids are "pp2-<source id>", and gsm8k
additionally requires every number in the question to appear as digits in the rewrite (closes the "one" loophole
found in the Part B test build). Output: data_partc/eval/para/<bench>/test.jsonl, a separate data root, so the
harness reads it with --data-root data_partc.

Usage: .venv/bin/python -P scripts/build_para2.py --endpoint http://host-3cc546d0.invalid:8681
"""
import argparse, importlib.util, json, random, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_para", ROOT / "scripts/build_para.py")
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
bp.CACHE = ROOT / "logs/para2/cache.jsonl"; bp.FLAGS = ROOT / "logs/para2/flags.jsonl"
OUT = ROOT / "data_partc/eval/para"
N = {"gsm8k_pp": 300, "bird_pp": 300, "mbpp_pp": 10**6, "bfcl_pp": 300, "mmlupro_pp": 300, "flores_pp": 300}


def draw(items, n, bench, used):
    pool = [it for it in items if it["id"] not in used]
    rng = random.Random(bp.SEED + 2)
    if n >= len(pool):
        chosen = pool
    elif bench == "bfcl_pp":
        by = defaultdict(list)
        for it in pool: by[it["meta"]["category"]].append(it)
        chosen = []
        for k, v in sorted(by.items()):
            chosen += rng.sample(v, min(len(v), round(n * len(v) / len(pool))))
    else:
        chosen = rng.sample(pool, n)
    order = {it["id"]: i for i, it in enumerate(items)}
    return sorted(chosen, key=lambda x: order[x["id"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--bench", default=",".join(bp.BENCH))
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    bp.CACHE.parent.mkdir(parents=True, exist_ok=True)
    for b in a.bench.split(","):
        cfg = bp.BENCH[b]
        items = [json.loads(l) for l in open(ROOT / "data/eval" / cfg["src"] / "test.jsonl", encoding="utf-8")]
        used = {it["id"] for it in bp.sample(items, cfg["n"], b)}  # the Part B draw, incl. items dropped then
        used |= {json.loads(l)["meta"]["source_id"] for l in open(ROOT / "data/eval/para" / b / "test.jsonl", encoding="utf-8")}
        chosen = draw(items, N[b], b, used)[: a.limit or None]
        assert not {it["id"] for it in chosen} & used

        def run(it):
            c = it["messages"][0]["content"]
            sk, lits, keep, what = cfg["sk"](c, it["meta"]) if b == "flores_pp" else cfg["sk"](c)
            style = bp.STYLES[int(re.sub(r"\D", "", it["id"]) or 0) % len(bp.STYLES)]
            return bp.rewrite(a.endpoint, "p2:" + it["id"], sk, lits, [k for k in set(keep) if len(k) > 0], what, style)
        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(run, chosen))
        rows, dropped = [], 0
        for it, (content, flag) in zip(chosen, outs):
            if content is None:
                dropped += 1; continue
            if b == "gsm8k_pp":
                q = it["messages"][0]["content"].rsplit("\n\n", 1)[0]
                body = content.replace("\\boxed{}", "")
                if any(not re.search(r"(?<![\d.])" + re.escape(x) + r"(?![\d])", body) for x in re.findall(r"\d+(?:\.\d+)?", q)):
                    dropped += 1; continue
            r = json.loads(json.dumps(it)); r["id"] = f"pp2-{it['id']}"; r["messages"][0]["content"] = content
            r.setdefault("meta", {}); r["meta"].update(source_domain=cfg["domain"], source_id=it["id"], source_benchmark=cfg["scorer"])
            rows.append(r)
        d = OUT / b; d.mkdir(parents=True, exist_ok=True)
        with open(d / "test.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(b, len(rows), "of", len(chosen), "(dropped", dropped, "; excluded", len(used), ")", flush=True)


if __name__ == "__main__":
    main()

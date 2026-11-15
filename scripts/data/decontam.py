"""13-gram decontamination of data/train_raw/<domain>/all.jsonl against all eval sets (dev+test), then writes pool/heldout."""
import sys, os, re, json, glob, random, multiprocessing as mp; sys.path.insert(0, os.path.dirname(__file__))
from common import *
N = 13
CJK = re.compile(r"[぀-ヿ㐀-鿿]")
RUN = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uff00-\uffef]{13,}")
def grams(text):
    t = re.sub(r"\s+", " ", text.lower()).strip()
    w = t.split(" ")
    g = {hash(" ".join(w[i:i+N])) for i in range(len(w) - N + 1)}
    for run in RUN.findall(t.replace(" ", "")):
        g |= {hash("c:" + run[i:i+N]) for i in range(len(run) - N + 1)}
    return g
def strip_template(bench, it):
    u = it["messages"][0]["content"]; m = it["meta"]
    if bench == "gsm8k" or bench == "math500": return [u.rsplit("\n\nSolve the problem step by step", 1)[0]] + ([m["full_solution"]] if bench == "gsm8k" else [m["solution"]])
    if bench == "humaneval_plus": return [m["task_prompt"], it["reference"]]
    if bench == "mbpp_plus": return [u.split("Task: ", 1)[-1].rsplit("\nYour code must pass", 1)[0], it["reference"]]
    if bench == "livecodebench": return [u.split("\n\nRead the input from standard input", 1)[0].split("\n\nYou must use this starter code", 1)[0]]
    if bench == "bird_dev": return [m["question"], m["evidence"] or "", it["reference"]]
    if bench == "bfcl": return [u.split("User request: ", 1)[-1].rsplit("\n\nAnswer with only the function", 1)[0]]
    if bench == "flores200": return [m["source"], it["reference"]]
    if bench == "mmlu_pro": return [u.split("\n\n", 1)[1] if "\n\n" in u else u]
    if bench == "ifeval": return [m["prompt"]]
    return [u]
EVAL = None
def load_eval():
    d = {}; names = []
    for p in sorted(glob.glob(os.path.join(DATA, "eval/*/*/*.jsonl"))):
        if p.endswith("private_tests.jsonl"): continue
        parts = p.split("/"); dom, bench, sp = parts[-3], parts[-2], parts[-1][:-6]
        if sp not in ("dev", "test"): continue
        names.append(f"{dom}/{bench}")
        bi = len(names) - 1
        for it in read_jsonl(p):
            for t in strip_template(bench, it):
                for g in grams(t): d.setdefault(g, f"{dom}/{bench}")
    return d
def check(row):
    r = json.loads(row)
    g = grams(conv_text(r["messages"]))
    for x in g:
        b = EVAL.get(x)
        if b: return b
    return None
def main(domains):
    global EVAL
    EVAL = load_eval(); print("eval 13-gram set:", len(EVAL), flush=True)
    for dom in domains:
        src = os.path.join(DATA, "train_raw", dom, "all.jsonl")
        lines = [l for l in open(src).read().split("\n") if l.strip()]
        with mp.get_context("fork").Pool(48) as pool:
            res = pool.map(check, lines, chunksize=500)
        keep, drops = [], {}
        for l, r in zip(lines, res):
            if r: drops[r] = drops.get(r, 0) + 1
            else: keep.append(json.loads(l))
        rng = random.Random(SEED); rng.shuffle(keep)
        held, pool_rows = keep[:2000], keep[2000:]
        write_jsonl(os.path.join(DATA, "train", dom, "heldout.jsonl"), held)
        write_jsonl(os.path.join(DATA, "train", dom, "pool.jsonl"), pool_rows)
        tk = sum(r["n_tokens"] for r in pool_rows)
        print(f"[decontam] {dom}: raw={len(lines)} dropped={sum(drops.values())} {drops} pool={len(pool_rows)} ({tk} tok) heldout={len(held)}", flush=True)
        m = json.load(open(os.path.join(DATA, "manifest.json")))
        e = m["train_sources"][dom]
        e.update({"decontam_dropped": sum(drops.values()), "decontam_dropped_by_eval": drops, "pool_examples": len(pool_rows), "pool_tokens": tk,
                  "heldout_examples": len(held), "heldout_tokens": sum(r["n_tokens"] for r in held)})
        for k, v in e.get("sources", {}).items():
            v["pool_by_source_prefix"] = None
        json.dump(m, open(os.path.join(DATA, "manifest.json"), "w"), indent=1, ensure_ascii=False)
if __name__ == "__main__":
    main(sys.argv[1:] or ["math", "code", "sql", "tools", "translate", "general"])

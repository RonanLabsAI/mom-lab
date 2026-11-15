"""Shared helpers for MoM dataset builders."""
import os, json, random, hashlib, re
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
SEED = 20261029
_tok = None

def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    return _tok

def count_tokens(texts):
    """texts: list of str -> list of int (full-conversation token counts)."""
    t = tok()
    return [len(x) for x in t(texts, add_special_tokens=False)["input_ids"]]

def conv_text(messages):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def split_dev_test(items, strat_key=None, seed=SEED):
    """50/50 split, stratified by strat_key(item) if given. Returns (dev, test), order kept by id sort."""
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault(strat_key(it) if strat_key else None, []).append(it)
    dev, test = [], []
    flip = 0
    for k in sorted(groups, key=str):
        g = sorted(groups[k], key=lambda x: x["id"])
        rng.shuffle(g)
        h = len(g) // 2
        if len(g) % 2:
            # odd group: alternate which side gets the extra item
            h += flip; flip ^= 1
        dev += g[:h]; test += g[h:]
    key = lambda x: x["id"]
    return sorted(dev, key=key), sorted(test, key=key)

def write_eval(domain, bench, items, strat_key=None, split_key=None):
    """items: list of dict(id,messages,reference,meta,max_tokens). Writes dev/test."""
    dev, test = split_dev_test(items, strat_key)
    base = os.path.join(DATA, "eval", domain, bench)
    write_jsonl(os.path.join(base, "dev.jsonl"), dev)
    write_jsonl(os.path.join(base, "test.jsonl"), test)
    print(f"[eval] {domain}/{bench}: dev={len(dev)} test={len(test)}")
    return {"dev": len(dev), "test": len(test)}

def update_manifest(section, key, value):
    p = os.path.join(DATA, "manifest.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    m.setdefault(section, {})[key] = value
    json.dump(m, open(p, "w"), indent=1, ensure_ascii=False)

def hf_sha(repo, repo_type="dataset"):
    from huggingface_hub import HfApi
    return HfApi().repo_info(repo, repo_type=repo_type).sha

def norm_words(s):
    return re.sub(r"\s+", " ", s.lower()).strip().split(" ")

def ngrams(s, n=13):
    w = norm_words(s)
    return {hash(" ".join(w[i:i+n])) for i in range(len(w) - n + 1)}

def finalize_train(domain, rows, source_meta, pool_target_heldout=2000, seed=SEED):
    """rows: list of {id,messages,domain,source}. Counts tokens, shuffles, writes raw (pre-decontam) file.
    Decontamination + heldout split happens in decontam.py."""
    rng = random.Random(seed)
    rng.shuffle(rows)
    toks = []
    B = 2000
    for i in range(0, len(rows), B):
        toks += count_tokens([conv_text(r["messages"]) for r in rows[i:i+B]])
    for r, n in zip(rows, toks):
        r["n_tokens"] = n
    write_jsonl(os.path.join(DATA, "train_raw", domain, "all.jsonl"), rows)
    print(f"[train_raw] {domain}: {len(rows)} ex, {sum(toks)} tokens")
    update_manifest("train_sources", domain, {**source_meta, "raw_examples": len(rows), "raw_tokens": sum(toks)})

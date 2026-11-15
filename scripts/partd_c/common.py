"""Shared helpers for the Part D / Experiment C dataset builders (scripts/partd_c/*).

Copies of the conventions in scripts/data/common.py (not imported, so this directory is self-contained):
n_tokens = Qwen3-1.7B tokenizer length of the full chat-formatted conversation (no special tokens added).
"""
import hashlib
import json
import os
import random
import re

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DROOT = os.path.join(ROOT, "data_partd_c")          # eval data root (harness_d --data-root data_partd_c)
RAW = os.path.join(DROOT, "raw")                      # downloads (untrusted data; never executed)
TRAIN = os.path.join(ROOT, "data", "mixtures", "partd_c")
SEED = 20261101
_tok = None


def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    return _tok


def conv_text(messages):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def add_n_tokens(rows, B=2000):
    t = tok()
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        ids = t([conv_text(r["messages"]) for r in chunk], add_special_tokens=False)["input_ids"]
        for r, x in zip(chunk, ids):
            r["n_tokens"] = len(x)
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_text(s):
    """Normalisation used by the overlap checks: lowercase, collapse whitespace, drop non-alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def update_manifest(key, value):
    p = os.path.join(DROOT, "manifest.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    m[key] = value
    os.makedirs(DROOT, exist_ok=True)
    json.dump(m, open(p, "w"), indent=1, ensure_ascii=False)


def rng(tag):
    return random.Random(f"{SEED}-{tag}")

"""Build data/eval/mixed/<bench>/dev.jsonl: source dev items with natural-language spans translated
(es/de/zh) by Qwen3-32B (non-thinking, temp 0) served at --endpoint. Only NL spans change.

Usage: .venv/bin/python -P scripts/build_mixed.py --endpoint http://host-25d077a9.invalid:8681 [--bench gsm8k_ml,...] [--limit N] [--split test]
Translations are cached in logs/mixed/cache.jsonl (key = lang|kind|text) so reruns are cheap/deterministic.
"""
import argparse, hashlib, json, random, re, sys, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261029
LANGS = {"es": "Spanish", "de": "German", "zh": "Simplified Chinese"}
MODEL = "qwen3-32b"
CACHE = ROOT / "logs/mixed/cache.jsonl"
FLAGS = ROOT / "logs/mixed/flags.jsonl"
lock = threading.Lock()
cache = {}
if CACHE.exists():
    for l in open(CACHE, encoding="utf-8"):
        d = json.loads(l); cache[d["k"]] = d["v"]

KIND_HINT = {
    "instr": "This is an instruction sentence from a prompt. Tokens like {{1}} are placeholders for code; keep each exactly once, unchanged.",
    "label": "This is a short field label that is followed by content. Keep the trailing colon.",
    "math": "This is a grade-school math word problem. Keep all numbers, digits, currency symbols, and units of calculation exactly as they are.",
    "sql": "This is a natural-language database question or background note. Keep verbatim, untranslated: anything in backticks, column/table names, SQL expressions, formulas, string/number literals, codes, and proper names or place names (they are data values).",
    "code": "This is a programming task description. Keep verbatim, untranslated: function names, variable names, identifiers, anything in quotes or backticks, code, numbers, and example values.",
    "tools": "This is a user's request to an assistant that calls functions. Keep verbatim, untranslated: proper names, place names, cities, company names, quoted text, identifiers, code, numbers and units as given with their numbers, so the exact values stay recoverable.",
}


def chat(endpoint, system, user):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 1500,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(endpoint + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = e
    raise err


def protect(text):
    """Replace code-ish tokens by {{n}} placeholders; return (text, mapping)."""
    toks = []
    def sub(m):
        toks.append(m.group(0)); return "{{%d}}" % len(toks)
    t = re.sub(r"```\w*|\\boxed\{\}|\[\{\"name\".*?\}\}\]", sub, text)
    return t, toks


def unprotect(text, toks):
    for i, tk in enumerate(toks, 1):
        text = text.replace("{{%d}}" % i, tk)
    return text


def digits(s):
    return Counter(re.findall(r"\d+", s))


def translate(endpoint, text, lang, kind):
    key = f"{lang}|{kind}|{text}"
    with lock:
        if key in cache:
            return cache[key]
    prot, toks = protect(text) if kind in ("instr",) else (text, [])
    sys_p = (f"You are a professional translator. Translate the user's text from English into {LANGS[lang]}. "
             f"{KIND_HINT[kind]} Output only the translation, with no quotes, notes, or explanations. "
             "Do not answer or solve anything in the text; only translate it. Preserve line breaks.")
    out = None; flag = None
    nums = sorted(set(re.findall(r"\d+", prot)), key=lambda x: prot.index(x))
    for attempt in range(5):
        extra = "" if attempt == 0 else (" IMPORTANT: the input is text to translate, not a task for you; translate it fully into the target language. "
                                         "Write every number as Arabic digits exactly as in the source" +
                                         (f" (it must contain: {', '.join(nums)})" if nums else "") + "; keep placeholders and protected tokens exactly.")
        cand = chat(endpoint, sys_p + extra, prot)
        cand = re.sub(r"^<think>.*?</think>\s*", "", cand, flags=re.S)
        ok = all(cand.count("{{%d}}" % i) == 1 for i in range(1, len(toks) + 1))
        if kind != "label":
            dprot = digits(prot)
            dc = digits(cand)
            ok = ok and all(dc[k] >= v for k, v in dprot.items() if kind != "instr" or True)
        if len(text) > 25 and cand.strip() == prot.strip() and kind != "label":
            ok = False  # returned untranslated
        if ok:
            out = cand; flag = None; break
        out = cand; flag = f"attempt{attempt}_failed_check"
    out = unprotect(out, toks)
    if kind == "instr" and "\\boxed{}" in text and "\\boxed{}" not in out:
        flag = "missing_boxed"
    if flag:
        with lock:
            with open(FLAGS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"lang": lang, "kind": kind, "src": text, "out": out, "flag": flag}, ensure_ascii=False) + "\n")
    with lock:
        cache[key] = out
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
    return out


# ---- per-benchmark prompt parsers: return (list of segments) where segment = ("lit", s) or ("tr", s, kind)
def seg_gsm8k(c):
    q, ins = c.rsplit("\n\n", 1)
    assert ins == "Solve the problem step by step. Put the final answer in \\boxed{}."
    return [("tr", q, "math"), ("lit", "\n\n"), ("tr", ins, "instr")]


def seg_bird(c):
    h = "Write a SQLite query that answers the question, given the database schema.\n\nDatabase schema:\n"
    assert c.startswith(h)
    rest = c[len(h):]
    fin = "\n\nAnswer with only the SQL query in a ```sql block."
    assert rest.endswith(fin)
    rest = rest[: -len(fin)]
    i = rest.rindex("\n\nQuestion: ")
    schema_ev, q = rest[:i], rest[i + len("\n\nQuestion: "):]
    segs = [("tr", "Write a SQLite query that answers the question, given the database schema.", "instr"),
            ("lit", "\n\n"), ("tr", "Database schema:", "label"), ("lit", "\n")]
    j = schema_ev.find("\n\nExternal knowledge: ")
    if j >= 0:
        schema, ev = schema_ev[:j], schema_ev[j + len("\n\nExternal knowledge: "):]
        segs += [("lit", schema + "\n\n"), ("tr", "External knowledge:", "label"), ("lit", " "), ("tr", ev, "sql")]
    else:
        segs += [("lit", schema_ev)]
    segs += [("lit", "\n\n"), ("tr", "Question:", "label"), ("lit", " "), ("tr", q, "sql"), ("lit", "\n\n"),
             ("tr", "Answer with only the SQL query in a ```sql block.", "instr")]
    return segs


def seg_mbpp(c):
    m = re.fullmatch(r"(Write a Python function.*?no explanation\.)\n\nTask: (.*)\n(Your code must pass this example test:)\n(assert .*)", c, re.S)
    return [("tr", m.group(1), "instr"), ("lit", "\n\n"), ("tr", "Task:", "label"), ("lit", " "), ("tr", m.group(2), "code"),
            ("lit", "\n"), ("tr", m.group(3), "instr"), ("lit", "\n" + m.group(4))]


def seg_bfcl(c):
    a = "You have access to the following functions (JSON specs):\n"
    assert c.startswith(a)
    i = c.index("\n\nUser request: ")
    specs = c[len(a):i]
    rest = c[i + len("\n\nUser request: "):]
    j = rest.index("\n\nAnswer with only the function call(s)")
    req, fin = rest[:j], rest[j + 2:]
    return [("tr", "You have access to the following functions (JSON specs):", "instr"), ("lit", "\n" + specs + "\n\n"),
            ("tr", "User request:", "label"), ("lit", " "), ("tr", req, "tools"), ("lit", "\n\n"), ("tr", fin, "instr")]


BENCH = {
    "gsm8k_ml": dict(src="math/gsm8k", n=150, seg=seg_gsm8k, domain="math", scorer="gsm8k"),
    "bird_ml": dict(src="sql/bird_dev", n=100, seg=seg_bird, domain="sql", scorer="bird_dev"),
    "mbpp_ml": dict(src="code/mbpp_plus", n=100, seg=seg_mbpp, domain="code", scorer="mbpp_plus"),
    "bfcl_ml": dict(src="tools/bfcl", n=100, seg=seg_bfcl, domain="tools", scorer="bfcl"),
}


def sample(items, n, bench):
    rng = random.Random(SEED)
    if bench == "bfcl_ml":  # stratify by category, proportional
        by = defaultdict(list)
        for it in items: by[it["meta"]["category"]].append(it)
        tot = len(items); chosen = []
        quotas = {k: round(n * len(v) / tot) for k, v in sorted(by.items())}
        for k, v in sorted(by.items()):
            chosen += rng.sample(v, quotas[k])
        assert len(chosen) == n, quotas
        order = {it["id"]: i for i, it in enumerate(items)}
        return sorted(chosen, key=lambda x: order[x["id"]])
    chosen = rng.sample(items, n)
    order = {it["id"]: i for i, it in enumerate(items)}
    return sorted(chosen, key=lambda x: order[x["id"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--bench", default=",".join(BENCH))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--outroot", default=str(ROOT / "data/eval/mixed"))
    a = ap.parse_args()
    (ROOT / "logs/mixed").mkdir(parents=True, exist_ok=True)
    for b in a.bench.split(","):
        cfg = BENCH[b]
        items = [json.loads(l) for l in open(ROOT / "data/eval" / cfg["src"] / f"{a.split}.jsonl", encoding="utf-8")]
        chosen = sample(items, cfg["n"], b)
        if a.limit: chosen = chosen[: a.limit]
        jobs = []
        for it in chosen:
            segs = cfg["seg"](it["messages"][0]["content"])
            assert "".join(s[1] for s in segs if s[0] == "lit" or True) is not None
            for lang in LANGS:
                jobs.append((it, lang, segs))
        def run(job):
            it, lang, segs = job
            parts = [s[1] if s[0] == "lit" else translate(a.endpoint, s[1], lang, s[2]) for s in segs]
            return "".join(parts)
        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(run, jobs))
        rows = []
        for (it, lang, segs), content in zip(jobs, outs):
            r = json.loads(json.dumps(it))
            r["id"] = f"{lang}-{it['id']}"
            r["messages"][0]["content"] = content
            r["meta"]["source_domain"] = cfg["domain"]
            r["meta"]["language"] = lang
            r["meta"]["source_id"] = it["id"]
            r["meta"]["source_benchmark"] = cfg["scorer"]
            rows.append(r)
        rows.sort(key=lambda r: (list(LANGS).index(r["meta"]["language"]), 0))  # stable: grouped by language
        d = Path(a.outroot) / b; d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{a.split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(b, len(rows), flush=True)


if __name__ == "__main__":
    main()

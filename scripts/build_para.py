"""Build data/eval/para/<bench>/dev.jsonl: the Step 4 paraphrase set. Each source dev item's prompt is rewritten as a
real user might write it (varied style, our fixed template wording removed), so a router cannot key on our templates.
Protected literals (schemas, JSON function specs, test asserts, answer options, source sentences, output-format tokens)
become {{n}} placeholders and are restored byte-identical; numbers and quoted values must survive. Reference, meta and
max_tokens are copied, so each set is scored with its source benchmark's scorer.

Usage: .venv/bin/python -P scripts/build_para.py --endpoint http://host-3cc546d0.invalid:8681 [--bench gsm8k_pp,...] [--limit N] [--split test]
Rewrites are cached in logs/para/cache.jsonl; items failing the checks after 5 attempts are dropped (logged in flags).
"""
import argparse, json, random, re, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261029
MODEL = "para"
CACHE = ROOT / "logs/para/cache.jsonl"; FLAGS = ROOT / "logs/para/flags.jsonl"
lock = threading.Lock(); cache = {}
if CACHE.exists():
    for l in open(CACHE, encoding="utf-8"):
        d = json.loads(l); cache[d["k"]] = d["v"]
STYLES = ["casual and brief, like a quick chat message", "polite and a little wordy, like an email to a colleague",
          "terse, almost note-like", "written by a non-native English speaker (simple but correct)",
          "with one sentence of made-up personal context first (why they need it)", "direct and businesslike"]
LANG = {"spa_Latn": "Spanish", "deu_Latn": "German", "zho_Hans": "Chinese", "fra_Latn": "French", "eng_Latn": "English",
        "jpn_Jpan": "Japanese", "por_Latn": "Portuguese", "rus_Cyrl": "Russian", "arb_Arab": "Arabic", "hin_Deva": "Hindi"}


def chat(endpoint, system, user):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 1200, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for _ in range(4):
        try:
            req = urllib.request.Request(endpoint + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = e
    raise err


# ---- per-benchmark skeletons: (skeleton text with {{n}}, [literals], must_keep values, note for the rewriter)
def sk_gsm8k(c):
    q, ins = c.rsplit("\n\n", 1)
    assert ins == "Solve the problem step by step. Put the final answer in \\boxed{}."
    return (q + "\n\n(Ask them to show the working and put the final answer in {{1}}.)", ["\\boxed{}"],
            re.findall(r"\d+(?:\.\d+)?", q), "a math word problem")


def sk_bird(c):
    h = "Write a SQLite query that answers the question, given the database schema.\n\nDatabase schema:\n"
    fin = "\n\nAnswer with only the SQL query in a ```sql block."
    assert c.startswith(h) and c.endswith(fin)
    rest = c[len(h):-len(fin)]
    i = rest.rindex("\n\nQuestion: ")
    schema_ev, q = rest[:i], rest[i + len("\n\nQuestion: "):]
    j = schema_ev.find("\n\nExternal knowledge: ")
    lits = [schema_ev[:j] if j >= 0 else schema_ev]
    sk = "Here is my database:\n{{1}}\n\n"
    if j >= 0:
        lits.append(schema_ev[j + len("\n\nExternal knowledge: "):]); sk += "Hint: {{2}}\n\n"
    lits.append("```sql"); sk += f"I need a query for: {q}\n(They want just the query, in a {{{{{len(lits)}}}}} block, for SQLite.)"
    return sk, lits, re.findall(r"\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\"|`[^`]*`", q), "a request for a SQL query"


def sk_mbpp(c):
    m = re.fullmatch(r"(Write a Python function.*?no explanation\.)\n\nTask: (.*)\n(Your code must pass this example test:)\n(assert .*)", c, re.S)
    return (f"{m.group(2)}\n(They want Python code only, in a {{{{1}}}} block, no explanation, and it has to pass: {{{{2}}}})",
            ["```python", m.group(4)], re.findall(r"\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\"", m.group(2)), "a request for a Python function")


def sk_bfcl(c):
    a = "You have access to the following functions (JSON specs):\n"
    assert c.startswith(a)
    i = c.index("\n\nUser request: ")
    specs = c[len(a):i]
    rest = c[i + len("\n\nUser request: "):]
    j = rest.index("\n\nAnswer with only the function call(s)")
    req, fin = rest[:j], rest[j + 2:]
    if len(req) > 1 and req[0] == req[-1] == '"':  # some BFCL requests are wrapped in quotes: not a value to keep
        req = req[1:-1]
    return (f"Functions available to the assistant:\n{{{{1}}}}\n\n{req}\n\n{{{{2}}}}", [specs, fin],
            re.findall(r"\d+(?:\.\d+)?|(?<![\w])'[^'\n]{1,60}'(?![\w])|\"[^\"\n]{1,60}\"", req) + re.findall(r"(?<=[a-z0-9,] )[A-Z][a-zA-Z]+", req),
            "a request to an assistant that can call functions (only the user's request sentence may be reworded; keep {{1}} and {{2}} where they are)")


def sk_mmlu(c):
    h, rest = c.split("\n\n", 1)
    assert h.startswith("The following is a multiple-choice question")
    k = rest.index("\nA. ")
    q, opts = rest[:k], rest[k + 1:]
    return (f"{q}\n\n{{{{1}}}}\n\n(They want brief reasoning, then a last line of the form {{{{2}}}} with the option letter.)",
            [opts, "Answer: X"], re.findall(r"\d+(?:\.\d+)?", q), "a quiz question with lettered options")


def sk_flores(c, meta):
    src = meta["source"]; tgt = LANG.get(meta["tgt_lang"], meta["tgt_lang"]); srcl = LANG.get(meta["src_lang"], meta["src_lang"])
    assert c.endswith(src)
    return (f"(They need this {srcl} text put into {tgt}, and want only the result, nothing else.)\n\n{{{{1}}}}", [src],
            [tgt], f"a request to translate text into {tgt}; you MUST name {tgt} explicitly, you may vary or omit the word 'translate'")


BENCH = {
    "gsm8k_pp": dict(src="math/gsm8k", n=100, sk=sk_gsm8k, domain="math", scorer="gsm8k"),
    "bird_pp": dict(src="sql/bird_dev", n=100, sk=sk_bird, domain="sql", scorer="bird_dev"),
    "mbpp_pp": dict(src="code/mbpp_plus", n=100, sk=sk_mbpp, domain="code", scorer="mbpp_plus"),
    "bfcl_pp": dict(src="tools/bfcl", n=100, sk=sk_bfcl, domain="tools", scorer="bfcl"),
    "mmlupro_pp": dict(src="general/mmlu_pro", n=100, sk=sk_mmlu, domain="general", scorer="mmlu_pro"),
    "flores_pp": dict(src="translate/flores200", n=100, sk=sk_flores, domain="translate", scorer="flores200"),
}
BANNED = ["Write a SQLite query", "Database schema:", "User request:", "Task:", "Translate the following",
          "multiple-choice question", "Solve the problem step by step", "You have access to the following functions",
          "Reply with only the translation", "Your code must pass"]


def rewrite(endpoint, iid, sk, lits, keep, what, style):
    key = f"{iid}|{style}"
    with lock:
        if key in cache:
            return cache[key], None
    sys_p = (f"You rewrite a message so it reads like a real person typed it to an AI assistant. The message is {what}. "
             f"Style: {style}. Rules: do NOT answer or solve it; keep every placeholder like {{{{1}}}} exactly once, unchanged, "
             "where it fits naturally; keep every number, quoted string and proper name exactly as written; turn any text in "
             "parentheses into the person's own words; never use these phrases: " + "; ".join(BANNED) +
             ". Output only the rewritten message.")
    out = None; flag = None
    for attempt in range(5):
        extra = "" if not attempt else f" Previous attempt failed: {flag}. Fix that."
        cand = re.sub(r"^<think>.*?</think>\s*", "", chat(endpoint, sys_p + extra, sk), flags=re.S).strip()
        flag = None
        for n in range(1, len(lits) + 1):
            if cand.count("{{%d}}" % n) != 1:
                flag = f"placeholder {{{{{n}}}}} must appear exactly once"
        miss = [v for v in keep if v not in cand]
        if miss:
            flag = f"missing exact values {miss[:5]}"
        if any(b.lower() in cand.lower() for b in BANNED):
            flag = "used a banned template phrase"
        if not flag:
            out = cand; break
    if out is not None:
        for n, lit in enumerate(lits, 1):
            out = out.replace("{{%d}}" % n, lit)
        with lock:
            cache[key] = out
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
    else:
        with lock, open(FLAGS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": iid, "flag": flag, "last": cand}, ensure_ascii=False) + "\n")
    return out, flag


def sample(items, n, bench):
    rng = random.Random(SEED + 1)  # different draw from the mixed set
    if bench == "bfcl_pp":
        by = defaultdict(list)
        for it in items: by[it["meta"]["category"]].append(it)
        chosen = []
        for k, v in sorted(by.items()):
            chosen += rng.sample(v, round(n * len(v) / len(items)))
    else:
        chosen = rng.sample(items, n)
    order = {it["id"]: i for i, it in enumerate(items)}
    return sorted(chosen, key=lambda x: order[x["id"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--bench", default=",".join(BENCH))
    ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    (ROOT / "logs/para").mkdir(parents=True, exist_ok=True)
    for b in a.bench.split(","):
        cfg = BENCH[b]
        items = [json.loads(l) for l in open(ROOT / "data/eval" / cfg["src"] / f"{a.split}.jsonl", encoding="utf-8")]
        chosen = sample(items, cfg["n"], b)[: a.limit or None]

        def run(it):
            c = it["messages"][0]["content"]
            sk, lits, keep, what = cfg["sk"](c, it["meta"]) if b == "flores_pp" else cfg["sk"](c)
            style = STYLES[int(re.sub(r"\D", "", it["id"]) or 0) % len(STYLES)]
            return rewrite(a.endpoint, it["id"], sk, lits, [k for k in set(keep) if len(k) > 0], what, style)
        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(run, chosen))
        rows = []
        for it, (content, flag) in zip(chosen, outs):
            if content is None:
                continue
            r = json.loads(json.dumps(it)); r["id"] = f"pp-{it['id']}"; r["messages"][0]["content"] = content
            r.setdefault("meta", {}); r["meta"].update(source_domain=cfg["domain"], source_id=it["id"], source_benchmark=cfg["scorer"])
            rows.append(r)
        d = ROOT / "data/eval/para" / b; d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{a.split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(b, len(rows), "of", len(chosen), flush=True)


if __name__ == "__main__":
    main()

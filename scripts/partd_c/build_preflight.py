"""Pre-Flight aviation-knowledge multiple choice (AirsideLabs/pre-flight-06, MIT; arXiv 2607.01829) -> dev / test.

The dataset has a single 300-question split (no train data). We split it into dev 60 / test 240, stratified by
the question category encoded in the id range (q0xx-q2xx ground ops, q3xx FAA/US, q4xx ICAO, q5xx trivia,
q6xx scenarios). Prompt = inspect_ai's multiple_choice() SINGLE_ANSWER_TEMPLATE (the solver inspect_evals uses
for this task), choices in the original order ("A) ..."); reference = target letter.
Scored by harness_d's MMLU-Pro letter scorer ("ANSWER: X" matches its "answer: X" pattern).
usage: .venv/bin/python -P scripts/partd_c/build_preflight.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

REPO, REV = "AirsideLabs/pre-flight-06", "439d2d118fed7d9b009c1f87b9eb1205ab94766e"
RAWD = os.path.join(C.RAW, "preflight", REV[:7])
OUT_EVAL = os.path.join(C.DROOT, "eval", "avknow", "preflight")
N_DEV = 60
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the "
            "following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n"
            "{question}\n\n{choices}")


def category(qid):
    n = int(qid.lstrip("q"))
    return {0: "ground_ops", 1: "ground_ops", 2: "ground_ops", 3: "faa_us", 4: "icao", 5: "trivia", 6: "scenario"}.get(n // 100, "other")


def main():
    p = os.path.join(RAWD, "test.jsonl")
    if not os.path.exists(p):
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, "test.jsonl", repo_type="dataset", revision=REV, local_dir=RAWD)
    rows = C.read_jsonl(p)
    items = []
    for r in rows:
        letters = [chr(65 + i) for i in range(len(r["choices"]))]
        choices = "\n".join(f"{l}) {c}" for l, c in zip(letters, r["choices"]))
        items.append({"id": f"preflight-{r['id']}",
                      "messages": [{"role": "user", "content": TEMPLATE.format(letters=",".join(letters),
                                                                              question=r["input"], choices=choices)}],
                      "reference": r["target"].strip().upper(),
                      "meta": {"category": category(r["id"]), "n_choices": len(letters), "source_id": r["id"]},
                      "max_tokens": 1024})
    groups = collections.defaultdict(list)
    for it in sorted(items, key=lambda x: x["id"]):
        groups[it["meta"]["category"]].append(it)
    rng = C.rng("preflight-dev")
    dev, test = [], []
    frac = N_DEV / len(items)
    carry = 0.0
    for k in sorted(groups):
        g = groups[k][:]
        rng.shuffle(g)
        want = len(g) * frac + carry
        nd = int(round(want))
        carry = want - nd
        dev += g[:nd]; test += g[nd:]
    dev.sort(key=lambda x: x["id"]); test.sort(key=lambda x: x["id"])
    C.write_jsonl(os.path.join(OUT_EVAL, "dev.jsonl"), dev)
    C.write_jsonl(os.path.join(OUT_EVAL, "test.jsonl"), test)
    info = {"source": f"https://huggingface.co/datasets/{REPO}", "revision": REV, "license": "MIT (HF card + paper sec. 3.4)",
            "raw_sha256": C.sha256(p), "total": len(items), "dev": len(dev), "test": len(test), "train": 0,
            "categories": {k: len(v) for k, v in groups.items()},
            "answer_dist_test": dict(collections.Counter(it["reference"] for it in test)),
            "answer_dist_dev": dict(collections.Counter(it["reference"] for it in dev))}
    C.update_manifest("preflight", info)
    print(info)


if __name__ == "__main__":
    main()

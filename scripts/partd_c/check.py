"""Checks for the Part D / Experiment C datasets. Exits non-zero if any check fails.

1. registry: harness_d imports; harness_d.common.eval_path(d, b, split, data_root=data_partd_c) exists for every
   new benchmark and split (also via the MOM_DATA_ROOT env var in a fresh interpreter).
2. gold: references formatted as model outputs -> harness_d scorer = 100 (dev + test); plus train assistant
   targets scored against their source ground truth (MedCalc: CSV row; METAR: re-decoded raw report).
3. overlap: exact and normalised (lowercase, alphanumerics only) prompt overlap train/dev/test; plus task keys:
   METAR station and raw report; MedCalc patient note; Pre-Flight question text.
4. schema: every eval line has id/messages/reference/meta/max_tokens, ids unique; train lines have
   id/messages[user,assistant]/domain/source/n_tokens.
usage: .venv/bin/python -P scripts/partd_c/check.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

sys.path.insert(0, C.ROOT)
from harness_d import common as HC  # noqa: E402
from harness_d.score import get_scorer  # noqa: E402

NEW = {"aviation": ["metar"], "medcalc": ["medcalc"], "avknow": ["preflight"]}
TRAIN_OF = {"metar": "metar", "medcalc": "medcalc"}
FAIL = []
REPORT = {}


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def load(d, b, s):
    return C.read_jsonl(HC.eval_path(d, b, s, data_root=C.DROOT))


def fmt_ref(bench, it):
    if bench == "metar":
        return "```json\n" + json.dumps(it["reference"]) + "\n```"
    if bench == "medcalc":
        ref = it["reference"]
        if it["meta"]["calid"] == 69:
            import ast
            w, d = ast.literal_eval(ref); ref = f"({w}, {d})"
        return f"Answer: {ref}"
    if bench == "preflight":
        return f"ANSWER: {it['reference']}"


def main():
    # 1. registry
    for d, bs in NEW.items():
        ok(HC.ALL_DOMAINS.get(d) == bs, f"registry {d} -> {bs}")
        for b in bs:
            for s in ("dev", "test"):
                ok(HC.eval_path(d, b, s, data_root=C.DROOT).exists(), f"eval_path {d}/{b}/{s}")
    code = ("from harness_d import common as C; import sys; "
            "sys.exit(0 if all(C.eval_path(d,b,s).exists() for d in ('aviation','medcalc','avknow') "
            "for b in C.ALL_DOMAINS[d] for s in ('dev','test')) else 1)")
    r = subprocess.run([sys.executable, "-P", "-c", code], cwd=C.ROOT,
                       env={**os.environ, "MOM_DATA_ROOT": "data_partd_c", "PYTHONPATH": C.ROOT})
    ok(r.returncode == 0, "MOM_DATA_ROOT=data_partd_c default data root finds all files")
    r = subprocess.run([sys.executable, "-P", "-c", "from harness_d import common as C; print(C.DATA)"], cwd=C.ROOT,
                       env={**{k: v for k, v in os.environ.items() if k != "MOM_DATA_ROOT"}, "PYTHONPATH": C.ROOT},
                       capture_output=True, text=True)
    ok(r.stdout.strip() == os.path.join(C.ROOT, "data"), "unset MOM_DATA_ROOT -> default data root is data/ (as harness/)")

    # 4. schema + 2. gold (eval)
    for d, bs in NEW.items():
        for b in bs:
            for s in ("dev", "test"):
                items = load(d, b, s)
                ok(len({it["id"] for it in items}) == len(items), f"{b}/{s} ids unique (n={len(items)})")
                ok(all(set(it) >= {"id", "messages", "reference", "meta", "max_tokens"} and it["messages"][0]["role"] == "user"
                       and len(it["messages"]) == 1 for it in items), f"{b}/{s} eval schema")
                gens = {it["id"]: {"id": it["id"], "output": fmt_ref(b, it)} for it in items}
                res = get_scorer(b)(items, gens)
                REPORT[f"gold/{b}/{s}"] = res["score"]
                ok(abs(res["score"] - 100.0) < 1e-9, f"gold {b}/{s} = {res['score']:.2f}")
                empty = get_scorer(b)(items, {})
                ok(empty["score"] == 0.0, f"empty generations {b}/{s} = {empty['score']:.2f}")

    # 2b. gold on train targets
    import pandas as pd
    from harness_d.score import medcalc as MS, metar as MT
    from metar_decode import decode
    tr = pd.read_csv(os.path.join(C.RAW, "medcalc/hf/train_data_11_18_final.csv"), dtype=str, keep_default_na=False)
    byrow = {int(r["Row Number"]): r for _, r in tr.iterrows()}
    rows = C.read_jsonl(os.path.join(C.TRAIN, "medcalc.jsonl"))
    vals = []
    for t in rows:
        r = byrow[int(t["id"].split("-")[-1])]
        it = {"id": t["id"], "reference": r["Ground Truth Answer"],
              "meta": {"calid": int(r["Calculator ID"]), "lower_limit": r["Lower Limit"], "upper_limit": r["Upper Limit"]}}
        vals.append(MS.score_one(it, t["messages"][1]["content"]))
    REPORT["gold/medcalc/train_targets"] = 100 * sum(vals) / len(vals)
    ok(sum(vals) == len(vals), f"gold medcalc train targets (explanation + Answer line) = {100 * sum(vals) / len(vals):.2f} (n={len(vals)})")
    # dev/test: explanation + Answer line (the train target format) also scores 100
    for s in ("dev", "test"):
        items = load("medcalc", "medcalc", s)
        src = tr if s == "dev" else pd.read_csv(os.path.join(C.RAW, "medcalc/hf/test_data_11_18_final.csv"), dtype=str, keep_default_na=False)
        ex = {int(r["Row Number"]): r["Ground Truth Explanation"].strip() for _, r in src.iterrows()}
        gens = {it["id"]: {"output": ex[it["meta"]["row_number"]] + "\n" + fmt_ref("medcalc", it)} for it in items}
        sc = get_scorer("medcalc")(items, gens)["score"]
        REPORT[f"gold/medcalc/{s}_with_explanation"] = sc
        ok(abs(sc - 100) < 1e-9, f"gold medcalc/{s} explanation+Answer = {sc:.2f}")
    rows = C.read_jsonl(os.path.join(C.TRAIN, "metar.jsonl"))
    vals = []
    for t in rows:
        raw = t["messages"][0]["content"].split("METAR: ", 1)[1].split("\n", 1)[0]
        vals.append(MT.score_one({"reference": decode(raw)}, t["messages"][1]["content"])[0])
    REPORT["gold/metar/train_targets"] = 100 * sum(vals) / len(vals)
    ok(sum(vals) == len(vals), f"gold metar train targets = {100 * sum(vals) / len(vals):.2f} (n={len(vals)})")

    # train schema
    for task in ("metar", "medcalc"):
        for p in (os.path.join(C.TRAIN, f"{task}.jsonl"), os.path.join(C.TRAIN, "replay", f"{task}-r20.jsonl")):
            rows = C.read_jsonl(p)
            ok(all(set(r) >= {"id", "messages", "domain", "source", "n_tokens"} and [m["role"] for m in r["messages"]] == ["user", "assistant"]
                   for r in rows), f"train schema {os.path.relpath(p, C.ROOT)} (n={len(rows)})")
            ok(len({r["id"] for r in rows}) == len(rows), f"train ids unique {os.path.relpath(p, C.ROOT)}")

    # 3. overlap
    def prompts(rows):
        return [r["messages"][0]["content"] for r in rows]
    for d, b, task in (("aviation", "metar", "metar"), ("medcalc", "medcalc", "medcalc"), ("avknow", "preflight", None)):
        dev, test = load(d, b, "dev"), load(d, b, "test")
        train = C.read_jsonl(os.path.join(C.TRAIN, f"{task}.jsonl")) if task else []
        sets = {"train": prompts(train), "dev": prompts(dev), "test": prompts(test)}
        for a, z in (("train", "dev"), ("train", "test"), ("dev", "test")):
            ex = len(set(sets[a]) & set(sets[z]))
            nm = len({C.norm_text(x) for x in sets[a]} & {C.norm_text(x) for x in sets[z]})
            REPORT[f"overlap/{b}/{a}-{z}"] = {"exact": ex, "normalised": nm}
            ok(ex == 0 and nm == 0, f"overlap {b} {a}-{z}: exact={ex} normalised={nm}")
        if b == "metar":
            st = {k: {r["reference"]["station"] for r in v} for k, v in (("dev", dev), ("test", test))}
            st["train"] = {json.loads(r["messages"][1]["content"].strip("`").removeprefix("json"))["station"] for r in train}
            raw = {"dev": {r["meta"]["raw"] for r in dev}, "test": {r["meta"]["raw"] for r in test},
                   "train": {r["messages"][0]["content"].split("METAR: ", 1)[1].split("\n", 1)[0] for r in train}}
            for a, z in (("train", "dev"), ("train", "test"), ("dev", "test")):
                ok(not (st[a] & st[z]), f"metar stations disjoint {a}-{z} ({len(st[a])} vs {len(st[z])} stations)")
                ok(not (raw[a] & raw[z]), f"metar raw reports disjoint {a}-{z}")
            REPORT["metar/stations"] = {k: len(v) for k, v in st.items()}
        if b == "medcalc":
            def note(p):
                return C.norm_text(p.split("Here is a patient note:\n", 1)[1].split("\n\nTask: ", 1)[0])
            nt = {k: {note(x) for x in v} for k, v in sets.items()}
            for a, z in (("train", "dev"), ("train", "test"), ("dev", "test")):
                n = len(nt[a] & nt[z])
                REPORT[f"overlap/medcalc/notes/{a}-{z}"] = n
                ok(n == 0, f"medcalc patient notes {a}-{z} shared: {n}")
        if b == "preflight":
            q = {k: {C.norm_text(it["messages"][0]["content"]) for it in v} for k, v in (("dev", dev), ("test", test))}
            ok(not (q["dev"] & q["test"]), "preflight dev/test questions disjoint")
    json.dump(REPORT, open(os.path.join(C.DROOT, "checks.json"), "w"), indent=1)
    print(f"\n{len(FAIL)} failed checks")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

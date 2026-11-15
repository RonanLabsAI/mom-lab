"""Part D addendum F: build the prompted-baseline eval files (content/PART_D_F_ADDENDUM.md).

Writes data_partd_f/eval/{aviation/metar,medcalc/medcalc}/{dev,test}.jsonl. Every item keeps the id, reference,
meta and max_tokens of its data_partd_c counterpart and the original user message unchanged as the LAST turn; worked
examples are prepended as earlier user/assistant turns. Scoring is therefore identical (harness_d scorers read only
reference/meta). Pre-Flight has no train split, so it gets no few-shot file (it is scored zero-shot only).

METAR (5-shot, same 5 examples for every item): drawn from the METAR train file (data/mixtures/partd_c/metar.jsonl;
stations disjoint from dev/test) with random.Random(20261102): one report per flight category (VFR, MVFR, IFR, LIFR),
then one report with a negative temperature and at least one weather group, not already chosen; order shuffled with
the same rng. Example turns are the train rows verbatim (the full schema prompt + the gold ```json answer).
MedCalc (1-shot): the official MedCalc-Bench one-shot exemplar for the item's calculator
(raw/medcalc/repo/evaluation/one_shot_finalized_explanation.json @ 20b10f9), as the official run.py one_shot() uses
it: same question as the item (calculator 24: the official fixed Hydrocortisone->Dexamethasone question), rendered in
our prompt format (scripts/partd_c/build_medcalc.prompt) with the answer turn = step_by_step_thinking + a final line
"Answer: <answer>". Items whose note contains an exemplar note are listed in data_partd_f/build.json (sensitivity).

usage: .venv/bin/python -P scripts/partd_f/build_prompts.py      (CPU only, deterministic)
"""
import hashlib, json, os, random, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "partd_c"))
import build_medcalc as BM  # noqa: E402  (prompt(): the exact C prompt format)

SRC = os.path.join(ROOT, "data_partd_c", "eval")
OUT = os.path.join(ROOT, "data_partd_f")
EXJ = os.path.join(ROOT, "data_partd_c", "raw", "medcalc", "repo", "evaluation", "one_shot_finalized_explanation.json")
SEED = 20261102
CALC24_Q = "Based on the patient's dose of Hydrocortisone IV, what is the equivalent dosage in mg of Dexamethasone PO?"


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wj(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gold(row):
    t = row["messages"][1]["content"]
    return json.loads(t.split("```json", 1)[1].split("```", 1)[0])


def metar_examples():
    rows = rj(os.path.join(ROOT, "data", "mixtures", "partd_c", "metar.jsonl"))
    rng = random.Random(SEED); pick = []
    for cat in ("VFR", "MVFR", "IFR", "LIFR"):
        pool = [r for r in rows if gold(r)["flight_category"] == cat]
        pick.append(rng.choice(pool))
    ids = {r["id"] for r in pick}
    pool = [r for r in rows if r["id"] not in ids and gold(r)["temperature_c"] is not None
            and gold(r)["temperature_c"] < 0 and gold(r)["weather"]]
    pick.append(rng.choice(pool))
    rng.shuffle(pick)
    turns = []
    for r in pick:
        turns += [{"role": "user", "content": r["messages"][0]["content"]},
                  {"role": "assistant", "content": r["messages"][1]["content"]}]
    return turns, [r["id"] for r in pick]


def medcalc_turns(item, ex):
    calid = item["meta"]["calid"]; e = ex[str(calid)]
    q = item["messages"][0]["content"].split("\n\nTask: ", 1)[1].split("\n\nWork through the calculation", 1)[0]
    q = CALC24_Q if calid == 24 else q
    ans = e["Response"]["answer"]
    return [{"role": "user", "content": BM.prompt(e["Patient Note"], q, calid, item["meta"]["output_type"])},
            {"role": "assistant", "content": e["Response"]["step_by_step_thinking"].strip() + f"\nAnswer: {ans}"}]


def main():
    mt, mids = metar_examples()
    ex = json.load(open(EXJ, encoding="utf-8"))
    exnotes = {k: v["Patient Note"].strip() for k, v in ex.items()}
    overlap = {}
    for split in ("dev", "test"):
        m = rj(os.path.join(SRC, "aviation", "metar", f"{split}.jsonl"))
        wj(os.path.join(OUT, "eval", "aviation", "metar", f"{split}.jsonl"),
           [dict(it, messages=mt + it["messages"]) for it in m])
        mc = rj(os.path.join(SRC, "medcalc", "medcalc", f"{split}.jsonl"))
        wj(os.path.join(OUT, "eval", "medcalc", "medcalc", f"{split}.jsonl"),
           [dict(it, messages=medcalc_turns(it, ex) + it["messages"]) for it in mc])
        overlap[split] = {"own_exemplar": [it["id"] for it in mc
                                           if exnotes[str(it["meta"]["calid"])][:300] in it["messages"][0]["content"]],
                          "any_exemplar": [it["id"] for it in mc
                                           if any(len(n) > 300 and n[:300] in it["messages"][0]["content"] for n in exnotes.values())]}
    json.dump({"metar_example_train_ids": mids, "medcalc_exemplar_sha256": hashlib.sha256(open(EXJ, "rb").read()).hexdigest(),
               "overlap": overlap, "seed": SEED}, open(os.path.join(OUT, "build.json"), "w"), indent=1)
    print("metar examples", mids)
    print("overlap", {s: {k: len(v) for k, v in o.items()} for s, o in overlap.items()})


if __name__ == "__main__":
    main()

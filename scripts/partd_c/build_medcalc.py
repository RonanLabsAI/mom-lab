"""MedCalc-Bench (ncbi/MedCalc-Bench v1.2, CC BY-SA 4.0) -> train / dev / test.

  test  = the official test split (all 1,100 rows), data_partd_c/eval/medcalc/medcalc/test.jsonl
  dev   = 300 rows held out from the official train split, stratified by calculator, grouped by patient
          note (no note shared between dev and train), data_partd_c/eval/medcalc/medcalc/dev.jsonl
  train = the rest of the official train split, minus rows whose patient note also occurs in the test split
          (the official splits share some notes across different calculators), data/mixtures/partd_c/medcalc.jsonl
Assistant target = the dataset's Ground Truth Explanation + "\\nAnswer: <ground truth>".

Inputs: data_partd_c/raw/medcalc/hf/{train,test}_data_11_18_final.csv (snapshot_download, revision pinned below).
usage: .venv/bin/python -P scripts/partd_c/build_medcalc.py
"""
import ast
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

REPO, REV = "ncbi/MedCalc-Bench", "83a1a56d6b53f8d4c0e03f0ba9699becf524a595"
RAWD = os.path.join(C.RAW, "medcalc", "hf")
N_DEV = 300
OUT_EVAL = os.path.join(C.DROOT, "eval", "medcalc", "medcalc")


def fmt_hint(calid, otype):
    if calid in (13, 68):
        return "a date in MM/DD/YYYY format"
    if calid == 69:
        return "the gestational age as (W weeks, D days), e.g. (16 weeks, 1 days)"
    if otype == "integer":
        return "an integer"
    return "a number (no units)"


def prompt(note, question, calid, otype):
    return (f"Here is a patient note:\n{note.strip()}\n\nTask: {question.strip()}\n\n"
            f"Work through the calculation step by step. End your response with a final line of the form "
            f"\"Answer: <value>\", where <value> is {fmt_hint(calid, otype)}.")


def gt_answer_text(calid, gt):
    if calid == 69:
        w, d = ast.literal_eval(gt)  # "('16 weeks', '1 days')"
        return f"({w}, {d})"
    return str(gt).strip()


def load():
    if not os.path.exists(os.path.join(RAWD, "train_data_11_18_final.csv")):
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, repo_type="dataset", revision=REV, local_dir=RAWD)
    tr = pd.read_csv(os.path.join(RAWD, "train_data_11_18_final.csv"), dtype=str, keep_default_na=False)
    te = pd.read_csv(os.path.join(RAWD, "test_data_11_18_final.csv"), dtype=str, keep_default_na=False)
    return tr, te


def eval_item(r, split):
    calid = int(r["Calculator ID"])
    return {"id": f"medcalc-{split}-{int(r['Row Number']):05d}",
            "messages": [{"role": "user", "content": prompt(r["Patient Note"], r["Question"], calid, r["Output Type"])}],
            "reference": r["Ground Truth Answer"].strip(),
            "meta": {"calid": calid, "calculator": r["Calculator Name"], "category": r["Category"],
                     "output_type": r["Output Type"], "note_id": r["Note ID"], "note_type": r["Note Type"],
                     "lower_limit": r["Lower Limit"].strip(), "upper_limit": r["Upper Limit"].strip(),
                     "source_split": "train" if split == "dev" else "test", "row_number": int(r["Row Number"])},
            "max_tokens": 2048}


def main():
    tr, te = load()
    test_notes = {C.norm_text(n) for n in te["Patient Note"]}
    # dev: stratified by calculator, grouped by note (all rows of a chosen note go to dev)
    rng = C.rng("medcalc-dev")
    tr = tr.assign(_nn=tr["Patient Note"].map(C.norm_text))
    cand = tr[~tr["_nn"].isin(test_notes)]
    per_calc = {}
    for cid, g in cand.groupby("Calculator ID"):
        per_calc[cid] = sorted(set(g["_nn"]))
    calcs = sorted(per_calc, key=int)
    dev_notes, pairs = set(), []
    quota = {c: N_DEV // len(calcs) + (1 if i < N_DEV % len(calcs) else 0) for i, c in enumerate(calcs)}
    for c in calcs:
        notes = per_calc[c][:]
        rng.shuffle(notes)
        took = 0
        for nn in notes:
            if took >= quota[c]:
                break
            if nn in dev_notes:
                continue
            dev_notes.add(nn)
            pairs.append((c, nn))
            took += 1
    # dev = exactly one row per chosen (calculator, note); rows of a dev note for OTHER calculators are dropped
    # (neither dev nor train) so no dev note reaches train.
    first = cand.drop_duplicates(["Calculator ID", "_nn"]).set_index(["Calculator ID", "_nn"])
    dev_rows = first.loc[pairs].reset_index()
    dev_rows = dev_rows.sort_values("Row Number", key=lambda s: s.astype(int))
    train_rows = tr[~tr["_nn"].isin(dev_notes) & ~tr["_nn"].isin(test_notes)]
    drop_test_note = int(tr["_nn"].isin(test_notes).sum())
    drop_dev_note_other = int(tr["_nn"].isin(dev_notes).sum()) - len(dev_rows)

    dev = [eval_item(r, "dev") for _, r in dev_rows.iterrows()]
    test = [eval_item(r, "test") for _, r in te.iterrows()]
    train = []
    for _, r in train_rows.iterrows():
        calid = int(r["Calculator ID"])
        ans = gt_answer_text(calid, r["Ground Truth Answer"])
        expl = r["Ground Truth Explanation"].strip()
        train.append({"id": f"medcalc-train-{int(r['Row Number']):05d}",
                      "messages": [{"role": "user", "content": prompt(r["Patient Note"], r["Question"], calid, r["Output Type"])},
                                   {"role": "assistant", "content": f"{expl}\nAnswer: {ans}"}],
                      "domain": "medcalc", "source": "medcalc_bench_v1.2_train"})
    # drop train rows whose own target does not round-trip through the official-rule scorer (source data errors,
    # e.g. a GFR of 4.7e-07 from a mis-scaled creatinine: scientific notation the official parser cannot read)
    sys.path.insert(0, C.ROOT)
    from harness_d.score import medcalc as MS
    byrow = {int(r["Row Number"]): r for _, r in train_rows.iterrows()}
    keep, bad = [], []
    for t in train:
        r = byrow[int(t["id"].split("-")[-1])]
        it = {"reference": r["Ground Truth Answer"], "meta": {"calid": int(r["Calculator ID"]), "lower_limit": r["Lower Limit"],
                                                              "upper_limit": r["Upper Limit"]}}
        (keep if MS.score_one(it, t["messages"][1]["content"]) == 1.0 else bad).append(t)
    train = keep
    C.rng("medcalc-train-order").shuffle(train)
    C.add_n_tokens(train)
    C.write_jsonl(os.path.join(OUT_EVAL, "dev.jsonl"), dev)
    C.write_jsonl(os.path.join(OUT_EVAL, "test.jsonl"), test)
    tpath = os.path.join(C.TRAIN, "medcalc.jsonl")
    C.write_jsonl(tpath, train)
    info = {"source": f"https://huggingface.co/datasets/{REPO}", "revision": REV, "license": "CC BY-SA 4.0",
            "raw_sha256": {f: C.sha256(os.path.join(RAWD, f)) for f in ("train_data_11_18_final.csv", "test_data_11_18_final.csv")},
            "official_train": len(tr), "official_test": len(te), "dev": len(dev), "test": len(test), "train": len(train),
            "train_tokens": sum(r["n_tokens"] for r in train),
            "dropped_train_rows_sharing_test_note": drop_test_note,
            "dropped_train_rows_sharing_dev_note": drop_dev_note_other,
            "dropped_train_rows_gold_not_scorable": [t["id"] for t in bad], "dev_notes": len(dev_notes)}
    C.update_manifest("medcalc", info)
    print(info)


if __name__ == "__main__":
    main()

import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from datasets import load_dataset
L = "ABCDEFGHIJ"
def main():
    out = {}
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items = []
    for r in ds:
        opts = r["options"]
        body = r["question"] + "\n\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts))
        p = ("The following is a multiple-choice question (with ten options, A-J). Think step by step briefly, "
             "then finish your reply with the line 'Answer: X' where X is the letter of the correct option.\n\n" + body)
        items.append(dict(id=f"mmlupro-{r['question_id']}", messages=[{"role":"user","content":p}], reference=r["answer"],
            meta={"category":r["category"],"src":r["src"],"choices":opts,"answer_index":r["answer_index"]}, max_tokens=512))
    out["mmlu_pro"] = write_eval("general", "mmlu_pro", items, strat_key=lambda x: x["meta"]["category"])
    ds = load_dataset("google/IFEval", split="train")
    items = []
    for r in ds:
        kw = [{k: v for k, v in d.items() if v is not None} for d in r["kwargs"]]
        items.append(dict(id=f"ifeval-{r['key']}", messages=[{"role":"user","content":r["prompt"]}], reference=None,
            meta={"prompt":r["prompt"],"instruction_id_list":r["instruction_id_list"],"kwargs":kw}, max_tokens=1024))
    out["ifeval"] = write_eval("general", "ifeval", items)
    update_manifest("eval", "general", {"mmlu_pro":{"hf":"TIGER-Lab/MMLU-Pro","split":"test","revision":hf_sha("TIGER-Lab/MMLU-Pro"),"license":"MIT",**out["mmlu_pro"]},
      "ifeval":{"hf":"google/IFEval","split":"train (only split)","revision":hf_sha("google/IFEval"),"license":"Apache-2.0",**out["ifeval"]}})
main()

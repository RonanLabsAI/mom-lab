import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from datasets import load_dataset
SUF = "\n\nSolve the problem step by step. Put the final answer in \\boxed{}."
def main():
    out = {}
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, r in enumerate(ds):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append(dict(id=f"gsm8k-{i}", messages=[{"role":"user","content":r["question"]+SUF}],
            reference=ans, meta={"full_solution": r["answer"]}, max_tokens=1024))
    out["gsm8k"] = write_eval("math", "gsm8k", items)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for r in ds:
        items.append(dict(id=r["unique_id"], messages=[{"role":"user","content":r["problem"]+SUF}],
            reference=r["answer"], meta={"subject":r["subject"],"level":r["level"],"solution":r["solution"]}, max_tokens=1024))
    out["math500"] = write_eval("math", "math500", items, strat_key=lambda x: x["meta"]["subject"])
    update_manifest("eval", "math", {"gsm8k":{"hf":"openai/gsm8k","config":"main","split":"test","revision":hf_sha("openai/gsm8k"),"license":"MIT",**out["gsm8k"]},
      "math500":{"hf":"HuggingFaceH4/MATH-500","split":"test","revision":hf_sha("HuggingFaceH4/MATH-500"),"license":"MIT (PRM800K/MATH; no explicit card license)",**out["math500"]}})
main()

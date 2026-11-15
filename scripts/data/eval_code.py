import sys, os, json, base64, zlib, pickle; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from datasets import load_dataset
from huggingface_hub import hf_hub_download
LCB_CUTOFF = "2025-02-01"  # nothing >= 2025-06-01 exists in the public release (max contest_date 2025-04-06)
def main():
    out = {}; man = {}
    ds = load_dataset("evalplus/humanevalplus", split="test")
    items = []
    for r in ds:
        p = ("Complete the following Python function. Reply with the complete function implementation "
             "(including the signature) in a single ```python code block, with no explanation.\n\n```python\n" + r["prompt"].strip("\n") + "\n```")
        items.append(dict(id=r["task_id"].replace("/", "-").replace("HumanEval", "humaneval"), messages=[{"role":"user","content":p}],
            reference=r["prompt"]+r["canonical_solution"], meta={"test":r["test"],"entry_point":r["entry_point"],"test_imports":[],"task_prompt":r["prompt"],
              "exec":"program = extracted_code ; then run reference (defines check(candidate)) ; check(entry_point)"}, max_tokens=1024))
    out["humaneval_plus"] = write_eval("code", "humaneval_plus", items)
    ds = load_dataset("evalplus/mbppplus", split="test")
    items = []
    for r in ds:
        tl = r["test_list"]
        p = ("Write a Python function for the following task. Reply with the complete code in a single ```python code block, with no explanation.\n\n"
             f"Task: {r['prompt']}\nYour code must pass this example test:\n{tl[0]}")
        items.append(dict(id=f"mbpp-{r['task_id']}", messages=[{"role":"user","content":p}], reference=r["code"],
            meta={"test":r["test"],"test_imports":(json.loads(r["test_imports"]) if isinstance(r["test_imports"],str) else list(r["test_imports"])),"entry_point":None,"test_list":tl,
                  "exec":"run test_imports, then extracted code, then reference script (evalplus-style plus inputs/results asserts)"}, max_tokens=1024))
    out["mbpp_plus"] = write_eval("code", "mbpp_plus", items)
    # LiveCodeBench
    p = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    items = []; side = []
    for l in open(p):
        r = json.loads(l)
        if r["contest_date"] < LCB_CUTOFF: continue
        md = json.loads(r["metadata"] or "{}"); pub = json.loads(r["public_test_cases"])
        func = md.get("func_name")
        q = r["question_content"]
        if r["starter_code"]:
            body = (f"{q}\n\nYou must use this starter code and complete it. Reply with the full solution in a single ```python code block, no explanation.\n\n"
                    f"```python\n{r['starter_code']}\n```")
        else:
            body = (f"{q}\n\nRead the input from standard input and write the answer to standard output. "
                    "Reply with the full Python program in a single ```python code block, no explanation.")
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"].encode()))))
        io_tests = [{"input":t["input"],"output":t["output"]} for t in pub + priv]
        iid = f"lcb-{r['question_id']}"
        items.append(dict(id=iid, messages=[{"role":"user","content":body}], reference=None,
            meta={"platform":r["platform"],"contest_date":r["contest_date"],"difficulty":r["difficulty"],"func_name":func,"starter_code":r["starter_code"],
                  "io_tests":io_tests}, max_tokens=1024))
    out["livecodebench"] = write_eval("code", "livecodebench", items, strat_key=lambda x: x["meta"]["platform"]+x["meta"]["difficulty"])
    update_manifest("eval", "code", {
      "humaneval_plus":{"hf":"evalplus/humanevalplus","split":"test","revision":hf_sha("evalplus/humanevalplus"),"license":"Apache-2.0",**out["humaneval_plus"]},
      "mbpp_plus":{"hf":"evalplus/mbppplus","split":"test","revision":hf_sha("evalplus/mbppplus"),"license":"Apache-2.0",**out["mbpp_plus"]},
      "livecodebench":{"hf":"livecodebench/code_generation_lite","file":"test6.jsonl","revision":hf_sha("livecodebench/code_generation_lite"),"license":"CC (card: cc)",
        "substitution":f"Requested >=2025-06-01 but public release v6 ends 2025-04-06 (0 problems qualify). Used contest_date >= {LCB_CUTOFF} (the LCB v6 Feb-Apr 2025 window; post Qwen3 data cutoff is not guaranteed).",**out["livecodebench"]}})
main()

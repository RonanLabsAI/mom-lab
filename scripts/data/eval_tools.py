import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from toolfmt import *
from huggingface_hub import hf_hub_download
R = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
def main():
    items = []; cnt = {}
    for cat in ["simple", "multiple", "parallel", "parallel_multiple"]:
        q = [json.loads(l) for l in open(hf_hub_download(R, f"BFCL_v3_{cat}.json", repo_type="dataset"))]
        a = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(hf_hub_download(R, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset"))}
        for r in q:
            msgs = r["question"][0]
            assert len(r["question"]) == 1 and len(msgs) == 1
            items.append(dict(id=f"bfcl-{r['id']}", messages=[{"role":"user","content":tools_prompt(r["function"], msgs[0]["content"])}],
                reference=a[r["id"]], meta={"category":cat,"function":r["function"],"bfcl_id":r["id"],
                 "reference_format":"BFCL ground_truth: list of {func_name: {param: [allowed values...]}}; model output is a JSON list of {name, arguments}"}, max_tokens=512))
        cnt[cat] = len(q)
    print(cnt)
    o = write_eval("tools", "bfcl", items, strat_key=lambda x: x["meta"]["category"])
    update_manifest("eval", "tools", {"bfcl":{"hf":R,"files":"BFCL_v3_{simple,multiple,parallel,parallel_multiple}.json + possible_answer/","revision":hf_sha(R),"license":"Apache-2.0","category_counts":cnt,**o}})
main()

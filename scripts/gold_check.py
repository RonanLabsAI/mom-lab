"""Feed gold references through each scorer as if they were model outputs. Expect ~100."""
import json,sys,importlib
from pathlib import Path
sys.path.insert(0,'.')
def load(p): return [json.loads(l) for l in open(p)]
def gold_output(dom,bench,it):
    r=it["reference"]
    if dom=="math": return f"The answer is \\boxed{{{r}}}."
    if dom=="sql": return f"```sql\n{r}\n```"
    if dom=="translate": return r
    if bench=="mmlu_pro": return f"Answer: {r}"
    if dom=="code": return f"```python\n{r}\n```" if isinstance(r,str) and r.strip() else None
    if dom=="tools":
        calls=[]
        for c in r:
            for fn,args in c.items(): calls.append({"name":fn,"arguments":{k:v[0] for k,v in args.items() if v and v[0]!=""}})
        return json.dumps(calls)
    return None
from harness.score import math_,sql,translate,general,code,tools
mods={"math":math_,"sql":sql,"translate":translate,"general":general,"code":code,"tools":tools}
for dom,bench in [("math","gsm8k"),("math","math500"),("sql","bird_dev"),("translate","flores200"),("general","mmlu_pro"),("code","humaneval_plus"),("code","mbpp_plus"),("tools","bfcl")]:
    items=load(f"data/eval/{dom}/{bench}/dev.jsonl")[:150]
    gens={}
    for it in items:
        o=gold_output(dom,bench,it)
        if o is not None: gens[it["id"]]={"output":o}
    items=[it for it in items if it["id"] in gens]
    m=mods[dom]
    fn={"gsm8k":"score_gsm8k","math500":"score_math500","bird_dev":"score_sql","flores200":"score_flores","mmlu_pro":"score_mmlu_pro","humaneval_plus":"score_code","mbpp_plus":"score_code","bfcl":"score_bfcl"}[bench]
    res=getattr(m,fn)(items,gens)
    print(f"{dom:10s} {bench:15s} n={len(items):4d} gold-score={res['score']:.1f}")

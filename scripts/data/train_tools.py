import sys, os, json, ast, re; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from toolfmt import *
from datasets import load_dataset
def mk(src, i, funcs, q, calls):
    if not calls or not funcs or not q.strip(): return None
    return dict(id=f"{src}-{i}", messages=[{"role":"user","content":tools_prompt(funcs, q)},{"role":"assistant","content":tools_answer(calls)}], domain="tools", source=src)
def xlam():
    out = []
    for r in load_dataset("lockon/xlam-function-calling-60k", split="train"):
        try: f = json.loads(r["tools"]); c = json.loads(r["answers"])
        except Exception: continue
        if all(isinstance(x, dict) and "name" in x and "arguments" in x for x in c):
            m = mk("xlam60k", r["id"], f, r["query"], c)
            if m: out.append(m)
    return out
def toolace():
    out = []
    for i, r in enumerate(load_dataset("Team-ACE/ToolACE", split="train")):
        conv = r["conversations"]
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant": continue
        s = r["system"]; k = s.find("[")
        try: funcs, _ = json.JSONDecoder().raw_decode(s[k:])
        except Exception: continue
        a = conv[1]["value"].strip()
        try:
            e = ast.parse(a, mode="eval").body
            if not isinstance(e, ast.List) or not all(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) for x in e.elts): continue
            calls = [{"name": x.func.id, "arguments": {kw.arg: ast.literal_eval(kw.value) for kw in x.keywords}} for x in e.elts]
        except Exception: continue
        m = mk("toolace", i, funcs, conv[0]["value"], calls)
        if m: out.append(m)
    return out
def hermes():
    out = []
    for cfg in ["func_calling_singleturn", "func_calling"]:
        for i, r in enumerate(load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train")):
            try: funcs = [t["function"] for t in json.loads(r["tools"])]
            except Exception: continue
            c = r["conversations"]
            h = [x for x in c if x["from"] == "human"]; g = [x for x in c if x["from"] == "gpt"]
            if not h or not g or "<tool_call>" not in g[0]["value"]: continue
            calls = []
            for t in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", g[0]["value"], re.S):
                try: d = json.loads(t)
                except Exception:
                    try: d = ast.literal_eval(t)
                    except Exception: d = None
                if isinstance(d, dict) and "name" in d and "arguments" in d: calls.append(d)
            m = mk("hermes_" + cfg, i, funcs, h[0]["value"], calls)
            if m: out.append(m)
    return out
def main():
    x, t, h = xlam(), toolace(), hermes(); print(len(x), len(t), len(h))
    rows = x + t + h
    finalize_train("tools", rows, {"sources":{
      "xlam60k":{"hf":"lockon/xlam-function-calling-60k (ungated mirror of Salesforce/xlam-function-calling-60k)","revision":hf_sha("lockon/xlam-function-calling-60k"),"license":"CC BY 4.0","n":len(x)},
      "toolace":{"hf":"Team-ACE/ToolACE","revision":hf_sha("Team-ACE/ToolACE"),"license":"Apache-2.0 (generated with GPT-4-class models)","n":len(t),"note":"first-turn single-call-list examples, pythonic calls parsed to JSON"},
      "hermes":{"hf":"NousResearch/hermes-function-calling-v1","revision":hf_sha("NousResearch/hermes-function-calling-v1"),"license":"Apache-2.0","n":len(h),"configs":["func_calling_singleturn","func_calling"]}},
      "substitution":"Salesforce/xlam-function-calling-60k is gated (401, no token); used the ungated lockon mirror (same 60k rows, CC-BY-4.0 permits redistribution; row-identity to Salesforce original not verified) plus ToolACE and Hermes."})
main()

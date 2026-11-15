import sys, os, random; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from datasets import load_dataset
EXCL = ["personahub_math_v5_regen_149960","numinamath_tir_math_decontaminated","tulu_v3.9_open_math_2_gsm8k_50k","tulu-3-sft-personas-math-grade",
        "tulu_v3.9_personahub_math_interm_algebra_20k","evol_codealpaca_heval_decontaminated","personahub_code_v2_34999"]
N = 150000
def main():
    R = "allenai/tulu-3-sft-mixture"
    d = load_dataset(R, split="train")
    keep = []
    for i, (m, s) in enumerate(zip(d["messages"], d["source"])):
        if any(s.endswith(e) for e in EXCL): continue
        if len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant": continue
        keep.append(i)
    print("eligible 2-turn non-math/code:", len(keep))
    random.Random(SEED).shuffle(keep); keep = keep[:N]
    rows = [dict(id=f"tulu3-{d[i]['id']}", messages=[{"role":m["role"],"content":m["content"]} for m in d[i]["messages"]], domain="general", source="tulu3_"+d[i]["source"].split("/")[-1]) for i in keep]
    finalize_train("general", rows, {"sources":{"tulu3_sft_mixture":{"hf":R,"revision":hf_sha(R),"license":"ODC-BY 1.0 (attribution; component datasets carry their own terms, e.g. WildChat ODC-BY, Aya Apache-2.0, FLAN, CoCoNot/WildGuard/WildJailbreak AI2 ImpACT-low-risk terms; outputs from GPT-4/Claude/Llama-3.1 in parts)"}},
      "excluded_sources":EXCL,"filters":"only single-turn user/assistant conversations; random 150k sample of the eligible remainder"})
main()

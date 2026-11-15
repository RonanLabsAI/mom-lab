import sys, os, json, random; sys.path.insert(0, os.path.dirname(__file__))
from common import *
D = os.path.join(DATA, "raw/flores/flores200_dataset/devtest")
NAMES = {"eng_Latn":"English","spa_Latn":"Spanish","deu_Latn":"German","zho_Hans":"Simplified Chinese","jpn_Jpan":"Japanese"}
DIRS = [("eng_Latn",t) for t in ["spa_Latn","deu_Latn","zho_Hans","jpn_Jpan"]] + [("spa_Latn","eng_Latn"),("deu_Latn","eng_Latn")]
def prompt(s, t, text):
    return f"Translate the following text from {NAMES[s]} to {NAMES[t]}. Reply with only the translation.\n\n{text}"
def main():
    lines = {l: [x.rstrip("\n") for x in open(os.path.join(D, f"{l}.devtest"), encoding="utf-8")] for l in NAMES}
    n = {len(v) for v in lines.values()}; assert n == {1012}, n
    ids = list(range(1012)); random.Random(SEED).shuffle(ids)
    dev_ids = set(ids[:506])
    split = {"dev": [], "test": []}
    for s, t in DIRS:
        for i in range(1012):
            it = dict(id=f"flores-{s}-{t}-{i}", messages=[{"role":"user","content":prompt(s, t, lines[s][i])}], reference=lines[t][i],
                meta={"src_lang":s,"tgt_lang":t,"direction":f"{s}-{t}","sentence_id":i,"source":lines[s][i]}, max_tokens=512)
            split["dev" if i in dev_ids else "test"].append(it)
    for k, v in split.items():
        write_jsonl(os.path.join(DATA, "eval/translate/flores200", f"{k}.jsonl"), v)
    o = {k: len(v) for k, v in split.items()}; print(o)
    update_manifest("eval", "translate", {"flores200":{"source":"https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (official FLORES-200, devtest, sha256 b8b0b767...)",
      "revision":"sha256:b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6","license":"CC BY-SA 4.0 (share-alike)",
      "substitution":"openlanguagedata/flores_plus is gated (401, needs token); used the official Meta FLORES-200 release instead (same sentences; zho_Hans label as in original).",
      "split":"506 sentence ids dev / 506 test, disjoint across all 6 directions","directions":[f"{s}-{t}" for s,t in DIRS],**o}})
main()

import sys, os, random, re; sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd
from huggingface_hub import hf_hub_download
NAMES = {"en":"English","es":"Spanish","de":"German","zh":"Simplified Chinese","ja":"Japanese"}
PER_DIR = 30000
def prompt(s, t, text): return f"Translate the following text from {NAMES[s]} to {NAMES[t]}. Reply with only the translation.\n\n{text}"
CJK = re.compile(r"[぀-ヿ㐀-鿿]")
def ok(s, t, a, b):
    a, b = a.strip(), b.strip()
    mn = {"en": 40, "es": 40, "de": 40, "zh": 12, "ja": 12}
    mx = {"en": 400, "es": 400, "de": 400, "zh": 150, "ja": 150}
    if not (mn[s] <= len(a) <= mx[s] and mn[t] <= len(b) <= mx[t]): return False
    r = len(a) / max(len(b), 1)
    lo, hi = (0.3, 3.5) if "zh" not in (s, t) and "ja" not in (s, t) else (0.8, 12) if s in ("en","es","de") else (0.08, 1.25)
    if not (lo <= r <= hi): return False
    if re.search(r"https?://|www\.|\d{4,}", a) and False: return False
    if a.lower() == b.lower(): return False
    for lang, x in ((s, a), (t, b)):
        if lang in ("zh", "ja") and len(CJK.findall(x)) < 0.4 * len(x): return False
        if lang in ("en", "es", "de") and CJK.search(x): return False
    return True
def main():
    R = "Helsinki-NLP/opus-100"; rows = []; counts = {}
    rng = random.Random(SEED)
    # (config, a, b, directions to emit from the pair-set: (src,tgt))
    jobs = [("en-es","en","es",[("en","es"),("es","en")]), ("de-en","de","en",[("en","de"),("de","en")]), ("en-zh","en","zh",[("en","zh")]), ("en-ja","en","ja",[("en","ja")])]
    for cfg, a, b, dirs in jobs:
        df = pd.read_parquet(hf_hub_download(R, f"{cfg}/train-00000-of-00001.parquet", repo_type="dataset"))
        pairs = [(t[a], t[b]) for t in df["translation"]]
        rng.shuffle(pairs); print(cfg, len(pairs))
        good = [(x, y) for x, y in pairs if ok(a, b, x, y)]
        seen = set(); g2 = []
        for x, y in good:
            if x in seen: continue
            seen.add(x); g2.append((x, y))
        print(" good", len(g2))
        per = PER_DIR
        for k, (s, t) in enumerate(dirs):
            chunk = g2[k * per:(k + 1) * per]
            for i, (x, y) in enumerate(chunk):
                src, tgt = (x, y) if (s, t) == (a, b) else (y, x)
                rows.append(dict(id=f"opus100-{s}-{t}-{i}", messages=[{"role":"user","content":prompt(s, t, src.strip())},{"role":"assistant","content":tgt.strip()}], domain="translate", source=f"opus100_{s}-{t}"))
            counts[f"{s}-{t}"] = len(chunk)
    print(counts)
    finalize_train("translate", rows, {"sources":{"opus100":{"hf":R,"revision":hf_sha(R),"license":"UNKNOWN: card lists none; underlying OPUS corpora (OpenSubtitles, GNOME, Ubuntu, Tatoeba, UN, etc.) have mixed licenses - verify before commercial release","per_direction":counts}},
      "filters":"char length bands (40-400 Latin, 12-150 CJK), length ratio, script check, exact-source dedupe, src!=tgt; en-es and de-en pairs split into disjoint halves for the two directions",
      "note":"OPUS-100 is noisy (subtitles-heavy); FLORES sentences are longer and encyclopedic, so expect domain gap"})
main()

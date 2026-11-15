"""translate: FLORES-200 chrF++ (sacrebleu CHRF word_order=2), corpus-level per direction.

Item schema: reference = target sentence; meta.direction ("eng_Latn-spa_Latn") or meta.src_lang +
meta.tgt_lang. Benchmark score = mean over directions of corpus chrF++. Per-item values are
sentence-level chrF++ (0-100), used only for bootstrapping; groups give the direction of each item.
"""
from __future__ import annotations

from collections import defaultdict

from ..common import strip_think
from . import output_of


def direction_of(it) -> str:
    m = it.get("meta") or {}
    if m.get("direction"):
        return m["direction"]
    return f"{m.get('src_lang', '?')}-{m.get('tgt_lang', '?')}"


def clean(out: str) -> str:
    out = strip_think(out)
    return out.strip().split("\n")[0].strip() if out.strip() else ""


def score_flores(items, gens, workers=None):
    from sacrebleu.metrics import CHRF
    corpus, sent = CHRF(word_order=2), CHRF(word_order=2)
    by_dir = defaultdict(lambda: ([], []))
    vals, groups = [], []
    for it in items:
        o = output_of(gens, it["id"])
        hyp = clean(o) if o is not None else ""
        ref = it["reference"]
        d = direction_of(it)
        by_dir[d][0].append(hyp)
        by_dir[d][1].append(ref)
        groups.append(d)
        vals.append(sent.sentence_score(hyp, [ref]).score if hyp else 0.0)
    per_dir = {d: corpus.corpus_score(h, [r]).score for d, (h, r) in by_dir.items()}
    return {"ids": [it["id"] for it in items], "correct": vals, "groups": groups,
            "score": sum(per_dir.values()) / max(1, len(per_dir)),
            "extra": {"per_direction": per_dir}}

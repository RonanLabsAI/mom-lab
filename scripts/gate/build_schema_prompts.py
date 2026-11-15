"""Build the schema-arm prompt variants for the METAR gate (preregs/2026-11-14-metar-schema-gate.md, "Prompts").

The Part D-C/F prompts ask for the answer "in a ```json block", and the F exemplar answers are fenced. A JSON-schema
grammar cannot emit backticks, so schema arms get the same prompts with exactly two edits:
  1. every user turn: "(in a ```json block)" -> "(raw JSON, no code fence)"
  2. every assistant (exemplar) turn: the "```json\n" ... "\n```" fence removed; the JSON inside is unchanged.
Nothing else changes (ids, references, meta, max_tokens). Deterministic; unconstrained arms keep the fenced prompts.

usage: python3 -P scripts/gate/build_schema_prompts.py   (reads the main checkout, writes data_gate_{zs,fs}/ here)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

MAIN = Path(os.environ.get("MOM_MAIN", str(Path(__file__).resolve().parents[2])))
WT = Path(__file__).resolve().parents[2]
OLD, NEW = "(in a ```json block)", "(raw JSON, no code fence)"
SOURCES = {"data_gate_zs": MAIN / "data_partd_c", "data_gate_fs": MAIN / "data_partd_f"}
REL = Path("eval/aviation/metar")


def _unfence(s: str) -> str:
    if not (s.startswith("```json\n") and s.endswith("\n```")):
        raise ValueError("exemplar answer is not a ```json fenced block")
    return s[len("```json\n"):-len("\n```")]


def transform(item: dict) -> dict:
    it = copy.deepcopy(item)
    n = 0
    for m in it["messages"]:
        if m["role"] == "user":
            if OLD not in m["content"]:
                raise ValueError(f"{it['id']}: user turn lacks the fence instruction")
            m["content"] = m["content"].replace(OLD, NEW)
            n += 1
        elif m["role"] == "assistant":
            m["content"] = _unfence(m["content"])
    assert n >= 1
    return it


def main():
    hashes = {}
    for dst, src in SOURCES.items():
        for split in ("dev", "test"):
            items = [json.loads(l) for l in (src / REL / f"{split}.jsonl").open() if l.strip()]
            out = WT / dst / REL / f"{split}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(json.dumps(transform(it), ensure_ascii=False) + "\n" for it in items)
            out.write_text(text)
            hashes[str(out.relative_to(WT))] = hashlib.sha256(text.encode()).hexdigest()
            print(out.relative_to(WT), len(items), hashes[str(out.relative_to(WT))][:16])
    (WT / "configs/gate/schema_prompts.sha256").write_text(
        "".join(f"{h}  {p}\n" for p, h in sorted(hashes.items())))


if __name__ == "__main__":
    main()

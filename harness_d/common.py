"""Shared paths, benchmark registry and jsonl helpers.

harness_d = copy of the frozen harness/ (Part D / Experiment C) with the niche-subject domains added to
EXTRA_DOMAINS (aviation, medcalc, avknow; data under data_partd_c/, built by scripts/partd_c/). Only change to
existing behaviour: the default data root can be set with the MOM_DATA_ROOT environment variable (so
scripts/evalq_d.py, which passes no --data-root, can point at data_partd_c). Unset -> data/ as before.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MOM_DATA_ROOT") or (ROOT / "data"))
if not DATA.is_absolute():
    DATA = ROOT / DATA
RESULTS = ROOT / "results"

DOMAINS = {
    "math": ["gsm8k", "math500"],
    "code": ["humaneval_plus", "mbpp_plus", "livecodebench"],
    "sql": ["bird_dev"],
    "tools": ["bfcl"],
    "translate": ["flores200"],
    "general": ["mmlu_pro", "ifeval"],
}
# Mixed-skill sets (translated prompts, scored with the source benchmark's scorer). NOT part of the
# six core domains: excluded from A; selected explicitly (--domains mixed / present in aggregate).
EXTRA_DOMAINS = {"mixed": ["gsm8k_ml", "bird_ml", "mbpp_ml", "bfcl_ml"],
                 # Step 4 paraphrase set: templates removed, prompts reworded (scripts/build_para.py)
                 "para": ["gsm8k_pp", "bird_pp", "mbpp_pp", "bfcl_pp", "mmlupro_pp", "flores_pp"],
                 # Part D / Experiment C (data root data_partd_c, scripts/partd_c/): niche SKILL tasks + KNOWLEDGE control
                 "aviation": ["metar"],        # METAR decoding -> fixed JSON schema (NOTAM skipped: no data licence)
                 "medcalc": ["medcalc"],       # MedCalc-Bench v1.2 (patient note -> clinical calculation)
                 "avknow": ["preflight"]}      # Pre-Flight aviation-knowledge multiple choice (control)
ALL_DOMAINS = {**DOMAINS, **EXTRA_DOMAINS}
BENCH_DOMAIN = {b: d for d, bs in ALL_DOMAINS.items() for b in bs}


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def eval_path(domain: str, bench: str, split: str, data_root: Path = DATA) -> Path:
    return Path(data_root) / "eval" / domain / bench / f"{split}.jsonl"


def run_dir(step: str, run_id: str, results_root: Path = RESULTS) -> Path:
    return Path(results_root) / step / run_id


def gen_path(rdir: Path, domain: str, bench: str, split: str) -> Path:
    return Path(rdir) / "gen" / domain / f"{bench}.{split}.jsonl"


def check_split(split: str, allow_test: bool):
    if split == "test" and not allow_test:
        raise SystemExit("Refusing to touch the test split without --allow-test "
                         "(CONTRACT: test is never read during Part A).")


def git_hash() -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Drop closed <think> blocks; if a think block never closes, drop everything after it."""
    text = _THINK.sub("", text or "")
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.replace("</think>", "").strip()

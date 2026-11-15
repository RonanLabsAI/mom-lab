#!/usr/bin/env bash
# Rebuild every Part D / Experiment C dataset from scratch (network needed; no GPU). Run from the repo root.
set -euo pipefail
PY=".venv/bin/python -P"
export HF_HUB_DISABLE_XET=1
$PY scripts/partd_c/fetch_metar.py            # ~25 min (polite sequential IEM requests)
$PY scripts/partd_c/build_metar.py
$PY scripts/partd_c/build_medcalc.py          # downloads ncbi/MedCalc-Bench @ pinned revision
$PY scripts/partd_c/build_preflight.py        # downloads AirsideLabs/pre-flight-06 @ pinned revision
$PY scripts/partd_c/make_replay.py 0.2 metar medcalc   # needs data/mixtures/N14M/spec/*.jsonl (main data pipeline)
$PY scripts/partd_c/check.py
$PY -m pytest -q -p no:cacheprovider harness_d/tests/test_partd_c.py
sha256sum data_partd_c/eval/*/*/*.jsonl data/mixtures/partd_c/*.jsonl data/mixtures/partd_c/replay/*.jsonl > data_partd_c/SHA256SUMS
# optional independent decoder validation (separate venv with the PyPI `metar` package):
#   python3 -m venv /tmp/metarvenv && /tmp/metarvenv/bin/pip install metar==1.11.0
#   /tmp/metarvenv/bin/python -I scripts/partd_c/validate_metar_decoder.py

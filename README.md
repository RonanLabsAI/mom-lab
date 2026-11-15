# mom-lab: Mixture of Models on desk hardware

Stephen J. Ronan, MD - RonanLabs

> **Preregistered locally; timestamps not externally verifiable.**
> The preregistrations, addenda and freeze manifests in this repository were written and hashed before the runs
> they govern, but only on our own machines. Nobody outside the lab can check those timestamps. Read them as
> locked plans with a local paper trail, not as externally timestamped preregistrations. From 2026-11-14,
> new work is preregistered in a public repository before it runs.

mom-lab asks one question: can a small frozen base model plus a few routed specialists do the job of the next
model size up? It runs on the RonanLabs fleet (two DGX Station GB300s with RTX PRO 6000s, DGX Spark GB10s). All
models are Qwen3 (1.7B, 4B, 8B, plus 14B and 32B as references). Every comparison is paired by item and
bootstrapped (2,000 resamples, 95% percentile CI). Outputs are deterministic: vLLM batch-invariant mode gave
byte-identical greedy outputs across chips (1,179/1,179 prompts on Spark, GB300, and later 9,262/9,262 items
across the two RTX PRO 6000s).

Short version: size buys knowledge, and specialists buy skills. Where a task needs a strict output format, a
small adapter beats much bigger prompted models. Where the task is arithmetic over a note, one worked example in
the prompt beats the adapter.

## Results

All numbers are test split unless marked. Sources: `results/analysis/*.json` and `notebook/`.

### Niche skills (Part D-C and D-F)

| Task (n) | 4B + LoRA adapter | Best prompted 8B | Qwen3-32B thinking | Output tokens per item, adapter vs best 8B |
|---|---|---|---|---|
| METAR to 14-field JSON, all fields exact (1,000) | **99.8** | 83.2 (few-shot + thinking) | 88.9 (one seed) | 143 vs 1,664 (11.6x fewer); 32B thinking 937 |
| MedCalc-Bench v1.2 accuracy (1,100) | 58.7 | **79.8** (few-shot + thinking) | 60.0 | 508 vs 2,209 |

- METAR: the adapter beats the best 8B condition by +16.7 [+15.0, +18.5] and 32B thinking by +10.9 [+9.1,
  +13.0]. A 1.7B adapter also beats the best 8B (+15.9 [+14.2, +17.7]). Caveat disclosed before the F runs: the
  untouched 8B often wrote times as "0355" instead of "03:55". Accepting that format moves zero-shot 8B from 2.1
  to 18.5, and the rest of its errors are field content (flight category, temperature and dewpoint, wind).
- MedCalc reverses. 8B few-shot + thinking beats the adapter by 21.2 [18.6, 23.9]. 4B with one worked example and
  no training (65.2) also beats the 4B adapter. On this task one worked example in the prompt did better than
  training on 9,827 notes.
- Constrained-decoding gate (Part G, locked in `preregs/2026-11-14-metar-schema-gate.md`, run 2026-11-13/14 PT):
  with a JSON schema under XGrammar, no untouched 8B or 32B arm came within 2 points of the adapter. The best,
  32B with thinking under the schema, scored 89.6 at 935 tokens per item; the adapter leads it by +10.2 [+8.4, +12.2]
  (n = 1,000). Constrained arms without thinking write about as many tokens as the adapter (139-159) but score
  20.4-66.2. The schema helped only where the error was format (zero-shot 8B 2.1 -> 20.4). The 11.6x figure above is
  against a thinking trace; against the best constrained arm the ratio is 6.5x. Deviations:
  `content/PART_G_DEVIATIONS.md` (three, none changing an arm or test). Result file:
  `results/analysis/gate_metar_schema.json`.

### Fresh SQL and tools (Part D-B; Spider n=2,000, BFCL live n=1,292)

- 4B + adapter library vs untouched 8B on SQL: +6.65 [+4.75, +8.52]. On tools it is non-inferior (+2.09 [+0.18,
  +3.97]).
- Caveat 1: Spider train is in our SQL training data, so this is a held-out-items test, not a held-out-dataset
  test. Spider dev items inside sql-create-context were removed (1,055 dropped).
- Caveat 2: a single pooled 4B fine-tune (75.1) and a distilled 4B (74.5) match or beat the library (74.1) on SQL.
  The library's advantage is that it does not lose other subjects (Part B), not that it gains more SQL.

### Six-domain study (Parts A-C, Qwen3-1.7B and 4B)

- Part A (dev only): keep the untouched base always on and switch in specialists only where they beat it (SQL and
  tools). Pooled retraining and merged specialists lose skills the base already had. Routing is not the hard part:
  a rule router and a small embedding router keep nearly all of the perfect-routing gain.
- Part B (test, frozen manifest `5b61273f...`): every primary hypothesis came back INCONCLUSIVE except H2-4bit,
  which FAILED (4B + LoRA vs 4-bit 8B on paraphrases: -5.34 [-8.41, -2.42]). Most of the open components are
  on the paraphrase set. H2 (4B + LoRA vs 8B, non-inferiority margin -2) also misses on core: -1.40 [-2.48,
  -0.27].
- Part C (fresh 1,562-item paraphrase set): the 1.7B replay library passes every component (fresh paraphrase vs
  base +4.02 [+2.34, +5.67], non-inferior to pooled +0.56 [-0.83, +2.02]).
- Nulls kept at full weight: merging (all merges below pooled), rank 64/128 over rank 16 (all 8 tests
  inconclusive), self-distilled replay in a routed library (FAIL, -0.50 [-0.82, -0.21]).

## Layout

| Path | What it holds |
|---|---|
| `GOAL.md`, `CONTRACT.md` | Study goal, eval protocol v2, data and harness contract |
| `content/PART_*_PREREG.md`, `content/PART_D_*_ADDENDUM.md`, `content/PART_D_DEVIATIONS.md` | Locked plans and every deviation |
| `artifacts/*.sha256` | Freeze manifests: code, configs, test files, weight and adapter hashes |
| `notebook/` | Dated lab notebook (PT) |
| `harness/`, `harness_d/` | Generation, scoring and aggregation (`harness_d` adds METAR, MedCalc, Pre-Flight) |
| `scripts/data/`, `scripts/partd_c/`, `scripts/partd_f/` | Deterministic data builders |
| `scripts/*_analysis.py` | Frozen analysis scripts, one per part |
| `results/<step>/<run>/scores.json`, `results/analysis/*.json` | Per-item correctness and test verdicts |
| `train/` | The training loop (full fine-tune and LoRA) |
| `preregs/` | New preregistrations (from 2026-11-14) |
| `RELEASE.md`, `LICENSES.md` | What ships, what does not, and under which licence |

Not in git: raw downloads, built training mixtures, model generations (`results/**/gen/`), weights and adapters.
See `RELEASE.md`.

## Rerun

1. Environment: Python 3.12, `python3 -m venv .venv && .venv/bin/pip install -r harness/requirements.txt`. Do not
   change pins without checking that file. Serving image: `docker/Dockerfile` (vLLM nightly `af7f9488`).
2. Data: `scripts/data/fetch_raw.sh` and `scripts/data/*.py` (Parts A-B), `scripts/partd_c/build_all.sh` (METAR,
   MedCalc, Pre-Flight; network needed, no GPU). The METAR, MedCalc and Pre-Flight eval files rebuild
   byte-identical (`sha256sum -c data_partd_c/SHA256SUMS`, checked again 2026-11-13).
3. Serve with batch-invariant vLLM: `VLLM_BATCH_INVARIANT=1 vllm serve <model> --max-model-len 8192 --seed 0`.
4. Generate and score:
   `.venv/bin/python -m harness_d.generate --endpoint <url>/v1 --model <name> --run-id <run> --step <step> --split test --allow-test --data-root data_partd_c --domains aviation,medcalc,avknow`
   then `.venv/bin/python -m harness_d.aggregate --data-root data_partd_c --run-id <run> --step <step> --split test`.
5. Analysis: `.venv/bin/python -P scripts/partd_f_analysis.py` (and the other `scripts/*_analysis.py`). Check the
   locks first with `scripts/release/verify_freezes.sh`.

## Licences

Code: Apache-2.0 (see `LICENSE`). Release preparation and remaining publication steps are in `RELEASE-CHECKLIST.md`. Data keeps its source licences (`LICENSES.md`, `DATA-LICENSE.md`). Anything derived
from MedCalc-Bench is CC BY-SA 4.0, and we treat MedCalc adapters the same way. METAR reports are US government
observations from the Iowa Environmental Mesonet archive. Pre-Flight is MIT (Airside Labs). Qwen3 weights are
Apache-2.0. The NOTAM datasets we looked at have no licence, so we did not use or redistribute them.

## AI assistance

Agents designed, ran, and drafted this work under PI supervision; the PI verified every number and is responsible
for every claim.

# Experiment C datasets (Part D): niche skill tasks + aviation knowledge control

Built 2026-11-01 on the Threadripper (CPU only, no model run). Builders: `scripts/partd_c/`. Base seed `20261101`
(`scripts/partd_c/common.py`). Every builder is deterministic: a second full rebuild reproduced every output
byte-for-byte (`sha256sum -c data_partd_c/SHA256SUMS`, all OK).

Hypothesis under test: LoRA specialists add most where the base model is weakest. These are SKILL tasks (METAR
decoding, MedCalc) plus a KNOWLEDGE control (Pre-Flight). The prediction is that adapters help the skill tasks much
more than the knowledge task.

## Layout

| benchmark (harness_d domain/bench) | dev | test | train (`data/mixtures/partd_c/`) | 20% replay |
|---|---|---|---|---|
| `aviation/metar`: METAR to JSON | 300 | 1,000 | `metar.jsonl` 8,000 (5.88M tok) | `replay/metar-r20.jsonl` 9,602 |
| `medcalc/medcalc`: MedCalc-Bench v1.2 | 300 | 1,100 | `medcalc.jsonl` 9,827 (12.37M tok) | `replay/medcalc-r20.jsonl` 11,795 |
| `avknow/preflight`: Pre-Flight MC (control) | 60 | 240 | none (the dataset has no train split) | none |
| NOTAM parsing | **skipped**: no licence, see below | | | |

- Eval files are at `data_partd_c/eval/<domain>/<bench>/{dev,test}.jsonl`, as `{id, messages:[user], reference, meta, max_tokens}`.
- Train files are `{id, messages:[user, assistant], domain, source, n_tokens}`. `n_tokens` is the Qwen3-1.7B tokenizer
  length of the chat-formatted conversation (`<|im_start|>role\n…<|im_end|>\n`), the same convention as `scripts/data/common.py`.
- Size limits: the longest train example is 3,882 tokens (MedCalc). The longest eval prompt is 2,566 tokens. Both fit
  the 8,192-token training and serving lengths together with 4,096 output tokens.
- `manifest.json` holds per-source details: revisions, raw-file sha256, drops, station split and category counts.
- `checks.json` holds the results of `check.py`. `metar_decoder_validation.json` holds the decoder cross-checks.
- `raw/` holds the downloads. It is untrusted data, nothing in it is executed, and it can be deleted and re-fetched.
  `raw/notam_probe/` holds the *unlicensed* Knots / NOTAM-Evolve clones, kept only for the licence audit.
  **Do not redistribute it.**

## Evaluation with harness_d

`harness_d/` is a copy of the frozen `harness/`. The changes are:
- The three domains above are registered in `EXTRA_DOMAINS`, so they are not part of the A score.
- Scorers `metar`, `medcalc` and `preflight` are added.
- The default data root can be set with `MOM_DATA_ROOT`. When it is unset the root is `data/`, as before.

`scripts/evalq_d.py` is `scripts/evalq.py` with `-m harness_d.generate` and `-m harness_d.aggregate` substituted.
Nothing else changed: batch-invariant serving, greedy decoding, 4,096 max tokens.

```
# direct
.venv/bin/python -m harness_d.generate --data-root data_partd_c --domains aviation,medcalc,avknow --split dev ...
.venv/bin/python -m harness_d.aggregate --data-root data_partd_c --run-id R --step S --split dev
# via the queue (Station slots honour EVAL_DOMAINS; the Spark path passes no --domains and runs only the 6 core domains)
MOM_DATA_ROOT=data_partd_c EVAL_DOMAINS=aviation,medcalc,avknow EVAL_SPARKS=0 EVAL_STATION_SLOTS=... python3 scripts/evalq_d.py <trainq> <step>
```

## Sources and licences

| source | URL / revision | licence | verdict |
|---|---|---|---|
| MedCalc-Bench v1.2 | https://huggingface.co/datasets/ncbi/MedCalc-Bench @ `83a1a56d6b53f8d4c0e03f0ba9699becf524a595` (train csv sha256 `aaadfcee…2b71c`, test `c9d219a3…90e3a`). Eval code: https://github.com/ncbi-nlp/MedCalc-Bench @ `20b10f9d66a8c3b0356ae3138533f5ec91666b6a` | CC BY-SA 4.0 | used. Derived files must stay CC BY-SA 4.0 |
| METAR reports | NWS/FAA ASOS observations from the Iowa Environmental Mesonet archive, https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py (`data=metar`, report types routine + SPECI). Per-day csv sha256 is in `raw/metar/fetch_log.json` | public domain (US government observations). IEM serves its archive freely | used |
| Pre-Flight | https://huggingface.co/datasets/AirsideLabs/pre-flight-06 @ `439d2d118fed7d9b009c1f87b9eb1205ab94766e`, the revision inspect_evals pins (`test.jsonl` sha256 `d02a0c67…bab17`, identical at HF main `fdb1e43`). Paper: arXiv 2607.01829 | MIT (HF card and paper §3.4; no LICENSE file or copyright line in the repo) | used. Attribute as "Airside Labs, MIT" |
| Knots | https://github.com/Estrellajer/Knots @ `8935617c` | **none**: GitHub `license: null`, no LICENSE file in the history, and the README's Apache-2.0 badge links to a 404 | **skipped** |
| NOTAM-Evolve | https://github.com/Estrellajer/NOTAM-Evolve @ `ed2f3a19` | **none**: same situation, badge links to a missing LICENSE | **skipped** |

**On METAR vs aviationweather.gov:**
- The AWC data API (`aviationweather.gov/api/data/metar`) serves only the last 15–30 days, so a build from it could
  not be reproduced. The fixed 2024 archive at IEM is used instead.
- AWC is used to validate flight categories (see Checks).

**On NOTAM:**
- Neither dataset has a licence, so by default all rights are reserved. Research use and redistribution of derived
  data are therefore not permitted, and NOTAM was skipped as instructed. The raw NOTAMs come from ADCC (China), not
  the FAA.
- Possible replacements, not yet audited for provenance:
  - `DEEL-AI/NOTAM` (MIT)
  - `AirsideLabs/notam-typed-decisions` (CC BY 4.0, ships a LICENSE)
  - `AeroAIAviation/NOTAM-TAF-SPECI-METAR-ACARS_Decode` (Apache-2.0)
- There is no open bulk archive of FAA NOTAMs. The FAA API and SWIM need accounts and give only current NOTAMs.

## Task definitions

### MedCalc (`scripts/partd_c/build_medcalc.py`, scorer `harness_d/score/medcalc.py`)

**Splits:**
- The official v1.2 split is 10,543 train / 1,100 test. The 10,053 / 1,047 figures in the paper are v1.0.
- **test** = all 1,100 official test rows.
- **dev** = 300 rows from official train:
  - stratified over the 55 calculators (5–6 each);
  - one row per sampled patient note;
  - only notes that do not occur in test.
- **train** = the rest of official train, minus three groups:
  - 316 rows whose patient note (normalised) also occurs in test. The official splits reuse notes across calculators.
  - 99 rows whose note is a dev note (for a different calculator).
  - 1 row whose own gold answer does not pass the official scorer. That is `medcalc-train-03175`, an MDRD GFR of
    4.7e-07 from a creatinine mis-scaled in the source.

**Prompt and target:**
- The prompt is the patient note, then `Task: <question>`, then "work step by step; end with a final line
  `Answer: <value>`", plus a format hint per output type:
  - number (no units)
  - integer
  - MM/DD/YYYY
  - `(W weeks, D days)`
- The train target is the dataset's Ground Truth Explanation followed by `\nAnswer: <ground truth>`.

**Scoring** is a faithful port of the official `extract_answer` + `check_correctness`:
- dates (calculators 13, 68): exact date;
- gestational age (69): (weeks, days) exact;
- the official integer calculator ids: `round(pred) == gt`;
- the official decimal ids: `lower_limit <= pred <= upper_limit`, with the dataset's limits;
- any parse error is wrong.

**Deviations from the official code:**
- **Answer location.** The answer string comes from the last `Answer:` line rather than from the JSON `answer` field.
  The official JSON pattern is kept as a fallback.
- **Expression evaluation.** `eval()` is replaced by an AST arithmetic evaluator with the same function whitelist,
  so model text cannot execute code.
- **Plain numbers.** These are read with `float()` first. The official `eval("05")` raises on a leading zero.

`extra.by_category` reports the paper's 7 categories.

### METAR decoding (`fetch_metar.py`, `build_metar.py`, `metar_decode.py`, scorer `harness_d/score/metar.py`)

**Collection:**
- Stations: 8 per state from the IEM `<ST>_ASOS` networks of all 50 states, seeded. A station qualifies if it is
  online, its archive starts on or before 2015-01-01, and it has a 3-character id. That gives 377 stations, of which
  367 had decodable reports.
- Dates: days 5 and 20 of every month of 2024, all hours, routine + SPECI. That gives 398,826 rows and 398,539
  unique reports.

**Ground truth:**
- It comes from a strict deterministic decoder of the report body (`metar_decode.py`). A report with any
  unrecognised body group is dropped: 21,379 dropped, mostly missing visibility, temperature or altimeter, or
  `///` cloud heights from AUTO stations. 377,160 reports were decoded.
- Schema, 14 fields:
  - station, day, time_utc
  - wind_dir_deg (int / "VRB" / null when calm), wind_speed_kt, wind_gust_kt
  - visibility_sm
  - weather (verbatim present-weather groups)
  - clouds [{cover, base_ft}]
  - ceiling_ft
  - temperature_c, dewpoint_c
  - altimeter_inhg
  - flight_category
- The flight category follows the FAA/AWC definition, taking the worse of ceiling and visibility:
  - LIFR: ceiling < 500 ft or visibility < 1 SM
  - IFR: ceiling < 1,000 ft or visibility < 3 SM
  - MVFR: ceiling ≤ 3,000 ft or visibility ≤ 5 SM
  - VFR: otherwise
- The prompt spells out the schema and these rules.

**Splits (by station, no station shared):**
- 265 train / 29 dev / 73 test stations.
- Each split is sampled stratified by flight category: VFR 40%, MVFR / IFR / LIFR 20% each. A per-station cap is 3x
  the uniform share.
- This yields train 8,000 / dev 300 / test 1,000.
- About 42% of test reports have present weather and about 14% have gusts.

**Scoring:**
- The output is parsed by taking the last ```json block, or else the first `{` … last `}`.
- Each field is compared as follows:
  - strings: case-insensitive;
  - integers: exact (numeric strings accepted);
  - visibility: within 1e-6;
  - altimeter: within 0.005;
  - weather and cloud lists: exact and in order;
  - a missing key is wrong.
- An item is correct only if all 14 fields are correct.
- `extra` reports `mean_field_accuracy`, per-field accuracy and the parse-failure rate.

**TAF was not built.** TAFs have a variable number of FM/TEMPO/BECMG/PROB change groups, so they would need a nested,
variable schema. Field-level all-or-nothing scoring would then be dominated by change-group segmentation rather than
decoding. This dataset is METAR only.

### Pre-Flight (`build_preflight.py`, scorer = harness_d `general.score_mmlu_pro`)

- The dataset has 300 questions and no train split.
- They are split dev 60 / test 240, stratified by the id-range category:
  - ground ops: 152
  - ICAO: 85
  - FAA/US: 51
  - trivia: 8
  - scenario: 4
- The prompt is inspect_ai's `multiple_choice()` single-answer template, which is what inspect_evals uses, with the
  original choice order. It asks for `ANSWER: $LETTER`.
- Scoring uses the MMLU-Pro letter extraction, which takes the last `answer: X`, else a trailing letter, and compares
  it exactly with the target.
- 162 items have 5 choices and 138 have 4. The last choice is always "no suitable option" and is never correct.
- Answer letters are skewed. In test: B 105, C 69, A 60, D 6. Always answering B scores 43.8% on test.

### 20% replay (`scripts/partd_c/make_replay.py`)

- This follows the same recipe as `scripts/make_replay.py`: own data + `round(0.2·n_own/6)` lines sampled from each
  of the six `data/mixtures/N14M/spec/*.jsonl` domains, then shuffled.
- All six domains are replay sources, because the new tasks are none of them.
- Seed is `20261029 + 6 + task index`.
- Result: metar 8,000 + 1,602 replay; medcalc 9,827 + 1,968 replay.
- The N14M spec files come from the main data pipeline (`data/README.md`). Outsiders must build those first.

## Checks (all run 2026-11-01; `check.py` reports 0 failed)

| check | result |
|---|---|
| Gold: reference formatted as output, then scorer (dev / test) | metar 100.00 / 100.00; medcalc 100.00 / 100.00; preflight 100.00 / 100.00 |
| Gold: MedCalc explanation + `Answer:` (the train-target format) on dev / test | 100.00 / 100.00 |
| Gold: train assistant targets vs source ground truth | medcalc 100.00 (n=9,827); metar 100.00 (n=8,000) |
| Empty generations | 0.00 on every file |
| Prompt overlap train–dev, train–test, dev–test (exact and normalised) | 0 for every pair and task |
| METAR station and raw-report overlap across splits | 0 |
| MedCalc patient-note overlap across splits (normalised) | 0 |
| Scorer unit tests (`harness_d/tests/test_partd_c.py`, hand-made right and wrong outputs) | 7 passed |
| harness_d registry; `eval_path(..., data_root=data_partd_c)` for all 6 files; `MOM_DATA_ROOT=data_partd_c` default; unset → `data/` | pass |
| End-to-end `harness_d.aggregate` on synthetic gens (every other item gold) | 50.00 on each benchmark, dev and test |
| Decoder vs PyPI `metar` 1.11.0 (independent parser), all 9,300 METAR items, 11 fields | 0 mismatches, 0 parse errors |
| Decoder flight category vs AWC `fltCat` on 4,481 live METARs (12 states, 6 h, 2026-11-01) | 4,481 / 4,481 agree |
| Frozen files (`artifacts/*FREEZE*.sha256`, local entries) | unchanged |

## Rebuild from scratch

```
cd mom-lab
bash scripts/partd_c/build_all.sh   # fetch_metar -> build_metar -> build_medcalc -> build_preflight -> make_replay -> check -> pytest
sha256sum -c data_partd_c/SHA256SUMS
```

- Requirements: `.venv` with transformers and the Qwen/Qwen3-1.7B tokenizer (for `n_tokens`), huggingface_hub and
  pandas.
- `fetch_metar.py` makes 50 + 24 sequential requests to IEM (about 25 minutes).
- The optional decoder validation needs a separate venv with `metar==1.11.0`. See the header of
  `scripts/partd_c/validate_metar_decoder.py`.

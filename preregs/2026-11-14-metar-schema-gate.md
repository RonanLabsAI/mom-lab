---
id: mom-metar-schema-gate
title: "METAR gate: JSON-schema constrained decoding for untouched Qwen3-8B and Qwen3-32B vs the 4B METAR adapter"
hardwareClass: rtx6000
valueType: bench
units: gate6000:10
budgetCap: 8
fingerprint: qwen3-8b+qwen3-32b;vllm-0.30.1rc1.dev245+gaf7f9488c;xgrammar-0.2.7;batch-invariant;metar-test-v1;schema-v1;prompts-raw-v1
lane: L2
template: none
preregUrl: pending
productType: standard
priority: 1
---

# Preregistration: METAR constrained-decoding gate (Part G)

Drafted 2026-11-13 ~21:05 PT and revised after a Grok adversarial review (2026-11-13 ~21:30 PT,
`docs/reviews/2026-11-14-grok-gate-prereg.md`), before any gate output exists on the dev or test split.
**Preregistered locally; timestamps not externally verifiable.** The public preregistration repository does not
exist yet (creating it is an approval item), so this card's only timestamp is the local git commit that freezes it.
It is a locked follow-up written after Parts A-F results were known, not a preregistration in the strict sense.
Frozen by `artifacts/GATE_FREEZE.sha256` (commit and manifest hash recorded in `notebook/2026-11-13.md`, the PT date of the freeze).

## Why this gate exists

Part D-C/F: Qwen3-4B with a rank-16 LoRA METAR adapter gets all 14 fields exact on 99.8% of 1,000 test reports. The
best prompted untouched 8B (5-shot + thinking, 3 seeds) gets 83.2 at 1,664 output tokens per item (11.6x the
adapter's 143); Qwen3-32B with thinking (one seed) gets 88.9 at 937 tokens (6.5x). None of the untouched arms was
given constrained decoding against a JSON schema, the obvious alternative to training for a strict-format task. This
gate measures it before the release and before the paper's METAR claims are final.

Disclosed before writing (already known from Part D-C/F, no new model run): untouched 8B 5-shot and 32B zero-shot
outputs already parse as JSON on 100% of items, and their misses are mostly field semantics (8B 5-shot field
accuracy: flight_category 69.4, temperature_c 76.2, ceiling_ft 83.3; 32B zero-shot: flight_category 67.2,
wind_dir_deg 73.2, temperature_c 87.3). The 5-shot and thinking runs already score the same with or without
accepting "HHMM" times (41.7, 83.2, 88.9). Untouched 8B zero-shot writes "0355"; accepting it lifts 2.1 to 18.5.
A format-only schema is therefore expected to help the zero-shot arms most and the strongest arms least. This gate
tests the format alternative, not every alternative to training.

## Systems (untouched Qwen3 checkpoints, evaluation only)

Serving: vLLM `0.30.1rc1.dev245+gaf7f9488c` (image `vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b`,
the vLLM commit inside `mom-lab:v1`), `VLLM_BATCH_INVARIANT=1`, bf16, `--seed 0`, `--gpu-memory-utilization 0.85`,
`--max-model-len 32768` for every arm (5-shot prompts reach 4,586 tokens), structured outputs
`{"backend": "xgrammar", "enable_in_reasoning": false}`, xgrammar 0.2.7, one RTX PRO 6000 Max-Q (Station LEFT, GPU 0).
Requests go only through LiteLLM routes `research-mom-gate-qwen3-8b` / `research-mom-gate-qwen3-32b`
(`docs/gate/litellm-routes.yaml`, route-scoped `drop_params: false`; `run_gate.py check` refuses to start unless both
routes are in the live router config with that setting). Generation: `scripts/gate/schema_generate.py`
(`harness_d.generate` plus `response_format`). Scorer: `harness_d/score/metar.py`, unchanged.

Schema v1 (`configs/gate/metar_schema.json`): the 14 keys (listed in prompt order; JSON Schema does not enforce order
and the scorer ignores it), all required, no extra keys, each field's type, `time_utc` as `HH:MM`, `wind_dir_deg`
integer or "VRB" or null, cloud cover in {FEW, SCT, BKN, OVC, VV}, flight category in {LIFR, IFR, MVFR, VFR}. It does
not encode the decoding rules the prompt states in words (calm wind, fractions, ceiling, altimeter, flight-category
thresholds): a schema that computes those would be a hand-written decoder, which scores 100 by construction because
the gold comes from one (`scripts/partd_c/metar_decode.py`). Checks before any GPU use: all 1,300 dev/test gold
objects validate (`tests/test_gate_schema.py`), and the grammar xgrammar 0.2.7 compiles from the schema accepts all
1,300 (compact, spaced and indented; also in fixed-separator mode) and rejects "0355" and an unknown category
(`scripts/gate/xgrammar_gold_check.py`, run by `check`).
Request field: `response_format = {"type": "json_schema", "json_schema": {"name": "metar", "schema": <v1>, "strict": true}}`.

Prompts: unconstrained arms use the frozen fenced prompts (C zero-shot `data_partd_c`, F 5-shot `data_partd_f`).
Schema arms use a raw-JSON variant built by `scripts/gate/build_schema_prompts.py` (hashes in
`configs/gate/schema_prompts.sha256`) with exactly two edits: "(in a ```json block)" becomes "(raw JSON, no code
fence)" in every user turn, and the five exemplar answers lose their fence. The grammar cannot emit backticks, so
the fenced instruction would fight it.

| run id (step `partdG`) | model | prompt | thinking | decoding | schema | role |
|---|---|---|---|---|---|---|
| `gate-qwen3-8b-zs-free` | 8B | zero-shot, fenced | off | greedy | no | parity control + G4 control |
| `gate-qwen3-8b-fs-free` | 8B | 5-shot, fenced | off | greedy | no | G4 control |
| `gate-qwen3-8b-zs-schema` | 8B | zero-shot, raw | off | greedy | yes | primary |
| `gate-qwen3-8b-fs-schema` | 8B | 5-shot, raw | off | greedy | yes | primary |
| `gate-qwen3-32b-zs-free` | 32B | zero-shot, fenced | off | greedy | no | G4 control |
| `gate-qwen3-32b-fs-free` | 32B | 5-shot, fenced | off | greedy | no | G4 control |
| `gate-qwen3-32b-zs-schema` | 32B | zero-shot, raw | off | greedy | yes | primary |
| `gate-qwen3-32b-fs-schema` | 32B | 5-shot, raw | off | greedy | yes | primary |
| `gate-qwen3-8b-fsthink-schema-s0` | 8B | 5-shot, raw | on | T 0.6, top_p 0.95, top_k 20, seed 0 | yes, after `</think>` | primary |
| `gate-qwen3-32b-think-schema-s0` | 32B | zero-shot, raw | on | same | yes, after `</think>` | primary |

Max new tokens 4,096 (non-thinking) and 16,384 (thinking), as C and F. Thinking servers add
`--reasoning-parser qwen3`; content then holds only the post-`</think>` answer. F's thinking runs had no parser and
the scorer stripped the trace, so an unclosed trace is wrong in both. `n_out_tokens` is `usage.completion_tokens`,
which in this vLLM includes reasoning tokens (checked on the smoke outputs: thinking arms must report hundreds of
tokens or more; if they report under 300 on average, token counts for thinking arms are marked unreliable).
Comparator x = `4B+metar` (Part D-C runs `pt4-lora16-metar-lr2e-4`, `-s1`, `-s2`, per-item mean over 3 training
seeds), not rerun.

## Parity control (stop rule)

`gate-qwen3-8b-zs-free` repeats C's `base-qwen3-8b` METAR run through the new path (other Station, image tag, router).
Expected: 1,000/1,000 identical output strings. Fewer than 1,000: the queue stops before any schema arm; the cause
is investigated and logged as a deviation; parity is rerun after a fix that touches no frozen file. The analysis
marks the headline VOID unless parity is 1,000/1,000. Parity cannot test parameters the zero-shot greedy request does
not send (`response_format`, `top_k`); those are covered by the canary below.

## Canary (the grammar was really applied)

For every schema arm, every non-truncated output must parse, validate under schema v1, contain no ``` fence, and the
arm's "HHMM accepted" score must not exceed its strict score. Any violation means the router or server did not apply
the grammar: the arm is VOID and so is the headline.

## Smoke test (dev split, first 30 items of each schema arm, before the test runs)

`scripts/gate/smoke_check.py`, called by the launcher; no scores are computed. Rules: canary violation, or a nullable
field null in at least 3 of the 30 gold items but never null in an arm's output (a lost null branch) -> stop, no
automatic change. Any schema arm truncating more than 2 of 30 -> every schema server gets
`disable_any_whitespace: true` (grammar-constrained greedy decoding can loop on whitespace), logged as a deviation,
the launcher state hashed in `artifacts/GATE_FREEZE_fix1.sha256`, and smoke rerun. `main` refuses to start until the
smoke has passed. On the test split, truncations count as wrong (harness rule) and are reported per arm.

## Tests (`scripts/gate/gate_analysis.py`, written and hashed with this card)

Statistics as `content/PART_D_PREREG.md`: per-item 0/100, seed-averaged per item where a system has several runs,
paired item bootstrap over the 1,000 test items, 2,000 resamples drawn once from numpy default_rng(0), 95% percentile
CI. PASS = CI low > 0, FAIL = CI high < 0, otherwise INCONCLUSIVE.

- **G1:** x beats the best 8B schema arm (highest test mean of the three 8B schema arms).
- **G2:** x beats the best 32B schema arm (same over the three 32B schema arms).
- **G3 (decides the headline):** d = x minus the best schema arm over all six. Two one-sided tests, margin 2.0:
  "schema ahead" if CI high of d < 0; "equivalent" if CI low > -2.0 and CI high < 2.0; otherwise "not equivalent".
  Picking the best arm after the fact makes G1/G2 harder for x and makes G3 "equivalent" easier; both are stated
  with the result. Ties in mean go to the arm listed first (zero-shot, 5-shot, thinking; 8B before 32B).
- **G4 (descriptive):** each schema arm minus its unconstrained control on the same server; thinking schema arms
  against F's runs, labelled "different serving path, no reasoning parser".
- **Sensitivity:** G1-G3 again with a station-cluster bootstrap (resample stations, then items within a station).
  Reported, not used for the headline.
- **Reported for every arm:** score, per-field accuracy, parse-failure and truncation rates, "HHMM accepted" score,
  mean output tokens and latency, token ratio to x (143.2). For the best schema arm: the share of the gap closed from
  8B zero-shot (2.1), from 8B 5-shot + thinking (83.2) and from 32B thinking (88.9) to x.

## Decision rule for the headline (fixed now)

- **VOID:** parity below 1,000/1,000, any primary arm missing, or any canary failure. No METAR claim changes in either
  direction until the cause is fixed and the gate is rerun.
- **Rewritten:** G3 "schema ahead" or "equivalent". The METAR result is withdrawn as a training win; the paper and
  README say constrained decoding of an untouched model matches (or beats, with the direction stated) the adapter, and
  METAR joins MedCalc as a task where training was not needed.
- **Stands:** G1 PASS, G2 PASS and CI low of d > 2.0. Wording is limited to: "with schema v1 under xgrammar, no
  untouched 8B or 32B arm came within 2 points of the adapter; d = D [lo, hi]". The token comparison is reported
  against the best constrained arm as well as against thinking; while the best constrained arm is more than 2 points
  behind, the thinking comparison stays in the lead and is labelled as such.
- **Narrowed:** anything else. Reported as "d = adapter minus best schema arm = D [lo, hi]" with no direction word.

## Predictions (written before any gate output)

The schema fixes format and little else. 8B zero-shot + schema 12-30; 8B 5-shot + schema 38-48; 32B zero-shot +
schema 35-55; 32B 5-shot + schema 50-75; 8B 5-shot + thinking + schema 80-86; 32B thinking + schema 86-91. Best
schema arm: 32B thinking + schema. G1 PASS, G2 PASS, G3 not equivalent, headline Stands. Non-thinking schema arms
emit 120-160 output tokens (no fence), so for them the token ratio to x is about 1x; the best arm is a thinking arm
at roughly 800-1,000 tokens. Parity 1,000/1,000. Smoke: no rule fires.

## Claim, preemption and budget

Fleet claim `mom-gate-left6000-<YYYYMMDD>` on Station LEFT RTX PRO 6000 (GPU 0) only. `glmf-daily` on the LEFT GB300,
Station RIGHT and Spark 1-3 are not touched. Preemptible: if Stephen or a production route needs the 6000, the run
stops (generation resumes by item id) and the claim is released. Budget cap 8 GPU-hours. Expected: non-thinking arms
~1.5 h including four model loads, 8B thinking ~0.7 h, 32B thinking ~4 h. The container is removed and the claim
released when the run ends.

## What this gate does not show

One schema (format-only, prompt-faithful) and one backend (xgrammar). It does not test a schema or decoder with
METAR domain rules, sampling for non-thinking arms, a schema with the 4B base, or MedCalc (free-text arithmetic,
where a schema has little to constrain). It does not change any Part A-F verdict.

## Deviations log

`content/PART_G_DEVIATIONS.md` (created at the first deviation).

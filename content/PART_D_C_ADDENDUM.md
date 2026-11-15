# Part D addendum C: niche subjects (aviation, medicine) on public data (FINAL at freeze)
Companion to content/PART_D_PREREG.md. Written 2026-11-01 ~08:05 PT, before any model output on this data exists (no
base check was run, on dev or test). Locked by artifacts/PARTD_C_FREEZE.sha256 (hash in notebook/2026-11-01.md).
Shared rules (serving, statistics, bars) are those of PART_D_PREREG.md. Deviations go in content/PART_D_DEVIATIONS.md.

## Why
Exploratory finding: adapter gain grows as the base gets weaker (r = -0.60 over 20 subject x set pairs), and "size buys
knowledge, specialists buy skills". C tests both on subjects Qwen is unlikely to have trained hard on, with public data
so anyone can rerun it.

## Data (data_partd_c/README.md; builders scripts/partd_c/; byte-identical on rebuild)
- METAR decoding (aviation/metar): raw METAR -> 14-field JSON; truth from a deterministic decoder cross-checked against
  the PyPI `metar` parser (0 mismatches on 9,300) and AWC flight category (4,481/4,481). Train 8,000 / test 1,000;
  stations disjoint across splits. Score = all 14 fields right (mean field accuracy reported as extra).
- MedCalc-Bench v1.2 (medcalc/medcalc, CC BY-SA 4.0): patient note -> calculation. Train 9,827 (notes shared with test
  removed) / test 1,100. Official scoring rules (ported; eval() replaced by a safe evaluator).
- Pre-Flight (avknow/preflight, MIT, Airside Labs): aviation knowledge multiple choice, test 240. Knowledge control; no
  training data. Caveat known in advance: always answering "B" scores 43.8%.
- NOTAM: skipped (no licence on either dataset).
Evaluation: scripts/evalq_p.py with harness=harness_d, data root data_partd_c, domains aviation,medcalc,avknow, step
partdC. Every system is scored on all three benchmarks.

## Systems
Untouched: Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-8B-AWQ.
Trained (LoRA r16, Part B recipe, seeds 0-2): pt17-lora16-{metar,medcalc}, pt4-lora16-{metar,medcalc} (12 runs);
pt4-lora16-replay-{metar,medcalc}-r20 (20% replay from the six original domains, 6 runs). 18 runs.
Each adapter is scored with oracle routing on its own task (single-domain test files); the other two benchmarks show what
the adapter does outside its lane (always-on).

## Tests (per task t in {metar, medcalc})
- C1 (primary): 4B + t adapter beats untouched 4B on t.
- C2 (primary): 4B + t adapter beats untouched 8B on t.
- C3: 1.7B + t adapter beats untouched 4B on t; and beats untouched 8B on t.
- C4: 4B replay adapter NI to plain 4B adapter on t; and beats plain on the other two benchmarks (less damage).
- C5 (descriptive, control): on Pre-Flight, the 8B - 4B gap vs the change from adding a METAR or MedCalc adapter. Prediction:
  size helps Pre-Flight; adapters do not (change ~0 or negative).
- C6 (descriptive): add (base score, adapter gain) for METAR and MedCalc to the gain-vs-base-weakness plot.

## Predictions
METAR: untouched 4B maybe 20-60% all-fields-exact (the prompt spells out the schema and rules, so the base may do better
than a true niche task); adapter 85%+. C1 PASS, C2 PASS, C3 PASS. MedCalc: untouched 4B ~20-35%, 8B a few points higher;
adapter +10 or more (the test shares calculator templates with train, so part of the gain is template learning). C1 PASS,
C2 PASS, C3 PASS vs 4B, PASS or INCONCLUSIVE vs 8B. C4: NI PASS; damage reduction PASS for at least one task.
C5: 8B > 4B on Pre-Flight by a few points; adapters change it by under 2 points.

## Deviations log
See content/PART_D_DEVIATIONS.md.

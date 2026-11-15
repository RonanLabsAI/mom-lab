# Part D addendum F: does the C result survive a fairly prompted 8B? (FINAL at freeze)
Companion to content/PART_D_PREREG.md and content/PART_D_C_ADDENDUM.md. Written 2026-11-01 ~23:00 PT, before any
model output for these conditions exists. Locked by artifacts/PARTD_F_FREEZE.sha256 (hash in notebook/2026-11-02.md).
Shared rules (statistics, bars) are those of PART_D_PREREG.md. Deviations go in content/PART_D_DEVIATIONS.md.

## Why
C found a 4B + LoRA adapter far ahead of the untouched 8B on METAR decoding (99.8 vs 2.1) and MedCalc (58.7 vs 39.0).
Every C system was run the same way: zero-shot, thinking mode off, greedy. The obvious objection is that the
untouched models were not given a fair chance. F gives the untouched models the standard help a practitioner would
use instead of training: worked examples in the prompt, Qwen3's thinking mode, and both. If the adapter still wins,
the C headline holds. If prompting closes the gap, Part 3 says so.

Disclosed before writing (exploratory, from the existing C outputs, no new model run): the untouched 8B usually wrote
METAR times as "0355" instead of "03:55". Accepting that format lifts its METAR all-fields score from 2.1 to 18.5
(4B 0.4 -> 4.1); the remaining 8B errors are mostly flight category, temperature/dewpoint and wind direction. The
per-field accuracies in C's output were also seen. No MedCalc or Pre-Flight output was re-examined.

## Conditions (untouched Qwen3 models, no training; evaluation only)
- few-shot (fs): METAR 5-shot (the same 5 worked examples from the METAR train split for every item: one per flight
  category plus one cold report with a weather group; seed 20261102); MedCalc 1-shot with the official MedCalc-Bench
  exemplar for the item's calculator (the benchmark's own "one-shot" setting, in our prompt format). Examples are
  earlier chat turns; the test prompt itself is unchanged. Built by scripts/partd_f/build_prompts.py into
  data_partd_f/ (deterministic; build.json lists the example ids). Pre-Flight has no train split: no few-shot.
  Greedy, thinking off, 4,096 max new tokens (as C).
- thinking (think): the C zero-shot prompts with enable_thinking on, Qwen3's recommended sampling (temperature 0.6,
  top_p 0.95, top_k 20), 16,384 max new tokens, sampling seeds 0, 1, 2. All three benchmarks.
- few-shot + thinking (fsthink): both, seeds 0, 1, 2. METAR and MedCalc.
Systems: 8B (fs, think, fsthink), 4B (fs, think, fsthink), 1.7B (fs). Descriptive extras if time allows: Qwen3-14B and
Qwen3-32B zero-shot (as C) and Qwen3-32B thinking seed 0 (station-b only; they are not on station-a).
Zero-shot bases and adapters are the existing C runs (results/partdC), not rerun.
Serving: scripts/evalq_f.py, a copy of evalq_p.py whose only changes are per-job generation keys (thinking, sampling,
max tokens, split/limit, a Station pin) and --max-model-len 32768 for every F job (few-shot prompts reach 4.6k tokens;
greedy outputs do not depend on this limit). Jobs: configs/partd_f_evaljobs.txt. Same container, batch-invariant
serving and harness_d scorers as C. Run order by priority: 8B seed 0 first, then 4B/1.7B seed 0, then seeds 1-2,
then the 14B/32B extras. If the window ends early, the analysis uses the seeds present (seed 0 required; seeds used
are reported). A thinking-mode answer whose thinking never closes counts as wrong (harness rule).

## Smoke test (before the test runs; dev split only)
configs/partd_f_smokejobs.txt: 8B fs, 8B think, 8B fsthink, 4B think on the first 30 dev items of each benchmark.
scripts/partd_f/smoke_check.py reports truncation, unparseable answers and output length only (no scores used).
Fixed rule: if any thinking job truncates more than 2 of 30 items on any benchmark, every thinking job's max new
tokens goes from 16,384 to 26,000 before the test runs (a deviation entry records it, and the edited jobs file is
hashed in artifacts/PARTD_F_FREEZE_fix1.sha256, as with PARTD_B_FREEZE_fix1). Nothing else may change.

## Tests (scripts/partd_f_analysis.py, written and hashed with this file; per task t in {metar, medcalc})
x = 4B + t adapter (C runs, 3 training seeds). Paired item bootstrap; beats = CI low > 0.
- F1: x beats 8B few-shot on t.
- F2: x beats 8B thinking on t.
- F3: x beats 8B few-shot + thinking on t.
- F4 (headline): x beats the best 8B condition on t (highest test mean among zero-shot, fs, think, fsthink; picking
  the strongest baseline after the fact can only make this test harder for the adapter).
- F5: 1.7B + t adapter beats the best 8B condition on t.
Descriptive: prompting gain (each condition minus zero-shot) for 1.7B / 4B / 8B; x vs the best 4B condition; x vs
14B / 32B / 32B thinking; share of the 8B-zero-shot-to-adapter gap closed by the best 8B condition; Pre-Flight
thinking minus zero-shot; MedCalc F1-F5 without the 5 test items whose note contains an exemplar note; METAR score
with "HHMM" accepted; output tokens and latency per item (cost); truncation rate per run.

## Predictions (written before any F output)
METAR: few-shot fixes the time format and most of the schema slips; 8B fs 30-60% all-fields, 8B think 15-45%,
8B fsthink 40-75%. Best 8B well below the adapter's 99.8. F1-F5 PASS on METAR.
MedCalc: the official one-shot helps a little (8B fs 40-48). Thinking helps more (8B think 45-60, fsthink 48-62),
because MedCalc is arithmetic over a note and thinking models are good at it. F1 PASS. F2, F3, F4 on MedCalc:
INCONCLUSIVE or FAIL is as likely as PASS; my single guess is F4 INCONCLUSIVE. F5 on MedCalc FAIL.
4B: prompting helps it less than the 8B. Pre-Flight: thinking changes it by under 3 points. Cost: thinking uses
roughly 5-20x the output tokens of the adapter per item.
If F4 fails on MedCalc, the Part 3 claim narrows to: "an adapter matches a thinking 8B at a fraction of the tokens",
and the METAR result is reported with the prompted baselines next to it.

## Deviations log
See content/PART_D_DEVIATIONS.md.

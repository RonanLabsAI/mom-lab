# Part C preregistration: fresh paraphrase set (FINAL, frozen; freeze time and hash in notebook/2026-11-01.md)

## Why, and what was known when this was written
Written 2026-10-31 ~23:00 PT, AFTER the Part B test results (notebook/2026-10-31.md, results/analysis/partb_test_*.json).
In Part B every primary hypothesis was INCONCLUSIVE except H2-4bit (FAIL). Most open components are on the paraphrase
set, which has only 590 items (95% CI half-width ~2.3-2.9 points). Part C asks one question: do the Part B paraphrase
components hold on a larger, fresh paraphrase set that no model has seen? Part B's verdicts stand as reported; Part C
is a follow-up chosen with the Part B test results in view, and is reported as such.

Dropped before running: a 4B replay-LoRA library as the main new arm. Part A dev already showed replay adds nothing
measurable to 4B LoRA (para 65.68 vs 65.34), so it is kept only as an exploratory secondary arm (seed 0 only).

## Data (built before any model output on it exists)
- data_partc/eval/para/<bench>/test.jsonl, built by scripts/build_para2.py with the same paraphraser
  (Qwen3.8-27B-NVFP4, greedy, served on the left RTX PRO 6000), styles, protected literals and checks as the Part B
  set (scripts/build_para.py, imported unchanged).
- Source items: TEST-split items of each source benchmark not drawn for the Part B paraphrase set (draw seed
  20261029+2; BFCL stratified by category). Target: gsm8k 300, bird 300, mbpp all 89 remaining, bfcl 300,
  mmlu-pro 300, flores 300 (~1,589 before drops). Items failing the checks after 5 attempts are dropped. Extra check:
  every number in a gsm8k question must appear as digits in the rewrite.
- Validation before the freeze: scripts/validate_para.py logic (literals, kept values, banned phrases, reference and
  meta copied) reports 0 problems; final counts recorded in the notebook.

## Built set (2026-10-31 22:52 PT, before any model output on it)
1,562 items, validate_para2.py 0 problems: gsm8k 294 (6 dropped: digit check), bird 300, mbpp 88 (1), bfcl 286 (14:
placeholder or kept-value checks), mmlu-pro 294 (6: placeholder), flores 300. No source item overlaps the Part B paraphrase
set (asserted in the builder). Paraphraser server stopped after the build.

## Models (no training; all weights already hashed in artifacts/FREEZE_MANIFEST.sha256)
The 50 Part B test-pass models, unchanged, plus the secondary arm pt4-lora16-replay-{sql,tools}-r20-lr2e-4 (Part A,
merged LoRA r16 + 20% replay, seed 0; restored from the TR archive, sha256 checked against the archive copy).
Served exactly as in the Part B test pass (mom-lab:v1 image, vLLM batch-invariant, greedy, 4,096 max new tokens),
RTX PRO 6000 on both Stations; results in results/partc/<run>/.

## Analysis (scripts/partc_analysis.py, reusing scripts/partb_analysis.py unchanged)
Frozen router artifacts/router_e5lr.pkl, same arms and routes (1.7B: code, sql, tools; 4B: sql, tools), seed
averaging, composite (mean over domains), paired item bootstrap (2,000 resamples within benchmark, numpy seed 0),
95% percentile CI, beats = CI low > 0, non-inferior = CI low > -2.0, FAIL if CI high < bar.

Primary: every paraphrase component of every Part B hypothesis, re-tested on the fresh set:
- H1-1.7B: lora > base; lora >= pooled - 2.   H1-4B: same.
- H2: 4B lora >= 8B - 2.   H2-4bit: AWQ-4B+lora >= AWQ-8B - 2.
- H3-1.7B, H3-4B: lora >= distilled - 2.
- H1-replay (secondary in Part B): 1.7B replay > base; replay >= pooled - 2.
Combined verdict per hypothesis: non-paraphrase components keep their Part B test verdicts; paraphrase components take
the fresh-set verdicts. PASS only if all pass; FAIL if any fails; else INCONCLUSIVE. Consequence, known in advance:
H2, H3-1.7B and H3-4B cannot reach PASS (each has an open core component in Part B); H2-4bit stays FAIL.
Only H1-1.7B, H1-4B and H1-replay can change.

Secondary (exploratory, single seed): 4B lora-replay >= pooled - 2; 4B lora-replay vs lora (beats).
Also reported: the same analysis with oracle routing; per-benchmark scores; router accuracy on the fresh set.

## Predictions (for the record, not bars; from the Part B test paraphrase differences, CIs expected ~40% narrower)
- H1-4B para vs pooled: -0.7 -> likely PASS (CI low around -2.0 to -1.6). Para vs base +2.1 -> PASS. => H1-4B likely PASS.
- H1-1.7B para vs base +2.2 -> likely PASS; vs pooled -1.2 -> likely INCONCLUSIVE (CI low ~ -2.7). => likely INCONCLUSIVE.
- H1-replay para vs pooled +0.05 -> likely PASS; vs base +3.4 -> PASS. => likely PASS.
- H2 para -3.6 -> likely FAIL or INCONCLUSIVE. H2-4bit para -5.3 -> FAIL.
- H3-1.7B para -2.2 -> INCONCLUSIVE or FAIL. H3-4B para +1.1 -> PASS (hypothesis stays INCONCLUSIVE on core).
- 4B lora-replay: about equal to lora (Part A dev).

## Deviations log (after the freeze)
(empty)

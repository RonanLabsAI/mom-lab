# Part D preregistration: follow-ups A-E (core sections FINAL at freeze; B and C added as locked addenda)

## Why, and what was known when this was written
Written 2026-11-01 ~08:10 PT. Written AFTER all Part B test and Part C results, and after an exploratory read of them
(workspace/research/mixture-of-models/article2/data_findings.md, results/analysis/explore_tables.json). Stephen
approved the five follow-ups ("Go ahead and begin A-E"). These are follow-up experiments chosen with every earlier
result in view, and are reported as such. Part B and Part C verdicts are unchanged by anything here.

Freeze rule: this file's sections A, D, E plus artifacts/PARTD_FREEZE.sha256 are written, and the manifest's sha256
is recorded in notebook/2026-11-01.md, BEFORE any Part D model output exists. Sections B and C are appended later as
addenda. Each addendum is locked the same way (own manifest + hash in the notebook) before any model output on its
data exists. Locked text is never edited; changes go in the deviations log.

## Shared rules (all experiments)
- Models served exactly as in Part B: mom-lab:v1 image, vLLM batch-invariant mode, greedy, non-thinking, 4,096 max new tokens.
  RTX PRO 6000s for every scored generation. GB300s for training.
- Training recipe = the Part B 4B LoRA recipe unless stated: post-trained Qwen3-4B, LoRA on all linear layers,
  alpha = 2 x rank, lr 2e-4, 2 epochs, micro-tokens 32,768, grad checkpointing, seeds 0, 1, 2.
- Library = untouched base + adapters routed by the frozen router artifacts/router_e5lr.pkl (4B routes: sql, tools;
  everything else to the base). Oracle routing reported as secondary.
- Statistics as Part B: seed-averaged per-item scores, composite = mean over domains, paired item bootstrap (2,000
  resamples within benchmark, numpy seed 0), 95% percentile CI. Beats = CI low > 0; non-inferior (NI) = CI low > -2.0;
  FAIL = CI high < bar; else INCONCLUSIVE. Each test is reported on its own (no conjunction across experiments).
- Sets: core / para (590) / mixed = the Part B test sets (data/); para2 = the Part C fresh set (data_partc/). These
  have all been scored before for other systems, so they are not new to us. They are new to the new systems only.
  The new held-out set is B's.
- Analysis: scripts/partd_analysis.py (written and added to a manifest before its first run; reuses partb_analysis.py
  unchanged).
- Memory: weights + adapters + router + bf16 KV at batch 1 x 8,192, as scripts/memory_report.py.

## A. FP8 4B + adapters vs 4-bit 8B (eval only)
System fp8-lib: Qwen3-4B-FP8 (served bf16 activations) + the existing Part B 4B r16 adapters (sql, tools; seeds
0-2, trained on bf16, NOT retrained), routed. Runs fp84-lora16-{sql,tools}-lr2e-4[-s1,-s2] (6 evals).
Memory ~7.13 GB (6.40 + 0.26 + 0.47) vs AWQ-8B 7.31 GB.
- A1 (primary): fp8-lib NI to AWQ-8B on core, para, para2, mixed (4 tests).
- A2: fp8-lib NI to the bf16 4B library on core, para, para2, mixed.
- A3 (secondary): fp8-lib vs untouched FP8-4B (beats) on core and para2.
Predictions: FP8 base ~= bf16 base (Part B: 65.4 vs 65.1 core), so fp8-lib ~= bf16 library. A1: core PASS
(~66.4 vs 66.7), mixed PASS (~69 vs 66.2), para INCONCLUSIVE (~63 vs 66.1), para2 PASS or INCONCLUSIVE (~63.5 vs
64.1). A2: PASS on all four.

## D. Replay with the model's own answers (self-distilled replay)
Data: for each of sql, tools, the same replay prompts as data/mixtures/N14M/replay/<d>-r20.jsonl (the 20% slice drawn
from the other five domains). Each replay prompt's answer is replaced with untouched Qwen3-4B's own greedy,
non-thinking answer (max 2,048 tokens; truncated answers dropped). Own-domain examples are unchanged. Generation is
not batch-invariant (training data, not scoring). Builder scripts/make_sdreplay.py, output
data/mixtures/N14M/sdreplay/<d>-r20.jsonl, counts in the notebook.
Systems: pt4-lora16-sdreplay-{sql,tools}-r20-lr2e-4[-s1,-s2] (6 training runs).
- D1 (primary): always-on sd-replay adapter beats the always-on plain r16 adapter, composite, on core and para2,
  for each of sql and tools (4 tests). This is the "does it stop the damage" test.
- D2: own-domain score of sd-replay NI to plain r16 (bird_dev core, bfcl core) (2 tests).
- D3: routed sd-replay library vs routed plain r16 library, composite, core and para2 (beats).
Predictions: D1 PASS for tools (plain tools adapter always-on wrecks code, 61 -> 11 on para2) and likely PASS for sql.
D2: PASS or INCONCLUSIVE (expect -0 to -2 own-domain). D3: INCONCLUSIVE (routing already hides most damage).

## E. Bigger adapters
Systems: pt4-lora{64,128}-{sql,tools}-lr2e-4[-s1,-s2] (12 training runs), same data and recipe as r16 except rank.
- E1 (primary): r64 beats r16 on own domain: SQL (bird_dev core; bird_pp para2) and tools (bfcl core; bfcl_ml mixed)
  (4 tests). Same for r128 (4 tests).
- E2: routed library with r64 / r128 vs routed r16 library, composite, core and para2 (beats).
Predictions: SQL +1 to +3 at r64 (full FT got +2 to +6 over r16 in Part A/B); r128 ~= r64; tools ~0 (base is at
91-94). E2 INCONCLUSIVE (one-domain gains are diluted 1/6 in the composite).

## Order of work (not a test; for the record)
GB300s: E training (12 runs) and, after the D data is built, D training (6 runs). 6000s: A evals, then D answer
generation, then E and D evals, then B and C when their addenda are locked. para2 is scored after core/para/mixed.

## Addendum B (fresh SQL and tool-calling test) — to be appended and locked before any model output on it
## Addendum C (niche subjects: aviation, medicine) — to be appended and locked before any model output on it

## Deviations log (after the freeze)
(empty)

# Part B preregistration — FINAL (frozen at the time recorded in notebook/2026-10-31.md)

Status: drafted 2026-10-30 09:55 PT; reviewed with Stephen 2026-10-31 ~05:50 PT; dev predictions filled 08:10 PT.
Frozen = this file + `artifacts/FREEZE_MANIFEST.sha256` written and the manifest's sha256 recorded in the notebook
BEFORE any test-split generation. Nothing in this file may be changed after the freeze.

## Question
Does an always-on post-trained base plus small switched-in domain adapters beat (a) the base, (b) retraining one
model on all the data (pooled), and (c) distilling the adapter library into one model, at two sizes, and does a
4B library match an 8B dense model at about half the serving memory?

## Fixed design (decided on dev in Part A; see PART_A_SUMMARY.md and notebook 2026-10-29/30)
- Inits: post-trained Qwen3-1.7B and Qwen3-4B (untouched base always served).
- Library (primary) = base + LoRA r16 adapters (lr 2e-4, 2 epochs, plain; Part A: replay adds nothing to LoRA).
  Routes, chosen on dev by cross-validated routing (scripts/cv_routes.py, 200 half-splits):
  1.7B -> code, sql, tools (routed share 0.52 / 1.00 / 0.99); 4B -> sql, tools (1.00 / 1.00; code 0.25 -> base).
- Secondary library (1.7B only) = base + full-FT specialists with 20% general replay (lr 2e-5, 2 epochs).
- Pooled = same init, all six domains (data/mixtures/N14M/pooled.jsonl), full FT lr 2e-5, 2 epochs (1.7B: 2e-5 beat
  1e-5 on dev, 55.36 vs 53.57). b4-pooled (Base init) = secondary only.
- Distilled = same init, trained like pooled (lr 2e-5, 2 epochs) on the library's own answers to the pooled prompts
  (teacher = seed-0 library routed by the prompt's training domain; other domains answered by the base; greedy,
  non-thinking, max 4,096 new tokens; truncated / over-length answers dropped). scripts/make_distill.py.
  Compute is matched by epochs (2), not by tokens. Teacher answers are longer than the pooled references, so the
  distilled arms train on ~2% fewer examples but more tokens and steps (4B: 102.2M vs 83.5M tokens, 56.9M vs 34.8M
  supervised, 2,379 vs 1,992 steps). Stephen approved this matching rule 2026-10-30 16:25 PT.
- Seeds: 3 per trained arm (s0, s1, s2). A library's seed k uses the seed-k adapters.
- Router on test: frozen multilingual-e5-small (rev 614241f6) + logistic regression trained only on training-mixture
  prompts (artifacts/router_e5lr.pkl, sha256 c3a1949e…). Oracle routing = secondary analysis.
- Eval protocol v2 (GOAL.md): vLLM batch-invariant, greedy, non-thinking, 4,096 max new tokens, docker-batch code
  sandbox; all test generations on the two RTX PRO 6000s (same chip type). AWQ checkpoints served in fp16 (AWQ kernels
  reject bf16); everything else bf16.

## Prompt sets (test split, never read before the freeze)
- core: the six domain benchmarks (gsm8k, math500, humaneval+, mbpp+, livecodebench, bird, bfcl, flores200,
  mmlu-pro, ifeval). Composite = mean over domains of the mean over that domain's benchmarks.
  Test items: gsm8k 660, math500 250, humaneval+ 82, mbpp+ 189, livecodebench 66, bird 767, bfcl 500,
  flores200 3,036 sentences, mmlu-pro 6,016 (dev 1,499; every item is scored), ifeval 271.
- para: 590 paraphrased test items (gsm8k 99, bird 100, mbpp 100, bfcl 96, mmlu-pro 95, flores 100), same paraphraser and checks as dev (validate_para.py: 0 problems).
- mixed: 1,350 multilingual (es/de/zh) mixed-skill test items (gsm8k 450, bird 300, mbpp 300, bfcl 300), validate_mixed + gold check 99-100.

## Arms (all on test)
1.7B: base; pooled x3; LoRA library x3; replay library x3; distilled x3.
4B: base; pooled x3; LoRA library x3; distilled x3; b4-pooled (Base init, secondary).
8B: base. Quantised (eval only): Qwen3-4B AWQ, Qwen3-8B AWQ, Qwen3-4B FP8, Qwen3-8B FP8; AWQ-4B + the 4B adapters (sql, tools)
x3 seeds (seed k = the seed-k adapters, as for the bf16 library).

## Statistics
Per system and set: per-item score averaged over seeds, composite as above. Differences: paired item bootstrap,
2,000 resamples within benchmark (numpy seed 0), 95% percentile CI. Bars: "beats" = CI low > 0; "non-inferior" =
CI low > -2.0 points. FAIL = CI high below the bar; otherwise INCONCLUSIVE. A hypothesis passes only if every
component passes. No multiplicity correction (conjunctions are conservative); every component is reported.
Implementation: scripts/partb_analysis.py (frozen), output results/analysis/partb_test_embed-lr.json.

## Hypotheses (at each size where listed)
H1 and H3 each get a separate verdict per size (H1-1.7B, H1-4B, H3-1.7B, H3-4B); a size's verdict uses only that
size's components.
H1 (library vs base and pooled), 1.7B and 4B:
  - beats base on core, para, mixed;
  - beats pooled on core;
  - non-inferior to pooled on para and mixed.
H2 (memory), 4B only:
  - 4B LoRA library non-inferior to Qwen3-8B on core, para, mixed, at a serving-memory ratio of 0.568
    (weights + adapters + router + bf16 KV at batch 1 x 8,192; scripts/memory_report.py; fixed, not tested);
  - H2-4bit: AWQ-4B + adapters (3 seeds) non-inferior to AWQ-8B on core, para, mixed (memory ratio 0.631; 0.60 with an fp16
    router).
H3 (library vs distilled), 1.7B and 4B: beats distilled on core; non-inferior on para and mixed.
Secondary (reported, not confirmatory): 1.7B replay library on the H1 components; oracle routing; b4-pooled;
quantised single models; seed SDs; router accuracy per benchmark.

## Predictions (final dev, 2026-10-31 08:10 PT, embed-lr router, 3 seeds everywhere; for the record, not bars)
Source: results/analysis/partb_dev_embed-lr.json (oracle routing within ~0.2 points everywhere). Dev = the split the
design was chosen on, so these are optimistic for the library.
- H1-1.7B: INCONCLUSIVE. Passes base x3, pooled core (+3.03 [+2.07, +4.01]) and mixed NI (-0.06 [-1.63, +1.55]);
  para vs pooled -0.81 [-3.42, +1.79] is the open component.
- H1-4B: INCONCLUSIVE. Passes base core/mixed, pooled core (+4.71 [+3.72, +5.75]) and para NI; base para
  +1.50 [-0.45, +3.23] and mixed vs pooled -0.44 [-2.08, +1.24] open.
- H2 (4B lora vs 8B): INCONCLUSIVE. core -1.10 [-2.23, -0.00], para -2.13 [-5.23, +0.83], mixed +0.64 PASS.
- H2-4bit: INCONCLUSIVE. core -1.60 [-2.75, -0.48], para -1.69 [-4.78, +1.34], mixed +0.47 PASS.
- H3-1.7B (lora vs distill): INCONCLUSIVE. core +0.69 [-0.18, +1.54], para -1.34 [-3.80, +1.08], mixed PASS.
- H3-4B: INCONCLUSIVE. core +0.82 [-0.01, +1.65]; para and mixed PASS.
- Secondary: 1.7B replay library passes every H1 component on dev (core +3.88 vs pooled, para +0.97, mixed +2.19).
- Context: post-trained 4B pooled is far weaker on core than the library (62.49 vs 67.21) but strong on mixed
  (67.74); Base-init 4B pooled is strongest on mixed (70.78, secondary arm).

## Changes before the freeze (Stephen's review, 2026-10-31 ~05:50 PT; made with dev results in view)
- Non-inferiority margin kept at -2.0 (not changed after seeing dev).
- H1/H3 verdicts per size instead of one pooled-over-sizes verdict (matches the "at each size" wording;
  partb_analysis.py had combined them).
- H2-4bit library uses all 3 adapter seeds (was seed-0 only), matching the bf16 library.
- No hypotheses added or removed.

## Deviations log (after the freeze)
(empty)

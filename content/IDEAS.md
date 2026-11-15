# Content ideas (DRAFT ONLY — nothing is posted until Stephen decides; kept private, no patent consult (Stephen 2026-10-30))

Each item: the claim, the evidence file, the figure to make.

1. **"Greedy decoding isn't deterministic."** Same model, same GPU, same server, same prompts, temperature 0: only 32% of MMLU-Pro answers and 27% of MATH-500 answers were identical on a rerun; MATH-500 moved 71.6 → 73.2. Evidence: notebook 2026-10-29 protocol table; results/protocol/p-greedy-4k vs p-greedy-4k-rerun. Figure: per-benchmark "% identical outputs" bars. Hook for anyone comparing models on a 1-2 point gap.
2. **"Your eval budget is lying to you."** A 1,024/512-token cap truncated 21-29% of MATH-500 and MMLU-Pro answers; raising it to 4K lifted Qwen3-1.7B MMLU-Pro 37.9 → 43.8. Evidence: step1 vs protocol runs. Figure: truncation % by benchmark at two budgets.
3. **"Same model, four chips — and a switch that makes them agree exactly."** Default vLLM: composite within ~0.5 but single benchmarks up to 3.4 apart, and even reruns on one chip differ. With batch-invariant mode: 100% identical outputs across DGX Spark and GB300 (1,179/1,179). Earlier text: Qwen3-1.7B on RTX PRO 6000, GB300, DGX Spark, RTX 5080 (pending): composite within ~0.1, single benchmarks up to ~2 points apart. Figure: per-benchmark dot plot by chip.
4. **"We reproduced Qwen3's numbers."** MATH-500 and IFEval within noise at 0.6B-8B. Table. Builds credibility for later claims.
5. (Running) "Does splitting a model into specialists beat training one model on everything?" — the Step 3 headline.

6. **"Don't fine-tune your good model on everything. Keep it, and add specialists."** Qwen3-1.7B: pooled fine-tune on 6 domains 53.6; six specialists with perfect routing 54.6; untouched base + only the specialists that beat it (SQL, tools) 58.9 (+3.2 over base, +5.4 over pooled, cross-validated). Figure: the 6x6 transfer heatmap + the 4 systems with CIs. Caveat until seeds land.
7. **"A SQL specialist that forgot how to translate."** lr 2e-5 SQL specialist: SQL 44.9, translation 0.4 (answers everything in SQL). Vivid forgetting example; screenshot of an output.

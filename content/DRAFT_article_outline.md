# DRAFT article / arXiv outline — DO NOT PUBLISH
Working title: "Keep the Base: Adding Skills to a Deployed LLM with Replay-Trained Adapters and Simple Routing"
Status: outline. Dev-split numbers only; the paper reports Part B test-split numbers. Hold for disclosure decision.

## Abstract (to write after Part B)
One-paragraph claim: always-on base + replay-trained specialists/adapters, routed simply, beats pooled
retraining at equal memory, never degrades the base, and reaches about one model size up at half the memory.
Honest limits: no gain where the base is already strong; does not match much larger models.

## 1. Introduction
- Problem: adding skills to a deployed model without breaking it. Retraining is costly and causes forgetting.
- Question as originally posed (library of small models vs big model per GB) and why the answer narrowed.

## 2. Related work
Mixture-of-experts; Branch-Train-Merge / Branch-Train-MiX; LLM routers (RouteLLM etc.); multi-LoRA serving
(S-LoRA, Punica) and LoRA routing (LoRAHub, MoLE, X-LoRA); model merging (task arithmetic, TIES); continual
learning / replay; BAR. Position: careful equal-budget comparisons, robustness sets, memory accounting.

## 3. Setup
Models, domains, data (14M tokens per domain), training recipe, deterministic evaluation (batch-invariant vLLM,
cross-chip identical outputs), three prompt sets (core / paraphrase / mixed-skill), paired item bootstrap.
Template confound and how Part B handles it.

## 4. Results
4.1 Specialists vs pooled vs base (Step 2-3), forgetting, Base vs post-trained init, crossover and its reversal
    under paraphrase.
4.2 Merging fails (Step 3).
4.3 Routing (Step 4): rules vs embeddings vs random; paraphrase robustness.
4.4 Replay (Step 6): forgetting and brittleness removed; seeds.
4.5 Memory (Step 5): curve at 1.7B; LoRA; 4B vs 8B "one size up".
4.6 Scale: 4B results.

## 5. Negative results and limits
Small libraries vs bigger dense models; quantisation interaction; Base-init LoRA stop-token failure; strong
domains; template familiarity; distillation baseline (Part B).

## 6. Engineering notes (appendix)
Nondeterministic batched greedy decoding and batch-invariant mode; IFEval scorer randomness bug (report
upstream); sandbox load bug; reproducibility details.

## Figures to make
F1 score-vs-memory (1.7B base, pooled, merges, base+k specialists, base+LoRA, 4B, 4B+LoRA, 8B).
F2 forgetting matrix (transfer heatmap) plain vs replay. F3 paraphrase drop per system. F4 router accuracy vs gain kept.

# Mixture-of-Models: Part A summary (dev split only)

Status 2026-10-30 00:40 PT (Part A complete). DRAFT, internal. Not for publication (kept private by Stephen's decision 2026-10-30; no patent consult).
Full record: `notebook/2026-10-29.md`, `notebook/2026-10-30.md`, `GOAL.md`. All numbers are on the dev split;
the test split is untouched for Part B.

## The question
Can a library of small specialist models with a router beat simpler options at equal or lower memory?
Compared against the simplest alternative at every step: one model trained on all the same data ("pooled").

## Setup in one paragraph
Qwen3-1.7B (and 4B, 8B as reference). Six domains: math, code, SQL, tool calling, translation, general.
Specialists = fine-tunes on one domain's public data (14M tokens each); pooled = all six (84M). Scored on
three prompt sets: the standard benchmarks (core), a paraphrase set (templates removed, prompts reworded,
595 items) and a mixed-skill set (math/SQL/code/tools questions in Spanish, German, Chinese; 1,350 items).
Deterministic vLLM (batch-invariant, identical outputs across chips). Paired item bootstrap for every comparison.

## What we found
1. **Keep the base, add specialists.** The winning design is the untouched post-trained model always on, with
   specialists switched in only for domains where they beat it (SQL, tools; sometimes code). Specialists that
   replace the base everywhere, pooled training and merged specialists all lose skills the base already has.
2. **Replay makes it robust.** Specialists trained on their domain plus 20% data from the other domains keep
   nearly all their gain (-1 to -3 points own-domain) and stop forgetting and stop breaking on reworded prompts.
   With replay, base + specialists beats the base on all three prompt sets (+3 core, +7 paraphrase, +10 mixed),
   beats pooled on core (+3.4 to +5.1) and ties pooled on paraphrase and mixed. Replicated on two seeds.
3. **Routing is not the hard part.** A no-training rule router and a small embedding router keep essentially
   all of the perfect-routing gain, even on reworded prompts (router accuracy 76-80%, cost <0.4 points).
4. **Adapters make it cheap.** LoRA rank-16 adapters (about 1% of a model each) keep 58-73% of the full
   fine-tuning gain. At about one model's memory, base + three adapters beats pooled on core and ties it on
   the other sets.
5. **It holds at 4B, and it buys about one model size.** Qwen3-4B + two LoRA adapters (~2.37 units of memory)
   statistically ties untouched Qwen3-8B (~4.8 units) on all three prompt sets. Specialists still add value on a
   stronger base (4B SQL 41.9 -> 52.5).

## What did not work / limits (must be stated)
- **No small library matches a much bigger model.** Plain Qwen3-4B beats every 1.7B library by 7-10 points.
  The original Part B headline (reach 90% of Qwen3-235B in less memory than any single model) is very unlikely
  at these sizes; quantised single models make it harder still.
- **Specialists only help where the base is weak.** Our public math, general, translation and code data did not
  beat Qwen's own post-training; the gains come from SQL and tool calling.
- **Template confound.** The SQL, tools, math and translation training data uses the same prompt templates as the
  evaluation. Part of the SQL/tools gain over the base is format familiarity (pooled shares it, so specialist vs
  pooled is fair). The paraphrase set is the clean test and the main results hold on it.
- **Mixed-skill questions favour pooled at 4B** (pooled 70.8 vs adapter library 67.6-68.0, significant). Replay
  fixed this at 1.7B but not at 4B (adapters with replay: +0.4, ns): it is not forgetting but cross-skill
  multilingual transfer that joint training provides. Candidate fix: multilingual specialist data or
  two-expert routing.
- **LoRA from a Base checkpoint cannot learn to stop** (the end-of-turn token is untrained). Use post-trained
  starting points for adapters.
- **Not yet built:** a distilled single-model baseline (required by Part B), self-distillation, KL-to-init.

## Recommendation for Part B
Proceed, reframed. Claim: "adding skills to a deployed model with an always-on base plus replay-trained
adapters beats retraining it, never makes it worse, and buys roughly one model size per memory doubling".
Before freezing: build the distilled baseline, make the paraphrase set (or template-free training data) the
primary score, set pass bars from these numbers, then run once on test.

## Operational record
Several bugs found and fixed during the run (all in the notebook): non-deterministic batched greedy decoding,
an IFEval scorer randomness bug, a per-program sandbox timeout that scored everything 0, Base-model stop tokens,
queue/process-group and orphan-server deadlocks, and an environment pin I broke and restored. None affected
reported numbers after their fixes; affected runs were re-scored.

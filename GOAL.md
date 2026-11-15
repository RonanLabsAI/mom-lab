# GOAL: Mixture-of-Models study, Part A (then Part B if A is strong) — PART A COMPLETE 2026-10-30 00:40 PT; see content/PART_A_SUMMARY.md

Set by Stephen 2026-10-29 (PT). Authority: claim + run on ALL GPUs (2 Stations GB300 + RTX PRO 6000, 3 Sparks, 5080, Threadripper); fix bugs; consult Muse/Codex/Grok; add experiments when data warrants; extra seeds for significance; be thorough over fast. Posts are DRAFT ONLY until Stephen decides (disclosure/patent).

- Plan: 2026-10-29_MoM_PoC_plan.md (v4), in the private research workspace (not in this repository)
- PI-visible mirror on the RTX 5080 desktop (lab/ = this repo's results, notebook, content)
- GPUs (8): 2x GB300, 2x RTX PRO 6000, 3x Spark GB10, 1x RTX 5080 (desktop under WSL, ~9 GB free; light jobs only, it drives the screen)
- Fleet claim: ALL RELEASED 2026-10-30 00:45 PT (Sparks 10-29 23:05, Stations 10-30 00:45); all services restored. Re-claim before Part B.
- Services DOWN while claim open (Stations only since 10-29 23:05 PT; Sparks released and restored): station-a glmf-daily + nemotron-super-rtx; station-b dsv41-daily + sidecar.

## Ladder status
| Step | Question | Status |
|---|---|---|
| 0 | Hardware claimed, services stopped, restore script | DONE 2026-10-29 |
| 1 | Test bench works (scores ≈ published) | DONE 2026-10-29: reproduction within noise; protocol v2 frozen |
| 2 | Specialising helps | DONE: post-trained init only where base is weak — SQL +16, tools +4 over base; math/code/general/translate specialists do not beat base. Base init: specialists beat pooled on own domain for code, sql, tools. |
| 3 | Room for a mixture: oracle routing vs pooled, tagged-pooled, random partitions, merged specialists; transfer matrix; 3 seeds; post-trained + Base inits | NEARLY DONE (10-29 16:20 PT): CROSSOVER - from Base the oracle library beats pooled (+1.3 to +1.7, sig, all LRs); from post-trained it does not. base+specialists +6 vs pooled both inits. Seed sd 0.3-0.8. Tagged-pooled +0.4 (ns); random partitions 2.0 below domain library. Merging: all merges below pooled (48.8-51.5); SQL gain lost. COMPLETE. |
| 4 | Routing works: format rules, embeddings+LR, random routing; paraphrase + mixed sets | DONE (10-30 01:05 UTC): routers keep the oracle gain on templates (within 0.15) and on paraphrases (within 0.35, router acc 76-80%). Paraphrase reveals Base-init specialists are brittle (b+s loses to pooled -5.7 sig at 2e-5); post-trained-init b+s holds (+5-6 vs base, ties pooled). Mixed-skill: b+s ties pooled. |
| 5 | Score vs memory curves | DONE (10-30): 1.7B: base+SQL (2 copies) gets 60-70% of the gain, 3-4 copies saturate; base + 3 LoRA r16 adapters (~1.03 copies) beats pooled on core (+3.0 sig), ties on para/mixed. Plain Qwen3-4B (2.35 copies) beats every 1.7B library by 7-10. 4B + 2 LoRA adapters (~2.37 copies) statistically ties Qwen3-8B (~4.8 copies) on core/para/mixed. |
| 6 | Improvements: replay, LoRA library, sizes, seeds | DONE (10-30): 20% replay removes forgetting and phrasing brittleness (Base-init code para 33->61) for <2-3 pts own-domain; b+s replay beats base everywhere and pooled on core, ties pooled on para/mixed; replicates across 2 seeds (6/6). LoRA from post-trained init keeps 58-73% of full-FT gain at ~1% memory; Base-init LoRA cannot emit <|im_end|> (dropped). Specialists add value at 4B (SQL 41.9->52.5). Merging loses (Step 3). Not run: KL-to-init, self-distillation (script ready), distilled single model, more domains. 4B LoRA+replay: no change (mixed gap vs 4B pooled persists, -2.8 sig). |

## Rules
- Compare every step to the simplest alternative (pooled model). Dev split for all decisions; test split untouched until Part B.
- Same hardware/software stack for scores within one comparison.
- Log every run: results/<step>/<run>.json with model, revision, settings, scores, hardware, git hash.
- notebook/ = dated lab notebook (human-readable). content/ = X/article/arXiv drafts + figures.
- Subagents: foreground commands < 9 min; long jobs are run by the controller.
- Scoring runs use HARNESS_SANDBOX=docker-batch: the whole code batch runs once in a --network none container (mom-sandbox:v1, numpy included); per-program docker containers timed out under 48-way load and scored everything 0 (bug found 2026-10-29). BIRD uses set comparison (official).
- Blocker rule (Stephen, /loop 2026-10-29): if a true blocker stops all progress, release the hardware claim (restore services) and wait for Stephen.

## Eval protocol v2 (frozen 2026-10-29 after the protocol experiment)
- Serving: vLLM with VLLM_BATCH_INVARIANT=1 (bit-exact repeatable outputs on the same chip; verified 1,179/1,179 identical on rerun, no speed loss), max-model-len 8192.
- Decoding: greedy, non-thinking, --max-tokens-override 4096 for every benchmark (all prompts < ~2.2K tokens).
- Scoring: docker-batch code sandbox; IFEval with Python `random` seeded per item (the official checkers draw missing args at random; scores drifted with call history before the fix).
- Hardware: all scores within one comparison from the same chip type. Plan: Sparks (3x GB10) = eval farm; Stations = training.
- Residual: greedy loops still truncate ~4-5% of math/IFEval and ~15% of LiveCodeBench at 4K; scored as wrong for every arm; truncation rate reported per run.
- trainq slots: ALWAYS pass `station-a:1,station-b:1` (GB300 = training; 6000s = eval) plus env `SOFT_SLOTS=station-a:0,station-b:0 SOFT_EXTRA=base-qwen3-1.7b EVAL_DONE=logs/queue/eval-step3.done` (6000s train only when eval backlog is empty). Launch queues ONLY with `setsid nohup python3 -P scripts/<q>.py ... < /dev/null &` from mom-lab/ (a queue in the caller's process group took the Claude session down with it on 10-29).
- Ops rule: stop_queue.sh evalq also removes the Stations' mom-eval-* servers (orphans deadlock the next evalq). Stop queues with scripts/stop_queue.sh evalq|trainq (the queue process started before 13:10 on 10-29 is still named scripts/queue.py: stop it with stop_queue.sh queue) (kills the process group, so no orphan generators double-write result files).
- 2026-10-30 (Stephen): no patent attorney; keep work private, continue research. Posts stay draft-only.

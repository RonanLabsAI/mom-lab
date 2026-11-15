I'll review the paper against the frozen JSON, the locked addenda, and the notebook. Read-only: no edits, no training, no GPU jobs.The paper is loaded. Next I will pull the analysis JSON keys and match every number in the abstract and Tables 1–4.Verdict: reject for a top cs.CL venue as written. The frozen numbers are mostly transcribed carefully, and several caveats are real. The headline is not. Every primary LoRA-library hypothesis is INCONCLUSIVE, the METAR "skill" result is still open, and the prose treats both as settled.

Audit basis: printed values checked against `results/analysis/*.json` at the displayed precision, round-half-up. I did not re-bootstrap from per-item scores. If `scripts/*_analysis.py` is wrong, the paper and the JSON are wrong together.

## 1. Number check

Abstract, Table 1 (`tab:main`), Table 2 (`tab:verdicts`), Table 3 (`tab:fresh`), Table 4 (`tab:niche`), and the intervals in Sections 4.1–4.5 match the JSON under that rounding, with the exceptions below.

Checked and consistent, so they are not listed one by one:

- Table 1 composites from `partb_test_embed-lr.json` `systems.*.{core,para,mixed}.score` and `partc_embed-lr.json` `systems.*.para2`. Half-integers land on the paper's side of the tie: 4B library mixed is exactly 69.25 and is printed 69.3; Spider 8B is 67.45 and is printed 67.5; distilled Spider is 74.45 and is printed 74.5 (`partd_b_embed-lr.json` `scores.sql`).
- Table 2 intervals match `partb_test_embed-lr.json` `tests[]` and the para2 rows match `partc_embed-lr.json` `tests[]`. Verdict labels H1-4B, H2, H3-4B = INCONCLUSIVE and the Part B H2-4bit para component = FAIL match `hypotheses` / that test's `verdict`.
- Table 3 point estimates and the three intervals in Section 4.3 match `partd_b_embed-lr.json` `scores` and `tests` B1 (`diff` 6.65, `lo` 4.75, `hi` 8.5167), B3 (7.30, 5.433, 9.167), B2 (2.090, 0.181, 3.973). `n.sql` = 2000, `n.tools` = 1292, `router_acc` = 1.0.
- Table 4 accuracies match `partd_f.json` `scores` and `partd_c.json` `scores` for the adapter rows. Tokens match `partd_f.json` `cost_per_item.*.{metar,medcalc}.out_tokens`. Section 4.5 intervals match `partd_f.json` `tests` F3, F5, F2, F1, F3 (medcalc) and `vs-large` vs 32B thinking, and `partd_c.json` `tests` C1, C2, and the three C5 Pre-Flight rows.
- Section 4.2 D-A intervals match `partd_embed-lr.json` `tests` A1. Section 4.4 matches D1, D3, D2 on `bird_dev`, E1/E2, and the Part C secondary `4B lora-replay vs lora [para2]` (−0.0325, −0.691, 0.623).
- Memory 9.99 GB, 17.59 GB, ratio 0.568 match `memory_report.json` `systems["Qwen3-4B + 2 LoRA (sql,tools) + router"].total_b1x8192_GB` = 9.988, `systems["Qwen3-8B"].total_b1x8192_GB` = 17.589, `h2_ratio_bf16` = 0.56785. 17.7 GB matches the 1.7B replay system, 17.664. 7.31 GB matches AWQ-8B, 7.307. 7.13 GB is not a stored field; it is `(bytes.weights["4B-FP8"] + bytes.adapters["4B"] + bytes.router + 1207959552) / 1e9` = 7.133.
- "About 0.3 points" of oracle gap: largest absolute composite gap I computed is 0.281 (`partc` 1.7B replay para2, embed 53.684 vs oracle 53.965).
- 11.6× is `1663.829 / 143.227` = 11.62, from `cost_per_item["8B few-shot+thinking"].metar.out_tokens` over `cost_per_item["4B+metar"].metar.out_tokens`.
- Lenient METAR 18.5 matches `partd_f.json` `metar_lenient_time["8B zero-shot"]`.
- 590 and 1,350 reconstruct as sums of benchmark `n` in `results/partbtest/base-qwen3-4b/scores.json`. 1,562, 1,055, and 9,300 are not in `results/analysis/`. They are in `content/PART_C_PREREG.md`, `content/PART_D_B_ADDENDUM.md`, and `data_partd_c/metar_decoder_validation.json` (`compared`: 9300).

Mismatches:

1. Abstract lines 41–42. Paper: "the 8B with thinking writes 1,664." File `results/analysis/partd_f.json`, path `cost_per_item["8B thinking"].metar.out_tokens` = 1486.356 (prints as 1,486). Path `cost_per_item["8B few-shot+thinking"].metar.out_tokens` = 1663.829 (prints as 1,664). The 83.2% "best-prompted" accuracy is the few-shot-plus-thinking row (`scores["8B few-shot+thinking"].metar` = 83.167). Section 4.5 lines 391–392 labels 1,664 correctly. The abstract does not. The thinking-only row is 80.5% and 1,486 tokens (`scores["8B thinking"].metar` = 80.5).

2. Section 4.1 lines 236–239. Paper: "Most router misses are on general-knowledge items (MMLU-Pro 71%) and reworded translation items (9%), which route to the base anyway." File `results/analysis/partb_test_embed-lr.json`, path `router_acc["4B/mmlu_pro"]` = 0.7141 and `router_acc["4B/flores_pp"]` = 0.09. Those are accuracies from `scripts/partb_analysis.py` (predicted domain equals gold domain), not miss rates. Miss rates are about 28.6% and 91%. Using benchmark `n` from `results/partbtest/base-qwen3-4b/scores.json`, MMLU-Pro accounts for about 1,720 of roughly 1,970 misses across core, para, and mixed, which is about 87% of misses, not 71%. The 9% figure is flores paraphrase accuracy (100 items, about 91 misses), not a miss share. "Route to the base anyway" is also not what the router does: at 4B only non-{sql, tools} predictions fall through to the base (`scripts/partb_analysis.py` lines 4–5 and 133). A wrong sql or tools prediction is served by an adapter. No confusion matrix is stored in the JSON, so the share that hits an adapter is unchecked.

3. Table 2 caption, lines 264–266, against the locked combination rule. Paper: "H2-4bit fails in Part B ... Part B's verdict stands." Part B's own para component does FAIL: `partb_test_embed-lr.json` `tests` where `test` = "AWQ-4B+lora >= AWQ-8B - 2 [para]", `diff` −5.343, `lo` −8.411, `hi` −2.419, `verdict` FAIL. The Part C replacement is INCONCLUSIVE: `partc_embed-lr.json` `tests` "AWQ-4B+lora >= AWQ-8B - 2 [para2]", `diff` −2.597, `lo` −4.503, `hi` −0.661. The frozen combined verdict, which keeps non-para Part B components and replaces the para component, is `partc_embed-lr.json` `combined["H2-4bit"]` = "INCONCLUSIVE". That rule is `content/PART_C_PREREG.md` lines 45–47 and `scripts/partc_analysis.py` lines 110–121. The same file's `combined["H1-replay (secondary)"]` = "PASS" is the verdict the paper uses at lines 232–234. The paper takes the combined upgrade and does not take the combined downgrade.

Sign hazard, not a different quantity: Section 4.2 lines 283–285 says the library "trails 8B by 1.40 points [−2.48, −0.27]" and "by 3.56 [−6.49, −0.71] and 2.58 [−4.51, −0.56]". Those intervals are the signed library-minus-8B intervals in Table 2 (`diff` −1.398, −3.557, −2.578). The positive word "trails by" is attached to a negative interval.

Wording that is weaker than the interval, not a mismatch: Section 4.3 line 320 calls BFCL "non-inferior" at +2.09 [0.18, 3.97]. `tests` B2 has `kind` "ni" and `lo` 0.181 > 0, so the superiority test would also pass. Table 2's para2-vs-pooled "+1.36 [−0.37, 3.18], a pass" (lines 221–223 and 254–255) is a non-inferiority pass: `lo` −0.369 is above −2 and below 0.

## 2. Overclaiming

The locked verdicts in `partb_test_embed-lr.json` `hypotheses` and `partc_embed-lr.json` `combined` are INCONCLUSIVE for H1-1.7B, H1-4B, H2, H3-1.7B, and H3-4B. The only PASS is the secondary 1.7B full-FT replay library. The paper says this at lines 232–234 and then does not let it govern the title, the introduction, or the discussion.

- Lines 20–21 and 85–88. "Adapters Buy Skills" and "a small adapter supplies the procedure and can beat a model twice the size." H2 does not pass. On core the 4B library is worse than 8B by a CI that excludes 0 (`diff` −1.398, `hi` −0.274) and only fails to FAIL the non-inferiority test because `hi` is above −2.
- Lines 440–444. "a small adapter can supply the procedure and beat a model twice the size, as on METAR and Spider." Spider: `partd_b_embed-lr.json` `scores.sql` is 74.1 (library), 74.45 (distilled), 75.07 (pooled), 67.45 (8B). Section 4.3 says the single models get the same gain. The discussion then cites Spider as evidence for the adapter. METAR is the pending gate. The discussion states it as an instance of the rule.
- Lines 35–36. "Confirmatory tests were locked with hash manifests before their test-split outputs existed." Part B was preregistered. `content/PART_C_PREREG.md` line 4 and `paper/main.tex` lines 175–177 say Part C and Part D were written after Part B test results. Part D is where METAR, MedCalc, Spider-2000, and the rank/replay nulls come from. Calling those confirmatory in the abstract, while Section 3.5 correctly says "locked follow-ups, not preregistrations," is the abstract doing the work the caveat undoes.
- Lines 78–79 and 411–416. The gate is "preregistered." `preregs/2026-11-14-metar-schema-gate.md` lines 18–21: local git only, public repository "pending," `preregUrl: pending`. It was written after Part D-C/F results, same epistemic status the paper refuses to call preregistration for Part D.
- Lines 275–276, Figure 2 caption. "The library clears every alternative on core prompts except Qwen3-8B." Distilled core is INCONCLUSIVE: `partb_test_embed-lr.json` test "4B lora > distill [core]", `diff` 0.796, `lo` −0.051, `hi` 1.624. Lines 227–229 state that correctly. The caption does not.
- Lines 74, 406–407, and the paragraph label "Knowledge goes with size." The only reported contrast is descriptive: Pre-Flight 8B minus 4B is `partd_c.json` C5 `diff` 4.583, `lo` −0.833, `hi` 10.417. Table 4 itself shows 32B zero-shot at 67.1, tied with 4B zero-shot (both `partd_f.json` `scores` preflight 67.083) and below 8B (71.667). `partd_f.json` tests preflight 32B zero-shot minus 8B: `diff` −4.583, `lo` −9.583, `hi` 0.833. Size does not win this control.
- Lines 406–407. "thinking adds nothing at any size." 32B thinking minus 32B zero-shot on Pre-Flight is `diff` 3.75, `lo` −1.667, `hi` 9.167, one seed, n = 240. The interval allows a gain of about 9 points.
- Lines 287–290. "ties on core" for FP8 4B library vs AWQ 8B. A1 core is a non-inferiority PASS, `diff` 0.022, `lo` −1.047, `hi` 1.129. That is not an equivalence result. The −2 point bar is large next to a 1.33 point base effect.
- Lines 391–392 and the abstract's 143 vs 1,664. Section 4.5 does say the comparison is against thinking traces and that the gate has not run. The abstract's last sentence says the same. The title, lines 85–88, and lines 440–444 do not wait. `preregs/2026-11-14-metar-schema-gate.md` lines 125–130 already predicts the token ratio against a schema arm is about 1× and that the 11.6× figure survives only against thinking. Leading with 11.6× is leaning on a comparison the preregistration says is the wrong one.
- Line 121. "Section 4.6 tests it." Section 4.6 has not been run. Lines 435–436 are still `[GATE RESULT PENDING]`.

What is not overclaiming: H1 is called inconclusive (lines 224–225). Nulls are in Section 4.4 with the right FAIL on D3. MedCalc's reversal is reported with the intervals that FAIL. Template overlap is in Section 3.3 and the limitations. Spider train contamination is in Section 4.3.

## 3. Prior work

Misstatements of papers that are cited:

- Lines 101–102, Arrow, arXiv:2405.11157. Arrow is the zero-shot router, not the library builder. The library method is model-based clustering. Routing is per token and per layer by alignment with LoRA factors. It is not "route a request to one adapter," which is this paper's router. "Match or outperform joint training" is the abstract's claim for the whole MBC+Arrow setup on held-out tasks. Citing it as a request-level library that already answered this paper's question is inaccurate, and it also undercuts line 59 ("what we could not find").
- Lines 95–96, BAR, arXiv:2604.18473. The 50.5 and 49.1 figures match the abstract: full retrain with mid-training 50.5, BAR 49.1. The sentence uses that pair to imply retraining beats modular experts. The same abstract's other retrain number is 47.8 without mid-training, which BAR beats, and the paper's claim is linear-cost updates without forgetting. "Post-training experts" also drops that each expert gets its own mid-training, SFT, and RL.
- Lines 92–99. "These works end in one larger model or measure mostly perplexity and pass rates." Branch-Train-Merge (arXiv:2208.03306) is an ensemble of separate experts; the previous sentence says so. SmallTalk LM (arXiv:2410.03529, Filippova et al., "No Need to Talk") routes a prefix to one small expert so inference uses a fraction of the mixture. Its headline metric is perplexity, so that half of the "or" fits SmallTalk and does not fit BTM.

Missing, with ids I am willing to stand behind:

- Biderman et al., "LoRA Learns Less and Forgets Less," arXiv:2405.09673. LoRA versus full fine-tuning on code and math, learning and forgetting. This is the comparison H1/H3 are made of. Not cited.
- Dou et al., "LoRAMoE," arXiv:2312.09979. Frozen backbone, routed LoRA plugins, explicitly to stop SFT from wiping world knowledge. Closest method to the system under test. Not cited.
- Pfeiffer et al., AdapterFusion, arXiv:2005.00247, and AdapterHub, arXiv:2007.07779. The adapter-library line did not start with Arrow.
- Zhou et al., LIMA, arXiv:2305.11206. The superficial-alignment claim (pretraining holds the knowledge, a small update teaches the format) is this paper's title. Not cited.
- Scholak et al., PICARD, arXiv:2109.05093. Grammar-constrained text-to-SQL. The Spider result has no constrained-decoding or execution-guided baseline. DIN-SQL, Pourreza and Rafiei, arXiv:2304.11015, is the prompting decomposition that beat fine-tuned Spider models and belongs next to "prompting versus training" on SQL.
- Luo et al., catastrophic forgetting in continual fine-tuning, arXiv:2308.08747. They cite SDFT (arXiv:2402.13669) and skip the forgetting literature the pooled-model collapse is an instance of.
- Yu et al., Spider, arXiv:1809.08887, is in `paper/refs.bib` and is not cited from `paper/main.tex`. The fresh-SQL table is a Spider result.
- Shuttleworth et al., "LoRA vs Full Fine-tuning: An Illusion of Equivalence," arXiv:2410.21228, belongs next to the rank-64/128 null (higher rank was the point of that paper). Optional relative to Biderman.
- Min et al., "Rethinking the Role of Demonstrations," arXiv:2202.12837, belongs on the MedCalc one-shot reversal.

Line 59 is not defensible once Arrow, BAR, Biderman, and LoRAMoE are in the related-work section.

## 4. Design weaknesses

Disclosed in the paper, and still load-bearing:

- One family, Qwen3 only. Limitations, lines 458–459.
- Template overlap on core prompts. Section 3.3 lines 160–162 and limitations lines 459–461. The clean tests are where H1 is inconclusive. The abstract still leads with the core +1.33 and +5.62.
- Part D chosen after results. Section 3.5. Timestamps are local. Limitations lines 464–465.
- No multiplicity correction. Section 3.4 lines 169–170. The rule "all components must pass" makes PASS hard and FAIL easy. H2-4bit's Part B FAIL is one component.
- Spider train is inside the SQL mixture; remaining items are new databases, not a new task. Section 4.3 lines 321–323.
- MedCalc names the calculator. Limitations lines 462–463.
- 32B thinking is one seed. Table 4 caption.
- Composites weight domains equally, and several CIs are wide. Limitations lines 470–471.
- The −2 non-inferiority margin was fixed before the Part B test (`content/PART_B_PREREG.md` lines 49–50 and 86). The paper states it and does not justify it. It is larger than the core effect the abstract leads with, which is why H2 can trail with a CI entirely below 0 and stay INCONCLUSIVE.

Not disclosed, or disclosed only as a definition that hides the problem:

- The bootstrap is an item bootstrap of seed-averaged scores (`scripts/partb_analysis.py`). Training-seed noise is not in the interval. Three seeds is the entire training uncertainty, and it is collapsed before the resample. Spider items are also clustered by database. Neither is mentioned. The intervals in Tables 2–4 are too narrow for the claims attached to them.
- Serving memory is batch-1, one 8,192-token KV cache. Section 3.6 says so. `memory_report.json` also has batch 16 × 4,096: the 4B library is 18.444 GB and Qwen3-8B is 26.045 GB, ratio about 0.71, not 0.568. The "half the memory" framing does not survive the KV regime the paper's own file already computed. Training FLOPs are also unmatched: the 4B library keeps two adapters, the pooled run trains all six domains, and the distilled run sees 102.2M tokens against 83.5M (Section 3.2). Distill still ties the library. That is reported. The memory claim is the one that is quoted at the flattering batch size.
- Thinking arms are sampled at temperature 0.6, top_p 0.95, top_k 20 (`content/PART_D_F_ADDENDUM.md`). The paper says greedy unless a condition says otherwise (lines 165–166) and never says thinking is sampled. Token counts and the 11.6× ratio compare greedy rank-16 LoRA with a sampled chain of thought. `content/PART_D_DEVIATIONS.md` item 7 also says one 8B thinking seed was paused mid-run and resumed, so the sample is not even one uninterrupted draw. The paper is silent.
- The paraphraser is Qwen3.8-27B (`content/PART_C_PREREG.md` lines 14–15). The paper says "a separate model" (line 157). Same family. Para and para2 are not an external-style test.
- "Mixed-skill" (abstract line 34, line 215) is one skill per item, in Spanish, German, or Chinese (lines 159–160). It is a language shift, not a multi-skill request. Router accuracy on mixed math is 0.844 (`router_acc["4B/gsm8k_ml"]`); on mixed SQL, code, and tools it is 1.0. Easy domains.
- The 4B library is two adapters, sql and tools, because the code adapter lost on dev (lines 144–145). That selection is legitimate if frozen before test. It also means "library versus pooled" compares a system that refused the domains that hurt with a system that was forced to train on them. Most of the +5.62 is the pooled model falling 4.3 points below the untouched base (65.219 − 60.923). Lines 217–218 say this. The abstract still leads with 5.62.
- Pretraining contamination is unmeasured. "Qwen is unlikely to have trained on heavily" (line 380) is a belief. METAR as a format is old and public. Spider test questions can sit in pretraining even after the 1,055 SFT-overlap drop.
- `content/PART_D_DEVIATIONS.md` is not in the paper. Two BFCL-live prompts were dropped after a 400 error (1,294 → 1,292). Deviation 6: seed-mean Part D-C scores were read before `partd_c_analysis.py` was written. The deviation says the tests were already in the addendum. A paper whose contribution is the lock has to print this log. Lines 175–176 as written are cleaner than the log.
- MedCalc sensitivity on 1,095 items after dropping exemplar-overlap (`partd_f.json` `tests` F1-sens through F5-sens) is almost identical to the full-set FAILs and is not in the paper. That omission does not help the authors. It should still be one sentence.

The rank null and the routed-replay null are reported at the right strength. The always-on damage numbers (+4.11 and +5.05) match D1 and are correctly separated from the routed FAIL.

## 5. The gate

Section 4.6 will not answer "does constrained decoding close the METAR gap?" It will answer a narrower question the authors have already predicted the answer to.

What is adequate: same 1,000 items, same 14-field scorer, greedy bf16, a parity rerun that must match stored 8B zero-shot bytes, XGrammar pinned, schema restricted to what the prompt says, best-of-arms chosen after the fact so the adapter's test gets harder, and a written rule that a matched result withdraws the training win. `preregs/2026-11-14-metar-schema-gate.md` lines 74–91 (parity stop, whitespace-loop smoke) are better than Section 4.6, which omits them. Lines 427–429 also loosen the preregistered "stands" rule. The preregistration requires the CI low of (adapter − best schema arm) to be above 2.0. The paper says "the adapter's lead is above 2 points," which a reader can take as the point estimate.

What a reviewer will say is missing:

- A parser baseline. Labels come from a deterministic decoder that already agrees with an independent parser on 9,300 reports (`data_partd_c/metar_decoder_validation.json`). The practitioner alternative to training a 4B model to imitate that decoder is to call it. A format schema does not stand in for that. On this task the tool scores 100% at a few hundred tokens of code and zero model tokens. Until that row exists, "adapters buy skills" on METAR is "adapters imitate a parser."
- A schema that encodes the decoding table the authors already say is the residual. The gate card lines 33–37, written before any gate run: 8B few-shot and 32B zero-shot already parse as JSON on 100% of items; 8B few-shot field accuracy is flight category 69.4, temperature 76.2, ceiling 83.3. Lines 125–128 predict 8B 5-shot plus schema at 38–48, against an adapter at 99.8, and predict G1 and G2 pass. A format constraint cannot move errors that are already inside valid JSON. Excluding a schema that computes flight category from ceiling and visibility (gate card lines 142–146) excludes the only constrained decoder that could threaten the result. G3 "matched" (CI high below 2) is then almost unreachable, and "stands" is the predicted branch. That is a locked measurement of a gap they already estimated, not a test that can close it.
- No 4B plus schema arm. The user in the introduction has a 4B model. Jumping to unconstrained-then-constrained 8B and 32B skips the model the adapter is attached to.
- No 32B few-shot unconstrained run, so G4 cannot separate schema from five-shot for the larger model. The card admits this. The paper's G4 sentence (line 427) does not.
- One backend, one schema, one GPU family, thinking-plus-schema at one seed and temperature 0.6. Disclosed in the limitations (lines 466–467) and still too thin to underwrite the decision rule.
- Output tokens only. Prefill for five-shot prompts is thousands of tokens (`max` prompt cited as 4,586 in the gate card). `cost_per_item` already has latency. The paper's cost claim does not use it.

Run the gate. Do not expect it, under this schema, to earn the right to keep the METAR headline.

## 6. Ten changes, by effect on the score

1. Put a deterministic METAR parser in Table 4, and stop using METAR as evidence that an adapter supplies a procedure a model lacks. That single row removes the title's best example. A format-constrained 8B/32B does not replace it.
2. Rewrite the title, lines 85–88, and lines 440–444 so they say only what passed. Primary LoRA hypotheses: inconclusive. Only full pass: 1.7B full-FT replay at 17.7 GB, which is the 8B memory budget (`memory_report.json` 17.664 vs 17.589). METAR: pending, and not a skill result until the parser row exists. Spider: any 4B fine-tune, not the router.
3. Report `combined["H2-4bit"]` = INCONCLUSIVE, or show both verdicts and the replacement rule. As written, lines 264–266 keep the Part B FAIL and lines 232–234 take the combined PASS. That is the integrity issue a reviewer will trust least.
4. Do not submit while lines 435–436 are red. Either the gate has numbers and the decision rule has been applied, or METAR is an appendix with the thinking comparison labeled secondary. Delete 11.6× from any lead position. The preregistration already says the honest token comparison is about 1×.
5. Replace Figure 2's caption (lines 275–276). Distilled core `lo` is −0.051. "Clears every alternative except 8B" is false.
6. Fix the two number bugs: abstract 1,664 is few-shot-plus-thinking, not "8B with thinking" (1,486); lines 236–239, 71% and 9% are `router_acc`, and misses do not "route to the base anyway." Soften "Knowledge goes with size" to the descriptive interval, and point at Table 4's 32B Pre-Flight tie with 4B.
7. Cite Biderman arXiv:2405.09673, LoRAMoE arXiv:2312.09979, AdapterFusion arXiv:2005.00247, LIMA arXiv:2305.11206, PICARD arXiv:2109.05093, and Spider arXiv:1809.08887. Correct Arrow (per-token, per-layer) and the BAR 50.5/49.1 gloss (BAR beats the 47.8 retrain). Delete "what we could not find."
8. Put training-seed resampling, or a seed-level sensitivity, into the intervals, and cluster Spider by database. State that the published CIs are item noise around a three-seed mean. Publish the batch-16 memory ratio from `memory_report.json` (about 0.71) next to 0.568.
9. Say in Section 4.5 that thinking is temperature 0.6 with three samples, that one 8B thinking job was resumed mid-run (`PART_D_DEVIATIONS.md` item 7), and that the paraphraser is Qwen3.8-27B. Move the deviations log into the paper. "Mixed-skill" should be renamed to a translation of single-skill items.
10. Add the missing baselines that change interpretation even if METAR is demoted: constrained or execution-guided SQL (PICARD-style) on the 2,000-item set, and a 4B-plus-schema METAR arm. Report the MedCalc n=1,095 sensitivity that is already in `partd_f.json`. Justify the −2 point margin in one sentence or run a margin sensitivity. None of these rescue the headline. They stop the next overclaim.

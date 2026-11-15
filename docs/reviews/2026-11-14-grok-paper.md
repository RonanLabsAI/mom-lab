# Grok adversarial review of paper/main.tex (draft v0), with dispositions

- Run: 2026-11-14 PT, Grok Build CLI 1.0.77 (subscription, default model, no `--model`), from the worktree root.
  Command: `~/.grok/bin/grok -p "$(cat <scratchpad>/grok-paper-prompt.txt)" --permission-mode acceptEdits --max-turns 300 --no-alt-screen --output-format plain`.
  The prompt told Grok to review only and edit nothing. Verified afterwards: `paper/main.tex` and `paper/refs.bib`
  sha256 unchanged, no new files except its output.
- Raw output: `docs/reviews/2026-11-14-grok-paper-raw.md` (25,133 bytes).
- Grok's overall verdict: "reject for a top cs.CL venue as written". The frozen numbers are "mostly transcribed
  carefully"; the headline overclaimed relative to the locked verdicts.
- Each point below was checked against the repo before acceptance. "Accepted" means the fix is in `paper/main.tex`
  (recompiled, 15 pages, no undefined references).

## 1. Number mismatches

| # | Grok's point | Check | Disposition |
|---|---|---|---|
| 1.1 | Abstract says "the 8B with thinking writes 1,664"; 1,664 is few-shot + thinking, thinking alone is 1,486 | `partd_f.json` `cost_per_item["8B few-shot+thinking"].metar.out_tokens` = 1663.83, `["8B thinking"]` = 1486.36 | Accepted. Abstract now says "with worked examples and thinking". |
| 1.2 | Router "misses 71% / 9%" are accuracies, not miss shares; misses do not all go to the base | `router_acc["4B/mmlu_pro"]` = 0.714, `["4B/flores_pp"]` = 0.09; `scripts/partb_analysis.py` header: a predicted domain in ROUTES goes to the adapter, everything else to the base | Accepted. Rewritten as "least accurate on MMLU-Pro (71%) and reworded translation (9%); a miss goes to the base unless the router predicts SQL or tools". |
| 1.3 | H2-4bit: paper kept the Part B FAIL but took the Part C combined PASS for replay; the combined H2-4bit is INCONCLUSIVE | `partc_embed-lr.json` `combined["H2-4bit"]` = INCONCLUSIVE; `content/PART_C_PREREG.md` lines 45-47 define the rule and also say "H2-4bit stays FAIL" (the text's expectation was wrong; notebook 2026-11-01 02:05 records this) | Accepted. Table 2 caption, Section 4.2 and the abstract report both: Part B FAIL, combined INCONCLUSIVE, and that the prereg text expected FAIL. |
| 1.4 | "trails by 1.40 [-2.48, -0.27]" attaches a positive word to a negative interval | Correct | Accepted. Rewritten as signed "library minus 8B" differences. |
| 1.5 | BFCL "non-inferior" undersells: lo > 0 | `partd_b` B2 lo = 0.181 | Accepted. Says it passes the locked NI test and the interval also excludes 0 (B2 was locked as NI, so it is not relabelled a superiority PASS). |
| 1.6 | Para2 vs pooled "a pass" is an NI pass | Correct | Already stated as non-inferiority in the table; text kept. |

Grok found no mismatch in Table 1, Table 3, Table 4, the memory numbers, or the 11.6x ratio.

## 2. Overclaiming

| Point | Disposition |
|---|---|
| Title "Size Buys Knowledge, Adapters Buy Skills" and the intro/discussion "rule" are not supported: every primary hypothesis is INCONCLUSIVE, METAR is pending, Spider is any 4B fine-tune | Accepted. New title "Adapter, Example, or a Bigger Model? A Locked Five-Way Comparison for Adding Skills to a Small Frozen LLM". Intro states the pattern as a working hypothesis, not a finding. The discussion says Spider supports fine-tuning, not routing, and that METAR is pending the gate. |
| Abstract "Confirmatory tests were locked..." blurs Part B (preregistered) with Part C/D (designed after results) | Accepted. Abstract separates the preregistered main test from follow-ups locked after its results. |
| Gate called "preregistered" though it was written after Part D-C/F results, the same status the paper refuses to call preregistration for Part D | Accepted. Section 4.6 is now "locked" and says it was written after D-C/F results. |
| Figure 2 caption "clears every alternative except 8B" is false for distilled (lo = -0.05) | Accepted. Caption rewritten. |
| "Knowledge goes with size" is descriptive and 32B zero-shot ties 4B on Pre-Flight; "thinking adds nothing" overstated | Accepted. Paragraph renamed "The knowledge control"; says size is not clearly better and thinking did not move it measurably, with the 32B interval. |
| FP8 lib vs AWQ-8B "ties" is an NI pass, not equivalence | Accepted. |
| "Section 4.6 tests it" when it has not run | Accepted ("is designed to test it"). |
| Abstract leads with +5.62 vs pooled, mostly pooled's own drop | Accepted. Abstract adds "mostly because pooled training pulls the model below its untouched base". |

## 3. Prior work

| Point | Check | Disposition |
|---|---|---|
| Arrow is the router; MBC builds the library; Arrow routing is per-token and per-layer | arXiv 2405.11157 abstract: MBC clusters tasks, Arrow selects adapters "for new inputs" zero-shot. The abstract does not state per-token/per-layer | Partly accepted. Description corrected to MBC + Arrow; the per-token/per-layer detail is not added because it was not verified from the source. |
| BAR gloss omits the 47.8 baseline BAR beats and its per-expert pipelines | arXiv 2604.18473 abstract confirms 49.1 vs 47.8 / 50.5 and per-expert mid-training, SFT and RL | Accepted. |
| "These works end in one larger model or measure perplexity" misfits BTM | Correct | Accepted; sentence now names FlexOlmo and BAR (end in one MoE) and SmallTalk (perplexity). |
| Missing: Biderman 2405.09673, LoRAMoE 2312.09979, AdapterFusion 2005.00247, LIMA 2305.11206, PICARD 2109.05093, DIN-SQL 2304.11015, Luo 2308.08747, Shuttleworth 2410.21228, Min 2202.12837 | All nine titles and first authors checked on arxiv.org on 2026-11-14 | Accepted; all nine added and cited. |
| Spider in refs.bib but not cited | Correct | Accepted. |
| "What we could not find" is indefensible next to Arrow, BAR, LoRAMoE | Fair | Accepted. Replaced with an explicit statement that routed LoRA plugins are prior art and the contribution is the matched comparison. |

## 4. Design weaknesses

| Point | Disposition |
|---|---|
| Item bootstrap excludes training-seed variance; Spider not clustered by database | Accepted as disclosure (Setup 3.4 and Limitations). Not re-analysed: a seed-and-cluster bootstrap is a new exploratory analysis; recommended before submission (open item). |
| Batch-16 memory ratio is ~0.71, not 0.568 | Accepted. `memory_report.json`: 18.444 / 26.045 = 0.708. Section 3.6 now gives both and says "about half" holds only at small batch. |
| Thinking arms are sampled (T 0.6), not greedy; one 8B thinking seed resumed mid-run | Accepted. Setup 3.4, Section 4.5 and Appendix B. |
| Paraphraser is Qwen3.8-27B (same family) | Accepted. Named in 3.3 and in Limitations. |
| "Mixed-skill" is single-skill items translated | Partly accepted. Described precisely; the set name is kept because the preregistration uses it. |
| Deviations log not in paper | Accepted. New Appendix B summarises all seven Part D entries. |
| MedCalc n=1,095 sensitivity omitted | Accepted. One sentence in 4.5 (F3-sens 21.04 [18.29, 23.71]). |
| -2 margin not justified | Partly accepted. The paper says it was fixed in the Part B prereg before dev results and that no margin sensitivity was run. No post-hoc justification added. |
| Pretraining contamination unmeasured | Accepted (Limitations). |
| 4B library is two adapters because code lost on dev; most of +5.62 is pooled's drop | Accepted (abstract wording; 4.1 already said it). |

## 5. The gate (Section 4.6 and `preregs/2026-11-14-metar-schema-gate.md`)

| Point | Disposition |
|---|---|
| Paper loosens the "stands" rule (CI low of adapter minus best schema arm > 2.0, not the point estimate) | Accepted. Section 4.6 now states the lower-bound rule exactly. |
| Paper omits the parity stop and the whitespace smoke rule | Accepted; both added. |
| G4 has no unconstrained 32B five-shot counterpart | Accepted; stated. |
| A parser baseline: the labels come from a deterministic decoder, so "adapters buy skills" on METAR is "adapters imitate a parser" | Partly accepted. Section 4.5 now says METAR has an exact non-learned solution, that one would call it in practice, and that METAR is a controlled probe of procedure learning. Rejected as a Table 4 row: the decoder generated the labels, so it scores 100% by construction and a row would imply an independent comparison that does not exist. |
| A format-only schema cannot fix errors already inside valid JSON; a flight-category schema, a 4B+schema arm are missing; G3 is nearly unreachable | Accepted as disclosure ("What the gate cannot show" paragraph). The gate card is owned by the parent workstream; these are forwarded as recommendations: consider adding a 4B + schema arm and a rule-encoding schema as secondary arms before the freeze, if it has not happened yet. |
| Output tokens only; prefill of 5-shot prompts ignored | Not changed in this draft; noted as an open item for the cost comparison once the gate runs. |

## 6. Grok's top-10 list

1 (parser row): partly accepted, see 5. 2 (title and claims): accepted. 3 (H2-4bit both verdicts): accepted.
4 (do not submit with the gate pending): accepted as a gate on submission, not a text change. 5 (Fig. 2 caption):
accepted. 6 (two number bugs, Pre-Flight): accepted. 7 (citations, Arrow, BAR): accepted (Arrow partly).
8 (seed/cluster intervals, batch-16 memory): memory accepted; intervals disclosed, re-analysis open.
9 (sampling, resumed seed, paraphraser, deviations, "mixed"): accepted ("mixed" name kept).
10 (PICARD-style SQL baseline, 4B+schema METAR arm, MedCalc sensitivity, margin): MedCalc sensitivity accepted;
the two new baselines need GPU and a prereg change and are open items; margin disclosed.

## Number check (`scripts/paper/check_numbers.py`)

```
python3 scripts/paper/check_numbers.py
362 decimal numbers, 362 found, 0 missing
```

How it matches: a decimal number in `main.tex` counts as found if its absolute value equals any number in
`results/analysis/*.json` rounded to the printed precision (0-1 values also tried as percentages), or if the literal
string appears in `notebook/*.md`, `content/*.md`, `GOAL.md` or `data_partd_c/README.md`. This is lenient: a wrong
number that happens to equal an unrelated JSON value would pass. So the following derived numbers were also checked
by hand:

- 4.3 (4B pooled below 4B base on core): 65.219 - 60.923 = 4.30 from `partb_test_embed-lr.json` systems.
- 0.71, 18.44, 26.05 (batch-16 memory): `memory_report.json` systems totals, 18.444 / 26.045 = 0.708.
- 7.13 GB (FP8 4B library): `content/PART_D_PREREG.md` line 34; Grok recomputed 7.133 from `memory_report.json` bytes.
- 17.7 GB (1.7B replay library): `memory_report.json` 17.664.
- 11.6x: 1663.829 / 143.227 = 11.62 (`partd_f.json`), also in `notebook/2026-11-02.md`.
- 21.04 [18.29, 23.71]: `partd_f.json` tests F3-sens (sign flipped: few-shot+thinking minus adapter).
- 0.3-0.8 seed SD: `content/PART_A_SUMMARY.md` and `GOAL.md` ("Seed sd 0.3-0.8").
- 1,562 / 1,055 / 9,300 are integers (not checked by the script): `content/PART_C_PREREG.md`,
  `content/PART_D_B_ADDENDUM.md` / notebook 2026-11-01, `data_partd_c/README.md`.

## Open items after this review

1. The gate result (Section 4.6 is still `\gatepending`). Do not submit until the decision rule has been applied.
2. Seed-and-cluster bootstrap sensitivity for the headline intervals (CPU only, exploratory, from saved per-item scores).
3. Body is 11 pages against an 8-10 target; trim Section 2 or move Table 2 to the appendix.
4. Gate design recommendations forwarded to the gate owner (4B + schema arm, a rule-encoding schema).
5. PICARD-style constrained SQL baseline on the 2,000-item Spider set, if a venue asks.

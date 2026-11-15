# Part D addendum B: fresh SQL and tool-calling test (FINAL at freeze)
Companion to content/PART_D_PREREG.md (kept as a separate file so the frozen plan's hash is unchanged). Written
2026-11-01 ~07:50 PT, before any model output on this data exists. Locked by artifacts/PARTD_B_FREEZE.sha256 (hash in
notebook/2026-11-01.md). Shared rules (serving, statistics, bars) are those of PART_D_PREREG.md.
(Clock note: PART_D_PREREG.md says "~08:10 PT"; it was written ~07:05 PT. PST began today.)

## Why
Exploratory (data_findings.md section 2): 4B + SQL adapter beat untouched 8B at SQL on 3 of 4 already-seen sets. B tests
that on items no model has been scored on.

## Data (built by scripts/build_partd_b.py; data_partd/README_B.md)
- SQL: data_partd/eval/sql/bird_dev/test.jsonl, 2,000 Spider 1.0 items (dev 38 + test 1,962, 55 databases), same prompt
  format and execution-accuracy scorer as BIRD. Spider dev is mostly excluded because sql-create-context (in our training
  data) contains it; contamination rules (exact, punctuation-normalised, gold-query match) drop 1,055 candidates.
  Gold check 100.0.
- Tools: data_partd/eval/tools/bfcl/test.jsonl, 1,294 BFCL v3 live items (simple 245, multiple 1,010, parallel 15,
  parallel_multiple 24). Gold check 100.0.
Note: Spider is easier than BIRD and its train split is in our SQL training data, so the SQL adapter has format
familiarity here; the base models do not. That is the same advantage it had on BIRD; it is stated, not removed.

## Systems (all scored on both files; step partdB; data root data_partd; domains sql,tools)
Untouched: Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-8B-AWQ, Qwen3-8B-FP8, Qwen3-4B-FP8.
4B r16 adapters (Part B, 3 seeds): pt4-lora16-sql-*, pt4-lora16-tools-* (served as base + LoRA).
FP8-4B + the same adapters (A): fp84-lora16-sql-*, fp84-lora16-tools-*.
1.7B: pt17-replay-sql-r20 (3 seeds, full FT), pt17-lora16-sql (3 seeds), pt17-lora16-tools (3 seeds).
4B single models: pt4-distill (3 seeds), pt4-pooled (3 seeds).
Secondary, added when trained: E (r64/r128) and D (sd-replay) adapters.

Routing: each file is single-domain. Primary = the frozen router (artifacts/router_e5lr.pkl) decides per item; items it
sends elsewhere are answered by the untouched base of that system (so router misses count against the library).
Secondary = oracle (every item to the adapter).

## Tests
- B1 (primary): 4B + SQL adapter beats untouched 8B on fresh SQL (CI low > 0).
- B2 (primary): 4B + tools adapter NI to untouched 8B on fresh tools (CI low > -2).
- B3: 4B + SQL adapter beats 4-bit 8B (AWQ, 7.3 GB, less memory than 4B+adapter) on fresh SQL.
- B4: 1.7B replay SQL specialist NI to untouched 4B on fresh SQL.
- B5: 4B + SQL adapter beats untouched 4B on fresh SQL; 4B + tools adapter vs untouched 4B on fresh tools (beats).
- B6 (secondary): distilled 4B beats untouched 8B on fresh SQL.
- B7 (secondary): FP8-4B + SQL adapter beats AWQ-8B on fresh SQL.

## Predictions
Spider is easier than BIRD: expect untouched 4B ~65-75, 8B a few points higher. B1 PASS (adapter +4 to +8 over 8B).
B2 INCONCLUSIVE or FAIL (on English tools the 4B adapter did not beat its own base: para2 -3.4). B3 PASS. B4 PASS or
INCONCLUSIVE. B5 SQL PASS, tools INCONCLUSIVE. B6 PASS. B7 PASS.

## Deviations log
(empty)

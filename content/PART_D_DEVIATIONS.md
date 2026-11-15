# Part D deviations log (applies to PART_D_PREREG.md and its addenda; those files are read-only)

1. 2026-11-01 07:39 PT, addendum B data. Two BFCL live items have prompts longer than 4,096 tokens
   (bfcl-live_multiple_985-216-0: 5,171; bfcl-live_multiple_217-93-0: 4,604, Qwen3 tokenizer with chat template), so
   with 4,096 max new tokens they exceed the 8,192 context and return HTTP 400 for every model. Found when the first
   job (untouched 1.7B) failed on exactly these two. Removed by prompt length only, not by any score. Tools set is now
   1,292 items. Original kept as data_partd/eval/tools/bfcl/test.locked_v1.jsonl. No SQL item exceeds the limit
   (longest 1,069 tokens). Seen before this fix: the untouched 1.7B's partial sql/tools aggregate (one model).
2. Clock note: PART_D_PREREG.md says it was written "~08:10 PT"; it was ~07:05 PT (PST began 2026-11-01).
3. configs/partd_evaljobs.txt is operational and will gain jobs after the B lock (C, retries) and had priorities
   changed (B before A) at 07:55 PT. Its locked copy's hash is in PARTD_B_FREEZE.sha256.
4. Clock note: the B and C addenda say "~07:50 PT" and "~08:05 PT"; both were written ~07:35-07:40 PT (notebook
   timestamps from `date` are authoritative). Both were still locked before any model output on their data.
5. 2026-11-01, operational (no effect on any score). At the 11:17 PT pause the eval queue (scripts/evalq_p.py,
   unit mom-partd-evalqp) was believed stopped but was not: scripts/stop_queue.sh evalq matches only evalq.py. It kept
   evaluating through the pause. station-b rebooted ~13:00 PT; jobs that hit it failed (copy/serve) and were
   re-run in full after the unit was restarted at 13:34 PT. Generation is greedy and batch-invariant, so a re-run
   gives the same outputs. No partial generation was scored.
6. 2026-11-01 17:30 PT. Raw seed-mean C scores were looked at (Stephen asked for progress) before
   scripts/partd_c_analysis.py was written (~18:20 PT). The script's tests are exactly those in PART_D_C_ADDENDUM.md;
   its only interpretive choice (C4 "other two benchmarks" = two separate tests) is stated in its docstring, and it
   was hashed (artifacts/PARTD_ACDE_ANALYSIS.sha256) before its first run.
7. 2026-11-02/03, F, operational. The F queue was paused 22:00 PT 11-02 at Stephen's request (he needed the 6000s)
   and resumed 19:20 PT 11-03 with the same frozen files (sha256sum -c OK at resume and before analysis).
   base-qwen3-8b-think-s1 was mid-run at the pause: items already written were kept and the rest generated after
   resume (generate.py skips ids present). Each item was generated exactly once with the job's settings; under
   sampling the per-item draws after resume are not the ones an unbroken run would have made, which does not
   change the distribution. No output was looked at between pause and resume. All 18 F jobs ran (the addendum and
   notebook said 19; the jobs file has 18). Analysis run once, 2026-11-04 04:27 PT, all seeds present.

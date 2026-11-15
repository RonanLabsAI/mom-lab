# Part B runbook (started 2026-10-30 08:40 PT; Stephen approved Part B + keep pooled seeds)

Plan of record: `content/DRAFT_part_b_plan.md`. This file = decisions made + live state + remaining steps.

## Live state (as of 2026-10-30 08:45 PT)
- Fleet claim `mom-partb-20261030`: both Stations (GB300 + RTX PRO 6000 each). Sparks NOT claimed, NOT used.
- Station dailies (glmf-daily, nemotron-super-rtx, dsv41-daily, dsv41-daily-sidecar) were ALREADY DOWN when claimed
  (left down by the dsv41-frontier session). On release: restore with `scripts/restore_services.sh` station lines.
- Queue `partb` (16 jobs, `configs/queues/partb.jsonl`) running on GB300 slots only (station-a:1, station-b:1),
  launched `setsid nohup python3 -P scripts/trainq.py partb station-a:1,station-b:1`. Events:
  `logs/queue/partb.events.log`. 6000s are deliberately left free for the builders/teacher generation below.
  Started 08:39 PT: pt4-pooled-lr2e-5 (left), pt4-pooled-lr2e-5-s1 (right); ~3.5 h each.
- DISK IS TIGHT: station-a / 69 GB free, station-b / 31 GB free (runs dir 360/330 GB). Before more 4B
  checkpoints land: rsync Part A runs not needed for Part B to TR `/srv/release-data/mom-lab-archive/runs/<host>/` (4.8 TB free),
  verify (sizes + sha256 of model.safetensors), then delete those on the Station (stat first). Keep every run named
  in "Arms" below.

## Live state (as of 2026-10-30 10:00 PT)
- DONE: disk archive + cleanup (395/321 GB free); test para (590) + mixed (1,350) sets built and validated; quantised
  models on both Stations; router frozen (artifacts/router_e5lr.pkl); cv_routes confirms routes; memory report
  (results/analysis/memory_report.json: H2 ratio 0.568 bf16, 0.631 AWQ); partb_analysis.py written + dev dry run;
  PART_B_PREREG.md DRAFT written.
- RUNNING: trainq partb (GB300s; pt4-pooled s0/s1 finish ~12:15 PT). Teacher servers on the 6000s: mom-serve-8690
  (left, 1.7B + code/sql/tools LoRA, max-num-seqs 512) and mom-serve-8691 (right, 4B + sql/tools LoRA); clients
  make_distill.py --size 17 / --size 4 (resumable, data/mixtures/N14M/distill/*.partial). ETA at 6000 speed:
  1.7B ~14:45 PT, 4B ~19:50 PT (power-capped 300 W Max-Q; LoRA compute-bound).
- evalq partb (step partb, 6000s, EVAL_SPARKS=0) WAITING for a free 6000 (<2 GB); extras: pt17-pooled s1/s2 (para
  missing), 4B/8B AWQ + FP8, AWQ-4B + sql/tools adapters (+lora= syntax; AWQ served fp16).
- PLAN at ~11:55 PT: restart trainq with station-a:1 only BEFORE the right pooled run finishes (trainq refills a
  freed slot in the same poll). Hand station-b (GB300 + 6000) to Stephen: stop the 4B teacher client + mom-serve-8691
  (resumable). Consider the left GB300 for teacher work (--only base shard on a GB300 server, --only lora on the
  6000s) before resuming training. After any sharding run make_distill once more to assemble (old clients assemble
  from their own partial only). Then append pt17-distill-lr2e-5{,-s1,-s2} / pt4-distill-lr2e-5{,-s1,-s2} to partb.jsonl
  (data distill/17.jsonl, distill/4.jsonl; args as pooled).

## Live state (as of 2026-10-30 22:10 PT)
- Queues run as USER SYSTEMD UNITS (a T3 Code session scope killed the setsid'd ones at 16:53 PT):
  `systemctl --user status mom-partb-trainq mom-partb-evalq`. Stop: scripts/stop_queue.sh trainq|evalq (now stops the
  unit too). Relaunch pattern: `systemd-run --user --unit=mom-partb-trainq --collect --working-directory=$PWD
  -p StandardOutput=append:$PWD/logs/queue/partb.trainq.out -p StandardError=append:$PWD/logs/queue/partb.trainq.out
  python3 -P scripts/trainq.py partb station-a:1,station-b:1` (evalq: same, plus -E EVAL_STATION_SLOTS=...,
  -E EVAL_SPARKS=0, -E EVAL_DOMAINS=math,code,sql,tools,translate,general,mixed,para, args logs/queue/evalq.args.1607
  from field 4). NEVER launch queues with setsid nohup from a T3 session again.
- Training: done pt4-pooled s0/s1/s2, pt4-distill s0. Running pt4-distill s1 (left), s2 (right), ~4.2 h each (~02:15 PT).

## Live state (as of 2026-10-30 16:10 PT)
- Claims: mom-partb-20261030-left AND mom-partb-20261030-right. Stephen's ds41/ds41-sidecar on right are STOPPED
  (not removed); on Part B release run `docker start ds41-sidecar ds41` on station-b if he still wants them.
- trainq partb (PID 1867204) on station-a:1,station-b:1: pt4-pooled-lr2e-5-s2 (left), pt4-distill-lr2e-5 (right).
- evalq partb (PID 1867844) on station-a:0,station-b:0, dev, extras as logs/queue/evalq.args.1607.
- Distill data on both Stations, sha256 verified.

## Live state (as of 2026-10-30 13:45 PT)
- Claim `mom-partb-20261030-left` (station-a only). station-b lent to Stephen since 12:41 PT: do not touch.
- Distill data DONE for both sizes, synced to station-a (sha256 checked): distill/4.jsonl 345,279; distill/17.jsonl
  345,684. Teacher server mom-serve-8692 removed. NOT yet on station-b (rsync + sha256 on reclaim).
- trainq partb RUNNING on station-a:1 (relaunched 13:36 PT): pt4-pooled-lr2e-5-s2 started 13:37 PT (~3.5 h).
  Queue now 22 jobs (pt17-distill x3 appended). State file still lists pt4-pooled-lr2e-5-s1 as running (right
  not in slot list); it is reaped when right rejoins.
- evalq partb RUNNING on station-a:0 (dev).
- On reclaim of right: new claim, rsync both distill files, relaunch trainq + evalq with both slots (step 5 of handoff).

## HANDOVER (Stephen, 2026-10-30 09:05 PT)
- Stephen needs ONE Station (GB300 + 6000) for ~30 min at a logical point = after the current pt4-pooled runs finish
  (~12:10 PT). Plan: when pt4-pooled-lr2e-5-s1 (station-b) finishes, stop trainq (scripts/stop_queue.sh trainq),
  relaunch partb with slots station-a:1 only, stop any mom-serve-* on station-b GPU0, tell Stephen station-b
  is free; resume both slots when he says done.
- 11:52 PT: trainq + evalq STOPPED; evalq relaunched with EVAL_STATION_SLOTS=station-a:0 only. trainq is left
  stopped on purpose (state file still lists both pt4-pooled runs as running; a relaunch reaps them from done.txt).
  When pt4-pooled-lr2e-5 (left) finishes: the left GB300 hosts the 4B distill teacher (mom-serve-8691 moves there;
  clients `--only base --tag gb` and `--only lora --tag gbl`; old 4B client pid 617720 killed first). When
  pt4-pooled-lr2e-5-s1 (right) finishes: stop mom-serve-8691 on the right if still there; station-b goes to
  Stephen. After teacher + Stephen: relaunch `trainq.py partb station-a:1,station-b:1`, evalq with both 6000s.
- 12:41 PT: fleet claim narrowed. Released mom-partb-20261030 and claimed mom-partb-20261030-left (station-a only). Re-claim right (new id) when Stephen says done.
- Archive went to /srv/release-data/sentinel/mom-lab-archive (not /srv/release-data/mom-lab-archive: /data is root-owned, sudo needs a password).

## Decisions (freeze these into PART_B_PREREG.md)
- Primary comparator = pooled from the SAME init as the library (post-trained). The existing 4B pooled was Base-init
  (b4-pooled, a confound vs post-trained 4B LoRA); so Part B trains pt4-pooled x3 seeds. b4-pooled s0 = secondary only.
- Seeds: 3 for every trained arm.
- Adapter recipe: plain LoRA r16 lr2e-4 (Part A: replay adds nothing to LoRA at 4B). Full-FT specialists: 20% replay.
- Routes: 1.7B -> code, sql, tools; 4B -> sql, tools (confirm with scripts/cv_routes.py on dev before freezing).
- Router on test: frozen multilingual-e5-small + LR (route_step4.py); oracle routing reported as secondary.
- Distilled baseline (H3): teacher = the adapter library (s0) routed by training-data domain; non-routed domains
  answered by the base itself; prompts = data/mixtures/N14M/pooled.jsonl (353,238); greedy, 4096 cap, drop truncated;
  student trained like pooled (same init, lr2e-5, 2 epochs), 3 seeds per size. Compare vs the adapter library
  (same memory).
- Multilingual fix for the 4B mixed-skill gap: NOT in the confirmatory run (would add data pooled does not get).
  Prereg predicts H1 at 4B may fail on mixed.
- Quantised single models (eval only): official Qwen3-4B/8B AWQ (4-bit) and FP8 (8-bit); plus AWQ-4B + LoRA adapters.
  Run them on dev first so pass bars use dev numbers.
- Margins: non-inferiority -2.0 points on the set composite; "beats" = paired-bootstrap 95% CI lower bound > 0.
- Freeze: PART_B_PREREG.md + sha256 manifest of scripts/, harness/, train/, configs/ (repo has no commits; a git
  commit needs Stephen's per-commit OK). Then one test pass.

## Arms (runs needed on test)
- 1.7B: base-qwen3-1.7b; pt17-pooled-lr2e-5{,-s1,-s2} (or lr1e-5 x3 if better on dev: pick by dev composite);
  pt17-replay-{code,sql,tools}-r20-lr2e-5{,-s1,-s2}; pt17-lora16-{code,sql,tools}-lr2e-4{,-s1,-s2}; pt17-distill x3.
- 4B: base-qwen3-4b; pt4-pooled-lr2e-5{,-s1,-s2}; pt4-lora16-{sql,tools}-lr2e-4{,-s1,-s2}; pt4-distill x3;
  b4-pooled-lr2e-5 (secondary).
- 8B: base-qwen3-8b. Quantised: 4B/8B AWQ + FP8; AWQ-4B + LoRA.

## Remaining steps
1. Archive + clean Station disks (above).
2. Build test-split paraphrase set (scripts/build_para.py, add a --split test option; same paraphraser) and
   test-split mixed set (scripts/build_mixed.py, --split test; Qwen3-32B translator) on a 6000; validate like dev
   (validate_mixed.py, gold_check_mixed.py).
3. Teacher generation for distillation (6000s, vLLM multi-LoRA), write data/mixtures/N14M/distill/{17,4}.jsonl,
   append pt17-distill / pt4-distill jobs to partb.jsonl.
4. Dev evals: new seeds, quantised models, distilled models (evalq on 6000s).
5. cv_routes + memory report (weights + adapters + router + KV at fixed batch/context; check against vLLM's
   reported model memory).
6. Write PART_B_PREREG.md with hypotheses H1-H3, bars, arms, routes; sha256 manifest; then the single test pass.
7. Analysis: system_compare on test (core / para-test / mixed-test), paired bootstrap, "inconclusive" when a CI
   straddles a bar. Update content drafts (still DRAFT ONLY, private).
8. Release claim, restore Station dailies, sync to the desktop mirror.

## Test pass (only after the freeze manifest is written and its sha256 is in the notebook)
```
cd ~/projects/RonanLabs/mom-lab && L=$PWD
systemd-run --user --unit=mom-partbtest-evalq --collect --working-directory=$L \
  -E EVAL_SPLIT=test -E EVAL_STATION_SLOTS=station-a:0,station-b:0 -E EVAL_SPARKS=0 \
  -E EVAL_DOMAINS=math,code,sql,tools,translate,general,mixed,para \
  -p StandardOutput=append:$L/logs/queue/partbtest.evalq.out -p StandardError=append:$L/logs/queue/partbtest.evalq.out \
  python3 -P scripts/evalq.py partb partbtest --extra $(grep -v '^#' configs/partbtest_extras.txt)
```
Expect 50 DONE lines in logs/queue/eval-partbtest.events.log; then
`.venv/bin/python -P scripts/partb_analysis.py --split test --strict` (and --router oracle, secondary).

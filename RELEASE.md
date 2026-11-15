# Release checklist and state

**Release-prep update, 2026-11-14 PT (W12b):** Stephen approved release of code,
data and adapters after scrub. Code is now Apache-2.0 (`LICENSE`); MedCalc-derived
material is CC BY-SA 4.0 (`DATA-LICENSE.md`). The tracked tree has been scrubbed;
see `RELEASE-CHECKLIST.md` and `release/SCRUB-MANIFEST.json` for the current state.
The historical W8 inventory below is preserved as a dated record, including its
then-pending decisions. No repository, dataset or adapter has been published.
The private git history must not be pushed as a public release.


State as of 2026-11-13 PT (W8, branch `feat/w8-mom-v2`). Nothing has been released. Publishing the repository,
the data and any weights is Stephen's call.

## What goes public

| Item | Ships? | State |
|---|---|---|
| Code: `harness/`, `harness_d/`, `train/`, `scripts/` (builders, queues, analysis) | yes | in git |
| Configs, prereg/addenda (`content/PART_*`), deviations log, `GOAL.md`, `CONTRACT.md` | yes | in git |
| Freeze manifests `artifacts/*.sha256` | yes, after the hostname scrub below. Their hashes then go in the already-public `RonanLabsAI/dgx-station-frontier` repo, so they are visible outside the lab. | in git |
| `results/**/scores.json`, `results/**/metadata.json`, `results/analysis/*.json` | yes, after the endpoint scrub | in git (93 MB) |
| Lab notebook | yes, after the scrub | in git |
| Eval files: METAR, Pre-Flight, MedCalc (`data_partd_c/eval/`) | yes, with the licences in `LICENSES.md` | not in git (8.4 MB); rebuild is byte-identical |
| Part A-C eval and paraphrase sets (`data/eval`, `data_partc/`) | only items whose source licence allows it. Re-check every "recorded" licence first. | partly in git (`data_partc/`) |
| Training mixtures | builders yes, files no (size; OPUS-100 licence unresolved) | not in git |
| Model generations (`results/**/gen/`, ~2.8 GB) | later, as a Hugging Face dataset | not in git |
| LoRA adapters, fine-tuned weights | only if Stephen approves (MedCalc adapters CC BY-SA 4.0) | Station and Threadripper archives |
| `data_partd_c/raw/notam_probe/` | **never** (no licence) | gitignored |
| `logs/` | no: operational logs with hostnames, IPs and paths. Only consult notes, if any, after review | in git locally only |

## Locks: `scripts/release/verify_freezes.sh` (main checkout, 2026-11-13 PT)

```
OK    artifacts/FREEZE_MANIFEST.sha256  local ok=111/111  remote(not checked here)=84
OK    artifacts/PARTC_FREEZE.sha256  local ok=29/29  remote(not checked here)=2
OK    artifacts/PARTD_ACDE_ANALYSIS.sha256  local ok=4/4  remote(not checked here)=0
OK    artifacts/PARTD_B_ANALYSIS.sha256  local ok=1/1  remote(not checked here)=0
FAIL  artifacts/PARTD_B_FREEZE_fix1.sha256  local ok=1/2  remote(not checked here)=0
      content/PART_D_DEVIATIONS.md: FAILED
FAIL  artifacts/PARTD_B_FREEZE.sha256  local ok=62/65  remote(not checked here)=0
      data_partd/eval/tools/bfcl/test.jsonl: FAILED
      configs/partd_evaljobs.txt: FAILED
      configs/partd_autojobs.txt: FAILED
OK    artifacts/PARTD_C_FREEZE.sha256  local ok=38/38  remote(not checked here)=0
OK    artifacts/PARTD_F_FREEZE.sha256  local ok=58/58  remote(not checked here)=0
FAIL  artifacts/PARTD_FREEZE.sha256  local ok=8/9  remote(not checked here)=16
      configs/queues/partd.jsonl: FAILED
OK    data_partd_c/SHA256SUMS  local ok=10/10  remote(not checked here)=0
```

All five local mismatches are expected and already logged:
- `data_partd/eval/tools/bfcl/test.jsonl`: deviation 1 removed two over-length prompts. The original is kept as
  `test.locked_v1.jsonl`, and it matches the PARTD_B_FREEZE hash (checked). The new file matches
  `PARTD_B_FREEZE_fix1.sha256` (checked).
- `content/PART_D_DEVIATIONS.md`: entries 4 to 7 were appended after fix1 was written (notebook 2026-11-01).
- `configs/partd_evaljobs.txt`, `configs/partd_autojobs.txt`: operational jobs files. Deviation 3 says they keep
  changing after the B lock.
- `configs/queues/partd.jsonl`: the 18 C training runs were appended after the Part D freeze (notebook
  2026-11-01, 07:40 PT).

Not checked from the Threadripper: 102 remote entries (weights and adapters on the Stations, in two spellings,
`station:<host> <sha> <file>` and `<sha>  station-a:<file>`). Checking them means hashing the files on each
Station over `ssh -o BatchMode=yes`, read-only. That is an open item; it needs no GPU.

## Data rebuild check (2026-11-13 PT)

Builders `build_metar.py`, `build_medcalc.py` and `build_preflight.py` ran offline in a scratch copy of the tree
(raw inputs copied there, nothing written in the main checkout). Result: 8/8 files byte-identical to
`data_partd_c/SHA256SUMS` (all six eval files plus `metar.jsonl` and `medcalc.jsonl`). The two replay mixtures
were not rebuilt, because they need the Part A training mixtures (`data/mixtures/N14M/`).

## Public-safety check: `scripts/release/check_public_safe.py`

Rules: FlyCurrent or RetenzAI in a file path (cannot be exempted), FlyCurrent or RetenzAI paths in content, the
PI's personal email, SSN-, MRN-, DOB- and US-phone-shaped strings. Run over every tracked file plus
`data_partd_c/eval/` (2,063 files). Raw hits before the allowlist, all triaged by hand:

| File | Rule | Hits | Verdict |
|---|---|---|---|
| `data_partc/eval/para/mbpp_pp/test.jsonl` | ssn | 30 | item pp2-mbpp-427: test strings for a date-format function such as `'2024-02-299'`. Not identifiers. |
| `data_partc/eval/para/bird_pp/test.jsonl` | phone | 3 | item pp2-bird-dev-1368: the 555 number in the public BIRD `student_club` database. Fictional. |
| checker, its tests, the W8 plan | several | 14 | the rules naming or testing themselves |
| `data_partd_c/eval/` (MedCalc, METAR, Pre-Flight) | any | 0 | clean |

Reviewed exceptions live in `scripts/release/public_safe_allow.txt`. With them, the checker exits 0. MedCalc notes
are de-identified case reports from PubMed Central (via PMC-Patients), MedQA vignettes and TREC cases. They are
public, but they do describe real patients, so they ship only inside the CC BY-SA dataset and are never
paraphrased into new public material. The checker is a screen, not a PHI audit. The source rules in
`docs/eval-set-plan.md` still apply.

## Scrub list before anything goes public

Counts over tracked files (`git grep`, 2026-11-13 PT):

| Pattern | Files | Lines | Where |
|---|---|---|---|
| `10.0.0.x` (fleet LAN IPs) | 864 | 889 | mostly the `endpoint` field of `results/**/metadata.json` and `scores.json` (845 files); also configs, scripts, logs, notebook |
| `station-a` / `station-b` | 65 | 1,300 | manifests, scripts, configs, notebook, logs |
| Tailscale `100.x` address | 2 | 2 | GOAL.md, scripts/sync_to_alien.sh |
| Station serial hostnames (`node-fba00626.invalid...`) | 3 | 5 | docker-inspect dumps in logs/ (these also hold full container configs: never release) |
| `/opt/release-user` | 8 | 30 | scripts and docs |
| `/srv/release-data/sentinel` | 6 | 7 | notebook, archive script |
| `sjronan` | 5 | 20 | GitHub handle in docs |
| "Threadripper" | 8 | 10 | prose |

Plan: endpoints become `http://<eval-host>:<port>/v1` in a release copy (results files are receipts, so the
originals stay unedited in the private repo; the release copy carries a mapping note). Hostnames become
`station-a` / `station-b`. Home paths become repo-relative. The manifests are rewritten into a release copy with
the same hashes and generic host names. Hashes cover file contents, not hostnames, so they still check.

## Approval items for Stephen

1. Make the repo public (under `RonanLabsAI`), and pick a code licence (suggest Apache-2.0, to match Qwen3).
2. Release the eval files and generations on Hugging Face (CC BY-SA 4.0 for anything derived from MedCalc).
3. Release adapters or weights at all, and under which terms.
4. Publish the freeze-manifest hashes in `RonanLabsAI/dgx-station-frontier`.

## Open items

- Hash-check the 102 remote manifest entries on the Stations (read-only).
- Re-check every "recorded" licence in `LICENSES.md`, and resolve OPUS-100 and LiveCodeBench.
- Build the scrubbed release copy and run `check_public_safe.py` on it.

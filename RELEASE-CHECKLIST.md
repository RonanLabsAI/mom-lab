# METAR / MoM v2 release preparation

Status: 2026-11-14 PT, W12b. Stephen approved public code, data and adapter release
after scrub. This branch prepares the local tree only; nothing was pushed,
merged, published, submitted or uploaded. Base: `feat/w8-mom-v2` at `cbf58ab`.

## Public export (W19, 2026-11-14 PT)

- Combined W12b's scrub (`08f409b`) with the W12 paper commit `519045d` (Part G
  deviation 2 disclosure). The tree was published as a single new commit with no
  private history. The private branch tip it was exported from is recorded in the
  public commit message, and every original freeze digest stays in
  `release/FREEZE-PROVENANCE.json`.
- Operational `logs/` removed from the export: queue events, generation logs,
  Docker inspect dumps, host summaries, claims and consult transcripts. Only
  `logs/analysis/` (analysis outputs the notebook cites) is kept. Excluded files
  keep their scrubbed hashes in `release/SCRUB-MANIFEST.json`
  (`excluded_from_public_export`).
- Independent privacy scan of the export tree and its commit metadata found no
  private IPv4/tailnet addresses, home or data paths, usernames, PI email, tokens or keys,
  FlyCurrent or RetenzAI material or private patient data. Clinical text is the public
  MedCalc-Bench v1.2 (CC BY-SA 4.0); email strings are the public BFCL fixtures.
  Residual prose that was rewritten: the desktop's host name, the private mirror
  and workspace paths, and one test fixture path.
- Kept on purpose: "Threadripper" as a product name (the lab's public fleet page
  names it); Spark run ids `spark1`-`spark3` and Station aliases `station-a/b`
  are ordinal labels with no address.
- Hugging Face upload: done 2026-11-14 ~17:55 PT (W21), public under the RonanLabs
  organization: https://huggingface.co/datasets/RonanLabs/mom-lab-metar-medcalc and
  https://huggingface.co/RonanLabs/mom-lab-qwen3-4b-adapters . Details below.

## Prepared

- [x] Project code under Apache-2.0 (`LICENSE`).
- [x] MedCalc-derived data notice and attribution under CC BY-SA 4.0
  (`DATA-LICENSE.md`); conservatively the same terms for MedCalc-trained adapters.
- [x] Neutral nonresolving endpoint placeholders, generic Station aliases and
  storage roots. GPU identifiers are placeholders; configure real devices before
  any rerun. `MOM_MAIN` optionally locates a separate main data checkout.
- [x] Scrub manifest with original and release SHA-256 per changed file:
  `release/SCRUB-MANIFEST.json`. No original private addresses are in the mapping.
- [x] Every original freeze digest preserved, with per-entry baseline/release
  status in `release/FREEZE-PROVENANCE.json`. A release hash certifies scrubbed
  bytes, not the bytes used by the frozen experiment. Original bytes remain
  verifiable against the baseline in the private history.
- [x] All tracked benchmark files unchanged; numeric JSON leaves in changed
  result files unchanged. No experimental scores, counts or result arrays edited.
- [x] CPU-only tests and privacy screens; see `release/VALIDATION.md`.

## Stephen's remaining publication steps

- [ ] Choose the public repository name, owner and visibility. Export the reviewed
  tree into a **new repository without the private commit history**. This branch
  descends from commits containing the original private operational details;
  changing the current tree does not scrub those commits. Do not push this branch
  or its ancestors into the public repository.
- [ ] Exclude `logs/` (especially Docker inspect/config dumps) from the public
  export even though tracked log strings were scrubbed. Exclude `.git`, virtual
  environments, caches, raw downloads, NOTAM probes, local fleet state and any
  unreviewed build output. Keep the release manifests and the experiment's result
  JSONs. Choose which dated internal planning/consult notes are useful to publish.
- [x] Create the Hugging Face dataset repository and upload reviewed METAR,
  Pre-Flight and MedCalc artifacts with source revisions, original notices,
  checksums, transformation descriptions and dataset cards. Mark MedCalc-derived
  data CC BY-SA 4.0 with attribution; retain other datasets' source terms.
  Done (W21): `RonanLabs/mom-lab-metar-medcalc`, eval and train files byte-identical
  to `data_partd_c/SHA256SUMS`, METAR 5-shot variants, the six 4B adapters' test
  generations, per-part licences on the card and a root `SHA256SUMS`. Held back:
  replay mixtures (OPUS-100 samples), MedCalc few-shot prompts (exemplars from a
  repository without a licence file), raw downloads, NOTAM probes.
- [x] Screen ignored generations and rebuilt/downloaded data separately before
  HF upload; the tracked-tree scan does not cover them. MedCalc contains public,
  de-identified clinical case reports and vignettes. Confirm that only the
  documented public sources are present; no lab/private patient data are allowed.
  Done (W21): pattern screen of every uploaded file found no private address, host,
  path, user name, email, token or proprietary material. All 11,227 MedCalc notes and
  9,827 training explanations occur verbatim in the MedCalc-Bench CSVs, which match
  revision `83a1a56d` byte for byte. The 18 generation files re-score to the
  published per-item results.
- [x] Upload approved adapters with model cards, exact base-model revisions,
  training-source inventory, checksums, evaluations and limitations. MedCalc-trained
  adapters retain CC BY-SA 4.0 attribution plus the Qwen3 base notices. Base weights
  are not relicensed by the code's Apache-2.0 licence.
  Done (W21): `RonanLabs/mom-lab-qwen3-4b-adapters`, METAR seeds 0-2 (Apache-2.0)
  and MedCalc seeds 0-2 (CC BY-SA 4.0) on `Qwen/Qwen3-4B` @ `1cfa9a72`, whose shard
  hashes match the training copies. Adapter weights are byte-identical to the
  training outputs; only the base path in `adapter_config.json` and
  `train_args.json` changed. Other adapters (1.7B, replay, Parts A-C) are not uploaded.
- [ ] Verify the 96 unavailable external freeze entries and six environment
  receipts on the original hosts/archives if certifying them. No SSH or GPU work
  was performed here. Rebuild or copy the 94 unavailable ignored local artifacts
  before certifying those entries. These counts are not successful verifications.
- [ ] Review the seven preexisting historical hash mismatches (three original
  gate entries superseded by dev1, two operational job configs, the appended
  deviation log and the queue). Their baseline hashes and status are recorded;
  the release scrub did not cause them.
- [ ] Resolve OPUS-100 and the unspecified LiveCodeBench licence version before
  redistributing affected data or derived artifacts. Exclude NOTAM probes.
  Re-check other upstream notices for the exact export selection (`LICENSES.md`).
- [ ] If publishing freeze hashes elsewhere, publish hashes of the clearly
  labeled release manifests alongside the original experiment digest records.

## Residual screened patterns

Private IPv4/CGNAT addresses, original Station/serial hostnames, personal home
roots and private data-storage roots: zero outside the scanner's generic rules.
19 email occurrences in operational logs were neutralized. Twelve email strings
remain in the unchanged public BFCL paraphrase benchmark: they are task/answer
fixtures, not the PI's contact details. Public programming benchmarks also contain
IP-shaped strings and synthetic identifier/phone/date examples; their bytes are
preserved and exceptions are recorded by `check_public_safe.py`.

FlyCurrent and RetenzAI name occurrences are exclusion-policy text, scanner rules and
synthetic tests, not proprietary project material. The source scanner's reviewed
exceptions are in `scripts/release/public_safe_allow.txt`; the audit inventory
is `release/PUBLIC-AUDIT.json`. Public author attribution and the existing public
GitHub handle remain. No matching live token/key or PI address was found. This
pattern screen is not an independent clinical de-identification audit.

## Verification commands (CPU, offline)

```sh
python3 scripts/release/prepare_public.py --verify
python3 release/audit_public.py
bash scripts/release/verify_freezes.sh
python3 -m pytest tests harness/tests harness_d/tests -q
```

The last command needs both `harness/requirements.txt` and
`requirements-gate.txt`; the local gate-only and full-harness environments were
used separately because neither contains all dependencies. See validation counts.
`prepare_public.py --seal` updates release hashes after reviewed portability or
release-document edits while retaining the baseline and frozen digests; review
those changes before sealing. It does not re-run or re-certify experiments.

`release/RELEASE-SHA256SUMS` inventories the final reviewed bytes. Recheck it with
`sha256sum -c release/RELEASE-SHA256SUMS`. Privacy scans are read-only by default;
use `release/audit_public.py --write` only to refresh the receipt, then review
and seal any changes. The checksum file itself is excluded from its inventory.

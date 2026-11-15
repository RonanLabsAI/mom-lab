# W12b validation record

2026-11-14 PT. CPU only. No serving, training, GPU probes, SSH, key-file reads,
pushes, merges, publication, submissions or uploads were performed.

## Tests

- Full harness environment: `python -m pytest harness/tests harness_d/tests -q`:
  **53 passed** (12.73 s).
- Gate environment: `python -m pytest tests -q`: **38 passed** (3.50 s), including
  five release-transform tests (private/CGNAT ranges, filename extensions,
  idempotence, numeric-result preservation and credential redaction).
- Total: **91 passed**, no skips in these final runs. The two environments were
  used because the gate venv lacks the full harness scorer dependencies and the
  full harness venv lacks jsonschema. Initial attempts in the wrong environment
  failed on missing dependencies; no scorer implementation was changed to make
  those failures pass. The tests themselves use synthetic local HTTP services.
- Python compilation checked the release scripts and portable gate scripts.

## Privacy and immutable results

`release/SCRUB-MANIFEST.json`: **976 existing files scrubbed**, including
**11 renamed log paths**. Replacement operations (content plus filenames):

| Pattern | Files | Operations |
|---|---:|---:|
| Private IPv4 / CGNAT | 873 | 912 |
| Station hostname | 77 | 1,477 |
| Personal home root | 42 | 132 |
| Private data root | 11 | 16 |
| Operational email | 10 | 19 |
| Serial hostname | 4 | 6 |
| GPU identifier | 9 | 29 |
| Assembled PI identity in checker/test source | 2 | 2 |

One file can match multiple rules. Additional new release documents, scripts and
portability updates are separate from this scrub count. Hashes of scrubbed files
are sealed after reviewed portability/document changes, retaining the original
baseline hashes.

`release/NUMERIC-PRESERVATION.json` compares every tracked result JSON against
`cbf58ab`: **893 files**, **2,692,256 numeric/bool/null leaves** unchanged;
**865** files have string-only release changes. All **12 tracked benchmark
files** remain byte-identical. All SHA-256 digest literals in **12 original
freeze manifests** remain unchanged. Literal addresses/paths/identifiers inside
operational strings were intentionally neutralized; that is not numerical-result
editing.

`python3 scripts/release/prepare_public.py --verify` validates release hashes,
benchmark hashes and absence of private-address/path/hostname/credential patterns.
`python3 release/audit_public.py` applies the existing source/PHI screens and
records reviewed exceptions and residual fixtures in `release/PUBLIC-AUDIT.json`.
Final screens report **zero refusals**. Twelve public BFCL email fixture strings
remain unchanged. The 34 proprietary-name hits are policy/scanner/synthetic test
text. SSN/phone/date-shaped benchmark fixtures and checker tests remain on the
reviewed allowlist (61 individual screen exceptions, including those 12 emails).
No real credential or additional private-patient source was identified; this is
a pattern screen, not an independent clinical de-identification audit.

## Frozen-run provenance

`release/FREEZE-PROVENANCE.json`: **458 entries**:

- 249 tracked entries matched the original expected digest at the baseline.
- 4 ignored local inputs and 2 external local inputs also matched their original
  expected digest when checked read-only.
- 7 mismatches predate release edits: three original gate code/test entries
  superseded by the dev1 manifest; two mutable operational job configs; the
  appended deviation log; and the queue.
- 94 ignored local artifacts are unavailable in this checkout.
- 96 external entries are unavailable here; no remote checks were attempted.
- 6 image/pipfreeze/driver receipts need the original environment to verify.

None of the unavailable entries is counted as a pass. Release SHA-256 values
attest current scrubbed bytes and do not replace the original experiment hashes.
The existing freeze verifier now checks the release receipts when present and
prints the historical-status inventory explicitly. All publication steps and
upstream-licence exclusions are in `RELEASE-CHECKLIST.md`.

`release/RELEASE-SHA256SUMS` seals the reviewed tree, excluding itself. Its entries
were checked with `sha256sum -c`; the release verifier also checks them. The
privacy audit is read-only by default; `--write` explicitly refreshes its JSON
receipt, after which reviewed changes must be sealed again.

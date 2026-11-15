# Eval-set plan: aviation and medical specialist library (lane L2, step 4)

Status: plan only, 2026-11-13 PT. Nothing here has been built or run. Spec: section 5 L2 of
`ronanlabs-hq/docs/superpowers/specs/2026-11-14-ronanlabs-research-strategy-design.md`.

## The rule

Public sources only. Never FlyCurrent or RetenzAI content, in any form: questions, explanations, lesson text,
generated items, or anything paraphrased from them. Never patient data. This holds for every eval set, receipt,
preregistration and figure. When we are unsure where a sentence came from, it does not go in.

Each set goes through three gates before anyone uses it:
1. **Source gate.** The source is on the table below, at a pinned revision, and its licence allows redistribution
   of derived items. A licence we have not read in a primary source (the LICENSE file or the dataset card at the
   pinned revision) counts as no licence.
2. **Screen gate.** `scripts/release/check_public_safe.py` exits 0 on the built files, with no new allowlist
   entries unless a person has reviewed them. It refuses FlyCurrent and RetenzAI paths, and SSN-, MRN-, DOB- and
   phone-shaped strings.
3. **Builder gate.** A deterministic builder in `scripts/` gives byte-identical output on a second run. It also
   writes the source URL, revision, raw-file hashes and licence into `manifest.json`, the way `scripts/partd_c/`
   already does.

## Candidate sets

Licence status: **verified** = read in a primary source; **likely** = US government work, or a licence tag seen
on a card, but not yet read at a pinned revision; **unverified** = not checked or unclear.

| Set | Source | Licence | Task | Target size | Notes |
|---|---|---|---|---|---|
| METAR v2 | Iowa Environmental Mesonet ASOS archive, `mesonet.agron.iastate.edu/cgi-bin/request/asos.py` (as in Part D-C) | US government observations; likely (IEM serves them freely) | raw METAR to 14-field JSON, plus a harder split: SPECI, RVR, remarks decoding, non-US ICAO stations | 1,000 test, 300 dev | Station-disjoint from the 2024 Part D-C set. Use a new year (2025) so no report text overlaps. |
| TAF | IEM TAF archive, `mesonet.agron.iastate.edu/request/taf.php` (CSV via `/cgi-bin/request/taf.py`; archive from 1996, with a known gap 2010-01-01 to 2015-04-04) | US government forecasts; likely | TAF to JSON: base forecast plus FM/TEMPO/PROB groups, each with time window, wind, visibility, weather, clouds; and "which group is in force at time T" questions | 1,000 test | Truth needs a deterministic decoder, checked against a second parser, as METAR was. Avoid the archive gap. |
| ACS knowledge | FAA Airman Certification Standards, e.g. Private Pilot Airplane FAA-S-ACS-6C (effective 2024-05-31), from faa.gov | US government work (17 U.S.C. 105); likely. FAA pages put responsibility for reproductions on the publisher, and commercial reprints carry their own notices, so build from the faa.gov PDF only. | Multiple choice and short answer written by an agent from ACS task elements, scored against the ACS text; plus "which ACS task covers this scenario" | 500 | This is a knowledge set (like Pre-Flight), the control for "size buys knowledge". |
| Handbook reasoning | FAA handbooks (PHAK FAA-H-8083-25, Airplane Flying Handbook, Aviation Weather Handbook) from faa.gov | US government work; likely. Same caveat as ACS: some figures inside are third-party, so use text only. | Computation items: density altitude, crosswind component, weight and balance, fuel planning, using the handbooks' own formulas and charts as described in text | 500 | This is the aviation analogue of MedCalc, and it tests whether "one worked example beats training" carries over. |
| POH / AFM | Manufacturer Pilot's Operating Handbooks | **Copyrighted by the manufacturers.** Do not use. | n/a | 0 | Flagged. If a POH-style task is wanted, write synthetic aircraft data sheets with invented numbers and say so. Never copy a real POH table. |
| NOTAM | Candidates from `data_partd_c/README.md`: `DEEL-AI/NOTAM` (card tag MIT, 13-class subject classification, English, 1K-10K rows); `AirsideLabs/notam-typed-decisions` (our Part D-C README recorded CC BY 4.0 with a LICENSE file; a web search on 2026-11-13 did not find it in Airside Labs' public dataset list); `AeroAIAviation/NOTAM-TAF-SPECI-METAR-ACARS_Decode` (Part D-C README recorded Apache-2.0) | all three **unverified** (needs a LICENSE file or card at a pinned revision, plus provenance of the NOTAM text itself) | classification first; decoding only if a licensed source has decoded truth | up to 1,000 | Knots and NOTAM-Evolve stay excluded: no licence. |
| MedCalc variants | MedCalc-Bench v1.2 (`ncbi/MedCalc-Bench` @ `83a1a56d`) | CC BY-SA 4.0, **verified** (card plus repo README) | (a) unnamed-calculator split: the note without the calculator's name, so the model has to pick it; (b) adapter + one-shot and adapter + thinking arms; (c) a calculator-held-out split (train on 40 calculators, test on the other 15) | reuse the 1,100 test items; (c) about 300 | Everything derived stays CC BY-SA 4.0. (a) answers the open question in `b-mom-program.md`: does the one-shot win only because the calculator is named? |
| Synthetic clinical calculations | Generated by our own code from published formulas (e.g. CKD-EPI, CHA2DS2-VASc, MELD) with invented values | ours (no source text) | numbers-in-a-template, no patient narrative | 500 | No narrative note, so there is nothing that could resemble a real patient. Every value is drawn at random; never sample from real records. |

## Contamination and dedup

- Exact and near-duplicate checks against every training mixture (`scripts/data/decontam.py` style: normalized
  13-gram overlap) and against the earlier eval sets.
- METAR and TAF: disjoint stations and a different year from anything trained on.
- MedCalc: drop training notes shared with test (as Part D-C did: 316 dropped).
- Generated items (ACS, handbook): the generator never sees the test answer key while writing train items. Train
  and test come from different ACS areas or handbook chapters.
- Record each dedup count in the builder manifest, as in `data_partd_c/manifest.json`.

## Spot-validation by Stephen

Agents draft and check the items. Stephen confirms the domain truth on a sample.

- Sample: 30 items per set, chosen at random with a fixed seed and stratified by category. Add 10 more from any
  category where the automatic check disagreed with the gold.
- What he checks: is the gold answer right, is the question unambiguous, is anything in it non-public? Three
  boxes per item.
- Time budget: about 20 minutes per set (about 40 seconds per item), and at most two sets a week, to stay inside
  the ~10-minutes-a-day triage target.
- Pass bar: 0 non-public items, and at most 1 wrong gold answer in 30. Above that, the set goes back for rework
  and a fresh sample. Results go in the set's `manifest.json`.

## Order

1. MedCalc variant (a), the unnamed-calculator split: cheapest, and it closes a known overclaim risk.
2. METAR v2 and TAF: the format-skill line, run after the JSON-schema gate (`preregs/`) has reported.
3. Handbook computations: tests whether the MedCalc reversal is a domain fact or a task-type fact.
4. ACS knowledge set: the knowledge control.
5. NOTAM, only after a licence is verified.

Each set gets its own preregistration (K2 card, public repo) before any model sees its test split.

Sources checked 2026-11-13 PT:
[IEM TAF request page](https://mesonet.agron.iastate.edu/request/taf.php),
[IEM taf.py documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py?help),
[FAA Aviation Handbooks & Manuals](https://www.faa.gov/regulations_policies/handbooks_manuals/aviation),
[FAA Private Pilot ACS change 1 (PDF)](https://www.faa.gov/sites/faa.gov/files/training_testing/testing/acs/private_airplane_acs_change_1.pdf),
[DEEL-AI/NOTAM card](https://huggingface.co/datasets/DEEL-AI/NOTAM/blob/main/README.md),
[Airside Labs on Hugging Face](https://huggingface.co/AirsideLabs/datasets),
[MedCalc-Bench paper](https://arxiv.org/abs/2406.12036); MedCalc licence read from the pinned card in
`data_partd_c/raw/medcalc/hf/README.md` and repo README.

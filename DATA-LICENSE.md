# Data and adapter licensing

Code written for this project is licensed under Apache-2.0; see LICENSE.
Third-party data, model weights and notices retain their original licences.

## MedCalc-derived material: CC BY-SA 4.0

Attribution: **MedCalc-Bench**, NLM/NCBI BioNLP Group and the MedCalc-Bench authors,
“MedCalc-Bench: Evaluating Large Language Models for Medical Calculations” (2024).
Source: https://huggingface.co/datasets/ncbi/MedCalc-Bench
and https://github.com/ncbi-nlp/MedCalc-Bench (paper: https://arxiv.org/abs/2406.12036).
The v1.2 source snapshot used by the builder is revision
`83a1a56d6b53f8d4c0e03f0ba9699becf524a595`; source CSV hashes and provenance
are recorded in `data_partd_c/manifest.json` and `data_partd_c/README.md`.

MedCalc-derived eval files (`data_partd_c/eval/medcalc/`), generated prompt sets,
training mixtures, generations and explanations are distributed under
**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**:
https://creativecommons.org/licenses/by-sa/4.0/ . The full legal code is at
https://creativecommons.org/licenses/by-sa/4.0/legalcode . Attribute the source, link this licence, indicate changes,
and distribute adaptations under the same licence. No endorsement is implied.
Preserve source attribution and notices, including the PMC-Patients lineage
(PMC-Patients/Open-Patients, CC BY-SA 4.0), when redistributing notes.

Changes made here: deterministic train/dev/test selection and note-overlap
removal; conversion to chat prompts and scorer reference records; calculator
format hints; explanation-plus-answer training targets; optional replay,
paraphrase or few-shot formatting as documented by the builders. No source
clinical notes were collected by this project. MedCalc includes public,
de-identified clinical case material; this notice does not claim it is
synthetic or independently certify de-identification.

For the approved release we conservatively apply CC BY-SA 4.0 to the
MedCalc-trained adapters and MedCalc-derived generations as well. Their model
cards must also retain the Qwen3 base-model attribution and Apache-2.0 notice.
Other adapters need their own training-source inventory; this notice does not
relicense all adapters or any base-model weights.

## Other datasets

METAR: NWS/FAA government observations served by the Iowa Environmental
Mesonet ASOS archive, https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py .
Pre-Flight: Airside Labs, MIT, source revision and attribution in
`data_partd_c/README.md`. Preserve the source notices on redistribution.

For Parts A–C and all other upstream datasets use `LICENSES.md`; Apache-2.0
for this project's code does not supersede those terms. OPUS-100 remains
unresolved and LiveCodeBench's licence version remains unverified: exclude
those data and affected redistribution artifacts until reviewed. NOTAM probe
downloads have no verified licence and must not be shipped.

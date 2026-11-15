# Data and model licences

Status of each entry:
- **verified**: checked against a primary source (dataset card or LICENSE file in our pinned download), 2026-11-13 PT.
- **recorded**: what the builder wrote into its manifest when it fetched the data (`scripts/data/*.py`,
  `scripts/partd_c/*.py`). It has not been re-checked yet.
- **unverified**: no licence found, or the finding is in doubt.

Re-check every "recorded" row before a public data release.

## Part D-C and D-F (niche skills)

| Dataset | Use | Licence | Status | Release notes |
|---|---|---|---|---|
| MedCalc-Bench v1.2 (`ncbi/MedCalc-Bench` @ `83a1a56d`) | train + eval | CC BY-SA 4.0 | verified (HF card `license: cc-by-sa-4.0`; repo README explains the share-alike chain from PMC-Patients) | Derived files stay CC BY-SA 4.0 with attribution. We treat the MedCalc adapters the same way (conservative reading). Notes come from Open-Patients: de-identified PMC-Patients case reports, MedQA vignettes (MIT), TREC CDS/CT (US government). |
| METAR reports, 2024 (Iowa Environmental Mesonet ASOS archive) | train + eval | US government observations (NWS/FAA), served freely by IEM | recorded (data_partd_c/README.md) | OK to redistribute with a source credit to IEM. AWC flight categories were used only for validation. |
| Pre-Flight (`AirsideLabs/pre-flight-06` @ `439d2d11`) | eval only | MIT, per the HF card and paper section 3.4 | recorded. The repo ships no LICENSE file. | Attribute as "Airside Labs, MIT". |
| Knots, NOTAM-Evolve | not used | none (all rights reserved) | verified absent (data_partd_c/README.md) | Never redistribute. `data_partd_c/raw/notam_probe/` stays out of every release. |

## Parts A-C (six domains)

| Dataset | Use | Licence as recorded | Status |
|---|---|---|---|
| GSM8K (`openai/gsm8k`) | eval | MIT | recorded |
| MATH-500 (`HuggingFaceH4/MATH-500`) | eval | MIT (PRM800K/MATH; card has none) | recorded, card silent |
| HumanEval+ / MBPP+ (`evalplus/*`) | eval | Apache-2.0 | recorded |
| LiveCodeBench (`livecodebench/code_generation_lite`, test6) | eval | "CC" on the card, version unclear | unverified |
| BIRD dev (dev_20240627 zip) | eval | CC BY-SA 4.0 | recorded |
| BFCL v3 (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) | eval | Apache-2.0 | recorded |
| FLORES-200 devtest (Meta release) | eval | CC BY-SA 4.0 | recorded |
| MMLU-Pro (`TIGER-Lab/MMLU-Pro`) | eval | MIT | recorded |
| IFEval (`google/IFEval`) | eval | Apache-2.0 | recorded |
| Spider (`xlangai/spider`), schemas `richardr1126/spider-schema` | train + Part D-B eval | CC BY-SA 4.0; schema file CC BY 4.0 | recorded |
| BIRD train (`xu3kev/BIRD-SQL-data-train`) | train | CC BY-SA 4.0 (the card itself has none) | recorded |
| sql-create-context (`b-mc2/sql-create-context`) | train | CC BY 4.0 (from WikiSQL BSD-3 + Spider CC BY-SA 4.0) | recorded |
| Magicoder-OSS-Instruct-75K | train | MIT (GPT-3.5 outputs; OpenAI terms apply) | recorded |
| OpenCoder opc-sft-stage2 | train | MIT (some subsets model-generated) | recorded |
| OpenMathInstruct-2 (`nvidia/OpenMathInstruct-2`) | train | CC BY 4.0 (Llama-3.1-405B outputs; Llama 3.1 licence terms on naming) | recorded |
| xLAM function calling 60k (ungated mirror `lockon/...`) | train | CC BY 4.0 | recorded. The original Salesforce release is gated, so check that its terms carry over to the mirror. |
| ToolACE (`Team-ACE/ToolACE`) | train | Apache-2.0 | recorded |
| Hermes function calling v1 | train | Apache-2.0 | recorded |
| Tulu 3 SFT mixture (`allenai/tulu-3-sft-mixture`) | train | ODC-BY 1.0; components carry their own terms | recorded |
| OPUS-100 (`Helsinki-NLP/opus-100`) | train | **none on the card**; the underlying OPUS corpora are mixed | unverified. Do not redistribute derived translation data until this is resolved. |

## Models

| Model | Licence | Status |
|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B / 14B / 32B, Base, AWQ, FP8 | Apache-2.0 | recorded (Qwen model cards); re-check the quantised repos |
| Our LoRA adapters and fine-tunes | Inherit Apache-2.0 from Qwen3, plus the terms of their training data (MedCalc adapters: CC BY-SA 4.0, conservative reading) | release approved 2026-11-14 PT; cards and upload pending |
| Paraphraser / translator models used to build the paraphrase and mixed-skill sets (Qwen3.8-27B-NVFP4, Qwen3-32B) | Model-generated rewrites of the benchmark items above; the item licences apply | recorded |

## Approved release terms (2026-11-14 PT)

Project code: Apache-2.0 (`LICENSE`). MedCalc-derived data and MedCalc-trained adapters: CC BY-SA 4.0 with attribution (`DATA-LICENSE.md`). Stephen approved code, data and adapter release after the scrub; repository creation and uploads remain unperformed. Upstream restrictions in this inventory still apply.

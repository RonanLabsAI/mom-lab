# Data / harness contract (v1, 2026-10-29)

Repo root: ~/projects/RonanLabs/mom-lab (Threadripper, CPU only). Python 3.12; use a venv at .venv (create if missing).

## Domains (ids are fixed)
math, code, sql, tools, translate, general

## Eval files
data/eval/<domain>/<benchmark>/{dev,test}.jsonl  — one JSON object per line:
- id: str, unique within benchmark
- messages: list of {"role","content"} — the user prompt only (system prompt optional); the harness adds nothing else
- reference: benchmark-specific gold (str or object)
- meta: dict (e.g. db_id + db path for SQL; test code + entry_point for code; target language for translate; instruction ids/kwargs for IFEval; function specs for tools; choices for MMLU-Pro)
- max_tokens: int (generation budget for this item)

dev/test split: fixed seed 20261029, 50/50 per benchmark (stratified by subject/category where the benchmark has one). The test split is never read during Part A.

Benchmarks per domain:
- math: gsm8k, math500
- code: humaneval_plus, mbpp_plus, livecodebench (problems released on/after 2025-06-01 only)
- sql: bird_dev (execution accuracy; databases under data/eval/sql/bird_dev/databases/)
- tools: bfcl (non-live single-turn AST categories: simple, multiple, parallel, parallel_multiple)
- translate: flores200 devtest, directions eng_Latn→{spa_Latn, deu_Latn, zho_Hans, jpn_Jpan} and {spa_Latn, deu_Latn}→eng_Latn
- general: mmlu_pro, ifeval

## Training pools
data/train/<domain>/pool.jsonl — {"id","messages":[user, assistant],"domain","source","n_tokens"} (n_tokens = Qwen3 tokenizer count of the full conversation)
data/train/<domain>/heldout.jsonl — 2,000 examples not in pool, for router fitting/calibration.
At least 60k examples or 40M tokens per domain where the source allows.

## Generations (harness output)
results/<step>/<run_id>/gen/<domain>/<benchmark>.<split>.jsonl — {"id","output","finish_reason","n_out_tokens","latency_s"}
results/<step>/<run_id>/scores.json — per benchmark score (0–100), per domain s_d (mean of its benchmarks), macro A, item-level correctness arrays, run metadata (model, revision, endpoint, hardware, settings, git hash, timestamp UTC).

## Generation settings (frozen defaults)
Qwen3 chat template with enable_thinking=False (chat_template_kwargs), temperature 0, top_p 1, seed 0, per-item max_tokens.

## v1.1 meta field layouts (fixed 2026-10-29 to match harness/score/*.py docstrings)
- math: reference = final answer string (gsm8k: the number; math500: the answer LaTeX). No meta needed.
- code: meta.test (HumanEval/MBPP-style test code appended after the solution; if it defines check(candidate), set meta.entry_point), meta.test_imports (optional list of import lines), meta.entry_point. LiveCodeBench: meta.io_tests = [{"input","output"}]; for functional-style problems also meta.func_name (inputs one JSON arg per line, output JSON; solution defines class Solution). reference = canonical solution if available.
- sql: reference = gold SQL; meta.db_path = sqlite path (absolute, or relative to repo root, or to data/eval/sql/bird_dev/databases/).
- tools: meta.function = list of BFCL function specs; reference = BFCL ground truth [{"fname": {"param": [allowed values]}}]; meta.category in simple|multiple|parallel|parallel_multiple.
- translate: reference = target sentence; meta.direction = "eng_Latn-spa_Latn" (src-tgt).
- general: mmlu_pro reference = answer letter, meta.choices; ifeval meta.instruction_id_list, meta.kwargs (list of dicts), meta.prompt = original prompt text.

# Part G (METAR schema gate) deviations log

Applies to `preregs/2026-11-14-metar-schema-gate.md` (read-only) and `artifacts/GATE_FREEZE.sha256` (read-only).

1. 2026-11-13 ~21:40 PT, before any gate output (results/partdG did not exist; no GPU used). Serving path and GPU
   selection, by controller decision: the production router (127.0.0.1:4000) is not restarted and not bypassed.
   - Router: requests go through a separate research-only LiteLLM, container `litellm-research` on 127.0.0.1:4001
     (/srv/release-data/services/litellm-research, same image digest as production
     `ghcr.io/berriai/litellm:v1.104.1@sha256:501909ec...`, same master key, only the two gate routes, global and
     route `drop_params: false`, no callbacks). The card's routes, route names and settings are unchanged; only the
     port and config path differ. `run_gate.py`: `ROUTER` -> :4001, `ROUTER_CFG` -> the research config.
   - GPU: the card said "GPU 0". Selected by UUID instead, after verifying by name on host-3cc546d0.invalid:
     index 0 = NVIDIA RTX PRO 6000 Blackwell Max-Q, `GPU-00000000-0000-0000-0000-2cd434a6b8f5` (47 MiB, no processes);
     index 1 = NVIDIA GB300, `GPU-00000000-0000-0000-0000-42943b2791b3` (glmf-daily, via CDI). The vLLM container gets
     `--gpus device=<6000 UUID>` and `CUDA_VISIBLE_DEVICES=<6000 UUID>`; memory checks query the UUID and verify the
     name; after every server start the launcher checks that the 6000 holds the model and that no new process appeared
     on the GB300, and stops the server if not.
   - `route_test.py` (fleet-ops `check-gpu-routing.py`) is hardcoded to :4000 and production names, so the router is
     tested with `/srv/release-data/services/litellm-research/bin/route_test.sh` instead.
   No change to arms, prompts, schema, decoding, scorer, tests or decision rule. The changed launcher is hashed in
   `artifacts/GATE_FREEZE_dev1.sha256`; every other frozen file still matches `GATE_FREEZE.sha256`.

2. 2026-11-13 ~22:05 PT, after the dev smoke and before any test-split output. Two findings.
   - Bug: `scripts/gate/smoke_check.py` read the run name from the wrong path component (`[-5]`, giving
     "partdG-smoke"), so `verdict()` treated every run as unconstrained and printed PASS without applying any schema
     rule; the launcher then wrote `logs/queue/gate.smoke.ok`. Fixed to `[-4]`, with a test
     (`tests/test_gate_smoke.py::test_run_name_from_path`). No other change to the script.
   - Applying the frozen rules with the fixed script: no truncation, unparseable, schema-invalid or fenced output in
     any of the six schema arms (0 of 180 items), so the whitespace rule and the canary do not fire. The null-branch
     rule fires for `gate-qwen3-8b-zs-schema` and `gate-qwen3-32b-zs-schema` only: `wind_dir_deg` is null for 5 of the
     30 gold items (calm wind, 00000KT) and both arms write 0 for all five. The frozen rule says stop, investigate,
     no automatic change. Investigation (CPU only, stored outputs): the unconstrained zero-shot 8B (Part C) and 32B
     (Part F) write 0 on all 268 calm test reports (0/268 null), while unconstrained 8B 5-shot writes null on 268/268;
     under the same grammar the four other schema arms write null (fs 5/5, thinking 4/5 and 5/5); xgrammar 0.2.7
     accepts every gold null (`xgrammar_gold_check.py`, 1,300/1,300). So the grammar's null branch is not lost: writing
     0 for calm wind is the zero-shot models' own reading of the prompt, and it is a real error the gate should score.
     Decision: the rule's purpose (detect a lost null branch) is shown not to apply; the main run proceeds with every
     arm unchanged. `logs/queue/gate.smoke.ok` was rewritten by hand to cite this entry. The smoke outputs are kept
     under results/partdG-smoke (dev only; no scores computed).
   The fixed script and this file are hashed in `artifacts/GATE_FREEZE_dev2.sha256`.

3. 2026-11-13 ~22:10 PT, operational. The first main launch generated all 1,000 parity-control items
   (`gate-qwen3-8b-zs-free`) and then stopped: `run_gate.py` ran `python -P -m harness_d.aggregate`, and `-P` removes
   the working directory from `sys.path`, so the module was not found. Fixed by passing `PYTHONPATH=<worktree>` to the
   aggregate call (`-P` kept). The parity run was aggregated by hand with the same command and checked with the
   launcher's own `parity()`: 1,000/1,000 outputs identical to Part C `base-qwen3-8b`, score 2.1 (C: 2.1). No schema
   arm had started. The queue was relaunched; generation resumes by item id, so the parity outputs are not
   regenerated. Hashes: run_gate.py in `GATE_FREEZE_dev1.sha256` and this file in `GATE_FREEZE_dev2.sha256` updated
   (earlier versions are in git history).

PI acceptance (2026-11-14): Stephen accepted deviation 2 above. The paper discloses it in
Appendix "Deviations from the locked files" entry (2): the frozen stop rule, why it fired (5 of 30 gold smoke
items), the stored-output investigation, the decision to continue with every arm unchanged, and the smoke-check
bug that produced the false PASS. Deviations 1 and 3 are disclosed alongside it in entries (1) and (3).

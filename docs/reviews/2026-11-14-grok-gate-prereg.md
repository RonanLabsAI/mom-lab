# Grok adversarial review of the METAR schema gate preregistration: disposition

Reviewer: Grok Build CLI (subscription, default model), 2026-11-13 ~21:15-21:30 PT, review only (no files changed; checked
with `git status`). Prompt: scratchpad `grok_gate_prompt.txt` (reproduced in the session log). Raw output:
`docs/reviews/2026-11-14-grok-gate-prereg-raw.md`. Verdict as given: "not a fair decisive test of the headline as
written". Each finding was checked against the files before acting.

| # | Severity | Finding | Checked | Disposition |
|---|---|---|---|---|
| 1 | blocker | 5-shot prompts (up to 4,586 tokens) + 4,096 max tokens exceed `--max-model-len` 8,192 -> HTTP 400, not retried | True: the card itself states 4,586; F used mml 32,768 | **Accepted.** Every arm now serves at 32,768. 32B route `max_input_tokens` raised to match. |
| 2 | blocker | Decision arms exclude thinking + schema and 32B controls; secondary arm cannot affect the headline | True | **Accepted (mostly).** Added `32b-think-schema-s0`, made both thinking + schema arms primary and part of G3's selection set, added unconstrained 8B/32B zero-shot and 5-shot controls on the gate server. Rejected the "schema pasted into the prompt" arm: the prompt already lists every key, type and allowed value, so it would duplicate the existing unconstrained arms. |
| 3 | blocker | Global `drop_params: true` could strip `response_format`; parity cannot see it; analysis ignores parity | True for the analysis; the drop risk is real and unverifiable until the route exists | **Accepted.** `check` now requires both routes with `drop_params: false` in the live router config. New canary: any non-truncated schema-arm output that is unparseable, schema-invalid or fenced, or an "HHMM accepted" score above strict, makes the arm and headline VOID. Parity below 1,000/1,000 makes the headline VOID. |
| 4 | major | "Matched" was one-sided; schema-ahead and wide CIs became Rewritten; selection effect on G3 unstated; tie-break unstated | True | **Accepted.** G3 is now TOST (lo > -2 and hi < 2 = equivalent; hi < 0 = schema ahead), selection effect stated, tie-break written down (list order). |
| 5 | major | Prompt and exemplars demand a ```json fence the grammar cannot emit | True (1,000/1,000 prompts contain the phrase) | **Accepted.** Schema arms use a raw-JSON prompt variant (two exact edits, deterministic builder, hashed). Fenced prompts kept for unconstrained arms. |
| 6 | major | Schema does not encode the prompt's stated rules; "training win" overclaims; key order not enforced | True | **Accepted for wording.** Stands wording limited to "schema v1 under xgrammar ... not within 2 points"; key-order claim removed. Encoding the decoding rules is rejected as out of scope: that is a hand-written decoder, which scores 100 by construction (gold comes from one). Stated in the card. |
| 7 | major | Possible xgrammar null-branch loss; early list closing; smoke rule not wired; unparseable not checked | Null-branch: tested, **not reproduced** (xgrammar 0.2.7 accepts all 1,300 golds, every null pattern, compact/spaced/indented and fixed-separator modes). Wiring: true | **Accepted.** Gold-membership check runs in `check`; smoke now calls `smoke_check.py`, fails on unparseable/invalid/fenced output or a lost null branch, sets the whitespace flag only by rule, and `main` refuses to start without a passed smoke. Test-set truncations stay "wrong" (harness rule) rather than voiding an arm: looping is a real property of the method and is reported. |
| 8 | major | Comparators served on another Station without the router; parity covers 8B only | True | **Accepted.** VOID unless parity is 1,000/1,000; G4 controls (8B and 32B, zero-shot and 5-shot) now run on the gate server. Thinking arms are compared with F's runs, labelled "different serving path, no reasoning parser". |
| 9 | major | Reasoning parser differs from F; `enable_in_reasoning` assumed; token counts may exclude reasoning | True | **Accepted.** `enable_in_reasoning: false` set explicitly; thinking-arm token counts flagged unreliable if they average under 300; parser difference labelled in G4. |
| 10 | major | Token wording mixes 8B and 32B; gap-closed denominator; Narrowed has a direction word; MISSING mapped to Narrowed | True | **Accepted.** 1,664 (8B 5-shot + thinking, 11.6x) and 937 (32B thinking, 6.5x) split; gap closed reported from 2.1, 83.2 and 88.9; Narrowed is direction-free; missing arms give VOID. |
| 11 | minor | `parse_fail_rate` not reported; lenient score only for gate rows | True | **Accepted.** Both reported; lenient-above-strict is part of the canary. |
| 12 | minor | Same-station items are correlated; item bootstrap CIs a little narrow | Plausible (1,000 items over many stations) | **Accepted as sensitivity only.** Station-cluster bootstrap added for G1-G3, reported, not used for the headline, so the statistics stay those of every earlier Part. |

Also raised by the paper-draft fork's review pass (not Grok's gate review): add a 4B + schema arm. Not added: the
question is whether the next size up, constrained, does the adapter's job, and the card's "does not show" section
names this gap.

Net effect: 10 arms instead of 6, budget 3 -> 8 GPU-hours, two new stop rules (parity VOID, canary VOID), a
decision rule that cannot turn a missing or broken run into a result, and wording that cannot claim more than
"format-only constrained decoding did or did not close the gap".

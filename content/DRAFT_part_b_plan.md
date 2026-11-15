# DRAFT Part B plan (for Stephen's review; nothing runs until approved)

## Reframed claim (replaces the 235B headline)
H1: At each size (1.7B, 4B), always-on base + replay-trained specialists/adapters beats the base on all three
    prompt sets and beats pooled retraining on core, without losing to pooled on paraphrase or mixed-skill
    (non-inferiority margin set from Part A, e.g. -2 points).
H2: 4B + adapters reaches Qwen3-8B within a pre-set margin at about half the memory (same precision), and the same
    holds for 4-bit 4B + adapters vs 4-bit 8B.
H3: Library beats a distilled single model of the same size.
Pass bars and margins fixed in a frozen file with the git hash before any test scoring.

## Before freezing (~4-6 h)
- Distilled baseline: generate answers from the routed library on the training prompts; train one 1.7B and one 4B.
- Paraphrase set for the test split (same builder, new items); make paraphrase + mixed primary to neutralise the
  template confound.
- Decide routes (which domains go to adapters) on dev with cross-validation; freeze.
- Memory measurement harness: real GPU memory including KV cache and router at a fixed batch/context.

## Runs (~8-12 h)
- 3 seeds: replay specialists and replay-LoRA adapters at 1.7B and 4B (sql, tools; code if dev shows gain).
- Pooled 1.7B and 4B: 3 seeds (4B pooled ~3.5 h each; the long pole; optional to cut to 1 extra seed).
- Quantised single models: 4B and 8B in 4-bit and 8-bit (eval only).

## One test pass (~4-6 h) and analysis (~3-4 h)
All arms on test (core, paraphrase-test, mixed-test). Paired bootstrap, "inconclusive" when a CI straddles a bar.

## Expected (from Part A): H1 likely (~80-90%), H2 coin flip (~50-60%), H3 unknown (~50%).

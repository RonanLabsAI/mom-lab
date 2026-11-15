# DRAFT X thread — DO NOT POST
Status: draft only. All numbers are DEV-SPLIT (Part A) and must be replaced with test-split numbers from Part B.
Hold until Part B confirms. Kept private by Stephen's decision (2026-10-30: no patent consult); nothing posted until he says so.
Voice: Stephen's to edit; this is a starting point, not final copy.

1/
We spent a day testing a simple idea: instead of retraining an LLM to add skills, keep the original model untouched and switch in small specialist adapters only where they help.
Short version: it works, it never makes the base worse, and it buys roughly one model size. Thread. [dev numbers, Part B pending]

2/
Setup: Qwen3 1.7B and 4B. Six domains (math, code, SQL, tool calling, translation, general). Every result compared with the obvious alternative: one model fine-tuned on all the same data ("pooled").
Three test sets: standard benchmarks, the same questions reworded, and questions in Spanish, German and Chinese.

3/
Finding 1: fine-tuning a strong model on public data mostly makes it worse. A specialist trained on SQL forgot how to translate (it answered translation requests in SQL). Pooling all six domains still lost skills the base already had.

4/
Finding 2: so keep the base. Route a question to a specialist only for domains where the specialist beats the base (for us: SQL and tool calling). Everything else goes to the untouched model.
That design beat both the base and the pooled model.

5/
Finding 3: specialists trained on one domain are brittle. Reword a coding question casually and one specialist dropped from 72% to 33% (it answered from the function name, not the request).
Fix: mix 20% data from other domains into each specialist ("replay"). Brittleness gone, specialty kept. Replicated across seeds.

6/
Finding 4: routing is the easy part. A few regex rules or a small embedding classifier kept essentially all of the gain of perfect routing, even on reworded questions.

7/
Finding 5: adapters make it cheap. LoRA adapters (~1% of a model each) keep most of the gain. Qwen3-4B + two adapters statistically tied untouched Qwen3-8B on all three test sets, at about half the memory.

8/
What didn't work: a library of small models does NOT beat a much bigger model (plain 4B beats every 1.7B library). Merging specialists into one model was the worst option. Specialists only help where the base is weak.

9/
Takeaway: if you need to add skills to a deployed model, don't retrain it. Keep it, add replay-trained adapters for the gaps, route simply. [link to write-up]

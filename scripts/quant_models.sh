#!/usr/bin/env bash
# Part B quantised single-model arms (eval only): official Qwen3 AWQ (4-bit) and FP8 checkpoints. Run on a Station.
docker run -i --rm -e HF_HUB_DISABLE_XET=1 -v /models/hf:/models/hf mom-lab:v1 python3 - <<'PY' > ~/mom-lab/download-quant.log 2>&1
from huggingface_hub import snapshot_download
for i in ["Qwen/Qwen3-4B-AWQ", "Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-4B-FP8", "Qwen/Qwen3-8B-FP8"]:
    p = snapshot_download(i, local_dir="/models/hf/" + i.replace("/", "__"), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "merges.txt", "vocab.*"])
    print("OK", i, p, flush=True)
print("ALL_DONE")
PY

#!/usr/bin/env bash
# on a Spark: download Qwen3 small models (instruct+base) to ~/models/hf using the vllm image's python
mkdir -p ~/models/hf
docker run -i --rm -e HF_HUB_DISABLE_XET=1 -v $HOME/models/hf:/models/hf --entrypoint python3 vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b - <<'PY'
from huggingface_hub import snapshot_download
for i in ["Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","Qwen/Qwen3-4B","Qwen/Qwen3-8B","Qwen/Qwen3-0.6B-Base","Qwen/Qwen3-1.7B-Base","Qwen/Qwen3-4B-Base","Qwen/Qwen3-8B-Base"]:
    snapshot_download(i, local_dir="/models/hf/"+i.replace("/","__"), allow_patterns=["*.json","*.safetensors","*.txt","merges.txt","vocab.*"]); print("OK",i,flush=True)
print("ALL_DONE")
PY

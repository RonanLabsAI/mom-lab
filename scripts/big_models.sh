#!/usr/bin/env bash
# on station-b after build_and_download: large reference models
until grep -q ALL_DONE ~/mom-lab/download.log 2>/dev/null; do sleep 30; done
docker run -i --rm -e HF_HUB_DISABLE_XET=1 -v /models/hf:/models/hf mom-lab:v1 python3 - <<'PY' > ~/mom-lab/download-big.log 2>&1
from huggingface_hub import snapshot_download
for i in ["Qwen/Qwen3-30B-A3B","Qwen/Qwen3-32B","Qwen/Qwen3-235B-A22B-FP8","Qwen/Qwen3-14B"]:
    snapshot_download(i, local_dir="/models/hf/"+i.replace("/","__"), allow_patterns=["*.json","*.safetensors","*.txt","merges.txt","vocab.*"]); print("OK",i,flush=True)
print("ALL_DONE")
PY

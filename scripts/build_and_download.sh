#!/usr/bin/env bash
# run on a Station: build image, then download Qwen3 models (instruct + base) into /models/hf
set -e
cd ~/mom-lab/docker && docker build -t mom-lab:v1 . > ~/mom-lab/build.log 2>&1 && echo BUILD_OK >> ~/mom-lab/build.log
docker run -i --rm -e HF_HUB_DISABLE_XET=1 -v /models/hf:/models/hf mom-lab:v1 python3 - <<'PY' > ~/mom-lab/download.log 2>&1
from huggingface_hub import snapshot_download
ids=["Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","Qwen/Qwen3-4B","Qwen/Qwen3-8B","Qwen/Qwen3-0.6B-Base","Qwen/Qwen3-1.7B-Base","Qwen/Qwen3-4B-Base","Qwen/Qwen3-8B-Base"]
for i in ids:
    p=snapshot_download(i, local_dir="/models/hf/"+i.replace("/","__"), allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken","merges.txt","vocab.*"])
    print("OK",i,p,flush=True)
print("ALL_DONE")
PY

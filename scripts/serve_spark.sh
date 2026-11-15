#!/usr/bin/env bash
# serve_spark.sh <port> <model_dir_under_~/models/hf> <served_name> <gpu_mem_util>
P=$1; M=$2; N=$3; U=${4:-0.3}
docker rm -f mom-serve-$P >/dev/null 2>&1
docker run -d --name mom-serve-$P ${VLLM_ENV:-} --gpus all --ipc host -p $P:8000 -v $HOME/models/hf:/models/hf \
  --entrypoint vllm vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b serve /models/hf/$M \
  --served-model-name $N --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization $U --seed 0

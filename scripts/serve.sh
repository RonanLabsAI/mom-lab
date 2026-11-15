#!/usr/bin/env bash
# serve.sh <gpu_index> <port> <model_path> <served_name> [extra vllm args...]   (run on a Station)
G=$1; P=$2; M=$3; N=$4; shift 4
docker rm -f mom-serve-$P >/dev/null 2>&1
docker run -d --name mom-serve-$P ${VLLM_ENV:-} --gpus "device=$G" --shm-size 16g -p $P:8000 -v /models/hf:/models/hf -v $HOME/mom-lab:/lab \
  --entrypoint vllm mom-lab:v1 serve $M --served-model-name $N --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --seed 0 "$@"

#!/bin/bash
# Gap check, Qwen side: serve each untouched Qwen3 model on station-a's RTX PRO 6000 (GPU 0), score the puzzle set,
# remove the container. Resumable: run_gap.py skips ids already in the output file.
set -u
cd "$(dirname "$0")/.."
HOST=station-a; IP=host-3cc546d0.invalid; PORT=8710; NAME=mom-gap-$PORT
IMAGE=vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b   # same vLLM build as mom-lab:v1
PY=~/.cache/rg-venv/bin/python
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*" | tee -a results/qwen_seq.log; }

for M in 0.6B 1.7B 4B 8B 32B; do
  tag=qwen3-$(echo $M | tr A-Z a-z)
  if [ "$M" = 32B ]; then
    while ssh $HOST 'systemctl --user is-active --quiet mom-gap-copy32b'; do log "waiting for 32B copy"; sleep 120; done
  fi
  conc=128; [ "$M" = 32B ] && conc=64
  log "serve $tag"
  ssh $HOST "docker rm -f $NAME >/dev/null 2>&1; docker run -d --name $NAME --gpus device=0 --shm-size 16g -p $PORT:8000 \
    -v /models/hf:/models/hf $IMAGE /models/hf/Qwen__Qwen3-$M --served-model-name $tag --dtype bfloat16 \
    --max-model-len 40960 --gpu-memory-utilization 0.88 --seed 0" >/dev/null || { log "FAIL docker run $tag"; continue; }
  ok=0
  for i in $(seq 1 120); do
    curl -s -m 3 http://$IP:$PORT/v1/models | grep -q "$tag" && { ok=1; break; }
    ssh $HOST "docker inspect -f '{{.State.Status}}' $NAME" | grep -q running || break
    sleep 10
  done
  [ $ok = 1 ] || { log "FAIL serve $tag"; ssh $HOST "docker logs --tail 40 $NAME" >> results/qwen_seq.log 2>&1; ssh $HOST "docker rm -f $NAME" >/dev/null; continue; }
  log "run $tag"
  $PY -I scripts/run_gap.py --endpoint http://$IP:$PORT/v1 --model $tag --out results/$tag.jsonl \
      --max-tokens 32768 --concurrency $conc >> results/qwen_seq.log 2>&1
  log "done $tag rc=$?"
  ssh $HOST "docker rm -f $NAME" >/dev/null
done
log "ALL DONE"

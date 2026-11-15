#!/usr/bin/env bash
# train_run.sh <gpu> <run_name> <model> <data...> -- [train.py args]   (run on a Station; logs to runs/<name>.log)
G=$1; N=$2; M=$3; shift 3
sed -i "/ $N\$/d" ~/mom-lab/runs/done.txt 2>/dev/null   # drop stale EXIT records for this run name
DATA=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do DATA+=("$1"); shift; done; shift
docker run --rm --name mom-train-$N --gpus "device=$G" --shm-size 32g -u 1000:1000 -e HOME=/tmp -v ~/mom-lab:/lab -v /models/hf:/models/hf -w /lab mom-lab:v1 \
  python3 train/train.py --model $M --data "${DATA[@]}" --out runs/$N "$@" > ~/mom-lab/runs/$N.log 2>&1
echo "EXIT $? $N" >> ~/mom-lab/runs/done.txt

#!/usr/bin/env bash
# Part D (D): after make_sdreplay finishes, stop its vLLM server, sync the data to both Stations, queue 6 training runs.
set -eu
cd ~/projects/RonanLabs/mom-lab
while systemctl --user is-active --quiet mom-partd-sdgen; do sleep 60; done
grep -q "^tools own" logs/partd-sdgen.out || { echo "sdgen did not finish tools"; exit 1; }
ssh station-b 'docker rm -f mom-sdgen'
for f in sql tools; do [ -s data/mixtures/N14M/sdreplay/$f-r20.jsonl ] || { echo "missing $f"; exit 1; }; done
for h in station-a station-b; do rsync -a data/mixtures/N14M/sdreplay $h:mom-lab/data/mixtures/N14M/; done
sha256sum data/mixtures/N14M/sdreplay/*.jsonl > logs/partd-sdreplay.sha256
for h in station-a station-b; do ssh $h 'cd ~/mom-lab && sha256sum data/mixtures/N14M/sdreplay/*.jsonl'; done >> logs/partd-sdreplay.sha256
python3 - <<'PY'
import json
J=[]
for s in [0,1,2]:
    for d in ['sql','tools']:
        n=f"pt4-lora16-sdreplay-{d}-r20-lr2e-4"+("" if s==0 else f"-s{s}")
        J.append({"name":n,"prefer":[],"model":"/models/hf/Qwen__Qwen3-4B","data":[f"data/mixtures/N14M/sdreplay/{d}-r20.jsonl"],"args":["--lr","2e-4","--epochs","2","--micro-tokens","32768","--seed",str(s),"--lora-rank","16","--grad-ckpt"]})
open('configs/queues/partd.jsonl','a').write(''.join(json.dumps(j)+'\n' for j in J))
PY
echo "D queued $(date -u)"

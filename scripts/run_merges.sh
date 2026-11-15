#!/usr/bin/env bash
# run_merges.sh — Step 3 merging baseline. Gathers the six lr1e-5 specialists (pt17 and b17) onto station-a,
# merges them on CPU inside mom-lab:v1 (avg and TIES), leaves merged models in ~/mom-lab/runs/<prefix>-merge-<m>-lr1e-5
# for evalq (--extra name@station-a:mom-lab/runs/<name>). Launch with setsid nohup from mom-lab/.
set -u
cd "$(dirname "$0")/.."
DOMS="math code sql tools translate general"
STATE=logs/queue/sweep17.state.json
for pre in pt17 b17; do
  for d in $DOMS; do
    n=$pre-spec-$d-lr1e-5
    host=$(.venv/bin/python -P -c "import json;print(json.load(open('$STATE'))['finished']['$n']['slot'].split(':')[0])")
    if [ "$host" != station-a ]; then
      ssh -o BatchMode=yes station-a "test -f ~/mom-lab/evalcache/$n/model.safetensors" && continue
      ssh -o BatchMode=yes $host "tar c -C ~/mom-lab/runs --exclude='epoch*' $n" | \
        ssh -o BatchMode=yes station-a "mkdir -p ~/mom-lab/evalcache && rm -rf ~/mom-lab/evalcache/$n && tar x -C ~/mom-lab/evalcache" \
        || { echo "COPY_FAIL $n"; exit 1; }
    fi
    echo "have $n on station-a"
  done
done
scp -q -o BatchMode=yes scripts/merge_specialists.py station-a:mom-lab/merge_specialists.py
for pre in pt17 b17; do
  [ $pre = pt17 ] && base=/models/hf/Qwen__Qwen3-1.7B || base=/models/hf/Qwen__Qwen3-1.7B-Base
  specs=""
  for d in $DOMS; do
    n=$pre-spec-$d-lr1e-5
    if ssh -o BatchMode=yes station-a "test -f ~/mom-lab/runs/$n/model.safetensors"; then specs="$specs /lab/runs/$n"; else specs="$specs /lab/evalcache/$n"; fi
  done
  for m in avg ties; do
    out=$pre-merge-$m-lr1e-5
    ssh -o BatchMode=yes station-a "docker run --rm --user \$(id -u):\$(id -g) --name mom-merge-$out --shm-size 16g -v /models/hf:/models/hf -v \$HOME/mom-lab:/lab \
      --entrypoint python3 mom-lab:v1 /lab/merge_specialists.py $base /lab/runs/$out $m $specs" > logs/merge-$out.log 2>&1
    grep -q MERGE_DONE logs/merge-$out.log && echo "MERGED $out" || { echo "MERGE_FAIL $out"; tail -5 logs/merge-$out.log; }
  done
done
echo ALL_MERGES_DONE

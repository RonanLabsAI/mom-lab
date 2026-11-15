#!/usr/bin/env bash
# Protocol experiment on Qwen3-1.7B: decoding x token budget, math500/gsm8k/mmlu_pro/ifeval/livecodebench (dev)
cd ~/projects/RonanLabs/mom-lab; export HARNESS_SANDBOX=docker-batch; PY=.venv/bin/python
B=math500,gsm8k,mmlu_pro,ifeval,livecodebench
until curl -s -m 3 http://host-b693d1b8.invalid:8621/v1/models >/dev/null && curl -s -m 3 http://host-3cc546d0.invalid:8622/v1/models >/dev/null; do sleep 10; done
g() { R=$1; E=$2; shift 2; $PY -m harness.generate --endpoint http://$E/v1 --model qwen3-1.7b --run-id $R --step protocol --split dev --benchmarks $B --concurrency 96 "$@" > logs/protocol/$R.gen.log 2>&1; $PY -m harness.aggregate --run-id $R --step protocol --split dev --benchmarks $B > logs/protocol/$R.agg.log 2>&1; echo "DONE $R" >> logs/protocol/done.txt; }
mkdir -p logs/protocol; : > logs/protocol/done.txt
( g p-greedy-4k host-b693d1b8.invalid:8621 --max-tokens-override 4096 --hardware GB300
  g p-greedy-pp-4k host-b693d1b8.invalid:8621 --max-tokens-override 4096 --presence-penalty 1.5 --hardware GB300
  g p-greedy-4k-rerun host-b693d1b8.invalid:8621 --max-tokens-override 4096 --hardware GB300 ) &
( for s in 0 1 2; do g p-sample-s$s-4k host-3cc546d0.invalid:8622 --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --seed $s --max-tokens-override 4096 --hardware "RTX PRO 6000"; done ) &
wait; echo ALL_PROTOCOL_DONE >> logs/protocol/done.txt

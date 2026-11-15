#!/usr/bin/env bash
# batch-invariant determinism test: two identical greedy runs on gsm8k+math500+ifeval at concurrency 96
cd ~/projects/RonanLabs/mom-lab; export HARNESS_SANDBOX=docker-batch; PY=.venv/bin/python
until curl -s -m 3 http://host-b693d1b8.invalid:8631/v1/models >/dev/null; do sleep 10; done
for r in p-bi-greedy-a p-bi-greedy-b; do
  /usr/bin/time -f "%e s" -o logs/protocol/$r.time $PY -m harness.generate --endpoint http://host-b693d1b8.invalid:8631/v1 --model qwen3-1.7b --run-id $r --step protocol --split dev --benchmarks gsm8k,math500,ifeval --concurrency 96 --max-tokens-override 4096 --hardware "GB300 batch-invariant" > logs/protocol/$r.gen.log 2>&1
  $PY -m harness.aggregate --run-id $r --step protocol --split dev --benchmarks gsm8k,math500,ifeval > logs/protocol/$r.agg.log 2>&1
  echo "DONE $r" >> logs/protocol/done.txt
done

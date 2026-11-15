#!/usr/bin/env bash
cd ~/projects/RonanLabs/mom-lab; export HARNESS_SANDBOX=docker-batch; PY=.venv/bin/python
until curl -s -m 3 http://host-25d077a9.invalid:8641/v1/models >/dev/null && curl -s -m 3 http://host-ed134442.invalid:8642/v1/models >/dev/null; do sleep 10; done
g() { R=$1; E=$2; /usr/bin/time -f "%e s" -o logs/protocol/$R.time $PY -m harness.generate --endpoint http://$E/v1 --model qwen3-1.7b --run-id $R --step protocol --split dev --benchmarks gsm8k,math500,ifeval --concurrency 128 --max-tokens-override 4096 --hardware "DGX Spark batch-invariant" > logs/protocol/$R.gen.log 2>&1; $PY -m harness.aggregate --run-id $R --step protocol --split dev --benchmarks gsm8k,math500,ifeval > /dev/null 2>&1; echo "DONE $R" >> logs/protocol/done.txt; }
( g p-bi-spark3-a host-25d077a9.invalid:8641; g p-bi-spark3-b host-25d077a9.invalid:8641 ) &
g p-bi-spark1-a host-ed134442.invalid:8642 &
wait

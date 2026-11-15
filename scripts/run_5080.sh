#!/usr/bin/env bash
cd ~/projects/RonanLabs/mom-lab; export HARNESS_SANDBOX=docker-batch; PY=.venv/bin/python
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8661/v1/models | grep -q qwen3 && break; sleep 10; done
/usr/bin/time -f "%e s" -o logs/protocol/p-bi-5080-a.time $PY -m harness.generate --endpoint http://127.0.0.1:8661/v1 --model qwen3-1.7b --run-id p-bi-5080-a --step protocol --split dev --benchmarks gsm8k,math500,ifeval --concurrency 32 --max-tokens-override 4096 --hardware "RTX 5080 (WSL) vLLM 0.30.0 batch-invariant" > logs/protocol/p-bi-5080-a.gen.log 2>&1
$PY -m harness.aggregate --run-id p-bi-5080-a --step protocol --split dev --benchmarks gsm8k,math500,ifeval > /dev/null 2>&1; echo "DONE p-bi-5080-a" >> logs/protocol/done.txt

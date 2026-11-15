#!/usr/bin/env bash
# Step 1: off-the-shelf Qwen3 on dev, primary protocol (greedy, non-thinking) + reproduction (Qwen non-thinking sampling) for math500/ifeval
cd ~/projects/RonanLabs/mom-lab; export HARNESS_SANDBOX=docker-batch
PY=.venv/bin/python
run() { # endpoint model tag hw conc
  E=$1; M=$2; T=$3; H=$4; C=$5
  $PY -m harness.generate --endpoint http://$E/v1 --model $M --run-id s1-$T-greedy --step step1 --split dev --concurrency $C --hardware "$H" > logs/step1/$T-greedy.gen.log 2>&1
  $PY -m harness.aggregate --run-id s1-$T-greedy --step step1 --split dev > logs/step1/$T-greedy.agg.log 2>&1
  $PY -m harness.generate --endpoint http://$E/v1 --model $M --run-id s1-$T-repro --step step1 --split dev --benchmarks math500,ifeval \
     --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --max-tokens-override 6144 --concurrency $C --hardware "$H" > logs/step1/$T-repro.gen.log 2>&1
  $PY -m harness.aggregate --run-id s1-$T-repro --step step1 --split dev --benchmarks math500,ifeval > logs/step1/$T-repro.agg.log 2>&1
  echo "DONE $T" >> logs/step1/done.txt
}
mkdir -p logs/step1; : > logs/step1/done.txt
run host-25d077a9.invalid:8616 qwen3-0.6b qwen3-0.6b-spark3 "DGX Spark GB10" 32 &
run host-3cc546d0.invalid:8611 qwen3-1.7b qwen3-1.7b-6000 "RTX PRO 6000 Max-Q" 64 &
run host-b693d1b8.invalid:8614 qwen3-1.7b qwen3-1.7b-gb300 "GB300" 64 &
run host-b55245c7.invalid:8615 qwen3-1.7b qwen3-1.7b-spark2 "DGX Spark GB10" 32 &
run host-b693d1b8.invalid:8613 qwen3-4b qwen3-4b-6000 "RTX PRO 6000 Max-Q" 64 &
run host-3cc546d0.invalid:8612 qwen3-8b qwen3-8b-gb300 "GB300" 64 &
wait; echo ALL_STEP1_DONE >> logs/step1/done.txt

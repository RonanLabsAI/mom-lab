#!/usr/bin/env bash
# re-aggregate (full) every finished step1 run with the fixed code sandbox
cd ~/projects/RonanLabs/mom-lab; export HARNESS_SANDBOX=docker-batch
for r in "$@"; do b=""; [[ $r == *repro ]] && b="--benchmarks math500,ifeval"; .venv/bin/python -m harness.aggregate --run-id $r --step step1 --split dev $b > logs/step1/$r.reagg.log 2>&1; echo "REAGG $r" >> logs/step1/done.txt; done

#!/usr/bin/env bash
# chain_step6_eval.sh — wait for the Step 4 paraphrase eval (29 models) to finish, stop that queue, then start the
# Step 6 eval queue (all step6a runs + Qwen3-4B base; core six + mixed + para) on the two RTX PRO 6000s.
cd "$(dirname "$0")/.."
until [ "$(sort -u logs/queue/eval-step4.done 2>/dev/null | wc -l)" -ge 29 ]; do sleep 60; done
echo "$(date -u +%H:%M:%S) step4 para eval complete"
./scripts/stop_queue.sh evalq; sleep 5
EVAL_SPARKS=0 EVAL_STATION_SLOTS=station-a:0,station-b:0 EVAL_DOMAINS=math,code,sql,tools,translate,general,mixed,para \
  setsid nohup python3 -P scripts/evalq.py step6a step6 base-qwen3-4b@station-a:/models/hf/Qwen__Qwen3-4B \
  >> logs/queue/eval-step6.out 2>&1 < /dev/null &
echo "$(date -u +%H:%M:%S) step6 eval queue started"

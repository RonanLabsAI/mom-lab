#!/usr/bin/env bash
# stop_queue.sh evalq|trainq — kill the queue process AND its process group (generators, aggregators).
# Safety: only kill the group when the queue is its own group leader (started with setsid) and the group is
# not this shell's own; otherwise kill just the process tree. Killing a shared group killed the Claude
# session on 2026-10-29 (the queue had been started without setsid, so its group was Claude's).
pat="^(/usr/bin/)?python3 (-P )?scripts/$1.py"
# queues launched as user systemd units (mom-<queue>-<trainq|evalq>.service; see runbook) are stopped via systemd
systemctl --user stop "mom-*-$1.service" 2>/dev/null
mypg=$(ps -o pgid= -p $$ | tr -d ' ')
for p in $(pgrep -f "$pat"); do
  pg=$(ps -o pgid= -p $p | tr -d ' ')
  if [ "$pg" = "$p" ] && [ "$pg" != "$mypg" ]; then
    kill -- -$pg 2>/dev/null
  else
    pkill -P $p 2>/dev/null; kill $p 2>/dev/null
    echo "warn: $p not a group leader; killed process + children only"
  fi
done
echo stopped
# evalq: its vLLM servers live on the Stations; left running they hold the GPU and the next evalq waits forever
if [ "$1" = evalq ]; then
  for h in station-a station-b; do ssh -o BatchMode=yes $h 'docker ps --format "{{.Names}}" | grep "^mom-eval-" | xargs -r docker rm -f >/dev/null'; done
  echo "removed eval servers"
fi

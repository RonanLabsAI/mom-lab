#!/usr/bin/env bash
# Part D addendum F launcher (content/PART_D_F_ADDENDUM.md). Run on the Threadripper from anywhere.
#   scripts/partd_f_launch.sh check   # hashes + both RTX PRO 6000s free? (no GPU use)
#   scripts/partd_f_launch.sh smoke   # dev smoke (4 jobs x 30 items), then: .venv/bin/python -P scripts/partd_f/smoke_check.py
#   scripts/partd_f_launch.sh main    # the test runs (configs/partd_f_evaljobs.txt), lowest prio first; exits when done
#   scripts/partd_f_launch.sh status  # unit state + last events
#   scripts/partd_f_launch.sh stop    # stop the unit and remove this queue's eval containers on both Stations
# Fleet claims are separate: ~/bin/fleet-claims claim / release (the 6000s, index 0 on each Station).
set -euo pipefail
L="$HOME/projects/RonanLabs/mom-lab"; cd "$L"
SLOTS="station-a:0,station-b:0"
EV=logs/queue/eval-partd-f.events.log

check() {
  sha256sum -c --quiet artifacts/PARTD_F_FREEZE.sha256 && echo "freeze hashes OK"
  for h in station-a station-b; do
    m=$(ssh "$h" 'nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits')
    echo "$h 6000 memory.used=${m} MiB $([ "$m" -lt 2000 ] && echo free || echo BUSY - the queue waits until it is free)"
  done
}

launch() {  # $1 unit  $2 jobs file
  check
  if systemctl --user is-active --quiet "$1"; then echo "$1 already running"; exit 1; fi
  mkdir -p logs/queue
  systemd-run --user --unit="$1" --collect --working-directory="$L" \
    -E EVAL_STATION_SLOTS="$SLOTS" \
    -p StandardOutput=append:"$L/logs/queue/$1.out" -p StandardError=append:"$L/logs/queue/$1.out" \
    python3 -P scripts/evalq_f.py none "$2" none
  echo "started $1; events: $EV"
}

case "${1:-}" in
  check) check ;;
  smoke) launch mom-partd-f-smoke configs/partd_f_smokejobs.txt ;;
  main) launch mom-partd-f configs/partd_f_evaljobs.txt ;;
  status)
    for u in mom-partd-f-smoke mom-partd-f; do echo "$u: $(systemctl --user is-active $u || true)"; done
    tail -n 20 "$EV" 2>/dev/null || true
    for f in logs/queue/eval-partdF-smoke.done logs/queue/eval-partdF.done; do [ -f "$f" ] && echo "$f: $(wc -l < "$f") done"; done; true ;;
  stop)
    for u in mom-partd-f-smoke mom-partd-f; do systemctl --user stop "$u" 2>/dev/null || true; done
    for h in station-a station-b; do ssh "$h" 'docker rm -f mom-eval-8681 mom-eval-8682 2>/dev/null' || true; done
    echo stopped ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac

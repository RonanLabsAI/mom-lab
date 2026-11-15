#!/usr/bin/env bash
# Copy every finished run on a Station to the Threadripper archive and verify it (sizes of all files + sha256 of
# every *.safetensors on both sides). Deletion is a separate, manual step after reading the report.
# usage: scripts/archive_runs.sh <host> [exclude-regex]   (run on the Threadripper)
set -euo pipefail
host=$1; excl=${2:-'^pt4-pooled|distill'}
dst=/srv/release-data/sentinel/mom-lab-archive/runs/$host
rep=logs/archive/$host.report
mkdir -p "$dst" logs/archive
runs=$(ssh -o BatchMode=yes "$host" 'ls ~/mom-lab/runs' | grep -v '^done.txt$' | grep -Ev "$excl" || true)
: > "$rep"
for r in $runs; do
  ssh -o BatchMode=yes "$host" "test -d ~/mom-lab/runs/$r" || continue
  rsync -a "$host:mom-lab/runs/$r/" "$dst/$r/"
  remote=$(ssh -o BatchMode=yes "$host" "cd ~/mom-lab/runs/$r && find . -type f -printf '%P %s\n' | sort")
  local_=$(cd "$dst/$r" && find . -type f -printf '%P %s\n' | sort)
  rsha=$(ssh -o BatchMode=yes "$host" "cd ~/mom-lab/runs/$r && find . -name '*.safetensors' -print0 | sort -z | xargs -0 -r -P8 sha256sum | sort -k2")
  lsha=$(cd "$dst/$r" && find . -name '*.safetensors' -print0 | sort -z | xargs -0 -r -P8 sha256sum | sort -k2)
  if [[ "$remote" == "$local_" && "$rsha" == "$lsha" && -n "$lsha" ]]; then
    echo "OK $r files=$(wc -l <<<"$local_") st=$(wc -l <<<"$lsha")" >> "$rep"
  else
    echo "FAIL $r" >> "$rep"
  fi
done
echo "DONE $(date -u +%FT%TZ)" >> "$rep"

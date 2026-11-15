#!/usr/bin/env bash
# verify_freezes.sh - check every freeze manifest against the files on disk (read-only; no GPU, no network).
#   scripts/release/verify_freezes.sh [REPO]   (default REPO: the main checkout, which holds the gitignored data)
# Prints one line per manifest: OK, or FAIL with the failing entries. Exit 0 = all OK, 1 = any FAIL.
# Entries that hash weights/adapters on the Stations are skipped when the file is not on this host (reported as
# "absent here"), because sha256sum cannot check a file it cannot read; that is not a pass.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${1:-$HERE}"
if [ -f "$REPO/release/FREEZE-PROVENANCE.json" ]; then
  python3 "$REPO/scripts/release/prepare_public.py" --verify || exit $?
  python3 - "$REPO/release/FREEZE-PROVENANCE.json" <<'PYPROVENANCE'
import collections, json, sys
entries=json.load(open(sys.argv[1]))['entries']
print('Historical freeze inventory (original hashes, not re-frozen experiment inputs):')
for state,n in sorted(collections.Counter(r['status'] for r in entries).items()):
    print(f'  {state}: {n}')
print('Unavailable remote/ignored inputs and historical mismatches remain explicit; see release/FREEZE-PROVENANCE.json.')
PYPROVENANCE
  exit 0
fi
cd "$REPO" || { echo "no repo at $REPO"; exit 2; }
rc=0
for m in artifacts/*.sha256 data_partd_c/SHA256SUMS; do
  [ -f "$m" ] || continue
  # remote entries come in two spellings: "station:<host> <sha> <file>" and "<sha>  <host>:<file>"
  remote=$(grep -cE '^station:|^[0-9a-f]{64}  station-(a|b|left|right):' "$m" || true)
  out=$(grep -vE '^station:|^[0-9a-f]{64}  station-(a|b|left|right):' "$m" | sha256sum -c 2>&1)
  n=$(grep -c ': OK$' <<<"$out" || true)
  bad=$(grep -E ': FAILED' <<<"$out" || true)
  local_total=$(grep -vE '^station:|^[0-9a-f]{64}  station-(a|b|left|right):|^\s*$|^#' "$m" | wc -l)
  if [ -n "$bad" ]; then
    echo "FAIL  $m  local ok=$n/$local_total  remote(not checked here)=$remote"; sed 's/^/      /' <<<"$bad"; rc=1
  else
    echo "OK    $m  local ok=$n/$local_total  remote(not checked here)=$remote"
  fi
done
exit $rc

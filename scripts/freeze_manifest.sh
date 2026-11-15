#!/usr/bin/env bash
# freeze_manifest.sh — write artifacts/FREEZE_MANIFEST.sha256 for the Part B preregistration freeze.
# Covers: code (scripts, harness, train, docker, configs), the router, the prereg + GOAL, every test-split prompt set,
# the exact weights of all 50 test-pass models, and the Station environment (train_run.sh, image IDs, pip freeze,
# GPU driver). Run once, from the project root, before any test-split generation. Refuses to overwrite.
# Verify later with: sha256sum -c <(grep -v '^#' artifacts/FREEZE_MANIFEST.sha256 | grep -v '^station:')
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${FREEZE_OUT:-artifacts/FREEZE_MANIFEST.sha256}   # FREEZE_OUT/FREEZE_EXPECT: dry runs only
EXPECT=${FREEZE_EXPECT:-50}
[ -e "$OUT" ] && { echo "refusing: $OUT exists"; exit 1; }
tmp=$(mktemp)
{
  echo "# Part B freeze manifest, written $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M %Z')"
  echo "# local files"
  find scripts harness train docker configs -type f -not -path '*/__pycache__/*' -not -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum
  sha256sum artifacts/router_e5lr.pkl content/PART_B_PREREG.md GOAL.md
  find data/eval -name test.jsonl -print0 | sort -z | xargs -0 sha256sum
  echo "# test-pass model weights (station:<host> <sha256> <file>)"
  arms=$(python3 - <<'EOF'
import json
st = json.load(open("logs/queue/partb.state.json"))
rows = [f"{n}@{v['slot'].split(':')[0]}:mom-lab/runs/{n}" for n, v in sorted(st["finished"].items()) if v.get("exit") == 0]
rows += [l.strip() for l in open("configs/partbtest_extras.txt") if l.strip() and not l.startswith("#")]
print("\n".join(rows))
EOF
)
  n=$(echo "$arms" | grep -c .)
  [ "$n" -eq "$EXPECT" ] || { echo "expected $EXPECT arms, got $n" >&2; exit 1; }
  echo "$arms" | while IFS= read -r a; do
    host=${a#*@}; host=${host%%:*}; path=${a#*:}; lora=""
    case "$path" in *+lora=*) lora=${path#*+lora=}; path=${path%%+lora=*};; esac
    [ -n "$host" ] && [ -n "$path" ] || { echo "bad arm $a" >&2; exit 1; }
    ssh -o BatchMode=yes "$host" "cd ~ && sha256sum $path/*.safetensors" < /dev/null | sed "s|^|station:$host |"
    if [ -n "$lora" ]; then
      lp="mom-lab/${lora#/lab/}"
      for h in station-a station-b; do
        ssh -o BatchMode=yes "$h" "cd ~ && sha256sum $lp/adapter_model.safetensors" < /dev/null | sed "s|^|station:$h |"
      done
    fi
  done
  echo "# station environment"
  for h in station-a station-b; do
    ssh -o BatchMode=yes "$h" 'cd ~/mom-lab && sha256sum train_run.sh train/train.py docker/Dockerfile;
      echo "image mom-lab:v1 $(docker image inspect -f {{.Id}} mom-lab:v1)";
      echo "pipfreeze $(docker run --rm --entrypoint pip mom-lab:v1 freeze 2>/dev/null | sort | sha256sum | cut -d" " -f1)";
      echo "driver $(nvidia-smi -i 0 --query-gpu=name,driver_version,power.limit --format=csv,noheader)"' < /dev/null | sed "s|^|station:$h |"
  done
} > "$tmp"
w=$(grep -c "^station:.*\.safetensors$" "$tmp" || true)
[ "$w" -ge "$EXPECT" ] || { echo "only $w weight hashes for $EXPECT arms; not writing" >&2; rm -f "$tmp"; exit 1; }
mv "$tmp" "$OUT"
echo "wrote $OUT ($(grep -vc '^#' "$OUT") entries); sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"

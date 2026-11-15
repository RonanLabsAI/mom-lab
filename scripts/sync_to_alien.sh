#!/usr/bin/env bash
# Mirror lab outputs + the research folder to the PI's desktop.
# Public release: the private destination and source paths were removed; set them before use.
set -e
: "${MIRROR_DEST:?set MIRROR_DEST=user@host:/path/to/mirror}"
: "${RESEARCH_DIR:?set RESEARCH_DIR to the private research folder}"
A=$MIRROR_DEST
rsync -a "$RESEARCH_DIR"/ $A/
rsync -a --exclude .venv --exclude 'data/train' --exclude 'data/eval/*/*/databases' --exclude '*.safetensors' --exclude 'results/*/*/gen' \
  "$(dirname "$0")"/../{GOAL.md,CONTRACT.md,README.md,notebook,content,results,configs,scripts,harness,docker} $A/lab/ 2>/dev/null || true
echo synced

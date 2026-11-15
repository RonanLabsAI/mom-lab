#!/usr/bin/env bash
# Back up Part D station state to the TR archive before the SSD swap (copy only; nothing deleted).
set -u
A=/srv/release-data/sentinel/mom-lab-archive
for h in station-a station-b; do
  [ -n "$h" ] || exit 1
  mkdir -p $A/runs/$h $A/adapters/$h $A/partd-done/$h
  rsync -a --exclude 'pt4-lora128-*-s2' $h:mom-lab/runs/ $A/runs/$h/ && echo "runs $h ok $(date -u)"
  rsync -a $h:mom-lab/adapters/ $A/adapters/$h/ && echo "adapters $h ok"
  rsync -a $h:mom-lab/runs/done.txt $A/partd-done/$h/ && echo "done.txt $h ok"
done
mkdir -p $A/docker; ssh station-a 'docker save mom-lab:v1' | gzip -1 > $A/docker/mom-lab-v1.tar.gz && echo "image ok $(ls -la $A/docker/mom-lab-v1.tar.gz)"
echo ALL_DONE $(date -u)

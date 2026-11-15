#!/bin/bash
# Raw non-HF downloads needed by eval_sql.py / eval_translate.py (run from repo root)
set -e
mkdir -p data/raw/bird data/raw/flores
[ -d data/raw/bird/dev_x ] || { curl -sL -o data/raw/bird/dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip; unzip -q data/raw/bird/dev.zip -d data/raw/bird/dev_x; (cd data/raw/bird/dev_x/dev_20240627 && unzip -q dev_databases.zip -d db_x); }
[ -d data/raw/flores/flores200_dataset ] || { curl -sL -o data/raw/flores/f200.tgz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz; tar xzf data/raw/flores/f200.tgz -C data/raw/flores; }

#!/usr/bin/env bash
# 取难负集重建所需的两份意图分类训练集（约 600KB）
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/_raw"
curl -sL -o "$ROOT/_raw/banking77_train.parquet" \
  "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/train-00000-of-00001.parquet"
curl -sL -o "$ROOT/_raw/clinc_plus_train.parquet" \
  "https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/plus/train-00000-of-00001.parquet"
echo "done -> $ROOT/_raw"

#!/usr/bin/env bash
# 下载 mammoth-1（27B）权重，约 54.7 GB
# mammoth-1 与 elephant-1 共用同一套 OneJev runtime，不需要重跑 setup.sh
set -uo pipefail

ROOT=/mnt/raid/home/mengguanlin/decision-eval
cd "$ROOT" || exit 1
mkdir -p models logs

export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo '')"
echo "token len=${#HF_TOKEN}"

if [ -f models/mammoth-1/model-00002-of-00002.safetensors ]; then
  echo "already downloaded"
  du -sh models/mammoth-1
  exit 0
fi

setsid nohup "$ROOT/.venv-decider/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('baselabs/mammoth-1', local_dir='models/mammoth-1',
                      allow_patterns=['*.json','*.safetensors','*.jinja','*.py','*.patch','*.sh','*.txt'])
print('DONE', p)
" > logs/dl_mammoth.log 2>&1 < /dev/null &

echo "download started -> logs/dl_mammoth.log (pid $!)"

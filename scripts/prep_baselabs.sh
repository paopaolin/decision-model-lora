#!/usr/bin/env bash
# 准备 Baset Labs（baseten）决策模型的部署环境：拉官方镜像 + 下载 elephant-1(9B) 权重
# 9B 和 27B 用同一套 setup.sh / OneJev / serve.patch，走通 9B 就等于把 27B 的路铺好
# 用法（服务器上）: bash prep_baselabs.sh

set -uo pipefail

ROOT=/mnt/raid/home/mengguanlin/decision-eval
IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

cd "$ROOT" || exit 1
mkdir -p models logs

if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  echo "HF_TOKEN loaded (len=${#HF_TOKEN})"
else
  echo "WARN: no HF token, will download anonymously (slower)"
fi

# 1) 官方指定镜像，9B/27B 共用
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image already present: $IMAGE"
else
  setsid nohup docker pull "$IMAGE" > logs/docker_pull_pytorch.log 2>&1 < /dev/null &
  echo "pulling image -> logs/docker_pull_pytorch.log (pid $!)"
fi

# 2) elephant-1 权重（约 19 GB）
if [ -f models/elephant-1/model.safetensors ]; then
  echo "elephant-1 already downloaded"
else
  setsid nohup .venv-decider/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('baselabs/elephant-1', local_dir='models/elephant-1')
print('DONE', p)
" > logs/hf_download_elephant1.log 2>&1 < /dev/null &
  echo "downloading elephant-1 -> logs/hf_download_elephant1.log (pid $!)"
fi

sleep 12
echo "--- image pull progress ---"
tail -3 logs/docker_pull_pytorch.log 2>/dev/null || echo "(no output yet)"
echo "--- weight download progress ---"
tail -3 logs/hf_download_elephant1.log 2>/dev/null || echo "(no output yet)"

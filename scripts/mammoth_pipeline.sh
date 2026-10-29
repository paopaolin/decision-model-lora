#!/usr/bin/env bash
# 27B（mammoth-1）自动流水线：等下载 -> 起服务 -> 等健康 -> 跑 231 条 -> 汇总
# 全程在服务器上跑，不依赖任何 ssh 会话
set -uo pipefail

ROOT=/mnt/raid/home/mengguanlin/decision-eval
cd "$ROOT" || exit 1
LOG=logs/mammoth_pipeline.log
exec >> "$LOG" 2>&1
echo "=== $(date '+%F %T') pipeline start ==="

# 1) 等下载完成（dl_mammoth.log 里出现 DONE）
for i in $(seq 1 240); do
  if grep -q "^DONE" logs/dl_mammoth.log 2>/dev/null; then
    echo "[1/4] download done: $(du -sh models/mammoth-1 | cut -f1)"
    break
  fi
  sleep 60
done
if ! grep -q "^DONE" logs/dl_mammoth.log 2>/dev/null; then
  echo "[1/4] TIMEOUT waiting download, abort"; exit 1
fi

# 2) 起服务（容器 jev-27b 挂 GPU 2）
echo "[2/4] starting server on :8200"
docker exec -d jev-27b bash -lc "cd /work && python serve_baseten.py --model /work/models/mammoth-1 --port 8200 > /work/logs/serve_mammoth.log 2>&1"

# 3) 等健康（27B 加载慢，最多等 15 分钟）
ok=0
for i in $(seq 1 45); do
  if docker exec jev-27b python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8200/health',timeout=5).read()" >/dev/null 2>&1; then
    echo "[3/4] health ok after $((i*20))s"
    docker exec jev-27b python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8200/health',timeout=10).read().decode())"
    ok=1
    break
  fi
  sleep 20
done
if [ "$ok" != "1" ]; then
  echo "[3/4] health TIMEOUT"
  tail -20 logs/serve_mammoth.log
  exit 1
fi

# 4) 评测
echo "[4/4] running jevbench public-231"
docker exec jev-27b mkdir -p /work/results/mammoth1_base
docker exec -w /work/jevbench jev-27b python -m jevbench.cli run \
  --tasks /work/data/jevbench_public_231.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8200 \
  --model mammoth-1 --key-env= \
  --cost-basis no_billable_local_selfhost --reserve-usd 0 \
  --results /work/results/mammoth1_base/results.jsonl \
  --raw-dir /work/results/mammoth1_base/raw \
  --ledger /work/results/mammoth1_base/ledger.jsonl \
  --manifest /work/results/mammoth1_base/manifest.json

docker exec -w /work/jevbench jev-27b python -m jevbench.cli summarize \
  --tasks /work/data/jevbench_public_231.jsonl \
  --results /work/results/mammoth1_base/results.jsonl \
  --ledger /work/results/mammoth1_base/ledger.jsonl \
  --public-export /work/results/mammoth1_base/summary.json

echo "=== $(date '+%F %T') pipeline done ==="

#!/bin/bash
# 串行跑一个模型的难负评测：起服务 -> 等健康 -> 跑 1200 条 -> 汇总 -> 停服务
# 用法：bash run_hard_eval.sh <容器名> <模型路径> <运行名> <端口>
set -u

C=$1
MODEL=$2
NAME=$3
PORT=$4
D=/work/results/crossbench_hard

echo "[$(date +%H:%M:%S)] === $NAME 开始 ==="
docker exec -d "$C" bash -lc "cd /work && python serve_baseten.py --model $MODEL --port $PORT > /work/logs/serve_${NAME}_hard.log 2>&1"

ok=0
for i in $(seq 1 50); do
    if docker exec "$C" python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=3)" >/dev/null 2>&1; then
        ok=1
        echo "[$(date +%H:%M:%S)] 服务就绪（等待 $((i*5))s）"
        break
    fi
    sleep 5
done
if [ "$ok" != "1" ]; then
    echo "ERROR: $NAME 服务未起来"
    exit 1
fi

docker exec "$C" mkdir -p "$D/$NAME"
docker exec -w /work/jevbench "$C" python -m jevbench.cli run \
    --tasks /work/data/crossbench/all_hard.jsonl \
    --adapter typesafe --endpoint "http://127.0.0.1:$PORT" --model "$NAME" \
    --key-env= --cost-basis no_billable_local_selfhost --reserve-usd 0 \
    --results "$D/$NAME/results.jsonl" --raw-dir "$D/$NAME/raw" \
    --ledger "$D/$NAME/ledger.jsonl" --manifest "$D/$NAME/manifest.json" 2>&1 | tail -2

docker exec -w /work/jevbench "$C" python -m jevbench.cli summarize \
    --tasks /work/data/crossbench/all_hard.jsonl \
    --results "$D/$NAME/results.jsonl" --ledger "$D/$NAME/ledger.jsonl" \
    --public-export "$D/$NAME/summary.json" >/dev/null 2>&1 && echo "汇总已写入"

docker exec "$C" bash -lc "pkill -f serve_baseten || true"
sleep 6
echo "[$(date +%H:%M:%S)] === $NAME 完成，服务已停 ==="

#!/usr/bin/env bash
# 等待容器内 setup.sh 把 causal-conv1d 编译安装完成
# 注意：不要用 pgrep -f "setup.sh" 判断，docker exec 自己的命令行里就含这个串，会自匹配

cd /mnt/raid/home/mengguanlin/decision-eval || exit 1

for i in $(seq 1 45); do
  ver=$(docker exec jev-baseten python -m pip show causal-conv1d 2>/dev/null | awk '/^Version/{print $2}')
  if [ -n "$ver" ]; then
    echo "causal-conv1d installed: $ver (after ~${i}min)"
    break
  fi
  sleep 60
done

sleep 20
echo "=== SETUP LOG TAIL ==="
tail -12 logs/setup_elephant.log

echo "=== IMPORT CHECK ==="
docker exec jev-baseten python -c 'import qev, qev.engine; print("qev ok")' 2>&1 | tail -3

echo "=== PIP KEY PKGS ==="
docker exec jev-baseten python -m pip list 2>/dev/null | grep -Ei "^(torch|transformers|causal-conv1d|flash|peft|qev) " | head -10

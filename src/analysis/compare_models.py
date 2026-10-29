#!/usr/bin/env python3
"""Decider-2B vs elephant-1(9B) 逐条对比：分档、校准、翻转、共同错误。"""

import json
from collections import Counter
from statistics import mean

tasks = {json.loads(l)["id"]: json.loads(l) for l in open("data_tasks.jsonl")}
A = {r["task_id"]: r for r in (json.loads(x) for x in open("results/results.jsonl"))}
B = {r["task_id"]: r for r in (json.loads(x) for x in open("results/elephant1_base/results.jsonl"))}


def conf(r):
    p = r.get("probs") or {}
    return max(p.values()) if p else 0.0


ids = sorted(A)

# ---------- 1. 分档 ----------
print("=== 1. 分档准确率 ===")
print(f"{'档位':<10}{'n':>5}{'2B':>9}{'9B':>9}{'delta':>9}")
for tier in ("easy", "original", "hard"):
    sel = [i for i in ids if i.startswith(tier)]
    ca = sum(1 for i in sel if A[i]["correct"])
    cb = sum(1 for i in sel if B[i]["correct"])
    print(f"{tier:<10}{len(sel):>5}{ca/len(sel):>9.3f}{cb/len(sel):>9.3f}{(cb-ca)/len(sel):>+9.3f}")

# ---------- 2. 置信度分箱 ----------
print("\n=== 2. 置信度分箱（max prob）===")
print(f"{'区间':<14}{'n':>5}{'2B acc':>9}{'9B acc':>9}{'2B conf':>9}{'9B conf':>9}")
for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)]:
    sel = [i for i in ids if lo <= conf(A[i]) < hi or lo <= conf(B[i]) < hi]
    sub = [i for i in ids if lo <= conf(B[i]) < hi]
    if not sub:
        continue
    aa = sum(1 for i in sub if A[i]["correct"]) / len(sub)
    bb = sum(1 for i in sub if B[i]["correct"]) / len(sub)
    print(f"[{lo:.2f},{hi:.2f}){'':<3}{len(sub):>5}{aa:>9.3f}{bb:>9.3f}"
          f"{mean(conf(A[i]) for i in sub):>9.3f}{mean(conf(B[i]) for i in sub):>9.3f}")

# ---------- 3. 高置信度错误 ----------
print("\n=== 3. 高置信度却答错（conf>=0.9）===")
for name, M in (("2B", A), ("9B", B)):
    bad = [i for i in ids if conf(M[i]) >= 0.9 and not M[i]["correct"]]
    print(f"  {name}: {len(bad)} 条 {bad[:8]}")

# ---------- 4. 翻转 ----------
onlyB = [i for i in ids if B[i]["correct"] and not A[i]["correct"]]
onlyA = [i for i in ids if A[i]["correct"] and not B[i]["correct"]]
both_wrong = [i for i in ids if not A[i]["correct"] and not B[i]["correct"]]
print(f"\n=== 4. 翻转 ===")
print(f"  只有 9B 对: {len(onlyB)} 条")
print(f"  只有 2B 对: {len(onlyA)} 条  {onlyA}")
print(f"  两个都错:   {len(both_wrong)} 条")
print("\n  只有 9B 对的，按 family 分布:")
for k, v in Counter(tasks[i]["family"] for i in onlyB).most_common():
    print(f"    {k:<18} {v}")

# ---------- 5. 共同错误 ----------
print(f"\n=== 5. 两个模型都错的 {len(both_wrong)} 条 ===")
for k, v in Counter(tasks[i]["family"] for i in both_wrong).most_common():
    print(f"  {k:<18} {v}")

# ---------- 6. 延迟 ----------
print("\n=== 6. 延迟对比 ===")
for name, M in (("2B", A), ("9B", B)):
    lat = sorted(M[i]["latency_s"] for i in ids if M[i].get("latency_s") is not None)
    n = len(lat)
    print(f"  {name}: p50={lat[n//2]:.3f}s  p90={lat[int(n*0.9)]:.3f}s  max={lat[-1]:.3f}s")

# ---------- 7. 校准误差谁更稳 ----------
print("\n=== 7. 按档位看 ECE ===")
for name, M in (("2B", A), ("9B", B)):
    for tier in ("easy", "original", "hard"):
        sel = [i for i in ids if i.startswith(tier)]
        bins = {}
        for i in sel:
            c = conf(M[i])
            k = min(int(c * 10), 9)
            bins.setdefault(k, []).append((c, M[i]["correct"]))
        ece = 0.0
        for k, arr in bins.items():
            mc = mean(x[0] for x in arr)
            ac = sum(1 for x in arr if x[1]) / len(arr)
            ece += len(arr) / len(sel) * abs(mc - ac)
        print(f"  {name} {tier:<9} ECE={ece:.4f}")

#!/usr/bin/env python3
"""对 Decider-2B 基线结果做二次分析：题型、置信度、错误模式、延迟、选项数。"""

import json
from collections import Counter, defaultdict
from statistics import mean

TASKS = "data_tasks.jsonl"
RESULTS = "results/results.jsonl"

tasks = {json.loads(line)["id"]: json.loads(line) for line in open(TASKS, encoding="utf-8")}
res = [json.loads(line) for line in open(RESULTS, encoding="utf-8")]


def conf(r):
    p = r.get("probs") or {}
    return max(p.values()) if p else None


def qtype(r):
    return tasks[r["task_id"]]["question"]["type"]


# ---------- 1. 按题型 ----------
print("=== 1. 按题型 ===")
by_q = defaultdict(list)
for r in res:
    if r["ok"]:
        by_q[qtype(r)].append(r)
for q, rs in sorted(by_q.items()):
    c = sum(1 for r in rs if r["correct"])
    print(f"  {q:<7} n={len(rs):>3}  acc={c/len(rs):.3f}  ({c}/{len(rs)})  mean_conf={mean(conf(r) for r in rs):.3f}")

# ---------- 2. 置信度分箱 ----------
print("\n=== 2. 置信度分箱（max prob）===")
print("  区间            n     acc    mean_conf   gap(conf-acc)")
for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)]:
    sub = [r for r in res if r["ok"] and conf(r) is not None and lo <= conf(r) < hi]
    if not sub:
        print(f"  [{lo:.2f},{hi:.2f})     0        -         -            -")
        continue
    c = sum(1 for r in sub if r["correct"]) / len(sub)
    mc = mean(conf(r) for r in sub)
    print(f"  [{lo:.2f},{hi:.2f})  {len(sub):>4}   {c:.3f}    {mc:.3f}      {mc-c:+.3f}")

# ---------- 3. 高置信度子集能不能当护栏 ----------
print("\n=== 3. 阈值路由模拟 ===")
for th in (0.9, 0.95, 0.99):
    sub = [r for r in res if r["ok"] and (conf(r) or 0) >= th]
    if not sub:
        print(f"  threshold {th}: 0 条")
        continue
    c = sum(1 for r in sub if r["correct"])
    print(f"  conf>={th}: 覆盖 {len(sub)}/231 ({len(sub)/231:.1%})  准确率 {c/len(sub):.3f}  错 {len(sub)-c} 条")
    rest = [r for r in res if r["ok"] and (conf(r) or 0) < th]
    if rest:
        cr = sum(1 for r in rest if r["correct"])
        print(f"            剩余 {len(rest)} 条准确率 {cr/len(rest):.3f}")

# ---------- 4. 错误样本 ----------
print("\n=== 4. 错误样本（按置信度降序，取前 25）===")
wrong = [r for r in res if r["ok"] and not r["correct"]]
print(f"  共 {len(wrong)} 条错误\n")
print(f"  {'task_id':<34} {'family':<17} {'conf':>6}  expected -> predicted")
for r in sorted(wrong, key=lambda x: -(conf(x) or 0))[:25]:
    t = tasks[r["task_id"]]
    print(f"  {r['task_id']:<34} {t['family']:<17} {conf(r):>6.3f}  {str(t['expected'])[:12]:<12} -> {str(r['predicted'])[:20]}")

# ---------- 5. 高置信度错误（最危险的一类） ----------
print("\n=== 5. 高置信度却答错（conf>=0.9）===")
bad = [r for r in wrong if (conf(r) or 0) >= 0.9]
print(f"  共 {len(bad)} 条")
for r in sorted(bad, key=lambda x: -(conf(x) or 0)):
    t = tasks[r["task_id"]]
    print(f"  {conf(r):.3f}  {r['task_id']:<34} {t['family']:<16} expected={t['expected']} predicted={r['predicted']}")

# ---------- 6. 延迟 ----------
print("\n=== 6. 延迟 ===")
lat = sorted(r["latency_s"] for r in res if r["ok"] and r["latency_s"] is not None)
n = len(lat)
print(f"  n={n}  min={lat[0]:.3f}  p50={lat[n//2]:.3f}  p90={lat[int(n*0.9)]:.3f}  p99={lat[int(n*0.99)]:.3f}  max={lat[-1]:.3f}")
slow = sorted(res, key=lambda r: -(r["latency_s"] or 0))[:5]
for r in slow:
    it = (r.get("usage") or {}).get("input_tokens")
    print(f"  {r['latency_s']:.3f}s  input_tokens={it}  {r['task_id']}")

# ---------- 7. 选项数 vs 准确率 ----------
print("\n=== 7. choice 题：选项数 vs 准确率 ===")
by_k = defaultdict(list)
for r in res:
    if r["ok"] and qtype(r) == "choice":
        k = len(tasks[r["task_id"]]["labels"])
        by_k[k].append(r)
for k in sorted(by_k):
    rs = by_k[k]
    c = sum(1 for r in rs if r["correct"])
    print(f"  {k:>2} 选项: n={len(rs):>3}  acc={c/len(rs):.3f}  随机基线={1/k:.3f}")

# ---------- 8. 预测分布是否偏斜 ----------
print("\n=== 8. noul 题预测分布 ===")
yes = no = 0
for r in res:
    if r["ok"] and qtype(r) == "noul":
        if r["predicted"] == "yes":
            yes += 1
        elif r["predicted"] == "no":
            no += 1
print(f"  预测 yes={yes}  no={no}  (实际 yes={sum(1 for r in res if r['ok'] and qtype(r)=='noul' and tasks[r['task_id']]['expected']=='yes')})")

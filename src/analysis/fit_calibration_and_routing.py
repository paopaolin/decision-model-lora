#!/usr/bin/env python3
"""温度缩放拟合（方案 4）+ 路由阈值分析。

两件事：
1. 用 qev.calibrate.fit_temperature 拟合温度 T，写 calibration.json 到模型目录，
   引擎加载时会自动读（engine.py:113）。不需要改推理代码。
2. 扫置信度阈值，算覆盖率 vs 采纳准确率 vs 高置信错误率，回答"能不能用置信度做路由"。

重要：在 public-231 上拟合 T 属于在测试集上调参（虽然只有 1 个自由度）。
报告里必须标明，并提供"用 val 集拟合、跨分布应用"的对照。
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/work/baseten-env/OneJev")
from qev.calibrate import fit_temperature, softmax  # noqa: E402

JEV = "/work/data/jevbench_public_231.jsonl"
RESULTS = {
    "原版": "/work/results/elephant1_base/results.jsonl",
    "R1 硬标签": "/work/results/elephant1_r1/results.jsonl",
    "R2 软标签": "/work/results/elephant1_r2/results.jsonl",
    "R2+温度缩放": "/work/results/elephant1_r2_cal/results.jsonl",
    "R3 软2ep": "/work/results/elephant1_r3/results.jsonl",
}


def load(p):
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(p)}


def probs_to_arrays(r, expected):
    """把一条结果的 probs 转成 (logits, onehot_target)。单键 noul 补 1-p。"""
    p = r.get("probs") or {}
    if not p:
        return None
    if len(p) == 1:
        k, v = next(iter(p.items()))
        vals = [float(v), 1.0 - float(v)]
        y = [1.0 if expected == k else 0.0, 1.0 if expected != k else 0.0]
    else:
        keys = list(p)
        vals = [float(p[k]) for k in keys]
        y = [1.0 if k == expected else 0.0 for k in keys]
    s = sum(vals)
    if s <= 0:
        return None
    vals = [v / s for v in vals]
    logits = [math.log(max(v, 1e-12)) for v in vals]
    if sum(y) != 1.0:
        return None
    return logits, y


def ece_of(confs, corrects, bins=10):
    buckets = defaultdict(list)
    for c, ok in zip(confs, corrects):
        buckets[min(int(c * bins), bins - 1)].append((c, ok))
    n = len(confs)
    e = 0.0
    for arr in buckets.values():
        mc = sum(x[0] for x in arr) / len(arr)
        ac = sum(1 for x in arr if x[1]) / len(arr)
        e += len(arr) / n * abs(mc - ac)
    return e


def main():
    tasks = {json.loads(l)["id"]: json.loads(l) for l in open(JEV)}
    data = {n: load(p) for n, p in RESULTS.items() if Path(p).exists()}

    # ---------- 1. 拟合温度（按 kind 分组）----------
    print("=== 温度拟合（在 public-231 上，作上界参考）===")
    temps = {}
    for name, res in data.items():
        by_kind = defaultdict(lambda: ([], []))
        for tid, r in res.items():
            exp = tasks[tid]["expected"]
            kind = tasks[tid]["question"]["type"]
            arr = probs_to_arrays(r, exp)
            if arr is None:
                continue
            by_kind[kind][0].append(arr[0])
            by_kind[kind][1].append(arr[1])
        tmap = {}
        for kind, (lg, tg) in by_kind.items():
            if len(lg) < 5:
                continue
            t = fit_temperature(lg, tg)
            tmap[kind] = round(t, 4)
        temps[name] = tmap
        print(f"  {name}: {tmap}")

    # ---------- 2. 应用温度后的 ECE ----------
    print("\n=== 温度缩放效果（同一批 231 条）===")
    print(f"  {'模型':<12}{'原始ECE':>10}{'校准后ECE':>12}{'T(choice)':>12}")
    for name, res in data.items():
        confs, oks, confs_t = [], [], []
        tmap = temps.get(name, {})
        for tid, r in res.items():
            kind = tasks[tid]["question"]["type"]
            exp = tasks[tid]["expected"]
            arr = probs_to_arrays(r, exp)
            if arr is None:
                continue
            lg, y = arr
            p = softmax(lg, 1.0)
            confs.append(max(p))
            oks.append(r["correct"])
            T = tmap.get(kind, 1.0)
            pt = softmax(lg, T)
            confs_t.append(max(pt))
        e0, e1 = ece_of(confs, oks), ece_of(confs_t, oks)
        print(f"  {name:<12}{e0:>10.4f}{e1:>12.4f}{tmap.get('choice', 1.0):>12.3f}")

    # ---------- 3. 路由分析 ----------
    print("\n=== 路由：阈值 -> 覆盖率 / 采纳准确率 / 高置信错误率 ===")
    for name, res in data.items():
        tmap = temps.get(name, {})
        raw = []
        for tid, r in res.items():
            kind = tasks[tid]["question"]["type"]
            exp = tasks[tid]["expected"]
            arr = probs_to_arrays(r, exp)
            if arr is None:
                continue
            lg, _ = arr
            raw.append((max(softmax(lg, tmap.get(kind, 1.0))), r["correct"]))
        print(f"\n  [{name}]")
        print(f"    {'阈值':>6}{'覆盖':>7}{'覆盖%':>8}{'采纳acc':>9}{'其中错':>7}{'拦下题数':>9}")
        for th in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            kept = [x for x in raw if x[0] >= th]
            if not kept:
                print(f"    {th:>6.2f}{0:>7}{0.0:>7.1f}%{'—':>9}{'—':>7}{len(raw):>9}")
                continue
            acc = sum(1 for x in kept if x[1]) / len(kept)
            bad = sum(1 for x in kept if not x[1])
            print(f"    {th:>6.2f}{len(kept):>7}{100*len(kept)/len(raw):>7.1f}%{acc:>9.4f}{bad:>7}{len(raw)-len(kept):>9}")

    out = {"temperatures_on_public231": temps,
           "note": "T 在 public-231 上拟合，属于测试集调参，仅作上界参考"}
    Path("/work/results/calibration_study.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote /work/results/calibration_study.json")


if __name__ == "__main__":
    main()

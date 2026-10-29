#!/usr/bin/env python3
"""把散落在 results/ 下的评测产物汇总成仓库可读的几张表。

输入（相对仓库根目录）：
  results/<run>/summary.json
  results/<run>/results.jsonl
  data_tasks.jsonl        231 条的 family / tier 元信息

输出：
  results/summary_all_models.csv
  results/per_family.csv
  results/routing_curve.csv
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

# 展示名 -> 结果目录
RUNS = [
    ("decider-2b", "decider2b_base"),
    ("elephant-1-9b", "elephant1_base"),
    ("elephant-1-r1-hard", "elephant1_r1"),
    ("elephant-1-r2-soft", "elephant1_r2"),
    ("elephant-1-r2-soft-temp", "elephant1_r2_cal"),
    ("elephant-1-r3-soft2ep", "elephant1_r3"),
    ("elephant-1-r3-soft2ep-temp", "elephant1_r3_cal"),
    ("mammoth-1-27b", "mammoth1_base"),
]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    tasks = {r["id"]: r for r in load_jsonl(ROOT / "data_tasks.jsonl")}

    # ---------- 总表 ----------
    rows = []
    present = []
    for name, d in RUNS:
        sp = RES / d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        present.append((name, d, s))
        rows.append({
            "model": name,
            "run_dir": d,
            "accuracy": round(s["accuracy"], 4),
            "n_correct": s["n_correct"],
            "n_scorable": s["n_scorable"],
            "brier": round(s["brier_mean"], 4),
            "ece": round(s["ece"]["ece"], 4),
            "p50_s": round(s["latency"]["p50_s"], 3),
            "p95_s": round(s["latency"]["p95_s"], 3),
        })
    with (RES / "summary_all_models.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"summary_all_models.csv: {len(rows)} rows")

    # ---------- 分族 ----------
    fam_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for name, d, _ in present:
        rp = RES / d / "results.jsonl"
        if not rp.exists():
            continue
        for r in load_jsonl(rp):
            fam = tasks.get(r["task_id"], {}).get("family", "?")
            fam_stats[fam][name][0] += 1
            if r["correct"]:
                fam_stats[fam][name][1] += 1
    model_names = [n for n, _, _ in present]
    with (RES / "per_family.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "n"] + model_names)
        for fam in sorted(fam_stats):
            n = max(v[0] for v in fam_stats[fam].values())
            line = [fam, n]
            for m in model_names:
                c, ok = fam_stats[fam][m]
                line.append(round(ok / c, 3) if c else "")
            w.writerow(line)
    print(f"per_family.csv: {len(fam_stats)} families")

    # ---------- 路由曲线 ----------
    with (RES / "routing_curve.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "threshold", "n_adopted", "coverage_pct",
                    "adopted_accuracy", "n_errors", "n_rejected"])
        for name, d, _ in present:
            rp = RES / d / "results.jsonl"
            if not rp.exists():
                continue
            data = []
            for r in load_jsonl(rp):
                p = r.get("probs") or {}
                if not p:
                    continue
                data.append((max(p.values()), r["correct"]))
            for th in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
                kept = [x for x in data if x[0] >= th]
                if not kept:
                    w.writerow([name, th, 0, 0.0, "", 0, len(data)])
                    continue
                acc = sum(1 for x in kept if x[1]) / len(kept)
                err = sum(1 for x in kept if not x[1])
                w.writerow([name, th, len(kept), round(100 * len(kept) / len(data), 1),
                            round(acc, 4), err, len(data) - len(kept)])
    print("routing_curve.csv done")


if __name__ == "__main__":
    main()

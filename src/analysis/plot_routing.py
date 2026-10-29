#!/usr/bin/env python3
"""画路由曲线：覆盖率 vs 采纳准确率。

这是决策模型部署时最关心的一张图——横轴是"直接采纳多少比例的请求"，
纵轴是"采纳的这部分有多准"。曲线越靠右上越好。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "routing_curve.csv"
OUT = ROOT / "docs" / "figures" / "routing_curve.png"

STYLE = {
    "elephant-1-9b": ("Baseline elephant-1 9B", "#7f7f7f", "-", "o"),
    "elephant-1-r1-hard": ("R1 hard label (calibration broken)", "#d62728", "-", "s"),
    "elephant-1-r3-soft2ep-temp": ("R3 soft label + temp scaling (ours)", "#1f77b4", "-", "^"),
    "mammoth-1-27b": ("Mammoth-1 27B (upper bound)", "#2ca02c", "-", "D"),
}


def main():
    rows = list(csv.DictReader(open(CSV)))
    series = {}
    for r in rows:
        if r["model"] not in STYLE or not r["coverage_pct"]:
            continue
        series.setdefault(r["model"], []).append(
            (float(r["coverage_pct"]), float(r["adopted_accuracy"]) * 100, float(r["threshold"]))
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    for model, pts in series.items():
        pts.sort()
        label, color, ls, marker = STYLE[model]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, ls, color=color, marker=marker, markersize=4.5,
                linewidth=1.6, label=label, clip_on=False)

    # 标注 0.9 阈值（部署最常用的工作点）
    target = series.get("elephant-1-r3-soft2ep-temp")
    if target:
        pt = [p for p in target if abs(p[2] - 0.9) < 1e-6]
        if pt:
            x, y, _ = pt[0]
            ax.scatter([x], [y], s=90, facecolors="none", edgecolors="#1f77b4", linewidths=1.8, zorder=5)
            ax.annotate(f"threshold 0.9: coverage {x:.1f}%, acc {y:.2f}%",
                        xy=(x, y), xytext=(x - 17, y - 1.4),
                        fontsize=7.5, color="#1f77b4",
                        arrowprops=dict(arrowstyle="-", color="#1f77b4", linewidth=0.7,
                                        shrinkA=0, shrinkB=4))

    ax.set_xlabel("Coverage: share of requests auto-adopted (%)", fontsize=9)
    ax.set_ylabel("Accuracy on adopted subset (%)", fontsize=9)
    ax.set_title("Routing curve: how much can be auto-accepted at what accuracy",
                 fontsize=10.5, pad=10)
    ax.set_xlim(45, 100)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.8, loc="lower right", frameon=False)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

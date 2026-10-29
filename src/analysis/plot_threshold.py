#!/usr/bin/env python3
"""画阈值-精度曲线（risk-coverage curve），对比简单流量与难流量。

回答一个部署问题：同一个 0.9 阈值，在两种流量上各自意味着什么？
左图简单流量：四条线都挤在 99% 以上，0.9 随便用。
右图难流量：同样 0.9，采纳精度掉到 93%，要保 98% 得把阈值推到 0.99。

横轴是覆盖率（自动采纳多少比例的请求），纵轴是采纳部分的精度，越靠右上越好。
两个面板的横轴范围不同，不要跨面板比水平位置。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "threshold_curves.csv"
OUT = ROOT / "docs" / "figures" / "threshold_curves.png"

STYLE = {
    "9B base": ("Baseline elephant-1 9B", "#7f7f7f", "o"),
    "9B+R3": ("9B + R3 (fine-tuned)", "#d62728", "s"),
    "9B+R3+T": ("9B + R3 + temperature", "#1f77b4", "^"),
    "27B": ("Mammoth-1 27B", "#2ca02c", "D"),
}
PANELS = [
    ("random", "(a) Easy traffic — random negatives", 87.5, 100.8, 97.6, 100.1),
    ("hard", "(b) Hard traffic — confusable negatives", 16, 67, 82, 100.1),
]
MIN_ADOPT = 30  # 采纳量太少时精度是噪声，不画


def load():
    data = {}
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["adopted"]) < MIN_ADOPT:
                continue
            data.setdefault((r["set"], r["model"]), []).append(
                (float(r["coverage_pct"]), float(r["precision_pct"]), float(r["threshold"]))
            )
    for k in data:
        data[k].sort(key=lambda p: p[0])
    return data


def crossing_at(pts, target=98.0):
    """满足精度的点里覆盖率最大的那个（曲线从右往左第一次跨过目标线）。"""
    for cov, prec, thr in sorted(pts, key=lambda p: -p[0]):
        if prec >= target:
            return cov, prec, thr
    return None


def at_threshold(pts, t=0.9):
    return next((p for p in pts if abs(p[2] - t) < 1e-9), None)


def main():
    data = load()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=300)

    for ax, (setname, title, x0, x1, y0, y1) in zip(axes, PANELS):
        for model, (label, color, marker) in STYLE.items():
            pts = data.get((setname, model))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    "-", color=color, marker=marker, markersize=3.2,
                    linewidth=1.6, label=label)

            # 空心圈标出 0.9 这个默认工作点
            p90 = at_threshold(pts)
            if p90:
                ax.scatter([p90[0]], [p90[1]], s=72, facecolors="none",
                           edgecolors=color, linewidths=1.7, zorder=6)

        ax.axhline(98, color="#444444", linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.75, zorder=2)
        if setname == "random":
            # 左图曲线在右侧会压到这条线上，标注放左边空白区
            ax.text(x0 + 0.3, 98.35, "98% precision target", fontsize=7, color="#444444",
                    ha="left", va="bottom")
        else:
            ax.text(x1 - 0.5, 98.4, "98% precision target", fontsize=7, color="#444444",
                    ha="right", va="bottom")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.set_xlabel("Coverage: share of requests auto-adopted (%)", fontsize=8.5)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Precision on adopted subset (%)", fontsize=8.5)
    axes[0].text(87.9, 98.8,
                 "hollow marker = 0.9 threshold\nall four models stay above 99%",
                 fontsize=6.9, color="#333333", va="bottom")

    # 右图：把 0.9 的落点和各模型达 98% 所需阈值放进一个信息框，不用长箭头
    lines = []
    p90 = at_threshold(data.get(("hard", "9B+R3"), []))
    if p90:
        lines.append(f"at 0.9 (default)  R3: {p90[0]:.1f}% cov / {p90[1]:.2f}% prec")
        base90 = at_threshold(data.get(("hard", "9B base"), []))
        if base90:
            lines.append(f"                  base: {base90[0]:.1f}% cov / {base90[1]:.2f}% prec")
    lines.append("")
    lines.append("threshold needed for 98% precision:")
    for model in ("9B base", "9B+R3", "9B+R3+T", "27B"):
        c = crossing_at(data.get(("hard", model), []))
        tag = f"{c[2]:.3f}  (cov {c[0]:.1f}%)" if c else "no operating point"
        lines.append(f"  {model:<8}{tag}")
    axes[1].text(16.8, 83.0, "\n".join(lines),
                 fontsize=6.6, va="bottom", ha="left", family="DejaVu Sans Mono",
                 bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                           edgecolor="#bbbbbb", linewidth=0.6, alpha=0.95))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(pad=1.1, rect=(0, 0.05, 1, 1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

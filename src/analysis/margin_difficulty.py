#!/usr/bin/env python3
"""能不能不靠标签判断"这条是难流量"？

思路：难流量 ≠ 低置信度。模型可以在难题上给出很高的置信度（这正是过度自信的定义）。
试一个更贴近"难"的代理指标：top-1 与 top-2 的概率差（margin）。
margin 小 = 模型在两个选项之间摇摆 = 这条本身模糊/困难。

如果 margin 能预测对错，就可以做分层阈值：margin 小的时候要求更高置信度才放行。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETS = {"random": ROOT / "results" / "crossbench", "hard": ROOT / "results" / "crossbench_hard"}
MODELS = {"9B base": "elephant1_base", "9B+R3": "elephant1_r3", "27B": "mammoth1_base"}


def load(setdir, name):
    rows = {}
    with (setdir / name / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p = r.get("probs") or {}
            if not p:
                continue
            top = sorted(p.values(), reverse=True)
            r["_margin"] = top[0] - (top[1] if len(top) > 1 else 0.0)
            rows[r["task_id"]] = r
    return rows


def quantile_split(rows, frac=0.5):
    ms = sorted(r["_margin"] for r in rows.values())
    cut = ms[int(len(ms) * frac)]
    low = [r for r in rows.values() if r["_margin"] < cut]
    high = [r for r in rows.values() if r["_margin"] >= cut]
    return cut, low, high


def err_rate(rs):
    return sum(1 for r in rs if not r["correct"]) / len(rs) if rs else 0.0


def main():
    for sname, sdir in SETS.items():
        print("=" * 88)
        print(f"[{sname}] margin 分半：错题率对比")
        print("=" * 88)
        for m, name in MODELS.items():
            rows = load(sdir, name)
            cut, low, high = quantile_split(rows)
            print(f"  {m:<9} 分界 margin={cut:.3f}  "
                  f"低 margin 错题率 {err_rate(low)*100:5.2f}%（n={len(low)}）  "
                  f"高 margin {err_rate(high)*100:5.2f}%（n={len(high)}）  "
                  f"倍数 {err_rate(low)/max(1e-9, err_rate(high)):.1f}×")
        print()

    # 分层阈值策略：margin 低的那半要求更高置信度
    print("=" * 88)
    print("分层阈值策略在难负集上的实测（低 margin 要求更高阈值）")
    print("=" * 88)
    policies = [(0.90, 0.90), (0.95, 0.90), (0.97, 0.93), (0.99, 0.95)]
    for m, name in MODELS.items():
        rows = load(SETS["hard"], name)
        cut, _, _ = quantile_split(rows)
        print(f"  [{m}]  margin 分界 {cut:.3f}")
        for t_low, t_high in policies:
            adopted = err = 0
            for r in rows.values():
                p = max(r["probs"].values())
                need = t_high if r["_margin"] >= cut else t_low
                if p >= need:
                    adopted += 1
                    err += 0 if r["correct"] else 1
            print(f"    低 {t_low:.2f} / 高 {t_high:.2f}"
                  f" → 覆盖 {adopted/len(rows)*100:5.1f}%  精度 {(adopted-err)/adopted*100:6.2f}%  错 {err}")
        print()


if __name__ == "__main__":
    main()

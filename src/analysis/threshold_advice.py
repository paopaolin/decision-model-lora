#!/usr/bin/env python3
"""阈值选点：给定目标采纳精度，各模型需要多高的置信度阈值，以及那个阈值下能采纳多少。

回答的是部署问题：同一个 0.9，在简单流量上够用，在难流量上不够。
缺的是"不够多少、要调到多少、代价是什么"。

输出：results/threshold_advice.json + 终端表
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETS = {
    "random": ROOT / "results" / "crossbench",
    "hard": ROOT / "results" / "crossbench_hard",
}
MODELS = {"9B base": "elephant1_base", "9B+R3": "elephant1_r3", "27B": "mammoth1_base"}
T_TEMP = 1.3096
TARGETS = [0.98, 0.99, 0.995]
GRID = [round(0.50 + 0.005 * i, 3) for i in range(101)]  # 0.500 ~ 1.000
MIN_ADOPT = 30  # 低于这个采纳量，精度估计不可靠


def load(setdir: Path):
    res = {}
    for m, name in MODELS.items():
        rows = {}
        with (setdir / name / "results.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[r["task_id"]] = r
        res[m] = rows
    return res


def probs_of(r, temp=None):
    p = r.get("probs") or {}
    if not p:
        return {}
    if temp:
        z = {k: v ** (1.0 / temp) for k, v in p.items()}
        s = sum(z.values())
        return {k: v / s for k, v in z.items()}
    return p


def sweep(rows, temp=None):
    out = []
    for t in GRID:
        adopted = err = 0
        for r in rows.values():
            p = probs_of(r, temp)
            if p and max(p.values()) >= t:
                adopted += 1
                if not r["correct"]:
                    err += 1
        if adopted:
            out.append({"threshold": t, "adopted": adopted,
                        "coverage": adopted / len(rows),
                        "precision": (adopted - err) / adopted,
                        "errors": err})
    return out


def pick(curve, target):
    """在采纳量足够的前提下，取满足目标精度的最低阈值（= 覆盖率最高的那个点）。"""
    ok = [c for c in curve if c["precision"] >= target and c["adopted"] >= MIN_ADOPT]
    return ok[0] if ok else None


def main():
    out = {}
    for sname, sdir in SETS.items():
        rows_by_model = load(sdir)
        print("=" * 88)
        print(f"[{sname}]  目标精度 → 所需最低阈值 / 覆盖率 / 采纳条数")
        print("=" * 88)
        out[sname] = {}
        variants = [(m, rows_by_model[m], None) for m in MODELS]
        variants.append(("9B+R3+T", rows_by_model["9B+R3"], T_TEMP))

        for label, rows, temp in variants:
            curve = sweep(rows, temp)
            at90 = next((c for c in curve if abs(c["threshold"] - 0.9) < 1e-9), None)
            out[sname][label] = {"at_0.9": at90, "targets": {}}
            head = f"  {label:<9}"
            if at90:
                head += f" @0.9: 覆盖 {at90['coverage']*100:5.1f}% / 精度 {at90['precision']*100:6.2f}% / 错 {at90['errors']:>3}"
            print(head)
            for tg in TARGETS:
                c = pick(curve, tg)
                if not c:
                    print(f"            目标 {tg*100:.1f}%：无法达到（样本不足或精度上限不够）")
                    continue
                out[sname][label]["targets"][f"{tg:.3f}"] = c
                print(f"            目标 {tg*100:5.1f}% → 阈值 {c['threshold']:.3f}"
                      f" / 覆盖 {c['coverage']*100:5.1f}% / 采纳 {c['adopted']:>3} 条")
            print()

    (ROOT / "results" / "threshold_advice.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("写入 results/threshold_advice.json")


if __name__ == "__main__":
    main()

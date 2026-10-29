#!/usr/bin/env python3
"""跨 benchmark 泛化评测的分析：三个模型 × 四个公开数据集。

准确率在这个集合上已经饱和（97.8-98.3%），所以主结论不能只看 argmax。
这里同时算四个对饱和度不敏感的指标：
  - gold 概率：模型给正确标签的平均概率（准确率相同时区分"勉强对"和"很确定"）
  - NLL：整体概率质量
  - 配对翻转：同一道题上两个模型的对错差异（McNemar 口径）
  - 0.9 阈值路由：跨域数据上置信度还能不能用来筛选

输出：终端表格 + results/crossbench/crossbench_summary.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CB = ROOT / "results" / "crossbench"

MODELS = {
    "elephant-1 9B (base)": "elephant1_base",
    "elephant-1 9B + R3": "elephant1_r3",
    "mammoth-1 27B": "mammoth1_base",
}
DATASETS = ["banking77", "clinc150", "ai2_arc", "imdb"]


def load(name: str):
    tasks = {}
    with (CB / "all.jsonl").open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tasks[t["id"]] = t
    results = {}
    with (CB / name / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            results[r["task_id"]] = r
    return tasks, results


FAM: dict[str, str] = {}

SHORT = {
    "elephant-1 9B (base)": "9B base",
    "elephant-1 9B + R3": "9B+R3",
    "elephant-1 9B + R3 + T": "9B+R3+T",
    "mammoth-1 27B": "27B",
}


def dataset_of(task_id: str) -> str:
    return FAM.get(task_id) or task_id.split("-")[1]


def metrics(tasks, results, ds=None):
    ids = [i for i in results if ds is None or dataset_of(i) == ds]
    n = len(ids)
    correct = sum(1 for i in ids if results[i]["correct"])
    gold_p, nll, conf = [], [], []
    lat = []
    hi_n = hi_err = 0
    for i in ids:
        r = results[i]
        probs = r.get("probs") or {}
        exp = tasks[i]["expected"]
        pg = probs.get(exp, 0.0)
        gold_p.append(pg)
        nll.append(-math.log(max(pg, 1e-12)))
        c = max(probs.values()) if probs else 0.0
        conf.append(c)
        if r.get("latency_s"):
            lat.append(r["latency_s"])
        if c >= 0.9:
            hi_n += 1
            if not r["correct"]:
                hi_err += 1
    return {
        "n": n,
        "acc": correct / n,
        "n_correct": correct,
        "gold_prob": sum(gold_p) / n,
        "nll": sum(nll) / n,
        "mean_conf": sum(conf) / n,
        "p50_lat": sorted(lat)[len(lat) // 2] if lat else float("nan"),
        "hi_n": hi_n,
        "hi_err": hi_err,
        "hi_cov": hi_n / n,
        "hi_prec": (hi_n - hi_err) / hi_n if hi_n else float("nan"),
    }


def main():
    res = {}
    tasks = None
    for label, name in MODELS.items():
        t, r = load(name)
        tasks = t  # 三个模型共用同一份任务定义
        res[label] = r
    FAM.update({i: t["family"] for i, t in tasks.items()})

    # 离线复现部署配置：R3 + 在 JevBench 上拟合的温度 T=1.3096。
    # 温度缩放 p' ∝ p^(1/T)：T>1 是软化（压置信度），不是锐化——方向搞反会得到相反结论。
    # 单调变换，argmax 不变，所以准确率必须与 R3 完全一致；变的只有置信度，
    # 正好用来检验"跨域时这个在 JevBench 上拟合的温度还管不管用"。
    T = 1.3096
    scaled = {}
    for tid, r in res["elephant-1 9B + R3"].items():
        probs = r.get("probs") or {}
        if probs:
            z = {k: v ** (1.0 / T) for k, v in probs.items()}
            s = sum(z.values())
            probs = {k: v / s for k, v in z.items()}
        scaled[tid] = {**r, "probs": probs}
    res["elephant-1 9B + R3 + T"] = scaled

    out = {"overall": {}, "per_dataset": {}, "flips": {}}

    print("=" * 96)
    print("总体（1200 条，4 个公开数据集）")
    print("=" * 96)
    print(f"{'模型':<24}{'acc':>8}{'对':>6}{'gold概率':>10}{'NLL':>8}{'均置信':>8}"
          f"{'≥0.9覆盖':>10}{'采纳精度':>10}{'误放':>6}")
    order = ["elephant-1 9B (base)", "elephant-1 9B + R3",
             "elephant-1 9B + R3 + T", "mammoth-1 27B"]
    for label in order:
        m = metrics(tasks, res[label])
        out["overall"][label] = m
        print(f"{label:<24}{m['acc']*100:>7.2f}%{m['n_correct']:>6}"
              f"{m['gold_prob']:>10.4f}{m['nll']:>8.4f}{m['mean_conf']:>8.4f}"
              f"{m['hi_cov']*100:>9.1f}%{m['hi_prec']*100:>9.2f}%{m['hi_err']:>6}")

    print()
    print("=" * 96)
    print("分数据集准确率")
    print("=" * 96)
    header = f"{'数据集':<12}" + "".join(f"{SHORT[l]:>12}" for l in order)
    print(header)
    for ds in DATASETS:
        row = f"{ds:<12}"
        per = {}
        for label in order:
            m = metrics(tasks, res[label], ds)
            per[label] = m
            row += f"{m['acc']*100:>13.2f}%"
        out["per_dataset"][ds] = per
        print(row)

    print()
    print("=" * 96)
    print("R3 vs 基线：配对翻转（同一道题上的对错变化）")
    print("=" * 96)
    base, r3 = res["elephant-1 9B (base)"], res["elephant-1 9B + R3"]
    ids = sorted(set(base) & set(r3))
    b2w = [i for i in ids if base[i]["correct"] and not r3[i]["correct"]]
    w2b = [i for i in ids if not base[i]["correct"] and r3[i]["correct"]]
    print(f"  基线对 / R3 错：{len(b2w)}   {[(i, dataset_of(i)) for i in b2w]}")
    print(f"  基线错 / R3 对：{len(w2b)}   {[(i, dataset_of(i)) for i in w2b]}")
    print(f"  净变化：{len(w2b) - len(b2w):+d} 条 / {len(ids)}")
    # McNemar 精确检验（双侧）
    n_d = len(b2w) + len(w2b)
    if n_d:
        k = min(len(b2w), len(w2b))
        p = min(1.0, 2 * sum(math.comb(n_d, j) for j in range(k + 1)) / (2 ** n_d))
    else:
        p = 1.0
    print(f"  McNemar 精确检验 p = {p:.3f}（不一致对 {n_d} 个）")
    out["flips"] = {"base_correct_r3_wrong": b2w, "base_wrong_r3_correct": w2b,
                    "n_discordant": n_d, "mcnemar_p": p}

    print()
    print("=" * 96)
    print("27B 参照：它在 JevBench 上比 9B 高 3.9 个点，在这里高多少")
    print("=" * 96)
    b, m27 = out["overall"]["elephant-1 9B (base)"], out["overall"]["mammoth-1 27B"]
    print(f"  准确率差：{(m27['acc']-b['acc'])*100:+.2f} 个点")
    print(f"  gold 概率差：{m27['gold_prob']-b['gold_prob']:+.4f}")
    print(f"  NLL 差：{m27['nll']-b['nll']:+.4f}（越低越好，27B 明显占优）")
    print(f"  延迟：{b['p50_lat']:.3f}s -> {m27['p50_lat']:.3f}s（约 {m27['p50_lat']/b['p50_lat']:.2f} 倍）")

    print()
    print("=" * 96)
    print("饱和度的量化：这套题还能分辨多大的差异")
    print("=" * 96)
    p0 = out["overall"]["elephant-1 9B (base)"]["acc"]
    se = math.sqrt(p0 * (1 - p0) / 1200)
    print(f"  基线准确率 {p0:.4f}，二项标准误 {se*100:.2f} 个点")
    print(f"  95% 置信区间约 ±{1.96*se*100:.2f} 个点 —— 小于这个幅度的差异不可分辨")

    (CB / "crossbench_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\n写入 {CB / 'crossbench_summary.json'}")


if __name__ == "__main__":
    main()

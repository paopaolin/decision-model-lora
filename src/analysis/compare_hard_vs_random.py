#!/usr/bin/env python3
"""随机负样本 vs 难负样本：同一批查询，只换干扰项，看两套评测各自能分辨什么。

两套任务共用 1,200 条里的查询与正确标签，task id 完全一致，可逐题配对。
arc / imdb 两套完全相同（它们没有"负样本"概念），因此这两族的数字应完全一致——
这一点本身就是数据一致性的自检。

输出：results/crossbench_hard/hard_vs_random.json
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETS = {
    "random": (ROOT / "results" / "crossbench", "all.jsonl"),
    "hard": (ROOT / "results" / "crossbench_hard", "all_hard.jsonl"),
}
MODELS = {
    "9B base": "elephant1_base",
    "9B+R3": "elephant1_r3",
    "27B": "mammoth1_base",
}
FAMILIES = ["banking77", "clinc150", "ai2_arc", "imdb"]

STOP = set("a an the is are of in on at for with and or to do does did not yes no".split())


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower())
            if len(t) >= 3 and t not in STOP}


def load_tasks(setdir: Path, fname: str) -> dict:
    out = {}
    with (setdir / fname).open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            out[t["id"]] = t
    return out


def load_results(setdir: Path, name: str) -> dict:
    out = {}
    with (setdir / name / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def ece_of(tasks, results, T=None):
    """10 等宽置信度分箱的 ECE。T 给定时先做温度缩放 p^(1/T)。

    与 jevbench summarize 的口径对齐，用未缩放的结果与 summary.json 对账校验。
    """
    bins = [[] for _ in range(10)]
    for i, r in results.items():
        p = r.get("probs") or {}
        if not p:
            continue
        if T:
            z = {k: v ** (1.0 / T) for k, v in p.items()}
            s = sum(z.values())
            p = {k: v / s for k, v in z.items()}
        c = max(p.values())
        bins[min(9, int(c * 10))].append((c, 1.0 if r["correct"] else 0.0))
    tot = sum(len(b) for b in bins)
    if not tot:
        return float("nan")
    e = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(a for _, a in b) / len(b)
        e += len(b) / tot * abs(acc - conf)
    return e


def hi_conf_stats(tasks, results, T=None, thr=0.9):
    hi = err = 0
    for i, r in results.items():
        p = r.get("probs") or {}
        if not p:
            continue
        if T:
            z = {k: v ** (1.0 / T) for k, v in p.items()}
            s = sum(z.values())
            p = {k: v / s for k, v in z.items()}
        if max(p.values()) >= thr:
            hi += 1
            err += 0 if r["correct"] else 1
    return hi, err


def metrics(tasks, results, family=None):
    ids = [i for i in results if family is None or tasks[i]["family"] == family]
    n = len(ids)
    if not n:
        return None
    correct = sum(1 for i in ids if results[i]["correct"])
    gp, nll, conf = [], [], []
    hi = hi_err = 0
    near = near_n = 0
    for i in ids:
        r = results[i]
        p = r.get("probs") or {}
        exp = tasks[i]["expected"]
        pg = p.get(exp, 0.0)
        gp.append(pg)
        nll.append(-math.log(max(pg, 1e-12)))
        c = max(p.values()) if p else 0.0
        conf.append(c)
        if c >= 0.9:
            hi += 1
            hi_err += 0 if r["correct"] else 1
        if not r["correct"] and r.get("predicted"):
            near_n += 1
            # 错选的标签与正确标签是否共享实词：衡量"差一点" vs "完全跑偏"
            if toks(r["predicted"]) & toks(exp):
                near += 1
    return {
        "n": n, "acc": correct / n, "n_correct": correct,
        "gold_prob": sum(gp) / n, "nll": sum(nll) / n,
        "mean_conf": sum(conf) / n,
        "hi_cov": hi / n, "hi_err": hi_err,
        "hi_prec": (hi - hi_err) / hi if hi else float("nan"),
        "near_miss_rate": near / near_n if near_n else float("nan"),
        "n_wrong": near_n,
    }


def main():
    data = {}
    for sname, (sdir, tf) in SETS.items():
        tasks = load_tasks(sdir, tf)
        data[sname] = {"tasks": tasks,
                       "res": {m: load_results(sdir, name) for m, name in MODELS.items()}}
        print(f"{sname}: {len(tasks)} 条任务, 模型 {len(MODELS)}")

    out = {}

    print("\n" + "=" * 92)
    print("一、难负把难度拉开了多少（同一批查询，只换 7 个干扰项）")
    print("=" * 92)
    print(f"{'数据集':<14}{'随机-9B':>10}{'难负-9B':>10}{'降幅':>9}"
          f"{'随机-27B':>10}{'难负-27B':>10}{'降幅':>9}")
    for fam in FAMILIES:
        row = f"{fam:<14}"
        hfam = fam + "_hard" if fam in ("banking77", "clinc150") else fam
        for m in ("9B base", "27B"):
            a = metrics(data["random"]["tasks"], data["random"]["res"][m], fam)
            b = metrics(data["hard"]["tasks"], data["hard"]["res"][m], hfam)
            row += f"{a['acc']*100:>9.2f}%{b['acc']*100:>9.2f}%{(b['acc']-a['acc'])*100:>+8.1f}"
        print(row)

    print("\n" + "=" * 92)
    print("二、谁在这套题上更强（准确率）")
    print("=" * 92)
    out["accuracy"] = {}
    for sname in SETS:
        print(f"\n  [{sname}]")
        for m in MODELS:
            o = metrics(data[sname]["tasks"], data[sname]["res"][m])
            out["accuracy"].setdefault(m, {})[sname] = o
            b77 = metrics(data[sname]["tasks"], data[sname]["res"][m], "banking77_hard"
                          if sname == "hard" else "banking77")
            c150 = metrics(data[sname]["tasks"], data[sname]["res"][m], "clinc150_hard"
                           if sname == "hard" else "clinc150")
            print(f"    {m:<10} 总 {o['acc']*100:>6.2f}%  banking77 {b77['acc']*100:>5.1f}%"
                  f"  clinc150 {c150['acc']*100:>5.1f}%")

    print("\n  27B 相对 9B 的领先幅度：")
    for sname in SETS:
        g = (out["accuracy"]["27B"][sname]["acc"]
             - out["accuracy"]["9B base"][sname]["acc"]) * 100
        print(f"    {sname:<7} {g:+.2f} 个点")
        out.setdefault("gap_27b", {})[sname] = g

    print("\n  R3 相对基线的变化：")
    for sname in SETS:
        d = (out["accuracy"]["9B+R3"][sname]["acc"]
             - out["accuracy"]["9B base"][sname]["acc"]) * 100
        print(f"    {sname:<7} {d:+.2f} 个点")
        out.setdefault("delta_r3", {})[sname] = d

    print("\n" + "=" * 92)
    print("三、校准与路由：难负版能不能把差异放大")
    print("=" * 92)
    print(f"{'集合':<8}{'模型':<10}{'ECE':>8}{'NLL':>9}{'均置信':>9}"
          f"{'≥0.9覆盖':>10}{'采纳精度':>10}{'高置信错误':>11}")
    out["calibration"] = {}
    for sname, (sdir, _) in SETS.items():
        for m in MODELS:
            o = metrics(data[sname]["tasks"], data[sname]["res"][m])
            ece = json.loads((sdir / MODELS[m] / "summary.json").read_text())["ece"]["ece"]
            o["ece"] = ece
            out["calibration"].setdefault(sname, {})[m] = o
            print(f"{sname:<8}{m:<10}{ece:>8.4f}{o['nll']:>9.4f}{o['mean_conf']:>9.4f}"
                  f"{o['hi_cov']*100:>9.1f}%{o['hi_prec']*100:>9.2f}%{o['hi_err']:>11}")

    print("\n" + "=" * 92)
    print("三之二、部署用的温度 T=1.31 在两套题上还管不管用")
    print("=" * 92)
    print(f"{'集合':<8}{'模型':<14}{'ECE':>8}{'ECE+T':>9}{'高置信错误':>12}{'+T 后':>8}")
    for sname in SETS:
        res_r3 = data[sname]["res"]["9B+R3"]
        tasks = data[sname]["tasks"]
        e0 = ece_of(tasks, res_r3)
        e1 = ece_of(tasks, res_r3, T=1.3096)
        h0, k0 = hi_conf_stats(tasks, res_r3)
        h1, k1 = hi_conf_stats(tasks, res_r3, T=1.3096)
        print(f"{sname:<8}{'9B+R3':<14}{e0:>8.4f}{e1:>9.4f}{k0:>12}{k1:>8}")
        out.setdefault("temp_transfer", {})[sname] = {
            "ece_raw": e0, "ece_scaled": e1,
            "hi_err_raw": k0, "hi_err_scaled": k1,
            "hi_cov_raw": h0 / len(res_r3), "hi_cov_scaled": h1 / len(res_r3),
        }
    print("  校验：上面 ECE 与 jevbench summary 的差值应接近 0（分箱口径一致）")
    for sname, (sdir, _) in SETS.items():
        ref = json.loads((sdir / "elephant1_r3" / "summary.json").read_text())["ece"]["ece"]
        mine = ece_of(data[sname]["tasks"], data[sname]["res"]["9B+R3"])
        print(f"    {sname:<8} jevbench {ref:.4f} vs 本脚本 {mine:.4f}  差 {abs(ref-mine):.4f}")

    print("\n" + "=" * 92)
    print("四、错在哪：错选标签与正确标签的相似度（值高说明是「差一点」而非完全跑偏）")
    print("=" * 92)
    for sname in SETS:
        for m in MODELS:
            o = metrics(data[sname]["tasks"], data[sname]["res"][m])
            print(f"  {sname:<7}{m:<10} 错 {o['n_wrong']:>3} 条，"
                  f"其中与正确标签共享实词 {o['near_miss_rate']*100:>5.1f}%")

    print("\n" + "=" * 92)
    print("五、配对一致性自检：arc / imdb 两套应完全相同")
    print("=" * 92)
    for m in MODELS:
        for fam in ("ai2_arc", "imdb"):
            a = metrics(data["random"]["tasks"], data["random"]["res"][m], fam)
            b = metrics(data["hard"]["tasks"], data["hard"]["res"][m], fam)
            same = abs(a["acc"] - b["acc"]) < 1e-9
            print(f"  {m:<10}{fam:<10}{a['acc']*100:>7.2f}% vs {b['acc']*100:>7.2f}%  "
                  f"{'一致' if same else '不一致（异常）'}")

    # R3 vs 基线：难负上的配对翻转
    print("\n" + "=" * 92)
    print("六、R3 vs 基线在难负集上的配对翻转")
    print("=" * 92)
    base, r3 = data["hard"]["res"]["9B base"], data["hard"]["res"]["9B+R3"]
    tasks = data["hard"]["tasks"]
    ids = sorted(set(base) & set(r3))
    b2w = [i for i in ids if base[i]["correct"] and not r3[i]["correct"]]
    w2b = [i for i in ids if not base[i]["correct"] and r3[i]["correct"]]
    print(f"  基线对/R3 错 {len(b2w)} 条，基线错/R3 对 {len(w2b)} 条，净 {len(w2b)-len(b2w):+d}")
    n_d = len(b2w) + len(w2b)
    if n_d:
        k = min(len(b2w), len(w2b))
        p = min(1.0, 2 * sum(math.comb(n_d, j) for j in range(k + 1)) / (2 ** n_d))
        print(f"  McNemar 精确检验 p = {p:.3f}（不一致对 {n_d} 个）")
    from collections import Counter
    print(f"  失分分布 {dict(Counter(tasks[i]['family'] for i in b2w))}")
    print(f"  得分分布 {dict(Counter(tasks[i]['family'] for i in w2b))}")

    out["flips_hard"] = {"base_correct_r3_wrong": b2w, "base_wrong_r3_correct": w2b}
    (ROOT / "results" / "crossbench_hard" / "hard_vs_random.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""合并第一轮 targeted 集：When2Call + OneJev 筛后数据。

同时做一件事：把这批数据的"任务类型"映射到 JevBench 的 family，
算出对 public-231 的覆盖度——这决定它能不能用来提总分。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

W2C = "/work/data/subsets/when2call_train.jsonl"
ONEJEV = "/work/data/subsets/train_targeted.jsonl"
JEV = "/work/data/jevbench_public_231.jsonl"
OUT = Path("/work/data/subsets")

# 训练任务 -> 它能影响的 JevBench family
TASK_TO_FAMILY = {
    "tool_use_decision": ["tool_selection", "routing"],
    "outcome": ["judge_hard", "adequacy"],
    "outcome_forecast": ["judge_hard"],
    "check_met": ["adequacy"],
    "next_action": ["tool_selection"],
    "security": ["policy", "adversarial"],
    "which_run": ["judge_hard"],
}


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w2c-limit", type=int, default=6507)
    ap.add_argument("--onejev-limit", type=int, default=94)
    ap.add_argument("--out", default="/work/data/subsets/merged_targeted.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w2c = load(W2C)[: args.w2c_limit]
    oj = load(ONEJEV)[: args.onejev_limit]
    for r in w2c:
        r["origin"] = "when2call"
    for r in oj:
        r["origin"] = "onejev"
    merged = w2c + oj
    random.Random(args.seed).shuffle(merged)

    # ---- 配比 ----
    print(f"when2call: {len(w2c)}")
    print(f"onejev:    {len(oj)}")
    print(f"merged:    {len(merged)}")
    print(f"ratio w2c:onejev = {len(w2c)/max(1,len(merged)):.1%} : {len(oj)/max(1,len(merged)):.1%}")

    print("\n=== 按 origin × question.type ===")
    for k, v in Counter((r["origin"], r["question"]["type"]) for r in merged).most_common():
        print(f"  {k[0]:<12} {k[1]:<8} {v}")

    print("\n=== 按 origin × task ===")
    for k, v in Counter((r["origin"], r.get("task")) for r in merged).most_common():
        print(f"  {k[0]:<12} {str(k[1]):<22} {v}")

    # ---- family 覆盖 ----
    jev = [json.loads(l) for l in open(JEV)]
    jev_fam = Counter(r["family"] for r in jev)
    covered_fam = defaultdict(int)
    for r in merged:
        for fam in TASK_TO_FAMILY.get(str(r.get("task")), []):
            covered_fam[fam] += 1

    print(f"\n=== 对 JevBench 18 个 family 的覆盖（共 {len(jev)} 条）===")
    print(f"{'family':<20}{'jevtasks':>9}{'train样本':>10}  状态")
    total_jevtasks_covered = 0
    for fam, n in jev_fam.most_common():
        c = covered_fam.get(fam, 0)
        if c:
            total_jevtasks_covered += n
        status = f"覆盖 {c}" if c else "**无**"
        print(f"{fam:<20}{n:>9}{c:>10}  {status}")
    print(f"\n  JevBench 被覆盖的题数: {total_jevtasks_covered}/{len(jev)} = {total_jevtasks_covered/len(jev):.1%}")
    print(f"  未被覆盖的 family 数: {sum(1 for f in jev_fam if f not in covered_fam)}/{len(jev_fam)}")

    # ---- 写文件 ----
    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(merged)} -> {args.out}")

    summary = {
        "when2call": len(w2c), "onejev": len(oj), "merged": len(merged),
        "origin_type": {f"{a}|{b}": v for (a, b), v in Counter((r["origin"], r["question"]["type"]) for r in merged).items()},
        "origin_task": {f"{a}|{b}": v for (a, b), v in Counter((r["origin"], str(r.get("task"))) for r in merged).items()},
        "jevbench_family": dict(jev_fam),
        "family_covered_by_train": dict(covered_fam),
        "jevbench_tasks_covered": total_jevtasks_covered,
        "jevbench_coverage": round(total_jevtasks_covered / len(jev), 4),
        "uncovered_families": sorted(set(jev_fam) - set(covered_fam)),
        "mapping": TASK_TO_FAMILY,
    }
    (OUT / "merged_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary -> {OUT/'merged_summary.json'}")


if __name__ == "__main__":
    main()

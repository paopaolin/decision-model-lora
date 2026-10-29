#!/usr/bin/env python3
"""第一轮训练集拼装：targeted 臂 + control 臂 + 验证集。

targeted（按 13 号报告的配比）：
  DROP 320（temporal_numeric，全量 train）
  OneJev 447（judge_hard 相关，剔除 security/check_met 两个已满分族）
  When2Call 400（通用/防遗忘，三类均衡采样）

control（对照臂）：
  When2Call 1,167 随机（与 targeted 同量，单一来源通用数据）

已知局限：control 是单一来源，与 targeted 的来源结构不同，
因此只能回答"定向混合 vs 单一通用"，不能分离来源效应。这一点写进清单。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SRC = Path("/work/data/subsets")
OUT = Path("/work/data/round1")

ONEJEV_USEFUL = {"outcome", "outcome_forecast", "which_run"}


def load(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def dump(rows, name):
    p = OUT / name
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(rows)}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w2c-general", type=int, default=400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- 1. DROP ----
    drop_tr = load(SRC / "drop_train.jsonl")
    drop_va = load(SRC / "drop_val.jsonl")
    drop_ho = load(SRC / "drop_heldout.jsonl")
    drop_train = drop_tr[:320]

    # ---- 2. OneJev（只保留对弱项有用的 task）----
    oj_tr = [r for r in load(SRC / "train_targeted.jsonl") if r.get("task") in ONEJEV_USEFUL]
    oj_ho = [r for r in load(SRC / "heldout_eval.jsonl") if r.get("task") in ONEJEV_USEFUL]

    # ---- 3. When2Call：targeted 取均衡 400，其余可用作 control ----
    w2c_tr = load(SRC / "when2call_train.jsonl")
    w2c_va = load(SRC / "when2call_val.jsonl")
    w2c_ho = load(SRC / "when2call_heldout.jsonl")

    by_label = {}
    for r in w2c_tr:
        lab = next(iter(r["target"]))
        by_label.setdefault(lab, []).append(r)
    per = args.w2c_general // 3
    general = []
    for lab, rs in by_label.items():
        rng.shuffle(rs)
        general.extend(rs[:per])
    used_ids = {r["id"] for r in general}

    # ---- targeted 臂 ----
    targeted = drop_train + oj_tr + general
    rng.shuffle(targeted)

    # ---- control 臂：When2Call 剩余里随机取同量 ----
    rest = [r for r in w2c_tr if r["id"] not in used_ids]
    rng.shuffle(rest)
    control = rest[: len(targeted)]

    # ---- 验证集 ----
    val = drop_va + w2c_va
    heldout = drop_ho + oj_ho + w2c_ho

    print("=== 输出 ===")
    n_t = dump(targeted, "train_targeted_r1.jsonl")
    n_c = dump(control, "train_control_r1.jsonl")
    n_v = dump(val, "val_r1.jsonl")
    n_h = dump(heldout, "heldout_r1.jsonl")

    # ---- 配比清单 ----
    def src_dist(rows):
        return dict(Counter(r.get("source", "?") for r in rows))

    def task_dist(rows):
        return dict(Counter(str(r.get("task")) for r in rows).most_common(8))

    mix = {
        "targeted": {"n": n_t, "source": src_dist(targeted), "task": task_dist(targeted),
                     "label": dict(Counter(k for r in targeted for k in r["target"]))},
        "control": {"n": n_c, "source": src_dist(control), "task": task_dist(control),
                    "label": dict(Counter(k for r in control for k in r["target"]))},
        "val": {"n": n_v}, "heldout": {"n": n_h},
        "note": "control 为单一来源(When2Call)随机，与 targeted 来源结构不同，只能回答'定向混合 vs 单一通用'",
        "seed": args.seed,
    }
    (OUT / "round1_mix.json").write_text(json.dumps(mix, indent=2, ensure_ascii=False))
    print(f"\n=== targeted 来源 ===\n{src_dist(targeted)}")
    print(f"=== targeted task ===\n{task_dist(targeted)}")
    print(f"=== control 来源 ===\n{src_dist(control)}")
    print(f"\nmix -> {OUT/'round1_mix.json'}")


if __name__ == "__main__":
    main()

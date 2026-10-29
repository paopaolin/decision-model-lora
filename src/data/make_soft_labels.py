#!/usr/bin/env python3
"""生成软标签：用原版 elephant-1 在训练集上的输出分布，给硬标签加不确定性。

设计（对应老师方案第 2 条）：
  teacher 答对 -> target = α·onehot + (1-α)·teacher_dist     （α=0.8）
  teacher 答错 -> target = onehot                              （错误分布不可信，不注入）

为什么不全用 teacher 分布：那等于蒸馏，会把原版的错误也学过来。
为什么要 α=0.8 而不是 1-α 更大：正确答案仍须主导，软标签只负责携带"这条题有多确定"。

软标签的价值在于：teacher 在正确项上是 0.99 时软标签≈onehot（无信息），
是 0.6 vs 0.3 时把这种犹豫保留下来——正好把原版的校准能力往训练数据里迁移。
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from collections import Counter
from pathlib import Path

SRC = "/work/data/round1/train_targeted_r1.jsonl"
OUT_DIR = Path("/work/data/round2")
LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")


def ask(endpoint: str, state, question, timeout: float = 60.0) -> dict:
    body = {"state": state, "model": "probe", "questions": {"decision": question}}
    req = urllib.request.Request(
        f"{endpoint}/v1/systemone",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8100")
    ap.add_argument("--alpha", type=float, default=0.8, help="硬标签权重")
    ap.add_argument("--out", default="/work/data/round2/train_soft.jsonl")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    print(f"input rows: {len(rows)} | alpha={args.alpha}", flush=True)

    out_rows = []
    stats = Counter()
    ent_teacher, ent_soft = [], []
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        gold = r["target"]
        gold_key = max(gold, key=gold.get)
        q = r["question"]
        crit = list(q.get("criteria") or {})
        try:
            resp = ask(args.endpoint, r["state"], q)
            ans = resp["answers"]["decision"]
            probs = ans.get("probabilities") or ({ans["noul"]: 1.0} if "noul" in ans else {})
            probs = {str(k): float(v) for k, v in probs.items()}
        except Exception as e:  # noqa: BLE001
            stats["request_failed"] += 1
            if stats["request_failed"] <= 3:
                print(f"  FAIL {r.get('id')}: {type(e).__name__}: {e}", flush=True)
            out_rows.append({**r, "target": gold, "teacher_status": "failed"})
            continue

        # teacher 与 gold 的对齐：只在候选集内比较
        cand = [k for k in crit if k in probs] or list(probs)
        if not cand:
            stats["no_overlap"] += 1
            out_rows.append({**r, "target": gold, "teacher_status": "no_overlap"})
            continue
        teacher_key = max(cand, key=lambda k: probs.get(k, 0.0))
        teacher_correct = teacher_key == gold_key

        if teacher_correct:
            soft = {}
            for k in crit:
                onehot = 1.0 if k == gold_key else 0.0
                soft[k] = args.alpha * onehot + (1 - args.alpha) * probs.get(k, 0.0)
            tot = sum(soft.values())
            soft = {k: round(v / tot, 6) for k, v in soft.items()}
            status = "soft"
            stats["teacher_correct"] += 1
            ent_teacher.append(-sum(p * math.log(max(p, 1e-12)) for p in probs.values() if p > 0))
            ent_soft.append(-sum(p * math.log(max(p, 1e-12)) for p in soft.values() if p > 0))
        else:
            soft = gold
            status = "hard_teacher_wrong"
            stats["teacher_wrong"] += 1

        out_rows.append({**r, "target": soft, "teacher_status": status})

        if i % 200 == 0:
            print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s  "
                  f"correct={stats['teacher_correct']} wrong={stats['teacher_wrong']}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== 结果 ===")
    print(f"total {len(out_rows)} | teacher_correct {stats['teacher_correct']} "
          f"({stats['teacher_correct']/len(out_rows):.1%}) | teacher_wrong {stats['teacher_wrong']} | failed {stats['request_failed']}")

    def stats_of(xs):
        if not xs:
            return {}
        s = sorted(xs)
        n = len(s)
        return {"mean": round(sum(s)/n, 4), "p50": round(s[n//2], 4),
                "p90": round(s[int(n*0.9)], 4), "max": round(s[-1], 4)}

    print(f"teacher 分布熵: {stats_of(ent_teacher)}")
    print(f"软标签熵:      {stats_of(ent_soft)}")

    summary = {
        "n": len(out_rows), "alpha": args.alpha,
        "teacher_correct": stats["teacher_correct"], "teacher_wrong": stats["teacher_wrong"],
        "failed": stats["request_failed"],
        "teacher_accuracy_on_train": round(stats["teacher_correct"]/len(out_rows), 4),
        "entropy_teacher": stats_of(ent_teacher), "entropy_soft": stats_of(ent_soft),
        "rule": "teacher 对 -> alpha*onehot+(1-alpha)*teacher；teacher 错 -> 保留 onehot",
    }
    (OUT_DIR / "soft_label_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    print(f"stats -> {OUT_DIR/'soft_label_stats.json'}")


if __name__ == "__main__":
    main()

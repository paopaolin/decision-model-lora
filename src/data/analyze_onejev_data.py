#!/usr/bin/env python3
"""分析 OneJev-Data 一个分片的构成，用来决定训练集怎么筛。
只读需要的列，避免把 images 读进内存。
"""

import json
from collections import Counter

import pyarrow.parquet as pq

PATH = "/work/data/onejev/data/train-00000-of-00018.parquet"
COLS = ["id", "source", "task", "split", "license", "state", "question", "target", "meta"]


def main():
    t = pq.read_table(PATH, columns=COLS)
    rows = t.to_pylist()
    print(f"rows: {len(rows)}\n")

    for field in ("task", "source", "license", "split"):
        print(f"=== {field} ===")
        for k, v in Counter(r.get(field) for r in rows).most_common(15):
            print(f"  {str(k):<42} {v}")
        print()

    print("=== question.type ===")
    qt = Counter((r.get("question") or {}).get("type") for r in rows)
    print(dict(qt))

    print("\n=== state 类型与长度 ===")
    types = Counter(type(r.get("state")).__name__ for r in rows)
    print("types:", dict(types))
    lens = sorted(len(json.dumps(r["state"], ensure_ascii=False)) if not isinstance(r["state"], str)
                  else len(r["state"]) for r in rows)
    n = len(lens)
    print(f"chars: min {lens[0]} | p50 {lens[n//2]} | p90 {lens[int(n*0.9)]} | p99 {lens[int(n*0.99)]} | max {lens[-1]}")
    for th in (500, 1000, 2000, 4000, 8000):
        print(f"  <= {th} chars: {sum(1 for x in lens if x <= th)/n:.1%}")

    print("\n=== target 形态 ===")
    hard = soft = 0
    for r in rows:
        tg = r.get("target") or {}
        vals = [float(v) for v in tg.values()]
        if vals and max(vals) >= 0.999:
            hard += 1
        else:
            soft += 1
    print(f"hard label(近 one-hot): {hard} | soft label: {soft}")

    print("\n=== judge（标签验证方式）===")
    print(dict(Counter((r.get("meta") or {}).get("judge") for r in rows).most_common(10)))

    print("\n=== 样例 2 条 ===")
    for r in rows[:2]:
        print("id:", r["id"])
        print("  task:", r["task"], "| license:", r["license"])
        print("  question:", json.dumps(r["question"], ensure_ascii=False)[:300])
        print("  target:", r["target"])
        print("  state head:", str(r["state"])[:300])


if __name__ == "__main__":
    main()

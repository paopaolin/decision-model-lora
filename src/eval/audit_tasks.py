#!/usr/bin/env python3
"""JevBench 公开 231 条的结构审计：state 类型、token 长度、是否会撞 4096 窗口。"""

import json
from collections import Counter

import torch
from transformers import AutoTokenizer

TASKS = "data/jevbench_public_231.jsonl"
MAX_LEN = 4096


def as_text(state):
    """把 state 归一化成可数 token 的文本（dict 就按 JSON 序列化近似）。"""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def main():
    rows = [json.loads(line) for line in open(TASKS, encoding="utf-8")]

    print("=== state 类型分布 ===")
    print(dict(Counter(type(r["state"]).__name__ for r in rows)))

    dicts = [r for r in rows if isinstance(r["state"], dict)]
    if dicts:
        print("\n=== dict 型 state 结构 ===")
        keys = Counter()
        for r in dicts:
            keys.update(r["state"].keys())
        print("出现过的 key:", dict(keys))
        sample = dicts[0]
        print("样例 id:", sample["id"], "| family:", sample["family"])
        print("样例内容:", json.dumps(sample["state"], ensure_ascii=False)[:800])
        print("\n=== dict 型任务清单 ===")
        for r in dicts:
            print(f"  {r['id']:<34} keys={list(r['state'].keys())}")

    print("\n=== token 长度（state 单独） ===")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    vals = []
    for r in rows:
        n = len(tok(as_text(r["state"]))["input_ids"])
        vals.append((n, r["id"], r["family"]))
    vals.sort()
    n = len(vals)
    print(f"min {vals[0][0]} | p50 {vals[n//2][0]} | p90 {vals[int(n*0.9)][0]} "
          f"| p99 {vals[int(n*0.99)][0]} | max {vals[-1][0]}")

    over = [v for v in vals if v[0] > MAX_LEN]
    print(f"超过 {MAX_LEN} tokens 的任务数: {len(over)}")
    for t, tid, fam in over:
        print(f"  {t:>6} tokens  {tid:<34} family={fam}")


if __name__ == "__main__":
    main()

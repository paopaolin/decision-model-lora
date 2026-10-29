#!/usr/bin/env python3
"""查清筛选链的瓶颈：text-only 子集内部的 license / judge / 长度联合分布，
验证"各条件能否独立相乘"这个假设为什么失败。
"""

import json
from collections import Counter

import pyarrow.parquet as pq
from pathlib import Path

GOOD_JUDGE = {
    "program", "program (task verifier; hack runs can fool it)",
    "dataset_label", "recorded_action", "human",
}


def as_obj(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def media_empty(m):
    if m is None:
        return True
    if isinstance(m, str):
        s = m.strip()
        if not s:
            return True
        try:
            return not json.loads(s)
        except Exception:
            return False
    return m in ([], {}, "")


def lic_group(s):
    s = (s or "").lower()
    if "non-commercial" in s or "cc-by-nc" in s:
        return "non-commercial"
    if "research-only" in s:
        return "research-only"
    if "none stated" in s or s in ("", "none", "unknown"):
        return "unknown"
    if "openrail" in s:
        return "openrail"
    if "source terms" in s:
        return "mixed"
    return "permissive"


rows = []
for p in sorted(Path("/work/data/onejev/data").glob("train-*.parquet")):
    rows.extend(pq.read_table(p, columns=[
        "id", "source", "task", "license", "state", "question", "target", "media", "meta"
    ]).to_pylist())

text = [r for r in rows if media_empty(r.get("media"))]
print(f"total {len(rows)} | text_only {len(text)} ({len(text)/len(rows):.1%})\n")

print("=== text_only 内的 license 分布 ===")
for k, v in Counter(lic_group(r.get("license")) for r in text).most_common():
    print(f"  {k:<16} {v:>5} {v/len(text):6.1%}")

print("\n=== text_only 内的 judge 分布 ===")
for k, v in Counter((as_obj(r.get('meta')) or {}).get('judge') for r in text).most_common(10):
    print(f"  {str(k):<50} {v:>5} {v/len(text):6.1%}")

print("\n=== text_only 内的 source 分布 top10 ===")
for k, v in Counter(str(r.get("source")) for r in text).most_common(10):
    print(f"  {k:<34} {v:>5} {v/len(text):6.1%}")

print("\n=== 逐个条件收窄（不是独立相乘）===")
cur = text
steps = [
    ("text_only", lambda r: True),
    ("+ license in {permissive,openrail,research-only}",
     lambda r: lic_group(r.get("license")) in {"permissive", "openrail", "research-only"}),
    ("+ judge in GOOD",
     lambda r: (as_obj(r.get("meta")) or {}).get("judge") in GOOD_JUDGE),
    ("+ state <= 8000 chars",
     lambda r: len(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])) <= 8000),
]
prev = len(rows)
for name, fn in steps:
    cur = [r for r in cur if fn(r)]
    print(f"  {name:<50} {len(cur):>5}  (占上一步 {len(cur)/prev:.1%}, 占全量 {len(cur)/len(rows):.2%})")
    prev = len(cur)

print("\n=== 最终池子的 task 分布 ===")
for k, v in Counter(str(r.get("task")) for r in cur).most_common(15):
    print(f"  {k:<28} {v:>4}")

print("\n=== 最终池子的 state 长度 ===")
lens = sorted(len(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])) for r in cur)
n = len(lens)
if n:
    print(f"  min {lens[0]} | p50 {lens[n//2]} | p90 {lens[int(n*.9)]} | max {lens[-1]}")

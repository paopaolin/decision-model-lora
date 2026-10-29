#!/usr/bin/env python3
"""When2Call train_sft 能否可靠转成 4 类决策题的可行性分析。

只看 assistant 回复的结构特征，判断能否推出
direct / tool_call / request_for_info / cannot_answer 四类标签。
"""

import json
from collections import Counter

P = "/work/data/when2call/train/when2call_train_sft.jsonl"

rows = [json.loads(l) for l in open(P)]
print(f"rows: {len(rows)}")

# assistant message 的结构
akeys = Counter()
roles = Counter()
n_turn = Counter()
for r in rows:
    msgs = r.get("messages") or []
    n_turn[len(msgs)] += 1
    for m in msgs:
        roles[m.get("role")] += 1
        if m.get("role") == "assistant":
            akeys[tuple(sorted(m.keys()))] += 1

print("\n=== message 轮数 ===")
print(dict(n_turn.most_common(6)))
print("\n=== role 分布 ===")
print(dict(roles))
print("\n=== assistant message 的字段组合 ===")
for k, v in akeys.most_common(8):
    print(f"  {k}: {v}")

# 内容模式
KW = {
    "tool_call_json": ['"name":', '"arguments"'],
    "cannot": ["unable to", "can't help", "cannot", "not able to", "sorry", "apologies"],
    "ask_info": ["could you", "can you provide", "please provide", "what is your",
                 "which", "do you have", "could you please", "may i have"],
}


def classify(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in KW["tool_call_json"]):
        return "tool_call_like"
    hits = {name: any(k in t for k in kws) for name, kws in KW.items() if name != "tool_call_json"}
    if hits.get("cannot") and not hits.get("ask_info"):
        return "cannot_like"
    if hits.get("ask_info") and not hits.get("cannot"):
        return "ask_info_like"
    if hits.get("cannot") and hits.get("ask_info"):
        return "ambiguous_both"
    return "direct_like"


cls = Counter()
examples = {k: [] for k in ("tool_call_like", "cannot_like", "ask_info_like", "ambiguous_both", "direct_like")}
for r in rows:
    for m in r.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        c = classify(m.get("content") or "")
        cls[c] += 1
        if len(examples[c]) < 2:
            examples[c].append((m.get("content") or "")[:220])

print("\n=== assistant 回复粗分类（关键词法）===")
total = sum(cls.values())
for k, v in cls.most_common():
    print(f"  {k:<18} {v:>6} {v/total:6.1%}")

print("\n=== 各类样例 ===")
for k, exs in examples.items():
    for e in exs[:1]:
        print(f"  [{k}] {e}")

# tools 情况
ntools = Counter(len(r.get("tools") or []) for r in rows)
print("\n=== 每条样本的工具数 ===")
print(dict(sorted(ntools.items())[:12]))

# question 长度（user 消息）
lens = sorted(len((m.get("content") or "")) for r in rows for m in (r.get("messages") or []) if m.get("role") == "user")
if lens:
    n = len(lens)
    print("\n=== user 消息长度（字符）===")
    print(f"  min {lens[0]} | p50 {lens[n//2]} | p90 {lens[int(n*.9)]} | max {lens[-1]}")

# tools 文本长度
tlens = sorted(sum(len(t) for t in (r.get("tools") or [])) for r in rows)
n = len(tlens)
print("\n=== 每条样本 tools 总字符数 ===")
print(f"  min {tlens[0]} | p50 {tlens[n//2]} | p90 {tlens[int(n*.9)]} | max {tlens[-1]}")

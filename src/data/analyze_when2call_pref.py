#!/usr/bin/env python3
"""When2Call train_pref：chosen_response 的 4 类可判别性统计。

chosen_response 带 <TOOLCALL> 标记，比 train_sft 的纯文本回复可靠得多。
"""

import json
from collections import Counter

P = "/work/data/when2call/train/when2call_train_pref.jsonl"
rows = [json.loads(l) for l in open(P)]
print(f"rows: {len(rows)}")


def classify(text):
    t = (text or "").strip()
    low = t.lower()
    if "<toolcall>" in low:
        return "tool_call"
    asks = ["could you", "can you provide", "please provide", "do you have",
            "which one", "could you please", "may i have", "what is your",
            "i'll need", "i need to know", "to proceed", "please specify",
            "which language", "would you like"]
    cannot = ["unable to", "can't help", "cannot", "not able to", "apologies",
              "i'm sorry", "i am sorry", "sorry,"]
    a = any(k in low for k in asks)
    c = any(k in low for k in cannot)
    if a and not c:
        return "request_for_info"
    if c and not a:
        return "cannot_answer"
    if a and c:
        return "ambiguous"
    return "direct"


cls = Counter()
ex = {}
for r in rows:
    ch = (r.get("chosen_response") or {}).get("content") or ""
    k = classify(ch)
    cls[k] += 1
    if k not in ex:
        ex[k] = ch[:260]

print("\n=== chosen_response 类别分布 ===")
for k, v in cls.most_common():
    print(f"  {k:<18} {v:>5} {v/len(rows):6.1%}")

print("\n=== 各类样例 ===")
for k, v in ex.items():
    print(f"  [{k}] {v}")

# rejected 是否包含不同类别（可用于构造对照）
diff = 0
for r in rows:
    a = classify((r.get("chosen_response") or {}).get("content") or "")
    b = classify((r.get("rejected_response") or {}).get("content") or "")
    if a != b:
        diff += 1
print(f"\nchosen 与 rejected 类别不同的比例: {diff}/{len(rows)} = {diff/len(rows):.1%}")

# 长度
ul = sorted(len(m.get("content") or "") for r in rows for m in (r.get("messages") or []))
tl = sorted(sum(len(t) for t in (r.get("tools") or [])) for r in rows)
print(f"\nuser 字符: p50 {ul[len(ul)//2]} max {ul[-1]}")
print(f"tools 字符: p50 {tl[len(tl)//2]} p90 {tl[int(len(tl)*.9)]} max {tl[-1]}")
print("\n=== 样例完整结构 ===")
r = rows[0]
print("tools[0]:", str(r["tools"][0])[:300])
print("user:", r["messages"][0]["content"][:200])
print("chosen:", (r.get("chosen_response") or {}).get("content")[:300])

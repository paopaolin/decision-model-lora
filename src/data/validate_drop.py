#!/usr/bin/env python3
"""用 OneJev 训练器解析链路验证 DROP 转换结果。"""
import sys
sys.path.insert(0, "/work/baseten-env/OneJev")
from qev.prompt import render_question
from train.common import build_question, load_rows, target_vector

PATH = "/work/data/subsets/drop_train.jsonl"
rows = load_rows(PATH)
print(f"load_rows -> {len(rows)}")

ok, fail = 0, []
pos = {0: 0, 1: 0, 2: 0, 3: 0}
for r in rows:
    try:
        q = build_question(r["question"])
        rendered = render_question("q", q)
        vec = target_vector(r["target"], rendered.labels)
        assert abs(sum(vec) - 1.0) < 1e-6
        assert len(rendered.labels) == 4
        pos[vec.index(1.0)] += 1
        ok += 1
    except Exception as e:
        fail.append((r.get("id"), f"{type(e).__name__}: {e}"))

print(f"parsed ok: {ok}/{len(rows)}")
for f in fail[:3]:
    print("  FAIL", f)

print("\n=== 正确答案在选项中的位置分布（检验有没有位置捷径）===")
tot = sum(pos.values())
for k, v in pos.items():
    print(f"  第{k+1}个选项: {v} ({v/tot:.1%})")

r = rows[0]
q = build_question(r["question"])
rendered = render_question("q", q)
print("\n=== 第一条渲染 ===")
print("labels:", len(rendered.labels))
print(rendered.suffix[:500])
print("target:", r["target"])

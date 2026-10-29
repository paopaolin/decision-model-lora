#!/usr/bin/env python3
"""用 OneJev 训练器自己的解析链路验证转换结果，而不是"看着像"。

调 train.common.load_rows / build_question / target_vector
+ qev.prompt.render_question，确认 rows 能一路走通到渲染与目标向量。
"""

import sys

sys.path.insert(0, "/work/baseten-env/OneJev")

from qev.prompt import render_question  # noqa: E402
from train.common import build_question, load_rows, target_vector  # noqa: E402

PATH = "/work/data/subsets/when2call_train.jsonl"

rows = load_rows(PATH, limit=200)
print(f"load_rows -> {len(rows)} rows")
print("keys:", list(rows[0].keys()))
print("question type:", type(rows[0]["question"]).__name__,
      "| target type:", type(rows[0]["target"]).__name__)

ok, fail = 0, []
for r in rows:
    try:
        q = build_question(r["question"])          # pydantic 校验
        rendered = render_question("q", q)          # 渲染出选项槽位
        vec = target_vector(r["target"], rendered.labels)  # 目标向量对齐
        assert abs(sum(vec) - 1.0) < 1e-6, f"target 未归一 {sum(vec)}"
        ok += 1
    except Exception as e:  # noqa: BLE001
        fail.append((r.get("id"), f"{type(e).__name__}: {e}"))

print(f"\nparsed ok: {ok}/{len(rows)}")
if fail:
    for fid, err in fail[:5]:
        print("  FAIL", fid, err)
else:
    print("  全部通过 build_question -> render_question -> target_vector")

r = rows[0]
q = build_question(r["question"])
rendered = render_question("q", q)
print("\n=== 第一条的渲染结果 ===")
print("kind:", rendered.kind)
print("labels:", rendered.labels)
print("suffix:\n", rendered.suffix[:600])
print("target:", r["target"])
print("target vec:", target_vector(r["target"], rendered.labels))

from collections import Counter  # noqa: E402
import json  # noqa: E402

allrows = load_rows(PATH)
print(f"\n全量 {len(allrows)} 条，label 分布:",
      dict(Counter(k for r in allrows for k in r["target"])))
lens = sorted(len(json.dumps(r["state"], ensure_ascii=False)) for r in allrows)
n = len(lens)
print(f"state JSON 字符长度: p50 {lens[n//2]} | p90 {lens[int(n*.9)]} | max {lens[-1]}")

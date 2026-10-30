#!/usr/bin/env python3
"""DROP -> 决策格式（数值推理 choice 题）。

对准 elephant-1 最大的错误族 temporal_numeric（15 题错 8）。
只转纯数字/年份答案（占 62.4%），文本类答案跳过（干扰项不可靠）。

三个设计要点：
1. 干扰项必须"有意义"：优先用 passage 里出现的同量级数字（对应"混淆实体"这类真实错误），
   不足时用 ±1 / ±2 / ×2 之类的计算近似错误。
2. **选项顺序必须打乱**，否则模型学到"答案总是在 A"的位置捷径。
3. 按 passage 哈希分组划分，同一段材料的改写不跨组。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SRC = "/work/data/drop/data/train-00000-of-00001.parquet"
OUT = Path("/work/data/subsets")

NUM = re.compile(r"^-?\d[\d,]*\.?\d*$")
YEAR = re.compile(r"^(1[89]\d{2}|20\d{2})$")
NUM_IN_TEXT = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

INSTRUCTIONS = (
    "仅依据材料内容回答问题。材料中的数字需要按题目要求做计算或比较，"
    "不要直接挑选材料中看起来相关、但并未回答问题的数字。"
)


def to_num(s: str):
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def fmt(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def passage_numbers(passage: str) -> list[float]:
    out = []
    for m in NUM_IN_TEXT.findall(passage or ""):
        v = to_num(m)
        if v is not None and v > 0:
            out.append(v)
    return out


def make_distractors(ans: float, passage: str, rng: random.Random, is_year: bool) -> list[float] | None:
    cands = []
    if is_year:
        for d in (1, 2, 3, 5, 10, 20):
            cands += [ans - d, ans + d]
    else:
        # 计算近似错误：少算/多算一步、翻倍、减半
        for d in (1, 2, 3, 5, 10):
            if ans - d > 0:
                cands.append(ans - d)
            cands.append(ans + d)
        cands += [ans * 2, ans / 2, ans * 3]
        # 材料里同量级的数字（真实干扰：混淆实体）
        if ans > 0:
            for v in passage_numbers(passage):
                if v != ans and 0.4 * ans <= v <= 2.5 * ans:
                    cands.append(v)

    seen, out = {ans}, []
    rng.shuffle(cands)
    for c in cands:
        if c <= 0:
            continue
        key = round(c, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= 3:
            break
    return out if len(out) == 3 else None


def group_of(passage: str) -> str:
    return hashlib.sha256((passage or "").encode()).hexdigest()[:16]


def bucket(g: str) -> str:
    h = int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "heldout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--max-passage", type=int, default=3000)
    ap.add_argument("--out", default="/work/data/subsets/drop_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = pq.read_table(SRC).to_pylist()
    stats = Counter()
    kept = []
    used_passage = Counter()

    for r in rows:
        spans = (r.get("answers_spans") or {}).get("spans") or []
        if len(spans) != 1:
            stats["multi_answer"] += 1
            continue
        a = spans[0].strip()
        is_year = bool(YEAR.match(a))
        if not (is_year or NUM.match(a)):
            stats["non_numeric"] += 1
            continue
        ans = to_num(a)
        if ans is None or ans <= 0:
            stats["bad_number"] += 1
            continue
        passage = r.get("passage") or ""
        if not passage or len(passage) > args.max_passage:
            stats["passage_too_long"] += 1
            continue
        g = group_of(passage)
        if used_passage[g] >= 3:          # 同一材料最多出 3 题，避免单段材料主导
            stats["passage_quota"] += 1
            continue

        dis = make_distractors(ans, passage, rng, is_year)
        if dis is None:
            stats["no_distractor"] += 1
            continue

        ans_s, dis_s = fmt(ans), [fmt(d) for d in dis]
        options = [ans_s] + dis_s
        rng.shuffle(options)                       # 打乱，消除位置捷径
        # 描述必须中性且统一。这里曾经写成 "正确计算结果"/"干扰项"，而 qev 把
        # criteria 的 value 渲染成 "选项: 描述" 并放进模型输入 —— 等于在提示里
        # 直接标出正确答案，模型可以只学"挑写着正确的那个"，完全不看材料。
        crit = {o: "" for o in options}

        used_passage[g] += 1
        kept.append({
            "id": f"drop-{r.get('query_id')}",
            "source": "drop",
            "task": "numeric_reasoning",
            "license": "cc-by-4.0",
            "state": {"passage": passage, "question": r.get("question")},
            "question": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": crit,
            },
            "target": {ans_s: 1.0},
            "_group": g,
            "_split": bucket(g),
            "_is_year": is_year,
        })
        if len(kept) >= args.target * 6:           # 多留些给划分后取样
            break

    print(f"scanned -> kept {len(kept)} | skip: {dict(stats)}")
    if not kept:
        raise SystemExit("没有可转换样本")

    pools = Counter(k["_split"] for k in kept)
    print("split dist:", dict(pools))
    print("year vs number:", dict(Counter("year" if k["_is_year"] else "number" for k in kept)))

    # 按目标数量等比取样，保证划分比例
    final = []
    for split in ("train", "val", "heldout"):
        sub = [k for k in kept if k["_split"] == split]
        share = {"train": 0.8, "val": 0.1, "heldout": 0.1}[split]
        final.extend(sub[: int(args.target * share)])

    OUT.mkdir(exist_ok=True)
    # 分 split 落盘，保留 split 字段——否则训练时无法区分 val/heldout，等于泄漏
    counts = {}
    for split in ("train", "val", "heldout"):
        data = [x for x in final if x["_split"] == split]
        p = Path(args.out).parent / f"drop_{split}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for x in data:
                row = {k: v for k, v in x.items() if not k.startswith("_")}
                row["id"] = x["id"]
                row["split"] = split
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(data)
        print(f"  drop_{split}.jsonl: {len(data)} -> {p}")

    print(f"\ntotal {sum(counts.values())} | {counts}")
    print("选项数分布:", dict(Counter(len(x["question"]["criteria"]) for x in final)))
    print("\n=== 样例 2 条 ===")
    for x in final[:2]:
        print("Q:", x["state"]["question"])
        print("options:", list(x["question"]["criteria"]))
        print("target:", x["target"])
        print()


if __name__ == "__main__":
    main()

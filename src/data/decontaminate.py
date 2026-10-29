#!/usr/bin/env python3
"""OneJev-Data × JevBench public-231 污染检查（Level 1/2/4/5）。

Level 1 exact match       : state / instructions / state+instructions 原文比对
Level 2 normalized match  : 小写、去标点、压缩空白、数字归一后比对
Level 4 source overlap    : OneJev.source × JevBench.provenance.source
Level 5 task overlap      : OneJev.task × JevBench.family（命名体系不同，仅作记录）

输出 /work/audit/contamination.json
"""

import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ONEJEV_GLOB = "/work/data/onejev/data/train-*.parquet"
JEV = "/work/data/jevbench_public_231.jsonl"
OUT = Path("/work/audit")


def as_obj(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def text_of(x):
    """把 state 归一成可比对文本；dict 按键排序序列化，避免键序差异。"""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def norm(s):
    s = s.lower()
    s = re.sub(r"\d+(?:\.\d+)?", "<num>", s)   # 数字归一
    s = re.sub(r"[^a-z<>]+", " ", s)           # 去标点、非字母
    return re.sub(r"\s+", " ", s).strip()


def main():
    train = []
    shards = sorted(glob.glob(ONEJEV_GLOB))
    for p in shards:
        train.extend(pq.read_table(
            p, columns=["id", "source", "task", "state", "question", "license", "media"]
        ).to_pylist())
    test = [json.loads(l) for l in open(JEV, encoding="utf-8")]
    print(f"shards: {len(shards)} | onejev rows: {len(train)} | jevbench rows: {len(test)}")

    # JevBench 侧的各种键
    test_state_raw, test_q_raw, test_pair_raw = set(), set(), set()
    test_state_n, test_q_n, test_pair_n = set(), set(), set()
    for r in test:
        st = text_of(r["state"])
        ins = (r.get("question") or {}).get("instructions") or ""
        pair = st + "\n@@\n" + ins
        test_state_raw.add(st.strip())
        test_q_raw.add(ins.strip())
        test_pair_raw.add(pair.strip())
        test_state_n.add(norm(st))
        test_q_n.add(norm(ins))
        test_pair_n.add(norm(pair))

    hits = {"exact_state": [], "exact_question": [], "exact_pair": [],
            "norm_state": [], "norm_question": [], "norm_pair": []}

    for r in train:
        st = text_of(r["state"])
        q = as_obj(r["question"])
        ins = (q or {}).get("instructions") or ""
        pair = st + "\n@@\n" + ins
        rid = r.get("id")
        if st.strip() in test_state_raw:
            hits["exact_state"].append(rid)
        if ins.strip() and ins.strip() in test_q_raw:
            hits["exact_question"].append(rid)
        if pair.strip() in test_pair_raw:
            hits["exact_pair"].append(rid)
        if norm(st) in test_state_n:
            hits["norm_state"].append(rid)
        if norm(ins) and norm(ins) in test_q_n:
            hits["norm_question"].append(rid)
        if norm(pair) in test_pair_n:
            hits["norm_pair"].append(rid)

    print("\n=== 文本重合 ===")
    for k, v in hits.items():
        print(f"  {k:<16} {len(v)}")
        for x in v[:5]:
            print(f"      {x}")

    # source / provenance
    src_train = Counter(str(r.get("source")) for r in train)
    prov_test = Counter()
    for r in test:
        prov = r.get("provenance") or {}
        prov_test[str(prov.get("source") or prov.get("source_id") or "unknown")] += 1

    print("\n=== JevBench provenance.source 分布 ===")
    for k, v in prov_test.most_common(20):
        print(f"  {k:<40} {v}")

    overlap_src = sorted(set(src_train) & set(prov_test))
    print("\n=== source 名称重合 ===")
    print(overlap_src or "无字面重合")

    # task/family
    task_train = Counter(str(r.get("task")) for r in train)
    fam_test = Counter(str(r.get("family")) for r in test)
    overlap_task = sorted(set(task_train) & set(fam_test))
    print("\n=== task 与 family 字面重合 ===")
    print(overlap_task or "无字面重合（命名体系不同）")

    result = {
        "n_onejev_rows": len(train),
        "n_shards": len(shards),
        "n_jevbench_rows": len(test),
        "text_overlap_counts": {k: len(v) for k, v in hits.items()},
        "text_overlap_ids": hits,
        "jevbench_provenance_source": dict(prov_test),
        "onejev_source": dict(src_train),
        "source_name_overlap": overlap_src,
        "task_family_name_overlap": overlap_task,
        "onejev_task": dict(task_train),
        "jevbench_family": dict(fam_test),
        "note": "Level 3(fuzzy/embedding) 与模板级相似度本轮未做，见报告说明。",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "contamination.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT/'contamination.json'}")


if __name__ == "__main__":
    main()

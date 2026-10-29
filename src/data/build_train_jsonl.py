#!/usr/bin/env python3
"""OneJev-Data parquet -> 训练用 JSONL（含筛选、去重、source-level 划分）。

解决 09 号报告指出的格式问题：parquet 里 question/target/meta 都是 JSON 字符串，
训练器期望对象，这里统一 json.loads 后按训练器的 unified format 写出。

筛选链：
  media 非空 -> 排除（纯文本模型不能用）
  license 不在白名单 -> 排除
  长度 > max-chars -> 排除
  judge 不在高可信集合 -> 排除
  source 在排除名单 -> 排除（distill 源待溯源）
  重复 id / 重复 (state,instructions) -> 去重

划分：source-level（同一 source 的全部样本进同一组），避免同源泄漏。

用法：
  python build_train_jsonl.py --out-dir /work/data/subsets --targeted 4000 --control 4000 --heldout 800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SHARDS_GLOB = "/work/data/onejev/data/train-*.parquet"

# 标签验证方式：只留可程序核算、数据集自带、真实记录、人工核验的
GOOD_JUDGE = {
    "program", "program (task verifier; hack runs can fool it)",
    "dataset_label", "recorded_action", "human",
}
# targeted 组的任务优先级（对应 elephant-1 的失败结构）
PRIORITY_TASKS = {
    "high": ["outcome", "step_correct", "check_met", "outcome_forecast"],
    "mid": ["action_type", "next_action", "progress_level", "current_step", "next_step"],
    "keep": ["choice", "noul", "mc_statement"],
}
EXCLUDE_SOURCES = {"onejev/distill", "retention/onejev/distill"}


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


def src_bucket(source: str) -> str:
    """确定性 source-level 划分：hash 到 0-99，80/10/10。"""
    h = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "heldout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/work/data/subsets")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--targeted", type=int, default=4000)
    ap.add_argument("--control", type=int, default=4000)
    ap.add_argument("--heldout", type=int, default=800)
    ap.add_argument("--allow-licenses", default="permissive",
                    help="逗号分隔：permissive,openrail,research-only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    allow = set(args.allow_licenses.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    shards = sorted(Path("/work/data/onejev/data").glob("train-*.parquet"))
    for p in shards:
        rows.extend(pq.read_table(p, columns=[
            "id", "source", "task", "split", "license", "state", "question", "target", "media", "meta"
        ]).to_pylist())
    total = len(rows)
    reasons = Counter()

    # ---------- 筛选 ----------
    kept = []
    seen_ids, seen_pair = set(), set()
    for r in rows:
        if not media_empty(r.get("media")):
            reasons["media_dependent"] += 1
            continue
        if lic_group(r.get("license")) not in allow:
            reasons["license_excluded"] += 1
            continue
        q = as_obj(r.get("question"))
        tg = as_obj(r.get("target"))
        if not isinstance(q, dict) or not isinstance(tg, dict) or not q.get("type"):
            reasons["bad_question_or_target"] += 1
            continue
        st = r.get("state")
        st_text = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
        if not st_text or len(st_text) > args.max_chars:
            reasons["too_long_or_empty"] += 1
            continue
        meta = as_obj(r.get("meta")) or {}
        if meta.get("judge") not in GOOD_JUDGE:
            reasons["low_confidence_label"] += 1
            continue
        if str(r.get("source")) in EXCLUDE_SOURCES:
            reasons["excluded_source"] += 1
            continue
        rid = r.get("id")
        key = hashlib.sha256((st_text + "\n@@\n" + str(q.get("instructions"))).encode()).hexdigest()
        if rid in seen_ids or key in seen_pair:
            reasons["duplicate"] += 1
            continue
        seen_ids.add(rid)
        seen_pair.add(key)

        kept.append({
            "id": rid,
            "source": str(r.get("source")),
            "task": str(r.get("task")),
            "license": str(r.get("license")),
            "state": st if isinstance(st, str) else st,
            "question": q,          # 已解析为对象
            "target": tg,           # 已解析为对象
            "judge": meta.get("judge"),
            "_split": src_bucket(str(r.get("source"))),
        })

    print(f"shards={len(shards)} rows={total} kept={len(kept)}")
    print("filter reasons:", dict(reasons.most_common()))
    print("task dist (kept):", dict(Counter(k["task"] for k in kept).most_common(12)))

    # ---------- 划分 ----------
    pool = {"train": [], "val": [], "heldout": []}
    for k in kept:
        pool[k.pop("_split")].append(k)
    print("pool sizes:", {k: len(v) for k, v in pool.items()})

    rng = random.Random(args.seed)
    train_pool = pool["train"]
    rng.shuffle(train_pool)

    # ---------- targeted：按任务优先级分桶取 ----------
    by_task = defaultdict(list)
    for k in train_pool:
        by_task[k["task"]].append(k)

    def take(bucket_names, quota):
        out = []
        per = max(1, quota // max(1, len(bucket_names)))
        for t in bucket_names:
            out.extend(by_task.get(t, [])[:per])
        return out

    n_high, n_mid = int(args.targeted * 0.4), int(args.targeted * 0.25)
    n_keep = args.targeted - n_high - n_mid
    targeted = take(PRIORITY_TASKS["high"], n_high)
    targeted += take(PRIORITY_TASKS["mid"], n_mid)
    targeted += take(PRIORITY_TASKS["keep"], n_keep)
    # 不足时从剩余池补
    used = {id(x) for x in targeted}
    for k in train_pool:
        if len(targeted) >= args.targeted:
            break
        if id(k) not in used:
            targeted.append(k)
    targeted = targeted[: args.targeted]

    # ---------- control：同池随机 ----------
    used = {x["id"] for x in targeted}
    rest = [k for k in train_pool if k["id"] not in used]
    control = rest[: args.control]

    # ---------- heldout：val + heldout 组 ----------
    heldout = (pool["val"] + pool["heldout"])[: args.heldout]

    def dump(name, data):
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for x in data:
                row = {k: v for k, v in x.items() if k != "id"}
                row["id"] = x["id"]  # 训练器不读，但保留可追溯
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} rows -> {p}")
        return len(data)

    print("\n=== outputs ===")
    n1 = dump("train_targeted.jsonl", targeted)
    n2 = dump("train_control.jsonl", control)
    n3 = dump("heldout_eval.jsonl", heldout)

    summary = {
        "shards": len(shards), "total_rows": total, "kept_after_filter": len(kept),
        "filter_reasons": dict(reasons), "allow_licenses": sorted(allow),
        "max_chars": args.max_chars, "seed": args.seed,
        "outputs": {"targeted": n1, "control": n2, "heldout": n3},
        "pool_sizes": {k: len(v) for k, v in pool.items()},
        "kept_task_dist": dict(Counter(k["task"] for k in kept).most_common(30)),
        "targeted_task_dist": dict(Counter(k["task"] for k in targeted).most_common(20)),
        "source_dist": dict(Counter(k["source"] for k in kept).most_common(20)),
    }
    (out_dir / "subset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary -> {out_dir/'subset_summary.json'}")


if __name__ == "__main__":
    main()

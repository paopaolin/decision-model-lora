#!/usr/bin/env python3
"""OneJev-Data 第一阶段审计：schema + 分布 + 质量 + modality。

只读必要列（不读 images 二进制），结果写到 /work/audit/。
对应执行标准 Task A / Task B。
"""

import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

PATHS = sorted(glob.glob("/work/data/onejev/data/train-*.parquet"))
OUT = Path("/work/audit")
OUT.mkdir(exist_ok=True)
COLS = ["id", "source", "task", "split", "license", "state", "question",
        "target", "media", "meta"]


def as_obj(x):
    """question/target 在不同行里可能是 dict 也可能是 JSON 字符串。"""
    if isinstance(x, (dict, list)):
        return x, "object"
    if isinstance(x, str):
        try:
            return json.loads(x), "json_string"
        except Exception:
            return {"_unparsed": x[:200]}, "plain_string"
    return {}, f"type_{type(x).__name__}"


def main():
    rows = []
    for p in PATHS:
        rows.extend(pq.read_table(p, columns=COLS).to_pylist())
    n = len(rows)
    print(f"shards: {len(PATHS)} | rows: {n}")

    dist = {"n_rows": n, "source_files": PATHS, "n_shards": len(PATHS)}

    # ---------- 字段原始类型 ----------
    field_types = {}
    for f in ("state", "question", "target", "media", "meta"):
        field_types[f] = dict(Counter(type(r.get(f)).__name__ for r in rows))
    dist["field_types"] = field_types
    print("\n=== 字段原始类型 ===")
    print(json.dumps(field_types, indent=1))

    # ---------- task / source / license ----------
    for field in ("task", "source", "license"):
        c = Counter(str(r.get(field)) for r in rows)
        dist[field] = dict(c.most_common())
        with (OUT / f"{field}_distribution.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([field, "count", "pct"])
            for k, v in c.most_common():
                w.writerow([k, v, round(v / n * 100, 2)])
        print(f"\n=== {field} top15 ===")
        for k, v in c.most_common(15):
            print(f"  {k:<46} {v:>5}  {v/n*100:5.1f}%")

    # ---------- license 归类 ----------
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
            return "mixed/source-terms"
        return "permissive"

    lg = Counter(lic_group(str(r.get("license"))) for r in rows)
    dist["license_group"] = dict(lg)
    print("\n=== license 归类 ===")
    for k, v in lg.most_common():
        print(f"  {k:<22} {v:>5}  {v/n*100:5.1f}%")

    # ---------- question / target 解析 ----------
    qtypes = Counter()
    q_raw_types = Counter()
    option_counts = Counter()
    target_raw_types = Counter()
    target_form = Counter()
    bad_rows = []
    parsed = []

    for r in rows:
        q, qkind = as_obj(r.get("question"))
        tg, tkind = as_obj(r.get("target"))
        q_raw_types[qkind] += 1
        target_raw_types[tkind] += 1

        qt = q.get("type") if isinstance(q, dict) else None
        qtypes[str(qt)] += 1

        crit = q.get("criteria") if isinstance(q, dict) else None
        if isinstance(crit, dict):
            option_counts[len(crit)] += 1
        elif isinstance(crit, list):
            option_counts[len(crit)] += 1
        else:
            option_counts["none"] += 1

        if isinstance(tg, dict) and tg:
            vals = []
            ok = True
            for v in tg.values():
                try:
                    vals.append(float(v))
                except Exception:
                    ok = False
            if not ok:
                target_form["non_numeric"] += 1
            elif max(vals) >= 0.999:
                target_form["hard_onehot"] += 1
            else:
                target_form["soft_distribution"] += 1
            if abs(sum(vals) - 1.0) > 0.02:
                target_form["sum_not_1"] += 1
        else:
            target_form["empty_or_not_dict"] += 1

        if not r.get("state") or not q or not tg:
            bad_rows.append({"id": r.get("id"), "reason": "missing_state_or_q_or_target"})
        parsed.append((r, q, tg))

    dist["question_raw_type"] = dict(q_raw_types)
    dist["question_type"] = dict(qtypes)
    dist["option_count"] = {str(k): v for k, v in option_counts.items()}
    dist["target_raw_type"] = dict(target_raw_types)
    dist["target_form"] = dict(target_form)

    print("\n=== question 原始类型 / 题型 ===")
    print(" raw:", dict(q_raw_types))
    print(" type:", dict(qtypes))
    print(" option_count:", dict(option_counts))
    print(" target raw:", dict(target_raw_types))
    print(" target form:", dict(target_form))

    # ---------- state 长度 ----------
    def stext(s):
        if isinstance(s, str):
            return s
        return json.dumps(s, ensure_ascii=False)

    lens = sorted(len(stext(r["state"])) for r, _, _ in parsed)
    dist["state_char_len"] = {
        "min": lens[0], "p50": lens[n // 2], "p90": lens[int(n * .9)],
        "p95": lens[int(n * .95)], "p99": lens[int(n * .99)], "max": lens[-1],
        "mean": round(sum(lens) / n, 1),
    }
    print("\n=== state 长度（字符）===")
    print(dist["state_char_len"])
    for th in (500, 1000, 2000, 4000, 8000, 16000):
        dist["state_char_len"][f"le_{th}"] = round(sum(1 for x in lens if x <= th) / n, 4)
        print(f"  <= {th}: {dist['state_char_len'][f'le_{th}']:.1%}")

    # ---------- modality ----------
    def media_empty(m):
        if m is None:
            return True
        if isinstance(m, str):
            s = m.strip()
            if not s:
                return True
            try:
                return not json.loads(s)   # '[]' -> 空，视为纯文本
            except Exception:
                return False
        return m in ([], {}, "")

    media_flag = Counter()
    for r in rows:
        media_flag["text_only" if media_empty(r.get("media")) else "has_media"] += 1
    dist["modality_by_media_field"] = dict(media_flag)
    print("\n=== media 字段 ===")
    print(dict(media_flag))

    # ---------- judge（meta 也是 JSON 字符串） ----------
    def meta_obj(m):
        if isinstance(m, dict):
            return m
        if isinstance(m, str):
            try:
                return json.loads(m)
            except Exception:
                return {}
        return {}

    judges = Counter(meta_obj(r.get("meta")).get("judge") for r in rows)
    dist["judge"] = dict(judges.most_common())
    print("\n=== judge（标签验证方式）===")
    print(dict(judges.most_common(10)))

    # ---------- 质量 ----------
    ids = [r.get("id") for r in rows]
    dup = len(ids) - len(set(ids))
    dist["quality"] = {
        "duplicate_id_rows": dup,
        "missing_core_fields": len(bad_rows),
        "empty_state": sum(1 for r in rows if not r.get("state")),
    }
    print("\n=== 质量 ===")
    print(dist["quality"])

    # ---------- 样例 ----------
    with (OUT / "sample_examples.jsonl").open("w") as f:
        # 每种题型挑几条，覆盖不同 task
        seen = set()
        picked = []
        for r, q, tg in parsed:
            key = (r.get("task"), (q or {}).get("type") if isinstance(q, dict) else None)
            if key in seen:
                continue
            seen.add(key)
            picked.append((r, q, tg))
            if len(picked) >= 20:
                break
        for r, q, tg in picked:
            f.write(json.dumps({
                "id": r.get("id"), "task": r.get("task"), "source": r.get("source"),
                "license": r.get("license"),
                "state": str(r.get("state"))[:1200],
                "question": q, "target": tg, "media": r.get("media"),
                "meta": r.get("meta"),
            }, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(picked)} sample examples")

    (OUT / "onejev_distribution.json").write_text(
        json.dumps(dist, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL OUTPUT -> {OUT}")


if __name__ == "__main__":
    main()

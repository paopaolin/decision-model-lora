#!/usr/bin/env python3
"""把四个公开数据集转成 jevbench 的 Task 格式，用于跨 benchmark 泛化评测。

为什么是这四个：它们都不在 LoRA 训练配比里（训练用的是 When2Call / DROP / OneJev-Data），
所以原版与微调版在这些集合上的差异，是纯粹的迁移效果，不存在训练污染。

  数据集            测什么能力                  选项构造
  ------------------------------------------------------------------
  banking77        细粒度意图路由（77 类）      8 选 1，1 正 + 7 负
  clinc150        更难的路由（150 类，含域外）  8 选 1，1 正 + 7 负
  ai2_arc         科学与常识推理                原生 3-5 选项，不改造
  imdb            长文本判断                    2 选 1（正/负）

注意：banking77 和 clinc150 用的是"8 选 1 构造集"，不是原数据集的完整标签空间协议，
所以绝对数字不能和文献里的分类准确率直接比；跨模型之间的差异才是可比的。

输出：
  /work/data/crossbench/<name>.jsonl   单数据集
  /work/data/crossbench/all.jsonl      合并（family 区分来源）
  /work/data/crossbench/meta.json      构造元信息
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

RAW = Path("/work/data/crossbench/raw")
OUT = Path("/work/data/crossbench")
N_PER_DATASET = 300
NEG_OPTIONS = 7  # 8 选 1
SEED = 42


# ---------------------------------------------------------------- 工具


def fetch(repo: str, filename: str, slug: str) -> Path:
    p = hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(RAW / slug))
    return Path(p)


def class_names(path: Path, field: str) -> list[str]:
    """从 parquet 的 arrow schema 元数据里取 ClassLabel 的类名（HF 写进去的）。"""
    md = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
    blob = md.get(b"huggingface")
    if not blob:
        raise RuntimeError(f"{path}: 没有 huggingface schema 元数据，取不到 {field} 的类名")
    info = json.loads(blob)
    feat = info.get("info", {}).get("features", {}).get(field)
    if not feat or feat.get("_type") != "ClassLabel":
        raise RuntimeError(f"{path}: {field} 不是 ClassLabel，实际为 {feat}")
    return list(feat["names"])


def humanize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip().lower()


def balanced_rows(rows: list[dict], key: str, n: int, rng: random.Random) -> list[dict]:
    """按 key 轮转抽样，尽量让每个类别都出现，而不是被高频类淹没。"""
    buckets: dict = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)
    for k in buckets:
        rng.shuffle(buckets[k])
    order = sorted(buckets, key=lambda k: (-len(buckets[k]), str(k)))
    picked, i = [], 0
    while len(picked) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        i += 1
    return picked


def make_choice_task(tid, family, state, instructions, options: list[str], expected: str, prov: dict):
    return {
        "id": tid,
        "family": family,
        "state": state,
        "question": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {o: "" for o in options},
        },
        "labels": list(options),
        "expected": expected,
        "split": "public",
        "group": None,
        "provenance": prov,
    }


# ---------------------------------------------------------------- 各数据集


def build_banking77(rng: random.Random):
    path = fetch("PolyAI/banking77", "data/test-00000-of-00001.parquet", "banking77")
    names = class_names(path, "label")
    t = pq.read_table(path)
    rows = t.to_pylist()
    print(f"banking77: {len(rows)} 行, {len(names)} 类")

    picked = balanced_rows(rows, "label", N_PER_DATASET, rng)
    tasks = []
    for i, r in enumerate(picked):
        correct = names[r["label"]]
        others = [names[j] for j in range(len(names)) if j != r["label"]]
        negatives = rng.sample(others, NEG_OPTIONS)
        opts = [humanize(x) for x in [correct] + negatives]
        exp = humanize(correct)
        if len(set(opts)) != len(opts):  # 极少见的归一化碰撞，回退到原名
            opts = [correct] + negatives
            exp = correct
        rng.shuffle(opts)
        tasks.append(
            make_choice_task(
                f"xb-banking77-{i:03d}",
                "banking77",
                r["text"],
                "Which banking intent does the customer message express?",
                opts,
                exp,
                {
                    "source": "PolyAI/banking77 test split",
                    "source_id": f"row{r['label']}-{i}",
                    "license": "CC-BY-4.0",
                    "notes": "8-way constructed: 1 gold + 7 sampled negatives, shuffled",
                    "label_basis": "human-annotated intent",
                },
            )
        )
    return tasks


def build_clinc150(rng: random.Random):
    path = fetch("clinc/clinc_oos", "plus/test-00000-of-00001.parquet", "clinc")
    names = class_names(path, "intent")
    t = pq.read_table(path)
    rows = t.to_pylist()
    oos_idx = next((i for i, n in enumerate(names) if n == "out_of_scope"), None)
    # 域外样本在"8 选 1 闭集"里无法正确回答，会把噪声当成能力差异，先剔除
    rows = [r for r in rows if r["intent"] != oos_idx]
    print(f"clinc150: {len(rows)} 行（已剔除 out_of_scope）, {len(names) - 1} 类")

    picked = balanced_rows(rows, "intent", N_PER_DATASET, rng)
    tasks = []
    for i, r in enumerate(picked):
        correct = names[r["intent"]]
        others = [names[j] for j in range(len(names)) if j != r["intent"] and j != oos_idx]
        negatives = rng.sample(others, min(NEG_OPTIONS, len(others)))
        opts = [humanize(x) for x in [correct] + negatives]
        exp = humanize(correct)
        if len(set(opts)) != len(opts):
            opts, exp = [correct] + negatives, correct
        rng.shuffle(opts)
        tasks.append(
            make_choice_task(
                f"xb-clinc150-{i:03d}",
                "clinc150",
                r["text"],
                "Which intent does the user's message express?",
                opts,
                exp,
                {
                    "source": "clinc/clinc_oos plus config, test split",
                    "source_id": f"row{r['intent']}-{i}",
                    "license": "CC-BY-3.0",
                    "notes": "8-way constructed; out_of_scope rows excluded",
                    "label_basis": "human-annotated intent",
                },
            )
        )
    return tasks


def build_arc(rng: random.Random):
    path = fetch("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet", "arc")
    t = pq.read_table(path)
    rows = t.to_pylist()
    print(f"arc-challenge: {len(rows)} 行")

    rng.shuffle(rows)
    tasks, seen = [], set()
    for r in rows:
        if len(tasks) >= N_PER_DATASET:
            break
        ch = r["choices"]
        texts = [x.strip() for x in ch["text"]]
        letters = ch["label"]
        key = r["answerKey"]
        if key not in letters:
            continue
        texts = [x for x in texts if x]
        if len(set(texts)) < 3 or len(texts) > 8:
            continue
        gold = texts[letters.index(key)]
        if len(set(texts)) != len(texts) or gold in seen:  # 选项重复或题目重复
            continue
        seen.add(gold)
        tasks.append(
            make_choice_task(
                f"xb-arc-{len(tasks):03d}",
                "ai2_arc",
                r["question"].strip(),
                "Which option correctly answers the science question above?",
                texts,
                gold,
                {
                    "source": "allenai/ai2_arc ARC-Challenge, test split",
                    "source_id": str(r.get("id")),
                    "license": "CC-BY-SA-4.0",
                    "notes": "native options, no construction",
                    "label_basis": "official answer key",
                },
            )
        )
    return tasks


def build_imdb(rng: random.Random):
    path = fetch("stanfordnlp/imdb", "plain_text/test-00000-of-00001.parquet", "imdb")
    t = pq.read_table(path, columns=["text", "label"])
    rows = t.to_pylist()
    print(f"imdb: {len(rows)} 行")

    # 只留"确实长"的评论：太短体现不出长文本判断，太长可能超窗口
    rows = [r for r in rows if 600 <= len(r["text"]) <= 2400]
    picked = balanced_rows(rows, "label", N_PER_DATASET, rng)

    tasks = []
    for i, r in enumerate(picked):
        exp = "positive" if r["label"] == 1 else "negative"
        opts = ["positive", "negative"]
        rng.shuffle(opts)
        tasks.append(
            make_choice_task(
                f"xb-imdb-{i:03d}",
                "imdb",
                r["text"],
                "What is the overall sentiment of this movie review?",
                opts,
                exp,
                {
                    "source": "stanfordnlp/imdb plain_text, test split",
                    "source_id": f"row{i}",
                    "license": "Apache-2.0 (dataset card: research use)",
                    "notes": "600-2400 chars to make it a genuine long-text task",
                    "label_basis": "human rating threshold (<=4 negative, >=7 positive)",
                },
            )
        )
    return tasks


# ---------------------------------------------------------------- 主流程


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    builders = {
        "banking77": build_banking77,
        "clinc150": build_clinc150,
        "arc": build_arc,
        "imdb": build_imdb,
    }

    meta, all_tasks = {}, []
    for name, fn in builders.items():
        tasks = fn(rng)
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        lens = [len(t["state"]) for t in tasks]
        pos = Counter(t["labels"].index(t["expected"]) for t in tasks)
        nopt = Counter(len(t["labels"]) for t in tasks)
        meta[name] = {
            "n": len(tasks),
            "state_chars": {
                "mean": round(sum(lens) / len(lens), 1),
                "median": sorted(lens)[len(lens) // 2],
                "max": max(lens),
            },
            "option_counts": dict(sorted(nopt.items())),
            "gold_position": dict(sorted(pos.items())),
        }
        print(f"  -> {path.name}: {len(tasks)} 条, 选项数 {dict(nopt)}, "
              f"state 中位 {meta[name]['state_chars']['median']} 字符")
        all_tasks.extend(tasks)

    with (OUT / "all.jsonl").open("w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    meta["_total"] = len(all_tasks)
    meta["_seed"] = SEED
    meta["_note"] = "8-way constructed for banking77/clinc150; native options for arc; binary for imdb"
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n合计 {len(all_tasks)} 条 -> {OUT}/all.jsonl")


if __name__ == "__main__":
    main()

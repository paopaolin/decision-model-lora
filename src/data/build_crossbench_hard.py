#!/usr/bin/env python3
"""构建难负样本版跨域评测集。

随机负样本让意图任务饱和（98.7% / 99.3%），测不出细粒度区分能力。
这版把干扰项换成"最容易混的意图"：用训练集给每个意图建 TF-IDF 质心，
按与查询的余弦相似度排名，取最接近的 7 个（排除正确项）。

关键设计：
  - **查询、正确标签、选项数量都不变**，只换 7 个干扰项。
    所以两组结果的差值，隔离出"负样本难度"这一个变量。
  - task id 与随机版完全一致，可逐题配对比较。
  - arc / imdb 原样带入（它们没有"负样本"这个概念），保证一次跑完 1,200 条。

输出：/work/data/crossbench/all_hard.jsonl + meta_hard.json
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

CB = Path("/work/data/crossbench")
RAW = CB / "raw"
NEG = 7
SEED = 42

STOP = set(
    "a an the is are was were be been being do does did to of in on at for with and or "
    "but if then than that this these those i you he she it we they my your his her its "
    "our their what which who whom whose how why when where can could should would will "
    "shall may might must not no yes please help need want".split()
)


def toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if len(t) >= 3 and t not in STOP]


def class_names(path: Path, field: str) -> list[str]:
    md = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
    info = json.loads(md[b"huggingface"])
    return list(info["info"]["features"][field]["names"])


def humanize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip().lower()


class TfidfSpace:
    """极简 TF-IDF：文档频率在全训练集上算，每个类别一个归一化质心。"""

    def __init__(self, docs_by_class: dict[str, list[str]]):
        self.df: Counter = Counter()
        for docs in docs_by_class.values():
            for d in docs:
                self.df.update(set(toks(d)))
        self.n_docs = sum(len(v) for v in docs_by_class.values())
        self.centroids: dict[str, dict[str, float]] = {}
        for cls, docs in docs_by_class.items():
            acc: Counter = Counter()
            for d in docs:
                ts = toks(d)
                if not ts:
                    continue
                for t, c in Counter(ts).items():
                    acc[t] += c / len(ts)
            vec = {t: (c / len(docs)) * self.idf(t) for t, c in acc.items()}
            self.centroids[cls] = self._norm(vec)

    def idf(self, t: str) -> float:
        return math.log((1 + self.n_docs) / (1 + self.df.get(t, 0))) + 1.0

    @staticmethod
    def _norm(vec: dict[str, float]) -> dict[str, float]:
        n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / n for k, v in vec.items()}

    def query(self, text: str) -> dict[str, float]:
        ts = toks(text)
        if not ts:
            return {}
        vec = {t: (c / len(ts)) * self.idf(t) for t, c in Counter(ts).items()}
        return self._norm(vec)

    def rank(self, text: str) -> list[tuple[str, float]]:
        q = self.query(text)
        scored = []
        for cls, cen in self.centroids.items():
            # 稀疏点积只走较短的一侧
            small, big = (q, cen) if len(q) <= len(cen) else (cen, q)
            s = sum(v * big.get(k, 0.0) for k, v in small.items())
            scored.append((cls, s))
        return sorted(scored, key=lambda x: -x[1])


def build_intent(name: str, repo: str, train_file: str, test_slug: str, field: str,
                 limit_classes: set[str] | None, existing: list[dict], rng: random.Random):
    train_path = Path(hf_hub_download(repo, train_file, repo_type="dataset",
                                      local_dir=str(RAW / test_slug)))
    names = class_names(train_path, field)
    rows = pq.read_table(train_path, columns=["text", field]).to_pylist()
    print(f"{name}: 训练集 {len(rows)} 行, {len(names)} 类")

    by_class: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        cls = names[r[field]]
        if limit_classes and cls not in limit_classes:
            continue
        by_class[cls].append(r["text"])

    space = TfidfSpace(by_class)
    allowed = set(by_class)

    out, hard_sims, rand_sims, same_prefix = [], [], [], 0
    for t in existing:
        gold = t["expected"]
        ranked = [(c, s) for c, s in space.rank(t["state"]) if c in allowed and c != gold]
        negs = ranked[:NEG]
        if len(negs) < NEG:
            continue
        opts = [humanize(gold)] + [humanize(c) for c, _ in negs]
        exp = humanize(gold)
        if len(set(opts)) != len(opts):  # 归一化碰撞，回退原名
            opts = [gold] + [c for c, _ in negs]
            exp = gold
        rng.shuffle(opts)

        hard_sims.append(sum(s for _, s in negs) / len(negs))
        # 随机基线：同样从候选里随机取 7 个，量一下"难"了多少倍
        pool = [c for c in allowed if c != gold]
        pick = rng.sample(pool, NEG)
        q = space.query(t["state"])
        rand_sims.append(sum(
            sum(v * space.centroids[c].get(k, 0.0) for k, v in q.items()) for c in pick
        ) / NEG)
        # 词面重叠：难负里有几个与查询共享至少一个实词
        tokset = set(toks(t["state"]))
        same_prefix += sum(1 for c, _ in negs if tokset & set(toks(humanize(c))))

        out.append({
            "id": t["id"],
            "family": f"{name}_hard",
            "state": t["state"],
            "question": {"type": "choice",
                         "instructions": t["question"]["instructions"],
                         "criteria": {o: "" for o in opts}},
            "labels": opts,
            "expected": exp,
            "split": "public",
            "group": None,
            "provenance": {
                **t.get("provenance", {}),
                "notes": "hard negatives: top-7 intents by TF-IDF cosine to the query",
                "hard_negative_sim": round(sum(s for _, s in negs) / len(negs), 4),
                "random_negative_sim": round(rand_sims[-1], 4),
            },
        })

    print(f"  -> {len(out)} 条 | 难负平均余弦 {sum(hard_sims)/len(hard_sims):.4f}"
          f" vs 随机 {sum(rand_sims)/len(rand_sims):.4f}"
          f"（{sum(hard_sims)/max(1e-9,sum(rand_sims)):.1f} 倍）"
          f" | 词面重叠的干扰项 {same_prefix} 个")
    return out, {"n": len(out),
                 "hard_neg_cosine": round(sum(hard_sims) / len(hard_sims), 4),
                 "random_neg_cosine": round(sum(rand_sims) / len(rand_sims), 4),
                 "negatives_sharing_a_token": same_prefix}


def main():
    rng = random.Random(SEED)
    base = [json.loads(l) for l in (CB / "all.jsonl").open(encoding="utf-8")]
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for t in base:
        by_fam[t["family"]].append(t)

    b77, s77 = build_intent("banking77", "PolyAI/banking77",
                            "data/train-00000-of-00001.parquet", "banking77", "label",
                            None, by_fam["banking77"], rng)
    clinc_classes = set(class_names(
        Path(hf_hub_download("clinc/clinc_oos", "plus/train-00000-of-00001.parquet",
                             repo_type="dataset", local_dir=str(RAW / "clinc"))),
        "intent")) - {"out_of_scope"}
    cl, sl = build_intent("clinc150", "clinc/clinc_oos",
                          "plus/train-00000-of-00001.parquet", "clinc", "intent",
                          clinc_classes, by_fam["clinc150"], rng)

    # arc / imdb 原样带入：它们没有负样本概念，换负样本对它们无意义
    keep = [t for t in base if t["family"] in ("ai2_arc", "imdb")]
    all_tasks = b77 + cl + keep

    with (CB / "all_hard.jsonl").open("w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    meta = {"_total": len(all_tasks), "_seed": SEED,
            "_note": "同查询、同正确标签、同选项数，仅把 7 个干扰项换成 TF-IDF 最相似意图",
            "banking77": s77, "clinc150": sl,
            "unchanged": {"ai2_arc": len(keep) // 2, "imdb": len(keep) // 2}}
    # 修正 unchanged 计数
    meta["unchanged"] = {k: sum(1 for t in keep if t["family"] == k) for k in ("ai2_arc", "imdb")}
    (CB / "meta_hard.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n合计 {len(all_tasks)} 条 -> {CB / 'all_hard.jsonl'}")


if __name__ == "__main__":
    main()

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

import os

# 从哪个目录读输入、写输出。默认当前目录（仓库根），可用 XB_ROOT 覆盖。
HERE = Path(os.environ.get("XB_ROOT", Path.cwd()))
CB = HERE / "results" / "crossbench"     # 输入：随机负样本版 all.jsonl
RAW = HERE / "_raw"                      # 两个意图分类训练集（见 scripts/fetch_raw.sh）
OUT = HERE / "results" / "crossbench_hard"   # 输出目录
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


def canon(s: str) -> str:
    """规范化到唯一形式：小写、非字母数字折叠成单空格。

    用于判定"同一个类别的不同拼写"——原始类别名 activate_my_card 与展示名
    activate my card 必须被视为同一个标签，否则会出现 gold 在选项里出现两次。
    """
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


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


def build_intent(name: str, parquet: Path, field: str,
                 limit_classes: set[str] | None, existing: list[dict], rng: random.Random):
    """构造难负集。

    关键：gold 在 all.jsonl 里存的是 humanize 后的展示名（"activate my card"），
    而质心空间里的类别名是原始形式（"activate_my_card"）。**两者必须用同一个
    canonical 形式比较**，否则 `c != gold` 永远成立，正确类别会被当成干扰项
    再放进选项里 —— 结果是同一个语义类别以两种拼写同时出现，模型只能猜哪个算对。
    """
    names = class_names(parquet, field)
    rows = pq.read_table(parquet, columns=["text", field]).to_pylist()
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
    skipped_short = 0
    for t in existing:
        gold = t["expected"]                 # 已 humanize，例如 "age limit"
        gcanon = canon(gold)                 # 比较与去重一律走 canonical
        ranked = [(c, s) for c, s in space.rank(t["state"])
                  if c in allowed and canon(c) != gcanon]

        # 按 canonical 去重后取前 NEG 个，避免同义不同拼写的类别重复出现
        negs, seen = [], {gcanon}
        for c, s in ranked:
            k = canon(c)
            if k in seen:
                continue
            seen.add(k)
            negs.append((c, s))
            if len(negs) >= NEG:
                break
        if len(negs) < NEG:
            skipped_short += 1
            continue

        # 展示名唯一，gold 只出现一次
        opts = [humanize(gold)] + [humanize(c) for c, _ in negs]
        exp = humanize(gold)
        assert len({canon(o) for o in opts}) == len(opts), f"{t['id']}: 选项 canonical 冲突"
        rng.shuffle(opts)

        hard_sims.append(sum(s for _, s in negs) / len(negs))
        # 随机基线：同样从候选里按 canonical 去重后随机取 7 个
        pool = sorted({canon(c): c for c in allowed if canon(c) != gcanon}.items())
        pick = rng.sample([c for _, c in pool], NEG)
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

    b77, s77 = build_intent("banking77", RAW / "banking77_train.parquet", "label",
                            None, by_fam["banking77"], rng)
    clinc_pq = RAW / "clinc_plus_train.parquet"
    clinc_classes = set(class_names(clinc_pq, "intent")) - {"out_of_scope"}
    cl, sl = build_intent("clinc150", clinc_pq, "intent",
                          clinc_classes, by_fam["clinc150"], rng)

    # arc / imdb 原样带入：它们没有负样本概念，换负样本对它们无意义
    keep = [t for t in base if t["family"] in ("ai2_arc", "imdb")]
    all_tasks = b77 + cl + keep

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "all_hard.jsonl").open("w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    meta = {"_total": len(all_tasks), "_seed": SEED,
            "_note": "同查询、同正确标签、同选项数，仅把 7 个干扰项换成 TF-IDF 最相似意图",
            "banking77": s77, "clinc150": sl,
            "unchanged": {"ai2_arc": len(keep) // 2, "imdb": len(keep) // 2}}
    # 修正 unchanged 计数
    meta["unchanged"] = {k: sum(1 for t in keep if t["family"] == k) for k in ("ai2_arc", "imdb")}
    (OUT / "meta_hard.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n合计 {len(all_tasks)} 条 -> {OUT / 'all_hard.jsonl'}")


if __name__ == "__main__":
    main()

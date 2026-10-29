#!/usr/bin/env python3
"""合并 JevBench 三份公开测试集为 231 条冻结文件，并做完整性校验。

用法（在 decision-eval 目录下）:
    python3 build_public_231.py

只依赖标准库，系统 python 即可运行，不需要 GPU、不需要装 jevbench。
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE / "jevbench"
NAMES = ["easy", "original", "hard"]

OUT = BASE / "data" / "jevbench_public_231.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = []
    with OUT.open("w", encoding="utf-8") as f:
        for name in NAMES:
            path = REPO / "datasets" / "public" / f"{name}.jsonl"
            if not path.exists():
                raise SystemExit(f"缺少文件: {path}")
            part = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            print(f"{name}: {len(part)}")
            for row in part:
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 硬校验：数量、唯一性、答案字段、题面字段
    if len(rows) != 231:
        raise SystemExit(f"总数不对: {len(rows)}")
    ids = {r["id"] for r in rows}
    if len(ids) != 231:
        raise SystemExit(f"重复 task id: {len(ids)} 个唯一")
    missing = [r.get("id") for r in rows if "expected" not in r]
    if missing:
        raise SystemExit(f"缺少 expected 字段: {missing[:5]}")
    if any(not r.get("state") for r in rows):
        raise SystemExit("存在空 state")

    print("总数:", len(rows))
    print("题型:", dict(Counter(r["question"]["type"] for r in rows)))
    print("任务族:", dict(Counter(r["family"] for r in rows)))
    print("输出:", OUT)

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (BASE / "data" / "jevbench_public_231.sha256").write_text(
        f"{digest}  jevbench_public_231.jsonl\n", encoding="utf-8"
    )
    print("sha256:", digest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""When2Call train_pref -> 决策模型训练格式（choice 三选一）。

任务本质：给定「可用工具 + 用户请求」，判断下一步该
  调用工具 / 追问缺失信息 / 声明无法完成
这正是决策模型相对通用 LM 的核心场景——一次 prefill、一个概率分布、不生成文本。

标签来源：chosen_response。tool_call 由 <TOOLCALL> 标记直接判定（结构化，不靠猜），
另两类用关键词，ambiguous/direct 一律剔除（宁缺勿错）。

划分：按 tools 组合哈希分组（同工具集的题进同一组），防同源泄漏。

输出 unified format，question/target 均为对象（不是 JSON 字符串），
可直接喂 OneJev train/sft.py。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PREF = "/work/data/when2call/train/when2call_train_pref.jsonl"

CRITERIA = {
    "call_tool": "所需工具和参数都已具备，直接调用工具完成任务。",
    "request_for_info": "缺少完成所必需的信息（如编号、地点、范围），先向用户追问。",
    "cannot_answer": "现有工具无法完成该请求，明确说明无法完成，不要编造结果。",
}
INSTRUCTIONS = (
    "根据可用工具与用户请求，判断下一步应当怎么做。"
    "若工具能完成任务且参数齐全，选择调用工具；"
    "若缺少必要信息，选择追问；"
    "若现有工具根本无法完成，选择无法回答。"
)

ASKS = ["could you", "can you provide", "please provide", "do you have",
        "which one", "could you please", "may i have", "what is your",
        "i'll need", "i need to know", "to proceed", "please specify",
        "would you like"]
CANNOT = ["unable to", "can't help", "cannot", "not able to", "apologies",
          "i'm sorry", "i am sorry", "sorry,"]


def classify(text: str) -> str | None:
    t = (text or "").strip()
    low = t.lower()
    if "<toolcall>" in low:
        return "call_tool"
    a = any(k in low for k in ASKS)
    c = any(k in low for k in CANNOT)
    if a and not c:
        return "request_for_info"
    if c and not a:
        return "cannot_answer"
    return None  # ambiguous 或 direct，剔除


def parse_tools(raw) -> list:
    out = []
    for t in raw or []:
        if isinstance(t, str):
            try:
                out.append(json.loads(t))
            except Exception:
                out.append({"raw": t})
        else:
            out.append(t)
    return out


def group_of(tools: list) -> str:
    names = sorted(str(t.get("name")) for t in tools if isinstance(t, dict))
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:12]


def bucket(group: str) -> str:
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "heldout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/work/data/subsets")
    ap.add_argument("--max-tools-chars", type=int, default=6000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(PREF)]
    print(f"input rows: {len(rows)}")

    kept, drop = [], Counter()
    for i, r in enumerate(rows):
        chosen = (r.get("chosen_response") or {}).get("content") or ""
        label = classify(chosen)
        if label is None:
            drop["unlabelable"] += 1
            continue
        tools = parse_tools(r.get("tools"))
        if not tools:
            drop["no_tools"] += 1
            continue
        tools_chars = len(json.dumps(tools, ensure_ascii=False))
        if tools_chars > args.max_tools_chars:
            drop["tools_too_long"] += 1
            continue
        user = ""
        for m in r.get("messages") or []:
            if m.get("role") == "user":
                user = m.get("content") or ""
                break
        if not user.strip():
            drop["no_user"] += 1
            continue

        g = group_of(tools)
        kept.append({
            "id": f"when2call-pref-{i}",
            "source": "when2call_pref",
            "task": "tool_use_decision",
            "license": "cc-by-4.0",
            "state": {"available_tools": tools, "user_request": user},
            "question": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": CRITERIA,
            },
            "target": {label: 1.0},
            "_label": label,
            "_group": g,
            "_split": bucket(g),
        })

    print(f"kept: {len(kept)} | dropped: {dict(drop)}")
    print("label dist:", dict(Counter(k["_label"] for k in kept)))
    print("split dist:", dict(Counter(k["_split"] for k in kept)))

    pools = defaultdict(list)
    for k in kept:
        pools[k["_split"]].append(k)
    print("pool sizes:", {k: len(v) for k, v in pools.items()})

    def dump(name, data):
        p = out / name
        with p.open("w", encoding="utf-8") as f:
            for x in data:
                row = {kk: vv for kk, vv in x.items() if not kk.startswith("_")}
                row["id"] = x["id"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} -> {p}")
        return len(data)

    print("\n=== outputs ===")
    a = dump("when2call_train.jsonl", pools["train"])
    b = dump("when2call_val.jsonl", pools["val"])
    c = dump("when2call_heldout.jsonl", pools["heldout"])

    summary = {
        "source": "nvidia/When2Call train_pref",
        "license": "cc-by-4.0",
        "input_rows": len(rows),
        "kept": len(kept),
        "dropped": dict(drop),
        "label_dist": dict(Counter(k["_label"] for k in kept)),
        "split_dist": {k: len(v) for k, v in pools.items()},
        "outputs": {"train": a, "val": b, "heldout": c},
        "criteria": CRITERIA,
        "instructions": INSTRUCTIONS,
        "assertion": "state 只含 available_tools 与 user_request，target 不进 state",
    }
    (out / "when2call_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary -> {out/'when2call_summary.json'}")


if __name__ == "__main__":
    main()

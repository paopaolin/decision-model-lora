#!/usr/bin/env python3
"""把 LoRA adapter 合回底座，产出可直接 serve 的完整模型目录。

要点：
- 复用训练器的 load_bundle，避免自己处理 Qwen3_5ForCausalLM / slot 校验的坑
- 必须把 adapter 目录里的 readout.json 一起带到输出目录，否则 DecisionEngine
  读不到 attention 配置，会用默认 causal，评测结果无意义
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/work/baseten-env/OneJev")

from peft import PeftModel  # noqa: E402
from train.common import load_bundle  # noqa: E402

DEFAULT_BASE = "/work/models/elephant-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/work/train_runs/r1_targeted_9b_lora/adapter")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default="/work/models/elephant-1-r1")
    args = ap.parse_args()

    BASE, ADAPTER, OUT = args.base, args.adapter, Path(args.out)
    t0 = time.time()
    bundle = load_bundle(BASE, device="cuda:0", dtype="bfloat16")
    print(f"base loaded in {time.time()-t0:.1f}s | slots={len(bundle.slot_ids)}", flush=True)

    model = PeftModel.from_pretrained(bundle.model, ADAPTER)
    print("adapter attached", flush=True)

    merged = model.merge_and_unload()
    print(f"merged in {time.time()-t0:.1f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(OUT), safe_serialization=True)
    bundle.tokenizer.save_pretrained(str(OUT))

    # readout.json 必须跟着走：训练产物里带 attention=noncausal
    src_readout = Path(ADAPTER) / "readout.json"
    if src_readout.exists():
        shutil.copy(src_readout, OUT / "readout.json")
        print("copied readout.json from adapter", flush=True)
    else:
        print("WARN: adapter 里没有 readout.json，需要手工补", flush=True)

    print(f"saved -> {OUT}", flush=True)
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()

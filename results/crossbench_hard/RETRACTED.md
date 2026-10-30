# 这个目录里哪些东西坏了

2026-10-30 发现的构造缺陷：旧版 `all_hard.jsonl` 把正确类别以两种拼写同时放进选项
（`age_limit` 与 `age limit`），只有一种被算作正确。1,200 条里 504 条受影响。

影响量化：旧结果里 9B 基线 367 个"错误"有 **315 个**（85.8%）是模型选了正确答案的另一种拼写；
R3 是 322/373（86.3%），27B 是 315/355（88.7%）。也就是说那套 69–70% 的准确率主要是这个缺陷造成的。

## 能用 / 不能用

| 文件 | 状态 |
|---|---|
| `all_hard.jsonl` | ✅ 已重建，gold 重复 0 条，可用于重跑 |
| `all_hard.buggy-alias.jsonl` | ❌ 旧的坏版本，只作留档 |
| `elephant1_base/`、`elephant1_r3/`、`mammoth1_base/` 下的 `results.jsonl` | ❌ 在旧版本上跑出来的，**不能**和新的 `all_hard.jsonl` 配对分析 |
| `summary.json`、`summary.csv`、`temp_transfer.csv`、`threshold_advice.csv`、`layered_policy.csv` | ❌ 由旧结果汇总而来，数字已作废 |
| `hard_vs_random.json` | ❌ 同上 |

## 要恢复这些结论需要做什么

三个模型在新 `all_hard.jsonl` 上各重跑一次（每个约 3 分钟推理 + 85 秒加载），
然后重跑 `analyze_crossbench.py`、`compare_hard_vs_random.py`、`threshold_advice.py`、
`margin_difficulty.py`。目前没有重跑，所以 README 里那些数字都标了作废。

随机负样本版（`results/crossbench/`）不受影响：它的 gold 从来没有进过干扰项池，
复核结果是 gold 重复 0 条。

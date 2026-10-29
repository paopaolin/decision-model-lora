# 13 · 第一轮 targeted 集配比评估（结论：当前配比不可用）

日期：2026-10-27
合并脚本：`merge_targeted.py`
产出：`/work/data/subsets/merged_targeted.jsonl`（6,601 条）

---

## 一、合并结果

| 来源 | 条数 | 占比 | 对应任务 |
|---|---:|---:|---|
| When2Call（pref 转换） | 6,507 | **98.6%** | tool_use_decision（三选一：调用/追问/拒答） |
| OneJev-Data（严格筛选） | 94 | 1.4% | outcome 43、security 18、which_run 14、outcome_forecast 14、check_met 5 |

比例 **70 : 1**。OneJev 那批在统计上几乎不存在。

## 二、对 JevBench 的覆盖度

| family | JevBench 题数 | 训练样本 | 状态 |
|---|---:|---:|---|
| tool_selection | 12 | 6,507 | 覆盖 |
| routing | 12 | 6,507 | 覆盖 |
| judge_hard | 17 | 71 | 覆盖 |
| adequacy | 12 | 48 | 覆盖 |
| policy | 12 | 18 | 覆盖 |
| adversarial | 6 | 18 | 覆盖 |
| **intent** | 24 | 0 | **无** |
| **extraction** | 24 | 0 | **无** |
| **long_policy** | 19 | 0 | **无** |
| **multi_hop** | 18 | 0 | **无** |
| **temporal_numeric** | 15 | 0 | **无** |
| fact | 12 | 0 | 无 |
| ordinal | 12 | 0 | 无 |
| probability | 10 | 0 | 无 |
| trap | 8 | 0 | 无 |
| ambiguous | 7 | 0 | 无 |
| tradeoff | 6 | 0 | 无 |
| routing_hard | 5 | 0 | 无 |

**覆盖 6/18 个 family，涉及 71/231 题（30.7%）。**

## 三、致命问题：覆盖到的恰好是 elephant-1 已经全对的

把 elephant-1 的 23 条错误按 family 拆开，再对照覆盖情况：

| family | 错题数 | 占全部错误 | 是否覆盖 |
|---|---:|---:|---|
| **temporal_numeric** | **8** | **34.8%** | **否** |
| **long_policy** | **5** | **21.7%** | **否** |
| **multi_hop** | **5** | **21.7%** | **否** |
| judge_hard | 4 | 17.4% | 是 |
| **tradeoff** | **1** | **4.3%** | **否** |

**落在已覆盖 family 的错误：4 条（17.4%）。落在未覆盖 family 的错误：19 条（82.6%）。**

也就是说，这批训练数据最多能改善 23 条错误里的 4 条；剩下 19 条（temporal_numeric 8、long_policy 5、multi_hop 5、tradeoff 1）一条训练样本都没有。

## 四、训练量投向的错配

| family | 训练样本 | 该族题数 | 错题数 | 每改善 1 条错误对应的训练量 |
|---|---:|---:|---:|---:|
| tool_selection | 6,507 | 12 | **0** | ∞（已经全对） |
| routing | 6,507 | 12 | **0** | ∞（已经全对） |
| judge_hard | 71 | 17 | 4 | 18 |
| adequacy | 48 | 12 | 0 | ∞ |
| policy | 18 | 12 | 0 | ∞ |
| adversarial | 18 | 6 | 0 | ∞ |

**98.6% 的训练量投在 tool_selection 和 routing 上，而这两个族 elephant-1 是 24 题全对。** 真正错的三个族（temporal/long_policy/multi_hop，18 条错误）零覆盖。

这不是"数据不够"，是**方向反了**。按这个配比训练，最可能的结果是：231 条准确率基本不动或微降（在已满分的题上过拟合，挤占原有能力），校准可能变差。

## 五、修正后的配比建议

按「错误占比 × 数据可得性」定权重。总量建议 1,500–2,000 条——公开数据凑不出更多，硬凑只会稀释质量。

| 目标 family | 错误占比 | 数据来源 | 建议条数 | 占比 |
|---|---:|---|---:|---:|
| temporal_numeric | 34.8% | DROP（train 77,400）转 choice | 400 | 20% |
| long_policy | 21.7% | ContractNLI + 自建长规则对照 | 350 | 17.5% |
| multi_hop | 21.7% | MuSiQue 训练部分 | 350 | 17.5% |
| judge_hard | 17.4% | OneJev 扩量 + 自建 | 300 | 15% |
| tradeoff | 4.3% | 自建对照 | 100 | 5% |
| **通用/防遗忘** | — | **When2Call 降为辅助** | **400** | **20%** |
| 自建跨族难例 | — | 按"标签随条件翻转"构造 | 100 | 5% |
| **合计** | | | **2,000** | 100% |

**When2Call 从 98.6% 降到 20%**，角色从"主干"改为"通用能力保留 + 场景数据"。它的价值在低延迟工具决策场景，不在 JevBench 提分，这一点必须写进实验设计，否则测完发现总分没动会很难解释。

## 六、现在这批数据怎么处理

`merged_targeted.jsonl` **保留但不用作第一轮主实验**，两种用途：

1. **作为对照组 B**（通用/场景 LoRA）——它本来就是"非定向"的数据，正好用来回答"通用数据能不能带来提升"这个问题（执行标准 16 节的 Model B）；
2. 等 DROP / MuSiQue / ContractNLI 转换完成后，其中 400 条并入 targeted 集的通用部分。

这样不浪费：6,601 条正好构成一个天然的对照臂。

## 七、要补的三块数据（按优先级）

1. **DROP → choice**（对准最大错误族 temporal_numeric，8 条）。把 passage+question 转成"给几个候选数值/日期，选正确的"，干扰项从其他样本的答案里抽，避免随机凑数。
2. **MuSiQue 训练部分 → 多跳判断**（对准 5 条）。保留证据链，构造"证据不足/结论成立"的判断。
3. **自建对照组 200–300 条**（覆盖 long_policy 和 tradeoff）。用"同一场景改一个决定性条件、标签随之翻转"的方法构造——这类数据的边际价值高于继续堆公开数据，因为它能精确对准失败模式。

长规则（long_policy）和权衡（tradeoff）没有现成同分布数据，**只能自建**。这是第一轮里最费人工、也最不可替代的部分。

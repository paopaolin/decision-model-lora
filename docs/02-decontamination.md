# 08 · 污染检查报告（OneJev-Data × JevBench public-231）

检查日期：2026-10-27
工具：`decontaminate.py`
数据：OneJev-Data 前 3 分片（15,852 行，覆盖全量 16.7%）× JevBench public-231（231 条）

---

## 一、检查方法

按执行标准 13 节的分级实现：

| 级别 | 方法 | 实现状态 |
|---|---|---|
| Level 1 | exact match（state / instructions / state+instructions 原文） | ✅ 已做 |
| Level 2 | normalized match（小写、去标点、压缩空白、数字→`<num>`） | ✅ 已做 |
| Level 3 | fuzzy / embedding 语义相似 | ⏳ 本轮未做 |
| Level 4 | source 名称交叉 | ✅ 已做 |
| Level 5 | task / family 模板级交叉 | ✅ 名称级已做，模板结构未做 |

## 二、结果

| 检查项 | 重合数 |
|---|---:|
| exact_state | **0** |
| exact_question | **0** |
| exact_pair | **0** |
| norm_state | **0** |
| norm_question | **0** |
| norm_pair | **0** |
| source 名称重合 | **无** |
| task × family 名称重合 | **无** |

## 三、为什么重合天然偏低

JevBench public-231 的 `provenance.source` 逐条查过：

| provenance.source | 条数 |
|---|---:|
| JevBench v1.2 hard tier (original authored scenario) | 111 |
| JevBench original authored scenario | 72 |
| JevBench v1.1 easy tier, original authored item | 48 |

**231 条全部是 JevBench 官方原创场景**，不是从公开数据集采集的。OneJev-Data 的 source（amex、mind2web、breakfast 等）与之没有字面交集，命题来源不同，这是零重合的根本原因。

## 四、局限（必须如实记录）

1. **覆盖率 16.7%**。剩余 15 个分片未检查。零重合的结论在已检样本上成立，不能自动推广到全量。正式训练前应把本 pipeline 跑满 18 个分片（脚本已支持 glob，分片下齐即可）。
2. **Level 3 未做**。语义近似的改写题（同场景换实体）检测不到。
3. **模板级重叠未深入**。OneJev 的 task 命名体系（outcome/next_action/...）与 JevBench family（temporal_numeric/long_policy/...）不同，名称无重合不代表任务结构无重合。`onejev/distill` 这个 source（1,017 条）名字上就有"蒸馏"含义，需要单独确认它的教师模型是否见过 JevBench。
4. **结论的适用范围**：本轮结论支持"没有逐字/近逐字泄漏"，不支持"完全无信息泄漏"。

## 五、进入训练前的待办

- [ ] 下载剩余 15 个分片，重跑 `decontaminate.py` 全量确认
- [ ] 对 `onejev/distill` 与 `retention/onejev/distill` 两个 source 单独溯源
- [ ] 补 Level 3：至少用 8-gram Jaccard 或 MinHash 扫一遍 state
- [ ] 训练集落盘时保留 `id` 与 `source`，出问题可回溯

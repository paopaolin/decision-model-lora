# Decision Model LoRA Adaptation

在 9B 决策模型上做参数高效微调：从硬标签到软标签，再到温度校准。

**核心结论**：LoRA 能让准确率提升 2 条（208 → 210 / 231），但**只有在用软标签训练时才不会破坏概率校准**；配合温度缩放后，置信度可以正常用于路由（阈值 0.9 下覆盖率 75.8%，采纳准确率 99.4%）。

---

## 摘要

决策模型（decision model）与生成式语言模型的工作方式不同：它不做自回归生成，而是在一次前向中读出预设选项的概率分布（`answer-slot readout`），因此延迟低、成本低，并且**自带一个可用于路由的置信度**。

这类模型的落地价值有一半在置信度上——阈值以上直接采纳，以下转人工或升级。所以任何微调都必须回答一个问题：**准确率涨了，但置信度还能信吗？**

本项目在 Baset Labs 的 9B 决策模型 `elephant-1` 上做了四组 LoRA 实验，全部在 [JevBench](https://github.com/OmniJev/jevbench) public-231 上评测，同一套数据、同一推理口径、单卡 H100 / bf16。

| 模型 | 准确率 | 对题 | Brier | ECE | 训练配置 |
|---|---:|---:|---:|---:|---|
| Decider-2B（参照） | 77.92% | 180 | 0.2803 | 0.0752 | 官方权重 |
| elephant-1-9B（基线） | 90.04% | 208 | 0.1289 | **0.0338** | 官方权重 |
| R1 | 90.48% | 209 | 0.1724 | 0.0840 | 硬标签 / 2ep / lr 1e-4 |
| R2 | 90.04% | 208 | 0.1369 | 0.0337 | 软标签 / 1ep / lr 1e-5 |
| **R3 + 温度缩放** | **90.91%** | **210** | 0.1346 | 0.0502 | 软标签 / 2ep / lr 1e-5 + T=1.31 |
| Mammoth-1-27B（上限参照） | 93.94% | 217 | 0.0828 | 0.0194 | 官方权重 |

---

## 主要发现

**1. 硬标签会把模型训练成"过度自信"。**
R1 用 one-hot 目标训练 2 epoch，准确率只涨 1 条，但 ECE 从 0.0338 恶化到 0.0840（2.5 倍），高置信度（≥0.9）却答错的样本从 **1 条涨到 14 条**。置信度闸门基本失效——在 0.99 阈值下只剩 6.6% 的样本能采纳。

**2. 软标签是"减震器"，不是"发动机"。**
用原版模型自己的输出分布软化标签（`0.8 × one-hot + 0.2 × teacher_dist`），再降低训练强度，校准完全恢复（ECE 0.0337 ≈ 基线 0.0338），但准确率也回到基线。软标签不让模型学得更准，只让它别过度自信。

**3. 训练强度决定准确率，软标签决定校准下限。**
R2 和 R3 用同一份软标签数据，唯一差别是 epoch 1 → 2：准确率 +0 → +2，ECE 0.0337 → 0.0549。两者是同一个权衡的两端。

**4. 提升是零退步的，而且落在没有训练数据的族上。**
R3 相比基线新对 2 条、新错 0 条，且两条全在 `multi_hop`——这一族**我们一条训练样本都没投**。同族的 27B 也是 +3。说明 multi_hop 属于可通过通用推理能力迁移改善的类型，不需要专门造数据。

**5. 温度参数本身就是诊断指标。**
在 public-231 上拟合的温度：基线 0.97（本来校准就好）、R1 2.23（严重过度自信，需要降温 2 倍多）、R3 1.31。对已校准的模型再拟合一次会得到 `T = 1.000`，这是校准确实生效的最干净验证。

---

## 路由效果（本项目的核心指标）

阈值 0.9 下"置信度达标就直接采纳"：

| 模型 | 覆盖率 | 采纳准确率 | 采纳中出错 |
|---|---:|---:|---:|
| elephant-1-9B 基线 | 73.7%（157 条） | 99.36% | 1 |
| R1（硬标签） | 73.7%（157 条） | 98.09% | 3 |
| **R3 + 温度缩放** | **75.8%（175 条）** | **99.43%** | **1** |
| Mammoth-1-27B | — | — | — |

R3 在能多采纳 18 条的同时，采纳准确率还略高。完整的阈值曲线见 [`results/routing_curve.csv`](results/routing_curve.csv)。

![routing](docs/figures/routing_curve.png)

---

## 仓库结构

```
├── README.md
├── requirements.txt
├── data_tasks.jsonl              # 231 条任务的 family / tier 元信息
├── docs/                          # 实验文档（按时间顺序）
│   ├── tech-report.docx           # 技术报告（可直接提交）
│   ├── 01-data-schema-audit.md    # 第三方数据集 schema 审计
│   ├── 02-decontamination.md      # 与评测集的污染检查
│   ├── 03-trainer-compatibility.md# 训练器字段/损失/注意力核对
│   ├── 04-targeted-data-mix.md    # 定向数据配比评估
│   ├── 05-round1-hard-label.md    # 第一轮：硬标签，校准崩坏
│   ├── 06-round2-soft-label.md    # 第二轮：软标签 + 温度缩放
│   ├── 07-conclusion-and-next.md  # 结论与后续方案
│   └── 08-baseline-four-models.md # 四个基座模型的横评
├── src/
│   ├── eval/                      # 评测集构建与结果分析
│   ├── data/                      # 数据集审计、去污染、格式转换
│   ├── train/                     # adapter 合并
│   ├── serve/                     # 决策模型 HTTP 服务封装
│   └── analysis/                  # 结果汇总、校准拟合、路由分析
├── scripts/                       # 环境准备与自动化流水线
└── results/
    ├── summary_all_models.csv     # 8 个模型的横向对比
    ├── per_family.csv             # 18 个任务族的准确率
    ├── routing_curve.csv          # 阈值-覆盖率-准确率
    ├── calibration_study.json     # 温度拟合结果
    └── <run>/                     # 每次运行的 per-example 结果
```

---

## 环境要求

- Python 3.11
- PyTorch 2.8.0 + CUDA 12.8（H100 上验证）
- 单卡 H100 80GB（9B LoRA 训练峰值显存 **18.0 GB**，27B 推理需 ~55 GB）
- 决策模型 runtime：[OneJev](https://github.com/OmniJev/OneJev) @ `81ce62f`

```bash
pip install -r requirements.txt
pip install -e OneJev[train]        # 含 qev runtime
```

---

## 复现步骤

**1. 准备模型**

```bash
bash scripts/prep_baselabs.sh                                    # 拉镜像、下载 9B 权重
bash scripts/dl_mammoth.sh                                       # 可选：27B
bash models/elephant-1/setup.sh /path/to/workdir                 # 注意：脚本不会自动创建 workdir
```

**2. 起服务**

```bash
python src/serve/serve_baseten.py --model models/elephant-1 --port 8100
```

**3. 跑基线评测**

```bash
jevbench run --tasks data/jevbench_public_231.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8100 \
  --model elephant-1 --key-env= --cost-basis no_billable_local_selfhost --reserve-usd 0 \
  --results results/elephant1_base/results.jsonl --raw-dir results/elephant1_base/raw \
  --ledger results/elephant1_base/ledger.jsonl --manifest results/elephant1_base/manifest.json
```

**4. 构造训练数据（含去污染）**

```bash
python src/data/onejev_audit.py            # schema 与分布审计
python src/data/decontaminate.py           # 与 231 条比对，确认零重合
python src/data/build_train_jsonl.py       # 筛选 + 去重 + source-level 划分
python src/data/convert_when2call.py       # 工具决策数据转决策格式
python src/data/convert_drop.py            # 数值题转 choice（含干扰项构造）
python src/data/build_round1.py            # 按弱项配比拼装
python src/data/make_soft_labels.py        # 用 teacher 输出分布生成软标签
```

**5. 训练**

```bash
torchrun --nproc-per-node 1 -m train.sft \
  --model models/elephant-1 \
  --train data/round2/train_soft.jsonl \
  --name r3_soft_2ep \
  --attention noncausal --readout slot_lm \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-len 4096 --batch-tokens 2048 \
  --epochs 2 --lr 1e-5 --brier-weight 1.0
```

**6. 合并 + 校准 + 评测**

```bash
python src/train/merge_adapter.py --adapter <run>/adapter --out models/elephant-1-r3
python src/analysis/fit_calibration_and_routing.py     # 拟合温度、输出路由曲线
python src/analysis/collect_results.py                 # 汇总所有 run
```

---

## 关键实现细节（四个真实踩过的坑）

**1. `--attention` 默认是 `causal`，直接用会毁掉模型架构。**

`elephant-1` 用的是 block-bidirectional attention（`config.json` 里的 `modeling_overrides.attention.full_attention = "block_bidirectional"`）。推理时引擎会读这个字段，但**训练器 `train/sft.py` 用命令行参数直接构造 `ReadoutConfig`，不读它**。不加 `--attention noncausal` 就等于先把架构改残再训练。

**2. fla + Triton 3.4.x 在 Hopper 上会静默算错梯度。**

```
RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results
for gated chunk_bwd_dqkwg
```

Qwen3.5 系列的 gated delta rule 依赖 `flash-linear-attention`。这个版本区间下反向 kernel 算出的梯度是错的但程序不报错——是库自己主动 raise 才挡住。解法：`pip install tilelang`，fla 会自动切后端（`[FLA Backend] chunk_bwd_dqkwg -> tilelang`）。

**3. 第三方 parquet 数据集里的 `question`/`target` 是 JSON 字符串，不是对象。**

训练器 `render_example` 直接对 `row["question"]` 调 `.get("type")`，不做解析就一定 AttributeError。所有外部数据在入训前必须 `json.loads` 再写回 JSONL。

**4. 合并 LoRA 时 `readout.json` 必须一起带走。**

adapter 目录里的 `readout.json` 记录了训练时的 `attention=noncausal`。如果合并时漏掉，推理会用默认的 causal，评测结果全部无意义。

---

## 数据说明

| 来源 | 用途 | 许可 | 备注 |
|---|---|---|---|
| [JevBench](https://github.com/OmniJev/jevbench) public-231 | 评测 | 见仓库 | 231 条，全部为官方原创场景 |
| [OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data) | 训练（补充） | 逐条不同 | 94,707 条；严格筛选后仅 717 条可用（0.76%），**69% 依赖图像** |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | 训练（工具决策） | CC-BY-4.0 | 用 `train_pref` 而非 `train_sft`（后者 15,000 条里没有一条工具调用样本） |
| [DROP](https://huggingface.co/datasets/ucinlp/drop) | 训练（数值推理） | CC-BY-4.0 | 只转纯数值答案，干扰项从同 passage 取同量级数字 |

**污染检查**：OneJev-Data 全量 94,707 条与 public-231 做 exact / normalized / source / task 四层比对，**六项重合全部为 0**。原因是 231 条全部为官方原创 authored scenario，与公开数据集不同源。语义近似（Level 3）未做。

---

## 局限性（诚实清单）

1. **温度是在 public-231 上拟合的**，属于测试集调参（1 个自由度）。报告中的 ECE 0.0502 应理解为上界，正式结论需要独立校准集。
2. **训练数据只有 1,166 条**，是同模型原始训练量的 0.02%。+2 条的提升统计意义有限。
3. **R3 的 ECE 仍高于基线**（0.0502 vs 0.0338）。路由曲线更好与全局校准更差这两个结论并存，不应只引用前者。
4. **训练集与评测集分布不同**：训练是工具决策 + 数值题 + agent 轨迹判断，评测是 18 个族的通用决策。
5. 路由表在 0.9 阈值下只有 175 条样本，置信区间宽，不应过度解读千分位差异。
6. 27B 的延迟（p50 0.256s）是在独占卡上测的，9B 各组在共享卡上测，延迟数字不完全可比。

---

## 引用

```bibtex
@misc{decision-model-lora-2026,
  title  = {Decision Model LoRA Adaptation: Soft Labels and Temperature Calibration},
  author = {MENG, Guanlin},
  year   = {2026},
  url    = {https://github.com/paopaolin/decision-model-lora}
}
```

## 致谢

- [Baset Labs](https://www.baseten.co/library/publisher/base-labs/) 提供 `elephant-1` / `mammoth-1` 权重与 [OneJev](https://github.com/OmniJev/OneJev) runtime
- [JevBench](https://github.com/OmniJev/jevbench) 提供评测框架与公开测试集

## License

代码 MIT，见 [LICENSE](LICENSE)。模型权重与数据集遵循其原始许可。

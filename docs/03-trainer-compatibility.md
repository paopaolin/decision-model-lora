# 09 · 训练器兼容性核对（OneJev-Data → OneJev trainer）

核对对象：`baseten-env/OneJev/train/sft.py` + `train/common.py`（@81ce62f + serve.patch）
核对日期：2026-10-27，逐行读代码，非文档推断

---

## 一、字段映射表

| OneJev-Data 字段 | 训练器入口 | 兼容性 | 需要的动作 |
|---|---|---|---|
| `state` | `serialize_state(row.get("state"))` | ⚠️ | parquet 里是 JSON 字符串；训练器把它当对象序列化。**必须 `json.loads`**，否则整段 state 会带引号进 prompt |
| `question` | `build_question(row["question"])`，直接 `.get("type")` | ❌ | **JSON 字符串会直接 AttributeError**。必须 `json.loads` |
| `target` | `target_vector(row["target"], labels)`，`.get(lbl, 0.0)` | ❌ | 同上，必须 `json.loads`。解析后单键硬标签可用（缺的键补 0） |
| `source` | `row.get("source", "unknown")` | ✅ | 直接可用 |
| `task` | `row.get("task", "unknown")` | ✅ | 直接可用 |
| `split` | `--split` 参数过滤 | ✅ | 直接可用 |
| `license` | 训练器不读 | — | 筛选阶段用 |
| `meta.judge` | 训练器不读 | — | 筛选阶段用 |
| `media` | `train/mm.py`（multimodal 模式） | ⚠️ | 纯文本训练必须**先排除 media 非空样本**，否则缺图 |
| `images` | 同上 | — | 同上 |

**结论：parquet → 训练 JSONL 需要一个转换脚本，做三件事：解析三个 JSON 字符串字段、过滤 media、写回 JSONL。** 这一步跳不过，也验证了执行标准第 5/27 节的禁令 8——字段同名不等于兼容。

## 二、损失函数（已确认）

`train/sft.py` 的 `slot_loss`（380–391 行）：

```python
ce    = 交叉熵（答案槽位上的受限 softmax）
brier = ((probs - targets) ** 2).sum(dim=-1)
loss  = (ce + brier_weight * brier).mean()
```

- 与 elephant-1 的训练口径一致：**CE + Brier**，`--brier-weight` 默认 1.0，不要改成 0；
- 只在序列最后一个答案槽位监督，没有生成目标；
- 监督信号形式：选项上的概率向量，硬标签等价 one-hot，软分布直接可用。

## 三、注意力架构（最容易踩的坑）

`train/sft.py` 617–626 行：

```python
bundle.readout = Readout.attach(bundle.model,
    ReadoutConfig(attention=args.attention, readout=args.readout, head_dim=args.head_dim), ...)
```

- **训练时的 readout 配置来自命令行参数，不读 elephant-1 自带的 `readout.json`**；
- `--attention` 默认 **`causal`**，`--readout` 默认 `slot_lm`；
- elephant-1 的架构是 `attention=noncausal`（block-bidirectional）+ `readout=slot_lm` + `calibration=none`。

**直接用默认参数对 elephant-1 训练，会把 bidirectional 结构改成 causal，等于先废掉模型再训。** 训练命令必须显式：

```
--attention noncausal --readout slot_lm
```

这一条对应执行标准第 18 节（attention mask topology 不允许被误改），不是理论风险，是默认参数就会触发的实际风险。

## 四、LoRA 实现（已确认）

- 不加 `--full-finetune` 即 LoRA；`peft.LoraConfig(r, lora_alpha, lora_dropout, bias="none", task_type="CAUSAL_LM")`；
- 默认 target 是 `common.linear_projection_modules(model)` 的全部线性层；执行标准 17 节建议第一版只动 `q_proj,k_proj,v_proj,o_proj`，用 `--target-modules q_proj,k_proj,v_proj,o_proj` 指定；
- LoRA 参数被强制转 fp32（`param.data.float()`），底座保持 bf16——显存估算按此口径；
- 梯度检查点默认开启，`--no-gradient-checkpointing` 才会关；
- `--merge` 可把 adapter 合回底座输出完整目录，merge 后能用现有 `serve_baseten.py` 直接起评测服务。

## 五、加载与显存

- `load_bundle` 支持本地路径（`local_files_only=True`），`Qwen3_5ForCausalLM` 文本路径可直接加载 elephant-1 的本地目录；
- LoRA 路径下 `load_dtype = args.dtype`（bf16），9B 冻结权重约 18GB；
- 按当前服务器最大空闲 33.9GB（GPU 0）：**9B LoRA 可行，但 `--batch-tokens` 建议从 8192 起步再降**，激活值是主要变量。

## 六、推荐训练命令（对 elephant-1 续训）

```bash
torchrun --nproc-per-node 1 -m train.sft \
  --model /work/models/elephant-1 \
  --train /work/data/train_targeted.jsonl \
  --name exp_b_9b_lora_v1 \
  --attention noncausal --readout slot_lm \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-len 4096 --batch-tokens 8192 \
  --epochs 1 --brier-weight 1.0 \
  --eval /work/data/heldout_eval.jsonl --eval-every 250
```

三个关键点都用粗体标过了：**`--attention noncausal`、`--brier-weight 1.0`、`--target-modules` 限定**。

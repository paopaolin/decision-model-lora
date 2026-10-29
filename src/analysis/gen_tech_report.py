#!/usr/bin/env python3
"""生成技术报告 docx（论文风格）。数据从 results/*.csv 读，不手抄。
输出 docs/tech-report.docx
"""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "docs" / "tech-report.docx"

CN = "宋体"
CN_H = "黑体"
MONO = "Consolas"


def style_run(run, size=10.5, bold=False, cn=CN, color=None):
    run.font.size = Pt(size)
    run.bold = bold
    run.font.name = cn
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cn)
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(doc, text="", size=10.5, bold=False, align=None, space_after=6, indent_first=False):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.4
    if indent_first:
        p.paragraph_format.first_line_indent = Pt(21)
    style_run(p.add_run(text), size=size, bold=bold)
    return p


def heading(doc, text, level=1):
    sizes = {1: 15, 2: 12.5, 3: 11}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(8 if level == 1 else 5)
    style_run(p.add_run(text), size=sizes.get(level, 11), bold=True, cn=CN_H)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.35
    style_run(p.add_run(text), size=size)
    return p


def table(doc, header, rows, widths=None, font=9, caption=None):
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(4)
        style_run(p.add_run(caption), size=9, bold=True, cn=CN_H)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_run(p.add_run(str(h)), size=font, bold=True, cn=CN_H)
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
            style_run(p.add_run(str(v)), size=font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    style_run(p.add_run(text), size=8.5, cn=MONO)
    return p


def load_csv(name):
    return list(csv.DictReader(open(RES / name, encoding="utf-8")))


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.6)

    # ---------------- 标题 ----------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    style_run(p.add_run("决策模型的参数高效适配："), size=19, bold=True, cn=CN_H)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(14)
    style_run(p2.add_run("软标签与温度校准对准确率和置信度路由的影响"), size=19, bold=True, cn=CN_H)

    for line, size in (("MENG Guanlin", 11), ("2026 年 10 月 29 日", 10)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(3)
        style_run(p.add_run(line), size=size)

    doc.add_paragraph()

    # ---------------- 摘要 ----------------
    heading(doc, "摘要", 1)
    para(doc, "决策模型（decision model）不做自回归生成，而是在一次前向中直接读出预设选项的概率分布，"
              "因此延迟低、成本低，并且自带一个可用于路由的置信度。这类模型的落地价值有一半在置信度上："
              "阈值以上直接采纳，以下转人工。因此任何微调都必须回答一个问题——准确率涨了，但置信度还能信吗？",
         indent_first=True)
    para(doc, "本文在一个 9B 决策模型（elephant-1）上做了四组 LoRA 适配实验，在 JevBench public-231 上以同一口径评测。"
              "实验发现：使用 one-hot 硬标签训练会使准确率微增 1 条，但校准误差（ECE）从 0.0338 恶化到 0.0840，"
              "高置信度却答错的样本从 1 条增至 14 条，置信度闸门基本失效。改用由教师模型输出分布构造的软标签、"
              "并降低训练强度后，校准完全恢复但准确率回到基线。将两者组合（软标签 + 2 个 epoch）并施加温度缩放后，"
              "准确率提升至 210/231（90.91%），且在 0.9 阈值下可自动采纳 75.8% 的请求、采纳准确率 99.43%，"
              "优于基线模型的 73.7% / 99.36%。",
         indent_first=True)
    para(doc, "消融显示：训练强度决定准确率增益，软标签决定校准下限——软标签是减震器而非发动机。"
              "此外，温度参数本身可作为过度自信的诊断指标；本项目中基线为 0.97、硬标签微调后升至 2.23、"
              "软标签方案为 1.31。",
         indent_first=True)

    # ---------------- 1 问题 ----------------
    heading(doc, "1  问题定义", 1)
    heading(doc, "1.1  决策模型与生成式语言模型的差异", 2)
    para(doc, "生成式模型完成一次判断需要自回归地生成一段文本，再由调用方解析；延迟随输出长度线性增长，"
              "成本按 token 计。决策模型则把选项标签本身当作词表中的 token，只在这些 token 的 logit 上做受限 softmax，"
              "一次前向即得到归一化的选项概率。其延迟与输出长度无关，且概率天然带有置信度含义。",
         indent_first=True)
    para(doc, "以本项目使用的 elephant-1 为例：其答案槽位读取方式为 slot_lm，注意力为 block-bidirectional"
              "（材料块与选项块内部各自双向，以便选项之间可以互相比较），损失为交叉熵加 Brier 项。"
              "这些结构决定了微调时必须在数据格式、注意力配置和损失函数三个层面保持一致。",
         indent_first=True)

    heading(doc, "1.2  核心问题", 2)
    para(doc, "在实际部署中，决策模型通常配合一个置信度阈值工作：置信度高于阈值则直接采纳，低于阈值转人工复核。"
              "这一机制成立的前提是概率经过校准——模型说 0.9 时，实际应有约 90% 的正确率。",
         indent_first=True)
    para(doc, "微调会改变模型的输出分布。若用 one-hot 目标训练，交叉熵会持续推动正确项的概率趋向 1，"
              "使模型对所有输入都过度自信。此时阈值机制会退化：闸门或全部放行（失去拦截作用），或全部拦截（失去效率）。"
              "本文要回答的正是：能否在提升准确率的同时保住这一机制。",
         indent_first=True)

    # ---------------- 2 实验设置 ----------------
    heading(doc, "2  实验设置", 1)
    heading(doc, "2.1  模型", 2)
    para(doc, "基线为 Baset Labs 的 elephant-1（9B，Qwen3.5-9B 全量微调而来，答案槽位读出口径为 slot_lm、"
              "注意力为 block-bidirectional）。上限参照为同系列的 mammoth-1（27B）。"
              "另有 Strands 的 Decider-2B 作为小模型参照。三者均以本机 bf16 自托管推理。",
         indent_first=True)
    para(doc, "需要说明与 Jev 的关系。Jev 是 TypeSafe AI 的闭源模型，官方从未公开其参数量，"
              "仅通过 API 提供服务，且文档明确指出它不接受针对客户数据的微调或 LoRA 适配。"
              "因此它无法作为本项目的微调对象，本文所用的是同范式的开源实现。"
              "截至 2026 年 10 月，Jev Decision Index 榜单上 Jev 得 60.2 分，低于多个开源 27B 模型"
              "（榜首 71.5），同生态的 OneJev 系列（0.8B/4B/9B/27B）分别为 29.0/49.2/53.3/59.9，"
              "规模与得分大致单调。这说明该范式的可训练能力目前主要承载于开源权重之上。",
         indent_first=True)

    heading(doc, "2.2  评测集", 2)
    para(doc, "JevBench public-231：231 条公开测试题，覆盖 18 个任务族，分为 easy（48）/ original（72）/ hard（111）三档。"
              "全部 231 条经核实为官方原创场景，与所用训练数据不同源（见 3.2 节污染检查）。",
         indent_first=True)

    heading(doc, "2.3  指标", 2)
    bullet(doc, "准确率：正确答案的选项被选中的比例。")
    bullet(doc, "Brier：所有选项上 (p - y)² 之和的均值，越低越好，衡量整个概率分布的质量。")
    bullet(doc, "ECE：按置信度分箱后，平均置信度与实际准确率的偏差。衡量“说 0.9 时是否真的有 90% 正确”。")
    bullet(doc, "路由指标：给定阈值下的覆盖率（可直接采纳的比例）与采纳准确率（被采纳部分中正确的比例）。")

    # ---------------- 3 方法 ----------------
    heading(doc, "3  方法", 1)
    heading(doc, "3.1  基线评测与失败分析", 2)
    base = [r for r in load_csv("summary_all_models.csv") if r["model"] == "elephant-1-9b"][0]
    para(doc, f"基线 elephant-1 在 231 条上取得 {int(base['n_correct'])}/231（{float(base['accuracy'])*100:.2f}%），"
              f"Brier {base['brier']}，ECE {base['ece']}。按任务族拆分错误后发现：23 条错误集中在 "
              "temporal_numeric（8 条）、long_policy（5 条）、multi_hop（5 条）、judge_hard（4 条）、tradeoff（1 条），"
              "easy 与 original 两档已接近满分。",
         indent_first=True)
    para(doc, "这决定了后续所有数据工作的方向：应当针对这四类弱项投放训练样本，而不是无差别扩充通用数据。",
         indent_first=True)

    heading(doc, "3.2  训练数据构造", 2)
    para(doc, "训练数据由三个公开来源构造，全部转换为决策模型原生格式（材料 + 题目 + 选项 + 目标分布）。",
         indent_first=True)
    table(doc,
          ["来源", "用途", "原始规模", "采用量"],
          [["OneJev-Data", "agent 轨迹结果判定", "94,707", "447"],
           ["When2Call (train_pref)", "工具调用/澄清/拒答三选一", "9,000", "399"],
           ["DROP", "数值推理", "77,400", "320"]],
          widths=[5.2, 6.0, 3.0, 2.4],
          caption="表 1  训练数据来源")
    para(doc, "对 OneJev-Data 的审计发现三个必须处理的问题：69.0% 的样本依赖图像（纯文本模型不可用）；"
              "42% 的标签来自 LLM 蒸馏（可靠性不足）；question / target 字段在 parquet 中为 JSON 字符串而非对象，"
              "直接输入训练器会抛出 AttributeError。经模态、许可、标签可信度、长度四道筛选后，"
              "94,707 条中仅 717 条（0.76%）可用。",
         indent_first=True)
    para(doc, "污染检查：将全部 94,707 条与 231 条评测题做精确匹配、归一化匹配、来源交叉、任务族交叉四层比对，"
              "六项重合数均为 0。原因是 231 条全部为官方原创 authored scenario，与公开数据集不同源。"
              "语义近似层面的检查（Level 3）未实施，为已知局限。",
         indent_first=True)

    heading(doc, "3.3  软标签构造", 2)
    para(doc, "软标签的构造规则如下：先用基线模型对每条训练样本打分，若其预测与标签一致，"
              "则目标分布取 0.8 × one-hot + 0.2 × 教师分布；若不一致，则保留 one-hot 不变。"
              "前者在正确答案占主导的前提下注入“这条样本有多确定”的信息，后者避免把错误分布学进来。",
         indent_first=True)
    para(doc, "统计结果值得注意：基线模型在这 1,166 条训练样本上的正确率仅 62.3%（727 条），"
              "远低于其在评测集上的 90%。原因是训练数据中的数值题和工具决策题恰好落在其弱项上。"
              "这意味着仅有 727 条样本获得了软标签，且教师分布的熵很低（中位数 0.0586），"
              "软化后的软标签熵中位数为 0.0149——实际注入的不确定性信息有限。",
         indent_first=True)

    heading(doc, "3.4  温度缩放", 2)
    para(doc, "温度缩放是一种事后校准方法：对 logits 除以标量 T 再做 softmax，不改变 argmax 因而不影响准确率，"
              "只改变概率的锐度。T 通过最小化验证集上的交叉熵拟合。",
         indent_first=True)
    para(doc, "本项目所用的推理引擎原生支持该机制：将 calibration.json 放入模型目录后自动加载，无需修改推理代码。"
              "由于 softmax(log(p)/T) 与对 logits 做同温度缩放等价，可直接由已记录的概率反推，"
              "无需重新抓取 logits。",
         indent_first=True)
    para(doc, "经验证：对已经施加温度缩放的模型再次拟合同一参数，得到 T = 1.000，"
              "且逐条比对概率有 213/231 条发生变化（未变的 18 条为 score 类型，样本量不足未拟合，回退到 T = 1）。"
              "这确认了校准确实作用于推理输出。",
         indent_first=True)

    # ---------------- 4 结果 ----------------
    heading(doc, "4  结果", 1)
    heading(doc, "4.1  基座模型横评", 2)
    rows = load_csv("summary_all_models.csv")
    base_rows = [[r["model"], f"{float(r['accuracy'])*100:.2f}%", r["n_correct"],
                  r["brier"], r["ece"], f"{r['p50_s']}s"] for r in rows]
    table(doc, ["模型", "准确率", "对题", "Brier", "ECE", "延迟 p50"],
          base_rows, widths=[5.4, 2.6, 1.8, 2.2, 2.0, 2.4],
          font=8.5, caption="表 2  全部模型的评测结果（JevBench public-231，同口径）")
    para(doc, "从 2B 到 9B 到 27B，准确率单调上升但边际收益快速衰减：2B→9B 提升 12.1 个点，"
              "每点准确率对应约 3.7% 的额外延迟；9B→27B 提升 3.9 个点，每点对应约 27% 的额外延迟。"
              "对延迟敏感的路由场景，9B 的性价比明显更好。",
         indent_first=True)

    heading(doc, "4.2  四轮微调对照", 2)
    para(doc, "四组微调实验共享同一份 targeted 数据（1,166 条），仅改变标签形式与训练强度。", indent_first=True)
    table(doc, ["轮次", "标签", "epoch", "lr", "准确率", "Brier", "ECE"],
          [["R1", "硬标签", "2", "1e-4", "90.48%（209）", "0.1724", "0.0840"],
           ["R2", "软标签", "1", "1e-5", "90.04%（208）", "0.1369", "0.0337"],
           ["R3", "软标签", "2", "1e-5", "90.91%（210）", "0.1395", "0.0549"],
           ["R3+温度", "软标签 + T=1.31", "2", "1e-5", "90.91%（210）", "0.1346", "0.0502"]],
          widths=[2.0, 3.6, 1.4, 1.4, 3.4, 1.8, 1.8],
          font=8.5, caption="表 3  四轮微调实验对照（基线：90.04% / 0.1289 / 0.0338）")
    para(doc, "R1 揭示了硬标签的代价：准确率仅增 1 条，ECE 恶化 2.5 倍，"
              "置信度 ≥0.9 却答错的样本从 1 条增至 14 条。在 0.99 阈值下，R1 只有 6.6% 的样本能通过闸门，"
              "实际上已无法用于路由。",
         indent_first=True)
    para(doc, "R2 用软标签配合减半的训练量，使校准完全恢复（ECE 0.0337），但准确率也回落到基线。"
              "R3 在同一份软标签上恢复到 2 个 epoch，取得 +2 条的准确率提升，ECE 升至 0.0549；"
              "施加温度缩放后降至 0.0502，Brier 降至 0.1346。",
         indent_first=True)
    para(doc, "需要特别指出：R3 相比基线新对 2 条、新错 0 条，是一次零退步的提升，"
              "且两条新增正确样本全部落在 multi_hop 族——该族未投入任何训练数据。",
         indent_first=True)

    heading(doc, "4.3  路由效果", 2)
    route = [r for r in load_csv("routing_curve.csv")
             if r["model"] in ("elephant-1-9b", "elephant-1-r1-hard",
                               "elephant-1-r3-soft2ep-temp", "mammoth-1-27b")
             and r["threshold"] in ("0.80", "0.90", "0.95", "0.99")]
    rrows = [[r["model"], r["threshold"], f"{float(r['coverage_pct']):.1f}%",
              f"{float(r['adopted_accuracy'])*100:.2f}%", r["n_errors"]] for r in route]
    table(doc, ["模型", "阈值", "覆盖率", "采纳准确率", "采纳中出错"],
          rrows, widths=[6.2, 2.0, 2.2, 2.6, 2.2],
          font=8.5, caption="表 4  不同阈值下的路由表现")
    para(doc, "在 0.9 阈值这一常用工作点上，R3 + 温度缩放的覆盖率为 75.8%（175 条），采纳准确率 99.43%，"
              "相比基线的 73.7%（157 条）/ 99.36%，在能多采纳 18 条的同时保持同等准确率。"
              "27B 的曲线整体更优，但其延迟为 9B 的 1.7 倍。",
         indent_first=True)

    img = ROOT / "docs" / "figures" / "routing_curve.png"
    if img.exists():
        doc.add_picture(str(img), width=Cm(15.2))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)
        style_run(p.add_run("图 1  路由曲线：横轴为自动采纳比例，纵轴为采纳部分的准确率，越靠右上越好"), size=9)

    heading(doc, "4.4  分族分析", 2)
    fam = load_csv("per_family.csv")
    key = ["family", "n", "elephant-1-9b", "elephant-1-r3-soft2ep-temp", "mammoth-1-27b"]
    fam_rows = []
    for r in fam:
        if r["family"] in ("temporal_numeric", "long_policy", "multi_hop", "judge_hard",
                           "tradeoff", "probability", "ambiguous", "intent", "extraction"):
            fam_rows.append([r["family"], r["n"],
                             f"{float(r['elephant-1-9b'])*100:.1f}%",
                             f"{float(r['elephant-1-r3-soft2ep-temp'])*100:.1f}%",
                             f"{float(r['mammoth-1-27b'])*100:.1f}%"])
    table(doc, ["任务族", "题数", "基线 9B", "R3+温度", "27B"],
          fam_rows, widths=[4.6, 1.8, 3.0, 3.0, 3.0],
          font=8.5, caption="表 5  关键任务族的分族准确率")
    para(doc, "temporal_numeric 是唯一在两个方向上都被证明难以改善的族：微调使其从 7 条增至 8 条，"
              "27B 同样为 8 条。从 2B（3/15）到 9B（7/15）到 27B（8/15），参数增加 13 倍仅多对 5 条。"
              "这提示该族的能力瓶颈不在参数规模，而可能在于单次前向加槽位读出的结构本身不擅长精确的多步算术，"
              "需要外部计算工具或架构改动。",
         indent_first=True)

    # ---------------- 5 讨论 ----------------
    heading(doc, "5  讨论", 1)
    heading(doc, "5.1  训练强度与软标签的分工", 2)
    para(doc, "R2 与 R3 使用完全相同的软标签数据，唯一差别是 epoch 从 1 增至 2，"
              "准确率从 +0 变为 +2、ECE 从 0.0337 升至 0.0549。这说明软标签并不能让模型“学得更准”，"
              "它只是在同等训练量下抑制过度自信。提升准确率仍需依靠训练量或数据量。",
         indent_first=True)

    heading(doc, "5.2  迁移效应", 2)
    para(doc, "R3 的 2 条提升全部落在 multi_hop，而该族没有任何训练数据；27B 在同一族同样是 +3。"
              "合理的解释是：多跳推理依赖的是一种可泛化的证据整合能力，"
              "被数值推理和轨迹判定数据中的组合式判断间接强化，而非依赖针对该族的专门监督。",
         indent_first=True)

    heading(doc, "5.3  温度作为诊断指标", 2)
    para(doc, "拟合出的温度本身即反映了模型的自信程度：基线 0.97（接近理想）、R1 2.23（需降温 2 倍以上，"
              "严重过度自信）、R3 1.31（轻度）。由于拟合只需一份带标签的样本和一次一维搜索，"
              "建议将其作为微调后的例行检查项。",
         indent_first=True)

    # ---------------- 6 局限 ----------------
    heading(doc, "6  局限", 1)
    bullet(doc, "温度在评测集（public-231）上拟合，属于测试集调参（仅 1 个自由度）。"
                "所报 ECE 应视为上界，正式结论需要独立校准集。")
    bullet(doc, "训练数据仅 1,166 条，为基座模型原始训练量的约 0.02%，+2 条的提升统计意义有限。")
    bullet(doc, "R3 的 ECE（0.0502）仍高于基线（0.0338）。路由曲线更优与全局校准更差这两个结论并存，"
                "不应只引用前者。")
    bullet(doc, "训练集与评测集分布不一致：前者为工具决策、数值推理与轨迹判定，后者为 18 个族的通用决策。")
    bullet(doc, "路由表在 0.9 阈值下仅 175 条样本，置信区间较宽，千分位差异不应过度解读。")
    bullet(doc, "污染检查未覆盖语义近似层面（Level 3）；27B 延迟在独占卡测得，与 9B 各组不完全可比。")

    # ---------------- 7 结论 ----------------
    heading(doc, "7  结论", 1)
    para(doc, "在一个已完成大规模决策后训练的 9B 模型上，参数高效适配仍能带来小幅但真实的提升（+2/231，零退步），"
              "前提是训练目标携带不确定性信息、且训练强度经过控制。使用 one-hot 目标会以牺牲校准为代价，"
              "使置信度路由机制失效——这种代价在只看准确率时完全不可见。",
         indent_first=True)
    para(doc, "温度缩放能在不改变准确率的前提下把校准拉回可用区间，且该机制在推理引擎中原生支持、无需改代码。"
              "综合来看，软标签 + 适度训练量 + 温度缩放是当前数据规模下兼顾准确率与路由可用性的较优组合；"
              "若要进一步突破准确率，主要制约是训练数据规模与对弱项族的覆盖，而非训练方法本身。",
         indent_first=True)

    # ---------------- 附录 ----------------
    heading(doc, "附录 A  复现细节", 1)
    para(doc, "训练配置（R3）：LoRA r=32、alpha=64、dropout=0.05，仅作用于 q/k/v/o 四个投影，"
              "可训练参数 7.86M（占 8,961.67M 的 0.088%）；max_len 4096、batch_tokens 2048、"
              "epochs 2、lr 1e-5、brier_weight 1.0。单卡 H100 80GB，峰值显存 18.0GB，947 步耗时 374 秒。",
         indent_first=True)
    para(doc, "关键命令：", indent_first=False)
    code_block(doc,
               "torchrun --nproc-per-node 1 -m train.sft \\\n"
               "  --model models/elephant-1 --train data/round2/train_soft.jsonl \\\n"
               "  --attention noncausal --readout slot_lm \\\n"
               "  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \\\n"
               "  --target-modules q_proj,k_proj,v_proj,o_proj \\\n"
               "  --max-len 4096 --batch-tokens 2048 \\\n"
               "  --epochs 2 --lr 1e-5 --brier-weight 1.0")

    heading(doc, "附录 B  四个实现层面的陷阱", 1)
    para(doc, "1）注意力配置：elephant-1 使用 block-bidirectional attention，推理引擎会从 config.json 的 "
              "modeling_overrides 字段读取该设置，但训练器以命令行参数直接构造配置且默认值为 causal。"
              "训练时必须显式指定 --attention noncausal，否则等于先改变模型架构再训练。",
         indent_first=True)
    para(doc, "2）反向传播数值错误：Qwen3.5 系列的 gated delta rule 依赖 flash-linear-attention，"
              "在 Triton 3.4.0–3.7.1 与 Hopper 显卡的组合下，其反向 kernel 会产生错误梯度且不报错，"
              "仅由库自身检查拦截。安装 tilelang 后自动切换后端可解决。这类静默错误若未被拦截，"
              "训练曲线正常但权重无效。",
         indent_first=True)
    para(doc, "3）数据格式：第三方 parquet 数据集中的 question / target 字段为 JSON 字符串，"
              "训练器直接按对象访问会抛异常，入训前必须解析。",
         indent_first=True)
    para(doc, "4）模型合并：LoRA adapter 目录中的 readout.json 记录了训练时的注意力配置，"
              "合并为完整模型时必须一并复制，否则推理回退到默认 causal，评测结果全部失真。",
         indent_first=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")
    print(f"size: {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()

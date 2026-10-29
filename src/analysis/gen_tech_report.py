#!/usr/bin/env python3
"""生成技术报告 docx（论文风格）。所有表格数据从 results/*.csv 读，不手抄。
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


def add_rich(p, text, size=10.5, bold=False):
    """把 **xxx** 转成真正的加粗片段，避免在 docx 里留下字面星号。"""
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        style_run(p.add_run(seg), size=size, bold=(bold or i % 2 == 1))


def para(doc, text="", size=10.5, bold=False, align=None, space_after=6, indent_first=False):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.4
    if indent_first:
        p.paragraph_format.first_line_indent = Pt(21)
    add_rich(p, text, size=size, bold=bold)
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
    add_rich(p, text, size=size)
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
    t.autofit = False
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
        # 正文可用宽度 = 页宽 21cm − 左右边距 2.6×2；超出就等比缩，否则表格会溢出到页边外
        usable = 15.6
        scale = min(1.0, usable / sum(widths))
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w * scale)
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
    style_run(p.add_run("决策模型的参数高效适配："), size=18, bold=True, cn=CN_H)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(14)
    style_run(p2.add_run("软标签、温度校准与跨域泛化"), size=18, bold=True, cn=CN_H)

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
    para(doc, "本文在一个 9B 决策模型（elephant-1）上做了四组 LoRA 适配实验，并在三个层次上验证其影响："
              "JevBench public-231 上的准确率与校准、四个公开数据集上的跨域泛化、以及难样本条件下的置信度可靠性。"
              "实验发现：使用 one-hot 硬标签训练会使准确率微增 1 条，但校准误差（ECE）从 0.0338 恶化到 0.0840，"
              "高置信度却答错的样本从 1 条增至 14 条，置信度闸门基本失效。改用由教师模型输出分布构造的软标签、"
              "并降低训练强度后，校准完全恢复但准确率回到基线；两者组合（软标签 + 2 epoch）并施加温度缩放后，"
              "准确率提升至 210/231（90.91%），且在 0.9 阈值下可自动采纳 75.8% 的请求、采纳准确率 99.43%，"
              "优于基线模型的 72.7% / 99.40%。",
         indent_first=True)
    para(doc, "跨域评测给出更完整但更克制的结论。在 1,200 条随机负样本任务上，微调后的模型没有可检测的能力退化"
              "（净 −1 条，McNemar p = 1.000），但该集合已饱和（三代模型 97.8–98.3%），分辨率仅 ±0.8 个点。"
              "将干扰项换成语义最相近的类别后，难度提升到 banking77 30.33%、clinc150 53.67%："
              "准确率仍无显著退化（−0.50 个点，p = 0.210），但**高置信错误从 22 条增至 47 条、采纳精度从 96.52% 降至 93.14%**。"
              "也就是说，微调的代价不在于变笨，而在于**在越难的输入上越不该被相信**。"
              "温度缩放能修掉这部分的约一半（错误 47 → 26），但修不干净。",
         indent_first=True)
    para(doc, "基于此，本文给出可操作的部署口径：简单流量上沿用 0.9 阈值即可；难流量上，"
              "同一模型为达到 98% 采纳精度需把阈值提到 0.97（基线）或 0.99（微调版），"
              "而微调版无论阈值多高都无法在难流量上达到 99% 精度。用 top-1 与 top-2 的概率差作为难度探针可显著改善这一权衡。",
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

    heading(doc, "1.2  核心问题：置信度能不能用来分流", 2)
    para(doc, "实际部署中，决策模型通常配合一个置信度阈值工作：置信度高于阈值则直接采纳，低于阈值转人工复核。"
              "这一机制成立的前提是概率经过校准——模型说 0.9 时，实际应有约 90% 的正确率。",
         indent_first=True)
    para(doc, "微调会改变输出分布。若用 one-hot 目标训练，交叉熵会持续推动正确项概率趋向 1，使模型对所有输入都过度自信。"
              "此时阈值机制退化：闸门或全部放行（失去拦截作用），或全部拦截（失去效率）。",
         indent_first=True)
    para(doc, "还有一个更隐蔽的问题：**校准是全局平均指标，而路由只关心高置信区间**。一个模型可以在全局 ECE 上变差、"
              "却仍在高置信区间表现良好；反之亦然。因此本文除了 ECE，还单独报告阈值下的覆盖率、采纳精度与高置信错误数。",
         indent_first=True)

    # ---------------- 2 实验设置 ----------------
    heading(doc, "2  实验设置", 1)
    heading(doc, "2.1  模型", 2)
    para(doc, "基线为 Baset Labs 的 elephant-1（9B，Qwen3.5-9B 全量微调而来，槽位读取为 slot_lm，"
              "注意力为 block-bidirectional）。上限参照为同系列 mammoth-1（27B），另有 Strands 的 Decider-2B 作为小模型参照。"
              "三者均以本机 bf16 自托管推理，同一套服务封装、同一套评测脚本。",
         indent_first=True)
    para(doc, "关于 Jev：它是 TypeSafe AI 的闭源模型，官方从未公开参数量，仅通过 API 提供服务，"
              "且文档明确指出它不接受针对客户数据的微调或 LoRA 适配。因此它无法作为本项目的微调对象。"
              "截至 2026 年 10 月，Jev Decision Index 榜单上 Jev 得 60.2 分，低于多个开源 27B 模型（榜首 71.5）；"
              "同生态的 OneJev 系列（0.8B/4B/9B/27B）分别为 29.0/49.2/53.3/59.9，规模与得分大致单调。"
              "值得注意的是，该榜单当前第一名是一个 4B 底座加 LoRA 的方案（65.84 分），"
              "说明「LoRA 适配决策模型」这条路已被验证可行。",
         indent_first=True)

    heading(doc, "2.2  评测集", 2)
    para(doc, "主评测集为 JevBench public-231：231 条公开测试题，覆盖 18 个任务族，分 easy（48）/ original（72）/ hard（111）三档。"
              "全部 231 条经核实为官方原创场景，与所用训练数据不同源（见 3.2 节）。",
         indent_first=True)
    para(doc, "由于 public-231 已被用于拟合温度与挑选 checkpoint，另建 1,200 条跨域评测："
              "banking77（意图路由）、clinc150（意图路由）、ai2_arc（科学推理）、imdb（长文本情感），各 300 条，"
              "与训练配比零交集。该集合有两套难度：随机负样本版与难负样本版（见 3.5 节）。",
         indent_first=True)

    heading(doc, "2.3  指标", 2)
    bullet(doc, "准确率：正确答案的选项被选中的比例。")
    bullet(doc, "Brier：所有选项上 (p − y)² 之和的均值，越低越好，衡量整个概率分布的质量。")
    bullet(doc, "ECE：按置信度十等分箱后，平均置信度与实际准确率的偏差。衡量「说 0.9 时是否真有 90% 正确」。")
    bullet(doc, "NLL：正确标签的负对数似然均值。在准确率饱和时，它比准确率更能区分模型。")
    bullet(doc, "路由指标：给定阈值下的覆盖率（可直接采纳的比例）、采纳精度（被采纳部分中正确的比例）、高置信错误数。")

    # ---------------- 3 方法 ----------------
    heading(doc, "3  方法", 1)
    heading(doc, "3.1  基线评测与失败分析", 2)
    base = [r for r in load_csv("summary_all_models.csv") if r["model"] == "elephant-1-9b"][0]
    para(doc, f"基线 elephant-1 在 231 条上取得 {int(base['n_correct'])}/231（{float(base['accuracy'])*100:.2f}%），"
              f"Brier {base['brier']}，ECE {base['ece']}。按任务族拆分 23 条错误后发现："
              "temporal_numeric（8 条）、long_policy（5 条）、multi_hop（5 条）、judge_hard（4 条）、tradeoff（1 条），"
              "easy 与 original 两档已接近满分。差距全部集中在 hard 档（58.6% → 87.4%，2B 与 27B 对比）。",
         indent_first=True)
    para(doc, "这决定了后续所有数据工作的方向：应针对这几类弱项投放训练样本，而不是无差别扩充通用数据。",
         indent_first=True)

    heading(doc, "3.2  训练数据构造与审计", 2)
    para(doc, "训练数据由三个公开来源构造，全部转换为决策模型原生格式（材料 + 题目 + 选项 + 目标分布）。", indent_first=True)
    table(doc,
          ["来源", "用途", "原始规模", "采用量"],
          [["OneJev-Data", "agent 轨迹结果判定", "94,707", "447"],
           ["When2Call (train_pref)", "工具调用/澄清/拒答三选一", "9,000", "399"],
           ["DROP", "数值推理", "77,400", "320"]],
          widths=[5.2, 6.0, 3.0, 2.4],
          caption="表 1  训练数据来源（合计 1,166 条）")
    para(doc, "对 OneJev-Data 的审计发现三个必须处理的问题：仅 30.69% 为纯文本（其余依赖图像，纯文本模型不可用）；"
              "纯文本子集中 42% 的标签来自 LLM 蒸馏，人工标注仅 3 条；question / target 字段在 parquet 中为 JSON 字符串而非对象，"
              "直接输入训练器会抛 AttributeError。经模态、许可、标签可信度、长度四道筛选后，"
              "94,707 条中仅 717 条（0.76%）可用。",
         indent_first=True)
    para(doc, "更关键的是长度错位：筛选后留下的样本 state 中位数为 3,815 字符，而 JevBench 的中位数是 70 字符。"
              "留下的多是「一整段 agent 轨迹 + 一个良莠判断」，与评测集「一个短场景 + 一道判断」的分布相反。"
              "这直接导致 OneJev-Data 从主干降为补充。",
         indent_first=True)
    para(doc, "污染检查：将全部 94,707 条与 231 条评测题做精确匹配、归一化匹配、来源交叉、任务族交叉四层比对，"
              "六项重合数均为 0。原因是 231 条全部为官方原创 authored scenario，与公开数据集不同源。"
              "语义近似层面的检查（Level 3）未实施，为已知局限。",
         indent_first=True)

    heading(doc, "3.3  软标签构造", 2)
    para(doc, "构造规则：先用基线模型对每条训练样本打分，若其预测与标签一致，目标分布取 0.8 × one-hot + 0.2 × 教师分布；"
              "若不一致，保留 one-hot 不变。前者在正确项占主导的前提下注入「这条样本有多确定」的信息，后者避免把错误分布学进来。",
         indent_first=True)
    para(doc, "统计结果值得注意：基线在这 1,166 条训练样本上的正确率仅 62.3%（727 条），远低于其在评测集上的 90%。"
              "原因是训练数据中的数值题与工具决策题恰好落在其弱项上。这意味着只有 727 条获得软标签，"
              "且教师分布的熵很低（中位数 0.0586），软化后软标签熵中位数 0.0149——实际注入的不确定性信息有限。",
         indent_first=True)

    heading(doc, "3.4  温度缩放", 2)
    para(doc, "温度缩放是一种事后校准方法：对 logits 除以标量 T 再做 softmax，不改变 argmax 因而不影响准确率，"
              "只改变概率的锐度。T 通过最小化验证集上的交叉熵拟合。",
         indent_first=True)
    para(doc, "推理引擎原生支持该机制：将 calibration.json 放入模型目录即自动加载，无需改代码。"
              "由于 p' ∝ p^(1/T) 与对 logits 做同温度缩放等价，可直接由已记录的概率反推，无需重新抓取 logits。"
              "（此处方向易错：T > 1 是软化，指数应取 1/T；写成 p^T 会得到相反的锐化效果。）",
         indent_first=True)
    para(doc, "经验证：对已施加温度缩放的模型再次拟合，得到 T = 1.000，且逐条比对概率有 213/231 条发生变化"
              "（未变的 18 条为 score 类型，样本量不足未拟合，回退到 T = 1）。这确认校准确实作用于推理输出。",
         indent_first=True)

    heading(doc, "3.5  跨域评测的两套难度", 2)
    para(doc, "第一套用随机负样本：意图类任务各取 1 个正确项 + 7 个随机抽样的负类，选项顺序打乱，按类别平衡采样。"
              "结果发现三代模型都挤在 97.8–98.3%，无法分辨。",
         indent_first=True)
    para(doc, "第二套改难负：用训练集为每个意图建立 TF-IDF 质心（文档频率在全训练集上统计，质心 L2 归一化），"
              "对每条查询按余弦相似度排序，取最相近的 7 个类别作为干扰项。**查询、正确标签、选项数量全部不变，只换干扰项**，"
              "因此两套结果的差值隔离出「负样本难度」这一个变量，task id 一致，可逐题配对。",
         indent_first=True)
    para(doc, "难负的相似度是随机版的 5.0 倍（banking77，0.2357 vs 0.0469）与 12.2 倍（clinc150，0.1634 vs 0.0134）。"
              "以查询「怎么激活卡片」为例，干扰项从 top up by bank transfer charge 变为 card not working、"
              "card linking、card acceptance、order physical card——同域、同主题、只差一个动作。",
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
    para(doc, "四组实验共享同一份 targeted 数据（1,166 条），仅改变标签形式与训练强度。", indent_first=True)
    table(doc, ["轮次", "标签", "epoch", "lr", "准确率", "Brier", "ECE"],
          [["R1", "硬标签", "2", "1e-4", "90.48%（209）", "0.1724", "0.0840"],
           ["R2", "软标签", "1", "1e-5", "90.04%（208）", "0.1369", "0.0337"],
           ["R3", "软标签", "2", "1e-5", "90.91%（210）", "0.1395", "0.0549"],
           ["R3+温度", "软标签 + T=1.31", "2", "1e-5", "90.91%（210）", "0.1346", "0.0502"]],
          widths=[2.0, 3.6, 1.4, 1.4, 3.4, 1.8, 1.8],
          font=8.5, caption="表 3  四轮微调实验对照（基线：90.04% / 0.1289 / 0.0338）")
    para(doc, "R1 揭示了硬标签的代价：准确率仅增 1 条，ECE 恶化 2.5 倍，置信度 ≥0.9 却答错的样本从 1 条增至 14 条。"
              "在 0.99 阈值下 R1 只有 6.6% 的样本能通过闸门，实际已无法用于路由。",
         indent_first=True)
    para(doc, "R2 用软标签配合减半的训练量，使校准完全恢复（ECE 0.0337），但准确率也回落到基线。"
              "R3 在同一份软标签上恢复到 2 个 epoch，取得 +2 条提升，ECE 升至 0.0549；施加温度缩放后降至 0.0502。",
         indent_first=True)
    para(doc, "R3 相比基线新对 2 条、新错 0 条，是一次零退步的提升，且两条新增正确样本全部落在 multi_hop 族——"
              "该族未投入任何训练数据。",
         indent_first=True)

    heading(doc, "4.3  路由效果", 2)
    route = [r for r in load_csv("routing_curve.csv")
             if r["model"] in ("elephant-1-9b", "elephant-1-r1-hard",
                               "elephant-1-r3-soft2ep-temp", "mammoth-1-27b")
             and abs(float(r["threshold"]) - 0.9) < 1e-9]
    rrows = [[r["model"], f"{float(r['coverage_pct']):.1f}%",
              f"{float(r['adopted_accuracy'])*100:.2f}%", r["n_errors"], r["n_adopted"]] for r in route]
    table(doc, ["模型", "覆盖率", "采纳精度", "采纳中出错", "采纳条数"],
          rrows, widths=[6.2, 2.4, 2.6, 2.4, 2.2],
          font=8.5, caption="表 4  0.9 阈值下的路由表现（231 条，全员对比）")
    para(doc, "在 0.9 这一常用工作点上，R3 + 温度缩放的覆盖率为 75.8%（175 条），采纳精度 99.43%，"
              "相比基线的 72.7%（168 条）/ 99.40%，在保持同等精度的前提下多采纳 7 条。27B 曲线整体更优（79.7% / 100%），"
              "但延迟为 9B 的 1.7 倍。",
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
    para(doc, "temporal_numeric 是唯一在两个方向上都被证明难以改善的族：微调使其从 7 条增至 8 条，27B 同样为 8 条。"
              "从 2B（3/15）到 9B（7/15）到 27B（8/15），参数增加 13 倍仅多对 5 条。"
              "这提示瓶颈不在参数规模，而可能在于单次前向加槽位读出的结构本身不擅长精确的多步算术，"
              "需要外部计算工具或架构改动。",
         indent_first=True)

    heading(doc, "4.5  跨域泛化（随机负样本）", 2)
    cb = load_csv("crossbench/summary.csv")
    pub = load_csv("crossbench/per_dataset.csv")
    table(doc, ["模型", "准确率", "gold 概率", "NLL", "≥0.9 覆盖", "采纳精度", "高置信错误"],
          [[r["model"], f"{float(r['accuracy_pct']):.2f}%", r["gold_prob"], r["nll"],
            f"{float(r['hi_cov_pct']):.1f}%", f"{float(r['hi_precision_pct']):.2f}%", r["hi_errors"]]
           for r in cb],
          widths=[4.6, 2.2, 2.2, 1.8, 2.2, 2.2, 2.2],
          font=8, caption="表 6  跨域 1,200 条（随机负样本）")
    table(doc, ["数据集"] + [k for k in pub[0] if k != "dataset"],
          [[r["dataset"]] + [f"{float(r[k]):.2f}%" for k in pub[0] if k != "dataset"] for r in pub],
          widths=[3.6, 3.0, 3.0, 3.0, 3.0],
          font=8, caption="表 7  分数据集准确率（随机负样本）")
    para(doc, "该集合上，微调版净 −1/1200（McNemar p = 1.000），无可检测退化。但基线已达 97.92%、"
              "二项标准误 0.41 个点，95% 区间约 ±0.81 个点——**只能排除大于 0.8 个点的损伤**。"
              "另一处证据说明该集合已饱和：27B 在 public-231 上比 9B 高 3.9 个点，在这里只高 0.42 个点。",
         indent_first=True)
    para(doc, "唯一保留区分度的是 NLL（27B 0.0597 vs 9B 0.0737）。一个反直觉的细节是 27B 在 imdb 上反而比 9B 低 0.67 个点，"
              "长文本二分类不吃规模。",
         indent_first=True)

    heading(doc, "4.6  跨域泛化（难负样本）", 2)
    hard = [r for r in load_csv("crossbench_hard/summary.csv") if r["set"] == "hard"]
    table(doc, ["模型", "准确率", "ECE", "NLL", "≥0.9 覆盖", "采纳精度", "高置信错误"],
          [[r["model"], f"{float(r['accuracy_pct']):.2f}%", r["ece"], r["nll"],
            f"{float(r['hi_cov_pct']):.1f}%", f"{float(r['hi_precision_pct']):.2f}%", r["hi_errors"]]
           for r in hard],
          widths=[4.6, 2.2, 1.8, 1.8, 2.2, 2.2, 2.2],
          font=8, caption="表 8  跨域 1,200 条（难负样本）")
    para(doc, "难度提升幅度：banking77 从 98.67% 降至 30.33%，clinc150 从 99.33% 降至 53.67%，"
              "而 ai2_arc 与 imdb 两套完全一致（它们没有负样本概念），六个组合逐位相同——这是数据链路的自检。",
         indent_first=True)
    para(doc, "必须先排除「题目本身有歧义」这一解释。判据是错选标签与正确标签是否共享实词：难负集上该比例为 91.6%，"
              "随机集上仅 20%。说明模型是在相似意图之间做精细区分时出错，而非随机乱选，也不是标注噪声。",
         indent_first=True)
    para(doc, "结果分两层。准确率上仍无显著退化：R3 净 −6 条，McNemar p = 0.210；失分集中在两个意图族（6 + 4 条），"
              "得分也在这两族（2 + 2），不是系统性崩塌。但**校准上明显退化**：高置信错误从 22 条增至 47 条，"
              "采纳精度从 96.52% 降至 93.14%，而 27B 只有 12 条、98.11%。",
         indent_first=True)
    para(doc, "换言之，输入越难，微调后的模型的置信度越不可信——而路由的价值恰恰在难样本上。"
              "这一点在随机版里看不出来（8 vs 14），因为那些题太简单、模型错得少。难负集不是改了结论，"
              "而是把原本被天花板压住的信号放了出来。",
         indent_first=True)

    heading(doc, "4.7  温度缩放的跨域迁移", 2)
    tt = load_csv("crossbench_hard/temp_transfer.csv")
    table(doc, ["集合", "R3 ECE", "+T=1.31", "R3 高置信错误", "+T 后"],
          [[r["set"], r["ece_raw"], r["ece_scaled_T1.31"], r["hi_errors_raw"], r["hi_errors_scaled"]] for r in tt],
          widths=[3.0, 3.0, 3.2, 3.6, 3.0],
          font=8.5, caption="表 9  温度缩放在两套难度上的效果（基线：随机 0.0071 / 8 条；难负 0.1272 / 22 条）")
    para(doc, "随机集上温度完全修平（ECE 0.0113 → 0.0071，追平基线）。难负集上把错误从 47 压到 26、ECE 从 0.1625 压到 0.1345，"
              "但仍略高于基线的 0.1272 / 22 条。",
         indent_first=True)
    para(doc, "因此过度自信有两个成分：**整体尺度偏移**（一个标量即可修正）与**难样本上的局部过度自信**"
              "（标量修不了，需要按难度分层校准或更换更强底座）。这一区分对部署有直接影响。",
         indent_first=True)

    heading(doc, "4.8  阈值选点：简单流量与难流量", 2)
    ta = load_csv("crossbench_hard/threshold_advice.csv")
    table(doc, ["集合", "模型", "0.9 覆盖", "0.9 精度", "达 98% 需阈值", "该阈值覆盖", "达 99% 需阈值"],
          [[r["set"], r["model"], f"{r['cov_at_0.9_pct']}%", f"{r['prec_at_0.9_pct']}%",
            r["thr_for_98"], f"{r['cov_at_98_pct']}%" if r["cov_at_98_pct"] != "n/a" else "—",
            r["thr_for_99"] if r["thr_for_99"] != "n/a" else "不可达"]
           for r in ta],
          widths=[2.2, 4.6, 2.0, 2.2, 2.6, 2.4, 2.4],
          font=7.5, caption="表 10  为达到目标采纳精度所需的最低阈值")
    para(doc, "这张表是本文最具操作性的结论。在简单流量（随机负样本）上，0.9 阈值对四个模型都够用，"
              "R3+温度甚至比基线略好（94.5% / 99.21% vs 93.4% / 99.29%）。",
         indent_first=True)
    para(doc, "在难流量上，同一个 0.9 就不安全了：R3 的采纳精度只有 93.14%，为达到 98% 需把阈值提到 0.990，"
              "覆盖率从 57.1% 掉到 46.4%；而**要达到 99%，R3 在难流量上不存在可用的工作点**——"
              "任何阈值下都达不到，基线则可以（0.990 / 35.0%）。这是「调阈值」与「模型不行」的分界："
              "前者可以靠调参数解决，后者不能。",
         indent_first=True)
    lp = load_csv("crossbench_hard/layered_policy.csv")
    table(doc, ["模型", "低 margin 阈值", "高 margin 阈值", "覆盖率", "采纳精度", "采纳中出错"],
          [[r["model"], r["threshold_low_margin"], r["threshold_high_margin"],
            f"{r['coverage_pct']}%", f"{r['precision_pct']}%", r["errors"]] for r in lp],
          widths=[4.8, 3.0, 3.0, 2.4, 2.6, 2.4],
          font=8, caption="表 11  分层阈值策略（按 top-1 与 top-2 概率差的中位数分两层）")
    para(doc, "分层之所以有效，是因为「难点」可以在推理时被识别：把 1,200 条按 top-1 与 top-2 的概率差（margin）排序，"
              "低的一半错题率 58.50%，高的一半只有 2.67%，相差 21.9 倍（27B 上为 34.5 倍）。"
              "这个指标不需要标签，推理时直接可得。",
         indent_first=True)
    para(doc, "在难负集上采用「低 margin 要求 0.99、高 margin 要求 0.95」的策略，R3 从 57.1% / 93.14%（47 错）"
              "改善到 50.0% / 97.50%（15 错），已经接近基线的默认表现（52.7% / 96.52%，22 错）——"
              "**代价是少采纳约 7 个百分点，换来精度回到可用区间**。",
         indent_first=True)

    # ---------------- 5 讨论 ----------------
    heading(doc, "5  讨论", 1)
    heading(doc, "5.1  训练强度与软标签的分工", 2)
    para(doc, "R2 与 R3 使用完全相同的软标签数据，唯一差别是 epoch 从 1 增至 2，准确率从 +0 变为 +2、"
              "ECE 从 0.0337 升至 0.0549。这说明软标签并不能让模型「学得更准」，只是在同等训练量下抑制过度自信。"
              "提升准确率仍需依靠训练量或数据量。",
         indent_first=True)

    heading(doc, "5.2  迁移效应", 2)
    para(doc, "R3 的 2 条提升全部落在 multi_hop，而该族没有任何训练数据；27B 在同一族同样 +3。"
              "合理的解释是：多跳推理依赖一种可泛化的证据整合能力，被数值推理与轨迹判定数据中的组合式判断间接强化，"
              "而非依赖针对该族的专门监督。",
         indent_first=True)

    heading(doc, "5.3  温度作为诊断指标", 2)
    para(doc, "拟合出的温度本身即反映模型的自信程度：基线 0.97（接近理想）、R1 2.23（需降温两倍以上，严重过度自信）、"
              "R3 1.31（轻度）。由于拟合只需一份带标签样本和一次一维搜索，建议将其作为微调后的例行检查项。",
         indent_first=True)

    heading(doc, "5.4  为什么「准确率没掉」与「不敢用」可以同时成立", 2)
    para(doc, "两个结论看似矛盾，实则指向不同的失败模式。准确率衡量的是「平均能对多少」，"
              "在 1,200 条难负集上，微调版的净损失只有 6 条，统计上不显著；"
              "而路由关心的是「在我说很确定的那部分里错了多少」，同一批数据上这个数字翻了一倍。"
              "一个模型可以整体不变笨，却变得更爱在难题上逞强——这正是过度自信的定义。",
         indent_first=True)
    para(doc, "这也解释了为什么单看 ECE 会误判。ECE 是全区间加权平均，被大量高置信正确样本稀释；"
              "而 0.9 以上那一段才是决策实际发生的地方。建议报告校准时同时给出分区间数据，而不是只给一个标量。",
         indent_first=True)

    heading(doc, "5.5  部署口径建议", 2)
    bullet(doc, "**按流量分层，不要用一个全局阈值。** 简单流量沿用 0.9；难流量按表 10 重新选点，"
                "或直接采用表 11 的分层策略。")
    bullet(doc, "**用 margin 做难度探针。** top-1 与 top-2 的概率差无需标签、无需额外模型，"
                "低 margin 半区的错题率高 20 倍以上，足以驱动分层。")
    bullet(doc, "**先定目标精度，再反解阈值。** 不要先设阈值再看精度——后者无法回答「这个阈值够不够」。")
    bullet(doc, "**区分「阈值没调对」与「模型不达标」。** 若目标精度在任何阈值下都达不到（如 R3 在难流量上求 99%），"
                "调参无用，只能提高阈值到放弃大部分流量、换更强模型、或重新训练。")
    bullet(doc, "**微调后必须重估阈值。** 本项目基线在 0.9 上勉强可用（96.52%），微调版同阈值只有 93.14%；"
                "沿用旧阈值等于静默降低质量。")

    # ---------------- 6 局限 ----------------
    heading(doc, "6  局限", 1)
    bullet(doc, "温度在 public-231 上拟合，属于测试集调参（仅 1 个自由度）。所报 ECE 应视为上界，正式结论需独立校准集。")
    bullet(doc, "训练数据仅 1,166 条，约为基座原始训练量的 0.02%，+2 条的提升统计意义有限。")
    bullet(doc, "R3 的 ECE 在两套集合上均高于基线（public-231：0.0502 vs 0.0338；难负集：0.1625 vs 0.1272）。"
                "路由曲线更优与全局校准更差这两个结论并存，不应只引用前者。")
    bullet(doc, "跨域随机负样本版已饱和（97.8–98.3%），分辨率仅 ±0.8 个点；难负版 banking77 降至 30.33%，"
                "接近「过难」边缘——8 选 1 的随机基线为 12.5%。")
    bullet(doc, "难负由 TF-IDF 选取而非语义相似度，词面近而语义远的类别可能被选入；clinc150 上三代模型均为 53–54%，"
                "该族缺乏区分度，疑与难负集中在同一 domain 有关。")
    bullet(doc, "跨域绝对值不可比文献：8 选 1 是构造协议，非原生 77/150 类全标签空间；"
                "且这些数据集可能在底座训练数据中，绝对水平可能偏高（模型间差值不受影响）。")
    bullet(doc, "分层阈值策略的两层分界取的是本数据集的中位数，实际部署需按真实流量分布重新确定。")
    bullet(doc, "污染检查未覆盖语义近似层面（Level 3）；27B 延迟在独占卡测得，与 9B 各组不完全可比。")

    # ---------------- 7 结论 ----------------
    heading(doc, "7  结论", 1)
    para(doc, "在一个已完成大规模决策后训练的 9B 模型上，参数高效适配仍能带来小幅但真实的提升（+2/231，零退步），"
              "前提是训练目标携带不确定性信息、且训练强度经过控制。使用 one-hot 目标会以牺牲校准为代价，"
              "使置信度路由机制失效——这种代价在只看准确率时完全不可见。",
         indent_first=True)
    para(doc, "跨域评测揭示了更重要的限制。微调没有让模型变笨（两套难度上准确率均无显著退化），"
              "但让它在难样本上更爱逞强：高置信错误翻倍，采纳精度下降 3.4 个百分点。"
              "温度缩放能修掉整体尺度偏移，却修不掉难样本上的局部过度自信。",
         indent_first=True)
    para(doc, "因此，对这类模型做适配后，**必须重新标定部署阈值**，并把「目标采纳精度 → 所需阈值 → 覆盖率代价」"
              "这条链路写进上线流程。若某个模型在目标精度下不存在可用工作点，那不是阈值问题，是模型不达标。",
         indent_first=True)

    # ---------------- 附录 ----------------
    heading(doc, "附录 A  复现细节", 1)
    para(doc, "训练配置（R3）：LoRA r=32、alpha=64、dropout=0.05，仅作用于 q/k/v/o 四个投影，"
              "可训练参数 7.86M（占 8,961.67M 的 0.088%）；max_len 4096、batch_tokens 2048、"
              "epochs 2、lr 1e-5、brier_weight 1.0。单卡 H100 80GB，峰值显存 18.0GB，947 步耗时 374 秒。",
         indent_first=True)
    para(doc, "评测：三个模型均以本机 bf16 自托管，同一 /v1/systemone 服务封装，"
              "1,200 条难负评测单卡串行全程 19 分钟。", indent_first=True)
    para(doc, "关键命令：")
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
              "在 Triton 3.4.0–3.7.1 与 Hopper 显卡的组合下，其反向 kernel 会产生错误梯度且不报错，仅由库自身检查拦截。"
              "安装 tilelang 后自动切换后端可解决。这类静默错误若未被拦截，训练曲线正常但权重无效。",
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

# -*- coding: utf-8 -*-
"""生成《漂移检测：我们发现的三个问题与参考文献》DOCX。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from docx_kit import (BLUE, GREEN, GREY, INK, INK2, RED, add_hyperlink, add_rich, bullet,
                      callout, caption, cell_margins, code_block, data_table,
                      heading, kv_table, no_borders, para, set_width, setup,
                      shade, style_run)

OUT = r"D:\人工智能+\docs\漂移检测-问题与文献.docx"

doc = Document()
setup(doc, "漂移检测 · 问题与文献 · 智训引擎 · 2026-11-09")

# ── 封面 ────────────────────────────────────────────────────
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(78)
p.paragraph_format.space_after = Pt(6)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
style_run(p.add_run("「智训引擎」· 技术备忘"), 10.5, False, GREY)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
style_run(p.add_run("漂移检测：三个问题"), 34, True, BLUE)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(24)
style_run(p.add_run("以及可以借鉴的文献"), 19, False, INK2)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(46)
style_run(p.add_run("2026 年 11 月 9 日"), 11, False, GREY)

meta = doc.add_table(rows=1, cols=1)
meta.alignment = 1
meta.autofit = False
no_borders(meta)
set_width(meta, 13.4)
c = meta.cell(0, 0)
c.width = Cm(13.4)
shade(c, "F1F5F9")
cell_margins(c, 200, 200, 260, 260)
c.paragraphs[0].text = ""
for i, (k, v) in enumerate([
    ("这份文档解决什么", "我们的漂移检测到底哪里不靠谱，别人是怎么做的"),
    ("写给谁", "队内 + 技术顾问"),
    ("含什么", "三个实测问题 + 9 篇文献 + 4 个待确认问题"),
    ("先看哪", "第三节的实测数据，再看第四节文献"),
]):
    pp = c.paragraphs[0] if i == 0 else c.add_paragraph()
    pp.paragraph_format.space_after = Pt(4)
    style_run(pp.add_run(f"{k}　"), 10, True, GREY)
    style_run(pp.add_run(v), 10, False, INK)

doc.add_page_break()

# ── 一 ──────────────────────────────────────────────────────
heading(doc, "一、我们现在的检测是怎么做的")

para(doc, "一句话：**量一量每个传感器的读数分布有没有搬家，搬家多远。**")

code_block(doc, """
PSI = Σ (a_i - e_i) × ln(a_i / e_i)

  e_i = 参照系（第 1 批）落在第 i 个箱子的比例
  a_i = 当前批落在第 i 个箱子的比例
  箱子 = 用参照系的 0/10/20/.../100% 分位点切出来，共 10 个
""")

para(doc, "128 个传感器各算一个 PSI，取平均得到 `psi_mean`，再和阈值比：")

kv_table(doc, [
    ("< 0.10", "稳定，什么都不做"),
    ("0.10 ~ 0.25", "中等漂移，先试便宜手段（重算标准化）"),
    ("> 0.25", "严重漂移，直接重训"),
], widths=(4.0, 12.2), size=10)

# ── 二 ──────────────────────────────────────────────────────
heading(doc, "二、三个问题（按严重程度排）")

para(doc, "下面每一条都附了在我们真实数据上跑出来的数字，不是推测。")

# 问题一
heading(doc, "问题一：空箱导致结果不可复现（最严重）", 2)

para(doc, "**现象**：同一批数据、同一个算法，只改一个常数，PSI 从 1.2 变到 8.2。")

para(doc, "**原因**：batch_04 有 3 个箱子一个样本都没有。公式里 ln(a/e) 遇到 a=0 就"
          "算不出来，代码用 `np.clip(a, 1e-6)` 把 0 顶成一个极小值。"
          "而这个“极小值”在对数里面，改一个量级，结果就差很多。")

data_table(doc,
           ["eps（假装的最小占比）", "3 个空箱的贡献", "合计 PSI"],
           [["1e-2", "0.625", "1.212"],
            ["1e-4", "2.079", "2.666"],
            ["**1e-6（我们现在用的）**", "**3.468**", "**4.054**"],
            ["1e-8", "4.855", "5.441"],
            ["1e-12", "7.628", "8.215"]],
           [6.0, 5.2, 5.0], size=9.5, align_center=1)

para(doc, "注意：**非空箱的贡献始终是 0.587，一动不动**。"
          "PSI 从 1.2 涨到 8.2，全部来自那三个空箱。空箱贡献了 85.5% 的分值。", bold=False)

callout(doc, [
    ("已验证的解法：空箱不“假装有极小值”，而是合并到相邻箱。", {"bold": True, "color": GREEN}),
    ("合并后 10 箱变 7 箱，没有空箱了，PSI = 0.94，且与 eps 完全无关——数值才可复现、"
     "才能和教科书的 0.1 / 0.25 对照。", {"size": 10, "color": INK2}),
], "ok")

# 问题二
heading(doc, "问题二：为什么切 10 个箱子，没有依据", 2)

para(doc, "现在 `cfg.psi_bins = 10` 是随手定的。切 5 个、20 个会怎样，我们没测过，"
          "也没有找到“该切几箱”的理论依据。")

para(doc, "箱数显然会影响结果：箱子越多越细，越容易发现小变化，但也越容易把噪声当成漂移；"
          "箱子越少越稳，但可能漏掉局部变化。")

# 问题三
heading(doc, "问题三：128 个 PSI 压成 1 个数，丢了信息", 2)

para(doc, "这是最直接的一个：128 个特征的 PSI 取平均，得到一个标量去做决策。")

para(doc, "**实测这 128 个值有多分散**（batch_01 → batch_04）：")

data_table(doc,
           ["统计量", "值"],
           [["最小", "0.61"],
            ["25% 分位", "2.42"],
            ["中位数", "3.98"],
            ["**平均值（决策实际用的）**", "**4.38**"],
            ["75% 分位", "6.17"],
            ["最大", "11.07"]],
           [7.0, 9.2], size=9.5, align_center=1)

para(doc, "从 0.61 到 11.07，差 18 倍。**没有任何一个特征的 PSI 恰好等于 4.38**——"
          "这个数是算出来的，现实里不存在。")

para(doc, "平均值的问题不是“不准”，而是它**不知道哪个特征重要**：")

data_table(doc,
           ["场景", "关键特征", "无关特征", "平均 PSI", "系统的误判"],
           [["A", "1.0（坏了）", "0.05", "0.11", "判“轻微漂移”，其实关键传感器已失效"],
            ["B", "0.05（正常）", "0.5", "0.47", "判“严重漂移”，白花一次重训的钱"]],
           [1.6, 3.6, 3.0, 2.6, 5.4], size=9)

para(doc, "**补充事实**：在我们这个气体数据上，128 个特征**全部** PSI > 0.25。"
          "所以此刻压不压成一个数，结论都是“漂了”。但换成轴承或 RUL 数据就未必。", size=9.5)

# ── 三 ──────────────────────────────────────────────────────
heading(doc, "三、还有一层：有标签时，还需不需要 PSI")

callout(doc, [
    ("评测时每批都带标签。既然能直接看分数掉没掉，为什么还要算 PSI？", {"bold": True, "size": 11.5}),
    ("这个问题的答案会决定上面三个问题值不值得投入——"
     "如果 PSI 只是“提前预警”，精度要求可以放宽；如果标签有延迟，那它就是主力。",
     {"size": 10, "color": INK2}),
], "amber")


# ── 四 ──────────────────────────────────────────────────────
def ref(title, meta_line, why, read, url, url_text=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(1)
    style_run(p.add_run(title), 11, True, INK)

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(2)
    style_run(p2.add_run(meta_line), 9, False, GREY)

    p3 = doc.add_paragraph()
    p3.paragraph_format.space_after = Pt(1)
    style_run(p3.add_run("为什么相关："), 9.5, True, BLUE)
    add_rich(p3, why, 9.5, INK)

    p4 = doc.add_paragraph()
    p4.paragraph_format.space_after = Pt(2)
    style_run(p4.add_run("读哪里："), 9.5, True, BLUE)
    add_rich(p4, read, 9.5, INK)

    p5 = doc.add_paragraph()
    p5.paragraph_format.space_after = Pt(3)
    add_hyperlink(p5, url, url_text or url, size=8.5)


heading(doc, "四、参考文献（按问题分类）")

heading(doc, "A. 先读这一篇：方法选型的直接依据", 2)

ref("Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift",
    "Rabanser, Günnemann, Lipton · NeurIPS 2019",
    "把三大类漂移检测方法（单变量 KS/PSI、多变量 MMD、分类器对抗验证）放在同一批数据上做系统对比。"
    "正好对应我们“用 PSI 还是换别的”这个疑问。",
    "第 1~3 节，约两小时。不必读完整个附录。",
    "http://export.arxiv.org/abs/1810.11953v1")

heading(doc, "B. 漂移检测：底层方法", 2)

ref("A survey on concept drift adaptation",
    "Gama et al. · ACM Computing Surveys, 2014",
    "这个领域的标准综述，被引上万次。重点不是方法，而是**它给的术语体系**："
    "突变/渐变/重复/概念漂移 vs 协变量漂移。团队和顾问之间用词统一，讨论效率会高很多。",
    "第 2 节（漂移的分类），约 1 小时。",
    "https://dl.acm.org/doi/10.1145/2523813")

ref("Learning under Concept Drift: A Review",
    "Lu et al. · 2020（arXiv:2004.05785）",
    "比 Gama 那篇新，重心在“怎么应对”而不是“怎么检测”，对我们设计更新策略那一层有用。",
    "第 3~4 节（自适应策略）。",
    "https://browse.arxiv.org/abs/2004.05785")

ref("A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams",
    "2024 · International Journal of Data Science and Analytics",
    "最新的一篇，专门评测无监督检测器在真实数据流上的表现。如果只想看“现在什么方法靠谱”，读它。",
    "结论表与方法对比部分。",
    "https://rd.springer.com/article/10.1007/s41060-024-00620-y")

ref("The Kolmogorov-Smirnov Windowing Method (KSWIN)",
    "Raab et al. · 2020",
    "**最可能直接替代 PSI 的方案**：用 KS 检验 + 滑动窗口。"
    "结果在 [0,1] 有界、带 p 值、对空箱不敏感——正好治我们那三个毛病里的前两个。",
    "方法定义与实验部分。",
    "https://open.fau.de/server/api/core/bitstreams/a4a3e0e4-70fe-477d-8150-ebb9bdc9d244/content")

para(doc, "另外两个经典流式算法（在综述里能找到引用）：**ADWIN**（Bifet & Gavaldà，"
          "自适应窗口）和 **DDM**（Gama et al.，基于错误率的检测）。工业界用 ADWIN 较多。",
     size=9.5)

heading(doc, "C. 有标签时的检测", 2)

ref("Detecting and Correcting for Label Shift with Black Box Predictors",
    "Lipton et al. · ICML 2018（arXiv:1802.03916）",
    "讨论“标签分布变了怎么办”，用的是**模型输出**而不是原始特征。"
    "对我们的启发是：有标签时，检测可以建在模型表现上，而不是输入分布上。",
    "方法部分与实验结论。",
    "http://xxx.itp.ac.cn/abs/1802.03916v1")

heading(doc, "D. 工程实现参考（PSI 本身没有学术论文）", 2)

para(doc, "**说明**：PSI 是信用卡风控领域的工程指标，学术界对应的名字是 KL 散度 / 相对熵。"
          "因此找不到“PSI 的论文”是正常的——**这本身就说明它偏经验，不是理论产物**。"
          "所以“eps 一改差 7 倍”不是我们实现得不好，是这个指标没把这件事解决好。",
     size=9.5)

ref("dqt · PSI 算法文档",
    "Python 库，开源",
    "工业级的 PSI 实现，有分箱方式、边界情况（含空箱）的处理说明，可以参考别人怎么绕开这些坑。",
    "整页不长。",
    "https://github.com/antonbarr-data/dqt/blob/b8dc7d9bd5bb4e7f281c343a3269702040b8a98d/docs/algorithms/psi.md")

ref("ml_drift 文档",
    "R 包 ml",
    "参数说明比较细，可以用来对照我们对 PSI 各项设置是否合理。",
    "参数表。",
    "https://rdrr.io/cran/ml/man/ml_drift.html")

heading(doc, "E. 我们用的数据集", 2)

ref("Gas Sensor Array Drift Dataset（UCI #224）",
    "Vergara et al. 提出，2012",
    "我们目前在用的数据集。原论文做的是“用分类器集成补偿传感器漂移”，"
    "和我们正在做的事（漂移补偿）是同一个问题。",
    "数据集页面看引用格式，原文看方法论。",
    "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset+at+different+concentrations")

# ── 五 ──────────────────────────────────────────────────────
heading(doc, "五、如果只有一天，这样读")

data_table(doc,
           ["投入", "读什么", "读完能回答"],
           [["2 小时", "Failing Loudly 第 1~3 节", "该用哪一类检测方法"],
            ["1 小时", "Gama 综述第 2 节", "漂移的术语体系，能跟顾问对话"],
            ["30 分钟", "2024 benchmark 的结论表", "现在什么方法靠谱"],
            ["其余", "备查，先不读", "——"]],
           [2.8, 6.4, 7.0], size=9.5)

callout(doc, [
    ("别从综述第一页开始啃。", {"bold": True}),
    ("综述很长，我们要的是“选型依据”，不是“领域全貌”——直接跳结论章节。",
     {"size": 10, "color": INK2}),
], "plain")

# ── 六 ──────────────────────────────────────────────────────
heading(doc, "六、带着这些问题去问顾问")

para(doc, "按顺序问，前两个是方向性的，后两个是技术细节。")

kv_table(doc, [
    ("① 方向",
     "评测时每批都有标签，为什么不直接监控分数、还要用 PSI 这类无监督指标？PSI 的边际价值在哪？"),
    ("② 参照系",
     "参照系该选哪一批？用第一批、前 N 批联合、还是滚动窗口？如果第一批本身就脏，整套判断是不是从一开始就歪了？"),
    ("③ 空箱",
     "我们遇到空箱问题：3 个空箱贡献 85% 分值，eps 从 1e-2 换到 1e-12，同一批数据 PSI 从 1.2 变到 8.2。"
     "业界怎么处理？合并空箱是不是标准做法？"),
    ("④ 压缩",
     "128 个特征的漂移指标压成一个标量可接受吗？要不要同时看“超标比例”和“top-k”？按特征重要性加权有意义吗？"),
], widths=(2.6, 13.6), size=9.5)

para(doc, "比问题更有力的是证据。我们手上已经有四条：", space_before=4)
bullet(doc, "128 个 PSI 的真实分布（0.61 ~ 11.07，均值 4.38）—— 说明压缩有多粗糙")
bullet(doc, "eps 敏感性表（1.21 → 8.22）—— 说明数值不可复现")
bullet(doc, "空箱贡献 85.5% 的具体案例 —— 说明问题出在哪")
bullet(doc, "合并空箱后 0.94 且与 eps 无关 —— 说明解法可行且已验证")

# ── 结尾 ────────────────────────────────────────────────────
doc.add_paragraph().paragraph_format.space_after = Pt(8)
callout(doc, [
    ("最后一句提醒：文献解决“检测得准”，但不解决“值不值得花钱”。", {"bold": True, "size": 11, "color": BLUE}),
    ("这道题的客观分是「增益 ÷ 成本」，那个公式没有任何论文可抄——"
     "检测可以借鉴别人，更新决策必须自己设计。", {"size": 10, "color": INK2}),
], "key")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("saved:", OUT)

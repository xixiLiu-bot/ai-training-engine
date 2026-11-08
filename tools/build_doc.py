# -*- coding: utf-8 -*-
"""生成《赛题说明 · 队内共识文档》DOCX。"""
import os
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = r"D:\人工智能+\docs\赛题说明-队内共识文档.docx"
FIG1 = r"D:\人工智能+\assets\fig1_loop.png"
FIG2 = r"D:\人工智能+\assets\fig2_layers.png"

HEI = "微软雅黑"
INK = RGBColor(0x1F, 0x29, 0x37)
INK2 = RGBColor(0x4B, 0x55, 0x63)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
RED = RGBColor(0xB9, 0x1C, 0x1C)
GREEN = RGBColor(0x04, 0x78, 0x57)
GREY = RGBColor(0x6B, 0x72, 0x80)


# ── 基础工具 ────────────────────────────────────────────────
def style_run(run, size=None, bold=None, color=None, font=HEI, italic=None):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def cell_margins(cell, top=90, bottom=90, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for tag, val in (("top", top), ("bottom", bottom), ("start", left), ("end", right)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def table_borders(table, color="D1D5DB", size=4, inside=True):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    edges = ["top", "left", "bottom", "right"] + (["insideH", "insideV"] if inside else [])
    for edge in edges:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tblPr.append(borders)


def set_width(table, cm):
    """强制表格总宽（cm），避免 Word 自动撑满版心。"""
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(int(cm / 2.54 * 1440)))
    w.set(qn("w:type"), "dxa")
    tblPr.append(w)


def no_split(table):
    """禁止表格行跨页断开。"""
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        el = OxmlElement("w:cantSplit")
        el.set(qn("w:val"), "true")
        trPr.append(el)


def no_borders(table):
    tbl = table._tbl
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    tbl.tblPr.append(borders)


def para(doc, text="", size=10.5, bold=False, color=INK, space_after=6,
         align=None, indent_first=False, line=1.45, space_before=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line
    if indent_first:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        style_run(r, size, bold, color)
    return p


def heading(doc, text, level=1, num=None):
    if level == 1:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(20)
        p.paragraph_format.space_after = Pt(9)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        style_run(r, 15.5, True, BLUE)
        pb = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:space"), "4")
        bottom.set(qn("w:color"), "BFDBFE")
        pb.append(bottom)
        p._p.get_or_add_pPr().append(pb)
        return p
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    style_run(r, 12, True, RGBColor(0x1E, 0x3A, 0x8A))
    return p


def bullet(doc, text, size=10.5, level=0, marker="·"):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(16 + level * 16)
    pf.first_line_indent = Pt(-10)
    pf.space_after = Pt(3)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.4
    r = p.add_run(f"{marker} ")
    style_run(r, size, True, BLUE)
    r2 = p.add_run(text)
    style_run(r2, size, False, INK)
    return p


def callout(doc, lines, kind="key"):
    palette = {
        "key": ("EFF6FF", "2563EB"),
        "warn": ("FEF2F2", "DC2626"),
        "ok": ("ECFDF5", "059669"),
        "plain": ("F8FAFC", "94A3B8"),
    }
    fill, border = palette[kind]
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    cell = t.cell(0, 0)
    cell.width = Cm(16.2)
    shade(cell, fill)
    cell_margins(cell, 130, 130, 200, 200)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, sz in (("top", 0), ("bottom", 0), ("right", 0), ("left", 28)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if sz else "none")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), border)
        borders.append(el)
    tcPr.append(borders)

    first = True
    cell.paragraphs[0].text = ""
    for txt, kwargs in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        pf = p.paragraph_format
        pf.space_after = Pt(3)
        pf.space_before = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.4
        r = p.add_run(txt)
        style_run(r, kwargs.get("size", 10.5), kwargs.get("bold", False),
                  kwargs.get("color", INK))
    set_width(t, 16.2)
    no_split(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def data_table(doc, headers, rows, widths, header_fill="1E3A8A", size=9.5,
               zebra=True, align_center=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t)
    set_width(t, sum(widths))
    for i, w in enumerate(widths):
        t.columns[i].width = Cm(w)
    hdr = t.rows[0]
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        c.width = Cm(widths[i])
        shade(c, header_fill)
        cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(h)
        style_run(r, size, True, RGBColor(0xFF, 0xFF, 0xFF))
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            c = cells[ci]
            c.width = Cm(widths[ci])
            cell_margins(c, 80, 80, 100, 100)
            if zebra and ri % 2 == 1:
                shade(c, "F8FAFC")
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            p.paragraph_format.line_spacing = 1.25
            if align_center and ci == align_center:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            bold = val.startswith("**")
            r = p.add_run(val.replace("**", ""))
            style_run(r, size, bold, INK if not bold else RED if "❌" in val else INK)
    # 表头重复
    trPr = t.rows[0]._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)
    if len(rows) <= 6:
        for row in t.rows[:-1]:
            for cell in row.cells:
                for pp in cell.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    no_split(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def code_block(doc, text):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t, "E2E8F0", 4, inside=False)
    cell = t.cell(0, 0)
    cell.width = Cm(16.2)
    shade(cell, "F8FAFC")
    cell_margins(cell, 120, 120, 180, 180)
    cell.paragraphs[0].text = ""
    for i, line in enumerate(text.strip("\n").split("\n")):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(0)
        pf.space_before = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.15
        r = p.add_run(line)
        style_run(r, 9, False, RGBColor(0x33, 0x41, 0x55), font="Consolas")
    set_width(t, 16.2)
    no_split(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(14)
    r = p.add_run(text)
    style_run(r, 9, False, GREY)


def add_page_field(paragraph):
    run = paragraph.add_run()
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = " PAGE "
    f2 = OxmlElement("w:fldChar")
    f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1)
    run._r.append(it)
    run._r.append(f2)
    style_run(run, 9, False, GREY)


# ── 文档骨架 ────────────────────────────────────────────────
doc = Document()

normal = doc.styles["Normal"]
normal.font.name = HEI
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), HEI)
normal.paragraph_format.space_after = Pt(6)

sec = doc.sections[0]
sec.page_width = Cm(21)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.3)
sec.bottom_margin = Cm(2.2)
sec.left_margin = Cm(2.4)
sec.right_margin = Cm(2.4)
sec.different_first_page_header_footer = True

footer_p = sec.footer.paragraphs[0]
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = footer_p.add_run("「智训引擎」赛题说明 · 队内共识文档   ")
style_run(r, 9, False, GREY)
add_page_field(footer_p)

# ── 封面 ────────────────────────────────────────────────────
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(70)
p.paragraph_format.space_after = Pt(6)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("第二届北京市大学生“人工智能 +”创新大赛 · 命题赛题")
style_run(r, 10.5, False, GREY)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("“智训引擎”")
style_run(r, 40, True, BLUE)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(26)
r = p.add_run("工业制造场景下的模型训练与超参数优化")
style_run(r, 14, False, INK2)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
r = p.add_run("我 们 到 底 要 做 一 个 什 么 东 西")
style_run(r, 19, True, INK)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(50)
r = p.add_run("—— 队内共识文档 ——")
style_run(r, 11, False, GREY)

meta = doc.add_table(rows=1, cols=1)
meta.alignment = WD_TABLE_ALIGNMENT.CENTER
meta.autofit = False
no_borders(meta)
set_width(meta, 12.4)
c = meta.cell(0, 0)
c.width = Cm(12.4)
shade(c, "F1F5F9")
cell_margins(c, 200, 200, 260, 260)
lines = [
    ("适用对象", "全体队员（先读这一份，再看别的）"),
    ("文档目的", "统一认知：这道题在考什么、我们要交什么、现在该干什么"),
    ("版    本", "v1.0"),
    ("日    期", "2026 年 11 月"),
]
c.paragraphs[0].text = ""
for i, (k, v) in enumerate(lines):
    pp = c.paragraphs[0] if i == 0 else c.add_paragraph()
    pp.paragraph_format.space_after = Pt(4)
    rr = pp.add_run(f"{k}    ")
    style_run(rr, 10, True, GREY)
    rr2 = pp.add_run(v)
    style_run(rr2, 10, False, INK)

doc.add_page_break()

# ── 一 ──────────────────────────────────────────────────────
heading(doc, "一、一句话说清我们在做什么")

callout(doc, [
    ("我们要做的不是一个“更准的模型”，而是一个“能让模型一直好用的系统”。", {"bold": True, "size": 12, "color": BLUE}),
    ("它自己发现模型变差了，自己想清楚为什么变差，自己挑一个最便宜的办法修好，"
     "自己验证修得对不对，修坏了就自动退回去，并把这次的经验记下来，让下次判断得更准。",
     {"size": 10.5}),
], "key")

para(doc, "换句话说：模型是“病人”，我们造的是“医院”。", bold=True, space_before=6)
para(doc, "病人（模型）会生病（被漂移拖垮），医院（系统）要能自己诊断、自己开药、自己判断疗效、"
          "治坏了能抢救回来，而且每次看完病都要写病历，越看越有经验。", space_after=10)

para(doc, "为什么是“医院”而不是“药”？因为组委会已经把“药”给你了——他们会给数据、给评测脚本，"
          "甚至有一个“躺平”的模型做对照。他们真正想看的，是你有没有能力管理这台机器长期运转。")

# ── 二 ──────────────────────────────────────────────────────
heading(doc, "二、为什么会有这道题：模型会“变笨”")

para(doc, "想象一条真实的产线，上面装着一个质检模型，判断产品合不合格。装上那天它很准。但半年后：",
     space_after=4)
bullet(doc, "设备磨损了，摄像头成像的色调整体偏移；")
bullet(doc, "换了新供应商的原料，产品外观的分布变了；")
bullet(doc, "季节性湿度变化，传感器读数整体漂移。")
para(doc, "数据变了，模型没变。于是它开始误报、漏报。", space_before=4)

para(doc, "传统做法是：等出事了，叫工程师回来重新标数据、重新训模型、重新上线。这个过程慢、贵，"
          "而工厂里根本没有那么多人天天守着模型。")

callout(doc, [
    ("“一次训练、长期有效”在工业现场是不成立的。", {"bold": True, "size": 11.5}),
    ("数据一直在变，所以模型必须一直跟着变——但“一直变”的成本极高，你不可能每来一批数据就重训一次。"
     "这就是这道题真正的矛盾：既要跟上漂移，又要少花钱。", {"size": 10.5, "color": INK2}),
], "plain")

para(doc, "所以大模型智能体出现了之后，一个新的思路成为可能：让 AI 自己来管这件事。"
          "它读实验记录、判断趋势、调度训练——这就是“由 AI 驱动模型持续优化”。"
          "这也是赛题背景里那段话的意思。")

# ── 三 ──────────────────────────────────────────────────────
heading(doc, "三、三个必须统一的概念（最重要的一节）")

para(doc, "队里最容易出现分歧的地方，是大家对“我们在做什么”的理解不一致。"
          "下面这三个东西，务必分清楚——它们是三个层次，不是一回事。")

doc.add_picture(FIG2, width=Cm(15.8))
_p = doc.paragraphs[-1]
_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_p.paragraph_format.space_before = Pt(6)
_p.paragraph_format.space_after = Pt(0)
_p.paragraph_format.keep_with_next = True
caption(doc, "图 1　模型、闭环、技能：三个层次，三种性质")

heading(doc, "3.1　模型（第 1 层）", 2)
para(doc, "就是我们平时说的“训练出来的那个东西”，一堆权重参数。它会被训练、被更新、被替换。")
para(doc, "关键事实：它对数据是“绑定”的。换个数据集，特征维度都对不上，权重一个都用不了。"
          "所以模型是这套系统里最不可复用的部分。", space_after=8)

heading(doc, "3.2　闭环（第 2 层）", 2)
_lead = para(doc, "闭环是我们写的代码——一个流程骨架，五个环节：")
_lead.paragraph_format.keep_with_next = True

doc.add_picture(FIG1, width=Cm(16.2))
_p = doc.paragraphs[-1]
_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_p.paragraph_format.space_before = Pt(6)
_p.paragraph_format.space_after = Pt(0)
_p.paragraph_format.keep_with_next = True
caption(doc, "图 2　闭环的五个环节，以及模型在其中被触碰的位置")

para(doc, "请注意图里一个很容易被忽略的细节：五个环节中，只有 ④ 和 ⑤ 会碰到模型，"
          "①②③ 全程不读模型内部，只看数据分布和分数。", bold=False)
para(doc, "这意味着：闭环的大部分工作跟“模型是什么”无关。换模型、换数据，①②③ 照样能跑。",
     space_after=8)

heading(doc, "3.3　技能 Skill（第 3 层）", 2)
para(doc, "技能是“把一段该怎么做事的经验，写成一个可以被调用的模块”。它长这样：")
code_block(doc, """
name: 轻度协变量漂移处置
when: 特征分布明显偏移（PSI > 0.25），但标签分布基本不变
do:
  1. 先试输出阈值 / 温度校准          # 成本 ≈ 0
  2. 校准后若增益 < 2%，再试小样本增量微调  # 学习率 = 原 1/10，3 轮
  3. 每次改动都必须过影子评估，不过就回滚
""")
para(doc, "注意：这段文字里没有出现任何模型名、任何特征名。")
para(doc, "这正是技能的价值所在——它描述的是“怎么判断、按什么顺序试、什么算通过”，"
          "而不是“模型长什么样”。所以它可以跨模型复用。")
callout(doc, [
    ("如果一条“技能”只能配一个模型用，那它就不是技能，是配置项。", {"bold": True, "color": RED}),
    ("恰恰因为它能跨模型、跨数据集迁移，才配叫“经验”。", {"size": 10.5, "color": INK2}),
], "warn")

heading(doc, "3.4　三个常见的理解偏差", 2)
data_table(doc,
           ["容易想成", "实际上是", "为什么要纠正"],
           [["我们要训一个更强的模型",
             "我们要造一个能维护模型的系统",
             "模型是成品，闭环是手艺；赛题考手艺"],
            ["技能就是改模型",
             "技能是写在文档里的处置经验，一个权重都不碰",
             "改技能 ≠ 改权重，两者互不影响"],
            ["用别的数据训的模型能搬过来",
             "权重搬不过来，但闭环和技能能搬",
             "所以先用替代数据跑通流程完全值得"],
            ["自动化 = 全交给大模型",
             "大模型只负责出主意，主流程必须是确定性代码",
             "大模型会卡死、会答非所问，不能让它挡住交付"]],
           [4.5, 6.4, 5.3])

# ── 四 ──────────────────────────────────────────────────────
heading(doc, "四、我们要交什么")

heading(doc, "4.1　四类交付物", 2)
para(doc, "口径依据 11/6 说明会，比最初规则 PDF 多了一类“源码与配置”和一类“合规与展示”。")
data_table(doc,
           ["交付物", "具体是什么", "谁负责"],
           [["① 系统镜像", "Docker 镜像，装着 Harness + 模型配置 + 一键运行脚本。\n"
                           "组委会把它拉起来，断网运行。", "C 线"],
            ["② 源码与配置", "完整源代码、依赖清单、复现说明。", "全员"],
            ["③ 报告与实验", "设计论证 + SwanLab 记录链接 + 运行说明。", "C 线 + 队长"],
            ["④ 合规与展示", "开源 / 模型 / 数据声明；按模板提交演示材料。", "队长"]],
           [3.4, 9.0, 3.8])

heading(doc, "4.2　评测是怎么进行的（很重要）", 2)
para(doc, "正式评测那天，你不在场。发生的事是这样的：")
code_block(doc, """
  1. 平台拉取你提交的镜像
  2. 扔进一台【离线 + 资源受限】的机器
  3. 启动容器，灌入隐藏测试流（一批批数据按时间顺序进来）
  4. 全程计时、计 GPU 时长、计标注配额、计大模型 Token
  5. 收集你写出来的结果文件
  6. 用它自己的评测脚本算分
  7. 与静态基线（部署后不再更新的模型）比较
""")
callout(doc, [
    ("你交的是一台自动售货机。", {"bold": True, "size": 11.5}),
    ("官方把它搬进一个没有网络的房间，锁上门，往里投币（数据），看它会不会正确出货。"
     "你人不在，不能打电话，不能重启，不能改配置。出不来货就是 0 分。",
     {"size": 10.5, "color": INK2}),
], "plain")

heading(doc, "4.3　三条铁律", 2)
data_table(doc,
           ["铁律", "具体意思", "违反的后果"],
           [["不依赖交互", "不能有 input()、不能等用户确认、不能有图形界面", "进程挂起 → 超时判失败"],
            ["不依赖网络", "评测环境完全离线，任何外部 API 都调不通", "调用失败 → 流程中断"],
            ["必须有兜底", "任何时刻都要能输出一个“当前最优结果”，哪怕很平庸",
              "没输出 = 0 分，没有辩解余地"]],
           [3.0, 8.2, 5.0])
para(doc, "第三条最容易被忽略，但最要命。设想你的 Agent 在某个决策点卡住、或者返回了无法解析的结果，"
          "如果整个流程就停在那儿——这一轮就是 0 分。", space_before=4)
para(doc, "所以架构上必须有：超时 → 降级到规则引擎 → 强制输出当前最优结果。"
          "宁可交一个平庸的结果，也不能什么都不交。", bold=True)

# ── 五 ──────────────────────────────────────────────────────
heading(doc, "五、什么时候算“赢”")

heading(doc, "5.1　分数不是精度，是“性价比”", 2)
callout(doc, [
    ("客观分 = 模型性能的增益 ÷ 消耗的资源", {"bold": True, "size": 13, "color": BLUE}),
    ("资源包括三样：GPU 时长 / 训练额度、标注配额、大模型 Token。",
     {"size": 10.5, "color": INK2}),
], "key")
para(doc, "这个公式决定了整套设计思路：")
para(doc, "分母可以被无限放大——如果你每来一批数据就重训一次，分母会爆炸，分数极低。"
          "所以核心问题不是“要不要更新”，而是“用多便宜的手段能更新好”。", bold=True)

heading(doc, "5.2　更新手段的性价比阶梯", 2)
data_table(doc,
           ["手段", "成本", "什么时候用", "效果"],
           [["什么都不做", "0", "只是噪声波动，会自己恢复", "无"],
            ["改输出（阈值 / 温度校准）", "≈ 0", "协变量漂移、类别比例变了", "中"],
            ["改特征（重标准化、加新特征）", "低", "输入整体平移、缩放", "中"],
            ["增量微调（少量新数据）", "中", "概念漂移，标注预算有限", "高"],
            ["从头重训", "高", "概念漂移剧烈，上面都不行", "高"]],
           [6.2, 2.2, 5.0, 2.8], align_center=1)
para(doc, "看出关键了吗：最便宜的两档成本接近于零，却能在相当大比例的漂移场景下解决问题。")
callout(doc, [
    ("“判断”的真正价值，不是判断“要不要更新”，", {"bold": True, "color": BLUE}),
    ("而是判断“能用多便宜的手段更新好”。", {"bold": True, "size": 11.5, "color": BLUE}),
    ("别人一掉分就重训（分母爆炸），你能看出“这次只是光照变了，调个阈值就回来”（分母≈0）——"
     "同样的增益，成本差几十倍。", {"size": 10.5, "color": INK2}),
], "key")

heading(doc, "5.3　五项基础门槛", 2)
para(doc, "以下五项不满足，成绩直接不进排名。注意它们都是“及格线”，不是加分项。"
          "口径依据 11/6 说明会：比最初规则 PDF 多了第③项。")
data_table(doc,
           ["门槛", "要求", "我们的对策"],
           [["① 跑完全程", "封闭评测中无人工干预地完成全部阶段并按时输出结果",
             "兜底路径 + 全流程超时保护"],
            ["② 记录完整", "全部训练、调整、评估实验按《实验记录规范》留存",
             "从第一次实验起就按规范记，不补账"],
            ["③ 三次循环", "至少 3 次“读历史 → 提假设 → 做实验 → 做决策”，过程可查",
             "每轮决策都写 decisions.jsonl，字段齐全"],
            ["④ 不低于基线", "封闭评测综合性能不低于静态基线",
             "候选必须过影子评估才上线，否则自动回滚"],
            ["⑤ 可复现", "换机器、断网也能一键跑通，材料完整",
             "镜像自包含 + 依赖锁定 + 复现说明"]],
           [2.8, 8.6, 4.8])

heading(doc, "5.4　一句话结论", 2)
callout(doc, [
    ("这题比的是“下降曲线”，不是“起跑线”。", {"bold": True, "size": 12, "color": GREEN}),
    ("哪怕我们的模型天生只有 0.88（低于基线），只要基线掉到 0.75、我们稳在 0.86，我们就赢了。",
     {"size": 10.5, "color": INK2}),
], "ok")

# ── 六 ──────────────────────────────────────────────────────
heading(doc, "六、我们打算怎么做")

heading(doc, "6.1　技术路线：确定性引擎打底，Agent 提上限", 2)
para(doc, "赛题说可以走“持续微调”路线，也可以走“大模型驱动的智能体系统”路线。我们的判断是：")
data_table(doc,
           ["路线", "优势", "风险", "我们的取舍"],
           [["纯微调", "稳定、成本可控、好复现", "拿不到“策略层自改进”的分", "作为底座"],
            ["纯 Agent", "演示效果好、创新性强", "会卡死、成本失控、可能跑挂",
             "作为上限，不作为底座"]],
           [2.6, 5.4, 5.4, 2.8])
callout(doc, [
    ("底层：确定性引擎（规则 + 上下文赌博机），保证不崩、成本可控。", {"bold": True}),
    ("上层：LLM Agent 负责归因解释和探索没被覆盖的策略空间。", {"bold": True}),
    ("约束：LLM 越权、超预算、超时 → 自动降级回确定性引擎。", {"bold": True, "color": RED}),
], "plain")
para(doc, "这样即使 Agent 全程失效，系统依然能跑完、能出分——门槛①和④都保住了。")

heading(doc, "6.2　第一件事不是写模型，是定接口", 2)
para(doc, "因为官方数据还没发，我们只能用替代数据集（UCI 气体传感器漂移数据集）来跑通流程。"
          "要让“换数据集”从重写变成换一行代码，必须先把接口抽象出来。")
code_block(doc, """
三个接口，闭环只认它们，不认具体实现：

  DatasetAdapter   吃任意数据 → 吐出统一的 (X, y, batch_id)
  ModelAdapter     任意模型   → 统一 fit / predict / save / load
  MetricAdapter    任意指标   → 统一 score()

现在：  GasSensorAdapter + LogRegAdapter + F1Adapter   → 跑通闭环
将来：  OfficialAdapter    + 换成官方模型  + 官方指标   → 闭环一行不改
""")
para(doc, "这半天的工作，决定了后面两周是“改一行”还是“重写一遍”。")

heading(doc, "6.3　目录结构", 2)
para(doc, "三条线在同一个仓库里并行开发，目录边界就是责任边界：")
data_table(doc,
           ["目录 / 文件", "职责", "归属"],
           [["adapters/", "三个适配器的实现：数据、模型、指标", "A + B"],
            ["loop/monitor.py", "① 漂移监测", "A"],
            ["loop/diagnose.py", "② 归因诊断", "A"],
            ["loop/decide.py", "③ 决策（含预算账本）", "C"],
            ["loop/evaluate.py", "④ 执行 + 影子评估", "C"],
            ["loop/release.py", "⑤ 上线 / 回滚", "C"],
            ["skills/", "技能库：markdown 描述 + 配套小脚本", "全员"],
            ["engine/", "确定性引擎 + Agent 接口（可替换）", "C"],
            ["ledger/", "预算账本：GPU / 标注 / Token 三重硬上限", "C"],
            ["logs/", "SwanLab 接入与本地日志", "C"],
            ["Dockerfile", "镜像构建，最终交付物", "C"]],
           [4.6, 9.6, 2.0], align_center=2)

# ── 七 ──────────────────────────────────────────────────────
heading(doc, "七、三个人怎么分工")

para(doc, "三条线并行，互不阻塞。每条线都可以在官方数据到位之前独立开工。")
data_table(doc,
           ["", "A 线 · 漂移与归因", "B 线 · 模型与手段", "C 线 · 闭环与工程"],
           [["负责内容",
             "造流、算分布距离、\n判别协变量 / 概念漂移",
             "基线模型、校准、\n增量微调、成本度量",
             "调度骨架、预算账本、\n门禁回滚、Docker"],
            ["核心产出",
             "monitor.py\ndiagnose.py",
             "model_sklearn.py\n参数扫描脚本",
             "decide.py / evaluate.py\nrelease.py / 镜像"],
            ["建议人选",
             "有分布/统计基础\n（联邦学习背景高度对口）",
             "有建模经验",
             "工程感最强、最细心"],
            ["姓名", "", "", ""]],
           [2.4, 4.6, 4.6, 4.6], align_center=0)

callout(doc, [
    ("A 线特别说明：联邦学习里最难的 non-IID 问题，本质就是分布漂移。",
     {"bold": True, "color": BLUE}),
    ("“客户端数据异构检测”“按分布聚类客户端”这些方法，搬到这题几乎一一对应。有这方面背景的同学"
     "应该直接吃下 A 线。", {"size": 10.5, "color": INK2}),
], "key")

# ── 八 ──────────────────────────────────────────────────────
heading(doc, "八、官方数据没发之前，我们做什么")

para(doc, "先说一个判断：官方数据包只是把“替代数据”换成“真题”，流程完全一样。"
          "等数据是最亏的选择——真正决定校赛能不能过的，是这一个月里有没有把闭环跑通一遍。")

heading(doc, "第 0 层 · 拆雷（各 2 小时，三人并行）", 2)
para(doc, "这几件事的共同点是：不卡智力，卡运气。出问题就是半天一天，而且必须解决。")
data_table(doc,
           ["谁", "做什么", "验收标准"],
           [["A", "装 WSL2 + Docker Desktop，数据目录指到 D 盘", "docker run hello-world 跑通"],
            ["B", "Python 3.10+ 环境 + scikit-learn / lightgbm", "import sklearn 不报错"],
            ["C", "注册 SwanLab，跑一个 hello world 训练", "网页上能看到那条曲线"],
            ["全员", "建 git 仓库（Gitee 或 GitHub），三个人都能提交", "push / pull 正常"]],
           [1.8, 9.4, 5.0], align_center=0)
para(doc, "第四行别小看。三个人并行开发，代码要是还在用微信传压缩包，两周内必乱。", bold=False)

heading(doc, "第 1 层 · 定契约（半天，三个人一起）", 2)
para(doc, "坐在一起，把下面五件事写成文档（不写代码，只写约定）。这是整件事里性价比最高的半天，"
          "而且只有一次机会定对。")
code_block(doc, """
1. 一条“数据批次”长什么样      →  { X, y, batch_id, timestamp }
2. 模型必须提供哪几个方法        →  fit / predict / save / load
3. 一次“决策”的输出长什么样      →  { action, reason, expected_gain, cost }
4. 账本记哪些字段               →  { action, gpu_sec, n_labels, tokens, ts }
5. 目录结构 + 谁负责哪个目录
""")
para(doc, "定歪了，三个人各写各的，两周后合不上，返工比重写还惨。")

heading(doc, "第 2 层 · 三条线并行（约 8~10 人天）", 2)
data_table(doc,
           ["线", "任务", "完成标志"],
           [["A", "GasSensorAdapter；切 10 个批次当流；\n实现 PSI / KS / MMD；\n判协变量 vs 概念漂移",
             "能对一条流输出每批的漂移报告"],
            ["B", "训逻辑回归 + GBDT 基线；\n实现「不动 / 校准 / 增量微调」三档；\n给每档标真实成本",
             "得到一张「手段 × 效果 × 成本」表"],
            ["C", "主循环 + 批次推进；预算账本硬上限；\n影子评估 + 回滚；SwanLab 接入；Docker",
             "一条命令跑完全流程，日志齐全"]],
           [1.2, 9.4, 5.6])

heading(doc, "第 3 层 · 合流，做出那条曲线（校赛的核心资产）", 2)
callout(doc, [
    ("在图上画两条线：静态基线（一路下滑） vs 我们的系统（稳住了）。", {"bold": True, "size": 12, "color": GREEN}),
    ("一张图，胜过十页 PPT。评委看一眼就懂我们在干什么，省下来的时间全用来讲“为什么能稳住”。",
     {"size": 10.5, "color": INK2}),
], "ok")

# ── 九 ──────────────────────────────────────────────────────
heading(doc, "九、现在千万别做的四件事")
data_table(doc,
           ["别做", "为什么"],
           [["训一个大模型", "这题不考模型精度，投入产出比极低"],
            ["急着接 LLM", "闭环没通就接大脑，80% 时间会耗在调 prompt 和修 JSON 上"],
            ["追精度指标", "0.95 但不会自维护，得分不如 0.85 但稳如老狗"],
            ["等官方数据", "等待是这三个月里最贵的动作"]],
           [4.2, 12.0])

# ── 十 ──────────────────────────────────────────────────────
heading(doc, "十、还没确认的问题（需要去问）")
para(doc, "这些问题的答案会显著改变设计，建议在答疑群里直接问，拿到答复后同步到队里。")
data_table(doc,
           ["#", "问题", "为什么重要"],
           [["1", "模型权重是预置在镜像里，还是评测时从零训练？", "决定初始训练成本算不算我们的账"],
            ["2", "单次评测的总时长限制是多少？", "决定兜底和降级策略要多激进"],
            ["3", "静态基线的分数是否公布？是否提供代码？", "决定能不能本地对标、算自己排第几"],
            ["4", "漂移类型是渐变还是突变？是否混合？", "决定监测窗口和阈值设计"],
            ["5", "评测指标是什么（F1 / MAE / 其他）？", "决定优化目标"],
            ["6", "评测期间调用大模型是否计入 Token 成本？", "决定 Agent 能“想”多少次"],
            ["7", "隐藏测试流是否有中间反馈？", "决定是闭眼跑还是能边跑边调"],
            ["8", "标注配额如何计量？我们能否主动“申请标注”？", "决定主动采样验证的策略"]],
           [1.0, 9.4, 6.0], align_center=0)

# ── 十一 ────────────────────────────────────────────────────
heading(doc, "十一、名词表（看不明白时回这里查）")
data_table(doc,
           ["词", "解释"],
           [["漂移（Drift）", "数据分布随时间变化，导致原来训练的模型失准"],
            ["协变量漂移", "输入分布变了，但“什么算合格”的规则没变（如光照变化）"],
            ["概念漂移", "输入和标签的关系变了（如客户改了验收标准），必须重训"],
            ["静态基线", "官方设定的对照模型：部署后永不更新。五项门槛之一的对标线"],
            ["单位增益", "每消耗一份资源换来的性能提升，客观分的核心指标"],
            ["影子评估", "新模型先在后台并行跑，不影响线上，用它的表现决定是否上线"],
            ["标注配额", "允许你花在“打标签”上的预算，是一种要计入分母的成本"],
            ["Harness", "套在大模型外面的那层脚手架：工具 + 循环 + 约束，让“想”变成“做”"],
            ["Skill（技能）", "把一段处置经验写成可复用的模块，跨模型通用"],
            ["RSI", "递归自改进：系统改进自身（在这里是改进自己的决策策略）"],
            ["Adapter", "适配器：把不同数据/模型/指标包装成统一接口的转换层"]],
           [3.4, 12.8])

# ── 结尾 ────────────────────────────────────────────────────
heading(doc, "十二、行动清单（对着勾）")
data_table(doc,
           ["", "动作", "负责人", "截止"],
           [["□", "装 WSL2 + Docker，数据目录指向 D 盘", "A", "本周"],
            ["□", "装 Python 3.10+ 环境，跑通 import sklearn", "B", "本周"],
            ["□", "注册 SwanLab，跑通一次记录上传", "C", "本周"],
            ["□", "建 git 仓库，三个人都能提交", "全员", "本周"],
            ["□", "读完本文档，把不懂的地方标出来，下次开会讲", "全员", "本周"],
            ["□", "在答疑群发出第十节那 8 个问题", "队长", "本周"],
            ["□", "向指导老师确认“第一版”到底要什么", "队长", "本周"],
            ["□", "三人开会定接口（3 小时，一次定完）", "全员", "下周"]],
           [1.2, 9.2, 2.8, 3.0], align_center=None)

doc.add_paragraph().paragraph_format.space_after = Pt(6)
callout(doc, [
    ("这道题看起来像机器学习题，本质是软件工程题。", {"bold": True, "size": 11.5, "color": BLUE}),
    ("它的核心矛盾是“会变”——数据会变、环境会变、模型会变。所以真正考的是："
     "你能不能区分出“什么该跟着变、什么不该跟着变”，并让该变的部分容易换掉。",
     {"size": 10.5, "color": INK2}),
    ("想清楚这一点，剩下的都是编码工作。", {"bold": True, "size": 10.5, "color": INK}),
], "key")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("saved:", OUT)

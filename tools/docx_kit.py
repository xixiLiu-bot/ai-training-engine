# -*- coding: utf-8 -*-
"""DOCX 排版工具集：中文文档常用的标题、表格、提示框、代码块。"""
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HEI = "微软雅黑"
INK = RGBColor(0x1F, 0x29, 0x37)
INK2 = RGBColor(0x4B, 0x55, 0x63)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
RED = RGBColor(0xB9, 0x1C, 0x1C)
GREEN = RGBColor(0x04, 0x78, 0x57)
GREY = RGBColor(0x6B, 0x72, 0x80)

PALETTE = {
    "key": ("EFF6FF", "2563EB"),
    "warn": ("FEF2F2", "DC2626"),
    "ok": ("ECFDF5", "059669"),
    "plain": ("F8FAFC", "94A3B8"),
    "amber": ("FFFBEB", "D97706"),
}


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


CODE_INK = RGBColor(0x1F, 0x4E, 0x79)


def add_rich(p, text, size=10.5, color=INK, bold=False, font=HEI, bold_color=None):
    """解析两种行内语法：**加粗** 与 `等宽代码`。"""
    wrote = False
    for k, chunk in enumerate(text.split("`")):
        if not chunk:
            continue
        if k % 2 == 1:                       # 反引号内 → 等宽
            style_run(p.add_run(chunk), max(size - 0.5, 7), bold, CODE_INK,
                      "Consolas")
            wrote = True
            continue
        for i, seg in enumerate(chunk.split("**")):   # 处理加粗
            if not seg:
                continue
            is_bold = bold or (i % 2 == 1)
            style_run(p.add_run(seg), size, is_bold,
                      bold_color if (is_bold and bold_color) else color, font)
            wrote = True
    if not wrote:
        style_run(p.add_run(""), size, bold, color, font)
    return p


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
    tblPr = table._tbl.tblPr
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


def no_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    table._tbl.tblPr.append(borders)


def set_width(table, cm):
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(int(cm / 2.54 * 1440)))
    w.set(qn("w:type"), "dxa")
    tblPr.append(w)


def no_split(table):
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        el = OxmlElement("w:cantSplit")
        el.set(qn("w:val"), "true")
        trPr.append(el)


def para(doc, text="", size=10.5, bold=False, color=INK, space_after=6,
         align=None, line=1.45, space_before=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    if text:
        add_rich(p, text, size, color, bold)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    if level == 1:
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
    else:
        p.paragraph_format.space_before = Pt(13)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        style_run(r, 12, True, RGBColor(0x1E, 0x3A, 0x8A))
    return p


def bullet(doc, text, size=10.5, marker="·", indent=16):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(indent)
    pf.first_line_indent = Pt(-10)
    pf.space_after = Pt(3)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.4
    r = p.add_run(f"{marker} ")
    style_run(r, size, True, BLUE)
    add_rich(p, text, size, INK)
    return p


def callout(doc, lines, kind="key", width=16.2):
    fill, border = PALETTE[kind]
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    cell = t.cell(0, 0)
    cell.width = Cm(width)
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

    cell.paragraphs[0].text = ""
    first = True
    for txt, kwargs in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        pf = p.paragraph_format
        pf.space_after = Pt(3)
        pf.space_before = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.4
        add_rich(p, txt, kwargs.get("size", 10.5), kwargs.get("color", INK),
                 kwargs.get("bold", False))
    set_width(t, width)
    no_split(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def data_table(doc, headers, rows, widths, header_fill="1E3A8A", size=9.5,
               zebra=True, align_center=None, first_col_bold=False):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t)
    set_width(t, sum(widths))
    for i, w in enumerate(widths):
        t.columns[i].width = Cm(w)

    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
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
            if align_center is not None and ci == align_center:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_rich(p, val, size, INK, first_col_bold and ci == 0)

    trPr = t.rows[0]._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)
    if len(rows) <= 6:                       # 小表尽量不跨页
        for row in t.rows[:-1]:
            for cell in row.cells:
                for pp in cell.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    no_split(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def code_block(doc, text, size=9, ls=1.15, width=16.2):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t, "E2E8F0", 4, inside=False)
    cell = t.cell(0, 0)
    cell.width = Cm(width)
    shade(cell, "F8FAFC")
    cell_margins(cell, 120, 120, 180, 180)
    cell.paragraphs[0].text = ""
    for i, line in enumerate(text.strip("\n").split("\n")):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(0)
        pf.space_before = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = ls
        r = p.add_run(line)
        style_run(r, size, False, RGBColor(0x33, 0x41, 0x55), font="Consolas")
    set_width(t, width)
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


def setup(doc, footer_text):
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

    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run(footer_text + "   ")
    style_run(r, 9, False, GREY)
    add_page_field(fp)
    return sec

def kv_table(doc, rows, widths=(3.0, 13.2), size=9.5, label_fill="F1F5F9"):
    """无表头的两列键值表；内容按换行拆成多个段落，多条时自动加圆点。"""
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t, "D1D5DB", 4, inside=True)
    set_width(t, sum(widths))
    for i, w in enumerate(widths):
        t.columns[i].width = Cm(w)

    for label, content in rows:
        cells = t.add_row().cells
        c0, c1 = cells[0], cells[1]
        for c, w in ((c0, widths[0]), (c1, widths[1])):
            c.width = Cm(w)
            cell_margins(c, 85, 85, 110, 110)
        shade(c0, label_fill)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        style_run(p0.add_run(label), size, True, INK)

        lines = [x for x in content.split("\n") if x.strip()]
        multi = len(lines) > 1
        c1.paragraphs[0].text = ""
        for k, ln in enumerate(lines):
            p = c1.paragraphs[0] if k == 0 else c1.add_paragraph()
            pf = p.paragraph_format
            pf.space_after = Pt(2 if multi else 0)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.3
            if multi:
                r = p.add_run("· ")
                style_run(r, size, True, BLUE)
            add_rich(p, ln, size, INK)

    # 让整张卡片尽量不跨页
    for row in t.rows[:-1]:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.keep_with_next = True
    no_split(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def add_hyperlink(paragraph, url, text, size=9, color=RGBColor(0x1D, 0x4E, 0xD8), font=HEI):
    """在段落里插入一个可点击的超链接。"""
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml import OxmlElement as _OE

    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)
    hl = _OE("w:hyperlink")
    hl.set(qn("r:id"), r_id)

    run = _OE("w:r")
    rPr = _OE("w:rPr")
    rFonts = _OE("w:rFonts")
    rFonts.set(qn("w:ascii"), font)
    rFonts.set(qn("w:hAnsi"), font)
    rFonts.set(qn("w:eastAsia"), font)
    rPr.append(rFonts)
    c = _OE("w:color")
    c.set(qn("w:val"), "%02X%02X%02X" % (color[0], color[1], color[2]))
    rPr.append(c)
    u = _OE("w:u"); u.set(qn("w:val"), "single"); rPr.append(u)
    sz = _OE("w:sz"); sz.set(qn("w:val"), str(int(size * 2))); rPr.append(sz)
    run.append(rPr)

    t = _OE("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    run.append(t)
    hl.append(run)
    paragraph._p.append(hl)
    return paragraph

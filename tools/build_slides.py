# -*- coding: utf-8 -*-
"""生成《系统全流程说明》PPTX。"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# ── 版式常量 ────────────────────────────────────────────────
SW, SH = 13.333, 7.5
M = 0.62                    # 左右边距
FONT = "微软雅黑"

NAVY = RGBColor(0x1E, 0x3A, 0x8A)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
BLUE_L = RGBColor(0xEF, 0xF6, 0xFF)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
ORANGE_L = RGBColor(0xFF, 0xF7, 0xED)
GREEN = RGBColor(0x05, 0x96, 0x69)
GREEN_L = RGBColor(0xEC, 0xFD, 0xF5)
RED = RGBColor(0xB9, 0x1C, 0x1C)
RED_L = RGBColor(0xFE, 0xF2, 0xF2)
GREY = RGBColor(0x6B, 0x72, 0x80)
GREY_L = RGBColor(0xF1, 0xF5, 0xF9)
INK = RGBColor(0x1F, 0x29, 0x37)
SUB = RGBColor(0x4B, 0x55, 0x63)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def font(run, size, bold=False, color=INK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT)


def textbox(slide, left, top, width, height, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top),
                                  Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.paragraphs[0].alignment = align
    return tb


def para(tf, text, size, bold=False, color=INK, space_before=0, space_after=4,
         first=False, align=PP_ALIGN.LEFT, line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    r = p.add_run()
    r.text = text
    font(r, size, bold, color)
    return p


def rect(slide, left, top, width, height, fill=None, line=None, radius=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height))
    if radius:
        shape.adjustments[0] = radius
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1.25)
    shape.shadow.inherit = False
    return shape


# ── 页面骨架 ────────────────────────────────────────────────
def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def head(slide, title, kicker=None):
    """标题 + 底部强调线。"""
    rect(slide, M, 0.34, 0.10, 0.56, fill=BLUE)
    tb = textbox(slide, M + 0.22, 0.30, SW - 2 * M, 0.68)
    para(tb.text_frame, title, 27, True, NAVY, first=True)
    if kicker:
        tb2 = textbox(slide, M + 0.22, 0.98, SW - 2 * M, 0.34)
        para(tb2.text_frame, kicker, 13.5, False, GREY, first=True)
    line = slide.shapes.add_connector(1, Inches(M), Inches(1.34),
                                      Inches(SW - M), Inches(1.34))
    line.line.color.rgb = RGBColor(0xDB, 0xE4, 0xF0)
    line.line.width = Pt(1)


def bullets(slide, items, left, top, width, size=15, gap=0.42):
    """items: [(文本, 强调色 or None), ...]"""
    for i, (text, color) in enumerate(items):
        y = top + i * gap
        rect(slide, left, y + 0.10, 0.075, 0.075, fill=color or BLUE)
        tb = textbox(slide, left + 0.24, y, width - 0.24, gap)
        tf = tb.text_frame
        para(tf, text, size, False, INK, first=True, line=1.15)


def table(slide, headers, rows, left, top, width, col_ratios,
          header_fill=NAVY, size=11.5, row_h=0.36, first_bold=False,
          highlight=None):
    n = len(headers)
    total_ratio = sum(col_ratios)
    gw = width - 0.0
    tbl_h = row_h * (len(rows) + 1)
    shape = slide.shapes.add_table(len(rows) + 1, n, Inches(left),
                                   Inches(top), Inches(width),
                                   Inches(tbl_h))
    t = shape.table
    t.first_row = False
    widths = [gw * r / total_ratio for r in col_ratios]
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for j, h in enumerate(headers):
        cell = t.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.margin_left = Inches(0.08)
        cell.margin_right = Inches(0.06)
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        font(r, size, True, WHITE)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i + 1, j)
            cell.fill.solid()
            is_hl = highlight is not None and i == highlight
            cell.fill.fore_color.rgb = (BLUE_L if is_hl else
                                        (GREY_L if i % 2 else WHITE))
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            font(r, size, (first_bold and j == 0) or is_hl, INK)
    return t


def kv_blocks(slide, items, left, top, width, gap=0.86, label_w=1.9):
    for i, (k, v) in enumerate(items):
        y = top + i * gap
        rect(slide, left, y, width, gap - 0.12, fill=GREY_L)
        tb = textbox(slide, left + 0.16, y + 0.09, label_w, 0.4)
        para(tb.text_frame, k, 12.5, True, GREY, first=True)
        tb2 = textbox(slide, left + 0.16 + label_w, y + 0.06,
                      width - label_w - 0.34, gap - 0.2)
        para(tb2.text_frame, v, 13.5, False, INK, first=True, line=1.1)


def note(slide, text, left, top, width, kind="blue", size=12.5):
    fills = {"blue": (BLUE_L, BLUE), "red": (RED_L, RED),
             "green": (GREEN_L, GREEN), "orange": (ORANGE_L, ORANGE)}
    fill, accent = fills[kind]
    h = 0.52 if len(text) < 58 else 0.84
    rect(slide, left, top, width, h, fill=fill)
    rect(slide, left, top, 0.06, h, fill=accent)
    tb = textbox(slide, left + 0.24, top, width - 0.42, h,
                 anchor=MSO_ANCHOR.MIDDLE)
    para(tb.text_frame, text, size, True, accent, first=True, line=1.2)


def picture(slide, path, left, top, width=None, height=None):
    if not os.path.exists(path):
        return None
    kw = {}
    if width:
        kw["width"] = Inches(width)
    if height:
        kw["height"] = Inches(height)
    return slide.shapes.add_picture(path, Inches(left), Inches(top), **kw)


# ── 开始 ────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)

A = r"D:\人工智能+\assets"
O = r"D:\人工智能+\out"
E = r"D:\人工智能+\out_tx_exp"

# ═══ 1 封面 ═══
s = new_slide(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x0F, 0x1B, 0x3A))
rect(s, 0, 0, 0.14, SH, fill=BLUE)

tb = textbox(s, 1.0, 1.5, 11.4, 0.5)
para(tb.text_frame, "「智训引擎」· 系统说明", 14, False, RGBColor(0x93, 0xB4, 0xE8),
     first=True)

tb = textbox(s, 1.0, 2.05, 11.4, 1.0)
para(tb.text_frame, "这套程序到底是怎么跑的", 40, True, WHITE, first=True)

tb = textbox(s, 1.0, 3.1, 11.0, 0.6)
para(tb.text_frame, "从一批数据进来，到一个决定做出来——每一步用了什么算法、什么模型",
     16, False, RGBColor(0xBF, 0xCD, 0xE4), first=True)

rect(s, 1.0, 4.2, 4.2, 0.05, fill=BLUE)
tb = textbox(s, 1.0, 4.45, 11.0, 0.9)
tf = tb.text_frame
para(tf, "第二屆北京市大学生“人工智能+”创新大赛  ·  命题赛题", 12,
     False, RGBColor(0x93, 0xB4, 0xE8), first=True)
para(tf, "2026 年 11 月 10 日", 12, False, RGBColor(0x6B, 0x7C, 0x99))

# ═══ 2 速览 ═══
s = new_slide(prs)
head(s, "一页看完：我们做的是什么", "不是“训一个更准的模型”，是“造一个会让模型一直好用的系统”")

cards = [
    ("在做的事", "在会漂移的数据上，\n无人干预地维持模型性能", BLUE, BLUE_L),
    ("现在的状态", "闭环跑通，两个真实数据集\n气体传感器 + 轴承退化", GREEN, GREEN_L),
    ("缺的东西", "Docker 镜像、SwanLab 记录\n（队友在做）", ORANGE, ORANGE_L),
]
for i, (t, d, col, bg) in enumerate(cards):
    x = M + i * 4.05
    rect(s, x, 1.62, 3.85, 1.62, fill=bg, line=col)
    tb = textbox(s, x + 0.26, 1.82, 3.35, 0.4)
    para(tb.text_frame, t, 15, True, col, first=True)
    tb = textbox(s, x + 0.26, 2.32, 3.35, 1.0)
    para(tb.text_frame, d, 13, False, SUB, first=True, line=1.3)

tb = textbox(s, M, 3.78, 12.1, 0.42)
para(tb.text_frame, "一次运行的数字（气体传感器，10 批）", 15, True, NAVY, first=True)

table(s, ["指标", "静态基线", "我们的系统", "说明"],
      [["平均 F1", "0.320", "0.786", "高出 2.4 倍"],
       ["末批 F1", "0.280", "0.718", "没有一路掉到底"],
       ["累计成本", "—", "4.5 秒", "含 5 次更新"],
       ["合规性", "—", "10 轮 / 违规 0", "协议走完"]],
      M, 4.24, 12.1, [2.2, 2.2, 2.4, 5.3], size=12, row_h=0.42, first_bold=True)

note(s, "这题考的是“系统是否长期可用”，不是“某一次模型的分数”。", M, 6.44, 12.1)

# ═══ 3 硬约束 ═══
s = new_slide(prs)
head(s, "赛题给的硬约束", "这些不是建议，是直接决定架构的前提")

table(s, ["约束", "内容", "逼出了什么设计"],
      [["算力", "4 核 8G Linux，无 GPU", "只能用 sklearn 级模型，神经网络也必须是小的"],
       ["体积", "提交包 ≤ 2G，严禁打包权重", "冷启动必须现场训练，训练成本进分母"],
       ["网络", "禁止外部 API，仅平台模型服务", "大模型只能接 Qwen，且必须可降级"],
       ["数据", "批次下发，先交卷才给下一批", "结构上无法偷看未来（防过拟合）"],
       ["标签", "每批都有标签，不耗配额", "可以真实验证，但也意味着会被查过拟合"],
       ["门槛", "含 Harness、≥3 次迭代、不低于基线", "缺一不进排名"]],
      M, 1.6, 12.1, [1.5, 4.6, 6.0], size=11.5, row_h=0.52, first_bold=True)

note(s, "最反直觉的一条：先交卷才给下一批。不是交互设计，是防过拟合的结构性手段。",
     M, 5.72, 12.1, "orange")

# ═══ 4 系统全貌 ═══
s = new_slide(prs)
head(s, "系统全貌", "左边是主流程，右边是四个“以后要改就改这里”的位置")
picture(s, os.path.join(A, "fig3_design.png"), 0.62, 1.50, width=8.0)

tb = textbox(s, 8.95, 1.55, 3.85, 5.4)
tf = tb.text_frame
para(tf, "五层结构", 14, True, NAVY, first=True, space_after=6)
for t, d in [("protocol/", "一问一答，禁止偷看未来"),
             ("adapters/", "数据、模型、指标适配器"),
             ("loop/", "监测→归因→决策→评估→回滚"),
             ("engine/", "规则 + 经验 + 大模型建议"),
             ("ledger/", "预算账本，超限即拒")]:
    para(tf, t, 12, True, BLUE, space_before=8, space_after=1)
    para(tf, d, 10.5, False, SUB, space_after=2)

# ═══ 5 闭环总览 ═══
s = new_slide(prs)
head(s, "每一批数据，都走这五步", "①②③ 不读模型内部，只有 ④⑤ 会碰权重")
picture(s, os.path.join(A, "fig1_loop.png"), 1.35, 1.50, width=9.0)

table(s, ["环节", "算法/方法", "输出"],
      [["① 漂移监测", "PSI（群体稳定性指数）", "128 个特征各一个值 → 平均 + 超标比例"],
       ["② 归因诊断", "标签分布 + 条件分布启发式", "协变量 / 概念 / 稳定"],
       ["③ 决策", "6 条规则 + 动作成本阶梯", "三档动作之一 + 理由"],
       ["④ 执行评估", "影子评估（同批对比）", "新旧模型的增益"],
       ["⑤ 上线回滚", "门控（阈值 0.02）", "部署 or 退回旧模型"]],
      M, 5.34, 12.1, [2.0, 4.6, 5.5], size=10.5, row_h=0.32, first_bold=True)

# ═══ 6 监测：PSI ═══
s = new_slide(prs)
head(s, "① 漂移监测：PSI 怎么算", "衡量“读数分布有没有搬家”，跟模型内部无关")

tb = textbox(s, M, 1.55, 6.4, 0.5)
para(tb.text_frame, "公式", 14, True, BLUE, first=True)

rect(s, M, 2.02, 6.4, 0.95, fill=BLUE_L, line=BLUE)
tb = textbox(s, M + 0.3, 2.14, 5.9, 0.75, anchor=MSO_ANCHOR.MIDDLE)
tf = tb.text_frame
para(tf, "PSI  =  Σ  (aᵢ − eᵢ) · ln(aᵢ / eᵢ)", 22, True, NAVY, first=True)
para(tf, "eᵢ = 参照系占比    aᵢ = 当前批占比", 11.5, False, SUB, space_before=2)

tb = textbox(s, M, 3.16, 6.4, 0.35)
para(tb.text_frame, "三步", 14, True, BLUE, first=True)
steps = ["按参照系的 0/10/…/100% 分位点切 10 个箱子",
         "数每箱里落了多少，算出占比",
         "套公式求和——搬得越远，值越大"]
for i, t in enumerate(steps):
    y = 3.58 + i * 0.44
    rect(s, M, y + 0.09, 0.075, 0.075, fill=BLUE)
    tb = textbox(s, M + 0.24, y, 6.0, 0.4)
    para(tb.text_frame, f"{i+1}. {t}", 13, False, INK, first=True)

note(s, "PSI 恒 ≥ 0：因为 (a−e) 和 ln(a/e) 永远同号，负偏差也是“搬家”。", M, 5.0, 6.4, "blue", 11.5)

tb = textbox(s, 7.35, 1.55, 5.3, 0.35)
para(tb.text_frame, "真实数据：1 号传感器", 13.5, True, NAVY, first=True)

table(s, ["批次", "中位读数", "PSI"],
      [["第 1 批（参照系）", "73,889", "—"],
       ["第 4 批", "8,029", "0.86"],
       ["第 8 批", "647", "7.12"]],
      7.35, 1.98, 5.3, [2.6, 1.5, 1.0], size=11, row_h=0.4, highlight=2)

tb = textbox(s, 7.35, 3.86, 5.3, 1.0)
para(tb.text_frame, "读数掉了一百倍，分布整体左移 → 传感器老化。", 12, False, SUB,
     first=True, line=1.2)
para(tb.text_frame, "PSI 就是把这个过程压成一个数。", 12, False, SUB, line=1.2)

note(s, "已知缺陷：空箱时 eps 一改，同一批数据 PSI 能从 1.2 变到 8.2。", 7.35, 5.0, 5.3,
     "red", 11)

tb = textbox(s, M, 5.86, 12.1, 0.32)
para(tb.text_frame, "经验阈值（但必须按数据集重新标定——这是已知问题）", 12.5, True, NAVY,
     first=True)
chips = [("PSI < 0.1", "稳定，什么都不用动", GREEN, GREEN_L),
         ("0.1 ~ 0.25", "中等，先试 0.05 秒的校准", ORANGE, ORANGE_L),
         ("PSI > 0.25", "显著，考虑重训", RED, RED_L)]
for i, (a, b, c, bg) in enumerate(chips):
    x = M + i * 4.05
    rect(s, x, 6.24, 3.85, 0.78, fill=bg, line=c)
    tb = textbox(s, x + 0.24, 6.32, 3.4, 0.3)
    para(tb.text_frame, a, 13, True, c, first=True)
    tb = textbox(s, x + 0.24, 6.64, 3.4, 0.3)
    para(tb.text_frame, b, 10.5, False, SUB, first=True)

# ═══ 7 归因 ═══
s = new_slide(prs)
head(s, "② 归因诊断：判断“为什么变差”", "这一步决定后面花多少钱——是校准（0.05 秒）还是重训（2 秒）")

table(s, ["输入信号", "算法", "说明"],
      [["psi_mean", "128 个特征 PSI 的平均", "整体漂了多少"],
       ["psi_ratio", "超阈值特征的比例", "是“少数坏了”还是“整体坏了”"],
       ["label_psi", "标签分布的 PSI", "哪些类出现了、哪些消失了"]],
      M, 1.6, 7.6, [2.0, 3.0, 2.6], size=11, row_h=0.46, first_bold=True)

tb = textbox(s, M, 3.42, 7.6, 0.35)
para(tb.text_frame, "判别规则", 14, True, BLUE, first=True)
table(s, ["条件", "结论", "动作"],
      [["PSI 低", "稳定", "不动"],
       ["PSI 高 + 标签稳", "协变量漂移", "先校准"],
       ["PSI 高 + 标签变", "概念漂移", "重训"]],
      M, 3.84, 7.6, [3.4, 2.0, 2.2], size=11, row_h=0.42)

note(s, "已知缺陷：把“这批没测某种气体”误判成概念漂移。等算法同学改。", M, 5.66, 7.6,
     "red", 11)

tb = textbox(s, 8.55, 1.6, 4.1, 5.2)
tf = tb.text_frame
para(tf, "为什么这一步最值钱", 14, True, NAVY, first=True, space_after=6)
para(tf, "选错了会怎样：", 12, True, INK, space_before=4, space_after=4)
para(tf, "· 明明是尺度问题，却花了 2 秒重训 —— 浪费 40 倍成本", 11.5, False, SUB,
     space_after=4, line=1.15)
para(tf, "· 明明是关系变了，却反复校准 —— 一直救不回来", 11.5, False, SUB,
     space_after=10, line=1.15)
para(tf, "赛题客观分 = 增益 ÷ 成本", 12, True, INK, space_after=4)
para(tf, "所以“判断准”直接等于“分数高”。", 11.5, False, SUB, line=1.15)

# ═══ 8 决策 ═══
s = new_slide(prs)
head(s, "③ 决策：6 条规则，从便宜到贵挑一档", "命中就停；顺序是刻意排的——先省钱，再考虑效果")

table(s, ["序", "条件", "动作", "为什么在这"],
      [["1", "预热期（前几批）", "不动", "先收集历史"],
       ["2", "PSI < 稳定阈值", "不动", "动了纯属浪费"],
       ["3", "距上次更新太近", "不动", "防止来回抖"],
       ["4", "上次校准没用（增益 <0.02）", "重训", "廉价手段已证明无效"],
       ["5", "PSI ≥ 重型阈值", "重训", "漂太猛，别浪费一次机会"],
       ["5b", "超标比例 ≥ 50%", "重训", "局部校准救不了全局"],
       ["6", "以上都不满足", "先校准", "省钱就省在这里"]],
      M, 1.58, 8.0, [0.7, 4.0, 1.4, 2.6], size=10.5, row_h=0.44,
      highlight=6)

tb = textbox(s, 8.95, 1.58, 3.75, 0.35)
para(tb.text_frame, "动作成本（按模型不同）", 13, True, NAVY, first=True)
table(s, ["模型", "校准", "重训"],
      [["logreg", "0.05s", "0.5s"],
       ["gbdt", "0.05s", "2.2s"],
       ["mlp", "0.05s", "1.2s"]],
      8.95, 1.98, 3.75, [1.3, 1.1, 1.1], size=11, row_h=0.4)

note(s, "动作清单由模型声明，决策器只从它能做的里面挑。换模型不用改代码。", 8.95, 3.84, 3.75,
     "green", 10.5)

tb = textbox(s, 8.95, 4.86, 3.75, 1.9)
tf = tb.text_frame
para(tf, "三层决策", 12.5, True, NAVY, first=True, space_after=4)
para(tf, "规则引擎（保底）", 11, True, BLUE, space_after=1)
para(tf, "↓", 11, False, GREY, space_after=1)
para(tf, "经验表（有证据才改判）", 11, True, GREEN, space_after=1)
para(tf, "↓", 11, False, GREY, space_after=1)
para(tf, "大模型（过四道闸才采纳）", 11, True, ORANGE)

# ═══ 9 执行 ═══
s = new_slide(prs)
head(s, "④ 执行：三档手段，各自到底做了什么", "成本差 40 倍，所以顺序很重要")

blocks = [
    ("什么都不做", "0 秒", "不碰模型。分布稳定时这是正确决策，不是偷懒。", GREY, GREY_L),
    ("重算标准化", "0.05 秒", "只换“量的尺子”，模型权重一个字节都不动。", GREEN, GREEN_L),
    ("重训模型", "0.5~2.2 秒", "在最近几批上继续训练（不是从零重建）。", ORANGE, ORANGE_L),
]
for i, (t, cost, d, col, bg) in enumerate(blocks):
    x = M + i * 4.05
    rect(s, x, 1.6, 3.85, 1.72, fill=bg, line=col)
    tb = textbox(s, x + 0.24, 1.76, 3.4, 0.4)
    para(tb.text_frame, t, 16, True, col, first=True)
    tb = textbox(s, x + 0.24, 2.16, 3.4, 0.3)
    para(tb.text_frame, f"成本 {cost}", 11.5, False, GREY, first=True)
    tb = textbox(s, x + 0.24, 2.52, 3.4, 0.72)
    para(tb.text_frame, d, 11.5, False, SUB, first=True, line=1.2)

tb = textbox(s, M, 3.6, 12.1, 0.35)
para(tb.text_frame, "“重算标准化”到底改了什么（真实数字）", 14, True, NAVY, first=True)
table(s, ["状态", "第 1 号特征的均值", "标准差"],
      [["校准前（参照系）", "125,598", "154,335"],
       ["校准后（当前批）", "49,496", "60,475"],
       ["模型权重", "−0.0818, 0.106", "→ 完全没变"]],
      M, 4.02, 7.4, [3.2, 2.4, 1.8], size=11, row_h=0.42, highlight=2)

tb = textbox(s, 8.5, 4.02, 4.2, 2.2)
tf = tb.text_frame
para(tf, "为什么它几乎免费", 13, True, NAVY, first=True, space_after=6)
para(tf, "模型学的是“读数偏高还是偏低”。", 11.5, False, SUB, space_after=3, line=1.2)
para(tf, "只要尺子和读数一起变，判断就不变。", 11.5, False, SUB, space_after=3, line=1.2)
para(tf, "所以只要重算尺子，不用动权重。", 11.5, True, GREEN, space_after=8, line=1.2)
para(tf, "实测：气体数据上，这一档让 F1 从 0.26 回到 0.60。", 11.5, False, SUB, line=1.2)

note(s, "顺序不能反：如果一上来就重训，等于每次都用最贵的刀切小问题 —— 成本翻 40 倍，效果还未必更好。",
     M, 5.72, 12.1, "blue", 12)

# ═══ 10 门控 ═══
s = new_slide(prs)
head(s, "⑤ 门控与回滚：绝不越改越差", "“不低于静态基线”这条门槛，就靠这一步保命")

tb = textbox(s, M, 1.55, 6.3, 0.35)
para(tb.text_frame, "判定方式：同一批数据上比", 14, True, NAVY, first=True)

flow = [("当前模型", 0.42), ("同一批打分", 0.42), ("候选模型", 0.42)]
y = 2.05
rect(s, M, y, 2.3, 0.62, fill=GREY_L, line=GREY)
tb = textbox(s, M, y + 0.12, 2.3, 0.4, align=PP_ALIGN.CENTER)
para(tb.text_frame, "当前模型", 13, True, SUB, first=True, align=PP_ALIGN.CENTER)

rect(s, M + 2.55, y, 1.35, 0.62, fill=BLUE_L, line=BLUE)
tb = textbox(s, M + 2.55, y + 0.16, 1.35, 0.35, align=PP_ALIGN.CENTER)
para(tb.text_frame, "比一比", 12, True, BLUE, first=True, align=PP_ALIGN.CENTER)

rect(s, M + 4.15, y, 2.3, 0.62, fill=ORANGE_L, line=ORANGE)
tb = textbox(s, M + 4.15, y + 0.12, 2.3, 0.4, align=PP_ALIGN.CENTER)
para(tb.text_frame, "候选模型", 13, True, ORANGE, first=True, align=PP_ALIGN.CENTER)

table(s, ["增益", "决定", "后果"],
      [["≥ 0.02", "上线", "版本 +1，参照系跟着挪"],
       ["< 0.02", "回滚", "什么都不换，继续用旧的"]],
      M, 2.98, 6.3, [1.6, 1.6, 3.1], size=11.5, row_h=0.46, first_bold=True)

note(s, "关键：两方必须在同一批上打分。拿不同批次的分数比，是两把尺子量身高。", M, 4.3,
     6.3, "red", 11)

tb = textbox(s, 7.35, 1.55, 5.35, 0.35)
para(tb.text_frame, "真实发生的一次回滚（第 6 批）", 13.5, True, NAVY, first=True)
table(s, ["时点", "F1"],
      [["校准前", "0.440"],
       ["校准后", "0.334"],
       ["增益", "−0.106 → 回滚"]],
      7.35, 1.96, 5.35, [2.8, 1.6], size=11, row_h=0.42, highlight=2)

tb = textbox(s, 7.35, 3.98, 5.35, 2.2)
tf = tb.text_frame
para(tf, "它在保护什么", 13, True, NAVY, first=True, space_after=6)
para(tf, "没有这道闸，一次失败的更新会直接把性能拉低，而且没人知道要退回去。", 11.5,
     False, SUB, line=1.25)
para(tf, "有了这道闸，最坏情况就是“什么都没变”—— 打平基线，不会更差。", 11.5, True,
     GREEN, space_before=6, line=1.25)

# ═══ 11 模型层 ═══
s = new_slide(prs)
head(s, "模型层：四个可选，默认最简单的", "不是越复杂越好——4 核 8G 无 GPU，能跑起来才有意义")

table(s, ["模型", "是什么", "训练耗时", "气体 F1", "轴承 F1"],
      [["logreg（默认）", "逻辑回归：128 个数加权求和", "0.5 秒", "0.786", "0.457"],
       ["mlp", "神经网络：64→32 两层，非线性", "1.2 秒", "0.641", "0.401"],
       ["rf", "随机森林：80 棵树投票", "1.5 秒", "0.546", "—"],
       ["gbdt", "梯度提升树：100 棵树互相纠错", "2.2 秒", "0.437", "0.411"]],
      M, 1.62, 12.1, [2.3, 5.0, 1.6, 1.6, 1.6], size=11, row_h=0.5, first_bold=True)

note(s, "单点精度高 ≠ 闭环表现好。GBDT 单批 F1 最高，但重训贵，在闭环里反而吃亏。", M, 4.28,
     12.1, "orange")

tb = textbox(s, M, 5.08, 12.1, 1.5)
tf = tb.text_frame
para(tf, "为什么默认用逻辑回归", 13.5, True, NAVY, first=True, space_after=4)
para(tf, "① 训练 0.5 秒 → 成本低，敢频繁更新   ② 权重可解释，答辩能讲清“它是看哪个传感器判断的”   "
         "③ 对“重算标准化”这一档反应最灵敏（+0.33 F1）", 11.5, False, SUB, line=1.25)

# ═══ 12 经验表 ═══
s = new_slide(prs)
head(s, "策略层自改进：系统会“记笔记”", "这是“越做越聪明”的实体，也是主观分的加分点")

tb = textbox(s, M, 1.55, 6.6, 0.35)
para(tb.text_frame, "每条记录长这样", 13.5, True, NAVY, first=True)
table(s, ["漂移类型", "做了什么", "次数", "累计增益", "单位成本增益"],
      [["协变量", "校准", "5", "+0.604", "2.42"],
       ["协变量", "重训", "3", "−0.362", "−0.06"],
       ["概念", "重训", "1", "+0.076", "0.04"]],
      M, 1.96, 6.6, [1.5, 1.3, 0.9, 1.5, 1.8], size=10.5, row_h=0.44,
      highlight=0)

tb = textbox(s, M, 3.66, 6.6, 1.0)
para(tb.text_frame, "翻译：这类漂移下，校准每分钟换 2.42 分，重训反而亏钱。", 12, False, SUB,
     first=True, line=1.25)
para(tb.text_frame, "下一批遇到同类情况，先查表，不再试错。", 12, True, GREEN, space_before=3)

tb = textbox(s, 7.6, 1.55, 5.1, 0.35)
para(tb.text_frame, "两个防坑设计（都是踩过才知道的）", 13, True, NAVY, first=True)
kv_blocks(s, [
    ("失败惩罚", "最近连着两次回滚的动作，不再推荐"),
    ("20% 探索", "有一定概率无视经验走规则，避免陷在过时经验里"),
], 7.6, 1.98, 5.1, gap=1.0, label_w=1.4)

note(s, "没有这两条时，经验表让分数从 0.722 掉到 0.709 —— 第 3 批成功后，它以为校准永远有效。",
     7.6, 4.16, 5.1, "red", 10.5)

note(s, "答辩价值：“系统在第 3 批学会了：协变量漂移下，校准的单位成本增益是重训的 40 倍。”",
     M, 4.86, 6.6, "green", 10.5)

# ═══ 13 一次完整运行 ═══
s = new_slide(prs)
head(s, "一次完整运行：10 批数据，10 次决策", "气体传感器数据，从头跑到尾")

table(s, ["批", "PSI", "判断", "动作", "分数", "结果", "读作"],
      [["1", "—", "初始", "训练", "0.672", "建立参照系", "先把基准立起来"],
       ["2", "0.77", "稳定", "不动", "0.672", "—", "还没到管的程度"],
       ["3", "1.42", "协变量", "校准", "0.582", "上线", "0.05 秒解决"],
       ["4", "4.38", "协变量", "重训", "0.805", "上线", "漂太猛，换模型"],
       ["5", "2.38", "协变量", "校准", "0.984", "回滚", "这次校准没用"],
       ["6", "2.30", "概念", "校准", "0.672", "回滚", "规则没变却掉分"],
       ["7", "2.11", "概念", "重训", "0.824", "上线", "升级到贵的"],
       ["8", "6.52", "概念", "重训", "0.882", "回滚", "重训也没救活"],
       ["9", "1.88", "概念", "重训", "0.737", "上线", "再来一次，成了"],
       ["10", "1.35", "概念", "校准", "0.718", "回滚", "维持现状"]],
      M, 1.58, 12.1, [0.6, 1.0, 1.4, 1.2, 1.3, 1.7, 3.6], size=10, row_h=0.42,
      first_bold=True)

note(s, "注意：不是每批都更新，10 批里只动了 7 次。另外 3 批判定“不该动”——那也是决策。",
     M, 6.28, 12.1, "blue")

# ═══ 14 实测结果 ═══
s = new_slide(prs)
head(s, "实测：基线下滑，我们稳住", "气体传感器，10 批，同一批数据上对比")
picture(s, os.path.join(O, "curve.png"), 0.7, 1.55, width=8.4)

tb = textbox(s, 9.35, 1.62, 3.4, 4.8)
tf = tb.text_frame
para(tf, "三个关键点", 13.5, True, NAVY, first=True, space_after=8)
for t, d in [("前两批不动", "PSI 只有 0.02 / 0.07，动了纯属浪费"),
             ("第 4 批拉回", "0.05 秒校准，0.475 → 0.976"),
             ("第 6 批到底", "判定标准变了，便宜手段失效，主动回滚")]:
    para(tf, t, 11.5, True, BLUE, space_before=4, space_after=1)
    para(tf, d, 10.5, False, SUB, space_after=4, line=1.15)

note(s, "平均 0.320 → 0.786；成本 4.5 秒。", 0.7, 5.86, 8.4, "green", 12)

# ═══ 15 迁移实验 ═══
s = new_slide(prs)
head(s, "最有说服力的一个实验：经验能带到新数据上", "在 10 个旧轴承上学，到 5 个全新轴承上用")
picture(s, os.path.join(E, "迁移实验结果.png"), 0.95, 1.52, width=11.4)

cols = [("做了什么", "前 10 个轴承攒经验 → 后 5 个（全新工况）验证。两者从没见过面。"),
        ("差在哪", "同一批、同一个 PSI（5.22）：没经验选重训，失败；有经验选校准，成功。"),
        ("结果", "分数 +32%，成本 −48%，性价比 4.1 倍。")]
for i, (k, v) in enumerate(cols):
    x = M + i * 4.05
    tb = textbox(s, x, 5.46, 3.75, 0.3)
    para(tb.text_frame, k, 12.5, True, NAVY, first=True)
    tb = textbox(s, x, 5.78, 3.75, 0.95)
    para(tb.text_frame, v, 10.5, False, SUB, first=True, line=1.2)

note(s, "局限：只有 5 批、同一种失效模式。这是“机制验证通过”，不是“效果已证明”。",
     M, 6.86, 12.1, "orange", 10.5)

# ═══ 16 决策记录 ═══
s = new_slide(prs)
head(s, "每一轮都留痕：这是“至少 3 次迭代”的证据", "门槛③要求“完整记录”，我们每批写一条")

rect(s, M, 1.55, 7.3, 4.2, fill=RGBColor(0xF8, 0xFA, 0xFC),
     line=RGBColor(0xE2, 0xE8, 0xF0))
tb = textbox(s, M + 0.22, 1.68, 6.9, 4.0)
tf = tb.text_frame
tf.word_wrap = True
code = [
    '{',
    '  "batch": 4,',
    '  "history_read": "近几批 PSI 0.02 → 0.07；当前分数 0.988",',
    '  "hypothesis": "诊断=covariate；特征整体平移",',
    '  "experiment": {',
    '      "action": "refresh_normalization",',
    '      "cost_sec": 0.05,',
    '      "decider": "rule+memory"',
    '  },',
    '  "result": { "f1_before": 0.4752, "f1_after": 0.9762 },',
    '  "decision": "accept",',
    '  "reason": "同批对比：增益 +0.5011 ≥ 门控 0.02"',
    '}',
]
for i, line in enumerate(code):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(3)
    p.line_spacing = 1.25
    r = p.add_run()
    r.text = line
    font(r, 11.5, False, RGBColor(0x33, 0x41, 0x55))
    r.font.name = "Consolas"

tb = textbox(s, 8.25, 1.58, 4.45, 0.35)
para(tb.text_frame, "四个字段是生死线", 13, True, NAVY, first=True)
kv_blocks(s, [
    ("history_read", "证明读过历史，不是拍脑袋"),
    ("hypothesis", "证明有假设，不是瞎试"),
    ("experiment", "证明真做过、花了多少"),
    ("reason", "证明有依据，不是碰运气"),
], 8.25, 2.0, 4.45, gap=0.94, label_w=1.35)

note(s, "这三样齐、跑够 3 条，门槛③就过了。也是答辩时最好讲的东西——一张卡，一句话。",
     8.25, 5.9, 4.45, "green", 10.5)

# ═══ 17 能力边界 ═══
s = new_slide(prs)
head(s, "现在能做什么，不能做什么", "说清楚边界，比吹牛有用")

tb = textbox(s, M, 1.58, 5.9, 0.35)
para(tb.text_frame, "能做", 14, True, GREEN, first=True)
items_do = ["判断数据漂不漂，漂了多少",
            "三档手段里挑一档，先便宜后贵",
            "改完自己验收，不行自动退回",
            "记账：算力 / 标注 / Token",
            "记住哪招好使，下次直接用",
            "换数据集 / 换模型不改主流程"]
for i, t in enumerate(items_do):
    y = 2.0 + i * 0.5
    rect(s, M, y + 0.12, 0.08, 0.08, fill=GREEN)
    tb = textbox(s, M + 0.26, y, 5.6, 0.42)
    para(tb.text_frame, t, 12.5, False, INK, first=True)

tb = textbox(s, 6.75, 1.58, 5.9, 0.35)
para(tb.text_frame, "不能做 / 还没做", 14, True, RED, first=True)
items_dont = ["归因判得不准（等算法同学改）",
              "门控没有统计检验（0.02 是拍的）",
              "成本是估算，不是真实计时",
              "只有分类，没有回归（RUL 未接）",
              "Docker 镜像、SwanLab 记录",
              "大模型接了框架，但没接真模型"]
for i, t in enumerate(items_dont):
    y = 2.0 + i * 0.5
    rect(s, 6.75, y + 0.12, 0.08, 0.08, fill=RED)
    tb = textbox(s, 7.01, y, 5.6, 0.42)
    para(tb.text_frame, t, 12.5, False, INK, first=True)

note(s, "前三条是“分数天花板”，后三条是“能不能交”。后者更急。", M, 5.24, 12.1, "orange")

tb = textbox(s, M, 6.02, 12.1, 0.3)
para(tb.text_frame, "接下来 3 天", 12.5, True, NAVY, first=True)
plan = [("① 镜像 + 记录", "队友在做 · 决定能不能交"),
        ("② 归因重构", "算法同学 · 决定分数天花板"),
        ("③ 深度学习", "先用 mlp · 不够再考虑 PyTorch")]
for i, (a, b) in enumerate(plan):
    x = M + i * 4.05
    rect(s, x, 6.38, 3.85, 0.68, fill=GREY_L)
    tb = textbox(s, x + 0.22, 6.44, 3.45, 0.28)
    para(tb.text_frame, a, 11.5, True, BLUE, first=True)
    tb = textbox(s, x + 0.22, 6.74, 3.45, 0.28)
    para(tb.text_frame, b, 10, False, SUB, first=True)

# ═══ 18 结尾 ═══
s = new_slide(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x0F, 0x1B, 0x3A))
rect(s, 0, 0, 0.14, SH, fill=BLUE)

tb = textbox(s, 1.0, 1.85, 11.4, 1.2)
tf = tb.text_frame
para(tf, "一句话讲完这套系统", 15, False, RGBColor(0x93, 0xB4, 0xE8), first=True,
     space_after=14)
para(tf, "它盯着一批批数据，决定该不该动模型；", 25, True, WHITE, space_after=6)
para(tf, "动完自己验收，不行就退回去；", 25, True, WHITE, space_after=6)
para(tf, "并把每次的经验记下来。", 25, True, WHITE)

rect(s, 1.0, 5.05, 4.2, 0.05, fill=BLUE)
tb = textbox(s, 1.0, 5.3, 11.4, 1.2)
tf = tb.text_frame
para(tf, "看起来像机器学习题，本质是软件工程题。", 14, False, RGBColor(0xBF, 0xCD, 0xE4),
     first=True, space_after=4)
para(tf, "它考的不是“你能训出多准的模型”，而是“你能不能判断什么时候该动、什么时候不该动”。",
     14, False, RGBColor(0x93, 0xB4, 0xE8))

out = r"D:\人工智能+\docs\系统全流程说明.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved:", out, "slides:", len(prs.slides))

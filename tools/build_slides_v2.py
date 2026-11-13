# -*- coding: utf-8 -*-
"""生成《上次之后我们改了什么》PPTX（简版，一页一个观点）。"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

SW, SH = 13.333, 7.5
M = 0.7
FONT = "微软雅黑"

NAVY = RGBColor(0x14, 0x2B, 0x5E)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
BLUE_L = RGBColor(0xEF, 0xF6, 0xFF)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
ORANGE_L = RGBColor(0xFF, 0xF4, 0xEA)
GREEN = RGBColor(0x04, 0x7A, 0x57)
GREEN_L = RGBColor(0xEC, 0xFD, 0xF5)
RED = RGBColor(0xB9, 0x1C, 0x1C)
RED_L = RGBColor(0xFE, 0xF2, 0xF2)
GREY = RGBColor(0x6B, 0x72, 0x80)
GREY_L = RGBColor(0xF3, 0xF6, 0xFA)
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
    for a in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, a, 0)
    tf.paragraphs[0].alignment = align
    return tb


def para(tf, text, size, bold=False, color=INK, sb=0, sa=4, first=False,
         align=PP_ALIGN.LEFT, line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(sb)
    p.space_after = Pt(sa)
    if line:
        p.line_spacing = line
    r = p.add_run()
    r.text = text
    font(r, size, bold, color)


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


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def head(slide, title, kicker=None):
    rect(slide, M, 0.36, 0.10, 0.58, fill=BLUE)
    tb = textbox(slide, M + 0.24, 0.30, SW - 2 * M, 0.72)
    para(tb.text_frame, title, 28, True, NAVY, first=True)
    if kicker:
        tb2 = textbox(slide, M + 0.24, 1.02, SW - 2 * M, 0.36)
        para(tb2.text_frame, kicker, 14, False, GREY, first=True)
    c = slide.shapes.add_connector(1, Inches(M), Inches(1.42),
                                   Inches(SW - M), Inches(1.42))
    c.line.color.rgb = RGBColor(0xDD, 0xE5, 0xF1)
    c.line.width = Pt(1)


def table(slide, headers, rows, left, top, width, ratios, size=12,
          row_h=0.44, hl=None, header_fill=NAVY, bold_first=True):
    n = len(headers)
    tot = sum(ratios)
    shape = slide.shapes.add_table(len(rows) + 1, n, Inches(left), Inches(top),
                                   Inches(width),
                                   Inches(row_h * (len(rows) + 1)))
    t = shape.table
    t.first_row = False
    for j, r in enumerate(ratios):
        t.columns[j].width = Inches(width * r / tot)
    for j, h in enumerate(headers):
        c = t.cell(0, j)
        c.fill.solid()
        c.fill.fore_color.rgb = header_fill
        for a in ("margin_left", "margin_right"):
            setattr(c, a, Inches(0.09))
        c.margin_top = Inches(0.02)
        c.margin_bottom = Inches(0.02)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        font(r, size, True, WHITE)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i + 1, j)
            c.fill.solid()
            is_hl = (hl is not None and i in hl)
            c.fill.fore_color.rgb = (BLUE_L if is_hl else (GREY_L if i % 2 else WHITE))
            for a in ("margin_left", "margin_right"):
                setattr(c, a, Inches(0.09))
            c.margin_top = Inches(0.02)
            c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(v)
            font(r, size, (bold_first and j == 0) or is_hl, INK)
    return t


def note(slide, text, left, top, width, kind="blue", size=13):
    fills = {"blue": (BLUE_L, BLUE), "red": (RED_L, RED),
             "green": (GREEN_L, GREEN), "orange": (ORANGE_L, ORANGE)}
    fill, acc = fills[kind]
    h = 0.60 if len(text) < 52 else 0.92
    rect(slide, left, top, width, h, fill=fill)
    rect(slide, left, top, 0.07, h, fill=acc)
    tb = textbox(slide, left + 0.26, top, width - 0.45, h,
                 anchor=MSO_ANCHOR.MIDDLE)
    para(tb.text_frame, text, size, True, acc, first=True, line=1.2)


def cards(slide, items, top, height=1.55, gap=0.22):
    """items: [(标题, 说明, 色, 底色), ...]  横向等分"""
    n = len(items)
    w = (SW - 2 * M - gap * (n - 1)) / n
    for i, (t, d, col, bg) in enumerate(items):
        x = M + i * (w + gap)
        rect(slide, x, top, w, height, fill=bg, line=col)
        tb = textbox(slide, x + 0.22, top + 0.16, w - 0.44, 0.4)
        para(tb.text_frame, t, 15, True, col, first=True)
        tb = textbox(slide, x + 0.22, top + 0.60, w - 0.44, height - 0.72)
        para(tb.text_frame, d, 11.5, False, SUB, first=True, line=1.25)


prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)

# ═══ 1 封面 ═══
s = new_slide(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x0E, 0x1A, 0x38))
rect(s, 0, 0, 0.16, SH, fill=BLUE)
tb = textbox(s, 1.0, 1.7, 11.4, 0.5)
para(tb.text_frame, "智训引擎 · 进展汇报", 15, False, RGBColor(0x93, 0xB4, 0xE8),
     first=True)
tb = textbox(s, 1.0, 2.2, 11.4, 1.0)
para(tb.text_frame, "上次聊完之后，我们改了什么", 42, True, WHITE, first=True)
tb = textbox(s, 1.0, 3.35, 11.0, 0.7)
para(tb.text_frame, "七处改动 · 三个反面结果 · 四个数据集的实测", 17, False,
     RGBColor(0xC3, 0xD3, 0xEA), first=True)
rect(s, 1.0, 4.45, 4.0, 0.05, fill=BLUE)
tb = textbox(s, 1.0, 4.7, 11.0, 1.0)
tf = tb.text_frame
para(tf, "第二屆北京市大学生「人工智能+」创新大赛 · 命题赛题", 12.5, False,
     RGBColor(0x93, 0xB4, 0xE8), first=True)
para(tf, "2026 年 11 月 13 日", 12.5, False, RGBColor(0x6B, 0x7C, 0x99))

# ═══ 2 一页看完 ═══
s = new_slide(prs)
head(s, "一页看完：现在到哪了", "四个数据集全部端到端跑通，两种任务、两类漂移")

table(s, ["数据集", "任务", "指标", "静态基线", "我们的系统", "变化"],
      [["合成漂移流", "分类", "F1 ↑", "0.503", "0.868", "+73%"],
       ["气体传感器", "分类", "F1 ↑", "0.320", "0.783", "+145%"],
       ["轴承退化", "分类", "F1 ↑", "0.402", "0.507", "+26%"],
       ["C-MAPSS", "回归", "MAE ↓", "16.414", "14.259", "−13%"]],
      M, 1.62, 12.0, [2.6, 1.1, 1.3, 1.5, 1.8, 1.3], size=13, row_h=0.52)

note(s, "MAE 越小越好，和分类的 F1 不可直接比。这一条我们会主动说明。",
     M, 4.35, 12.0, "orange")

cards(s, [("两个任务", "分类 + 回归\n都覆盖", GREEN, GREEN_L),
          ("两类漂移", "协变量 + 概念\n都有真实案例", BLUE, BLUE_L),
          ("240 个测试", "含 13 个故障注入\n0 协议违规", NAVY, GREY_L)], 5.15, 1.6)

# ═══ 3 总览：上次 → 现在 ═══
s = new_slide(prs)
head(s, "核心对比：上次 vs 现在", "这张表回答“你们这段时间做了什么”")

table(s, ["", "上次汇报时", "现在"],
      [["数据", "只有合成数据", "4 个数据集（真实数据 3 份）"],
       ["分类成绩", "0.836（仅合成）", "0.868 / 0.783 / 0.507"],
       ["回归任务", "没有", "C-MAPSS，MAE 14.26"],
       ["成本阶梯", "3 档", "4 档（多了“增量学习”）"],
       ["经验表", "只有机制，没验证", "跨数据集迁移 +37%"],
       ["PSI", "有已知缺陷（空箱不可复现）", "已修：与 eps 无关"],
       ["模型", "1 个", "7 个 + 自动选标准化方式"],
       ["测试", "42 个", "240 个"]],
      M, 1.58, 12.0, [1.6, 4.4, 5.2], size=12, row_h=0.52, hl={4, 5})

# ═══ 4 改动一：PSI 空箱 ═══
s = new_slide(prs)
head(s, "改动 1：PSI 的空箱问题——修了", "上次我们主动交代过这个缺陷，现在它没了")

tb = textbox(s, M, 1.58, 5.6, 0.4)
para(tb.text_frame, "问题：同一批数据，改一个常数，结果差 5 倍", 14, True, RED,
     first=True)
table(s, ["做法", "eps=1e-2", "eps=1e-6", "eps=1e-12"],
      [["改之前（硬顶）", "1.59", "4.38", "8.43"],
       ["改之后（Laplace）", "1.66", "1.66", "1.66"]],
      M, 2.02, 5.6, [1.9, 1.2, 1.2, 1.3], size=12, row_h=0.46, hl={1})

tb = textbox(s, 6.85, 1.58, 5.85, 0.4)
para(tb.text_frame, "我们还试过“合并空箱”，但它是错的", 14, True, ORANGE, first=True)
table(s, ["场景", "旧做法", "合并空箱", "Laplace"],
      [["部分重叠", "4.38", "1.75", "1.66"],
       ["完全不重叠", "12.43", "0.00 ✗", "5.49"]],
      6.85, 2.02, 5.85, [1.8, 1.3, 1.5, 1.3], size=12, row_h=0.46, hl={1})

note(s, "合并空箱会把“两个分布完全不重叠”错误地压成 0 —— 方向反了。所以改用 Laplace。",
     M, 3.52, 5.6, "red", 11.5)
note(s, "最后一种在两种场景下方向都正确，而且没有需要调的常数。",
     6.85, 3.52, 5.85, "green", 11.5)

tb = textbox(s, M, 4.95, 12.0, 0.4)
para(tb.text_frame, "顺带发现一件必须更正的事", 14, True, NAVY, first=True)
note(s, "我们之前到处引用的 PSI ≈ 4.38，是被空箱虚高的。真值约 1.66。以后引用要用新值。",
     M, 5.4, 12.0, "orange")

# ═══ 5 改动二：阈值重标定 ═══
s = new_slide(prs)
head(s, "改动 2：阈值必须跟着 PSI 的尺度一起改", "这是改动 1 的连带影响，我们一开始漏了")

cards(s, [("改 PSI 之前", "阈值倍数\n1.0 / 2.0", GREY, GREY_L),
          ("改 PSI 之后", "PSI 尺度变了 2.6 倍\n阈值还是老倍数 → 触发变少", RED, RED_L),
          ("修正", "倍数改 0.55 / 1.6\n阈值跟着尺度走", GREEN, GREEN_L)], 1.62, 1.7)

tb = textbox(s, M, 3.62, 12.0, 0.4)
para(tb.text_frame, "这一处的效果（同一套代码，只改阈值倍数）", 14, True, NAVY,
     first=True)
table(s, ["数据集", "改之前", "改之后"],
      [["气体传感器", "0.776", "0.783"],
       ["轴承退化", "0.436", "0.477"]],
      M, 4.05, 7.6, [3.0, 2.0, 2.0], size=13, row_h=0.5, hl={1})

note(s, "教训：改一个指标的尺度，所有依赖它的阈值都要重新推一遍。", M, 5.95, 12.0)

# ═══ 6 改动三：百分位归一 ═══
s = new_slide(prs)
head(s, "改动 3：轴承——把特征换成“寿命百分位”", "轴承最大的问题是：个体差异比退化阶段还大")

tb = textbox(s, M, 1.58, 12.0, 0.7)
tf = tb.text_frame
para(tf, "同一个轴承内部能分到 0.89~0.98，但换一个轴承就崩 —— 因为每个轴承的振动绝对水平不同（装配公差、传感器位置）。",
     13, False, SUB, first=True, line=1.25)

tb = textbox(s, M, 2.38, 7.4, 0.4)
para(tb.text_frame, "办法：标签是“寿命百分位”，那让特征也变成百分位", 13.5, True,
     NAVY, first=True)
table(s, ["训练 → 测试", "原来的标准化", "百分位归一"],
      [["批1 → 批3", "0.725", "0.762"],
       ["批1 → 批4", "0.189", "0.401"],
       ["批2 → 批3", "0.430", "0.834"],
       ["批2 → 批4", "0.185", "0.569"],
       ["平均", "0.426", "0.649"]],
      M, 2.82, 7.4, [2.2, 2.2, 2.0], size=12, row_h=0.44, hl={4})

tb = textbox(s, 8.5, 2.38, 4.2, 0.4)
para(tb.text_frame, "进完整闭环", 13.5, True, NAVY, first=True)
table(s, ["", "系统分"], [["标准化", "0.436"], ["百分位", "0.493"], ["自动选", "0.507"]],
      8.5, 2.82, 4.2, [1.6, 1.4], size=12, row_h=0.44, hl={2})

note(s, "“自动选”的意思是：系统在第 3 批拿前两批比较两种方式，自己挑好的，不用手调。",
     M, 5.55, 12.0, "blue")

# ═══ 7 改动四：回归 ═══
s = new_slide(prs)
head(s, "改动 4：C-MAPSS——换模型 + 加滚动特征", "RUL 和传感器读数不是线性关系，线性模型吃不下")

table(s, ["做法", "MAE（越小越好）", "说明"],
      [["线性回归（原来）", "17.94", "只能拟合直线"],
       ["梯度提升树", "14.26", "能吃到“退化前慢后快”的形状"],
       ["改善", "−21%", "加滚动均值/标准差/差分做时序特征"]],
      M, 1.62, 12.0, [3.4, 3.0, 5.6], size=13, row_h=0.52, hl={2})

note(s, "这是队友做的。我原本以为“C-MAPSS 数字不好看是因为选错了数据子集”，实测证明是模型选错了。",
     M, 3.82, 12.0, "orange")

cards(s, [("滚动特征是什么", "把前几个周期的读数\n聚合成趋势，而不只看当前点", BLUE, BLUE_L),
          ("为什么对 C-MAPSS 有效", "每行只是“一个周期的读数”，\n没有时间上下文", GREEN, GREEN_L),
          ("为什么对轴承无效", "轴承每个样本本来就是\n1.28 秒振动的汇总", RED, RED_L)],
      4.9, 1.75)

# ═══ 8 改动五：增量学习 ═══
s = new_slide(prs)
head(s, "改动 5：成本阶梯从 3 档变成 4 档", "中间补了一级：真的改权重，但比重训便宜 4 倍")

table(s, ["档位", "成本", "做了什么"],
      [["不动", "0 秒", "分布稳定时这就是正确决策"],
       ["校准", "0.05 秒", "只换标准化统计量，权重一个字节不动"],
       ["增量更新", "0.10 秒", "只喂新数据，在现权重上学 3 轮"],
       ["全量重训", "0.40 秒", "旧数据 + 新数据一起从头训"]],
      M, 1.62, 12.0, [2.4, 2.2, 7.4], size=13, row_h=0.5, hl={2})

tb = textbox(s, M, 4.28, 7.4, 0.4)
para(tb.text_frame, "单步实验（同一批数据）", 13.5, True, NAVY, first=True)
table(s, ["做法", "耗时", "F1"], [["不更新", "—", "0.340"], ["增量 3 轮", "17 ms", "0.843"],
                                  ["全量重训", "158 ms", "0.830"]],
      M, 4.7, 7.4, [2.4, 2.0, 1.8], size=12, row_h=0.44, hl={1})

note(s, "全局效果：成本降低 50~70%，绝对分低 5~10%。按“单位增益”口径，性价比提升 2~3 倍。",
     8.5, 4.7, 4.2, "green", 11.5)
note(s, "轮数不能多：10 轮反而掉到 0.657 —— 只喂一批数据会把它背下来，冲掉之前学的。",
     8.5, 5.85, 4.2, "red", 11)

# ═══ 9 改动六：经验表 ═══
s = new_slide(prs)
head(s, "改动 6：经验表——从“有机制”到“能带走”", "上次只有设计，这次有了实测证据")

tb = textbox(s, M, 1.58, 6.0, 0.4)
para(tb.text_frame, "记什么", 13.5, True, NAVY, first=True)
table(s, ["漂移类型", "动作", "次数", "单位成本增益"],
      [["协变量", "校准", "5", "+2.42"],
       ["协变量", "重训", "3", "−0.06"]],
      M, 2.0, 6.0, [1.6, 1.3, 1.0, 2.0], size=12, row_h=0.46, hl={0})

tb = textbox(s, M, 3.55, 6.0, 1.6)
tf = tb.text_frame
para(tf, "怎么用（四条限制）", 13.5, True, NAVY, first=True, sa=6)
para(tf, "试过 ≥2 次才算数 · 连输两次就划掉", 11.5, False, SUB, sa=3, line=1.2)
para(tf, "要高出 1.5 倍才改判 · 另有 20% 概率故意不听", 11.5, False, SUB, line=1.2)

tb = textbox(s, 7.1, 1.58, 5.6, 0.4)
para(tb.text_frame, "真正的价值：能跨数据集带走", 13.5, True, GREEN, first=True)
table(s, ["配置", "系统分"],
      [["前 10 个轴承上学经验", "0.544"],
       ["后 5 个新轴承 · 不带经验", "0.272"],
       ["后 5 个新轴承 · 带经验", "0.372"]],
      7.1, 2.0, 5.6, [3.4, 1.6], size=12, row_h=0.48, hl={2})
note(s, "同一批、同一个 PSI：没经验选重训失败，有经验选校准成功。", 7.1, 3.95, 5.6,
     "green", 11.5)

note(s, "已知局限（主动说）：经验表没按模型分桶，换模型后旧经验的经济账就不对了。",
     M, 5.5, 12.0, "red")

# ═══ 10 改动七：多种子 ═══
s = new_slide(prs)
head(s, "改动 7：结论从“一个数”变成“一个区间”", "只跑一次的数字没有说服力")

table(s, ["配置", "系统分", "种子标准差", "95% 置信区间"],
      [["合成 + 重造数据", "0.867", "±0.002", "[0.810, 0.917]"],
       ["气体 + 逻辑回归", "0.783", "±0.000（确定性）", "[0.755, 0.813]"],
       ["气体 + 随机森林", "0.549", "±0.023", "[0.519, 0.580]"],
       ["轴承 + 逻辑回归", "0.477", "±0.000", "[0.420, 0.530]"]],
      M, 1.62, 12.0, [3.2, 1.8, 2.8, 2.6], size=12, row_h=0.5)

note(s, "逻辑回归是确定性的，换种子结果一模一样 —— 标准差为 0 不是 bug，是事实。",
     M, 3.85, 12.0, "blue")
note(s, "真正的不确定性来自批次抽样（自助法区间 ±0.03~0.05），而不是模型随机性。",
     M, 4.62, 12.0, "blue")
note(s, "所以以后引用数字应该写「0.78 [0.76, 0.81]」，而不是「0.783」。", M, 5.4, 12.0,
     "green")

# ═══ 11 三个反面结果 ═══
s = new_slide(prs)
head(s, "三个反面结果（我们主动做了对照实验）", "这些比“我们用了什么”更能说明问题")

cards(s, [("深度学习无收益", "MLP 在三个数据集上\n全部低于逻辑回归\n0.634 / 0.692 / 0.424", RED, RED_L),
          ("集成学习无效", "三模型软投票\nF1 低于最好的单模型\n错误高度重叠", ORANGE, ORANGE_L),
          ("局部指标 ≠ 全局", "加强正则让单批 +44%\n闭环里却掉 0.12", BLUE, BLUE_L)],
      1.62, 1.95)

tb = textbox(s, M, 3.95, 12.0, 0.4)
para(tb.text_frame, "为什么深度学习不行，一句话说清", 14, True, NAVY, first=True)
table(s, ["", "数值"],
      [["训练样本", "445 条"],
       ["MLP 参数", "10,534 个"],
       ["平均每个参数", "0.04 条样本"],
       ["在 60 条样本上", "训练集准确率就能到 100%"]],
      M, 4.4, 7.2, [3.6, 2.6], size=12, row_h=0.42)

note(s, "它记住的是那 60 条数据，不是规律。所以一换新批就掉到 0.58。", 8.3, 4.4, 4.4,
     "red", 11.5)

# ═══ 12 Harness ═══
s = new_slide(prs)
head(s, "Harness 部分：它到底是什么", "模型是员工，Harness 是经理")

table(s, ["组成", "作用", "一句话"],
      [["协议层", "交卷才给下一批", "结构上堵死“偷看未来”"],
       ["五环节", "监测→归因→决策→执行→门控", "每批走一遍"],
       ["三层决策", "规则 → 经验表 → 大模型", "任何一层挂了，下层顶上"],
       ["门控回滚", "新旧模型同批对比", "最坏情况是“什么都没变”"]],
      M, 1.62, 12.0, [1.8, 5.0, 5.2], size=12, row_h=0.5)

tb = textbox(s, M, 4.3, 12.0, 0.4)
para(tb.text_frame, "最重要的一点：我们故意让大模型不是必需的", 14, True, GREEN,
     first=True)

cards(s, [("13 个故障注入", "JSON 乱码 / 输出截断 /\n编造动作 / 服务端 500 /\n30 秒超时 / 端点不可达", GREEN, GREEN_L),
          ("结果", "分数一次没掉\n10 轮全跑完\n0 协议违规", GREEN, GREEN_L),
          ("反直觉的实验", "换成“永远说不动”的假模型\n分数正好掉回静态基线", ORANGE, ORANGE_L)],
      4.75, 1.8)

note(s, "结论：接入大模型本身没有价值，有价值的是模型的判断力。所以我们的设计是——它只能让系统更好，不能更差。",
     M, 6.75, 12.0, "blue", 12)

# ═══ 13 数据处理 ═══
s = new_slide(prs)
head(s, "数据处理部分：四件事", "从原始文件到一个能喂给闭环的批次流")

table(s, ["环节", "做了什么", "关键数字"],
      [["① 切批", "三个预处理脚本，把原始数据切成批次流", "气体 10 批 / 轴承 15 批 / C-MAPSS 10 批"],
       ["② 特征", "轴承：9216 个 CSV、每个 32768 点 → 16 个时域统计量", "11 GB → 1 MB"],
       ["③ 标准化", "standard（均值方差）/ quantile（本批内百分位）", "轴承跨设备 +52%"],
       ["④ 数据治理", "每个数据源带身份标识，用途不符直接拦", "防“用测试集调参”"]],
      M, 1.62, 12.0, [1.6, 6.0, 4.4], size=12, row_h=0.62)

tb = textbox(s, M, 4.88, 12.0, 0.4)
para(tb.text_frame, "第 ④ 条值得多说一句", 13.5, True, NAVY, first=True)
table(s, ["数据身份", "允许用途"],
      [["validation", "唯一允许用来选阈值"],
       ["official_public", "只能格式体检，不许进闭环"],
       ["official_test", "必须已冻结配置 + 一次性凭据"]],
      M, 5.3, 7.6, [2.6, 5.0], size=11.5, row_h=0.42)
note(s, "这不是约定，是代码里抛异常拦的。", 8.6, 5.3, 4.1, "green", 11.5)

# ═══ 14 已知局限 ═══
s = new_slide(prs)
head(s, "已知局限（主动交代）", "说清边界，比吹牛有用")

table(s, ["局限", "影响", "状态"],
      [["经验表没按模型分桶", "换模型后旧经验的经济账失效", "待改"],
       ["归因仍偏粗", "把“这批没测某类”误判成概念漂移", "待改"],
       ["成本是手写常数", "不是实测标定，数据规模变化时会失真", "待改"],
       ["只用了 FD001", "C-MAPSS 的单工况子集，漂移较弱", "待换 FD002"],
       ["大模型无额度", "框架就绪，但没接真 Qwen", "企业赛"]],
      M, 1.62, 12.0, [3.4, 6.6, 2.0], size=12, row_h=0.58)

note(s, "前三条是方法上的债，后两条是资源限制。我们不打算在汇报里藏起来。", M, 5.35, 12.0,
     "orange")

# ═══ 15 结尾 ═══
s = new_slide(prs)
rect(s, 0, 0, SW, SH, fill=RGBColor(0x0E, 0x1A, 0x38))
rect(s, 0, 0, 0.16, SH, fill=BLUE)

tb = textbox(s, 1.0, 1.5, 11.4, 0.5)
para(tb.text_frame, "一句话总结这段时间", 15, False, RGBColor(0x93, 0xB4, 0xE8),
     first=True)
tb = textbox(s, 1.0, 2.0, 11.4, 2.0)
tf = tb.text_frame
para(tf, "上次交代的缺陷，修了；", 26, True, WHITE, first=True, sa=10)
para(tf, "只有机制没有证据的地方，补了实测；", 26, True, WHITE, sa=10)
para(tf, "以为能用的东西，有一部分实测证明没用。", 26, True, WHITE)

rect(s, 1.0, 4.55, 4.0, 0.05, fill=BLUE)
tb = textbox(s, 1.0, 4.8, 11.4, 1.6)
tf = tb.text_frame
para(tf, "接下来想听你的意见：", 15, True, RGBColor(0xC3, 0xD3, 0xEA), first=True,
     sa=6)
para(tf, "① 归因这一块该做到什么程度  ② C-MAPSS 要不要换 FD002 验漂移  ③ 企业赛的优先级怎么排",
     14, False, RGBColor(0x93, 0xB4, 0xE8), line=1.3)

out = r"D:\人工智能+\docs\进展汇报-上次之后改了什么.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved:", out, "slides:", len(prs.slides))

# -*- coding: utf-8 -*-
"""生成《项目交接说明》DOCX（给队员看）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from docx_kit import (BLUE, GREEN, GREY, INK, INK2, RED, bullet, callout,
                      caption, code_block, data_table, heading, kv_table,
                      no_borders, para, set_width, setup, shade, style_run,
                      cell_margins)

OUT = r"D:\人工智能+\docs\项目交接说明.docx"
FIG = r"D:\人工智能+\assets\fig1_loop.png"

doc = Document()
setup(doc, "项目交接说明 · 智训引擎 · 2026-11-08")

# ── 封面 ────────────────────────────────────────────────────
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(80)
p.paragraph_format.space_after = Pt(6)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("「智训引擎」· 队内交接文档")
style_run(r, 10.5, False, GREY)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("我们现在有什么")
style_run(r, 38, True, BLUE)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(26)
r = p.add_run("以及，你明天该做什么")
style_run(r, 17, False, INK2)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(50)
r = p.add_run("2026 年 11 月 8 日")
style_run(r, 11, False, GREY)

meta = doc.add_table(rows=1, cols=1)
meta.alignment = 1
meta.autofit = False
no_borders(meta)
set_width(meta, 13.0)
c = meta.cell(0, 0)
c.width = Cm(13.0)
shade(c, "F1F5F9")
cell_margins(c, 200, 200, 260, 260)
c.paragraphs[0].text = ""
for i, (k, v) in enumerate([
    ("阅读时间", "15 分钟"),
    ("适合谁", "还没跟上进度的队员（先读这一份）"),
    ("读完你会知道", "现在有什么、怎么跑起来、你负责哪块"),
]):
    pp = c.paragraphs[0] if i == 0 else c.add_paragraph()
    pp.paragraph_format.space_after = Pt(4)
    rr = pp.add_run(f"{k}    ")
    style_run(rr, 10, True, GREY)
    style_run(pp.add_run(v), 10, False, INK)

doc.add_page_break()

# ── 一 ──────────────────────────────────────────────────────
heading(doc, "一、先说结论")

callout(doc, [
    ("我们已经有一个能自己跑的系统了，它已经在笔记本上跑出了结果。", {"bold": True, "size": 12, "color": BLUE}),
    ("你不需要看懂全部代码，但需要知道它在干什么、你负责哪一块。"
     "这份文档就是干这个的。", {"size": 10.5}),
], "key")

para(doc, "三句话概括现状：", space_before=4)
bullet(doc, "**代码写好了** —— 1200 多行，能跑通完整闭环")
bullet(doc, "**结果出来了** —— 静态基线一路掉到 0.50，我们的系统稳在 0.84")
bullet(doc, "**缺三样东西** —— SwanLab 记录、Docker 镜像、演示材料")

para(doc, "这三样东西，正好是你们三个人的活。见第五节。", space_before=4, bold=True)

# ── 二 ──────────────────────────────────────────────────────
heading(doc, "二、五分钟跑起来")

para(doc, "打开终端，进到项目目录，敲三条命令：")

code_block(doc, """
  cd D:\\人工智能+

  # 1. 造一条"会漂移"的数据流（10 个批次）
  D:\\anaconda\\python.exe scripts\\make_data.py

  # 2. 跑完整闭环，看结果
  D:\\anaconda\\python.exe scripts\\run_pipeline.py
""")

para(doc, "你应该看到这样的输出：")
code_block(doc, """
  数据集：synthetic_drift  批次：10  模型：logreg
    b02 | 基线 0.795 | 系统 0.795 | PSI 0.04 | stable    | do_nothing  | —
    b03 | 基线 0.629 | 系统 0.629 | PSI 0.11 | stable    | do_nothing  | —
    b04 | 基线 0.475 | 系统 0.976 | PSI 0.21 | covariate | refresh_... | ✔ 上线
    b05 | 基线 0.371 | 系统 0.978 | PSI 0.31 | covariate | refresh_... | ✔ 上线
    b06 | 基线 0.440 | 系统 0.334 | PSI 0.38 | covariate | refresh_... | ✕ 回滚
    b07 | 基线 0.448 | 系统 0.955 | PSI 0.34 | concept   | refit_model | ✔ 上线
    b08 | 基线 0.446 | 系统 0.951 | PSI 0.02 | stable    | do_nothing  | —
    ...
    平均分  基线 0.499  系统 0.837
""")

para(doc, "然后打开 `out\\curve.png`，你会看到两条线：**红线一路下滑，蓝线稳在高位**。")
para(doc, "**这张图就是我们全部工作的意义。** 后面所有事都是围绕它做的。")

callout(doc, [
    ("跑不起来怎么办？", {"bold": True}),
    ("最常见的原因：没用 Anaconda 那个 Python（自带 sklearn）。"
     "检查 `D:\\anaconda\\python.exe -V` 能不能打出版本号。", {"size": 10, "color": INK2}),
], "amber")

# ── 三 ──────────────────────────────────────────────────────
heading(doc, "三、它到底在干什么")

para(doc, "一句话：**它盯着模型的表现，决定要不要动它，动了之后自己验收，不行就退回去。**")

doc.add_picture(FIG, width=Cm(16.2))
_p = doc.paragraphs[-1]
_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_p.paragraph_format.space_before = Pt(6)
_p.paragraph_format.space_after = Pt(0)
_p.paragraph_format.keep_with_next = True
caption(doc, "五个环节，只有 ④⑤ 会碰模型，①②③ 只看数据")

para(doc, "举个真实发生的例子，第 4 批数据进来时它干了什么：")

kv_table(doc, [
    ("① 发现", "数据分布偏了（PSI = 0.21，超过阈值 0.10）"),
    ("② 判断", "是“数据偏了”，不是“判定规则变了”"),
    ("③ 决定", "用最便宜的办法：重算标准化（花 0.05 秒，几乎免费）"),
    ("④ 验证", "同一批数据上，F1 从 0.475 涨到 0.976"),
    ("⑤ 上线", "通过门控，替换当前模型"),
], widths=(2.6, 13.6), size=10)

para(doc, "**这五步以前是人干的，现在是代码干的。** 这就是“自进化系统”。")

callout(doc, [
    ("关键点：它先试便宜的，不行才上贵的。", {"bold": True, "color": GREEN}),
    ("第 6 批判定标准变了，便宜办法失效 → 它自动回滚 → 第 7 批升级为重训 → "
     "0.955 上线。整个过程没有人插手。", {"size": 10, "color": INK2}),
], "ok")

# ── 四 ──────────────────────────────────────────────────────
heading(doc, "四、代码在哪里，谁管哪块")

para(doc, "整个项目按“三条线”切开，**各改各的文件，互不打架**：")

data_table(doc,
           ["目录", "管什么", "谁碰"],
           [["evolver/adapters/", "数据 / 模型 / 指标 三个适配器", "A"],
            ["evolver/loop/", "监测 → 归因 → 决策 → 评估 → 回滚", "A + B"],
            ["evolver/engine/", "决策大脑（规则 + 大模型建议）", "B"],
            ["evolver/ledger/", "预算账本（算力/标注/Token）", "C"],
            ["scripts/", "跑数据、跑流程、画图", "C"]],
           [4.4, 8.6, 3.2], size=9.5)

callout(doc, [
    ("一句话记住：改策略去 loop/，改模型去 adapters/，别互相踩。", {"bold": True, "color": BLUE}),
], "plain")

para(doc, "换数据集只需要改一个文件：`adapters/dataset.py` 里加一个类。"
          "闭环一行都不用动——这就是当初把接口拆出来的原因。")

# ── 五 ──────────────────────────────────────────────────────
heading(doc, "五、你明天该做什么（最重要的一节）")

para(doc, "技术部分基本齐了，缺的三样东西正好一人一件。**都不难，但都不做不行。**")

kv_table(doc, [
    ("A · SwanLab",
     "注册 → pip install swanlab → 在 run_pipeline.py 里加三行 → 跑一遍 → 网页上能看到曲线\n"
     "完成标志：SwanLab 项目页里有一条完整的 run\n耗时：半天\n为什么急：门槛② 卡这个，没记录直接不排名"),
    ("B · Docker",
     "装 WSL2 → 装 Docker → 写 Dockerfile → 打包 → 断网跑一遍\n"
     "完成标志：关掉网络，一条 docker run 跑完并输出四个文件\n耗时：半天到一天\n为什么急：这是唯一会卡死交付的环境问题，越早越好"),
    ("C · 材料",
     "把 out/curve.png 和 decisions.jsonl 做成 3 张卡片 + 2 分钟视频 + 一页说明\n"
     "完成标志：找个没参与的人看完，能复述我们在做什么\n耗时：一天\n为什么急：校赛评委只看这个"),
], widths=(3.2, 13.0), size=9.5)

callout(doc, [
    ("三个人今天就要做同一件事：各自在自己电脑上把第二节那三条命令跑一遍。", {"bold": True, "color": RED}),
    ("三个人都能打出那张图，这个项目才算真的开始了。在那之前，分工都是空的。",
     {"size": 10, "color": INK2}),
], "warn")

# ── 六 ──────────────────────────────────────────────────────
heading(doc, "六、十个词（看不懂就回这里查）")

data_table(doc,
           ["词", "一句话解释"],
           [["模型", "被训练、被更新的那堆参数。会变，但换个数据集就用不了"],
            ["闭环", "我们写的代码：监测→归因→决策→评估→回滚。结构固定，不训练"],
            ["技能（Skill）", "把处置经验写成可复用的说明书，跨模型通用"],
            ["漂移", "数据分布随时间变化，导致原来训练的模型失准"],
            ["协变量漂移", "输入变了，但“什么算什么”的规则没变（如光照变化）"],
            ["概念漂移", "输入和标签的关系变了（如验收标准改了），必须重训"],
            ["PSI", "一种衡量分布偏移程度的数字。越大越偏，0.1 小、0.25 明显"],
            ["静态基线", "官方设定的对照模型：部署后永不更新。五项门槛之一的对标线"],
            ["门控", "候选模型必须比当前好一定幅度，才允许上线；否则自动回滚"],
            ["Harness", "套在大模型外面的脚手架：工具 + 状态 + 策略 + 执行机制"]],
           [3.4, 12.8], size=9.5)

# ── 七 ──────────────────────────────────────────────────────
heading(doc, "七、你一定会问的六个问题")

kv_table(doc, [
    ("我需要会机器学习吗？",
     "不需要很会。你需要理解“模型会随时间变差、所以要有人管”这个逻辑。"
     "具体的训练代码已经写好了，你改的是流程和参数。"),
    ("为什么不用大模型？",
     "校赛阶段拿不到额度（PyTRiO 只给企业赛），你也跑不动 27B。"
     "现在的决策是规则做的，反而更稳、更省钱。接口已经留好，额度一到改两行就能接。"),
    ("校赛到底要交什么？",
     "学校自己定，通常是 PPT + 演示 + 报告。因为 11/13 就截止，而评测平台 11 月初才上线，"
     "所以校赛比的是“讲得清楚 + 有东西能跑”，不是分数。"),
    ("我要改哪里？",
     "看第四节。改策略去 loop/，改模型去 adapters/。两边尽量不要同时改同一个文件。"),
    ("官方数据还没发怎么办？",
     "先用合成数据。这一点都不将就——我们的合成流同时包含协变量漂移和概念漂移，"
     "和真实场景同构。官方数据一到，写个 adapter 就换过去。"),
    ("我改坏了怎么办？",
     "仓库是 git，随时可以回到上一个版本：git checkout . 就恢复了。"
     "另外系统本身有回滚机制，就算策略写错，门控也会拦住坏模型。"),
], widths=(3.6, 12.6), size=9.5)

# ── 八 ──────────────────────────────────────────────────────
heading(doc, "八、接下来的时间")

data_table(doc,
           ["日期", "要发生的事"],
           [["11/08（今天）", "三人各自跑通第二节的三条命令"],
            ["11/09 — 11/10", "SwanLab 打通、Docker 打通"],
            ["11/11 — 11/12", "材料定稿：曲线、决策卡、演示视频"],
            ["**11/13（周五）**", "**校赛推荐截止**"],
            ["11/15 — 12/05", "企业赛：接官方数据、接真模型、跑客观分"],
            ["12/26", "总决赛"]],
           [4.0, 12.2], size=9.5)

doc.add_paragraph().paragraph_format.space_after = Pt(8)
callout(doc, [
    ("最后一句，也是最重要的一句。", {"bold": True, "size": 11, "color": BLUE}),
    ("这道题看起来像机器学习题，本质是软件工程题。"
     "它考的不是“你能训出多准的模型”，而是“你能不能判断什么时候该动、什么时候不该动”。",
     {"size": 10.5, "color": INK2}),
    ("想清楚这一点，剩下的都是编码和打包。", {"bold": True, "size": 10.5, "color": INK}),
], "key")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("saved:", OUT)

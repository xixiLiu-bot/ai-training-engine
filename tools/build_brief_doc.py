# -*- coding: utf-8 -*-
"""生成《项目技术简报》DOCX（给技术顾问/研究生学长）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from docx_kit import (BLUE, GREEN, GREY, INK, INK2, RED, bullet, callout,
                      caption, cell_margins, code_block, data_table, heading,
                      kv_table, no_borders, para, set_width, setup, shade,
                      style_run)

OUT = r"D:\人工智能+\docs\项目技术简报.docx"
FIG = r"D:\人工智能+\assets\fig1_loop.png"

doc = Document()
setup(doc, "项目技术简报 · 智训引擎 · 2026-11-08")

# ── 封面 ────────────────────────────────────────────────────
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(80)
p.paragraph_format.space_after = Pt(6)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
style_run(p.add_run("「智训引擎」赛题 · 技术简报"), 10.5, False, GREY)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
style_run(p.add_run("我们做到哪了，卡在哪"), 34, True, BLUE)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(26)
style_run(p.add_run("请学长指路用 · 不是科普材料"), 15, False, INK2)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(46)
style_run(p.add_run("2026 年 11 月 8 日　·　距校赛截止 5 天"), 11, False, GREY)

meta = doc.add_table(rows=1, cols=1)
meta.alignment = 1
meta.autofit = False
no_borders(meta)
set_width(meta, 13.6)
c = meta.cell(0, 0)
c.width = Cm(13.6)
shade(c, "F1F5F9")
cell_margins(c, 200, 200, 260, 260)
c.paragraphs[0].text = ""
for i, (k, v) in enumerate([
    ("我们是谁", "三名本科生，零基础起步，第 5 天"),
    ("已有", "一套能跑通的闭环 + 合成漂移流 + 完整决策留痕"),
    ("缺的", "统计严谨性、真实数据集验证、工程交付"),
    ("最需要你帮忙的", "第八节那五件事，按优先级排好了"),
]):
    pp = c.paragraphs[0] if i == 0 else c.add_paragraph()
    pp.paragraph_format.space_after = Pt(4)
    style_run(pp.add_run(f"{k}　"), 10, True, GREY)
    style_run(pp.add_run(v), 10, False, INK)

doc.add_page_break()

# ── 一 ──────────────────────────────────────────────────────
heading(doc, "一、一页速览")

kv_table(doc, [
    ("赛题", "工业模型自进化系统：在漂移数据流上，无人干预地自主维持并提升模型性能"),
    ("交付", "Docker 镜像 + Harness 配置 + 源码 + SwanLab 记录 + 论证文档 + 合规声明"),
    ("评测", "4 核 8G Linux，无 GPU；提交包 ≤ 2G；禁止外部 API；批次下发、交卷换下一批"),
    ("路线", "B（Harness 优化为主）：不拼模型精度，拼“决策是否聪明、成本是否低”"),
    ("现在", "闭环跑通，合成流上基线 0.499 / 系统 0.837；协议合规；6 项回归测试通过"),
    ("缺", "归因是启发式、门控无统计检验、多任务未覆盖、Docker 与 SwanLab 未接"),
], widths=(3.0, 13.2), size=9.5)

callout(doc, [
    ("一句话说明我们的判断：这道题考的是“运维系统”，不是“模型能力”。", {"bold": True, "size": 12, "color": BLUE}),
    ("如果这个判断错了，后面所有设计都白做——所以这是我们最想请你先看一眼的地方。",
     {"size": 10, "color": INK2}),
], "key")

# ── 二 ──────────────────────────────────────────────────────
heading(doc, "二、赛题的硬约束")

para(doc, "这些是说明会上明确的口径，直接决定架构，不是软要求。")

data_table(doc,
           ["约束", "内容", "对我们的影响"],
           [["算力", "4 核 8G Linux，无 GPU", "sklearn 级模型；任何深度学习/Embedding 都排除"],
            ["体积", "提交包 ≤ 2G，严禁打包模型权重", "冷启动必须现场训练，训练成本进分母"],
            ["网络", "禁止外部 API，仅可用平台模型服务", "LLM 只能接 Qwen3.8-27B / Qwen3.5-4B"],
            ["数据", "批次下发，**先交卷才给下一批**", "在线决策，结构上无法偷看未来"],
            ["标签", "评测时每批都有标签，不耗配额", "归因可以做真；但过拟合成为评分项"],
            ["门槛", "含 Harness、≥3 次迭代、不低于静态基线", "缺一不进排名"],
            ["记录", "SwanLab 必须记录且**设为公开**", "不公开评委打不开，直接失分"]],
           [2.4, 7.4, 6.4], size=9)

callout(doc, [
    ("最反直觉的一条是“先交卷才给下一批”。", {"bold": True, "color": RED}),
    ("它不是交互设计，是**防过拟合的结构性手段**——你看不到未来，就没法为测试流调参。"
     "评分里“是否过拟合”那一项就是靠这个兜底的。", {"size": 10, "color": INK2}),
], "warn")

# ── 三 ──────────────────────────────────────────────────────
heading(doc, "三、路线选择：为什么是 B")

data_table(doc,
           ["路线", "做法", "要求", "我们的判断"],
           [["A 模型为主", "闭环跑起来，重点升级工业模型", "一个训得好的模型", "5 天做不出来，且评分权重不在模型"],
            ["**B Harness 为主**", "模型用合规基模，重点做策略与决策", "够用的基线 + 扎实的闭环", "**选了这条**"],
            ["C 联合", "模型层 + 策略层协同", "两头都要强", "企业赛阶段再升级"]],
           [3.0, 5.2, 4.2, 5.4], size=9)

para(doc, "理由有三条：")
bullet(doc, "客观分是「增益 ÷ 成本」，**成本完全由决策决定**，模型精度只是分子的一部分")
bullet(doc, "官方口径反复强调“系统是否长期可用”，而不是“某一次模型的分数”")
bullet(doc, "5 天里我们唯一有可能做好的是决策层，不是建模")

# ── 四 ──────────────────────────────────────────────────────
heading(doc, "四、系统结构")

doc.add_picture(FIG, width=Cm(15.2))
_p = doc.paragraphs[-1]
_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_p.paragraph_format.space_before = Pt(4)
_p.paragraph_format.space_after = Pt(0)
_p.paragraph_format.keep_with_next = True
caption(doc, "闭环五环节：①②③ 不读模型内部，只有 ④⑤ 碰权重")

para(doc, "代码按职责切成五层，层间只通过接口通信：")

data_table(doc,
           ["层", "职责", "关键文件", "可替换点"],
           [["protocol/", "评测协议：批次下发、交卷、禁止偷看未来", "batch_stream.py", "换 RemoteEvaluator 即接官方接口"],
            ["adapters/", "数据 / 模型 / 指标 三个适配器", "dataset.py, model.py, metric.py", "换数据集只改 dataset.py"],
            ["loop/", "监测、归因、决策、评估、回滚", "monitor / diagnose / decide / evaluate / release", "策略实验的主战场"],
            ["engine/", "决策大脑：规则 + 大模型建议", "policy.py, llm_agent.py", "LLM 后端可切换，失败降级"],
            ["ledger/", "预算账本：算力 / 标注 / Token", "ledger.py", "超预算抛异常，不进决策"]],
           [2.6, 6.2, 4.6, 4.2], size=8.5)

# ── 五 ──────────────────────────────────────────────────────
heading(doc, "五、当前实现与实测")

para(doc, "合成漂移流（10 批）：前 5 批协变量漂移，第 6 批起判定标准整体改变。")

data_table(doc,
           ["指标", "静态基线", "我们的系统"],
           [["平均 F1（macro）", "0.499", "**0.837**"],
            ["末批 F1", "0.446", "0.956"],
            ["决策次数", "—", "4 次（2 校准 + 1 重训 + 1 失败回滚）"],
            ["累计成本", "—", "2.15 秒 / 2000 条标注"],
            ["协议合规", "—", "10 轮 | 部署 4 | 维持 6 | 违规 0"]],
           [5.4, 4.4, 6.4], size=9.5)

para(doc, "完整跑一遍：")
code_block(doc, """
  D:\\anaconda\\python.exe scripts\\make_data.py
  D:\\anaconda\\python.exe scripts\\run_pipeline.py
  D:\\anaconda\\python.exe -m unittest tests.test_protocol -v
""")

# ── 六 ──────────────────────────────────────────────────────
heading(doc, "六、几个关键设计决策（欢迎挑刺）")

kv_table(doc, [
    ("先规则，后 LLM",
     "决策层永远是规则引擎先算一遍，LLM 只提建议，过四道闸才采纳；LLM 挂了直接降级。\n"
     "**可质疑**：四道闸是拍出来的，尤其“不许重复无效便宜动作”那条，阈值 0.02 没有依据。"),
    ("成本阶梯：先便宜后贵",
     "三级动作：不动 / 重算标准化（≈0.05s）/ 重训（≈2s）。默认先试最便宜的。\n"
     "**可质疑**：秒数是编的常数，不是真实测量；也没有内存/IO 成本。"),
    ("门控用“同批对比”",
     "候选与当前模型在同一批上打分，增益 ≥ 0.02 才上线，否则保留旧模型（= 自动回滚）。\n"
     "**可质疑**：单批 2000 样本，0.02 的差值可能纯属噪声，**没有做任何显著性检验**。"),
    ("重训只用 b-1 及更早的数据",
     "用第 b 批训练、又在第 b 批验收 = 数据泄漏，分数会虚高到 1.0。\n"
     "**可质疑**：样本权重是指数衰减（tau=0.5），衰减率同样是拍的。"),
    ("模型换代后重置漂移参照系",
     "否则参照系永远钉在部署那天，PSI 虚高，系统会反复做无用更新。\n"
     "**可质疑**：重置时机与旧数据保留策略，没有理论依据。"),
    ("协议层结构性强约束",
     "不是“约定不要偷看未来”，而是 `LocalEvaluator` 直接抛异常；有回归测试证明只加载已下发批次。"),
], widths=(3.8, 12.4), size=9)

# ── 七 ──────────────────────────────────────────────────────
heading(doc, "七、已知局限（我们自己不满意的地方）")

data_table(doc,
           ["问题", "现状", "严重度"],
           [["归因基本是启发式",
             "只用 PSI + 标签边缘分布。**判不出“局部区域规则改变”这类概念漂移**——"
             "因为标签边缘不变。实际靠“试了没用再升级”反推，不是真识别", "高"],
            ["门控没有统计检验",
             "0.02 阈值是拍的；单批对比噪声大，可能把噪声当增益上线，也可能错过真增益", "高"],
            ["成本模型是伪的",
             "`cost_sec` 是硬编码常数，与实际 CPU 时间无关；评分要算“单位增益消耗”，我们这块是空的", "高"],
            ["只有分类器",
             "A 包三个数据集里，发动机 RUL 是回归任务，`model.py` 里没有回归器", "中"],
            ["只验证过合成数据",
             "气体传感器 adapter 写了但没跑通；官方三个数据集都没实测", "中"],
            ["LLM 层只跑过 mock",
             "链路与降级验证过了，但没有接过真模型（校赛不发额度）", "低"],
            ["Docker / SwanLab 未接",
             "交付物缺两样，正在分工做", "中"]],
           [4.4, 10.6, 1.6], size=9, align_center=2)

# ── 八 ──────────────────────────────────────────────────────
heading(doc, "八、最想请你帮忙的五件事")

para(doc, "按“我们能自己解决吗”排序，前面的是我们确实做不了的。")

kv_table(doc, [
    ("① 归因：怎么在没有/少标签下判别漂移类型",
     "现状是 PSI + 实验反推。想知道学界有没有更可靠的做法（残差分析？分区域性能代理？"
     "对抗验证？），以及在我们这种“批次只有 2000 条”的规模下是否可行。"),
    ("② 门控的统计严谨性",
     "同批对比 + 固定阈值，等于没做检验。要不要上 bootstrap 置信区间 / 配对检验？"
     "样本量要多少才有统计意义？**这条直接关系到“不低于静态基线”这条门槛稳不稳**。"),
    ("③ 怎么证明“不是过拟合”",
     "B 包是同领域全新数据集，评分明确要查过拟合。我们的策略实验该怎么设计，"
     "才能既调好又不至于对 B 包过拟合？有没有可量化的“泛化裕度”指标？"),
    ("④ 多任务统一",
     "A 包三个任务：气体传感器（分类）、轴承退化（分类）、发动机 RUL（回归）。"
     "是写三套 adapter 各跑各的，还是抽一套统一 pipeline？后者更漂亮但风险更高。"),
    ("⑤ 成本模型怎么建",
     "客观分是“单位增益 ÷ 成本”。我们现在的成本是编的。"
     "在 4 核 8G 无 GPU 的环境里，哪些量该记、怎么记才既准又便宜？"),
], widths=(5.0, 11.2), size=9)

callout(doc, [
    ("如果时间只够你帮一件事，请帮第 ①② 条。", {"bold": True, "color": RED}),
    ("归因不准 + 门控不严，会同时让我们在“客观分”和“安全可控”两个维度上失分，"
     "而且这两处错在外行看不出来，只有懂的人能发现。", {"size": 10, "color": INK2}),
], "warn")

# ── 九 ──────────────────────────────────────────────────────
heading(doc, "九、时间线与分工")

data_table(doc,
           ["时间", "事", "状态"],
           [["11/08（今天）", "闭环跑通、协议合规、仓库上线", "✅"],
            ["11/09", "官方评测平台上线，SwanLab 规范发布", "待发布"],
            ["11/09 — 11/12", "SwanLab 接入、Docker 打包、材料定稿", "进行中"],
            ["**11/13（周五）**", "**校赛推荐截止**", "倒计时 5 天"],
            ["11/15 — 12/05", "企业赛：接 A/B 包、接真模型、跑客观分", ""],
            ["12/26", "总决赛", ""]],
           [4.2, 8.4, 3.6], size=9.5)

para(doc, "三个人当前分工：A 负责策略层（监测+归因+决策），B 负责模型层（手段池+成本测量），"
          "C 负责工程层（协议、Docker、SwanLab、材料）。")

# ── 十 ──────────────────────────────────────────────────────
heading(doc, "十、五分钟上手")

code_block(doc, """
  git clone https://gitee.com/gusliuy0y0/human--ai-training-engine.git
  cd human--ai-training-engine

  D:\\anaconda\\python.exe scripts\\make_data.py      # 造漂移流
  D:\\anaconda\\python.exe scripts\\run_pipeline.py   # 跑闭环，出 out\\curve.png
""")

para(doc, "仓库里还有三份文档：赛题说明（概念）、五天冲刺（排期）、项目交接（给队员）。"
          "这份是第四份，专门给你看的——前面那些不用读。")

doc.add_paragraph().paragraph_format.space_after = Pt(8)
callout(doc, [
    ("最后说句实话：我们知道这套东西还很不成熟。", {"bold": True, "size": 11, "color": BLUE}),
    ("闭环跑通了、留痕齐全了，但统计上是糙的、数据集上只验证了合成流。"
     "如果学长看完觉得“方向就错了”，请直接说——现在改还来得及，11/13 之后就来不及了。",
     {"size": 10, "color": INK2}),
], "key")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("saved:", OUT)

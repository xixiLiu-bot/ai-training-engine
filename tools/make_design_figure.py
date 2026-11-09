# -*- coding: utf-8 -*-
"""生成程序结构图：主流程 + 扩展点（重点是“神经网络插在哪”）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_figures import Canvas, F, arrow

W, H = 1400, 900

BLUE = "#2563EB"
BLUE_BG = "#EFF6FF"
ORANGE = "#EA580C"
ORANGE_BG = "#FFF7ED"
GRAY = "#64748B"
GRAY_BG = "#F1F5F9"
GREEN = "#059669"
GREEN_BG = "#ECFDF5"
VIOLET = "#7C3AED"
VIOLET_BG = "#F5F3FF"
INK = "#1F2937"
SUB = "#4B5563"


def main():
    c = Canvas(W, H)

    L0, L1 = 45, 890
    cx = (L0 + L1) / 2

    # ── 协议（最上面）──
    c.rect((250, 26, 680, 92), radius=12, fill=GRAY_BG, outline="#CBD5E1", width=2)
    c.put((250, 26, 680, 60), "评测器：一问一答", F(21, True), INK)
    c.put((250, 58, 680, 92), "next_batch() → 处理 → submit() → 才给下一批", F(17), SUB)
    arrow(c, (cx, 92), (cx, 118), GREEN, 3, 12)

    # ── 五个环节（显式给 y，避免叠在一起）──
    steps = [
        (124, 84, "① 漂移监测", "loop/monitor.py",
         "当前批 vs 参照系，128 个特征各算一个 PSI，取平均", BLUE, BLUE_BG),
        (242, 84, "② 归因诊断", "loop/diagnose.py",
         "PSI 高 + 标签分布也变 → 概念漂移；否则 → 协变量漂移", BLUE, BLUE_BG),
        (360, 84, "③ 决策", "loop/decide.py",
         "6 条规则，从便宜到贵挑一档；LLM 可插嘴，但要过四道闸", BLUE, BLUE_BG),
        (478, 124, "④ 执行 + 影子评估", "loop/evaluate.py", None, ORANGE, ORANGE_BG),
        (636, 84, "⑤ 门控 + 回滚", "loop/release.py",
         "候选比当前高 ≥ 0.02 才上线，否则保留旧模型（自动回滚）", ORANGE, ORANGE_BG),
    ]
    for y0, hh, title, f, sub, col, bg in steps:
        c.rect((L0, y0, L1, y0 + hh), radius=12, fill=bg, outline=col, width=2)
        c.text((L0 + 26, y0 + 13), f"{title}    {f}", F(23, True), col)
        if sub:
            c.text((L0 + 26, y0 + 46), sub, F(18), SUB)

    for y0, hh in ((208, 34), (326, 34), (444, 34), (602, 34)):
        arrow(c, (cx, y0), (cx, y0 + 30), BLUE, 3, 11)

    # ④ 里的三个动作档位
    by0 = 478 + 58
    chips = [("什么都不做", "0 秒", "#94A3B8", "#F8FAFC"),
             ("重算标准化", "0.05 秒", GREEN, GREEN_BG),
             ("重训模型", "2 秒", ORANGE, ORANGE_BG)]
    cw, ch, cgap = 228, 48, 16
    x = L0 + 26
    for label, cost, col, bg in chips:
        c.rect((x, by0, x + cw, by0 + ch), radius=9, fill=bg, outline=col, width=2)
        c.text((x + 14, by0 + 13), label, F(18, True), col)
        w = c.measure(cost, F(16))[0]
        c.text((x + cw - w - 14, by0 + 15), cost, F(16), col)
        x += cw + cgap

    # ── 交卷 ──
    arrow(c, (cx, 720), (cx, 748), GREEN, 3, 12)
    c.rect((255, 750, 675, 816), radius=12, fill=GREEN_BG, outline=GREEN, width=2)
    c.put((255, 750, 675, 784), "交卷 submit()", F(21, True), GREEN)
    c.put((255, 782, 675, 816), "deploy（部署）/ keep（维持）+ 本批预测", F(17), SUB)

    # ── 右侧：扩展点 ──
    RX0, RX1 = 930, 1360
    c.text((RX0, 6), "以后要改，就改这几个地方", F(21, True), VIOLET)

    cards = [
        ("改这里 →  ①", "换数据集", "adapters/dataset.py",
         "写一个类，实现 n_batches / get()。其余代码一行不动",
         BLUE, BLUE_BG),
        ("改这里 →  ④", "换模型 / 加神经网络", "adapters/model.py",
         "神经网络就插在这里。实现 fit / predict / refresh 就行",
         VIOLET, VIOLET_BG),
        ("改这里 →  ③", "加一档新动作", "loop/decide.py 的 COST 表",
         "加一行「动作名: 成本」，决策器自动就能选它",
         GREEN, GREEN_BG),
        ("改这里 →  ③", "接大模型", "engine/policy.py",
         "LLM 提建议 → 四道闸 → 采纳或退回规则",
         "#0891B2", "#ECFEFF"),
    ]
    cy = 40
    for tag, title, file, desc, col, bg in cards:
        c.rect((RX0, cy, RX1, cy + 150), radius=12, fill=bg, outline=col, width=2)
        c.text((RX0 + 18, cy + 10), tag, F(15, True), "#94A3B8")
        c.text((RX0 + 18, cy + 32), title, F(21, True), col)
        c.text((RX0 + 18, cy + 62), file, F(16), col)
        # 描述按宽度手工折行
        words = desc
        c.text((RX0 + 18, cy + 92), words[:26], F(16), SUB)
        if len(words) > 26:
            c.text((RX0 + 18, cy + 116), words[26:], F(16), SUB)
        cy += 166

    # ── 底部结论 ──
    c.rect((45, 830, 1360, 886), radius=12, fill="#FFFBEB",
           outline="#D97706", width=2)
    c.text((68, 840),
           "神经网络只能落在 adapters/model.py 这一层 —— 五个环节、协议层、门控回滚全都不用改。",
           F(19, True), "#B45309")
    c.text((68, 862), "这就是当初把接口拆出来的原因：换模型不该牵动流程。",
           F(17), SUB)

    out = r"D:\人工智能+\assets\fig3_design.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    c.save(out)
    print("saved:", out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""生成赛题说明文档所需的两张示意图（统一超采样，避免坐标错位）。"""
import math
import os
from PIL import Image, ImageDraw, ImageFont

FONT_R = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
SS = 3  # 超采样倍数


class Canvas:
    """所有坐标都用设计坐标（逻辑像素），内部自动乘以 SS。"""

    def __init__(self, w, h, bg="white"):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SS, h * SS), bg)
        self.d = ImageDraw.Draw(self.img)

    def _p(self, pt):
        return (pt[0] * SS, pt[1] * SS)

    def line(self, pts, fill, width=1):
        self.d.line([self._p(p) for p in pts], fill=fill, width=width * SS, joint="curve")

    def rect(self, box, radius=0, fill=None, outline=None, width=1):
        self.d.rounded_rectangle([self._p(box[:2]), self._p(box[2:])],
                                 radius=radius * SS, fill=fill,
                                 outline=outline, width=max(1, width * SS))

    def poly(self, pts, fill):
        self.d.polygon([self._p(p) for p in pts], fill=fill)

    def text(self, xy, txt, f, fill):
        self.d.text(self._p(xy), txt, font=f, fill=fill)

    def measure(self, txt, f):
        b = self.d.textbbox((0, 0), txt, font=f)
        return (b[2] - b[0]) / SS, (b[3] - b[1]) / SS, b[0] / SS, b[1] / SS

    def put(self, box, txt, f, fill, dy=0):
        w, h, ox, oy = self.measure(txt, f)
        cx = (box[0] + box[2]) / 2 - w / 2 - ox
        cy = (box[1] + box[3]) / 2 - h / 2 - oy + dy
        self.text((cx, cy), txt, f, fill)

    def save(self, path):
        self.img.resize((self.w, self.h), Image.LANCZOS).save(path, "PNG")


def F(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, int(size * SS))


def arrow(c: Canvas, p1, p2, color, width=3, head=12):
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    hx, hy = p2[0] - head * math.cos(ang), p2[1] - head * math.sin(ang)
    c.line([p1, (hx, hy)], color, width)
    a = math.pi / 7
    c.poly([
        p2,
        (p2[0] - head * 1.5 * math.cos(ang - a), p2[1] - head * 1.5 * math.sin(ang - a)),
        (p2[0] - head * 1.5 * math.cos(ang + a), p2[1] - head * 1.5 * math.sin(ang + a)),
    ], color)


# ─────────────────────────────────────────────────────────────
# 图 1：闭环五个环节
# ─────────────────────────────────────────────────────────────
def fig_loop(path):
    W, H = 1140, 470
    c = Canvas(W, H)

    BLUE, ORANGE, GREEN, RED = "#2563EB", "#EA580C", "#059669", "#DC2626"

    items = [
        ("① 漂移监测", BLUE), ("② 归因诊断", BLUE), ("③ 决 策", BLUE),
        ("④ 执行 + 评估", ORANGE), ("⑤ 上线 / 回滚", ORANGE),
    ]
    n = len(items)
    mx, gap, bh, by = 58, 26, 100, 46
    bw = (W - 2 * mx - gap * (n - 1)) / n
    boxes = []
    for i, (label, color) in enumerate(items):
        x0 = mx + i * (bw + gap)
        boxes.append((x0, by, x0 + bw, by + bh))
        c.rect(boxes[-1], radius=13, fill=color)
        f = F(24, True)
        while c.measure(label, f)[0] > bw - 12:
            f = F(int(f.size / SS) - 1, True)
        c.put(boxes[-1], label, f, "white")
        if i:
            arrow(c, (boxes[i - 1][2] + 2, by + bh / 2), (x0 - 3, by + bh / 2), "#9CA3AF", 3, 10)

    cx = [(b[0] + b[2]) / 2 for b in boxes]

    # ①②③ 不碰模型
    lx0, lx1, ly = boxes[0][0] + 8, boxes[2][2] - 8, 172
    c.line([(lx0, ly), (lx1, ly)], BLUE, 2)
    c.line([(lx0, ly), (lx0, ly + 8)], BLUE, 2)
    c.line([(lx1, ly), (lx1, ly + 8)], BLUE, 2)
    note = "这三步全程不读模型内部，只看数据分布和分数"
    f = F(21)
    nw = c.measure(note, f)[0]
    c.text(((lx0 + lx1) / 2 - nw / 2, ly + 11), note, f, BLUE)

    # 模型框
    mbox = (boxes[3][0] - 8, 268, boxes[4][2] + 8, 358)
    c.rect(mbox, radius=13, fill="#FEF2F2", outline=RED, width=3)
    c.put((mbox[0], mbox[1] + 6, mbox[2], mbox[1] + 56), "工业模型（权重）", F(26, True), "#B91C1C")
    c.put((mbox[0], mbox[1] + 50, mbox[2], mbox[3] - 8),
          "整个系统唯一被读写的对象", F(18), RED)

    for i in (3, 4):
        arrow(c, (cx[i], 268), (cx[i], boxes[i][3] + 4), RED, 3, 12)

    # 闭环回路
    rx, ly2 = W - 22, 404
    y0 = by + bh / 2
    c.line([(boxes[4][2] + 3, y0), (rx, y0)], GREEN, 3)
    c.line([(rx, y0), (rx, ly2)], GREEN, 3)
    c.line([(rx, ly2), (28, ly2)], GREEN, 3)
    c.line([(28, ly2), (28, y0)], GREEN, 3)
    arrow(c, (28, y0), (boxes[0][0] - 4, y0), GREEN, 3, 12)

    cap = "每来一批新数据就走一遍 —— 循环往复，这就是“闭环”"
    f2 = F(21, True)
    cw = c.measure(cap, f2)[0]
    c.text(((boxes[0][0] + boxes[4][2]) / 2 - cw / 2, ly2 + 12), cap, f2, GREEN)

    c.save(path)


# ─────────────────────────────────────────────────────────────
# 图 2：三层结构
# ─────────────────────────────────────────────────────────────
def fig_layers(path):
    W, H = 1140, 562
    c = Canvas(W, H)

    layers = [
        ("第 3 层　技能（Skill）",
         "√ 跨模型通用　　√ 写完不碰权重　　√ 越攒越多 = 真正的“进化”",
         "#7C3AED", "#F5F3FF"),
        ("第 2 层　闭环（系统 / Harness）",
         "· 监测 → 归因 → 决策 → 评估 → 回滚　　· 结构固定，不训练，只调试",
         "#2563EB", "#EFF6FF"),
        ("第 1 层　模型（权重）",
         "× 换数据集必须重训　　× 权重不可迁移　　× 会被漂移拖垮",
         "#DC2626", "#FEF2F2"),
    ]
    bx0, bx1 = 40, 1100
    top, bh, gap = 28, 132, 56
    boxes = []
    for i, (title, desc, color, fill_light) in enumerate(layers):
        y0 = top + i * (bh + gap)
        boxes.append((bx0, y0, bx1, y0 + bh))
        c.rect(boxes[-1], radius=15, fill=fill_light, outline=color, width=3)
        c.text((bx0 + 34, y0 + 16), title, F(30, True), color)
        c.text((bx0 + 34, y0 + 74), desc, F(21), "#4B5563")

    for i in range(2):
        ax = (bx0 + bx1) / 2
        arrow(c, (ax, boxes[i][3] + 5), (ax, boxes[i + 1][1] - 5), "#6B7280", 3, 12)

    for txt, bx in (("决定“怎么做”", boxes[0]), ("操作 / 更新", boxes[1])):
        f = F(19)
        c.text(((bx[0] + bx[2]) / 2 + 22, bx[3] + 17), txt, f, "#6B7280")

    c.save(path)


if __name__ == "__main__":
    out = r"D:\人工智能+\assets"
    os.makedirs(out, exist_ok=True)
    fig_loop(os.path.join(out, "fig1_loop.png"))
    fig_layers(os.path.join(out, "fig2_layers.png"))
    print("done")

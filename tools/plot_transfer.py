# -*- coding: utf-8 -*-
"""把经验迁移实验的结果画成一张对比图（答辩用）。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = json.load(open(os.path.join(path, "metrics.json"), encoding="utf-8"))
    rows = d["rows"]
    used = d["ledger"]["used"]
    base = sum(r["f1_baseline"] for r in rows) / len(rows)
    sysf = sum(r["f1_system"] for r in rows) / len(rows)
    return base, sysf, used["gpu_sec"]


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    b, s_no, c_no = load("out_tx_noexp")
    _, s_exp, c_exp = load("out_tx_exp")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=160)

    # ── ① 分数 ──
    ax = axes[0]
    names = ["静态基线", "无经验", "带经验"]
    vals = [b, s_no, s_exp]
    colors = ["#DC2626", "#94A3B8", "#2563EB"]
    bars = ax.bar(names, vals, color=colors, width=0.6)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("平均 F1（macro）")
    ax.set_title("分数：新数据上更高", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # ── ② 成本 ──
    ax = axes[1]
    names2 = ["无经验", "带经验"]
    vals2 = [c_no, c_exp]
    bars = ax.bar(names2, vals2, color=["#94A3B8", "#059669"], width=0.5)
    for r, v in zip(bars, vals2):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.08, f"{v:.2f}s",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals2) * 1.3)
    ax.set_ylabel("累计算力成本（秒）")
    ax.set_title("成本：反而降了一半", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # ── ③ 性价比 ──
    ax = axes[2]
    eff = [(s_no - b) / max(c_no, 1e-9), (s_exp - b) / max(c_exp, 1e-9)]
    bars = ax.bar(names2, eff, color=["#94A3B8", "#7C3AED"], width=0.5)
    for r, v in zip(bars, eff):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(eff) * 1.3)
    ax.set_ylabel("增益 ÷ 成本（分/秒）")
    ax.set_title(f"性价比：{eff[1]/max(eff[0],1e-9):.1f} 倍", fontsize=12,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle("经验迁移：在 10 个旧轴承上学，到 5 个全新轴承上用",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs("out_tx_exp", exist_ok=True)
    out = "out_tx_exp/迁移实验结果.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()

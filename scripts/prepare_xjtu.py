# -*- coding: utf-8 -*-
"""把 XJTU-SY 轴承数据转成「批次 npz」，和 pipeline 对齐。

原始数据：每个 CSV = 一次采样（1.28 秒，32768 点，水平+垂直两通道）。
一个轴承 = 一次完整的「跑到失效」，采样数从 42 到 2538 不等。

设计：
  · 每个 CSV 压成 16 个时域特征（两通道各 8 个）
  · 一个轴承 = 一个批次（它本身就是一个完整的退化过程）
  · 标签 = 退化阶段：按寿命三等分 → 健康 / 退化中 / 临近失效
  · 15 个轴承按工况排序，构成一条「工况逐渐变化」的漂移流

为什么这么切：
  气体数据是「同类内分布漂移」，轴承是「跨个体 + 跨工况漂移」，
  漂移类型不同 —— 正好用来验证经验表能不能迁移。

用法：
    python scripts/prepare_xjtu.py
    python scripts/prepare_xjtu.py --limit 3      # 每个工况只处理 3 个轴承（调试用）
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

FEATURE_NAMES = []
for ch in ("H", "V"):
    FEATURE_NAMES += [f"{ch}_rms", f"{ch}_kurt", f"{ch}_skew", f"{ch}_peak",
                      f"{ch}_crest", f"{ch}_shape", f"{ch}_impulse", f"{ch}_clear"]


def time_features(sig: np.ndarray) -> np.ndarray:
    """8 个时域指标。这些是轴承健康监测里的标准量，不需要频域也够用。"""
    x = np.asarray(sig, dtype=np.float64)
    absx = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2))
    peak = absx.max()
    mean_abs = absx.mean()
    # 峭度：对冲击（早期故障）最敏感
    mu = x.mean()
    sd = x.std() + 1e-12
    kurt = np.mean(((x - mu) / sd) ** 4)
    skew = np.mean(((x - mu) / sd) ** 3)
    return np.array([
        rms,
        kurt,
        skew,
        peak,
        peak / (rms + 1e-12),          # 峰值因子
        rms / (mean_abs + 1e-12),      # 波形因子
        peak / (mean_abs + 1e-12),     # 脉冲因子
        peak / (np.mean(np.sqrt(absx)) ** 2 + 1e-12),   # 裕度因子
    ])


def bearing_id(name: str) -> tuple[str, int]:
    m = re.match(r"Bearing(\d+)_(\d+)", name)
    if not m:
        raise ValueError(name)
    return m.group(1), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"data/raw/xjtu_sy/XJTU-SY_Bearing_Datasets")
    ap.add_argument("--out", default="data/xjtu")
    ap.add_argument("--limit", type=int, default=0, help="每个工况只取前 N 个轴承")
    ap.add_argument("--stages", type=int, default=3, help="退化阶段数")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"没找到 {args.src}")

    conds = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
    os.makedirs(args.out, exist_ok=True)

    index = []
    batch = 0
    total_files = 0

    for cond in conds:
        cdir = os.path.join(args.src, cond)
        if not os.path.isdir(cdir):
            continue
        bears = sorted(os.listdir(cdir))
        if args.limit:
            bears = bears[:args.limit]

        for b in bears:
            files = sorted(glob.glob(os.path.join(cdir, b, "*.csv")),
                           key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            if len(files) < args.stages * 2:
                print(f"  跳过 {cond}/{b}：只有 {len(files)} 个采样，太少"
                      f"（至少要 {args.stages * 2} 个）")
                continue

            # 【修复】单个轴承读取失败不应中断整批，但必须逐条报出来，
            # 而不是让异常冒到顶层或悄悄跳过。
            try:
                import pandas as pd
                feats = []
                for f in files:
                    df = pd.read_csv(f)
                    arr = df.to_numpy(dtype=np.float64)
                    if arr.ndim != 2 or arr.shape[1] < 2:
                        raise ValueError(f"{os.path.basename(f)} 不是两列数值"
                                         f"（实际形状 {arr.shape}）")
                    feats.append(np.concatenate([time_features(arr[:, 0]),
                                                 time_features(arr[:, 1])]))
                X = np.vstack(feats)
            except Exception as exc:              # noqa: BLE001
                print(f"  跳过 {cond}/{b}：读取失败 "
                      f"（{type(exc).__name__}: {str(exc)[:80]}）")
                continue

            # 标签：按寿命三等分（相对位置，不按绝对时间）
            n = len(X)
            rel = np.arange(n) / max(n - 1, 1)
            y = np.minimum((rel * args.stages).astype(int), args.stages - 1)

            batch += 1
            path = os.path.join(args.out, f"batch_{batch:02d}.npz")
            np.savez_compressed(path, X=X, y=y,
                                condition=cond, bearing=b, n_samples=n)
            index.append({"batch": batch, "condition": cond, "bearing": b,
                          "n": n, "file": os.path.basename(path)})
            total_files += n
            print(f"  batch_{batch:02d}  {cond:<12} {b:<12} {n:>5} 条  "
                  f"类别 {np.bincount(y, minlength=args.stages).tolist()}")

    # 【修复】一个批次都没产出 = 输入不可用，必须非零退出，不能生成"空的成功产物"
    if not index:
        print("\n[!] 没有产出任何批次：输入目录里没有可用的 CSV。")
        print(f"    查找路径：{args.src}")
        print(f"    期望子目录：{', '.join(conds)}/<BearingX_Y>/*.csv（两列数值）")
        # 清掉可能已创建的空输出目录，避免留下"看起来成功"的痕迹
        try:
            if os.path.isdir(args.out) and not os.listdir(args.out):
                os.rmdir(args.out)
                print(f"    已清理空输出目录：{args.out}")
        except OSError:
            pass
        raise SystemExit(2)

    with open(os.path.join(args.out, "index.txt"), "w", encoding="utf-8") as f:
        for r in index:
            f.write(f"{r['batch']:02d}\t{r['condition']}\t{r['bearing']}\t{r['n']}\n")

    print(f"\n完成：{len(index)} 个批次，{total_files} 条样本，写入 {args.out}/")
    print(f"特征维数：{len(FEATURE_NAMES)}")
    if index:
        print("\n批次顺序（按工况排列，工况变化本身就是漂移）：")
        for r in index:
            print(f"  {r['batch']:02d}  {r['condition']:<12} {r['bearing']}")


if __name__ == "__main__":
    main()

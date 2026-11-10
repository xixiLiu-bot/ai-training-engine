# -*- coding: utf-8 -*-
"""把 NASA C-MAPSS 转成批次 npz，供 RULAdapter（回归任务）使用。

原始数据：train_FD00X.txt，26 列
    单元号, 周期, 3 个工况设置, 21 个传感器读数
    每台发动机从不同起点跑到失效，每行是一个运行周期。

标签口径（重要，直接影响 MAE）：
    RUL = 这台发动机的最大周期 - 当前周期
    并按惯例截断到 125（piecewise linear）——超过 125 的部分视为"健康"，
    因为早期退化信号不可辨识，强行拟合会伤害 MAE。

切批方式：
    默认按「发动机」切——100 台分成 10 批，每批 10 台。
    每批是不同发动机的完整寿命段，漂移来自个体差异与磨损状态差异。
    这对应赛题里"平台一批批下发新设备数据"的场景。

    另有 --by cycle：按周期窗口切，每批是所有发动机在同一时间窗的切片。

用法：
    python scripts/prepare_rul.py                    # FD001，默认 10 批
    python scripts/prepare_rul.py --subset FD002     # 更复杂（6 个工况）
    python scripts/prepare_rul.py --by cycle --batches 8
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

RUL_CAP = 125.0


def load_cmapss(path: str) -> np.ndarray:
    d = np.loadtxt(path)
    if d.ndim != 2 or d.shape[1] < 26:
        raise ValueError(f"{path} 列数异常：{d.shape}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"data/CMAPSSData")
    ap.add_argument("--subset", default="FD001",
                    choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--out", default="data/rul")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--by", default="unit", choices=["unit", "cycle"])
    ap.add_argument("--rul-cap", type=float, default=RUL_CAP)
    args = ap.parse_args()

    path = os.path.join(args.root, f"train_{args.subset}.txt")
    if not os.path.exists(path):
        # 允许根目录下再套一层
        cand = glob.glob(os.path.join(args.root, "**", f"train_{args.subset}.txt"),
                         recursive=True)
        if not cand:
            raise SystemExit(f"没找到 {args.subset} 的训练文件（找过 {args.root}）")
        path = cand[0]

    d = load_cmapss(path)
    unit = d[:, 0].astype(int)
    cycle = d[:, 1].astype(int)
    feat = d[:, 5:]                       # 21 个传感器；工况列先不用（FD001 恒定）

    # 去掉常数传感器：没有信息，反而让标准化分母为 0
    std = feat.std(axis=0)
    keep = std > 1e-6
    feat = feat[:, keep]

    # 标签：每台发动机的剩余寿命，截断到上限
    max_cycle = {u: cycle[unit == u].max() for u in np.unique(unit)}
    rul = np.array([max_cycle[u] - c for u, c in zip(unit, cycle)], dtype=float)
    rul = np.clip(rul, 0.0, args.rul_cap)

    units = np.unique(unit)
    os.makedirs(args.out, exist_ok=True)

    if args.by == "unit":
        groups = np.array_split(units, args.batches)
        slices = [(np.isin(unit, g), f"发动机 {g[0]}~{g[-1]}（{len(g)} 台）")
                  for g in groups]
    else:
        order = np.argsort(cycle)
        parts = np.array_split(order, args.batches)
        slices = [(np.isin(np.arange(len(cycle)), p),
                   f"周期 {cycle[p].min()}~{cycle[p].max()}") for p in parts]

    index = []
    for i, (mask, desc) in enumerate(slices, 1):
        X = feat[mask]
        y = rul[mask]
        if len(y) == 0:
            print(f"  跳过第 {i} 批：空")
            continue
        path_out = os.path.join(args.out, f"batch_{i:02d}.npz")
        np.savez_compressed(path_out, X=X, y=y,
                            units=unit[mask], cycles=cycle[mask])
        index.append((i, desc, len(y), float(y.mean())))
        print(f"  batch_{i:02d}  {desc:<26} {len(y):>5} 条  "
              f"RUL 均值 {y.mean():6.1f}  范围 {y.min():.0f}~{y.max():.0f}")

    with open(os.path.join(args.out, "index.txt"), "w", encoding="utf-8") as f:
        f.write(f"subset\t{args.subset}\tby={args.by}\trul_cap={args.rul_cap}\n")
        f.write("batch\tdescription\tn\trul_mean\n")
        for r in index:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.1f}\n")

    print(f"\n完成：{len(index)} 个批次 → {args.out}/")
    print(f"保留传感器 {int(keep.sum())}/{len(keep)} 个（去掉 {int((~keep).sum())} 个常数）")
    print(f"标签：RUL 截断到 {args.rul_cap:.0f}，回归任务（指标 = MAE / RMSE）")


if __name__ == "__main__":
    main()

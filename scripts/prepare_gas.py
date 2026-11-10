# -*- coding: utf-8 -*-
"""把 UCI 气体传感器漂移数据集（libsvm 稀疏格式 .dat）转成批次 npz。

原始格式长这样（每行一个样本）：

    6 1:15596.16 2:1.868 3:2.371 ... 128:5.1

第一列是气体类别（1~6），后面是 `特征索引:数值`，索引从 1 开始。

为什么要转：评测环境要反复读，稀疏文本解析慢；转成稠密 npz 后加载是毫秒级。
而且转完就和合成数据同构，闭环代码一行都不用改。

用法：
    python scripts/prepare_gas.py
    python scripts/prepare_gas.py --src "data/Dataset (1)/Dataset" --out data/gas
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np


def parse_libsvm(path: str, n_features: int):
    """读一个 .dat，返回 (X 稠密矩阵, y 从 0 开始的标签)。"""
    ys, rows, cols, vals = [], [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for r, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ys.append(int(float(parts[0])))
            for p in parts[1:]:
                i, _, v = p.partition(":")
                rows.append(r)
                cols.append(int(i) - 1)      # libsvm 从 1 开始
                vals.append(float(v))
    n = len(ys)
    X = np.zeros((n, n_features), dtype=np.float64)
    X[np.array(rows), np.array(cols)] = np.array(vals)
    y = np.array(ys, dtype=int)
    y = y - y.min()                          # 1..6 → 0..5
    return X, y


def batch_id(path: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"文件名里没有批次号：{path}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    # 默认锚定到代码包根目录，避免从别的 cwd 跑时把数据写到奇怪的地方
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--src", default=os.path.join(repo, "data", "gas", "raw", "Dataset"))
    ap.add_argument("--out", default=os.path.join(repo, "data", "gas", "processed"))
    ap.add_argument("--features", type=int, default=128)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "batch*.dat")),
                   key=batch_id)
    if not files:
        raise SystemExit(f"没找到 {args.src} 下的 batch*.dat")

    os.makedirs(args.out, exist_ok=True)
    print(f"找到 {len(files)} 个批次，特征维数 {args.features}\n")

    total = 0
    for path in files:
        b = batch_id(path)
        X, y = parse_libsvm(path, args.features)
        np.savez_compressed(os.path.join(args.out, f"batch_{b:02d}.npz"), X=X, y=y)
        dist = np.bincount(y, minlength=6)
        total += len(y)
        print(f"  batch_{b:02d}: {len(y):5d} 条  类别分布 {dist.tolist()}")

    print(f"\n合计 {total} 条，写入 {args.out}/")


if __name__ == "__main__":
    main()

"""生成一条「会漂移」的合成数据流。

用途：官方数据包到手之前，先用它把闭环跑通、把曲线画出来。

关键设计：先一次性生成一个**固定的底层分布**，再从里面抽样组成各批次。
否则每批都重新生成聚类几何结构，批间差异会变成随机噪声，
PSI 就测不出“漂移”，只剩抖动。

两段漂移，正好对应两种处置手段：
  第 2~5 批：协变量漂移（整体缩放 + 平移）→ 便宜的“重算标准化”就能救
  第 6~10 批：叠加概念漂移（部分样本标签关系改变）→ 必须动模型权重

用法：
    python scripts/make_data.py
    python scripts/make_data.py --batches 12 --n 3000 --shift 1.2
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def build_pool(seed, n_pool, n_features, n_informative, n_classes):
    """一次性生成底层数据池，后面所有批次都从这里抽。"""
    from sklearn.datasets import make_classification

    Z, _ = make_classification(
        n_samples=n_pool,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=1.0,
        random_state=seed,
    )
    # 用固定的标签规则，而不是 make_classification 自带的那套
    signal = Z[:, :n_informative].sum(axis=1)
    cuts = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(signal, cuts) % n_classes
    return Z, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--n", type=int, default=2000, help="每批样本数")
    ap.add_argument("--features", type=int, default=20)
    ap.add_argument("--informative", type=int, default=8)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--shift", type=float, default=0.42,
                    help="末批的平移幅度（以特征标准差为单位）")
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(42)

    n_pool = args.n * args.batches * 2
    Z_pool, y_pool = build_pool(42, n_pool, args.features,
                                args.informative, args.classes)
    std = Z_pool.std(axis=0)
    order = rng.permutation(n_pool)          # 打散，各批互不重叠

    for b in range(1, args.batches + 1):
        t = (b - 1) / max(1, args.batches - 1)      # 0 → 1

        idx = order[(b - 1) * args.n: b * args.n]
        Z, y = Z_pool[idx].copy(), y_pool[idx].copy()

        # ---- 协变量漂移：前 5 批逐渐放大 + 平移，之后**保持不再变化** ----
        # 这样第 6 批起的分数下滑只能来自标签关系改变（纯概念漂移），
        # 才能验证“便宜的校准救不了 → 必须升级”这条路径。
        t_cov = min(1.0, (b - 1) / max(1, min(5, args.batches) - 1))
        scale = 1.0 + 0.25 * t_cov
        offset = args.shift * t_cov
        X = Z * scale + offset * std

        # ---- 概念漂移：一部分样本的标签被改写 ----
        # ---- 概念漂移：第 6 批起，类别 0 与 1 的判定标准整体互换 ----
        # 这是“规则变了”而不是“数据变脏了”：重新训练**能完全学回来**。
        # 若做成随机翻标签，标签就不可学了，重训反而更差——那是错误建模。
        concept_flag = 0.0 if b <= 5 else 1.0
        if concept_flag > 0:
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))

        np.savez_compressed(
            os.path.join(args.out, f"batch_{b:02d}.npz"),
            X=X, y=y, scale=scale, offset=offset, concept_flag=concept_flag,
        )
        print(f"  batch_{b:02d}: n={len(y)} scale={scale:.2f} "
              f"offset={offset:.2f}σ 换类={concept_flag:.0f}")

    print(f"\n完成：{args.batches} 个批次写入 {args.out}/")


if __name__ == "__main__":
    main()

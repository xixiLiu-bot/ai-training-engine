"""数据适配器：把不同来源的数据切成统一的「批次流」。

所有适配器都实现同一个协议：

    adapter.name                 -> str
    adapter.feature_names        -> list[str]
    adapter.n_batches            -> int
    adapter.get(batch_id)        -> Batch

Batch 就是一条「产线数据」：一批特征 + 这批的标签。
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    """一个数据批次。"""
    batch_id: int
    X: np.ndarray          # (n_samples, n_features)
    y: np.ndarray          # (n_samples,)
    meta: dict | None = None

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y)


class DatasetAdapter:
    """数据集适配器的最小协议（子类实现 name / feature_names / n_batches / get）。"""

    name: str = "base"
    feature_names: list[str] = []
    n_batches: int = 0

    def get(self, batch_id: int) -> Batch:  # pragma: no cover - 接口声明
        raise NotImplementedError

    def __iter__(self):
        for i in range(1, self.n_batches + 1):
            yield self.get(i)

    def __repr__(self):
        return f"<{type(self).__name__} name={self.name} n_batches={self.n_batches}>"


# ─────────────────────────────────────────────────────────────
# 1) 合成漂移流：官方数据没到之前的替身，随时可用
# ─────────────────────────────────────────────────────────────
class SyntheticDriftAdapter(DatasetAdapter):
    """读 data/synthetic/batch_XX.npz（由 scripts/make_data.py 生成）。"""

    def __init__(self, root: str = "data/synthetic"):
        self.root = root
        files = sorted(glob.glob(os.path.join(root, "batch_*.npz")))
        if not files:
            raise FileNotFoundError(
                f"{root} 下没有批次文件。先跑：python scripts/make_data.py"
            )
        self._files = files
        self.name = "synthetic_drift"
        self.n_batches = len(files)
        with np.load(files[0]) as d:
            n_feat = d["X"].shape[1]
        self.feature_names = [f"f{i}" for i in range(n_feat)]

    def get(self, batch_id: int) -> Batch:
        with np.load(self._files[batch_id - 1]) as d:
            meta = {"source": self.name}
            for k in d.files:
                if k not in ("X", "y"):
                    meta[k] = float(d[k])
            return Batch(batch_id, d["X"], d["y"], meta)


# ─────────────────────────────────────────────────────────────
# 2) UCI 气体传感器漂移数据集：真实的物理漂移
# ─────────────────────────────────────────────────────────────
class GasSensorAdapter(DatasetAdapter):
    """UCI Gas Sensor Array Drift（官方 A 包三个数据集之一）。

    真实漂移来源：传感器在数月内老化，同一气体的读数持续偏移。
    原始文件是 libsvm 稀疏格式（.dat），先用 scripts/prepare_gas.py
    转成 batch_XX.npz，运行时直接读，加载是毫秒级。

    批次顺序就是时间顺序：batch1 最早、batch10 最晚。
    """

    def __init__(self, root: str = "data/gas"):
        files = sorted(
            glob.glob(os.path.join(root, "batch_*.npz")),
            key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())),
        )
        if not files:
            raise FileNotFoundError(
                f"没找到 {root} 下的 batch_*.npz。先跑：python scripts/prepare_gas.py")
        self._files = files
        self.name = "gas_sensor_drift"
        self.n_batches = len(files)
        with np.load(files[0]) as d:
            self.feature_names = [f"s{i}" for i in range(d["X"].shape[1])]

    def get(self, batch_id: int) -> Batch:
        with np.load(self._files[batch_id - 1]) as d:
            return Batch(batch_id, d["X"], d["y"], {"source": self.name})


# ─────────────────────────────────────────────────────────────
# 3) 官方数据包：等它到货，只改这一个文件
# ─────────────────────────────────────────────────────────────
class OfficialAdapter(DatasetAdapter):
    """官方数据包适配器（占位）。

    拿到数据包后，只需要改这个类的 __init__ 和 get()，
    loop/ 下面的代码一行都不用动。
    """

    def __init__(self, root: str = "data/official"):
        raise NotImplementedError(
            "官方数据包尚未下发。拿到后在这里实现：读训练集/验证集，按批次切分。"
        )


def build(kind: str = "synthetic", **kwargs) -> DatasetAdapter:
    """工厂：按名字拿适配器。"""
    table = {
        "synthetic": SyntheticDriftAdapter,
        "gas": GasSensorAdapter,
    }
    if kind not in table:
        raise ValueError(f"未知数据集 '{kind}'，可选：{list(table)}")
    return table[kind](**kwargs)

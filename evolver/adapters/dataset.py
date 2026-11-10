"""数据适配器：把不同来源的数据切成统一的「批次流」。

所有适配器都实现同一个协议：

    adapter.name                 -> str
    adapter.task_type            -> "classification" | "regression"   【整改 A4 新增】
    adapter.feature_names        -> list[str]
    adapter.n_batches            -> int
    adapter.get(batch_id)        -> Batch

Batch 就是一条「产线数据」：一批特征 + 这批的标签。

【整改 A4 / B7】原来只有 synthetic 与 gas 两个数据集，且没有任务类型，
主流程写死 f1_macro，导致 RUL 这类回归任务根本无法正确评估。
本轮补充：
    - task_type 字段（决定指标分派）
    - SyntheticRULAdapter   回归用合成流（无需外部数据即可验证回归链路）
    - BearingAdapter        轴承文件加载器（data/bearing/batch_*.npz）
    - RULAdapter            官方 RUL 加载器（data/rul/batch_*.npz）
原有 synthetic / gas 的行为与接口完全不变。
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
    task_type: str = "classification"
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

    task_type = "classification"

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
def _repo_root() -> str:
    """evolver/adapters/dataset.py → 上溯三级 = 代码包根目录。

    数据路径必须锚定到这里，不能用相对 CWD 的 "data/gas"：
    否则从 workspace 其它目录跑 run_pipeline 就会 FileNotFoundError。
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_batch_npz(root: str) -> list[str]:
    """在 root 与 root/processed 下找 batch_*.npz，按批次号排序。

    兼容两种布局：老的数据直接放在 data/gas/ 下，新结构放在 data/gas/processed/ 下。
    """
    files: list[str] = []
    for d in (os.path.join(root, "processed"), root):
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "batch_*.npz")))
    uniq = {os.path.abspath(p): p for p in files}
    return sorted(uniq.values(),
                  key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))


class GasSensorAdapter(DatasetAdapter):
    """UCI Gas Sensor Array Drift（官方 A 包三个数据集之一）。

    真实漂移来源：传感器在数月内老化，同一气体的读数持续偏移。
    原始文件是 libsvm 稀疏格式（.dat），先用 scripts/prepare_gas.py
    转成 batch_XX.npz，运行时直接读，加载是毫秒级。

    批次顺序就是时间顺序：batch1 最早、batch10 最晚。
    """

    task_type = "classification"

    def __init__(self, root: str | None = None):
        root = root or os.path.join(_repo_root(), "data", "gas")
        files = _find_batch_npz(root)
        if not files:
            raise FileNotFoundError(
                f"没找到 {root}（或其 processed/）下的 batch_*.npz。"
                f"先跑：python scripts/prepare_gas.py")
        self._files = files
        self.name = "gas_sensor_drift"
        self.n_batches = len(files)
        with np.load(files[0]) as d:
            self.feature_names = [f"s{i}" for i in range(d["X"].shape[1])]

    def get(self, batch_id: int) -> Batch:
        with np.load(self._files[batch_id - 1]) as d:
            return Batch(batch_id, d["X"], d["y"], {"source": self.name})


# ─────────────────────────────────────────────────────────────
# 3) 合成 RUL 流：把"回归链路"在无外部数据的情况下跑通
# ─────────────────────────────────────────────────────────────
class SyntheticRULAdapter(DatasetAdapter):
    """合成剩余寿命流（回归任务）。生成方式与 make_data.py 同构：

        隐状态 h ~ U(0,1)          # 退化程度，0=全新 1=报废
        X = h * mu + noise         # 可观测信号随退化增强
        y = RUL_MAX * (1 - h)      # 剩余寿命（天/循环）

    漂移：
        第 2~5 批：协变量漂移（传感器增益 scale 逐渐放大 + 零点 offset）
        第 6 批起：概念漂移（同样的信号对应更短的剩余寿命 → 映射关系改变）

    这是**开发用替身**，不是官方 RUL 数据。官方数据到手后用 RULAdapter。
    """

    task_type = "regression"

    def __init__(self, batches: int = 10, n: int = 800, features: int = 12,
                 rul_max: float = 125.0, shift: float = 0.35, seed: int = 42):
        self.name = "synthetic_rul"
        self.task_type = "regression"
        self.n_batches = batches
        self.feature_names = [f"s{i}" for i in range(features)]
        self._batches: dict[int, Batch] = {}
        rng = np.random.default_rng(seed)
        mu = np.linspace(1.0, 3.0, features)          # 各特征对退化的敏感度不同

        for b in range(1, batches + 1):
            h = rng.uniform(0.0, 1.0, size=n)
            X = h[:, None] * mu[None, :] + rng.normal(scale=0.4, size=(n, features))

            t_cov = min(1.0, (b - 1) / max(1, min(5, batches) - 1))
            scale = 1.0 + 0.20 * t_cov
            offset = shift * t_cov
            X = X * scale + offset

            concept = 0.0 if b <= 5 else 1.0
            if concept > 0:
                # 退化加速：同样的 h 对应更短的剩余寿命
                y = rul_max * (1.0 - h) ** 1.6
            else:
                y = rul_max * (1.0 - h)

            self._batches[b] = Batch(
                b, X, y,
                {"source": self.name, "scale": scale, "offset": offset,
                 "concept_flag": concept},
            )

    def get(self, batch_id: int) -> Batch:
        return self._batches[batch_id]


# ─────────────────────────────────────────────────────────────
# 4) 轴承退化识别（分类）—— 文件加载器
# ─────────────────────────────────────────────────────────────
class BearingAdapter(DatasetAdapter):
    """轴承故障分类（正常/内圈/外圈/滚动体），读 data/bearing/batch_*.npz。

    与 gas 适配器同构，只要求 npz 里有 X 与 y。
    公开数据来源：CWRU Bearing Data Center / XJTU-SY。
    拿到原始数据后写一个 prepare_bearing.py，切批并转成 npz 即可，闭环不用改。
    """

    task_type = "classification"

    def __init__(self, root: str = "data/bearing"):
        files = sorted(
            glob.glob(os.path.join(root, "batch_*.npz")),
            key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())),
        )
        if not files:
            raise FileNotFoundError(
                f"没找到 {root} 下的 batch_*.npz。\n"
                "  1) 下载 CWRU 或 XJTU-SY 轴承数据\n"
                "  2) 写 scripts/prepare_bearing.py 切成 batch_XX.npz（X, y）\n"
                "  3) 重跑本适配器")
        self._files = files
        self.name = "bearing"
        self.n_batches = len(files)
        with np.load(files[0]) as d:
            self.feature_names = [f"v{i}" for i in range(d["X"].shape[1])]

    def get(self, batch_id: int) -> Batch:
        with np.load(self._files[batch_id - 1]) as d:
            return Batch(batch_id, d["X"], d["y"], {"source": self.name})


# ─────────────────────────────────────────────────────────────
# 5) 官方 RUL 数据（回归）—— 文件加载器
# ─────────────────────────────────────────────────────────────
class RULAdapter(DatasetAdapter):
    """发动机剩余寿命评估（回归），读 data/rul/batch_*.npz。

    公开数据来源：NASA C-MAPSS（FD001~FD004）。
    注意：C-MAPSS 原始格式是 train/test/RUL 三个 txt，需要先按"时间切片"切批，
    再转成 batch_XX.npz（X=传感器特征, y=剩余寿命）。详见 scripts/prepare_rul.py（待补）。

    标签口径提醒：C-MAPSS 的 RUL 常被截断到 125 上限（piecewise linear），
    这个截断点会显著影响 MAE，务必和官方评测脚本对齐。
    """

    task_type = "regression"

    def __init__(self, root: str = "data/rul"):
        files = sorted(
            glob.glob(os.path.join(root, "batch_*.npz")),
            key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())),
        )
        if not files:
            raise FileNotFoundError(
                f"没找到 {root} 下的 batch_*.npz。\n"
                "  1) 下载 NASA C-MAPSS（FD001 起）\n"
                "  2) 写 scripts/prepare_rul.py 按时间切批 → batch_XX.npz（X, y）\n"
                "  3) 重跑本适配器")
        self._files = files
        self.name = "rul"
        self.n_batches = len(files)
        with np.load(files[0]) as d:
            self.feature_names = [f"c{i}" for i in range(d["X"].shape[1])]

    def get(self, batch_id: int) -> Batch:
        with np.load(self._files[batch_id - 1]) as d:
            return Batch(batch_id, d["X"], d["y"], {"source": self.name})


# ─────────────────────────────────────────────────────────────
# 6) 官方数据包：等它到货，只改这一个文件
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


class SubsetAdapter(DatasetAdapter):
    """只暴露 inner 适配器的 start..end 这几批。

    迁移实验要「前 N 批攒经验、后 M 批验证」，用这个视图切，
    闭环代码一行都不用改。
    """

    def __init__(self, inner: DatasetAdapter, start: int = 1,
                 end: int | None = None):
        self._inner = inner
        last = inner.n_batches if end is None else min(end, inner.n_batches)
        if start < 1 or start > last:
            raise ValueError(
                f"批次范围非法：{start}..{last}（共 {inner.n_batches} 批）")
        self.start = start
        self.end = last
        self.n_batches = last - start + 1
        self.name = f"{inner.name}[{start}-{last}]"
        self.task_type = getattr(inner, "task_type", "classification")
        self.feature_names = inner.feature_names

    def get(self, batch_id: int) -> Batch:
        b = self._inner.get(self.start + batch_id - 1)
        return Batch(batch_id, b.X, b.y,
                     {**(b.meta or {}), "orig_batch": b.batch_id})


def _build_gas_drift(**kwargs) -> DatasetAdapter:
    """走新数据源层（sources.GasSource + bridge.BridgeAdapter）。

    与旧的 `gas` 适配器读同一份 npz，区别是：新层额外提供 split / 元信息 /
    许可证说明 / 每批缺哪些类别，便于核验与文档化。
    延迟 import 避免 dataset ↔ bridge 循环依赖。
    """
    from .bridge import BridgeAdapter
    from .sources import GasSource

    batch_ids = kwargs.pop("batch_ids", None)
    return BridgeAdapter(GasSource(**kwargs), batch_ids=batch_ids)


def build(kind: str = "synthetic", **kwargs) -> DatasetAdapter:
    """工厂：按名字拿适配器。默认行为与旧版一致。"""
    table = {
        "synthetic": SyntheticDriftAdapter,
        "gas": GasSensorAdapter,
        "gas_drift": _build_gas_drift,      # 新数据源层（含 split/元信息）
        "synthetic_rul": SyntheticRULAdapter,
        "bearing": BearingAdapter,
        "rul": RULAdapter,
    }
    if kind not in table:
        raise ValueError(f"未知数据集 '{kind}'，可选：{list(table)}")
    return table[kind](**kwargs)

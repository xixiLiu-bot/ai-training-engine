"""统一数据源接口（DatasetSource）。

【为什么新增这一层】
现有闭环用的是极简协议 `name / task_type / n_batches / get(batch_id)`，
够用但不足以描述"数据从哪来、怎么切、许可证是什么、hash 是多少"。
直接在 DatasetAdapter 上加 8 个方法会让闭环和协议层都被迫改动，
所以这里单独建一层"数据源"，再用 bridge.py 桥接成旧协议——闭环零改动。

统一记录格式（iter_batches 的产物）：

    {
      "dataset":   "gas",
      "task_type": "classification",
      "batch_id":  1,               # 数据源内部的原始批次号（从 1 开始，按时间递增）
      "timestamp": None,            # 原始数据若没有真实时间戳，必须写 None 并说明
      "features":  np.ndarray (n, d),
      "labels":    np.ndarray (n,),
      "metadata":  {...},           # 批次级元信息
    }

【本文件当前只实现 Gas】
C-MAPSS 与 XJTU-SY 的下载地址、许可证、任务定义尚未确认，按用户要求暂不实现。
"""
from __future__ import annotations

from . import identities

import glob
import os
from typing import Iterator

import numpy as np


class DataSourceError(RuntimeError):
    """数据缺失或格式不符。**必须抛出明确指引**，不能静默返回空数据。"""


# ─────────────────────────────────────────────────────────────
# 基类：定义契约，子类必须全部实现
# ─────────────────────────────────────────────────────────────
class DatasetSource:
    name: str = "base"
    task_type: str = "classification"
    #: 数据身份。子类必须声明，默认 unknown（会被选择类操作拒绝）。
    #: 见 adapters/identities.py 的取值与含义。
    data_identity: str = identities.UNKNOWN

    def load(self) -> "DatasetSource":
        raise NotImplementedError

    def split(self) -> dict:
        """返回 {'train': [...], 'stream': [...], 'holdout': [...]}（原始批次号）。

        契约：
          - 三个集合两两不相交
          - holdout 必须是时间上**最后**的批次
          - train 只能是最早的批次（模拟"部署那天"）
        """
        raise NotImplementedError

    def iter_batches(self) -> Iterator[dict]:
        raise NotImplementedError

    def get_features(self) -> list[str]:
        raise NotImplementedError

    def get_labels(self) -> list[str]:
        raise NotImplementedError

    def get_task_type(self) -> str:
        return self.task_type

    def get_metric(self) -> str:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# Gas Sensor Array Drift（UCI 224）
# ─────────────────────────────────────────────────────────────
GAS_CLASSES = ["Ammonia", "Acetaldehyde", "Acetone", "Ethylene", "Ethanol", "Toluene"]
GAS_N_FEATURES = 128
GAS_EXPECTED_SIZES = [445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600]  # 合计 13910

# UCI 文档对 128 维向量的描述：每 8 个特征属于一个传感器，传感器共 16 个
_GAS_FEATURE_BASE = ["dR", "abs_dR",
                     "EMAi0.001", "EMAi0.01", "EMAi0.1",
                     "EMAd0.001", "EMAd0.01", "EMAd0.1"]

GAS_SOURCE_URL = "https://archive.ics.uci.edu/dataset/224/gas"
GAS_LICENSE = ("UCI 页面原文：dataset can be used exclusively for research purposes. "
               "Commercial purposes are fully excluded.")
GAS_LICENSE_ZH = "仅限研究用途，明确排除商业用途。"


def _repo_root() -> str:
    """evolver/adapters/sources.py → 上溯三级 = 代码包根目录。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _gas_feature_names() -> list[str]:
    return [f"{base}_s{sensor}"
            for sensor in range(1, 17) for base in _GAS_FEATURE_BASE]


class GasSource(DatasetSource):
    """UCI Gas Sensor Array Drift（10 批，6 类，128 维）。

    数据形态：`data/gas/processed/batch_01.npz … batch_10.npz`（X, y）。
    原始 `.dat`（libsvm 稀疏）用 `scripts/prepare_gas.py` 转换。

    批次顺序 = 时间顺序（batch1 最早、batch10 最晚），**必须保留**。
    """

    name = "gas"
    # 公开替代数据：允许开发期使用，但必须声明"非官方"
    data_identity = identities.PUBLIC_SUBSTITUTE
    task_type = "classification"

    def __init__(self, root: str | None = None, strict_sizes: bool = True):
        # 默认新结构 data/gas/processed；若不存在则回退到 data/gas（老布局）
        base = root or os.path.join(_repo_root(), "data", "gas")
        self.root = base
        self.strict_sizes = strict_sizes
        self._batches: list[dict] = []
        self._loaded = False

    def _dirs(self) -> list[str]:
        return [os.path.join(self.root, "processed"), self.root]

    # ---------------- 加载 ----------------
    def load(self) -> "GasSource":
        files: list[str] = []
        for d in self._dirs():
            if os.path.isdir(d):
                files.extend(glob.glob(os.path.join(d, "batch_*.npz")))
        uniq = {os.path.abspath(p): p for p in files}
        files = sorted(uniq.values(),
                       key=lambda p: int("".join(c for c in os.path.basename(p)
                                                 if c.isdigit()) or 0))
        if not files:
            raise DataSourceError(
                f"没有找到气体批次文件：{self.root}/processed 或 {self.root}\n"
                "  1) 下载并解压 UCI 数据：\n"
                f"     {GAS_SOURCE_URL}\n"
                "  2) 生成批次：python scripts/prepare_gas.py\n"
                "  3) 重跑本数据源")

        batches = []
        for path in files:
            with np.load(path) as d:
                X = np.asarray(d["X"], dtype=float)
                y = np.asarray(d["y"]).ravel()
            if X.shape[1] != GAS_N_FEATURES:
                raise DataSourceError(
                    f"{os.path.basename(path)} 特征维数是 {X.shape[1]}，"
                    f"预期 {GAS_N_FEATURES}")
            batches.append({
                "batch_id": len(batches) + 1,          # 按时间顺序重编号，1..N
                "file": os.path.basename(path),
                "X": X,
                "y": y,
            })

        if self.strict_sizes and len(batches) == len(GAS_EXPECTED_SIZES):
            actual = [len(b["y"]) for b in batches]
            if actual != GAS_EXPECTED_SIZES:
                raise DataSourceError(
                    "批次样本数与 UCI 官方口径不一致：\n"
                    f"  实际 {actual}\n  预期 {GAS_EXPECTED_SIZES}\n"
                    "数据可能被重新切分或损坏，请重新运行 prepare_gas.py")

        self._batches = batches
        self._loaded = True
        return self

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    # ---------------- 划分 ----------------
    def split(self) -> dict:
        """train=第1批，stream=中间批，holdout=最后一批。

        为什么要显式切：门槛③/④的核验与"防过拟合"都依赖"留出批次只看一次"。
        默认 10 批 → train=[1] stream=[2..9] holdout=[10]。
        """
        self._ensure()
        ids = [b["batch_id"] for b in self._batches]
        if len(ids) < 2:
            return {"train": ids, "stream": [], "holdout": []}
        return {"train": [ids[0]], "stream": ids[1:-1], "holdout": [ids[-1]]}

    # ---------------- 迭代 ----------------
    def iter_batches(self) -> Iterator[dict]:
        self._ensure()
        for b in self._batches:
            yield {
                "dataset": self.name,
                "task_type": self.task_type,
                "batch_id": b["batch_id"],
                # UCI 未提供每一批的确切采集日期，只有"批次顺序=时间顺序"，
                # 所以这里如实写 None，不编造时间戳。
                "timestamp": None,
                "features": b["X"],
                "labels": b["y"],
                "metadata": {
                    "source_file": b["file"],
                    "n_samples": int(len(b["y"])),
                    "time_order": b["batch_id"],
                    "time_note": "UCI 未提供每批确切采集日期；批次顺序即时间顺序",
                    "classes_present": sorted(int(c) for c in np.unique(b["y"])),
                },
            }

    # ---------------- 元信息 ----------------
    def get_features(self) -> list[str]:
        # 名称按 UCI 文档描述的排列生成（16 传感器 × 8 特征）。
        # libsvm 文件本身没有表头，所以这是"按文档推断"，不是从数据校验出来的。
        return _gas_feature_names()

    def get_labels(self) -> list[str]:
        return list(GAS_CLASSES)

    def get_metric(self) -> str:
        return "f1_macro"

    def get_metadata(self) -> dict:
        self._ensure()
        sizes = {b["batch_id"]: int(len(b["y"])) for b in self._batches}
        missing = {}
        for b in self._batches:
            present = set(int(c) for c in np.unique(b["y"]))
            lack = [i for i in range(len(GAS_CLASSES)) if i not in present]
            if lack:
                missing[b["batch_id"]] = [GAS_CLASSES[i] for i in lack]
        return {
            "dataset": self.name,
            "task_type": self.task_type,
            "metric": self.get_metric(),
            # 数据身份：决定这份数据能用来干什么（见 adapters/identities.py）
            "data_identity": self.data_identity,
            "data_identity_zh": identities.describe(self.data_identity),
            "n_batches": len(self._batches),
            "n_samples": int(sum(sizes.values())),
            "n_features": GAS_N_FEATURES,
            "n_classes": len(GAS_CLASSES),
            "classes": list(GAS_CLASSES),
            "batch_sizes": sizes,
            # 真实数据特性：这三个月份没测 Toluene，宏平均 F1 会因此系统性偏低
            "batches_missing_classes": missing,
            "split": self.split(),
            "source_url": GAS_SOURCE_URL,
            "license": GAS_LICENSE,
            "license_zh": GAS_LICENSE_ZH,
            "license_status": "已确认（仅限研究用途，排除商业用途）",
            "is_official": False,
            "official_relation": "公开替代数据。官方 A/B 包尚未下发，映射关系未公开——未确认",
            "label_protocol_status": (
                "TODO/未确认：官方评测是否在交付当批预测前提供当批标签，"
                "尚未写进规则文档。当前流程按'当批标签可用于当批评估'实现，"
                "若官方为标签延迟，门控将使用不可用信息，需加 labels_available 开关"),
            "processed_dir": self.root,
        }

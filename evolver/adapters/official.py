"""官方数据集适配（**契约先行，尚未接入真实数据**）。

状态（务必如实表述）
--------------------
- 官方公开集：**未提供**。赛题规则写明"公开集随评测平台同步开放"（预计 11 月初），
  仓库里没有 `data/official/` 目录。
- 官方隐藏测试流：**本地永远拿不到**，只能在平台跑镜像后回传分数。
  红线：禁止获取隐藏测试集。
- 因此本模块目前只做两件事：**发现文件** + **按契约校验**。
  真实读取逻辑要等看到真实文件格式后再补。

契约（我们向数据方提出的最小要求）
----------------------------------
目录结构：

    data/official/
      manifest.json          ← 必需：声明格式、列名、批次、标签
      train.<ext>            ← manifest.files.train
      valid.<ext>            ← manifest.files.valid
      evaluate.py            ← 可选：官方评测脚本

manifest.json 示例：

    {
      "format": "csv",                 // csv | npz
      "label_column": "label",
      "feature_columns": ["f0", "f1", ...],   // 或 null = 除 label 外全部
      "batch_column": "batch_id",
      "batch_order": "ascending",      // 时间顺序必须单调
      "task_type": "classification",
      "metric": "f1_macro",
      "files": {"train": "train.csv", "valid": "valid.csv"},
      "identity": "official_public",
      "license": "未提供",
      "source": "比赛平台公开集（待下载）"
    }

没有 manifest 就**直接报错**，不做任何猜测式解析——猜错了会静默产生
错误的实验结论，比报错危险得多。
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import identities
from .sources import DataSourceError, DatasetSource

MANIFEST_NAME = "manifest.json"
SUPPORTED_FORMATS = ("csv", "npz")


class OfficialSource(DatasetSource):
    """官方数据源（公开集）。只读，不做任何写回。"""

    name = "official"
    task_type = "classification"
    data_identity = identities.OFFICIAL_PUBLIC

    def __init__(self, root: str = "data/official", *, identity: str | None = None,
                 strict: bool = True):
        self.root = root
        # 身份必须是官方公开集；传 official_test 会被拒绝（本地没有原始测试流）
        self.data_identity = identities.assert_inspectable(
            identity or identities.OFFICIAL_PUBLIC)
        self.strict = strict
        self._manifest: dict | None = None
        self._batches: list[dict] = []
        self._features: list[str] = []

    # ------------------------------------------------------------------
    def discover(self) -> dict:
        """只做发现：返回目录状态，不解析数据内容。"""
        if not os.path.isdir(self.root):
            raise DataSourceError(
                f"官方数据目录不存在：{self.root}\n"
                f"  说明：官方公开集**尚未提供**，需从比赛评测平台下载后放入该目录。\n"
                f"  本仓库不会伪造下载地址或占位数据。")
        entries = sorted(os.listdir(self.root))
        if not entries:
            raise DataSourceError(
                f"官方数据目录是空的：{self.root}\n"
                f"  至少需要：{MANIFEST_NAME} + 训练/验证数据文件。")
        manifest_path = os.path.join(self.root, MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            raise DataSourceError(
                f"缺少 {MANIFEST_NAME}（{manifest_path}）。\n"
                f"  没有清单就无法确定列名/标签/批次，本模块**拒绝猜测式解析**。\n"
                f"  现有文件：{entries}\n"
                f"  请参照 evolver/adapters/official.py 顶部注释补齐清单。")
        try:
            with open(manifest_path, encoding="utf-8-sig") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise DataSourceError(
                f"{MANIFEST_NAME} 不是合法 JSON：{exc}") from exc
        return {"entries": entries, "manifest": manifest, "path": manifest_path}

    # ------------------------------------------------------------------
    def load(self) -> "OfficialSource":
        found = self.discover()
        self._manifest = self._check_manifest(found["manifest"])
        self._check_files(self._manifest)
        self._read(self._manifest)
        return self

    def _check_manifest(self, m: dict) -> dict:
        problems = []
        fmt = str(m.get("format", "")).lower()
        if fmt not in SUPPORTED_FORMATS:
            problems.append(f"format 必须是 {SUPPORTED_FORMATS} 之一，收到 {m.get('format')!r}")
        if not m.get("label_column"):
            problems.append("缺少 label_column")
        if not isinstance(m.get("files"), dict) or not m["files"].get("train"):
            problems.append("files.train 未声明")
        bat = str(m.get("batch_order", "ascending")).lower()
        if bat not in ("ascending", "descending"):
            problems.append(f"batch_order 只能是 ascending/descending，收到 {bat!r}")
        task = str(m.get("task_type", "classification")).lower()
        if task not in ("classification", "regression"):
            problems.append(f"task_type 只能是 classification/regression，收到 {task!r}")
        if problems:
            raise DataSourceError(
                f"{MANIFEST_NAME} 字段不完整：\n  - " + "\n  - ".join(problems))
        return m

    def _check_files(self, m: dict) -> None:
        missing = []
        for key, rel in (m.get("files") or {}).items():
            if not rel:
                continue
            if not os.path.isfile(os.path.join(self.root, rel)):
                missing.append(f"{key}: {rel}")
        if missing:
            raise DataSourceError(
                f"{MANIFEST_NAME} 声明的文件不存在：\n  - " + "\n  - ".join(missing))

    # ------------------------------------------------------------------
    def _read(self, m: dict) -> None:
        fmt = str(m["format"]).lower()
        if fmt == "csv":
            self._read_csv(m)
        elif fmt == "npz":
            self._read_npz(m)
        else:                                       # 上面已校验，兜底
            raise DataSourceError(f"不支持的 format：{fmt}")

    def _read_csv(self, m: dict) -> None:
        import csv
        path = os.path.join(self.root, m["files"]["train"])
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:                    # noqa: BLE001
            raise DataSourceError(f"读取 {path} 失败：{exc}") from exc
        if not rows:
            raise DataSourceError(f"{path} 是空的")
        header = list(rows[0].keys())
        label_col = m["label_column"]
        if label_col not in header:
            raise DataSourceError(
                f"{path} 里没有标签列 {label_col!r}；实际列：{header[:12]}…")
        feats = m.get("feature_columns") or [c for c in header if c != label_col]
        missing = [c for c in feats if c not in header]
        if missing:
            raise DataSourceError(f"{path} 缺少特征列：{missing[:8]}…")
        self._features = list(feats)
        batch_col = m.get("batch_column")
        if not batch_col or batch_col not in header:
            raise DataSourceError(
                f"{MANIFEST_NAME} 未声明有效的 batch_column（收到 {batch_col!r}）；"
                f"没有批次列就无法还原漂移流。")
        groups: dict[int, list[dict]] = {}
        try:
            for r in rows:
                groups.setdefault(int(r[batch_col]), []).append(r)
        except (TypeError, ValueError) as exc:
            raise DataSourceError(
                f"列 {batch_col!r} 含有非整数批次号：{exc}") from exc
        self._assemble(groups, lambda rs, f=feats, lc=label_col: (
            np.asarray([[float(r[c]) for c in f] for r in rs], dtype=float),
            np.asarray([r[lc] for r in rs]),
        ))

    def _read_npz(self, m: dict) -> None:
        path = os.path.join(self.root, m["files"]["train"])
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:                    # noqa: BLE001
            raise DataSourceError(f"读取 {path} 失败：{exc}") from exc
        X = np.asarray(data["X"], dtype=float) if "X" in data else None
        y = np.asarray(data["y"]).ravel() if "y" in data else None
        if X is None or y is None:
            raise DataSourceError(
                f"{path} 里缺 X 或 y（实际键：{list(data.keys())}）")
        b = np.asarray(data[m["batch_column"]]).ravel() if m.get("batch_column") in data else None
        if b is None:
            raise DataSourceError(
                f"{path} 里没有批次列 {m.get('batch_column')!r}")
        self._features = list(m.get("feature_columns") or
                              [f"f{i}" for i in range(X.shape[1])])
        self._batches = []
        for bid in sorted(set(int(v) for v in b)):
            mask = b.astype(int) == bid
            self._batches.append({"batch_id": bid, "X": X[mask], "y": y[mask]})

    def _assemble(self, groups: dict[int, list[dict]], builder) -> None:
        if not groups:
            raise DataSourceError("没有解析出任何批次")
        ids = sorted(groups)
        self._batches = []
        for bid in ids:
            X, y = builder(groups[bid])
            if len(X) != len(y):
                raise DataSourceError(f"批次 {bid} 特征/标签行数不一致")
            self._batches.append({"batch_id": bid, "X": X, "y": y})
        # 时间顺序自检：批次号必须单调递增且无空洞
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise DataSourceError(f"批次号不连续：{ids}")

    # ------------------------------------------------------------------
    def _ensure(self):
        if not self._batches:
            raise DataSourceError("尚未 load()，没有批次可用")

    def split(self) -> dict:
        self._ensure()
        ids = [b["batch_id"] for b in self._batches]
        if len(ids) < 3:
            raise DataSourceError(
                f"批次太少（{len(ids)}），无法划出 train/stream/holdout")
        return {"train": [ids[0]], "stream": ids[1:-1], "holdout": [ids[-1]]}

    def iter_batches(self):
        self._ensure()
        for b in self._batches:
            yield {
                "batch_id": b["batch_id"],
                "dataset": self.name,
                "features": self._features,
                "data": b["X"],
                "X": b["X"],
                "y": b["y"],
                "metadata": {"data_identity": self.data_identity},
                "timestamp": None,
            }

    def get_features(self) -> list[str]:
        if not self._features:
            raise DataSourceError("尚未 load()，没有特征信息")
        return list(self._features)

    def get_labels(self) -> list[str]:
        self._ensure()
        return sorted({str(v) for b in self._batches for v in b["y"]})

    def get_task_type(self) -> str:
        return (self._manifest or {}).get("task_type", self.task_type)

    def get_metric(self) -> str:
        m = (self._manifest or {}).get("metric")
        if not m:
            raise DataSourceError(f"{MANIFEST_NAME} 未声明 metric")
        return str(m)

    def get_metadata(self) -> dict:
        m = self._manifest or {}
        return {
            "dataset": self.name,
            "data_identity": self.data_identity,
            "data_identity_zh": identities.describe(self.data_identity),
            "is_official": True,
            "root": self.root,
            "format": m.get("format"),
            "n_batches": len(self._batches),
            "batch_ids": [b["batch_id"] for b in self._batches],
            "n_samples": int(sum(len(b["y"]) for b in self._batches)),
            "split": self.split() if self._batches else None,
            "license": m.get("license", "未提供"),
            "source": m.get("source", "未提供"),
            "note": "官方公开集；只允许格式体检，不参与阈值/超参选择",
        }

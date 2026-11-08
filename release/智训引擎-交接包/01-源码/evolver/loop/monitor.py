"""① 漂移监测：只看数据，不看模型内部。成本≈0。"""
from __future__ import annotations

import numpy as np


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10,
         eps: float = 1e-6) -> float:
    """群体稳定性指数（PSI）。用 expected 的分位数切箱。"""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    e = np.histogram(expected, bins=edges)[0] / expected.size
    a = np.histogram(actual, bins=edges)[0] / actual.size
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


class Monitor:
    """以训练批次为参照，逐批计算特征分布偏移。"""

    def __init__(self, reference_X: np.ndarray, bins: int = 10):
        self.reference_X = np.asarray(reference_X, dtype=float)
        self.bins = bins
        self.history: list[dict] = []

    def reset_reference(self, X: np.ndarray):
        """模型更新上线后，把参照系挪到“当前已知良好状态”。

        否则参照系永远钉在部署那天，PSI 会一直虚高，
        系统会反复做无用更新——这正是成本失控的常见原因。
        """
        self.reference_X = np.asarray(X, dtype=float)

    def check(self, X: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        per_feature = [
            _psi(self.reference_X[:, j], X[:, j], self.bins)
            for j in range(self.reference_X.shape[1])
        ]
        psi = np.array(per_feature)
        report = {
            "psi_max": float(psi.max()) if psi.size else 0.0,
            "psi_mean": float(psi.mean()) if psi.size else 0.0,
            "psi_p90": float(np.quantile(psi, 0.9)) if psi.size else 0.0,
            "worst_feature": int(psi.argmax()) if psi.size else -1,
            "per_feature": [float(v) for v in psi],
        }
        self.history.append(report)
        return report


def label_psi(ref_y: np.ndarray, cur_y: np.ndarray, eps: float = 1e-6) -> float:
    """标签分布偏移（真实场景拿不到标签时，用伪标签或置信度分布替代）。"""
    ref_y = np.asarray(ref_y).ravel()
    cur_y = np.asarray(cur_y).ravel()
    classes = np.union1d(ref_y, cur_y)
    e = np.array([(ref_y == c).mean() for c in classes]) + eps
    a = np.array([(cur_y == c).mean() for c in classes]) + eps
    e /= e.sum()
    a /= a.sum()
    return float(np.sum((a - e) * np.log(a / e)))

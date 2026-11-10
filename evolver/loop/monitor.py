"""① 漂移监测：只看数据，不看模型内部。成本≈0。

【整改 B1】原来只有 PSI。PSI 有三个已知弱点：
  - 对分箱数敏感，样本少时方差大
  - 只测边缘分布，测不到相关结构变化
  - 依赖参照批的分位切分，量纲不可跨数据集比较

本轮补两种主流方法（纯 numpy，不引入 scipy）：

  PSI          Population Stability Index，对称化 KL 的离散近似
               适合：工业现场最常用、可解释、对"整体平移/缩放"敏感
  KS           Kolmogorov–Smirnov 两样本统计量（ECDF 最大距离）
               适合：无需分箱、对分布形状变化敏感、值域固定在 [0,1]，
                     比 PSI 更容易跨数据集设阈值
  Wasserstein  一维 Wasserstein-1 距离（按分位点积分），已按参照批标准差归一化
               适合：对"整体平移幅度"给出有物理意义的距离（单位=标准差），
                     三种方法里最直观

默认 detector="psi"，旧行为完全不变。
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────
# 三种分布距离
# ─────────────────────────────────────────────────────────────
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


def _ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """两样本 KS 统计量：两条经验分布函数的最大距离，值域 [0,1]。"""
    a = np.sort(np.asarray(expected, dtype=float).ravel())
    b = np.sort(np.asarray(actual, dtype=float).ravel())
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein(expected: np.ndarray, actual: np.ndarray,
                 n_grid: int = 200) -> float:
    """一维 Wasserstein-1 距离，按参照批标准差归一化（单位=标准差）。

    不做分箱、不假设分布，纯按分位点积分：W1 = ∫|F_a⁻¹(q) − F_b⁻¹(q)|dq。
    归一化后跨特征可比，这是它比 PSI 更适合做"统一阈值"的原因。
    """
    a = np.asarray(expected, dtype=float).ravel()
    b = np.asarray(actual, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    q = np.linspace(0.0, 1.0, n_grid)
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    scale = float(a.std()) or 1.0
    return float(np.mean(np.abs(qa - qb)) / scale)


DISTANCES = {"psi": _psi, "ks": _ks, "wasserstein": _wasserstein}


# ─────────────────────────────────────────────────────────────
# 监测器
# ─────────────────────────────────────────────────────────────
class Monitor:
    """以训练批次为参照，逐批计算特征分布偏移。

    detector:
        "psi"          只算 PSI（旧行为，默认）
        "ks"           只算 KS
        "wasserstein"  只算 W1
        "all"          三种都算（便于对比与标定）
    """

    def __init__(self, reference_X: np.ndarray, bins: int = 10,
                 detector: str = "psi"):
        if detector not in ("psi", "ks", "wasserstein", "all"):
            raise ValueError(f"未知 detector '{detector}'")
        self.reference_X = np.asarray(reference_X, dtype=float)
        self.bins = bins
        self.detector = detector
        self.history: list[dict] = []

    def reset_reference(self, X: np.ndarray):
        """模型更新上线后，把参照系挪到"当前已知良好状态"。

        否则参照系永远钉在部署那天，PSI 会一直虚高，
        系统会反复做无用更新——这正是成本失控的常见原因。
        """
        self.reference_X = np.asarray(X, dtype=float)

    def _one(self, name: str, ref_j, cur_j) -> float:
        if name == "psi":
            return _psi(ref_j, cur_j, self.bins)
        if name == "ks":
            return _ks(ref_j, cur_j)
        return _wasserstein(ref_j, cur_j)

    def check(self, X: np.ndarray, threshold: float = 0.25) -> dict:
        X = np.asarray(X, dtype=float)
        names = (["psi", "ks", "wasserstein"] if self.detector == "all"
                 else [self.detector])

        per_feature: dict[str, list[float]] = {}
        for name in names:
            per_feature[name] = [
                self._one(name, self.reference_X[:, j], X[:, j])
                for j in range(self.reference_X.shape[1])
            ]

        primary = per_feature[names[0]]
        arr = np.array(primary) if primary else np.array([0.0])
        report = {
            # 这三个键名保持原样：diagnose.py 依赖它们
            "psi_max": float(arr.max()) if arr.size else 0.0,
            "psi_mean": float(arr.mean()) if arr.size else 0.0,
            "psi_p90": float(np.quantile(arr, 0.9)) if arr.size else 0.0,
            # 有多少比例的特征超标 —— 区分「少数几个坏了」和「整体都坏了」
            "psi_ratio": float((arr >= threshold).mean()) if arr.size else 0.0,
            "worst_feature": int(arr.argmax()) if arr.size else -1,
            "per_feature": [float(v) for v in primary],
            "detector": names[0],
        }
        for name, vals in per_feature.items():
            if not vals:
                continue
            a = np.array(vals)
            report[f"{name}_max"] = float(a.max())
            report[f"{name}_mean"] = float(a.mean())
            report[f"{name}_per_feature"] = [float(v) for v in vals]

        self.history.append(report)
        return report


# ─────────────────────────────────────────────────────────────
# 标签漂移（分类 / 回归通用）
# ─────────────────────────────────────────────────────────────
def label_psi(ref_y: np.ndarray, cur_y: np.ndarray, eps: float = 1e-6) -> float:
    """标签分布偏移（分类任务：对类别频率算 PSI）。

    【整改 A4】原实现对连续标签会失效：np.union1d 之后每个"类别"频次≈1/n，
    算出来的 PSI 恒接近 0，等于什么都没测。回归任务请用 label_drift()。
    """
    ref_y = np.asarray(ref_y).ravel()
    cur_y = np.asarray(cur_y).ravel()
    classes = np.union1d(ref_y, cur_y)
    e = np.array([(ref_y == c).mean() for c in classes]) + eps
    a = np.array([(cur_y == c).mean() for c in classes]) + eps
    e /= e.sum()
    a /= a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def label_drift(ref_y: np.ndarray, cur_y: np.ndarray,
                task_type: str = "classification") -> float:
    """按任务类型分派标签漂移度量。

    分类：类别频率的 PSI（量纲无界，经验阈值 0.1）
    回归：连续标签的 KS 统计量（值域 [0,1]，建议阈值 0.10~0.15）

    注意：回归用 KS 是本轮整改的取舍，不是标准做法。
    更严谨的做法是对"给定 X 的条件分布"做检验，需要额外建模。
    """
    ref_y = np.asarray(ref_y).ravel()
    cur_y = np.asarray(cur_y).ravel()
    if task_type == "regression":
        return _ks(ref_y, cur_y)
    return label_psi(ref_y, cur_y)


def label_drift_from_proba(ref_proba: np.ndarray, cur_proba: np.ndarray,
                           bins: int = 10) -> float:
    """无标签场景下的替代信号：用最大置信度的分布偏移近似标签漂移。

    真实工业场景标签常常延迟到不了，这时用模型置信度分布变化当代理指标。
    这是**近似**，不是标签漂移本身，答辩时要说清楚。
    """
    ref_conf = np.asarray(ref_proba, dtype=float).max(axis=1)
    cur_conf = np.asarray(cur_proba, dtype=float).max(axis=1)
    return _psi(ref_conf, cur_conf, bins)

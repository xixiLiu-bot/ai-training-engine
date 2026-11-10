"""模型适配器：统一 fit / predict / save / load，并提供不同价位的更新手段。"""
from __future__ import annotations

import copy
import os

import numpy as np


def _make_estimator(kind: str, random_state: int):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=random_state)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=80, random_state=random_state,
                                      n_jobs=-1)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        # 刻意做小：4 核 8G 无 GPU，且镜像有 2G 限制
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                             early_stopping=True, random_state=random_state)
    raise ValueError(f"未知模型 '{kind}'，可选：logreg / gbdt / rf / mlp")


class SklearnModelAdapter:
    """把 sklearn 模型包装成闭环能调的统一对象。

    刻意把「标准化」和「分类器」拆开存：
    这样“只重算均值方差、不动模型权重”这种**零成本**更新才做得到。
    """

    def __init__(self, kind: str = "logreg", random_state: int = 0):
        from sklearn.preprocessing import StandardScaler
        self.kind = kind
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.est = _make_estimator(kind, random_state)
        self.n_features_in_: int | None = None

    # ---- 训练 ----
    def fit(self, X, y, sample_weight=None):
        self.n_features_in_ = X.shape[1]
        self.scaler.fit(X, sample_weight=sample_weight)
        Xs = self.scaler.transform(X)
        if sample_weight is not None:
            self.est.fit(Xs, y, sample_weight=sample_weight)
        else:
            self.est.fit(Xs, y)
        return self

    # ---- 推理 ----
    def predict(self, X):
        return self.est.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        if hasattr(self.est, "predict_proba"):
            return self.est.predict_proba(self.scaler.transform(X))
        raise AttributeError(f"{self.kind} 不支持 predict_proba")

    # ---- 续训（warm start）：把“之前学到的”带到下一轮 ----
    def fit_continue(self, X, y, sample_weight=None, refresh_scaler: bool = False):
        """在当前权重的基础上继续训练，而不是从零重来。

        这就是“训练后的经验可以用到之后的训练”：
        模型本身就是经验的载体，重建等于把学过的全丢了。

        注意：默认**不刷新标准化**——因为权重是配着旧尺度训出来的，
        换了尺子权重就失配。需要换尺度时，应该先 refresh_normalization 再续训。
        """
        if refresh_scaler:
            self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        est = self.est
        if hasattr(est, "warm_start"):
            try:
                est.set_params(warm_start=True)
            except Exception:
                pass

        if sample_weight is not None:
            try:
                est.fit(Xs, y, sample_weight=sample_weight)
            except TypeError:
                est.fit(Xs, y)
        else:
            est.fit(Xs, y)
        return self

    # ---- 便宜的更新手段 ----
    def refresh_normalization(self, X):
        """只重算标准化统计量（均值/方差），模型权重不动。成本≈0。"""
        self.scaler.fit(X)
        return self

    def is_fitted(self) -> bool:
        return hasattr(self.est, "classes_")

    def clone(self):
        return copy.deepcopy(self)

    # ---- 持久化 ----
    def save(self, path: str):
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        import joblib
        return joblib.load(path)

    def __repr__(self):
        return f"<SklearnModelAdapter {self.kind}>"


def build(kind: str = "logreg", **kwargs) -> SklearnModelAdapter:
    return SklearnModelAdapter(kind=kind, **kwargs)


# ─────────────────────────────────────────────────────────────
# 动作清单：每个模型自己声明「我能做哪些更新手段、各花多少秒」
#
# 这里只声明**能力**（能不能做、多少钱），不声明**效用**（做了有没有用）。
# 为什么这么分：
#
#   · 能不能做 —— 由模型本身决定。树模型技术上也能重算标准化，只是快慢不同。
#   · 有没有用 —— 由数据决定，应该交给经验表去学，而不是在这里写死。
#
# 曾经试过「把对树有害的校准直接从清单里删掉」，结果反而更差：
#   带校准 0.478 → 删掉校准 0.437
# 因为它不只是"修"，还是"探"——先花 0.05 秒试一下，不行下一批再升级。
# 删掉它，系统就变得急躁，每次都直接上最贵的重训。
#
# 所以：清单管成本，经验表管取舍。
# ─────────────────────────────────────────────────────────────
ACTION_CATALOGS: dict[str, dict[str, float]] = {
    # 线性模型：重训很快
    "logreg": {"do_nothing": 0.0, "refresh_normalization": 0.05, "refit_model": 0.5},
    # 梯度提升树：重训很慢（100 棵树）
    "gbdt": {"do_nothing": 0.0, "refresh_normalization": 0.05, "refit_model": 2.2},
    # 随机森林：中等
    "rf": {"do_nothing": 0.0, "refresh_normalization": 0.05, "refit_model": 1.5},
    # 多层感知机：重训中等，但标准化是网络输入的一部分
    "mlp": {"do_nothing": 0.0, "refresh_normalization": 0.05, "refit_model": 1.2},
}


def action_catalog(kind: str) -> dict[str, float]:
    """返回这个模型支持的动作及成本（秒）。"""
    if kind not in ACTION_CATALOGS:
        raise ValueError(
            f"未知模型 '{kind}'，可选：{list(ACTION_CATALOGS)}")
    return dict(ACTION_CATALOGS[kind])


__all__ = ["SklearnModelAdapter", "build", "action_catalog", "ACTION_CATALOGS", "np"]

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
    raise ValueError(f"未知模型 '{kind}'，可选：logreg / gbdt / rf")


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


__all__ = ["SklearnModelAdapter", "build", "np"]

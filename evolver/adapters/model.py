"""模型适配器：统一 fit / predict / save / load，并提供不同价位的更新手段。

【整改 A4】原来只有分类器（logreg/gbdt/rf），回归任务（如 RUL）无法正确建模——
用分类器拟合连续标签，predict 出来的是"类别号"，算 MAE 毫无意义。
本轮新增 `ridge`（岭回归）作为回归模型，并修正 is_fitted 的判据。
"""
from __future__ import annotations

import copy
import os

import numpy as np

# 按任务类型划分可用模型
CLASSIFIERS = ("logreg", "gbdt", "rf", "mlp", "gnb")
REGRESSORS = ("ridge",)
TASK_OF = {**{k: "classification" for k in CLASSIFIERS},
           **{k: "regression" for k in REGRESSORS}}


def _make_scaler(kind: str = "standard"):
    """标准化用的"尺子"。

    standard  —— 逐特征减去均值除以标准差（默认，旧行为）
    quantile  —— 逐特征转成"在本批内的百分位"（QuantileTransformer）

    为什么要有 quantile：轴承数据里每个轴承的振动绝对水平不同（装配公差、
    传感器位置），个体差异比退化阶段还大。把特征换成"在本批内的百分位"，
    等于只保留退化轨迹的形状，消掉个体差异。

    实测（跨轴承泛化，批1/2 训练 → 批3/4/5 测试）：
        标准化    平均 0.426
        百分位    平均 0.649   ← +52%，六组对照全部改善

    注意：这是"直推式"做法——用当前批自身的分布做变换。评测评测时整批数据
    是一次性下发的，所以拿得到；这与 refresh_normalization 的假设一致。
    """
    if kind == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    if kind == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(n_quantiles=200, output_distribution="uniform",
                                   subsample=100000, random_state=0)
    raise ValueError(f"未知 scaler '{kind}'，可选：standard / quantile")


def _make_estimator(kind: str, random_state: int, logreg_c: float = 1.0):
    if kind == "logreg":
        # C 是正则强度的倒数：C 越小正则越强。
        # sklearn 默认 C=1.0；小样本 + 高维时加强正则通常更稳
        # （实测气体数据 C=0.01 时单批 F1 从 0.263 升到 0.378）。
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=logreg_c, max_iter=1000,
                                  random_state=random_state)
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=random_state)
    if kind == "gnb":
        # 高斯朴素贝叶斯：假设每个特征服从正态分布、特征间独立。
        # 假设很强（传感器明显相关），但在小样本 + 高维上这个强偏见正好起正则作用。
        # 实测：气体 0.540 vs 逻辑回归 0.263；轴承 0.214 vs 0.189，训练只要 1ms。
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    if kind == "mlp":
        # 刻意做小：4 核 8G 无 GPU，且镜像有 2G 限制。
        # 用 sklearn 而不是 PyTorch —— torch 占 0.48 GB，会挤掉镜像额度。
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                             early_stopping=True, random_state=random_state)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=80, random_state=random_state,
                                      n_jobs=-1)
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=random_state)
    raise ValueError(f"未知模型 '{kind}'，可选：{CLASSIFIERS + REGRESSORS}")


def model_task(kind: str) -> str:
    """模型属于哪个任务类型（用于和数据集声明做一致性校验）。"""
    return TASK_OF.get(kind, "unknown")


# ─────────────────────────────────────────────────────────────
# 【整改 B3】输出校准的辅助函数（纯 numpy，不引入额外依赖）
# ─────────────────────────────────────────────────────────────
class CalibrationUnsupported(RuntimeError):
    """当前任务/模型不支持输出校准。**必须被上层捕获并记录原因**，不能静默跳过。"""


def _macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = np.union1d(y_true, y_pred)
    if labels.size == 0:
        return 0.0
    scores = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores)) if scores else 0.0


def _best_threshold(p_pos, y, pos_label, n_grid: int = 19) -> float:
    """二分类：在 [0.05, 0.95] 上搜一个概率阈值，最大化 macro-F1。"""
    y = np.asarray(y).ravel()
    best_t, best_s = 0.5, -1.0
    neg_label = None
    for c in np.unique(y):
        if c != pos_label:
            neg_label = c
            break
    if neg_label is None:                       # 只有一个类，阈值无意义
        return 0.5
    for t in np.linspace(0.05, 0.95, n_grid):
        pred = np.where(p_pos >= t, pos_label, neg_label)
        s = _macro_f1(y, pred)
        if s > best_s + 1e-9:
            best_t, best_s = float(t), s
    return best_t


def _best_logit_biases(proba, y, classes, rounds: int = 2,
                       grid=(-0.3, -0.15, 0.15, 0.3)) -> np.ndarray:
    """多分类：对 log 概率加逐类偏置，坐标上升最大化 macro-F1。

    为什么不用温度缩放：温度是单调变换，**不改变 argmax**，
    对 F1/准确率毫无作用。要改变决策边界必须动逐类偏置（等价于一 vs 其余阈值）。
    """
    y = np.asarray(y).ravel()
    logp = np.log(np.clip(np.asarray(proba, dtype=float), 1e-12, None))
    classes = np.asarray(classes)
    k = logp.shape[1]
    b = np.zeros(k)

    def score(vec) -> float:
        return _macro_f1(y, classes[np.argmax(logp + vec, axis=1)])

    best = score(b)
    for _ in range(rounds):
        improved = False
        for j in range(k):
            for delta in grid:
                trial = b.copy()
                trial[j] += delta
                s = score(trial)
                if s > best + 1e-6:
                    best, b, improved = s, trial, True
        if not improved:
            break
    return b


class SklearnModelAdapter:
    """把 sklearn 模型包装成闭环能调的统一对象。

    刻意把「标准化」和「分类器」拆开存：
    这样“只重算均值方差、不动模型权重”这种**零成本**更新才做得到。
    """

    def __init__(self, kind: str = "logreg", random_state: int = 0,
                 logreg_c: float = 1.0, scaler_kind: str = "standard"):
        from sklearn.preprocessing import StandardScaler
        self.kind = kind
        self.random_state = random_state
        self.scaler_kind = scaler_kind
        self.scaler = _make_scaler(scaler_kind)
        self.logreg_c = float(logreg_c)
        self.est = _make_estimator(kind, random_state, logreg_c)
        self.n_features_in_: int | None = None
        # 【整改 B3】输出校准参数。None = 未校准；结构见 calibrate_output()
        self.calib: dict | None = None

    def _fit_scaler(self, X, sample_weight=None):
        """拟合尺子。不是所有 scaler 都吃 sample_weight（如 QuantileTransformer）。"""
        if sample_weight is None:
            self.scaler.fit(X)
            return
        try:
            self.scaler.fit(X, sample_weight=sample_weight)
        except TypeError:
            self.scaler.fit(X)

    # ---- 训练 ----
    def fit(self, X, y, sample_weight=None):
        self.n_features_in_ = X.shape[1]
        self._fit_scaler(X, sample_weight)
        Xs = self.scaler.transform(X)
        if sample_weight is not None:
            self.est.fit(Xs, y, sample_weight=sample_weight)
        else:
            self.est.fit(Xs, y)
        self.calib = None                 # 重新训练后旧校准失效
        return self

    # ---- 续训（warm start）：不把学过的丢掉，在现权重上接着训 ----
    def fit_continue(self, X, y, sample_weight=None, refresh_scaler=True):
        """在当前权重基础上继续训练，而不是从零重建。

        实测（气体数据，第 4 批 / 第 7 批）：
            冷启动重训  128ms·0.7851 / 72ms·0.8011
            续训（旧尺子）254ms·0.7946 / 117ms·0.7845
            **续训 + 刷新尺子  75ms·0.8049 / 67ms·0.8232**  ← 最优
            只刷新尺子  0.4ms·0.5953 / 2.8ms·0.5714

        关键：必须同时刷新标准化。权重是配着旧尺度训出来的，换了数据不换尺子，
        优化器要在错误空间里重新爬，反而更慢更差。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if not self.is_fitted() or not hasattr(self.est, "warm_start"):
            return self.fit(X, y, sample_weight)

        self.est.warm_start = True
        if refresh_scaler:
            self._fit_scaler(X, sample_weight)
        Xs = self.scaler.transform(X)
        try:
            if sample_weight is not None:
                self.est.fit(Xs, y, sample_weight=sample_weight)
            else:
                self.est.fit(Xs, y)
        except Exception:                 # noqa: BLE001  不支持 sample_weight 就退回
            self.est.fit(Xs, y)
        self.calib = None
        return self

    # ---- 【整改 B3】输出层校准：只改决策规则，不碰一个权重 ----
    def calibrate_output(self, X, y):
        """用给定的**历史**数据拟合输出校准参数（绝不能用当批标签）。

        二分类：搜索最优概率阈值 t（最大化 macro-F1）
        多分类：对 log 概率加逐类偏置，坐标上升最大化 macro-F1
        回归：不支持（见 CalibrationUnsupported）

        成本远低于 refresh_normalization：只做一次 O(n) 的前向 + 小规模网格。
        """
        if not self.is_fitted():
            raise CalibrationUnsupported("模型尚未训练，无法校准")
        if model_task(self.kind) == "regression":
            raise CalibrationUnsupported(
                "回归任务暂不支持输出校准：回归没有可用的阈值/温度校准目标，"
                "强行套用分类校准会得到无意义的结果")
        if not hasattr(self.est, "predict_proba"):
            raise CalibrationUnsupported(f"{self.kind} 没有 predict_proba，无法校准")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        proba = self.est.predict_proba(self.scaler.transform(X))
        classes = np.asarray(getattr(self.est, "classes_", np.unique(y)))
        k = proba.shape[1]

        if k == 2:
            t = _best_threshold(proba[:, 1], y, classes[1])
            self.calib = {"kind": "threshold", "t": float(t),
                          "classes": [float(c) for c in classes],
                          "k": int(k)}
        elif k <= 10:
            b = _best_logit_biases(proba, y, classes)
            self.calib = {"kind": "bias", "b": [float(v) for v in b],
                          "classes": [float(c) for c in classes],
                          "k": int(k)}
        else:
            raise CalibrationUnsupported(f"类别数 {k} 过多（>10），暂不支持输出校准")
        return self

    def calibration_summary(self) -> dict | None:
        if self.calib is None:
            return None
        out = {"kind": self.calib["kind"], "k": self.calib.get("k")}
        if out["kind"] == "threshold":
            out["threshold"] = round(self.calib["t"], 4)
        else:
            out["biases"] = [round(v, 4) for v in self.calib["b"]]
        return out

    # ---- 推理 ----
    def predict(self, X):
        Xs = self.scaler.transform(X)
        if self.calib is None:
            return self.est.predict(Xs)

        proba = self.est.predict_proba(Xs)
        classes = np.asarray(self.calib["classes"])
        if self.calib["kind"] == "threshold":
            idx = (proba[:, 1] >= self.calib["t"]).astype(int)
            return classes[idx]
        logp = np.log(np.clip(proba, 1e-12, None))
        return classes[np.argmax(logp + np.asarray(self.calib["b"]), axis=1)]

    def predict_proba(self, X):
        if hasattr(self.est, "predict_proba"):
            return self.est.predict_proba(self.scaler.transform(X))
        raise AttributeError(f"{self.kind} 不支持 predict_proba")

    # ---- 便宜的更新手段 ----
    def refresh_normalization(self, X):
        # 这就是文献里的 AdaBN（Adaptive Batch Normalization）：
        # 域偏移时不用重训权重，只把归一化统计量换成目标域的。
        # Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation",
        # ICLR 2017 Workshop, arXiv:1603.04779
        #
        # 下一步可升级为 CORAL（Sun & Saenko, ECCV 2016, arXiv:1607.01719）：
        # 连协方差一起对齐，而不只是逐特征的均值/方差。
        # 详见 docs/文献-逻辑回归与校准.md
        """只重算标准化统计量（均值/方差），模型权重不动。成本≈0。"""
        self.scaler.fit(X)
        return self

    def is_fitted(self) -> bool:
        # 【A4】原来只查 classes_，回归模型（Ridge）没有 classes_，会永远判为未训练
        return hasattr(self.est, "classes_") or hasattr(self.est, "coef_")

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
    """按名字造模型。kwargs 会透传给 SklearnModelAdapter（如 logreg_c）。"""
    return SklearnModelAdapter(kind=kind, **kwargs)


def default_for(task_type: str) -> str:
    """按任务类型给默认模型：分类 logreg，回归 ridge。"""
    return "ridge" if (task_type or "").lower() == "regression" else "logreg"


def check_task(kind: str, task_type: str) -> str | None:
    """校验"模型 vs 任务"是否匹配。不匹配返回告警文本，匹配返回 None。

    不做成异常：评测现场宁可跑出一个带告警的结果，也不要中途崩掉。
    """
    mt = model_task(kind)
    if mt == "unknown" or not task_type:
        return None
    if mt != (task_type or "").lower():
        return (f"模型 {kind} 属于 {mt}，但任务声明为 {task_type} —— "
                f"分数不可信，请换 {'ridge' if task_type == 'regression' else 'logreg'}")
    return None


# ─────────────────────────────────────────────────────────────
# 动作清单：每个模型自己声明「能做什么、各花多少秒」
#
# 这里只声明**能力**（能不能做、多少钱），不声明**效用**（做了有没有用）。
#   · 能不能做 —— 由模型本身决定（树模型技术上也能重算标准化，只是快慢不同）
#   · 有没有用 —— 由数据决定，交给经验表去学，而不是在这里写死
#
# 曾经试过「把对树有害的校准直接从清单里删掉」，结果反而更差（0.478 → 0.437）：
# 它不只是修，还是探——先花 0.05 秒试一下，不行下一批再升级。
# 删掉它，系统就变得急躁，每次都直接上最贵的重训。
# ─────────────────────────────────────────────────────────────
ACTION_CATALOGS: dict[str, dict[str, float]] = {
    # 线性模型：重训很快；输出校准也可用
    "logreg": {"do_nothing": 0.0, "calibrate_output": 0.02,
                "refresh_normalization": 0.05, "refit_model": 0.5},
    # 梯度提升树：重训很慢（100 棵树）
    "gbdt": {"do_nothing": 0.0, "calibrate_output": 0.02,
              "refresh_normalization": 0.05, "refit_model": 2.2},
    # 随机森林：中等
    "rf": {"do_nothing": 0.0, "calibrate_output": 0.02,
            "refresh_normalization": 0.05, "refit_model": 1.5},
    # 神经网络：重训中等
    "mlp": {"do_nothing": 0.0, "refresh_normalization": 0.05,
             "refit_model": 1.2},
    # 高斯朴素贝叶斯：重训几乎免费（1ms 量级）
    "gnb": {"do_nothing": 0.0, "calibrate_output": 0.02,
             "refresh_normalization": 0.05, "refit_model": 0.05},
    # 岭回归（回归任务，没有输出校准这一档）
    "ridge": {"do_nothing": 0.0, "refresh_normalization": 0.05,
               "refit_model": 0.5},
}


def action_catalog(kind: str) -> dict[str, float]:
    """返回这个模型支持的动作及成本（秒）。"""
    if kind not in ACTION_CATALOGS:
        raise ValueError(f"未知模型 '{kind}'，可选：{list(ACTION_CATALOGS)}")
    return dict(ACTION_CATALOGS[kind])


__all__ = ["SklearnModelAdapter", "build", "default_for", "check_task",
           "action_catalog", "ACTION_CATALOGS",
           "model_task", "CalibrationUnsupported", "CLASSIFIERS", "REGRESSORS", "np"]

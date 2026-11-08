"""指标适配器：闭环只认 score()，不关心具体算的是哪个指标。"""
from __future__ import annotations


class MetricAdapter:
    name = "base"

    def score(self, y_true, y_pred) -> float:  # pragma: no cover
        raise NotImplementedError


class F1Macro(MetricAdapter):
    name = "f1_macro"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


class Accuracy(MetricAdapter):
    name = "accuracy"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y_true, y_pred))


class MAE(MetricAdapter):
    """回归任务用（官方若是回归指标，换成这个）。"""
    name = "mae"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import mean_absolute_error
        return float(-mean_absolute_error(y_true, y_pred))   # 越大越好


def build(kind: str = "f1_macro") -> MetricAdapter:
    table = {"f1_macro": F1Macro, "accuracy": Accuracy, "mae": MAE}
    if kind not in table:
        raise ValueError(f"未知指标 '{kind}'，可选：{list(table)}")
    return table[kind]()

"""指标适配器：闭环只认 score()，不关心具体算的是哪个指标。

【整改 A4】主流程原来写死 `f1_macro`，接 RUL（回归）会把 RMSE/MAE 当分类指标算。

【整改 R1】新增 `direction`（higher/lower）与 `raw_score()`，为门控的
margin 归一化提供量纲信息。约定如下，改指标时必须一起遵守：

    score()      返回 **utility**：一律"越大越好"（回归指标取负号）
    raw_score()  返回 **人类可读的原始值**（如 MAE 为正的天数）
    direction    "higher"（f1/accuracy）| "lower"（mae/rmse）

    utility = raw        （direction == "higher"）
    utility = -raw       （direction == "lower"）

为什么内部坚持用 utility：`decide.py` 的 `best = max(best, score)`、
门控的 `candidate - reference`、决策阈值全都假设"越大越好"。
把方向信息放在 direction 里，而不是让每个调用点自己取反，才不会双重取反。
"""
from __future__ import annotations


class MetricAdapter:
    name = "base"
    task_type = "classification"
    direction = "higher"

    def score(self, y_true, y_pred) -> float:  # pragma: no cover
        raise NotImplementedError

    def raw_score(self, y_true, y_pred) -> float:
        """人类可读的原始指标值。默认与 utility 相同。"""
        return self.score(y_true, y_pred)


class F1Macro(MetricAdapter):
    name = "f1_macro"
    task_type = "classification"
    direction = "higher"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


class Accuracy(MetricAdapter):
    name = "accuracy"
    task_type = "classification"
    direction = "higher"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y_true, y_pred))


class MAE(MetricAdapter):
    """回归任务用。score 取负号（越大越好），raw_score 返回正的 MAE。"""
    name = "mae"
    task_type = "regression"
    direction = "lower"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import mean_absolute_error
        return float(-mean_absolute_error(y_true, y_pred))

    def raw_score(self, y_true, y_pred) -> float:
        from sklearn.metrics import mean_absolute_error
        return float(mean_absolute_error(y_true, y_pred))


class RMSE(MetricAdapter):
    """回归任务用。score 取负号，raw_score 返回正的 RMSE。"""
    name = "rmse"
    task_type = "regression"
    direction = "lower"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import mean_squared_error
        return float(-(mean_squared_error(y_true, y_pred) ** 0.5))

    def raw_score(self, y_true, y_pred) -> float:
        from sklearn.metrics import mean_squared_error
        return float(mean_squared_error(y_true, y_pred) ** 0.5)


_TABLE = {"f1_macro": F1Macro, "accuracy": Accuracy, "mae": MAE, "rmse": RMSE}

# 任务类型 → 默认指标
_DEFAULT_BY_TASK = {
    "classification": "f1_macro",
    "regression": "mae",
}


def build(kind: str = "f1_macro") -> MetricAdapter:
    if kind not in _TABLE:
        raise ValueError(f"未知指标 '{kind}'，可选：{list(_TABLE)}")
    return _TABLE[kind]()


def build_for(task_type: str, metric_name: str | None = None) -> MetricAdapter:
    """按任务类型分派指标。显式给了 metric_name 就以它为准（并校验一致性）。"""
    task = (task_type or "classification").lower()
    if metric_name:
        metric = build(metric_name)
        if metric.task_type != task:
            raise ValueError(
                f"指标 {metric_name} 属于 {metric.task_type} 任务，"
                f"但数据集声明为 {task} —— 这会把分数算错")
        return metric
    if task not in _DEFAULT_BY_TASK:
        raise ValueError(f"未知任务类型 '{task}'，可选：{list(_DEFAULT_BY_TASK)}")
    return build(_DEFAULT_BY_TASK[task])

"""② 归因诊断：判断“为什么变差”，输出该做什么，而不是直接重训。"""
from __future__ import annotations


def diagnose(monitor_report: dict, label_shift: float, cfg) -> dict:
    """区分协变量漂移与概念漂移。

    协变量漂移：输入变了，但“什么算什么”的规则没变 → 便宜的校准通常能救。
    概念漂移：输入和标签的关系变了 → 往往必须动模型。

    用 mean PSI 判定（比 max 稳，不会被单个小方差特征带偏）。
    真实工业场景拿不到标签，label_shift 要用伪标签分布 / 置信度分布替代。
    """
    psi_x = monitor_report["psi_mean"]

    if psi_x < cfg.psi_low:
        kind, why = "stable", "特征分布无明显偏移"
    elif label_shift >= cfg.label_psi_th:
        kind, why = "concept", "特征与标签同时偏移，疑似输入-输出关系改变"
    else:
        kind, why = "covariate", "特征偏移但标签分布稳定，疑似输入分布改变"

    return {
        "kind": kind,
        "psi_mean": psi_x,
        "psi_max": monitor_report["psi_max"],
        "label_shift": label_shift,
        "reason": why,
        "suggest": {
            "stable": "do_nothing",
            "covariate": "refresh_normalization",
            "concept": "refit_model",
        }[kind],
    }

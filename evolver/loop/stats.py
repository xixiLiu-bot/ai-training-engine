# -*- coding: utf-8 -*-
"""配对显著性检验：McNemar。

为什么需要它：
    门控现在的判据是「候选分数 − 当前分数 ≥ 0.02」。但分数本身是**估计值**。
    轴承 b9 那一批只有 42 条（每类 14 条），macro-F1 的估计标准差约 0.095，
    是 0.02 门槛的 4~5 倍 —— 在那样的批次上，"上线还是回滚"等于抛硬币。

    文献依据：Dietterich, "Approximate Statistical Tests for Comparing
    Supervised Classification Learning Algorithms", Neural Computation 1998.
    结论是：在小测试集上比较两个分类器，原始差值没有意义，必须做统计检验。

    McNemar 是这里最合适的检验：同一批样本上，两个模型"一个对一个错"的
    样本数 b 和 c，在原假设下服从 Binomial(b+c, 0.5)。

    它比"把阈值调大"更有原则：样本多时自动放行小改进，样本少时自动要求
    更强的证据，不需要人按数据集调参。

与门控的关系（两条都要满足）：
    ① margin：改动要有**实际意义**的增益（成本效益）
    ② 显著性：这个增益不能是**噪声**（统计意义）
    只看①会在小批次上被噪声骗；只看②会把"统计显著但经济上不值"的改动放进来。
"""
from __future__ import annotations

import numpy as np


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """配对 McNemar 检验：判断 a 是否显著优于 b。

    参数：
        y_true  : 真实标签
        pred_a  : 候选模型的预测
        pred_b  : 现役模型的预测

    返回：
        {"b": a对b错的条数, "c": a错b对的条数, "n": 分歧数,
         "p_value": 双侧 p 值, "significant": 是否显著, "note": 人类可读说明}

    说明：
        · n == 0（两个模型预测完全一致）→ p=1，不显著。
          这是对的：没有任何证据说明换掉会更好。
        · 用精确二项分布而不是卡方近似：分歧数少时近似不可靠，
          而小批次恰恰是分歧数少的场合。
    """
    y = np.asarray(y_true).ravel()
    a = np.asarray(pred_a).ravel()
    b_ = np.asarray(pred_b).ravel()
    if not (len(y) == len(a) == len(b_)):
        raise ValueError("y_true / pred_a / pred_b 长度必须一致")

    a_ok = a == y
    b_ok = b_ == y
    b_cnt = int(np.sum(a_ok & ~b_ok))     # 候选对、现役错
    c_cnt = int(np.sum(~a_ok & b_ok))     # 候选错、现役对
    n = b_cnt + c_cnt

    if n == 0:
        return {"b": 0, "c": 0, "n": 0, "p_value": 1.0,
                "significant": False,
                "note": "两模型预测完全一致，无证据支持更换"}

    try:
        from scipy.stats import binom
        k = min(b_cnt, c_cnt)
        p = float(2.0 * binom.cdf(k, n, 0.5))
        p = min(1.0, p)
    except Exception:                      # pragma: no cover - scipy 缺失时兜底
        # 正态近似（仅用于 scipy 不可用时，不作为默认路径）
        from math import erfc, sqrt
        z = abs(b_cnt - c_cnt) / max(sqrt(n), 1e-9)
        p = float(erfc(z / sqrt(2.0)))

    return {
        "b": b_cnt, "c": c_cnt, "n": n,
        "p_value": round(p, 6),
        "significant": p < 0.05,
        "note": (f"分歧 {n} 条（候选独对 {b_cnt} / 现役独对 {c_cnt}），"
                 f"p={p:.4f}"),
    }


def required_margin_for(n_per_class: int, alpha: float = 0.05) -> float:
    """给定"每类样本数"，返回在当前检验力下能被分辨的最小差距（粗略估计）。

    用途：写文档、做复盘时说明"这批数据能测出多大的改动"，
    而不是拿它当门控阈值（门控用的是 mcnemar 本身）。
    """
    n = max(int(n_per_class), 1)
    # 二项分布标准差的 2 倍（约 95% 覆盖），再按三类平均
    return float(2.0 * np.sqrt(0.15 / n))

def macro_f1_local(y_true, y_pred) -> float:
    """就地算 macro-F1（不依赖 metric 适配层，保证检验与判据同源）。"""
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    scores = []
    for c in np.unique(y):
        tp = float(np.sum((p == c) & (y == c)))
        fp = float(np.sum((p == c) & (y != c)))
        fn = float(np.sum((p != c) & (y == c)))
        if tp == 0:
            scores.append(0.0)
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def paired_bootstrap_test(y_true, pred_a, pred_b, n_boot: int = 500,
                          seed: int = 0) -> dict:
    """对「macro-F1 之差」做按类别分层的配对自助检验。

    做法（Dietterich 1998 推荐用于非准确率指标的重采样检验）：
        每类内部有放回重采样，重算两模型的 macro-F1 之差；
        p 值 = 差值 <= 0 的比例（单侧，只关心候选更好）。

    返回 observed_diff / p_value / 95% 区间 / significant / note。
    """
    y = np.asarray(y_true).ravel()
    a = np.asarray(pred_a).ravel()
    b = np.asarray(pred_b).ravel()
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_true / pred_a / pred_b 长度必须一致")

    obs = macro_f1_local(y, a) - macro_f1_local(y, b)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    diffs = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True)
                              for g in groups])
        diffs[i] = macro_f1_local(y[idx], a[idx]) - macro_f1_local(y[idx], b[idx])

    p_value = float(np.mean(diffs <= 0.0))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return {
        "observed_diff": round(float(obs), 6),
        "p_value": round(p_value, 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
        "n_boot": int(n_boot),
        "significant": p_value < 0.05,
        "note": (f"macro-F1 差 {obs:+.4f}，分层自助 p={p_value:.4f}，"
                 f"95% 区间 [{lo:+.4f}, {hi:+.4f}]"),
    }

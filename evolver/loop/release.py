"""⑤ 上线 / 回滚：候选必须过门控才允许替换，否则保留旧版（= 自动回滚）。

注意：门控比较的两方**必须在同一批数据上打分**，否则是拿不同尺子量身高。

这一步是门槛④（不低于静态基线）的保命绳：
只要回滚可靠，最坏情况也就是打平基线。

────────────────────────────────────────────────────────────
【整改 A3】原实现有两个洞：

  1. 只记录 history（决策文本），**不保存模型快照**。
     `self.current = candidate` 之后旧模型立即被回收 →
     若后面发现"这次上线是错的"，无法回到旧版本。
     原实现的"回滚"= 拒绝即维持，只能撤销**当前这一批**的决定，
     撤销不了**已经上线过**的版本。

  2. 门控只看**当前这一批**的增益，固定 margin 0.02。
     单批 F1 的抽样方差可能远大于 0.02 → 噪声就能驱动接受/拒绝。
     （统计缺陷属 B 类，接口在此预留）

本轮改动：
  - 保存模型快照（深拷贝），支持 rollback_to(version)
  - 支持多批增益窗口 + Bootstrap 置信下界（默认关闭，保持旧行为）
  - 默认参数下行为与原实现**完全一致**
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _deep(obj):
    """尽量深拷贝；对象不支持就退回引用（快照仍可用，只是共享权重）。"""
    clone = getattr(obj, "clone", None)
    if callable(clone):
        try:
            return clone()
        except Exception:                       # noqa: BLE001
            pass
    import copy
    try:
        return copy.deepcopy(obj)
    except Exception:                           # noqa: BLE001
        return obj


def bootstrap_lower_mean(values, n_boot: int = 500, alpha: float = 0.05,
                         seed: int = 0) -> float:
    """对均值做 Bootstrap，返回 (1−alpha) 置信下界。

    样本少于 3 个时不做重采样（统计上没有意义），直接返回均值。
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.size < 3 or n_boot <= 0:
        return float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, alpha))


class ReleaseManager:
    """门控 + 版本快照 + 回滚。

    参数
    ----
    model            初始（现役）模型
    reference_score  初始分数（**utility，越大越好**）
    margin           允许上线的最小增益。语义随 margin_mode 变化：
                       absolute → utility 差（如 F1 的 0.02）
                       relative → 相对改进比例（0.02 = 2%）
    max_snapshots    最多保留多少历史快照（防止内存无限增长）
    gate_window      用最近多少批的增益做判据（1 = 只看当前批，旧行为）
    n_boot           Bootstrap 重采样次数（0 = 不做，用均值）
    seed             Bootstrap 随机种子，固定以保证可复现
    margin_mode      "absolute" | "relative"      【整改 R1】
    metric_name      指标名，仅用于日志
    metric_direction "higher" | "lower"，仅用于日志与 raw 换算

    【整改 R1 · margin 归一化】
    原实现直接拿 `candidate - reference` 与常数 0.02 比较，与指标量纲无关：
      - F1 值域 0~1，0.02 是合理的 2 个百分点
      - MAE 值域 0~125，0.02 天等于"任何正增益都放行"或"永不更新"
    实测：回归任务 8 次动作里 7 次回滚。

    现在改为：
        raw_gain        = candidate_u - reference_u          （见下方恒等式）
        normalized_gain = raw_gain                     （absolute）
                        = raw_gain / max(|reference_u|, eps)   （relative）
        accept ⇔ normalized_gain ≥ margin

    关于"越低越好"的指标：内部一律使用 utility（MAE 取负号），因此
        utility = raw          （higher）
        utility = -raw         （lower）
    代入可得恒等式：
        candidate_u - reference_u
          = (cand_raw - ref_raw)   当 direction == higher
          = (ref_raw - cand_raw)   当 direction == lower
    恰好等于你要求的 gain 定义，所以**不会双重取反**。
    `raw_gain` 字段记录的是原始量纲下的增益（如 MAE 改善 2.3 天）。
    """

    def __init__(self, model, reference_score: float = 0.0, margin: float = 0.02,
                 max_snapshots: int = 20, gate_window: int = 1,
                 n_boot: int = 0, seed: int = 0,
                 margin_mode: str = "absolute",
                 metric_name: str = "score",
                 metric_direction: str = "higher",
                 eps: float = 1e-9,
                 gate_alpha: float = 0.05,
                 require_significance: bool = True,
                 n_boot_test: int = 300):
        if margin_mode not in ("absolute", "relative"):
            raise ValueError(f"未知 margin_mode '{margin_mode}'")
        self.current = model
        self.current_score = float(reference_score)
        self.margin = margin
        self.version = 0
        self.history: list[dict] = []

        self.margin_mode = margin_mode
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.eps = float(eps)
        # 显著性检验：Dietterich 1998 指出小测试集上的原始差值不可靠。
        # alpha=0.05；门控窗口 >1 时用窗口均值，不走单批检验。
        self.gate_alpha = float(gate_alpha)
        self.require_significance = bool(require_significance)
        self.last_test: dict | None = None
        self.n_boot_test = max(50, int(n_boot_test))
        # 相对模式下实际生效的模式（baseline 接近 0 时会退回 absolute）
        self.last_mode_used = margin_mode

        self.max_snapshots = max(1, int(max_snapshots))
        self.snapshots: list[dict] = [
            {"version": 0, "model": model, "score": self.current_score}
        ]

        self.gate_window = max(1, int(gate_window))
        self.n_boot = max(0, int(n_boot))
        self.seed = int(seed)
        self.gain_window: deque = deque(maxlen=self.gate_window)

    # ------------------------------------------------------------------
    # 判据
    # ------------------------------------------------------------------
    def _to_raw(self, utility: float) -> float:
        """utility → 人类可读的原始值（只用于日志）。"""
        return utility if self.metric_direction == "higher" else -utility

    def _normalize(self, raw_gain: float, reference_score: float) -> tuple[float, str]:
        """把增益归一化到与 margin 可比的尺度。

        relative 模式按 |baseline| 归一；baseline 接近 0 时**退回 absolute**，
        绝不做除法——否则会得到 inf 或一个荒谬的巨大增益。
        """
        if self.margin_mode == "relative":
            denom = abs(float(reference_score))
            if denom < self.eps:
                self.last_mode_used = "absolute_fallback"
                return raw_gain, f"baseline≈0（|{denom:.2e}|<{self.eps}），退回 absolute"
            self.last_mode_used = "relative"
            return raw_gain / denom, ""
        self.last_mode_used = "absolute"
        return raw_gain, ""

    def gate(self, candidate_score: float, reference_score: float,
             paired: tuple | None = None) -> tuple[bool, str]:
        """记一笔增益，并按"归一化增益 + 窗口"判据决定是否放行。

        参数都是 **utility**（越大越好）。
        返回 (是否放行, 说明)。
        """
        # 恒等式：utility 差 ≡ 按 direction 定义好的 raw gain（见类文档）
        raw_gain = float(candidate_score) - float(reference_score)
        norm_gain, note = self._normalize(raw_gain, reference_score)
        self.gain_window.append(norm_gain)

        unit = "%" if self.last_mode_used == "relative" else ""
        scale = 100.0 if unit == "%" else 1.0
        margin_txt = f"{self.margin * scale:+.4f}{unit}" if unit else f"{self.margin:+.4f}"
        gain_txt = f"{norm_gain * scale:+.4f}{unit}" if unit else f"{norm_gain:+.4f}"
        tail = f"（{note}）" if note else ""

        if self.gate_window == 1 or len(self.gain_window) < self.gate_window:
            # 单批路径：增益要够（经济意义），并且（若给了配对预测）差异要显著。
            # 没有 paired 时退回旧行为（回归任务没有"对错"，McNemar 不适用）。
            if paired is not None and self.require_significance:
                # 检验的统计量必须与判据同源：判据是 macro-F1，就用 macro-F1 的自助检验。
                # （McNemar 检验的是逐样本对错≈准确率，类别不均衡时会与 macro-F1 背离：
                #   实测一例 F1 差 0.33 而 McNemar p=0.70 —— 会系统性误拒。）
                from .stats import paired_bootstrap_test
                test = paired_bootstrap_test(*paired, n_boot=self.n_boot_test,
                                             seed=self.seed)
                margin_ok = norm_gain >= self.margin
                stats_ok = test["p_value"] < self.gate_alpha
                passed = margin_ok and stats_ok
                # 反事实留痕：只比大小时会怎么判。用于事后核对新门控
                # 到底拦下了什么、放过了什么（不改变行为）。
                test = {**test, "margin_ok": bool(margin_ok),
                        "stats_ok": bool(stats_ok),
                        "margin_only_would_accept": bool(margin_ok)}
                verdict = "放行" if passed else "拒绝上线"
                why = (f"同批对比：候选 {self._to_raw(candidate_score):.4f} vs 当前 "
                       f"{self._to_raw(reference_score):.4f}，"
                       f"归一化增益 {gain_txt}（门控 {margin_txt}）；"
                       f"{test['note']}（alpha={self.gate_alpha}）→ {verdict}"
                       f"（{self.metric_name}/{self.last_mode_used}）{tail}")
                self.last_test = test
                return passed, why
            if norm_gain >= self.margin:
                return True, (f"同批对比：候选 {self._to_raw(candidate_score):.4f} vs 当前 "
                              f"{self._to_raw(reference_score):.4f}，"
                              f"归一化增益 {gain_txt} ≥ 门控 {margin_txt}"
                              f"（{self.metric_name}/{self.last_mode_used}）{tail}")
            return False, (f"同批对比：候选 {self._to_raw(candidate_score):.4f} vs 当前 "
                           f"{self._to_raw(reference_score):.4f}，"
                           f"归一化增益 {gain_txt} < 门控 {margin_txt}"
                           f"（{self.metric_name}/{self.last_mode_used}），拒绝上线{tail}")

        gains = list(self.gain_window)
        if self.n_boot > 0:
            stat, label = (bootstrap_lower_mean(gains, self.n_boot, seed=self.seed),
                           "Bootstrap 95% 下界")
        else:
            stat, label = float(np.mean(gains)), "窗口均值"

        detail = (f"近 {len(gains)} 批归一化增益 "
                  f"{['%+.3f' % (g * scale) for g in gains]}{unit}；"
                  f"{label} {stat * scale:+.4f}{unit} vs 门控 {margin_txt}"
                  f"（{self.metric_name}/{self.last_mode_used}）{tail}")
        return (stat >= self.margin), detail

    def try_release(self, candidate, candidate_score: float,
                    reference_score: float,
                    paired: tuple | None = None) -> dict:
        """候选过门控则上线并快照，否则维持现役（= 自动回滚）。

        【R1】返回的记录里带上完整门控证据，供 decisions.jsonl / run.jsonl 留痕。
        """
        raw_gain = float(candidate_score) - float(reference_score)
        self.last_test = None
        ok, why = self.gate(candidate_score, reference_score, paired)
        norm_gain, _note = self._normalize(raw_gain, reference_score)

        if ok:
            self.current = candidate
            self.version += 1
            self.current_score = float(candidate_score)
            self._snapshot()
            decision = "accept"
        else:
            decision = "rollback"          # 什么也不换 = 自动回滚到上一版
            self.current_score = float(reference_score)

        rec = {
            "decision": decision,
            "release_decision": decision,
            "reason": why,
            "version": self.version,
            "score": self.current_score,
            # ---- 门控证据（整改 R1 要求留痕的字段）----
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "baseline_score": round(self._to_raw(reference_score), 6),
            "candidate_score": round(self._to_raw(candidate_score), 6),
            "baseline_utility": round(float(reference_score), 6),
            "candidate_utility": round(float(candidate_score), 6),
            "raw_gain": round(float(raw_gain), 6),
            "normalized_gain": round(float(norm_gain), 6),
            "margin": self.margin,
            "margin_mode": self.margin_mode,
            "margin_mode_used": self.last_mode_used,
            # 显著性检验证据（回归任务为 None）
            "mcnemar": self.last_test,
        }
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------------
    # 快照与回滚
    # ------------------------------------------------------------------
    def _snapshot(self) -> None:
        self.snapshots.append({
            "version": self.version,
            "model": _deep(self.current),
            "score": self.current_score,
        })
        if len(self.snapshots) > self.max_snapshots:
            # 丢掉最旧的一版（但永远保留 version 0，它是"部署基线"）
            for i, snap in enumerate(self.snapshots):
                if snap["version"] != 0:
                    del self.snapshots[i]
                    break

    def rollback_to(self, version: int) -> dict:
        """把现役模型退回到某个已上线过的版本。

        返回 {'ok': bool, 'reason': str, 'version': int}。
        找不到该版本时**不抛异常、不改变状态**——回滚失败本身也要能继续运行。
        """
        for idx, snap in enumerate(self.snapshots):
            if snap["version"] == version:
                self.current = _deep(snap["model"])
                self.current_score = float(snap["score"])
                self.version = int(version)
                # 丢掉该版本之后的快照（它们已不再是"历史"）
                self.snapshots = self.snapshots[:idx + 1]
                self.gain_window.clear()
                rec = {"decision": "rollback_to", "reason": f"回退到版本 {version}",
                       "version": version, "score": self.current_score}
                self.history.append(rec)
                return {"ok": True, "reason": rec["reason"], "version": version}
        return {"ok": False,
                "reason": f"版本 {version} 不在快照里（现有 {self.snapshot_versions()}）",
                "version": self.version}

    def snapshot_versions(self) -> list[int]:
        return [s["version"] for s in self.snapshots]

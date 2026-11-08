"""③ 决策器：把“要不要动、动哪一档”变成一个确定的函数。

刻意写成规则版——门槛①要求“无人工干预”，规则引擎最稳。
大模型可以在后面作为“提建议”的一层挂上来，但不能挡住主流程。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class State:
    batch_id: int
    psi_mean: float
    current_score: float
    best_score: float
    steps_since_update: int = 99
    last_action: str | None = None
    last_gain: float = 0.0
    warmup: bool = False
    history: list = field(default_factory=list)

    def read_history(self) -> str:
        """给决策记录用的“读历史”描述——门槛③要的就是这句话。"""
        recent = self.history[-3:]
        if not recent:
            return f"第 {self.batch_id} 批，无历史可读"
        seq = " → ".join(f"PSI={h['psi_mean']:.2f}" for h in recent)
        last = ""
        if self.last_action:
            last = f"；上次动作 {self.last_action}（增益 {self.last_gain:+.3f}）"
        return (f"近几批 PSI {seq}；当前分数 {self.current_score:.3f}，"
                f"历史最好 {self.best_score:.3f}{last}")


ACTIONS = ("do_nothing", "refresh_normalization", "refit_model")

# 成本粗略换算成“秒”，对应赛题的「单位增益消耗」
COST = {"do_nothing": 0.0, "refresh_normalization": 0.05, "refit_model": 2.0}


def decide(state: State, diagnosis: dict, cfg) -> dict:
    """返回 {action, hypothesis, reason, cost_sec}。"""
    psi = state.psi_mean

    # 1) 预热期：只观察
    if state.warmup:
        return _mk("do_nothing", "处于预热期，先收集历史再动手",
                   f"第 {state.batch_id} 批仍在预热窗口内")

    # 2) 分布稳定：不动才是正确决策（这一档最省钱，也常常最正确）
    if psi < cfg.psi_low:
        return _mk("do_nothing", "分布稳定，更新无收益",
                   f"PSI {psi:.3f} < {cfg.psi_low}，不满足最小更新门槛")

    # 3) 触发间隔不足：防止频繁抖动
    if state.steps_since_update < cfg.min_gap:
        return _mk("do_nothing", "距上次更新过近，等下一批再判断",
                   f"距上次更新 {state.steps_since_update} 批 < {cfg.min_gap}")

    # 4) 上次便宜手段没奏效 → 升级
    if state.last_action == "refresh_normalization" and state.last_gain < cfg.min_gain:
        return _mk("refit_model",
                   f"上次仅校准标准化只带来 {state.last_gain:+.3f} 增益，说明不只是尺度问题",
                   "零成本手段已证明无效，按性价比阶梯升级到重训")

    # 5) 漂移很猛 → 直接上重训
    if psi >= cfg.psi_mid:
        return _mk("refit_model", f"PSI {psi:.2f} 已超过重型阈值，校准多半不够",
                   "按性价比阶梯直接选择更贵但更有效的一档")

    # 6) 默认：先试零成本那一档（省钱就省在这里）
    return _mk("refresh_normalization",
               "特征整体平移/缩放，属协变量漂移，重算标准化统计量应可恢复",
               "先用成本≈0 的手段试，不行下一轮再升级")


def _mk(action: str, hypothesis: str, reason: str) -> dict:
    return {"action": action, "hypothesis": hypothesis, "reason": reason,
            "cost_sec": COST[action]}

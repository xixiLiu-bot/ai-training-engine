"""③ 决策器：把“要不要动、动哪一档”变成一个确定的函数。

关键设计：**动作清单由模型提供，不写死在这里。**

因为不同模型能做的事不一样：
  · 逻辑回归 —— 「刷新标准化」非常有效（实测 +0.33 F1）
  · 梯度提升树 —— 「刷新标准化」反而有害（实测 -0.08 F1），
                  因为它会冲垮树原来的切分阈值

所以决策器只在该模型**支持且划算**的动作里挑。换模型时，坏档位自动消失，
规则本身不用改。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 兼容用：没传 catalog 时的默认清单（按逻辑回归标定）
DEFAULT_COST = {"do_nothing": 0.0, "refresh_normalization": 0.05, "refit_model": 2.0}
ACTIONS = tuple(DEFAULT_COST)
COST = DEFAULT_COST


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
    psi_ratio: float = 0.0        # 有多少比例的特征超标（0~1）
    psi_top: float = 0.0          # 最严重的那个特征
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


def decide(state: State, diagnosis: dict, cfg, catalog: dict | None = None) -> dict:
    """返回 {action, hypothesis, reason, cost_sec}。

    catalog：这个模型支持的动作及成本。不传则用默认（逻辑回归标定）。
    """
    cost = catalog or DEFAULT_COST
    psi = state.psi_mean
    can_refresh = "refresh_normalization" in cost
    can_refit = "refit_model" in cost

    # 1) 预热期：只观察
    if state.warmup:
        return _mk("do_nothing", "处于预热期，先收集历史再动手",
                   f"第 {state.batch_id} 批仍在预热窗口内", cost)

    # 2) 分布稳定：不动才是正确决策（这一档最省钱，也常常最正确）
    if psi < cfg.psi_low:
        return _mk("do_nothing", "分布稳定，更新无收益",
                   f"PSI {psi:.3f} < {cfg.psi_low}，不满足最小更新门槛", cost)

    # 3) 触发间隔不足：防止频繁抖动
    if state.steps_since_update < cfg.min_gap:
        return _mk("do_nothing", "距上次更新过近，等下一批再判断",
                   f"距上次更新 {state.steps_since_update} 批 < {cfg.min_gap}", cost)

    # 4) 上次便宜手段没奏效 → 升级
    if (can_refresh and can_refit
            and state.last_action == "refresh_normalization"
            and state.last_gain < cfg.min_gain):
        return _mk("refit_model",
                   f"上次仅校准标准化只带来 {state.last_gain:+.3f} 增益，说明不只是尺度问题",
                   "廉价手段已证明无效，按性价比阶梯升级到重训", cost)

    # 5) 漂移很猛 → 能重训就重训
    if psi >= cfg.psi_mid:
        if can_refit:
            return _mk("refit_model", f"PSI {psi:.2f} 已超过重型阈值，校准多半不够",
                       "按性价比阶梯直接选择更贵但更有效的一档", cost)
        return _mk("do_nothing", "漂移明显，但这个模型没有可用的重训手段",
                   "动作清单里没有 refit_model，只能维持现状", cost)

    # 5b) 大面积特征超标（不只是平均偏高）→ 局部校准救不了
    if state.psi_ratio >= 0.5 and can_refit:
        return _mk("refit_model",
                   f"{state.psi_ratio:.0%} 的特征都超标，不是个别传感器的问题",
                   "超标比例过半，校准只能救局部，直接重训更划算", cost)

    # 6) 默认：先试零成本那一档（省钱就省在这里）
    if can_refresh:
        return _mk("refresh_normalization",
                   "特征整体平移/缩放，属协变量漂移，重算标准化统计量应可恢复",
                   "先用成本≈0 的手段试，不行下一轮再升级", cost)

    if can_refit:
        return _mk("refit_model", "这个模型没有廉价校准档，只能重训",
                   "动作清单里没有 refresh_normalization", cost)

    return _mk("do_nothing", "没有可用的更新手段", "动作清单为空", cost)


def _mk(action: str, hypothesis: str, reason: str, cost: dict) -> dict:
    return {"action": action, "hypothesis": hypothesis, "reason": reason,
            "cost_sec": cost.get(action, 0.0)}

"""③ 决策器：把“要不要动、动哪一档”变成一个确定的函数。

刻意写成规则版——门槛①要求“无人工干预”，规则引擎最稳。
大模型可以在后面作为“提建议”的一层挂上来，但不能挡住主流程。

【整改 B3】补齐性价比阶梯最便宜的一档 `calibrate_output`（改输出）。

原来的阶梯其实只有两档真正可用：
    do_nothing  →  refresh_normalization（改特征：重算标准化）
                →  refit_model（重训）
缺了文档里承诺的“改输出（阈值/温度校准）”——那一档成本最低（≈0.02s），
在“判定阈值偏移”“类别先验变化”这类漂移下能直接救回来，却完全没实现。

现在（仅在分类任务且 cfg.enable_output_calibration=True 时启用）：
    do_nothing → calibrate_output → refresh_normalization → refit_model

默认 `enable_output_calibration=False`，行为与旧版**逐位一致**，
保证既有分类结果（0.836）不被改动。需要开启时显式传参。
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


ACTIONS = ("do_nothing", "calibrate_output", "refresh_normalization", "refit_model")

# 成本粗略换算成“秒”，对应赛题的「单位增益消耗」
# calibrate_output 比 refresh_normalization 更便宜：只做一次前向 + 小网格搜索，
# 不重新拟合标准化统计量（后者要重扫全批数据算均值方差）。
COST = {
    "do_nothing": 0.0,
    "calibrate_output": 0.02,
    "refresh_normalization": 0.05,
    "refit_model": 2.0,
}


def _calibration_enabled(cfg) -> bool:
    """输出校准只在"分类任务 + 显式开启"时可用。

    回归任务没有合理的输出校准目标（见 model.CalibrationUnsupported），
    所以这里直接不给它出场机会，而不是让它执行后再失败。
    """
    return bool(getattr(cfg, "enable_output_calibration", False)) and \
        (getattr(cfg, "task_type", "classification") == "classification")


def decide(state: State, diagnosis: dict, cfg, catalog: dict | None = None) -> dict:
    """返回 {action, hypothesis, reason, cost_sec}。

    catalog：这个模型支持的动作及成本（来自 adapters.model.action_catalog）。
    不传则用默认 COST，行为与旧版一致。
    """
    cost = dict(catalog) if catalog else dict(COST)
    psi = state.psi_mean
    can_calib = _calibration_enabled(cfg) and "calibrate_output" in cost

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

    # 4) 【B3】上次改输出没奏效 → 升级到改特征
    if (can_calib and state.last_action == "calibrate_output"
            and state.last_gain < cfg.min_gain):
        return _mk("refresh_normalization",
                   f"上次输出校准仅 {state.last_gain:+.3f} 增益，说明不只是判定阈值问题",
                   "最便宜的一档已证明无效，按性价比阶梯升级到改特征", cost)

    # 5) 上次改特征没奏效 → 升级到重训
    if state.last_action == "refresh_normalization" and state.last_gain < cfg.min_gain:
        return _mk("refit_model",
                   f"上次仅校准标准化只带来 {state.last_gain:+.3f} 增益，说明不只是尺度问题",
                   "零成本手段已证明无效，按性价比阶梯升级到重训", cost)

    # 6) 漂移很猛 → 直接上重训
    if psi >= cfg.psi_mid:
        return _mk("refit_model", f"PSI {psi:.2f} 已超过重型阈值，校准多半不够",
                   "按性价比阶梯直接选择更贵但更有效的一档", cost)

    # 6b) 大面积特征超标 → 局部手段救不了整体
    #     比例按**重型阈值 psi_mid** 统计（不是 psi_low）。这一点很关键：
    #     按 psi_low 统计时，合成数据几乎 100% 特征都"超标"，规则会一直误触发，
    #     实测把合成数据从 0.864 拖到 0.587；改按 psi_mid 后，
    #     合成 0~15%、气体最高 84%，区分度才出来。
    if (state.psi_ratio >= float(getattr(cfg, "psi_ratio_th", 0.7))
            and "refit_model" in cost):
        return _mk("refit_model",
                   f"{state.psi_ratio:.0%} 的特征超过重型阈值，不是个别传感器的问题",
                   "整体漂移，局部校准救不了，直接重训更划算", cost)

    # 7) 【B3】轻度过失：先试最便宜的"改输出"
    if can_calib:
        return _mk("calibrate_output",
                   "判定阈值/类别先验可能偏移，改输出应可恢复，无需动特征与权重",
                   "按性价比阶梯从最便宜的一档（改输出）开始试", cost)

    # 8) 默认：改特征（重算标准化）——与旧版一致
    return _mk("refresh_normalization",
               "特征整体平移/缩放，属协变量漂移，重算标准化统计量应可恢复",
               "先用成本≈0 的手段试，不行下一轮再升级", cost)


def _mk(action: str, hypothesis: str, reason: str, cost: dict) -> dict:
    return {"action": action, "hypothesis": hypothesis, "reason": reason,
            "cost_sec": cost.get(action, 0.0)}

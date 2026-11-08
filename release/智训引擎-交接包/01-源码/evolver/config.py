"""全局配置：所有阈值都放这里，方便后面接官方数据时统一调。"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- 监测 ----
    psi_bins: int = 10
    psi_low: float = 0.10       # 低于它：视为稳定，不动
    psi_mid: float = 0.25       # 超过它：明显漂移，可以直接上重训
    label_psi_th: float = 0.10  # 标签分布漂移判定阈值

    # ---- 决策 ----
    min_gap: int = 1            # 两次更新之间至少隔几个批次
    min_gain: float = 0.02      # 增益低于它：认为这次更新没用

    # ---- 门控 ----
    gate_margin: float = 0.02   # 候选必须比当前高这么多才允许上线

    # ---- 预算（对应赛题的“单位增益消耗”）----
    budget_gpu_sec: float = 600.0
    budget_labels: int = 30000
    budget_tokens: int = 0      # 暂时不接大模型

    # ---- 其它 ----
    warmup_batches: int = 1     # 前几批只观察不动作
    random_state: int = 0
    extra: dict = field(default_factory=dict)

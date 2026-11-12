"""全局配置：所有阈值都放这里，方便后面接官方数据时统一调。

【整改说明】原配置有两个问题：
  1. 没有"任务类型"，导致指标写死分类；
  2. 没有"检测器选择 / 门控窗口 / Bootstrap"等开关，
     新能力只能硬编码，无法在不改代码的前提下做消融实验。

本轮新增字段**全部带默认值**，默认路径与原实现完全一致。
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- 任务 ----
    task_type: str = "classification"   # classification | regression
    metric_name: str = ""               # 空 = 按 task_type 自动分派

    # ---- 监测 ----
    psi_bins: int = 10
    psi_low: float = 0.10       # 低于它：视为稳定，不动
    psi_mid: float = 0.25       # 超过它：明显漂移，可以直接上重训
    # 超标比例阈值：超过这个比例的特征都漂了，才判定「整体漂移」而不是局部
    # 调高到 0.85 是因为气体数据上 92%~95% 特征常年超标，0.5 会让重训成为唯一选择
    psi_ratio_th: float = 0.70
    label_psi_th: float = 0.10  # 标签分布漂移判定阈值
    detector: str = "psi"       # psi | ks | wasserstein | all

    # ---- 决策 ----
    min_gap: int = 1            # 两次更新之间至少隔几个批次
    min_gain: float = 0.02      # 增益低于它：认为这次更新没用
    enable_output_calibration: bool = False   # 是否启用"改输出"这一档（默认关，见 README）

    # ---- 门控 ----
    gate_margin: float = 0.02   # 候选必须比当前高这么多才允许上线
    # 【整改 R1】margin 的量纲：absolute=F1 的绝对差；relative=相对 baseline 的比例
    # 默认 absolute 保持分类流程不变；回归任务由 run_pipeline 自动切到 relative
    margin_mode: str = "absolute"
    margin_relative_floor: float = 1e-9   # baseline≈0 时退回 absolute 的判据
    gate_window: int = 1        # 用最近几批增益做判据（1=只看当前批，旧行为）
    gate_bootstrap: int = 0     # Bootstrap 重采样次数（0=关闭，用窗口均值）
    # 【调参】重训窗口：用最近几批（b-W .. b-1），以及时间衰减系数
    refit_window: int = 4
    refit_tau: float = 0.5

    # ---- 预算（对应赛题的“单位增益消耗”）----
    budget_gpu_sec: float = 600.0
    budget_labels: int = 30000
    budget_tokens: int = 0      # 暂时不接大模型

    # ---- 其它 ----
    warmup_batches: int = 1     # 前几批只观察不动作
    random_state: int = 0
    logreg_c: float = 1.0    # 逻辑回归的正则强度倒数（越小越强）
    scaler_kind: str = "standard"   # standard | quantile（百分位归一）
    incremental_epochs: int = 3      # 增量更新喂几轮（多了会灾难性遗忘）
    # 【改进 4】特征筛选：none | var | fclass；feature_k = 保留比例
    feature_select: str = "none"
    feature_k: float = 0.6
    # 【改进 1】训练后自动输出校准（留出集验证，不改善不启用）
    auto_calibrate: bool = False
    max_snapshots: int = 20     # 保留多少个模型快照（回滚用）
    extra: dict = field(default_factory=dict)

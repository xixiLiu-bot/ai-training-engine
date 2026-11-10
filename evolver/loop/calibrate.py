"""阈值自标定：不用手调 PSI 阈值，让系统从前几批自己定。

为什么需要：
    合成数据 PSI 0.1~0.3，气体传感器 1~4，轴承 1.8~7.3。
    同一套绝对值（0.1 / 0.25）在真实数据上必然失效——
    要么从不动作，要么永远走最贵那档。

    以前每换一个数据集都要手动试 --
        --psi-low 1.2 --psi-mid 2.5   （气体）
        --psi-low 1.5 --psi-mid 3.0   （轴承）
    这还叫「换数据集只改一个文件」吗？当然不算。

做法：
    预热期（前 N 批）只观察，记下 PSI 的自然水位；
    然后按倍数定阈值——
        low = 中位数 × 1.0     （比平时偏一点就该看）
        mid = 中位数 × 2.0     （比平时偏一倍，别浪费廉价手段）

    这样换数据集、换模型都不用手调，阈值跟着数据走。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThresholdCalibrator:
    """观察前几批的 PSI，自动定出稳定阈值和重型阈值。"""

    warmup: int = 2            # 观察几批
    low_mult: float = 1.0      # 稳定阈值 = 中位数 × 这个倍数
    mid_mult: float = 2.0      # 重型阈值 = 中位数 × 这个倍数
    floor_low: float = 0.05    # 下限，防止标定出 0
    floor_mid: float = 0.15
    ceil_low: float = 5.0      # 上限，防止标定出荒谬的值
    ceil_mid: float = 12.0

    samples: list = field(default_factory=list)

    def observe(self, psi_mean: float) -> None:
        """预热期每批调一次。"""
        if len(self.samples) < self.warmup:
            self.samples.append(float(psi_mean))

    @property
    def ready(self) -> bool:
        return len(self.samples) >= self.warmup

    @property
    def base(self) -> float:
        return float(np.median(self.samples)) if self.samples else 0.0

    def calibrate(self) -> tuple[float, float]:
        """返回 (psi_low, psi_mid)。"""
        if not self.ready:
            raise RuntimeError(f"预热还没完成：{len(self.samples)}/{self.warmup}")
        base = self.base
        low = float(np.clip(base * self.low_mult, self.floor_low, self.ceil_low))
        mid = float(np.clip(base * self.mid_mult, max(low * 1.5, self.floor_mid),
                            self.ceil_mid))
        return low, mid

    def explain(self) -> str:
        if not self.ready:
            return f"采样中 {len(self.samples)}/{self.warmup}"
        low, mid = self.calibrate()
        vals = " → ".join(f"{v:.2f}" for v in self.samples)
        return (f"观察值 {vals}；中位数 {self.base:.2f} → "
                f"稳定阈值 {low:.2f}，重型阈值 {mid:.2f}")

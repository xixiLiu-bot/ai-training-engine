"""⑤ 上线 / 回滚：候选必须过门控才允许替换，否则保留旧版（= 自动回滚）。

注意：门控比较的两方**必须在同一批数据上打分**，否则是拿不同尺子量身高。

这一步是门槛④（不低于静态基线）的保命绳：
只要回滚可靠，最坏情况也就是打平基线。
"""
from __future__ import annotations


class ReleaseManager:
    def __init__(self, model, reference_score: float = 0.0, margin: float = 0.02):
        self.current = model
        self.current_score = float(reference_score)
        self.margin = margin
        self.version = 0
        self.history: list[dict] = []

    def gate(self, candidate_score: float, reference_score: float) -> tuple[bool, str]:
        gain = candidate_score - reference_score
        if gain >= self.margin:
            return True, (f"同批对比：候选 {candidate_score:.4f} vs 当前 {reference_score:.4f}，"
                          f"增益 {gain:+.4f} ≥ 门控 {self.margin}")
        return False, (f"同批对比：候选 {candidate_score:.4f} vs 当前 {reference_score:.4f}，"
                       f"增益 {gain:+.4f} < 门控 {self.margin}，拒绝上线")

    def try_release(self, candidate, candidate_score: float,
                    reference_score: float) -> dict:
        ok, why = self.gate(candidate_score, reference_score)
        if ok:
            self.current = candidate
            self.version += 1
            decision = "accept"
            self.current_score = float(candidate_score)
        else:
            decision = "rollback"          # 什么也不换 = 自动回滚到上一版
            self.current_score = float(reference_score)
        rec = {"decision": decision, "reason": why, "version": self.version,
               "score": self.current_score}
        self.history.append(rec)
        return rec

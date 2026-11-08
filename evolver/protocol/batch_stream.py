"""评测协议层。

依据 11/6 说明会答疑：

> B 包数据按批次下发，**必须先输出当前批次模型或不上线的判断，才能获取下一批数据**。

这意味着评测不是「把一整个数据流灌给你」，而是**一问一答**：

    要第 1 批 → 拿到 → 处理 → 【交卷】 → 才给第 2 批 → ...

这条约束的真正作用不是添麻烦，而是**从结构上堵死“偷看未来”**：
你看不到后面的数据，就没法为了测试流去调参——这正是评分里“是否过拟合”那一项要防的事。

所以本地开发也必须遵守同一套协议，否则本地跑通 ≠ 评测跑通。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ProtocolViolation(RuntimeError):
    """违反评测协议（例如没交卷就要下一批、或偷看未来批次）。"""


@dataclass
class Submission:
    """每一轮必须交给评测方的判断。"""
    round_no: int
    batch_id: int
    decision: str                 # "deploy"（采用当前模型）| "keep"（不上线，维持原样）
    model_version: int
    reason: str
    predictions: np.ndarray | None = None
    self_reported: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.decision not in ("deploy", "keep"):
            raise ProtocolViolation(
                f"decision 只能是 deploy / keep，收到 {self.decision!r}")

    def brief(self) -> dict:
        return {"round": self.round_no, "batch": self.batch_id,
                "decision": self.decision, "model_version": self.model_version,
                "reason": self.reason[:120]}


class LocalEvaluator:
    """本地模拟版评测器：一次只放一批，必须交卷才给下一批。

    strict=True 时，任何违规都直接抛异常——本地就把问题暴露出来，
    不要等上了评测平台才发现流程走不通。
    """

    def __init__(self, adapter, strict: bool = True):
        self.adapter = adapter
        self.strict = strict
        self.released: dict[int, object] = {}
        self.max_released = 0
        self.pending: int | None = None
        self.submissions: list[Submission] = []
        self.violations: list[str] = []

    # ---- 协议动作 ----
    def next_batch(self):
        if self.pending is not None:
            self._violate(f"第 {self.pending} 批尚未提交判断，不能请求下一批")
        nxt = self.max_released + 1
        if nxt > self.adapter.n_batches:
            return None
        batch = self.adapter.get(nxt)
        self.released[nxt] = batch
        self.max_released = nxt
        self.pending = nxt
        return batch

    def submit(self, sub: Submission) -> None:
        if self.pending is None:
            self._violate("当前没有待提交的批次")
        if sub.batch_id != self.pending:
            self._violate(
                f"提交的批次 {sub.batch_id} 与待处理批次 {self.pending} 不一致")
        self.submissions.append(sub)
        self.pending = None

    # ---- 只读：已下发的历史 ----
    def history(self, start: int, end: int):
        """只能取**已经下发过**的批次；请求未来批次会被拦下。"""
        if end > self.max_released:
            self._violate(
                f"试图访问尚未下发的第 {end} 批（当前最多到 {self.max_released}）")
            end = self.max_released
        return [(i, self.released[i]) for i in range(max(1, start), end + 1)
                if i in self.released]

    def _violate(self, msg: str):
        self.violations.append(msg)
        if self.strict:
            raise ProtocolViolation(msg)

    # ---- 汇总 ----
    def summary(self) -> dict:
        return {
            "rounds": len(self.submissions),
            "deploy": sum(1 for s in self.submissions if s.decision == "deploy"),
            "keep": sum(1 for s in self.submissions if s.decision == "keep"),
            "violations": len(self.violations),
            "violation_detail": self.violations[:5],
        }


class RemoteEvaluator:
    """企业赛真评测器（占位）。

    拿到官方 SDK / 接口文档后，只需要实现三个方法：
        next_batch()  -> Batch
        submit(sub)   -> None
        history(s,e)  -> [(id, Batch)]
    上层的闭环代码一行都不用动。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "企业赛评测接口尚未下发。等官方 SDK 到手，在这里实现 next_batch / submit 即可。")

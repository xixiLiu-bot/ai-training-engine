"""预算账本：每一次动作都要记账。

对应赛题的客观分口径——「单位增益消耗的算力 / 标注 / Token」。
这里只是最简版：记录耗时、标注用量、调用次数，并做硬上限拦截。

【整改 A1】原实现有两个会导致整轮 0 分的问题：
  1. 构造时无条件 open(path,"w") —— 评测容器若是只读文件系统，整条流程起不来；
  2. 每次 log 都重新 open —— 中途磁盘满/权限变化会让主循环直接抛异常。
现在：写不了就降级为**纯内存账本**，`summary()` 与 `records` 仍然可用，
只把 `write_ok=False` 记下来，交由上层决定是否告警。
"""
from __future__ import annotations

import json
import os
import time


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: str, budget_gpu_sec: float = 600.0,
                 budget_labels: int = 2000, budget_tokens: int = 0):
        self.path = path
        self.budget = {"gpu_sec": budget_gpu_sec, "labels": budget_labels,
                       "tokens": budget_tokens}
        self.used = {"gpu_sec": 0.0, "labels": 0, "tokens": 0}
        self.records: list[dict] = []
        self.write_ok = False
        self.write_error: str | None = None
        try:                                   # 只读文件系统不能让流程起不来
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "w", encoding="utf-8").close()
            self.write_ok = True
        except Exception as exc:               # noqa: BLE001
            self.write_error = f"{type(exc).__name__}: {exc}"

    # ---- 记账 ----
    def charge(self, gpu_sec: float = 0.0, labels: int = 0, tokens: int = 0):
        if self.used["gpu_sec"] + gpu_sec > self.budget["gpu_sec"]:
            raise BudgetExceeded(f"算力预算超限：{self.used['gpu_sec']:.1f}+{gpu_sec} > {self.budget['gpu_sec']}")
        if self.used["labels"] + labels > self.budget["labels"]:
            raise BudgetExceeded(f"标注预算超限：{self.used['labels']}+{labels} > {self.budget['labels']}")
        if self.budget["tokens"] and self.used["tokens"] + tokens > self.budget["tokens"]:
            raise BudgetExceeded("Token 预算超限")
        self.used["gpu_sec"] += gpu_sec
        self.used["labels"] += labels
        self.used["tokens"] += tokens

    # ---- 落盘（decisions.jsonl 就是门槛③的证据）----
    def log(self, record: dict):
        record = {**record, "ts": time.time()}
        self.records.append(record)
        if not self.write_ok:
            return                             # 降级为纯内存，账本语义不变
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:               # noqa: BLE001
            self.write_ok = False
            self.write_error = f"{type(exc).__name__}: {exc}"

    def summary(self) -> dict:
        return {"used": self.used, "budget": self.budget,
                "write_ok": self.write_ok,
                "write_error": self.write_error,
                "n_actions": len([r for r in self.records
                                   if r.get("experiment", {}).get("action")
                                   not in (None, "do_nothing")])}

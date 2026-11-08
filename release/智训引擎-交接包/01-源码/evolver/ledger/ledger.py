"""预算账本：每一次动作都要记账。

对应赛题的客观分口径——「单位增益消耗的算力 / 标注 / Token」。
这里只是最简版：记录耗时、标注用量、调用次数，并做硬上限拦截。
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "w", encoding="utf-8").close()

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
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        return {"used": self.used, "budget": self.budget,
                "n_actions": len([r for r in self.records
                                   if r.get("experiment", {}).get("action")
                                   not in (None, "do_nothing")])}

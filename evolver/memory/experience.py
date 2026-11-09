"""L2 经验表：记下“哪种漂移、用了什么动作、结果如何”，供后续决策查表。

这是「策略层自改进」的最小可行实现——不训练任何模型，
只是把实验记录整理成一张可查询的表，然后让决策器参考它。

      (漂移类型, 动作)  →  用了几次 / 累计增益 / 累计成本 / 单位成本增益

为什么它有用：

1. 第一轮没有经验，全靠规则；
2. 跑过几轮之后，表里有了证据，比如“PSI 中等时校准的单位成本增益是重训的 90 倍”；
3. 下一轮遇到同类情况，直接查表，不用再试错。

这就是“越做越聪明”，而且成本为零。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Record:
    """某一类漂移下，某个动作的累计战绩。"""
    drift_kind: str
    action: str
    n: int = 0
    total_gain: float = 0.0        # 累计增益（可为负）
    total_cost: float = 0.0        # 累计成本（秒）
    wins: int = 0                  # 上线次数
    losses: int = 0                # 回滚次数
    recent: list = field(default_factory=list)   # 最近几次结果（accept/rollback）

    @property
    def gain_per_cost(self) -> float:
        """单位成本换来的增益——这就是赛题客观分的口径。"""
        return self.total_gain / max(self.total_cost, 1e-9)

    @property
    def avg_gain(self) -> float:
        return self.total_gain / max(self.n, 1)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gain_per_cost"] = round(self.gain_per_cost, 4)
        d["avg_gain"] = round(self.avg_gain, 4)
        d["total_gain"] = round(self.total_gain, 4)
        d["total_cost"] = round(self.total_cost, 4)
        return d


@dataclass
class ExperienceTable:
    """经验表本体。可以存盘、可以打印成人能看的样子。"""
    records: dict = field(default_factory=dict)   # (kind, action) -> Record
    min_samples: int = 2          # 少于这个次数，不足以当证据
    advantage: float = 1.5        # 要超过规则选择这么多倍，才值得改判
    epsilon: float = 0.2          # 探索率：有一定概率忽略经验，走规则
    rng: object = None            # 随机数发生器（固定种子可复现）

    @staticmethod
    def _norm(kind: str) -> str:
        """归因类型归一化。

        diagnose 会给出 concept / concept_inferred 两种叫法（后者是“用实验结果反推的”），
        它们在策略上是同一类，若不合并，各自的样本永远凑不够 min_samples，
        经验表就永远不会生效。
        """
        k = (kind or "").strip().lower()
        if k.startswith("concept"):
            return "concept"
        if k.startswith("covariate"):
            return "covariate"
        return k or "unknown"

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    def _key(self, kind: str, action: str):
        return (self._norm(kind), action)

    # ---- 记账 ----
    def record(self, kind: str, action: str, gain: float, cost: float,
               decision: str = "accept") -> None:
        if action == "do_nothing":
            return                      # 不动作不产生“经验”
        k = self._key(kind, action)
        rec = self.records.get(k) or Record(kind, action)
        rec.n += 1
        rec.total_gain += float(gain)
        rec.total_cost += float(cost)
        if decision == "accept":
            rec.wins += 1
        elif decision == "rollback":
            rec.losses += 1
        rec.recent = (rec.recent + [decision])[-5:]     # 只留最近 5 次
        self.records[k] = rec

    # ---- 查表 ----
    def best_action(self, kind: str, candidates: list[str]) -> tuple[str, float] | None:
        """在这类漂移下，返回单位成本增益最高的动作。

        返回 None 表示“经验不足，别听我的”。
        """
        pool = [self.records.get(self._key(kind, a)) for a in candidates]
        pool = [r for r in pool if r is not None and r.n >= self.min_samples]
        # 过滤一：最近连着两次都回滚 → 这招现在不好使了，别再推
        pool = [r for r in pool
                if not (len(r.recent) >= 2 and r.recent[-1] == r.recent[-2] == "rollback")]
        # 过滤二：单位成本增益必须为正
        pool = [r for r in pool if r.gain_per_cost > 0]
        if not pool:
            return None
        best = max(pool, key=lambda r: r.gain_per_cost)
        return best.action, best.gain_per_cost

    def gain_per_cost_of(self, kind: str, action: str) -> float | None:
        rec = self.records.get(self._key(kind, action))
        if rec is None or rec.n < self.min_samples:
            return None
        return rec.gain_per_cost

    # ---- 输出 ----
    def summary(self) -> list[dict]:
        rows = [r.as_dict() for r in self.records.values()]
        return sorted(rows, key=lambda d: (d["drift_kind"], -d["gain_per_cost"]))

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"records": self.summary()}, f, ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        rows = self.summary()
        if not rows:
            print("  经验表：还没有积累（前几批只会按规则走）")
            return
        print(f"  {'漂移类型':<18}{'动作':<24}{'次数':>5}{'累计增益':>10}"
              f"{'累计成本':>10}{'单位成本增益':>13}")
        print("  " + "-" * 82)
        for d in rows:
            print(f"  {d['drift_kind']:<18}{d['action']:<24}{d['n']:>5}"
                  f"{d['total_gain']:>+10.3f}{d['total_cost']:>10.2f}"
                  f"{d['gain_per_cost']:>13.2f}")

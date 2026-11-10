"""把「规则引擎」「经验记忆」「大模型建议」合成一个决策出口。

顺序固定：先规则，再查经验，最后问大模型，全部过闸才采纳。

    ┌─────────┐
    │ 规则引擎 │  ← 永远先算一遍，作为保底与对照
    └────┬────┘
         │ rule_action
    ┌────▼─────┐
    │ 经验记忆 │  ← 有足够证据时改判；保留 ε 探索，避免过早锁定
    └────┬─────┘
         │ base_action
    ┌────▼─────┐
    │ 大模型建议 │  ← 可选；超时/报错/胡说 → 返回 None
    └────┬─────┘
         │ suggestion
    ┌────▼──────────────────────┐
    │ 四道闸：白名单 / 预算 / 可解析 / 不重复无效 │
    └────┬──────────────────────┘
         │
    采纳 or 降级到 base_action

**动作清单由模型提供**（catalog）。不同模型能做的事和代价不一样，
所以决策器只从它能做的里面挑。
"""
from __future__ import annotations

from ..loop import decide as dc

# 最便宜的那几档：受「不许重复无效」闸约束
_CHEAP = ("calibrate_output", "refresh_normalization")


def decide(state, diagnosis, cfg, agent=None, memory=None,
           catalog: dict | None = None) -> dict:
    """返回 {action, hypothesis, reason, cost_sec, source, llm, memory}。"""
    cost = dict(catalog) if catalog else dict(dc.COST)
    rule = dc.decide(state, diagnosis, cfg, cost)
    base = {**rule, "source": "rule", "llm": None, "memory": None}

    # ── 第二层：经验记忆 ──
    # 保留 ε 探索：不能因为一次成功就锁死，否则会陷在过时的经验里
    available = list(cost.keys())
    explore = (memory is not None
               and getattr(memory, "rng", None) is not None
               and memory.rng.random() < getattr(memory, "epsilon", 0.0))
    if memory is not None and not explore:
        best = memory.best_action(diagnosis["kind"], available)
        if best is not None:
            act, gpc = best
            cur = memory.gain_per_cost_of(diagnosis["kind"], rule["action"])
            if act != rule["action"] and (cur is None or gpc >= cur * memory.advantage):
                base = {
                    "action": act,
                    "hypothesis": f"经验表显示：{diagnosis['kind']} 类漂移下 {act} 更划算",
                    "reason": (f"经验：{act} 单位成本增益 {gpc:.2f}"
                               + (f"，规则选的 {rule['action']} 只有 {cur:.2f}" if cur else "")),
                    "cost_sec": cost.get(act, 0.0),
                    "source": "memory",
                    "llm": None,
                    "memory": {"chosen": act,
                               "gain_per_cost": round(gpc, 3),
                               "rule_action": rule["action"],
                               "rule_gain_per_cost": None if cur is None else round(cur, 3)},
                }

    # ── 第三层：大模型 ──
    if agent is None or not getattr(agent, "enabled", False):
        return base

    llm_catalog = {k: {"cost_sec": v} for k, v in cost.items()}
    state_dict = {
        "batch_id": state.batch_id,
        "psi_mean": state.psi_mean,
        "current_score": state.current_score,
        "best_score": state.best_score,
        "steps_since_update": state.steps_since_update,
        "last_action": state.last_action or "无",
        "last_gain": state.last_gain,
    }

    sug = agent.suggest(state_dict, diagnosis, llm_catalog, base["action"])

    # 闸一：没给建议 / 给的是错误信息
    if not sug or "action" not in sug:
        return {**base, "llm": sug, "note": "大模型不可用，已降级"}

    # 闸二：动作必须真的在这个模型的能力清单里
    if sug["action"] not in cost:
        return {**base, "llm": sug,
                "note": f"动作 {sug['action']} 不在本模型可用清单里，已拒绝"}

    # 闸三：不许重复“已经被证明无效”的便宜动作
    # 实测：只靠白名单，大模型会一直选 refresh，结果把一个 0.84 的系统拖到 0.56
    if (sug["action"] in _CHEAP
            and state.last_action == sug["action"]
            and state.last_gain < cfg.min_gain):
        return {**base, "llm": sug,
                "note": f"上次 {sug['action']} 仅 {state.last_gain:+.3f}，拒绝重复无效动作"}

    # 闸四：已接近历史最好时，拒绝大模型的重训建议
    if sug["action"] == "refit_model" and state.current_score >= state.best_score - 0.01:
        return {**base, "llm": sug, "note": "当前已接近历史最好，拒绝大模型的重训建议"}

    if sug["action"] == base["action"]:
        return {**base, "llm": sug,
                "source": base["source"] + "+llm",
                "note": "大模型与当前判断一致"}

    return {
        "action": sug["action"],
        "hypothesis": f"大模型建议：{sug.get('reason', '')}",
        "reason": f"采纳大模型建议（原判 {base['action']}，来源 {base['source']}）",
        "cost_sec": cost.get(sug["action"], 0.0),
        "source": "llm",
        "llm": sug,
        "memory": base.get("memory"),
    }

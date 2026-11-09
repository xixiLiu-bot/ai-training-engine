"""把「规则引擎」「经验记忆」「大模型建议」合成一个决策出口。

优先级（后面的可以覆盖前面的，但都要过闸）：

    规则引擎   → 永远先算，作为保底
    经验记忆   → 有足够证据时，可以改判（免费，先用）
    大模型     → 最后提建议，过四道闸才采纳

任何一层出问题，都退回上一层，永远不会卡死。
"""
from __future__ import annotations

from ..loop import decide as dc


def decide(state, diagnosis, cfg, agent=None, memory=None) -> dict:
    """返回 {action, hypothesis, reason, cost_sec, source, llm, memory}。"""
    rule = dc.decide(state, diagnosis, cfg)
    base = {**rule, "source": "rule", "llm": None, "memory": None}

    # ── 第二层：经验记忆 ──
    # 保留 20% 的探索：不能因为一次成功就锁死，否则会陷在过时的经验里
    explore = (memory is not None and getattr(memory, "rng", None) is not None
               and memory.rng.random() < memory.epsilon)
    if memory is not None and not explore:
        best = memory.best_action(diagnosis["kind"], list(dc.ACTIONS))
        if best is not None:
            act, gpc = best
            cur = memory.gain_per_cost_of(diagnosis["kind"], rule["action"])
            # 只有明显更好才改判，避免早期一两次实验就把系统带偏
            if act != rule["action"] and (cur is None or gpc >= cur * memory.advantage):
                base = {
                    "action": act,
                    "hypothesis": f"经验表显示：{diagnosis['kind']} 类漂移下 {act} 更划算",
                    "reason": (f"经验：{act} 单位成本增益 {gpc:.2f}"
                               + (f"，规则选的 {rule['action']} 只有 {cur:.2f}" if cur else "")),
                    "cost_sec": dc.COST[act],
                    "source": "memory",
                    "llm": None,
                    "memory": {"chosen": act, "gain_per_cost": round(gpc, 3),
                               "rule_action": rule["action"],
                               "rule_gain_per_cost": None if cur is None else round(cur, 3)},
                }

    # ── 第三层：大模型（在记忆的基础上再建议） ──
    if agent is None or not getattr(agent, "enabled", False):
        return base

    catalog = {k: {"cost_sec": v} for k, v in dc.COST.items()}
    state_dict = {
        "batch_id": state.batch_id,
        "psi_mean": state.psi_mean,
        "current_score": state.current_score,
        "best_score": state.best_score,
        "steps_since_update": state.steps_since_update,
        "last_action": state.last_action or "无",
        "last_gain": state.last_gain,
    }

    sug = agent.suggest(state_dict, diagnosis, catalog, base["action"])

    if not sug or "action" not in sug:
        return {**base, "llm": sug, "note": "大模型不可用，已降级"}

    if sug["action"] not in dc.COST:
        return {**base, "llm": sug, "note": f"动作 {sug['action']} 不在白名单，已拒绝"}

    if (sug["action"] == "refresh_normalization"
            and state.last_action == "refresh_normalization"
            and state.last_gain < cfg.min_gain):
        return {**base, "llm": sug,
                "note": f"上次校准仅 {state.last_gain:+.3f}，拒绝重复无效动作"}

    if sug["action"] == "refit_model" and state.current_score >= state.best_score - 0.01:
        return {**base, "llm": sug, "note": "已接近历史最好，拒绝大模型的重训建议"}

    if sug["action"] == base["action"]:
        return {**base, "llm": sug,
                "source": base["source"] + "+llm",
                "note": "大模型与当前判断一致"}

    return {**base,
            "action": sug["action"],
            "hypothesis": f"大模型建议：{sug.get('reason', '')}",
            "reason": f"采纳大模型（原判 {base['action']}，来源 {base['source']}）",
            "cost_sec": dc.COST[sug["action"]],
            "source": "llm",
            "llm": sug}

"""把「规则引擎」和「大模型建议」合成一个决策出口。

顺序固定：先规则，再问大模型，最后过三道闸。

    ┌─────────┐
    │ 规则引擎 │  ← 永远先算一遍，作为保底与对照
    └────┬────┘
         │ rule_action
    ┌────▼─────┐
    │ 大模型建议 │  ← 可选；超时/报错/胡说 → 返回 None
    └────┬─────┘
         │ suggestion
    ┌────▼──────────────────┐
    │ 三道闸：白名单 / 预算 / 可解析 │
    └────┬──────────────────┘
         │
    采纳 or 降级到 rule_action
"""
from __future__ import annotations

from ..loop import decide as dc


def decide(state, diagnosis, cfg, agent=None) -> dict:
    """返回 {action, hypothesis, reason, cost_sec, source, llm}。"""
    rule = dc.decide(state, diagnosis, cfg)

    if agent is None or not getattr(agent, "enabled", False):
        return {**rule, "source": "rule", "llm": None}

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

    sug = agent.suggest(state_dict, diagnosis, catalog, rule["action"])

    # 闸一：没给建议 / 给的是错误信息
    if not sug or "action" not in sug:
        return {**rule, "source": "rule", "llm": sug,
                "note": "大模型不可用，已降级到规则"}

    # 闸二：动作必须真的存在（agent 内部已校验，这里再兜一层）
    if sug["action"] not in dc.COST:
        return {**rule, "source": "rule", "llm": sug,
                "note": f"动作 {sug['action']} 不在白名单，已拒绝"}

    # 闸三：不许重复“已经被证明无效”的便宜动作
    # 这是实测出来的：只靠白名单和预算，大模型会一直选 refresh，
    # 而 refresh 早就不管用了，结果把一个 0.84 的系统拖到 0.56。
    if (sug["action"] == "refresh_normalization"
            and state.last_action == "refresh_normalization"
            and state.last_gain < cfg.min_gain):
        return {**rule, "source": "rule", "llm": sug,
                "note": f"上次校准仅 {state.last_gain:+.3f}，拒绝重复无效动作"}

    # 闸四：预算（贵动作要额外确认，避免被大模型带飞）
    if sug["action"] == "refit_model" and state.current_score >= state.best_score - 0.01:
        return {**rule, "source": "rule", "llm": sug,
                "note": "当前已接近历史最好，拒绝大模型的重训建议"}

    if sug["action"] == rule["action"]:
        return {**rule, "source": "rule+llm", "llm": sug,
                "note": "大模型与规则一致"}

    return {
        "action": sug["action"],
        "hypothesis": f"大模型建议：{sug.get('reason', '')}",
        "reason": f"采纳大模型建议（规则原判 {rule['action']}）",
        "cost_sec": dc.COST[sug["action"]],
        "source": "llm",
        "llm": sug,
    }

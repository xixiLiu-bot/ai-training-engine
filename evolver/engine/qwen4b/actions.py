"""动作空间：让 4B 只做「三选一」，不让它写自由文本决策。

设计理由（这是整个包的核心）：
4B 的短板是「严格 JSON」和「长指令遵循」，强项是「短分类」。所以把任务
改造成 3 路分类题，而不是让模型生成结构化决策。这样：

    27B 能做的事        4B 也能做对
    ─────────────────────────────────
    读 6 个数字 → 选 1 个动作

代价是理由字段由本地模板拼（不花 token，也永远不会解析失败）。
"""
from __future__ import annotations

DO_NOTHING = "do_nothing"
REFRESH = "refresh_normalization"
REFIT = "refit_model"

# 顺序很重要：解析时要先匹配长名字，否则 refresh_normalization 会被误判
ACTIONS: tuple[str, ...] = (REFRESH, REFIT, DO_NOTHING)

# 与 evolver/loop/decide.py 的 COST 保持一致（单位：秒）
COST: dict[str, float] = {DO_NOTHING: 0.0, REFRESH: 0.05, REFIT: 2.0}

# 动作的中文释义：只用于拼理由，不参与决策
ACTION_ZH: dict[str, str] = {
    DO_NOTHING: "不动",
    REFRESH: "零成本校准",
    REFIT: "重训",
}

# 4B 在自由文本模式下可能吐中文/近义词，兜底映射一下
ALIASES: dict[str, str] = {
    "do_nothing": DO_NOTHING,
    "donothing": DO_NOTHING,
    "none": DO_NOTHING,
    "noop": DO_NOTHING,
    "keep": DO_NOTHING,
    "no_action": DO_NOTHING,
    "不动": DO_NOTHING,
    "什么都不做": DO_NOTHING,
    "保持": DO_NOTHING,
    "维持": DO_NOTHING,
    "refresh_normalization": REFRESH,
    "refresh": REFRESH,
    "recalibrate": REFRESH,
    "calibrate": REFRESH,
    "normalization": REFRESH,
    "校准": REFRESH,
    "标准化": REFRESH,
    "重算标准化": REFRESH,
    "refit_model": REFIT,
    "refit": REFIT,
    "retrain": REFIT,
    "re_train": REFIT,
    "retrain_model": REFIT,
    "重训": REFIT,
    "重新训练": REFIT,
    "增量微调": REFIT,
}


_SENTENCE_MARKS = "。！？.!?\n；;，,"


def normalize_action(raw: str | None, strict: bool = False) -> str | None:
    """把模型吐出来的写法收敛到白名单，收敛不了就**返回 None**。

    `strict=True`（结构化输出走这条）：只接受精确匹配/精确近义词。
        目的：拦住幻觉。`{"action": "retrain_everything"}` 必须被拒，
        而不是被子串匹配"猜"成 refit_model——那等于替模型编造决策。

    `strict=False`（自由文本兜底走这条）：允许子串匹配，但有硬前提——
        文本必须短、且不含句读。否则一句"不需要重训，保持不动"
        会因为出现"重训"二字被误判成 refit_model。
    """
    if not raw:
        return None
    key = str(raw).strip().strip("\"'`。.,， ").lower()
    if not key:
        return None
    if key in ALIASES:
        return ALIASES[key]
    if strict:
        return None
    if len(key) > 30 or any(mark in key for mark in _SENTENCE_MARKS):
        return None
    for token in (REFRESH, REFIT, DO_NOTHING):
        if token in key:
            return token
    for alias, action in ALIASES.items():
        if alias and alias in key:
            return action
    return None


# 给支持 guided decoding 的后端用：输出被强制压成这 3 个 token 串之一
GUIDED_CHOICE: list[str] = list(ACTIONS)

# 给只支持 JSON Schema 的后端用
DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "reason": {"type": "string", "maxLength": 60},
    },
    "required": ["action"],
    "additionalProperties": False,
}


def reason_for(action: str, diagnosis: dict | None = None) -> str:
    """本地拼理由：不花 token，也永远不会解析失败。"""
    kind = (diagnosis or {}).get("kind", "unknown")
    base = {
        DO_NOTHING: "判为无需动作",
        REFRESH: "判为零成本校准",
        REFIT: "判为需要重训",
    }[action]
    return f"4B {base}：归因={kind}"


def catalog_text(cost: dict | None = None) -> str:
    """给模型看的动作清单。

    cost 传入时用真实成本（每个模型不同：logreg 重训 0.5s、gbdt 2.2s），
    不传则退回本模块写死的默认值（仅作兜底）。
    """
    table = cost or COST
    return " | ".join(
        f"{a}({table.get(a, COST[a])}s)" for a in (DO_NOTHING, REFRESH, REFIT))

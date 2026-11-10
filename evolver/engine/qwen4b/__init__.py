"""qwen4b —— 给 Qwen3.5-4B 用的决策层（纯标准库，零外部依赖）。

替换 `evolver/engine/llm_agent.py` 即可接入现有闭环，上层代码一行不用改。
"""
from .actions import (ACTIONS, COST, DECISION_SCHEMA, DO_NOTHING, GUIDED_CHOICE,
                      REFIT, REFRESH, normalize_action, reason_for)
from .client import (CapabilityError, ChatClient, Completion, LLMError,
                     ServerError, TransportError)
from .decider import STRATEGIES, DisabledDecider, Qwen4BDecider
from .parsing import parse_decision

__all__ = [
    "ACTIONS", "COST", "DECISION_SCHEMA", "DO_NOTHING", "REFRESH", "REFIT",
    "GUIDED_CHOICE", "normalize_action", "reason_for",
    "ChatClient", "Completion", "LLMError", "TransportError",
    "CapabilityError", "ServerError",
    "Qwen4BDecider", "DisabledDecider", "STRATEGIES",
    "parse_decision",
]

__version__ = "1.0.0"

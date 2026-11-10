"""把 `evolver.engine.llm_agent.build()` 的返回接到 4B 决策器上。

这一层的存在意义：**让上层代码零改动**。
`policy.decide()` 只认三个成员 —— `.enabled` / `.suggest(...)` / `.stats()`，
所以换决策器不需要动闭环、门控、账本任何一行。

关于 PyTRiO（实测口径，2026-11）：

    base_url  https://pytrio.com/api/openai/v1
    模型 id   带命名空间前缀：Qwen/Qwen3.5-4B、Qwen/Qwen3.8-27B
    思考模式  **必须显式关掉**，参数是顶层 enable_thinking: false
              （chat_template_kwargs 会被 400 拒绝，/no_think 后缀无效）
    约束解码  response_format(json_schema / json_object) 生效；
              guided_choice / guided_json 返回 200 但**不生效**（静默忽略）
"""
from __future__ import annotations

import os

from .decider import DisabledDecider, Qwen4BDecider

# 后端名 -> (base_url 环境变量, model 环境变量, 默认 model)
_BACKENDS: dict[str, tuple[str, str, str]] = {
    "qwen4b": ("QWEN_BASE_URL", "QWEN_4B_MODEL", "Qwen/Qwen3.5-4B"),
    "pytrio": ("QWEN_BASE_URL", "QWEN_MODEL", "Qwen/Qwen3.8-27B"),
    "ollama": ("OLLAMA_BASE_URL", "OLLAMA_MODEL", "qwen3:4b"),
}

_OLLAMA_DEFAULT = "http://127.0.0.1:11434/v1"
_PYTRIO_DEFAULT = "https://pytrio.com/api/openai/v1"

# 不同模型对 max_tokens 的需求差一个量级：4B 约 80，27B 要 1024+
_DEFAULT_MAX_TOKENS = {"qwen4b": 96, "pytrio": 1024, "ollama": 128}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _truthy(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def build(backend: str = "qwen4b", **overrides) -> Qwen4BDecider | DisabledDecider:
    """和原版 `llm_agent.build()` 同签名，可直接替换。"""
    if backend in ("", "none", None):
        return DisabledDecider()

    url_env, model_env, default_model = _BACKENDS.get(backend, _BACKENDS["qwen4b"])
    base_url = overrides.get("base_url") or os.getenv(url_env, "")
    if not base_url and backend == "ollama":
        base_url = _OLLAMA_DEFAULT
    if not base_url and backend in ("qwen4b", "pytrio"):
        # PyTRiO 的地址是固定的，没配就用默认值，少一个出错点
        base_url = os.getenv("PYTRIO_BASE_URL", _PYTRIO_DEFAULT)

    api_key = overrides.get("api_key") or os.getenv(
        "QWEN_API_KEY", "ollama" if backend == "ollama" else "EMPTY"
    )
    model = overrides.get("model") or os.getenv(model_env, default_model)

    if not base_url:
        # 没配地址时**不要崩**，只是不启用 —— 离线评测里崩一次就是 0 分
        return DisabledDecider()

    # PyTRiO / Qwen3.x 默认开思考，不关掉就只会返回 reasoning_content
    extra_body = overrides.get("extra_body")
    if extra_body is None:
        extra_body = {}
        if backend in ("qwen4b", "pytrio", "ollama") and _truthy(
                "QWEN_DISABLE_THINKING", True):
            extra_body["enable_thinking"] = False

    return Qwen4BDecider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        backend=backend,
        timeout=overrides.get("timeout") or _float("QWEN_TIMEOUT_SECONDS", 30.0),
        max_attempts=overrides.get("max_attempts") or _int("QWEN_4B_MAX_ATTEMPTS", 2),
        budget_tokens=overrides.get("budget_tokens")
        if overrides.get("budget_tokens") is not None
        else _int("QWEN_4B_BUDGET_TOKENS", 20000),
        max_consecutive_failures=overrides.get("max_consecutive_failures")
        or _int("QWEN_4B_MAX_FAILURES", 3),
        max_calls=overrides.get("max_calls")
        if overrides.get("max_calls") is not None
        else _int("QWEN_4B_MAX_CALLS", 0),
        deadline_sec=overrides.get("deadline_sec")
        or _float("QWEN_4B_DEADLINE_SEC", 120.0),
        min_psi_for_refit=overrides.get("min_psi_for_refit")
        if overrides.get("min_psi_for_refit") is not None
        else _float("QWEN_4B_MIN_PSI_REFIT", 0.10),
        max_tokens=overrides.get("max_tokens")
        or _int("QWEN_MAX_TOKENS", _DEFAULT_MAX_TOKENS.get(backend, 96)),
        max_tokens_cap=overrides.get("max_tokens_cap")
        or _int("QWEN_MAX_TOKENS_CAP", 2048),
        extra_body=extra_body,
    )

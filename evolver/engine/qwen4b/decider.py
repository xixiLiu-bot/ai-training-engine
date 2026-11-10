"""4B 决策器：约束解码 → 修复重试 → 预算 → 熔断 → 记账。

和原版 `llm_agent.py` 的区别只有一句话：
**原版赌模型一次就吐对 JSON；这一版假设它一定会吐错，然后设计好吐错之后怎么办。**

四条必须守住的性质：

    1. 不抛异常            —— suggest() 任何路径都返回 dict 或 None
    2. 不阻塞              —— 单次决策有硬截止时间
    3. 不拖累分数          —— 解析失败自动降级到规则，分数不低于纯规则
    4. 不隐瞒成本          —— 每次调用的 token 都回填，失败的那次也算

另外三个坑是实测出来的（见 README「PyTRiO 实测」）：

    5. **思考模式必须显式关掉**（`enable_thinking: false`）。否则 content 恒为 null，
       全部 token 花在 reasoning_content 上，看起来像"模型什么也没说"。
    6. **约束解码可能被静默忽略**：PyTRiO 对 `guided_choice` / `guided_json`
       返回 200 但完全不生效。所以每次成功都要核对"输出形态是否与策略相符"，
       不符就永久降级——否则会假装自己在受约束，实际全是裸文本。
    7. **不同模型需要的 max_tokens 差一个量级**（4B 约 80，27B 要 1024+）。
       遇到"被长度截断且 content 为空"就自动放大预算重试，而不是当成失败。
"""
from __future__ import annotations

import re
import time

from .actions import (ACTIONS, DECISION_SCHEMA, GUIDED_CHOICE, REFIT, reason_for)
from .client import (CapabilityError, ChatClient, LLMError, ServerError,
                     TransportError)
from .parsing import parse_decision
from .prompts import build_messages, repair_messages

# 解码策略，从「最可信」到「最兜底」。
# 顺序是**按实测效果排的**，不是按理论优雅度：
#   json_schema / json_object 在 PyTRiO 上真的生效；
#   guided_choice / guided_json 会被静默忽略（200 但不生效），所以排在后面。
STRATEGIES: tuple[str, ...] = (
    "json_schema",     # OpenAI 风格: response_format(json_schema) —— PyTRiO 生效
    "json_object",     # OpenAI 风格: response_format(json_object) —— PyTRiO 生效
    "guided_choice",   # vLLM 原生: 三个动作名之一
    "guided_json",     # vLLM 原生: JSON Schema 约束
    "text",            # 什么都不支持时的裸文本
)

# 每种策略「诚实生效」时，解析结果应该长什么样
_HONORED_HOW = {
    "json_schema": {"json", "json_scan"},
    "json_object": {"json", "json_scan"},
    "guided_json": {"json", "json_scan"},
    "guided_choice": {"bare"},
    "text": {"bare", "field", "keyword", "json", "json_scan"},
}

_PARAM_IN_ERROR = re.compile(
    r"(guided_choice|guided_json|response_format|enable_thinking|"
    r"chat_template_kwargs|reasoning_effort)", re.I)


def payload_for(strategy: str, max_tokens: int) -> dict:
    if strategy == "guided_choice":
        return {"guided_choice": GUIDED_CHOICE, "max_tokens": min(max_tokens, 16),
                "stop": ["\n"]}
    if strategy == "guided_json":
        return {"guided_json": DECISION_SCHEMA, "max_tokens": max_tokens}
    if strategy == "json_schema":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "strict": True,
                                "schema": DECISION_SCHEMA},
            },
            "max_tokens": max_tokens,
        }
    if strategy == "json_object":
        return {"response_format": {"type": "json_object"}, "max_tokens": max_tokens}
    return {"max_tokens": min(max_tokens, 32), "stop": ["\n"]}


class Qwen4BDecider:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen3.5-4B",
        backend: str = "qwen4b",
        timeout: float = 30.0,
        max_attempts: int = 2,
        budget_tokens: int = 0,
        max_consecutive_failures: int = 3,
        max_calls: int = 0,
        deadline_sec: float = 120.0,
        min_psi_for_refit: float = 0.10,
        max_tokens: int = 96,
        max_tokens_cap: int = 2048,
        extra_body: dict | None = None,
        enabled: bool = True,
        client: ChatClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.backend = backend
        self.mode = "openai"
        self.enabled = bool(enabled) and bool(base_url)
        self.max_attempts = max(1, int(max_attempts))
        self.budget_tokens = int(budget_tokens)
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.max_calls = int(max_calls)
        self.deadline_sec = float(deadline_sec)
        self.min_psi_for_refit = float(min_psi_for_refit or 0.0)
        self.max_tokens = int(max_tokens)
        self.max_tokens_cap = int(max_tokens_cap)
        # 平台私有参数，例如 PyTRiO 的 {"enable_thinking": False}
        self.extra_body = dict(extra_body or {})
        self.client = client or ChatClient(base_url, api_key, model, timeout)

        # 计数（stats() 会原样打出来，也进 decisions.jsonl）
        self.tokens_used = 0
        self.calls = 0
        self.failures = 0
        self.filtered = 0
        self.skipped = 0
        self.network_calls = 0
        self.consecutive_failures = 0
        self.tripped = False
        self.trip_reason = ""
        self.dropped_params: set[str] = set()
        self.silently_ignored: dict[str, str] = {}

        # 自适应策略：探测到某个策略不被支持就永久降级，不再浪费请求
        self.strategy_i = 0
        self.strategy_ok: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 对外唯一接口（和原版 LLMAgent.suggest 完全同签名）
    # ------------------------------------------------------------------
    def suggest(self, state: dict, diagnosis: dict, catalog: dict,
                rule_action: str) -> dict | None:
        if not self.enabled:
            return None
        if self.tripped:
            self.skipped += 1
            return None
        if self.max_calls and self.network_calls >= self.max_calls:
            self._trip(f"调用上限 {self.max_calls} 已达")
            self.skipped += 1
            return None
        if self.budget_tokens and self.tokens_used >= self.budget_tokens:
            self._trip(f"Token 预算 {self.budget_tokens} 已用尽")
            self.skipped += 1
            return None

        t0 = time.monotonic()
        start_tokens = self.tokens_used
        base_messages = build_messages(state, diagnosis, rule_action)

        messages = base_messages
        repairs = 0
        attempts = 0
        last_error = "未发起调用"
        tried = 0
        max_tries = len(STRATEGIES) + self.max_attempts + 2

        while tried < max_tries:
            tried += 1
            if self.strategy_i >= len(STRATEGIES):
                last_error = "所有解码策略均不被支持"
                break
            if time.monotonic() - t0 > self.deadline_sec:
                last_error = f"超出单次决策时限 {self.deadline_sec}s"
                break
            if self.max_calls and self.network_calls >= self.max_calls:
                self._trip(f"调用上限 {self.max_calls} 已达")
                last_error = "调用上限已达"
                break

            strategy = STRATEGIES[self.strategy_i]
            payload = payload_for(strategy, self.max_tokens)
            if self.extra_body:
                payload["extra_body"] = self.extra_body

            self.network_calls += 1      # 失败的那次也要计数，它是真花的钱
            attempts += 1
            try:
                comp = self.client.complete(messages, **payload)
            except CapabilityError as exc:
                # 先看是不是"我们自己注入的某个参数"不被支持 → 拔掉它重试
                dropped = self._drop_unsupported(str(exc))
                if dropped:
                    last_error = f"平台不支持 {dropped}，已移除该参数"
                    continue
                # 否则是这个解码策略不被支持 → 降级
                self.strategy_i += 1
                last_error = f"{strategy} 不被支持: {exc}"
                continue
            except (TransportError, ServerError) as exc:
                # 网络断 / 服务端崩：换解码策略没有意义，立刻收手
                last_error = f"{type(exc).__name__}: {exc}"
                break
            except LLMError as exc:                     # 兜底，理论上到不了
                last_error = str(exc)
                break

            self.tokens_used += comp.total_tokens

            if self.budget_tokens and self.tokens_used >= self.budget_tokens:
                self._trip(f"Token 预算 {self.budget_tokens} 已用尽")
                last_error = "Token 预算用尽"
                break

            # 坑 5：思考模式吃光预算 → 放大 max_tokens 重试，不记为失败
            if comp.thinking_ate_budget:
                if self.max_tokens < self.max_tokens_cap:
                    self.max_tokens = min(self.max_tokens * 4, self.max_tokens_cap)
                    last_error = (f"{self.model} 在思考模式里耗光 token，"
                                  f"已把 max_tokens 提到 {self.max_tokens}")
                    continue
                last_error = "思考模式耗光 token，且已达上限"
                break

            action, reason, how = parse_decision(comp.text)
            if action and not self._strategy_honored(strategy, how):
                # 坑 6：策略声称生效但输出形态不符 → 静默忽略，永久降级
                self.silently_ignored[strategy] = how
                self.strategy_i += 1
                last_error = f"{strategy} 被静默忽略（输出形态={how}）"
                continue

            if action:
                self.calls += 1
                self.consecutive_failures = 0
                self.strategy_ok[strategy] = self.strategy_ok.get(strategy, 0) + 1
                tokens_used = self.tokens_used - start_tokens

                # 成本预筛：分布稳定（PSI 低于稳定阈值）时重训**不可能**有增益，
                # 花 2 秒去验证一个必然为负的假设是纯浪费——客观分的分母就这么被撑大的。
                # 注意这不算"模型变傻"，不计入熔断。
                psi = float(state.get("psi_mean", 0.0) or 0.0)
                if (action == REFIT and self.min_psi_for_refit > 0
                        and psi < self.min_psi_for_refit):
                    self.filtered += 1
                    return {
                        "error": (f"refit 被成本预筛拦下：PSI {psi:.3f} < "
                                  f"{self.min_psi_for_refit}，重训不可能有增益"),
                        "filtered": True,
                        "strategy": strategy,
                        "attempts": attempts,
                        "tokens": tokens_used,
                        "latency": round(time.monotonic() - t0, 3),
                        "backend": self.backend,
                        "model": self.model,
                    }

                return {
                    "action": action,
                    "reason": reason or reason_for(action, diagnosis),
                    "how": how,
                    "strategy": strategy,
                    "attempts": attempts,
                    "tokens": tokens_used,
                    "usage_estimated": comp.usage_estimated,
                    "latency": round(time.monotonic() - t0, 3),
                    "backend": self.backend,
                    "model": self.model,
                }

            # 解析失败 → 把坏输出喂回去重问一次（4B 对此非常敏感）
            last_error = f"{strategy} 输出无法解析: {comp.text[:60]!r}"
            if repairs < self.max_attempts - 1:
                repairs += 1
                messages = repair_messages(base_messages, comp.text)
                continue
            break

        # 一次决策只记一次失败：重试/降级是同一件事，不该把熔断器触发得更快
        self._record_failure()
        return {
            "error": last_error[:200],
            "strategy": STRATEGIES[min(self.strategy_i, len(STRATEGIES) - 1)],
            "attempts": attempts,
            "tokens": self.tokens_used - start_tokens,
            "latency": round(time.monotonic() - t0, 3),
            "backend": self.backend,
            "model": self.model,
        }

    # ------------------------------------------------------------------
    def _strategy_honored(self, strategy: str, how: str) -> bool:
        return how in _HONORED_HOW.get(strategy, {"json", "json_scan", "bare"})

    def _drop_unsupported(self, message: str) -> str | None:
        """如果 400 报的是我们自己注入的参数，就把它拔掉，返回参数名。"""
        m = _PARAM_IN_ERROR.search(message or "")
        if not m:
            return None
        name = m.group(1)
        if name in self.extra_body:
            self.extra_body.pop(name, None)
            self.dropped_params.add(name)
            return name
        return None

    def _record_failure(self) -> None:
        self.failures += 1
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self._trip(f"连续 {self.consecutive_failures} 次失败")

    def _trip(self, reason: str) -> None:
        self.tripped = True
        self.trip_reason = reason

    def reset(self) -> None:
        self.tokens_used = 0
        self.calls = 0
        self.failures = 0
        self.filtered = 0
        self.skipped = 0
        self.network_calls = 0
        self.consecutive_failures = 0
        self.tripped = False
        self.trip_reason = ""
        self.strategy_i = 0
        self.strategy_ok = {}
        self.silently_ignored = {}

    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "filtered": self.filtered,
            "skipped": self.skipped,
            "tokens": self.tokens_used,
            "network_calls": self.network_calls,
            "strategy": STRATEGIES[min(self.strategy_i, len(STRATEGIES) - 1)],
            "strategies_ok": dict(self.strategy_ok),
            "silently_ignored": dict(self.silently_ignored),
            "dropped_params": sorted(self.dropped_params),
            "max_tokens": self.max_tokens,
            "tripped": self.tripped,
            "trip_reason": self.trip_reason,
        }


class DisabledDecider:
    """`--llm none` 的占位：保持同样的接口，但一次网络都不发。"""

    enabled = False
    mode = "none"
    backend = "none"
    model = "-"
    tokens_used = 0
    calls = 0
    failures = 0

    def suggest(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def reset(self) -> None:
        return None

    def stats(self) -> dict:
        return {"backend": "none", "model": "-", "calls": 0,
                "failures": 0, "filtered": 0, "tokens": 0, "tripped": False}

"""OpenAI 兼容的极简客户端，**只用标准库**。

为什么不直接用 `openai` SDK（这是原版代码的一个真 bug）：

    evolver/engine/llm_agent.py 顶层 `from openai import OpenAI`
    requirements.txt 里却没有 openai
    → 官方 2G 离线镜像里只要少了这个包，`import llm_agent` 就炸，
      连 `--llm none` 都跑不起来，整条流水线起不来 = 0 分。

这个客户端只用 urllib，零依赖，也不吃 SDK 的版本坑。
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(Exception):
    """所有 LLM 相关错误的基类。调用方只需 catch 这一个。"""


class TransportError(LLMError):
    """连不上 / 超时 / DNS 失败 —— 网络层问题。"""


class CapabilityError(LLMError):
    """4xx —— 多半是这个后端不支持我们请求的参数（如 guided_choice）。

    单独分出来，是为了能立刻降级到下一个解码策略，而不是当成"模型变傻"。
    """


class AuthError(LLMError):
    """401 / 403 —— 没 Key 或 Key 无效。

    单独分出来是因为它**不该重试**：换解码策略、换参数都救不回一个坏 Key，
    重试只会浪费配额，还会把"缺少 Key"伪装成"模型不稳定"。
    """


class ServerError(LLMError):
    """5xx —— 服务端崩了。"""


@dataclass
class Completion:
    text: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    usage_estimated: bool = False
    finish_reason: str = ""
    has_reasoning: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def truncated_by_length(self) -> bool:
        return self.finish_reason == "length"

    @property
    def thinking_ate_budget(self) -> bool:
        """思考模式把 max_tokens 吃光了：content 空 + 有 reasoning + 被长度截断。

        这是 Qwen3.x 在 PyTRiO 上**默认就会踩**的坑：不给
        `enable_thinking: false`，content 恒为 null，看起来像"模型什么也没说"。
        """
        return (not self.text.strip()) and self.has_reasoning and self.truncated_by_length


def estimate_tokens(text: str) -> int:
    """没有 usage 字段时的兜底估算。

    Qwen 的分词器对中文大约 1 字 1 token，英文约 4 字符 1 token。
    只用来记账，不追求精确——但**必须记**，因为 Token 是客观分的分母。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


class ChatClient:
    """对 `/chat/completions` 的一层薄封装，带类型化的异常。"""

    def __init__(self, base_url: str, api_key: str = "EMPTY",
                 model: str = "Qwen3.5-4B", timeout: float = 8.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.timeout = float(timeout)

    # ------------------------------------------------------------------
    def complete(self, messages: list[dict], *, max_tokens: int = 24,
                 temperature: float = 0.0, stop: list[str] | None = None,
                 response_format: dict | None = None,
                 guided_choice: list[str] | None = None,
                 guided_json: dict | None = None,
                 extra_body: dict | None = None) -> Completion:
        if not self.base_url:
            raise TransportError("base_url 为空")

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        # vLLM / SGLang 的约束解码参数，走顶层字段
        if guided_choice is not None:
            payload["guided_choice"] = guided_choice
        if guided_json is not None:
            payload["guided_json"] = guided_json
        # 平台私有参数（如 PyTRiO 的 enable_thinking），由调用方注入
        if extra_body:
            payload.update(extra_body)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # 有些网关会直接重置默认 Python-urllib UA 的请求
                "User-Agent": "zhixun-engine/0.1",
                "Accept": "application/json",
            },
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if exc.code in (401, 403):
                raise AuthError(f"HTTP {exc.code}: {detail}") from exc
            if 400 <= exc.code < 500:
                raise CapabilityError(f"HTTP {exc.code}: {detail}") from exc
            raise ServerError(f"HTTP {exc.code}: {detail}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise TransportError(f"timeout after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            raise TransportError(f"URLError: {exc.reason}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise ServerError(f"响应不是合法 JSON: {exc}") from exc
        except OSError as exc:
            # ConnectionResetError / ConnectionAbortedError 等裸 OS 级错误。
            # 原来没兜住，网络抖一下就会把整轮跑挂 —— 门槛①要求无人干预跑完。
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc

        latency = time.monotonic() - t0
        return self._parse(raw, latency, messages)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: dict, latency: float, messages: list[dict]) -> Completion:
        choices = raw.get("choices") or []
        text = ""
        finish_reason = ""
        has_reasoning = False
        if choices:
            choice = choices[0]
            finish_reason = str(choice.get("finish_reason") or "")
            msg = choice.get("message") or {}
            has_reasoning = bool(msg.get("reasoning_content"))
            content = msg.get("content")
            if isinstance(content, list):        # 有的后端返回分段内容
                content = "".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict)
                )
            text = (content or "")

        usage = raw.get("usage") or {}
        total = int(usage.get("total_tokens") or 0)
        estimated = False
        if total <= 0:
            prompt_text = "".join(str(m.get("content", "")) for m in messages)
            total = estimate_tokens(prompt_text) + estimate_tokens(text)
            estimated = True

        return Completion(
            text=text,
            total_tokens=total,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency=latency,
            usage_estimated=estimated,
            finish_reason=finish_reason,
            has_reasoning=has_reasoning,
            raw=raw,
        )

"""决策大脑（可选层）：用大模型给闭环“提建议”，但绝不进主路径。

设计原则：

1. **它不是主流程**。规则引擎 `loop/decide.py` 才是。
2. **它只能“建议”**。建议必须过 policy.py 的四道闸才会被采纳。
3. **失败一律降级**。超时 / 连不上 / 返回垃圾 → 返回 None，规则接管。
   门槛① 要求“无人干预跑完”，所以这一层不能有任何阻塞点。

后端（同一个 OpenAI 兼容协议，可随时切换）：

    none            关闭（默认）
    mock            手写假模型，验证「提示词 → 解析 → 校验」
    mock_bad        故意返回垃圾，验证降级
    pydantic-test   ★ Pydantic AI 框架 + TestModel，离线验证框架链路
    pydantic        ★ Pydantic AI 框架 + 任意 OpenAI 兼容端点（企业赛/Qwen）
    pytrio          手写 HTTP 直连（保留，兼容性最好）
    ollama          本地 ollama

为什么用 Pydantic AI：

我们把「动作必须在三档之内」写成 `Literal[...]`，框架会在模型返回时做
schema 校验；不合格就自动重试，重试还不行就抛异常被我们降级。
**白名单从“事后用正则检查”变成了“结构约束”。**
"""
from __future__ import annotations

import json
import re
import time
from typing import Literal

from pydantic import BaseModel, Field

PRESETS = {
    "none": {"enabled": False},
    "mock": {"enabled": True, "mode": "mock", "model": "mock-llm"},
    "mock_bad": {"enabled": True, "mode": "mock", "model": "mock-bad"},
    # ★ 框架 + 离线假模型：不需要网络，也不需要下载任何权重
    "pydantic-test": {
        "enabled": True, "mode": "pydantic", "model": "TestModel",
        "base_url": None, "api_key": "EMPTY", "use_test_model": True,
    },
    # ★ 框架 + 模拟模型：仍走 Pydantic AI，但决策由本地启发式给出（验收用）
    "pydantic-sim": {
        "enabled": True, "mode": "pydantic", "model": "SimModel",
        "base_url": None, "api_key": "EMPTY", "use_test_model": True,
        "simulate": True,
    },
    # ★ 框架 + 真模型：改成赛题组给的地址即可
    "pydantic": {
        "enabled": True, "mode": "pydantic", "model": "Qwen3.8-27B",
        "base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY",
        "use_test_model": False,
    },
    "pytrio": {
        "enabled": True, "mode": "openai",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen3.8-27B",
        "api_key": "EMPTY",
    },
    "ollama": {
        "enabled": True, "mode": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "api_key": "ollama",
    },
}

SYSTEM_PROMPT = """你是工业模型运维的决策助手。数据在漂移，一个工业模型正在逐渐失准。
你只能从候选动作中选择一个，并给出一句话理由。
原则：能用便宜手段解决，就不要用贵的。"""


class Decision(BaseModel):
    """模型必须返回的结构。action 用 Literal 锁死，非法值框架直接拒绝。"""

    action: Literal["do_nothing", "refresh_normalization", "refit_model"]
    reason: str = Field(default="", max_length=200)


def build_prompt(state: dict, diagnosis: dict, catalog: dict,
                 rule_action: str) -> str:
    """把当前局势翻译成给模型看的说明书。"""
    return f"""## 当前局势
- 批次：第 {state['batch_id']} 批
- 漂移强度 PSI：{state['psi_mean']:.3f}（越高越严重）
- 当前分数：{state['current_score']:.3f}
- 历史最好：{state['best_score']:.3f}
- 距上次更新：{state['steps_since_update']} 批
- 上次动作：{state['last_action']}（增益 {state['last_gain']:+.3f}）
- 归因结论：{diagnosis['kind']}（{diagnosis['reason']}）

## 可选动作（成本单位：秒）
{json.dumps(catalog, ensure_ascii=False, indent=2)}

## 规则引擎的初步判断
{rule_action}

## 要求
选一个动作，并给一句话理由。"""


class LLMAgent:
    """大模型建议器。任何异常都返回 None，由调用方降级到规则。"""

    def __init__(self, backend: str = "none", **overrides):
        preset = dict(PRESETS.get(backend, PRESETS["none"]))
        preset.update({k: v for k, v in overrides.items() if v is not None})
        self.backend = backend
        self.enabled = bool(preset.pop("enabled", False))
        self.mode = preset.pop("mode", "mock")
        self.model = preset.pop("model", "mock-llm")
        self.base_url = preset.pop("base_url", None)
        self.api_key = preset.pop("api_key", "EMPTY")
        self.use_test_model = bool(preset.pop("use_test_model", False))
        self.simulate = bool(preset.pop("simulate", False))
        self.timeout = 10.0
        self.tokens_used = 0
        self.calls = 0
        self.failures = 0
        self._agent = None                 # Pydantic AI 的 Agent，懒加载

    # ---- 对外唯一接口 ----
    def suggest(self, state: dict, diagnosis: dict, catalog: dict,
                rule_action: str) -> dict | None:
        if not self.enabled:
            return None
        prompt = build_prompt(state, diagnosis, catalog, rule_action)
        t0 = time.time()
        try:
            if self.mode == "pydantic":
                parsed = self._pydantic(prompt, catalog)
            else:
                raw = self._call(prompt)
                parsed = self._parse(raw)
            if parsed is None:
                self.failures += 1
                return None
            if parsed["action"] not in catalog:
                self.failures += 1
                return None
            self.calls += 1
            return {**parsed, "latency": round(time.time() - t0, 2),
                    "backend": self.backend, "model": self.model}
        except Exception as e:                      # noqa: BLE001
            self.failures += 1
            return {"error": f"{type(e).__name__}: {e}"[:200]}

    # ---- Pydantic AI：结构化输出，不需要解析 ----
    def _ensure_agent(self):
        if self._agent is not None:
            return self._agent
        from pydantic_ai import Agent

        if self.use_test_model:
            from pydantic_ai.models.test import TestModel
            model = TestModel()
        else:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            model = OpenAIChatModel(
                self.model,
                provider=OpenAIProvider(base_url=self.base_url,
                                        api_key=self.api_key),
            )
        self._agent = Agent(model, output_type=Decision,
                            system_prompt=SYSTEM_PROMPT, retries=2)
        return self._agent

    def _simulate_decision(self, prompt: str) -> dict:
        """模拟一个“还算靠谱的模型”：读 PSI，按成本阶梯给建议。

        它不是真模型，只是让框架路径能跑出有意义的结果。
        真模型来了，这里整段作废。
        """
        m = re.search(r"PSI：([\d.]+)", prompt)
        psi = float(m.group(1)) if m else 0.0
        if psi >= 2.5:
            return {"action": "refit_model", "reason": "漂移显著，建议重训"}
        if psi >= 1.0:
            return {"action": "refresh_normalization", "reason": "先试零成本校准"}
        return {"action": "do_nothing", "reason": "分布稳定，无需动作"}

    def _pydantic(self, prompt: str, catalog: dict) -> dict | None:
        if self.simulate:
            from pydantic_ai import Agent
            from pydantic_ai.models.test import TestModel
            args = self._simulate_decision(prompt)
            sim = TestModel(custom_output_args=args)
            agent = Agent(sim, output_type=Decision,
                          system_prompt=SYSTEM_PROMPT, retries=2)
            result = agent.run_sync(prompt)
            d: Decision = result.output
            if d.action not in catalog:
                return None
            return {"action": d.action, "reason": d.reason}

        agent = self._ensure_agent()
        result = agent.run_sync(prompt)
        d: Decision = result.output            # 已经是 Decision 实例，无需解析
        # 框架只保证类型合法，不保证动作在当前白名单里（比如某档被禁用）
        if d.action not in catalog:
            return None
        return {"action": d.action, "reason": d.reason}

    # ---- 手写后端（保留，兼容性最好）----
    def _call(self, prompt: str) -> str:
        if self.mode == "mock":
            return self._mock(prompt)
        if self.mode == "openai":
            return self._openai(prompt)
        raise RuntimeError(f"未知后端模式 {self.mode}")

    def _openai(self, prompt: str) -> str:
        """任何 OpenAI 兼容端点：PyTRiO / vLLM / ollama / LM Studio。"""
        import requests

        resp = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self.tokens_used += int(data.get("usage", {}).get("total_tokens", 0))
        return data["choices"][0]["message"]["content"]

    def _mock(self, prompt: str) -> str:
        """本地假模型：只用来验证「提示词 → 解析 → 校验 → 采纳」这条链路。"""
        if self.backend == "mock_bad":
            return "我觉得应该重新训练一下吧"      # 故意不返回 JSON
        m = re.search(r"PSI：([\d.]+)", prompt)
        psi = float(m.group(1)) if m else 0.0
        if psi >= 0.25:
            return json.dumps({"action": "refit_model",
                               "reason": "漂移已明显，建议直接重训"},
                              ensure_ascii=False)
        if psi >= 0.10:
            return json.dumps({"action": "refresh_normalization",
                               "reason": "先试零成本校准"},
                              ensure_ascii=False)
        return json.dumps({"action": "do_nothing",
                           "reason": "分布稳定，无需动作"}, ensure_ascii=False)

    # ---- 手写解析（只给 openai/mock 用；pydantic 路径不需要）----
    @staticmethod
    def _parse(raw: str) -> dict | None:
        if not raw:
            return None
        text = raw.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict) or "action" not in obj:
            return None
        return {"action": str(obj["action"]).strip(),
                "reason": str(obj.get("reason", "")).strip()[:200]}

    def stats(self) -> dict:
        return {"backend": self.backend, "model": self.model,
                "calls": self.calls, "failures": self.failures,
                "tokens": self.tokens_used}


def build(backend: str = "none", **kw) -> LLMAgent:
    return LLMAgent(backend=backend, **kw)

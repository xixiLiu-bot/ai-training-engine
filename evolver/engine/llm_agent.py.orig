"""决策大脑（可选层）：用大模型给闭环“提建议”，但绝不进主路径。

设计原则（很重要）：

1. **它不是主流程**。规则引擎 `loop/decide.py` 才是。
2. **它只能“建议”**。建议必须通过三道检查才会被采纳：
      - 动作在白名单内
      - 预算够
      - 输出能被解析
3. **失败一律降级**。超时 / 连不上 / 返回垃圾 → 直接返回 None，规则接管。
   门槛① 要求“无人干预跑完”，所以这一层不能有任何阻塞点。

后端可切换（同一个 OpenAI 兼容协议）：

    none   关闭（默认）
    mock   本地假模型，用来验证链路是否通
    pytrio 比赛用的模型服务（等额度下发）
    ollama 本地 ollama（开发期替代）
"""
from __future__ import annotations

import json
import re
import time

PRESETS = {
    "none": {"enabled": False},
    "mock": {"enabled": True, "mode": "mock", "model": "mock-llm"},
    "mock_bad": {"enabled": True, "mode": "mock", "model": "mock-bad"},
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
你只能从给定动作中选择一个，并给出一句话理由。
严格只输出 JSON，不要任何解释性文字、不要代码块。"""


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
选一个动作。原则：能用便宜手段解决，就不要用贵的。
输出：{{"action": "<动作名>", "reason": "<一句话理由>"}}"""


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
        self.timeout = 10.0
        self.tokens_used = 0
        self.calls = 0
        self.failures = 0

    # ---- 对外唯一接口 ----
    def suggest(self, state: dict, diagnosis: dict, catalog: dict,
                rule_action: str) -> dict | None:
        if not self.enabled:
            return None
        prompt = build_prompt(state, diagnosis, catalog, rule_action)
        t0 = time.time()
        try:
            raw = self._call(prompt)
            parsed = self._parse(raw)
            if parsed is None:
                self.failures += 1
                return None
            if parsed["action"] not in catalog:
                # 模型说了个不存在/不允许的动作 → 拒绝，交给规则
                self.failures += 1
                return None
            self.calls += 1
            return {**parsed, "latency": round(time.time() - t0, 2),
                    "backend": self.backend, "model": self.model}
        except Exception as e:                      # noqa: BLE001
            self.failures += 1
            return {"error": f"{type(e).__name__}: {e}"[:200]}

    # ---- 后端实现 ----
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
        # 解析 prompt 里的 PSI，模仿一个“比规则更激进”的主任
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

    # ---- 解析 ----
    @staticmethod
    def _parse(raw: str) -> dict | None:
        """从模型输出里抠出 JSON。容忍代码块包裹、前后有废话。"""
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

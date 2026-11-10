"""把 4B 的任意输出解析成一个合法动作。

按「越可靠越先用」的顺序试，任何一步成功就返回：

    1. 裸动作名        —— guided_choice 的正常输出，只有 1~2 个 token
    2. 整体 JSON       —— 模型严格照 schema 输出
    3. 围栏内的 JSON   —— ```json ... ```
    4. 文本里的 JSON   —— 括号配平扫描，比正则贪婪匹配可靠
    5. `"action": "x"` —— JSON 被截断但字段还在
    6. 英文动作名      —— 输出里出现过白名单里的词
    7. 中文/近义词     —— 走 ALIASES

解析失败返回 None，调用方负责降级，**永远不抛异常**。
"""
from __future__ import annotations

import json
import re

from .actions import ACTIONS, normalize_action

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
_ACTION_FIELD = re.compile(r'"?action"?\s*[:=]\s*"?([A-Za-z_\u4e00-\u9fff]+)"?', re.I)


def strip_fences(text: str) -> str:
    out = _FENCE.sub("", text or "")
    return out.strip()


def iter_json_objects(text: str):
    """括号配平扫描：逐个吐出可解析的 JSON 对象。

    比 `re.search(r"\\{.*\\}")` 可靠得多——后者在理由里带 `}` 时会切错。
    从后往前找，因为模型通常把最终答案放在最后。
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    found: list[str] = []
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    found.append(text[start:i + 1])
                    start = -1
    for chunk in reversed(found):
        try:
            yield json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue


def _from_obj(obj) -> str | None:
    """结构化字段一律 strict：宁可拒绝，也不替模型猜。"""
    if not isinstance(obj, dict):
        return None
    for key in ("action", "decision", "next_action", "choice"):
        if key in obj:
            got = normalize_action(obj.get(key), strict=True)
            if got:
                return got
    return None


def _reason_from_obj(obj) -> str:
    if isinstance(obj, dict):
        for key in ("reason", "why", "explanation"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:120]
    return ""


def parse_decision(raw: str) -> tuple[str | None, str, str]:
    """返回 (action, reason, how)。action 为 None 表示解析失败。"""
    if raw is None:
        return None, "", "empty"
    text = str(raw).strip()
    if not text:
        return None, "", "empty"

    # 1. 裸动作名（guided_choice 的典型输出）
    bare = normalize_action(text)
    if bare and len(text) <= 40 and " " not in text.strip():
        return bare, "", "bare"

    # 2/3/4. 整体 JSON / 围栏 / 文本中的 JSON
    body = strip_fences(text)
    try:
        obj = json.loads(body)
        got = _from_obj(obj)
        if got:
            return got, _reason_from_obj(obj), "json"
    except (json.JSONDecodeError, ValueError):
        pass

    for obj in iter_json_objects(body):
        got = _from_obj(obj)
        if got:
            return got, _reason_from_obj(obj), "json_scan"

    # 5. 字段还在但 JSON 已经断了（4B 被 max_tokens 截断的典型形态）
    #    注意用 strict：`"refit_mo` 这种截断只能放弃，不能靠子串猜。
    m = _ACTION_FIELD.search(body)
    if m:
        got = normalize_action(m.group(1), strict=True)
        if got:
            return got, "", "field"

    # 6/7. 白名单词 / 中文近义词出现在任意位置（只在短文本上生效）
    #      但如果整段以 `{` 开头，说明是**残缺的 JSON**——
    #      这时候从 `"refit_mo` 里猜出 refit_model 就是替模型编造决策，宁可拒绝。
    if not body.lstrip().startswith("{"):
        got = normalize_action(body)
        if got:
            return got, "", "keyword"

    return None, "", "unparsed"

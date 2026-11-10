"""给 4B 的提示词。

三条硬约束，都是被 4B 的实际表现逼出来的：

1. **短**。4B 的有效注意力有限，超过 400 token 的指令会被忽略一半。
2. **数字给全、别给叙述**。PSI / 分数 / 上次增益六个数字就是全部决策依据，
   写"近期数据分布出现了一定程度的变化"这种话对 4B 只有负作用。
3. **示例用 2 条就够**。3 条以上反而让 4B 开始模仿示例里的具体数值。
"""
from __future__ import annotations

from .actions import catalog_text

SYSTEM_PROMPT = (
    "你是工业模型运维助手。数据在漂移，模型在失准。\n"
    "你只从三个动作里选一个，只回答动作名，不要任何解释。\n"
    "原则：能用便宜手段解决，就不要用贵的。"
)

# 两条示例：一条"别动"，一条"便宜的够用"。刻意不给"重训"的例子——
# 否则 4B 会过度模仿，倾向选最贵的动作。
FEW_SHOT: list[dict] = [
    {"role": "user", "content": "PSI=0.030 分数=0.951 最好=0.955 上次增益=+0.000"},
    {"role": "assistant", "content": "do_nothing"},
    {"role": "user", "content": "PSI=0.150 分数=0.812 最好=0.951 上次增益=+0.000"},
    {"role": "assistant", "content": "refresh_normalization"},
]

REPAIR_PROMPT = (
    "你上一次的输出无法解析。只输出一个动作名，不要标点、不要解释。"
    "可选：do_nothing 或 refresh_normalization 或 refit_model"
)


def build_messages(state: dict, diagnosis: dict, rule_action: str) -> list[dict]:
    """把局势压成 4 行数字 + 1 行归因。"""
    line = (
        f"PSI={float(state.get('psi_mean', 0.0)):.3f} "
        f"分数={float(state.get('current_score', 0.0)):.3f} "
        f"最好={float(state.get('best_score', 0.0)):.3f} "
        f"上次增益={float(state.get('last_gain', 0.0)):+.3f} "
        f"距上次更新={int(state.get('steps_since_update', 0))}批 "
        f"上次动作={state.get('last_action') or '无'}"
    )
    body = (
        f"{line}\n"
        f"归因={diagnosis.get('kind', 'unknown')}\n"
        f"规则初判={rule_action}\n"
        f"可选={catalog_text()}\n"
        "只回答一个动作名。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT,
        {"role": "user", "content": body},
    ]


def repair_messages(messages: list[dict], bad_output: str) -> list[dict]:
    """把坏输出原样喂回去，再要一次——比换一套提示词有效。"""
    return [
        *messages,
        {"role": "assistant", "content": (bad_output or "")[:200]},
        {"role": "user", "content": REPAIR_PROMPT},
    ]

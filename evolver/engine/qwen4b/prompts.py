"""给 4B 的提示词。

三条硬约束，都是被 4B 的实际表现逼出来的：

1. **短**。4B 的有效注意力有限，超过 400 token 的指令会被忽略一半。
2. **数字给全、别给叙述**。PSI / 分数 / 上次增益六个数字就是全部决策依据，
   写"近期数据分布出现了一定程度的变化"这种话对 4B 只有负作用。
3. **示例用 2 条就够**。3 条以上反而让 4B 开始模仿示例里的具体数值。
"""
from __future__ import annotations

import os

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

# ─────────────────────────────────────────────────────────────
# rich 变体：给 27B 这类"能吃更多信息"的模型用
#
# 与默认版的区别（默认版是为 4B 压缩过的）：
#   1. 多给 psi_ratio（多少比例的特征超标）和 psi_topk（最严重的几个）
#      —— 这两个能把"整体老化"和"个别传感器坏了"分开，而 psi_mean 分不出
#   2. 给真实成本（每个模型不同），而不是写死的 2.0s
#   3. 给一条历史经验（这类漂移下哪种动作的单位成本增益更高）
#   4. 多一条"该重训"的示例 —— 默认版刻意不给，因为 4B 会过度模仿；
#      27B 不会，而且没有这条它会以为重训永远不该选
# ─────────────────────────────────────────────────────────────
RICH_SYSTEM = (
    '你是工业模型运维助手。数据在漂移，模型在失准。\n'
    '你从三个动作里选一个，只回答动作名。\n'
    '原则：能用便宜手段解决，就不要用贵的。\n'
    '判断依据是 psi_rel（当前 PSI 相对本数据集重型阈值的倍数，1.0 表示刚好到重型阈值）：\n'
    '  psi_rel < 0.5 一般不动；0.5~1.0 先试校准；\n'
    '  psi_rel >= 1 且上次校准已证明无效时，才重训。\n'
    'psi_ratio 高说明是整体老化（局部校准救不了）；\n'
    'psi_ratio 低但 psi_topk 高，说明只是少数传感器异常，先校准更划算。'
)

RICH_FEW_SHOT: list[dict] = [
    # 别动：远低于重型阈值
    {"role": "user", "content":
        "psi_rel=0.02 psi_ratio=0.00 psi_topk=0.05 分数=0.951 最好=0.955 "
        "上次增益=+0.000 上次动作=无 归因=stable 成本=不动0s/校准0.05s/重训0.5s"},
    {"role": "assistant", "content": "do_nothing"},
    # 校准：接近但在重型阈值以下，超标比例不高
    {"role": "user", "content":
        "psi_rel=0.60 psi_ratio=0.12 psi_topk=0.31 分数=0.812 最好=0.951 "
        "上次增益=+0.000 上次动作=无 归因=covariate 成本=不动0s/校准0.05s/重训0.5s"},
    {"role": "assistant", "content": "refresh_normalization"},
    # 重训：超过重型阈值 + 大面积超标 + 上次校准已证明无效
    {"role": "user", "content":
        "psi_rel=1.35 psi_ratio=0.63 psi_topk=2.80 分数=0.580 最好=0.951 "
        "上次增益=-0.004 上次动作=refresh_normalization 归因=covariate "
        "成本=不动0s/校准0.05s/重训0.5s"},
    {"role": "assistant", "content": "refit_model"},
]


def build_messages_rich(state: dict, diagnosis: dict, rule_action: str,
                        catalog: dict | None = None) -> list[dict]:
    """信息更全的版本：多给 psi_ratio / psi_topk / 真实成本 / 历史经验。"""
    ratio = state.get("psi_ratio")
    topk = state.get("psi_topk")
    ratio_s = f"{float(ratio):.2f}" if ratio is not None else "?"
    topk_s = f"{float(topk):.3f}" if topk is not None else "?"

    if catalog:
        # catalog 可能是 {动作: 秒} 或 {动作: {"cost_sec": 秒}}（policy 传的是后者）
        def _sec(v):
            if isinstance(v, dict):
                v = v.get("cost_sec", 0.0)
            return float(v)
        cost_s = "/".join(f"{k}{_sec(v):g}s" for k, v in catalog.items())
    else:
        cost_s = catalog_text()

    psi_mean = float(state.get("psi_mean", 0.0))
    psi_mid = float(state.get("psi_mid", 0.0) or 0.0)
    if psi_mid > 0:
        rel_s = f"{psi_mean / psi_mid:.2f}"
    else:
        rel_s = f"{psi_mean:.3f}"      # 还没标定出来时退回绝对值

    body = (
        f"psi_rel={rel_s} psi_mean={psi_mean:.3f} psi_mid={psi_mid:.3f} "
        f"psi_ratio={ratio_s} psi_topk={topk_s} "
        f"分数={float(state.get('current_score', 0.0)):.3f} "
        f"最好={float(state.get('best_score', 0.0)):.3f} "
        f"上次增益={float(state.get('last_gain', 0.0)):+.3f} "
        f"上次动作={state.get('last_action') or '无'} "
        f"归因={diagnosis.get('kind', 'unknown')}\n"
        f"成本={cost_s}\n"
        f"规则初判={rule_action}\n"
        "只回答一个动作名。"
    )
    return [
        {"role": "system", "content": RICH_SYSTEM},
        *RICH_FEW_SHOT,
        {"role": "user", "content": body},
    ]


def prompt_variant() -> str:
    """当前 prompt 变体：默认 default，设 QWEN_PROMPT_VARIANT=rich 切到增强版。"""
    v = (os.getenv("QWEN_PROMPT_VARIANT") or "default").strip().lower()
    return "rich" if v == "rich" else "default"

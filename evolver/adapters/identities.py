"""数据身份标识（data identity）与使用权限。

为什么需要这层
--------------
同一个代码库会接触四类数据，它们的**允许用途完全不同**。
如果不把身份写进数据源、并在入口做硬校验，就极容易在标定脚本里
误用官方公开集甚至官方测试流——那属于"用测试集调参"，
是评测里最严重的违规之一。

身份清单
--------
    validation          本地验证段（Gas 的 b05–b08）→ **唯一**允许选阈值
    public_holdout      本地留出段（Gas 的 b10）  → 一次性验收，已消耗
    official_public     官方公开集（训练/验证）    → 只做格式体检，禁止调参
    official_test       官方隐藏测试流的结果快照    → 冻结后一次性评估
    public_substitute   公开替代数据（UCI Gas）    → 开发期允许，须声明非官方
    synthetic           合成数据                  → 开发期允许

关键区分
--------
- `official_public` 与 `official_test` **都不是** selection 允许的身份；
- `official_test` 不是"一堆本地文件"，而是**平台回测后回传的结果**，
  本地不可能读取它的原始数据（赛题红线：禁止获取隐藏测试集）。
"""
from __future__ import annotations

VALIDATION = "validation"
PUBLIC_HOLDOUT = "public_holdout"
OFFICIAL_PUBLIC = "official_public"
OFFICIAL_TEST = "official_test"
PUBLIC_SUBSTITUTE = "public_substitute"
SYNTHETIC = "synthetic"
UNKNOWN = "unknown"

ALL_IDENTITIES = (
    VALIDATION, PUBLIC_HOLDOUT, OFFICIAL_PUBLIC, OFFICIAL_TEST,
    PUBLIC_SUBSTITUTE, SYNTHETIC, UNKNOWN,
)

# 允许用于"选阈值 / 调参 / 网格搜索"的身份。
# 注意：这里**没有** official_public，也没有 official_test。
SELECTION_ALLOWED = frozenset({VALIDATION, PUBLIC_SUBSTITUTE, SYNTHETIC})

# 官方身份：任何训练/选择路径都必须拒绝
OFFICIAL_IDENTITIES = frozenset({OFFICIAL_PUBLIC, OFFICIAL_TEST})

# 只允许"体检"的身份（读结构、算统计，不拟合、不选参）
INSPECT_ONLY = frozenset({OFFICIAL_PUBLIC})

IDENTITY_ZH = {
    VALIDATION: "本地验证段",
    PUBLIC_HOLDOUT: "本地留出段",
    OFFICIAL_PUBLIC: "官方公开集",
    OFFICIAL_TEST: "官方隐藏测试流（仅平台）",
    PUBLIC_SUBSTITUTE: "公开替代数据（非官方）",
    SYNTHETIC: "合成数据",
    UNKNOWN: "未标注身份",
}


class DataIdentityError(RuntimeError):
    """数据身份不允许当前用途。必须显式报错，不能降级放行。"""


def normalize(identity: str | None) -> str:
    v = (identity or "").strip().lower()
    if v not in ALL_IDENTITIES:
        raise DataIdentityError(
            f"未知的数据身份 {identity!r}；允许值：{sorted(ALL_IDENTITIES)}")
    return v


def assert_selectable(identity: str, *, where: str = "选阈值/调参") -> str:
    """选阈值、网格搜索、调参前必须调用。

    官方数据一律拒绝；未标注身份的也拒绝（宁可报错，不要默认放行）。
    """
    v = normalize(identity)
    if v in OFFICIAL_IDENTITIES:
        raise DataIdentityError(
            f"拒绝在 {where} 中使用 {v}（{IDENTITY_ZH[v]}）。"
            f"官方数据只能做格式体检或冻结后的一次性评估，绝不可参与调参。")
    if v == PUBLIC_HOLDOUT:
        raise DataIdentityError(
            f"拒绝在 {where} 中使用 {v}（{IDENTITY_ZH[v]}）。"
            f"留出段已消耗，不得再次用于选择。")
    if v not in SELECTION_ALLOWED:
        raise DataIdentityError(
            f"拒绝在 {where} 中使用 {v}（{IDENTITY_ZH.get(v, v)}）。"
            f"允许的身份：{sorted(SELECTION_ALLOWED)}")
    return v


def assert_inspectable(identity: str, *, where: str = "格式体检") -> str:
    """只做结构检查（不拟合、不选参）时使用。"""
    v = normalize(identity)
    if v == OFFICIAL_TEST:
        raise DataIdentityError(
            f"{v} 是平台隐藏测试流的结果快照，本地没有原始数据可检查。"
            f"请改为检查 {OFFICIAL_PUBLIC}。")
    return v


def assert_official_test(identity: str, *, where: str = "正式评估") -> str:
    """一次性正式评估只接受 official_test。"""
    v = normalize(identity)
    if v != OFFICIAL_TEST:
        raise DataIdentityError(
            f"{where} 期望 {OFFICIAL_TEST}，实际是 {v}（{IDENTITY_ZH.get(v, v)}）。"
            f"本地数据不能冒充官方测试流。")
    return v


def describe(identity: str) -> str:
    return IDENTITY_ZH.get(identity, identity)

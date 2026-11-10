"""评测纪律：配置冻结 + 一次性运行凭据。

两道闸门
--------
1. **冻结**（`assert_frozen`）：`official_test` 评估前，阈值/门控参数必须已经
   写进 `thresholds.json` 并带 `frozen_at` 与 `config_sha`。参数一改，指纹就变，
   旧冻结失效，必须重新冻结（人工动作，不能脚本自动续命）。
2. **一次性**（`eval_ticket.json`）：同一份官方测试流只允许正式评估一次。
   凭据被标记 used 后再次申请直接报错，杜绝"跑一次看分数再改参数"。

凭据字段
--------
    dataset / identity / config_sha / frozen_at / created_at
    status: pending | used
    used_at / result_sha   （跑完写入）

设计原则
--------
- 只用标准库；不联网、不调平台 API。
- 报错要说清"下一步该做什么"，而不是只抛异常。
"""
from __future__ import annotations

import hashlib
import json
import os
import time

TICKET_NAME = "eval_ticket.json"


class FreezeError(RuntimeError):
    """配置未冻结或冻结已失效。"""


class TicketError(RuntimeError):
    """一次性运行凭据相关错误。"""


# ---------------------------------------------------------------- 指纹
def config_fingerprint(cfg: dict) -> str:
    """参数字典 → 稳定 sha256（键排序、紧凑序列化）。"""
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_params(psi_low: float, psi_mid: float, *, detector: str,
                      gate_margin: float, margin_mode: str) -> dict:
    """参与冻结的"影响决策"的参数集合。

    只放**会改变动作**的量：阈值、检测器、门控 margin 与模式。
    batch 数、随机种子、输出目录之类不影响策略的，不放进来。
    """
    return {
        "psi_low": float(psi_low),
        "psi_mid": float(psi_mid),
        "detector": str(detector),
        "gate_margin": float(gate_margin),
        "margin_mode": str(margin_mode),
    }


# ---------------------------------------------------------------- 冻结
def freeze_thresholds(path: str, *, psi_low: float, psi_mid: float,
                      detector: str = "psi", gate_margin: float = 0.02,
                      margin_mode: str = "absolute",
                      extra: dict | None = None) -> dict:
    """写入/更新冻结信息。**只应在人工确认参数后调用一次。**"""
    params = _canonical_params(psi_low, psi_mid, detector=detector,
                               gate_margin=gate_margin, margin_mode=margin_mode)
    data = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data.update({
        "active_thresholds": {"psi_low": params["psi_low"],
                              "psi_mid": params["psi_mid"]},
        "frozen_params": params,
        "config_sha": config_fingerprint(params),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    if extra:
        data.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def assert_frozen(path: str, *, psi_low: float, psi_mid: float,
                  detector: str = "psi", gate_margin: float = 0.02,
                  margin_mode: str = "absolute") -> dict:
    """校验当前要跑的参数与冻结记录完全一致。"""
    if not os.path.isfile(path):
        raise FreezeError(
            f"找不到冻结配置：{path}\n"
            f"  官方测试集评估前必须先冻结参数。下一步：\n"
            f"    python -m evolver.evaluation freeze --path {path} "
            f"--psi-low {psi_low} --psi-mid {psi_mid}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise FreezeError(f"冻结配置不是合法 JSON：{exc}") from exc

    if not data.get("frozen_at") or not data.get("config_sha"):
        raise FreezeError(
            f"{path} 里没有 frozen_at / config_sha，说明尚未冻结。\n"
            f"  先用 calibrate.py 给出结论并人工确认，再执行冻结。")

    want = _canonical_params(psi_low, psi_mid, detector=detector,
                             gate_margin=gate_margin, margin_mode=margin_mode)
    got = data.get("frozen_params") or {}
    if config_fingerprint(want) != config_fingerprint(got):
        diff = {k: (got.get(k), want.get(k)) for k in want
                if got.get(k) != want.get(k)}
        raise FreezeError(
            f"当前参数与冻结配置不一致（冻结在 {data.get('frozen_at')}）。\n"
            f"  差异（冻结值 → 当前值）：{diff}\n"
            f"  官方测试集只允许用**冻结后**的参数跑一次；改参数请先重新冻结。")
    return data


# ---------------------------------------------------------------- 凭据
def _load_ticket(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_ticket(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def acquire_ticket(path: str, *, dataset: str, identity: str,
                   config_sha: str, frozen_at: str | None = None) -> dict:
    """申请一次性运行凭据。已用过则拒绝。"""
    from .adapters import identities as ids

    ids.assert_official_test(identity, where="一次性评估凭据")
    data = _load_ticket(path)
    if data.get("status") == "used":
        raise TicketError(
            f"这份官方测试流的评估凭据已被使用：\n"
            f"  数据集 {data.get('dataset')}　使用时间 {data.get('used_at')}\n"
            f"  凭据文件 {path}\n"
            f"  规则：官方测试流只允许正式评估一次。如需重跑，"
            f"请走赛题允许的**正式重跑申请**，不要删凭据文件绕过。")
    if data.get("status") == "pending" and data.get("config_sha") != config_sha:
        raise TicketError(
            f"已存在**不同配置**的待用凭据（{data.get('config_sha')[:12]}…）。"
            f"放弃原凭据需人工删除 {path} 并说明原因。")

    ticket = {
        "dataset": dataset,
        "identity": ids.OFFICIAL_TEST,
        "config_sha": config_sha,
        "frozen_at": frozen_at,
        "created_at": data.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pending",
        "used_at": None,
        "result_sha": None,
    }
    _save_ticket(path, ticket)
    return ticket


def mark_used(path: str, *, result_sha: str | None = None) -> dict:
    """评估跑完后标记凭据为已使用。"""
    data = _load_ticket(path)
    if not data:
        raise TicketError(f"凭据不存在：{path}（应先 acquire_ticket）")
    if data.get("status") == "used":
        raise TicketError(f"凭据已被标记使用：{path}")
    data["status"] = "used"
    data["used_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if result_sha:
        data["result_sha"] = result_sha
    _save_ticket(path, data)
    return data


def ticket_status(path: str) -> dict:
    data = _load_ticket(path)
    if not data:
        return {"exists": False, "status": "absent"}
    return {"exists": True, "status": data.get("status"),
            "dataset": data.get("dataset"), "used_at": data.get("used_at"),
            "config_sha": data.get("config_sha")}

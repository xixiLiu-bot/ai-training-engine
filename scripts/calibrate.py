# -*- coding: utf-8 -*-
"""阈值标定：在时间顺序分段上网格搜索 psi_low × psi_mid。

    python scripts/calibrate.py --dataset gas --detector psi
    python scripts/calibrate.py --grid "0.6,0.8,1.0,1.2|2.0,2.5,3.0"

设计约束（重要，别改）
--------------------
1. **b10 是严格 holdout**：
   - 默认**不读**、不跑（`--run-holdout` 才跑，且只跑一次）
   - 网格搜索只允许用 b01..b09
   - holdout 标签永远不参与任何选择
2. **选择与报告分离**：
   - calibration 段（默认 b02–b04）用来观察
   - validation 段（默认 b05–b08）**唯一**用于挑阈值
   - 选定后在 b02–b09 上出一份 pre-holdout 汇总，**仅用于报告，不回头改阈值**
3. **动作顺序合法性**：只接受 0 < psi_low < psi_mid，保证
   do_nothing → calibrate/refresh → refit 的阶梯单调。
4. **不碰核心算法**：本脚本只通过命令行参数（--psi-low/--psi-mid）驱动
   `scripts/run_pipeline.py`，不改任何 loop/decide/policy 代码。

选择目标（可调）
----------------
    utility = validation_gain
              - lambda_cost     * normalized_cost
              - lambda_rollback * rollback_rate

硬约束（任一不满足直接淘汰）：
    - 协议违规数 == 0
    - validation_gain >= 0（不能比"从不更新"的静态基线差）
    - rollback_rate <= --max-rollback-rate
    - 门控未被关闭（margin > 0）

多个候选落在最优 utility 的容差内时，**优先选成本最低的那个**，
而不是动作最多、分数最高的那个。
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import io
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ---------------------------------------------------------------- 分段
DEFAULT_CALIBRATION = (2, 4)
DEFAULT_VALIDATION = (5, 8)
PRE_HOLDOUT = (2, 9)
HOLDOUT = (10, 10)

# 默认网格：气体数据 PSI 量级 0.7~6.5，所以低位从 0.6 起、高位到 4.0
DEFAULT_GRID = "0.6,0.8,1.0,1.2,1.5|2.0,2.5,3.0,4.0"

# 旧阈值（冻结基线，来自旧文档，未标定）
OLD_THRESHOLD = (1.2, 2.5)


class CalibrationError(RuntimeError):
    """标定前置条件不满足。必须显式报错，不能静默降级。"""


# ---------------------------------------------------------------- 参数解析
def parse_grid(spec: str) -> tuple[list[float], list[float]]:
    """'0.6,0.8|2.0,2.5' → ([0.6,0.8], [2.0,2.5])"""
    if not spec or "|" not in spec:
        raise CalibrationError(
            f"--grid 格式应为 '低值列表|高值列表'，例如 '0.6,1.0|2.0,3.0'；收到 {spec!r}")
    low_s, mid_s = spec.split("|", 1)
    try:
        lows = [float(x) for x in low_s.split(",") if x.strip()]
        mids = [float(x) for x in mid_s.split(",") if x.strip()]
    except ValueError as exc:
        raise CalibrationError(f"--grid 里有非数字：{exc}") from exc
    if not lows or not mids:
        raise CalibrationError("--grid 两侧都不能为空")
    return sorted(set(lows)), sorted(set(mids))


def parse_batches(spec: str, name: str) -> tuple[int, int]:
    """'5-8' → (5, 8)"""
    try:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
    except Exception as exc:                        # noqa: BLE001
        raise CalibrationError(f"--{name} 格式应为 '起始-结束'，收到 {spec!r}") from exc
    if lo < 1 or hi < lo:
        raise CalibrationError(f"--{name} 区间非法：{spec}")
    return lo, hi


def validate_point(low: float, mid: float) -> None:
    """动作阶梯合法性：必须 0 < psi_low < psi_mid。"""
    if not (low > 0):
        raise CalibrationError(f"psi_low 必须 > 0，收到 {low}")
    if not (mid > low):
        raise CalibrationError(f"必须 psi_low < psi_mid，收到 low={low} mid={mid}")
    if mid <= 0:
        raise CalibrationError(f"psi_mid 必须 > 0，收到 {mid}")


def assert_no_holdout(batch_ids, holdout: tuple[int, int] = HOLDOUT) -> None:
    """硬闸：任何一次网格/验证运行都不许包含 holdout 批次。"""
    lo, hi = holdout
    bad = [b for b in batch_ids if lo <= b <= hi]
    if bad:
        raise CalibrationError(
            f"内部错误：试图在选阈值阶段访问 holdout 批次 {bad}。这会造成数据泄漏。")


# ---------------------------------------------------------------- 跑一次
def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _numpy_backend(module):
    """没有 sklearn 时，用外部 np_stack 顶替 model/metric（仅开发机验证用）。"""
    path = os.getenv("ZHIXUN_NP_STACK_PATH")
    if not path:
        return False
    sys.path.insert(0, path)
    try:
        np_stack = __import__("np_stack")
    except Exception:                               # noqa: BLE001
        return False
    module.md = np_stack.NpModels()
    module.mt = np_stack.NpMetrics()
    return True


def run_config(*, low: float, mid: float, upto: int, out_dir: str,
               seed: int, detector: str, dataset: str,
               verbose: bool = False, allow_holdout: bool = False) -> dict:
    """跑一次闭环，只到第 upto 批（含）。返回 metrics.json 内容 + 逐批决策。

    批次是**连续的 1..upto**，所以行里的 batch 号就是原始批次号，不用做映射。
    """
    batch_ids = list(range(1, upto + 1))
    if not allow_holdout:
        assert_no_holdout(batch_ids)                # 选阈值阶段的双重保险

    os.makedirs(out_dir, exist_ok=True)
    rp_path = os.path.join(REPO, "scripts", "run_pipeline.py")
    rp = _load_module(rp_path, f"rp_calib_{abs(hash((low, mid, upto, out_dir)))}")

    # 用桥接层限定批次范围（数据源本身支持子集）
    from evolver.adapters import identities
    from evolver.adapters.bridge import BridgeAdapter
    from evolver.adapters.sources import GasSource

    # 纪律闸门：官方数据/留出段一律不得进入标定流程
    _src = GasSource()
    identities.assert_selectable(_src.data_identity, where="阈值标定")

    class _Provider:
        @staticmethod
        def build(kind: str = "", **kwargs):
            return BridgeAdapter(GasSource(), batch_ids=batch_ids)

    rp.ds = _Provider()
    rp.plot = lambda *a, **k: None
    _numpy_backend(rp)                              # 有 sklearn 时什么也不做

    # 固定随机种子
    from evolver.config import Config as RealConfig
    rp.Config = lambda: RealConfig(random_state=seed)

    argv = ["run_pipeline.py", "--dataset", dataset,
            "--psi-low", str(low), "--psi-mid", str(mid),
            "--detector", detector, "--llm", "none", "--out", out_dir,
            "--no-recorder"]                        # 网格阶段不写实验记录、不上传
    old_argv = sys.argv
    sys.argv = argv
    buf = io.StringIO()
    try:
        if verbose:
            rp.main()
        else:
            with contextlib.redirect_stdout(buf):
                rp.main()
    except Exception as exc:                        # noqa: BLE001
        tail = "\n".join(buf.getvalue().splitlines()[-15:])
        raise CalibrationError(
            f"psi_low={low}, psi_mid={mid} 运行失败：{type(exc).__name__}: {exc}\n"
            f"---- 管道输出尾部 ----\n{tail}") from exc
    finally:
        sys.argv = old_argv

    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["decisions"] = _read_decisions(os.path.join(out_dir, "decisions.jsonl"))
    return metrics


def _read_decisions(path: str) -> list[dict]:
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------- 汇总
def segment(rows: list[dict], lo: int, hi: int) -> list[dict]:
    return [r for r in rows if lo <= r.get("batch", -1) <= hi]


def summarize(rows: list[dict], decisions: list[dict], protocol: dict) -> dict:
    """按**批次区间**统计指标与成本。

    成本必须来自 decisions.jsonl 的逐批 charge，不能用账本累计值——
    否则 calibration 段的成本会被算进 validation，导致"验证更贵"的假象。
    """
    valid = [r for r in rows if r.get("f1_system") is not None]
    empty = {"n": 0, "system": float("nan"), "baseline": float("nan"),
             "gain": float("nan"), "actions": 0, "rollbacks": 0,
             "rollback_rate": 0.0, "violations": protocol.get("violations", 0),
             "gpu_sec": 0.0, "labels": 0, "tokens": 0,
             "wasted_gpu_sec": 0.0, "wasted_ratio": 0.0}
    if not valid:
        return empty

    base_rows = [r for r in valid if r.get("f1_baseline") is not None]
    system = sum(r["f1_system"] for r in valid) / len(valid)
    base = (sum(r["f1_baseline"] for r in base_rows) / len(base_rows)) if base_rows else float("nan")

    by_batch = {d.get("batch"): d for d in decisions if d.get("batch") is not None}
    gpu = labels = tokens = wasted = 0.0
    rollbacks = 0
    for r in valid:
        d = by_batch.get(r.get("batch"))
        if not d:
            continue
        c = d.get("charge") or {}
        g = float(c.get("gpu_sec", 0) or 0)
        gpu += g
        labels += float(c.get("labels", 0) or 0)
        tokens += float(c.get("tokens", 0) or 0)
        if d.get("decision") == "rollback":
            rollbacks += 1
            wasted += g

    actions = sum(1 for r in valid if r.get("action") not in (None, "do_nothing"))
    return {
        "n": len(valid),
        "system": system,
        "baseline": base,
        "gain": system - base,
        "actions": actions,
        "rollbacks": rollbacks,
        "rollback_rate": rollbacks / len(valid),
        "violations": protocol.get("violations", 0),
        "gpu_sec": gpu,
        "labels": int(labels),
        "tokens": int(tokens),
        "wasted_gpu_sec": wasted,
        "wasted_ratio": (wasted / gpu) if gpu > 1e-12 else 0.0,
    }


def normalized_cost(gpu_sec: float, ref_gpu_sec: float, floor: float = 1e-6) -> float:
    return gpu_sec / max(ref_gpu_sec, floor)


def utility(seg: dict, *, lam_cost: float, lam_rollback: float,
            ref_gpu_sec: float) -> float:
    return (seg["gain"]
            - lam_cost * normalized_cost(seg["gpu_sec"], ref_gpu_sec)
            - lam_rollback * seg["rollback_rate"])


def is_feasible(seg: dict, *, max_rollback_rate: float, need_gain: float,
                max_wasted_ratio: float = 1.0) -> tuple[bool, str]:
    if seg["violations"] != 0:
        return False, f"协议违规 {seg['violations']}"
    if seg["gain"] < need_gain:
        return False, f"增益 {seg['gain']:+.4f} < 下限 {need_gain:+.4f}"
    if seg["rollback_rate"] > max_rollback_rate:
        return False, f"回滚率 {seg['rollback_rate']:.0%} > 上限 {max_rollback_rate:.0%}"
    if seg["wasted_ratio"] > max_wasted_ratio:
        return False, (f"回滚浪费占成本 {seg['wasted_ratio']:.0%} > "
                       f"上限 {max_wasted_ratio:.0%}")
    return True, "ok"


def select_best(cands: list[dict], *, tol: float = 0.01) -> dict:
    """在可行候选里选 utility 最高者；落在容差内的，改选成本最低的。"""
    feasible = [c for c in cands if c["feasible"]]
    if not feasible:
        raise CalibrationError("没有任何候选满足硬约束，无法给出推荐阈值")
    best_u = max(c["utility"] for c in feasible)
    near = [c for c in feasible if c["utility"] >= best_u - tol]
    # 优先低成本，其次回滚少，最后才看 utility
    near.sort(key=lambda c: (c["seg"]["gpu_sec"], c["seg"]["rollback_rate"], -c["utility"]))
    chosen = near[0]
    chosen["selection_note"] = (f"utility 最高 {best_u:.4f} 的容差 {tol} 内共有 "
                                f"{len(near)} 个候选，按'最低成本优先'选中 "
                                f"psi_low={chosen['psi_low']}, psi_mid={chosen['psi_mid']}")
    return chosen


# ---------------------------------------------------------------- 稳健性
def actions_by_batch(decisions: list[dict]) -> dict:
    """{批次号: 动作名}，只保留发生过决策的批次。

    注意字段位置：run_pipeline 写进 decisions.jsonl 的动作在
    `experiment.action`（不是顶层 `action`），这里两种都兼容。
    """
    out = {}
    for d in decisions:
        b = d.get("batch")
        if b is None:
            continue
        exp = d.get("experiment") or {}
        out[b] = exp.get("action", d.get("action"))
    return out


def evidence_batches(cand_dec: list[dict], ref_dec: list[dict]) -> list[int]:
    """候选与参照**动作不同**的批次列表。

    这是"阈值优势是否只来自一个批次"的判据：
    若只差一个批次，就说不清是阈值更优，还是那一批恰好运气好。
    """
    a, b = actions_by_batch(cand_dec), actions_by_batch(ref_dec)
    common = set(a) & set(b)
    return sorted(x for x in common if a[x] != b[x])


def plateau_members(cands: list[dict], best: dict) -> list[dict]:
    """与 best 产生**完全相同运行结果**的候选（平台区间）。

    平台 = 系统均值 / 成本 / 回滚数三者都相同。同平台分不出高下，
    说明阈值在该区间内对结果没有影响。
    """
    key = (round(best["seg"]["system"], 4), round(best["seg"]["gpu_sec"], 2),
           best["seg"]["rollbacks"])
    return [c for c in cands if c["feasible"] and
            (round(c["seg"]["system"], 4), round(c["seg"]["gpu_sec"], 2),
             c["seg"]["rollbacks"]) == key]


def axis_affected_batches(rows: list[dict], v1: float, v2: float) -> list[int]:
    """把阈值从 v1 改到 v2，**可能**受影响的批次（PSI 落在两值之间的）。

    这是"单批次证据"的可证明判据：若只有 1 个批次的 PSI 落在区间内，
    那么该参数的全部作用就只能通过这 1 个批次发生，与其他批次无关。
    """
    lo, hi = min(v1, v2), max(v1, v2)
    out = []
    for r in rows:
        psi = r.get("psi_mean")
        if psi is None:
            continue
        if lo <= psi < hi:
            out.append(r.get("batch"))
    return sorted(b for b in out if b is not None)


def single_batch_flip(cands: list[dict], chosen: dict,
                      eps: float = 0.005) -> dict | None:
    """判断"优势是否只来自单个批次"。

    做法：找只改动**一个轴**且数值相邻的网格点，计算该改动会波及哪些批次
    （PSI 落在两阈值之间）。若只波及 1 个批次，而 validation 结果明显不同，
    则结论由那一个批次单独决定。

    这比"动作差异批次数"可靠：后者会把相位级联
    （b02 一动，后面每批动作都跟着 shift）误当成分散证据。
    """
    base = chosen.get("_decisions") or []
    rows = chosen.get("_rows") or []
    best = None
    for c in cands:
        if c is chosen or not c.get("feasible"):
            continue
        same_mid = c["psi_mid"] == chosen["psi_mid"]
        same_low = c["psi_low"] == chosen["psi_low"]
        if same_mid == same_low:            # 必须恰好一个轴变化
            continue
        if same_mid:                        # 只改 psi_low
            affected = axis_affected_batches(rows, chosen["psi_low"], c["psi_low"])
            axis = "psi_low"
        else:                               # 只改 psi_mid
            affected = axis_affected_batches(rows, chosen["psi_mid"], c["psi_mid"])
            axis = "psi_mid"
        if len(affected) != 1:
            continue
        delta = abs(c["seg"]["system"] - chosen["seg"]["system"])
        if delta <= eps:
            continue
        if best is None or delta > best["delta"]:
            best = {"batch": affected[0], "axis": axis,
                    "other_low": c["psi_low"], "other_mid": c["psi_mid"],
                    "other_system": c["seg"]["system"],
                    "system": chosen["seg"]["system"],
                    "delta": round(delta, 4)}
    return best


def robust_pick(cands: list[dict], *, active: tuple[float, float],
                tol: float = 0.01) -> dict:
    """稳健推荐：utility 最优可能很脆，默认值必须站得住。

    规则（按顺序）：
      1. 先按 utility 选出实验最优候选；
      2. 若它与现役配置**行为完全相同** → 没有替换的理由，回退；
      3. 若存在"只差一个批次决策就翻转结果"的相邻点 → single_batch_evidence，
         **不采纳**，回退到现役配置；
      4. 否则在"平台区间"内选最接近现役配置的那个（并列看动作数与成本）。
    """
    best = select_best(cands, tol=tol)
    act = next((c for c in cands if c.get("_is_active")), None)
    div = evidence_batches(best.get("_decisions") or [],
                           act.get("_decisions") if act else [])

    if act is not None and not div:
        return {
            "psi_low": active[0], "psi_mid": active[1],
            "source": "active（回退）", "reason": "no_behavioral_difference",
            "evidence_batches": [],
            "candidate": {"psi_low": best["psi_low"], "psi_mid": best["psi_mid"],
                          "utility": best["utility"], "seg": best["seg"]},
            "note": (f"候选 {best['psi_low']}/{best['psi_mid']} 与现役配置在每个批次上"
                     f"动作完全相同，没有更换的理由，保持 {active[0]}/{active[1]}"),
        }

    flip = single_batch_flip(cands, best)
    if flip is not None:
        return {
            "psi_low": active[0], "psi_mid": active[1],
            "source": "active（回退）", "reason": "single_batch_evidence",
            "evidence_batches": [flip["batch"]],
            "flip": flip,
            "candidate": {"psi_low": best["psi_low"], "psi_mid": best["psi_mid"],
                          "utility": best["utility"], "seg": best["seg"]},
            "note": (f"实验最优点 {best['psi_low']}/{best['psi_mid']} 与相邻网格点 "
                     f"{flip['other_low']}/{flip['other_mid']} 的动作**只差 b{flip['batch']} "
                     f"一个批次**，但 validation 系统均值从 {flip['system']:.4f} 变成 "
                     f"{flip['other_system']:.4f}（差 {flip['delta']:.4f}）。"
                     f"说明结论由 b{flip['batch']} 单独决定，单批次证据不足以替换默认值，"
                     f"回退到现役 {active[0]}/{active[1]}"),
        }

    pool = plateau_members(cands, best) or [best]
    al, am = active
    pool.sort(key=lambda c: (abs(c["psi_low"] - al) + abs(c["psi_mid"] - am),
                             c["seg"]["actions"], c["seg"]["gpu_sec"]))
    pick = pool[0]
    return {
        "psi_low": pick["psi_low"], "psi_mid": pick["psi_mid"],
        "source": "plateau（平台内最保守）",
        "reason": "multi_batch_evidence",
        "evidence_batches": div,
        "flip": None,
        "candidate": {"psi_low": best["psi_low"], "psi_mid": best["psi_mid"],
                      "utility": best["utility"], "seg": best["seg"]},
        "note": (f"最优候选与现役配置相差 {len(div)} 个批次动作，且找不到"
                 f"\"单批次翻转\"的证据；在平台 {len(pool)} 个点中取最接近现役 "
                 f"{active[0]}/{active[1]} 的 psi_low={pick['psi_low']}/"
                 f"psi_mid={pick['psi_mid']}"),
    }


def sensitivity(cands: list[dict], *, lo: int, hi: int,
                drop: tuple[int, ...] = (2, 5, 6, 7, 8)) -> dict:
    """留一敏感性：逐个移除某批次后重算 validation 统计。

    只做**统计层面**重算（用同一次运行里的逐批结果），
    不重跑、不读 holdout、不得用于宣称最终成绩。
    """
    out = {"dropped": list(drop), "validation_range": [lo, hi], "rows": []}
    refs = [c for c in cands if c.get("_is_active") or c.get("_is_candidate")]
    for b in drop:
        entry = {"dropped_batch": b,
                 "in_validation": lo <= b <= hi, "candidates": {}}
        for c in refs:
            name = c.get("_label") or f"{c['psi_low']}/{c['psi_mid']}"
            rows = [r for r in (c.get("_rows") or [])
                    if lo <= r.get("batch", -1) <= hi and r.get("batch") != b]
            dec = [d for d in (c.get("_decisions") or []) if d.get("batch") != b]
            entry["candidates"][name] = summarize(rows, dec, {"violations": 0})
        out["rows"].append(entry)
    return out


# ---------------------------------------------------------------- 报告
def render_markdown(report: dict) -> str:
    L = []
    L.append("# 阈值标定报告（Gas 公开数据 · 时间顺序）\n")
    L.append(f"- 数据集：`{report['dataset']}`")
    L.append(f"- 检测器：`{report['detector']}`　随机种子：`{report['seed']}`")
    L.append(f"- 标定段 calibration：b{report['split']['calibration'][0]}–"
             f"b{report['split']['calibration'][1]}")
    L.append(f"- 验证段 validation：b{report['split']['validation'][0]}–"
             f"b{report['split']['validation'][1]}　**唯一用于选阈值**")
    _hc = report.get("holdout_carried")
    if report["holdout_run"]:
        _hol = "**本次已运行一次**"
    elif _hc:
        _c = _hc.get("config") or {}
        _hol = (f"**本次未重跑**；下方结果结转自上次"
                f"（配置 {_c.get('psi_low')}/{_c.get('psi_mid')}）")
    else:
        _hol = "未运行"
    L.append(f"- 留出 holdout：b{report['split']['holdout'][0]}–"
             f"b{report['split']['holdout'][1]}　{_hol}")
    L.append("")
    L.append("> ⚠️ 本报告是 Gas 公开数据上的**时间顺序标定结果**，"
             "不代表跨数据集泛化能力。样本只有 10 个批次。\n")

    L.append("## 1. 阈值：默认值 vs 实验候选\n")
    L.append("| 角色 | psi_low | psi_mid | 说明 |")
    L.append("|---|---|---|---|")
    L.append(f"| **默认（现役）** | **{report['active_thresholds']['psi_low']}** | "
             f"**{report['active_thresholds']['psi_mid']}** | "
             f"实际采用；未因本轮标定而改变 |")
    L.append(f"| 实验候选 | {report['calibrated_candidate']['psi_low']} | "
             f"{report['calibrated_candidate']['psi_mid']} | "
             f"utility 最优，**不设为默认** |")
    L.append("")
    rb = report["robust_choice"]
    L.append(f"**稳健推荐依据**：{rb['note']}")
    L.append("")
    if report.get("fallback_reason"):
        _ev = rb.get("evidence_batches") or []
        _evt = "、".join(f"b{b}" for b in _ev) if _ev else "（无可证明批次）"
        L.append(f"> ⚠️ 回退原因：`{report['fallback_reason']}` —— "
                 f"候选优势只来自 {_evt}，换个批次就不成立。")
    if not report["changed_default"]:
        L.append(">")
        L.append("> **默认配置未改变**：仍是 "
                 f"{report['active_thresholds']['psi_low']}/"
                 f"{report['active_thresholds']['psi_mid']}。"
                 "0.6/4.0 仅作为 Gas 公开数据上的实验候选保留。")
    L.append("")

    L.append("## 2. 分段结果（分开列，不混成一个平均分）\n")
    L.append("| 段 | 批次数 | 系统均值 | 静态基线均值 | 增益 | 动作 | 回滚 | 回滚率 | GPU秒 | 标注 | 违规 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, label in (("calibration", "calibration"), ("validation", "validation"),
                       ("pre_holdout", "pre-holdout(b02–b09，仅报告)"),
                       ("holdout", "holdout（一次性）")):
        seg = report["segments"].get(key)
        if not seg or seg.get("n", 0) == 0:
            L.append(f"| {label} | – | – | – | – | – | – | – | – | – | – |")
            continue
        L.append(f"| {label} | {seg['n']} | {seg['system']:.4f} | {seg['baseline']:.4f} | "
                 f"{seg['gain']:+.4f} | {seg['actions']} | {seg['rollbacks']} | "
                 f"{seg['rollback_rate']:.0%} | {seg['gpu_sec']:.2f} | "
                 f"{seg['labels']} | {seg['violations']} |")
    L.append("")

    if report.get("holdout_carried"):
        _c = report["holdout_carried"].get("config") or {}
        L.append(f"> 注意：上表 holdout 行对应配置 **{_c.get('psi_low')}/"
                 f"{_c.get('psi_mid')}**，不是本轮稳健推荐 "
                 f"{report['active_thresholds']['psi_low']}/"
                 f"{report['active_thresholds']['psi_mid']}。为遵守"
                 f"\"b10 只跑一次\"，本轮未重跑，因此没有稳健推荐的 holdout 数字。\n")

    L.append("## 3. 与旧阈值对比（同一 validation 段）\n")
    L.append("| 配置 | psi_low/psi_mid | validation 系统 | 增益 | 动作 | 回滚 | GPU秒 | 单位成本增益 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, ref in report["references"].items():
        s = ref.get("validation") or {}
        if not s or s.get("n", 0) == 0:
            continue
        unit = (f"{s['gain'] / s['gpu_sec']:+.4f}"
                if s["gpu_sec"] > 1e-9 else "—（无成本）")
        L.append(f"| {name} | {ref['psi_low']}/{ref['psi_mid']} | {s['system']:.4f} | "
                 f"{s['gain']:+.4f} | {s['actions']} | {s['rollbacks']} | "
                 f"{s['gpu_sec']:.2f} | {unit} |")
    L.append("")

    L.append("## 4. 网格明细（仅 calibration/validation 结果，**不是最终测试结果**）\n")
    L.append("| psi_low | psi_mid | val 系统 | val 增益 | 动作 | 回滚 | GPU秒 | utility | 可行 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in report["candidates"]:
        s = c["seg"]
        L.append(f"| {c['psi_low']} | {c['psi_mid']} | {s['system']:.4f} | {s['gain']:+.4f} | "
                 f"{s['actions']} | {s['rollbacks']} | {s['gpu_sec']:.2f} | "
                 f"{c['utility']:+.4f} | {'✅' if c['feasible'] else '✗ ' + c['reason']} |")
    L.append("")

    sat = report.get("grid_saturation") or {}
    if sat:
        L.append("## 4.1 网格饱和\n")
        L.append(f"{sat['n_points']} 个网格点只产生了 "
                 f"**{sat['distinct_outcomes']} 种不同的运行结果**。")
        if sat["distinct_outcomes"] <= 3:
            L.append("")
            L.append("> ⚠️ 说明当前阈值空间对这份数据**区分度很低**："
                     "多数网格点走出同一条轨迹。此时\"最优阈值\"的置信度低，"
                     "不能当作稳健结论。")
        L.append("")
    L.append("## 5. 不确定性、敏感性与限制\n")
    sens = report.get("sensitivity") or {}
    if sens.get("rows"):
        L.append("### 5.1 留一敏感性（统计层面重算，**不是**新的独立实验）\n")
        L.append("| 移除批次 | 配置 | 系统 | 增益 | 动作 | 回滚 | GPU秒 | 单位成本增益 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for e in sens["rows"]:
            for name, sg in e["candidates"].items():
                unit = (f"{sg['gain'] / sg['gpu_sec']:+.4f}"
                        if sg["gpu_sec"] > 1e-9 else "—")
                L.append(f"| b{e['dropped_batch']} | {name} | {sg['system']:.4f} | "
                         f"{sg['gain']:+.4f} | {sg['actions']} | {sg['rollbacks']} | "
                         f"{sg['gpu_sec']:.2f} | {unit} |")
        L.append("")
        L.append("> b02 不在 validation 段（b05–b08），移除它对统计无影响；"
                 "但它正是候选阈值行为差异的来源，见第 1 节的回退原因。")
        L.append("")
    L.append("### 5.2 限制\n")
    for item in report["limitations"]:
        L.append(f"- {item}")
    L.append("")
    L.append("### 5.3 官方测试集接入时的冻结规则\n")
    L.append("1. 先用 b02–b09 冻结全部配置（阈值、门控 margin、动作策略），写入 `thresholds.json`；")
    L.append("2. 冻结后**只跑一次**官方测试集，结果单独记录，不回写任何参数；")
    L.append("3. 若官方测试集明显低于 validation，**不调参**，把差异作为分布偏移如实报告；")
    L.append("4. 任何阈值变更都要重新走 calibration→validation，并使用新的留出集"
             "（本轮 b10 已消耗，不可复用）。")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gas")
    ap.add_argument("--detector", default="psi",
                    choices=["psi", "ks", "wasserstein", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=os.path.join(REPO, "out", "calibration"))
    ap.add_argument("--grid", default=DEFAULT_GRID)
    ap.add_argument("--calibration-batches", default="2-4")
    ap.add_argument("--validation-batches", default="5-8")
    ap.add_argument("--holdout-batches", default="10-10")
    ap.add_argument("--max-rollback-rate", type=float, default=0.75,
                    help="回滚率硬上限。默认 0.75；设为 0.5 时本数据上会无可行点")
    ap.add_argument("--max-wasted-ratio", type=float, default=0.85,
                    help="回滚浪费占总成本的比例上限")
    ap.add_argument("--lambda-cost", type=float, default=0.5)
    ap.add_argument("--lambda-rollback", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="utility 容差：容差内优先选最低成本配置")
    ap.add_argument("--active-low", type=float, default=OLD_THRESHOLD[0],
                    help="现役 psi_low（默认值，稳健推荐会回退到它）")
    ap.add_argument("--active-mid", type=float, default=OLD_THRESHOLD[1],
                    help="现役 psi_mid")
    ap.add_argument("--carry-holdout-from", default=None,
                    help="从旧报告结转 holdout 结果（不重跑 b10）")
    ap.add_argument("--run-holdout", action="store_true",
                    help="在阈值冻结后**运行一次** holdout（b10）。默认不跑")
    ap.add_argument("--keep-runs", action="store_true",
                    help="保留每次网格运行的中间产物（默认只留报告）")
    args = ap.parse_args()

    cal = parse_batches(args.calibration_batches, "calibration-batches")
    val = parse_batches(args.validation_batches, "validation-batches")
    hold = parse_batches(args.holdout_batches, "holdout-batches")
    lows, mids = parse_grid(args.grid)

    # ---- 前置校验：分段必须都在 holdout 之前，且互不重叠 ----
    if cal[1] >= val[0]:
        raise CalibrationError(f"calibration {cal} 与 validation {val} 必须按时间前后分开")
    if val[1] >= hold[0]:
        raise CalibrationError(
            f"validation 结束于 b{val[1]}，必须早于 holdout 起始 b{hold[0]}")
    for name, seg in (("calibration", cal), ("validation", val)):
        if seg[1] - seg[0] + 1 < 2:
            raise CalibrationError(f"{name} 段只有 1 个批次，样本不足，结果不稳定")

    points = [(lo, mi) for lo in lows for mi in mids if mi > lo]
    if not points:
        raise CalibrationError("网格里没有合法点（要求 psi_low < psi_mid）")
    for lo, mi in points:
        validate_point(lo, mi)

    upto = max(val[1], cal[1])          # 网格阶段最多跑到 validation 结束
    assert_no_holdout(list(range(1, upto + 1)), hold)

    print(f"数据集={args.dataset} 检测器={args.detector} 种子={args.seed}")
    print(f"calibration=b{cal[0]}–b{cal[1]}  validation=b{val[0]}–b{val[1]}  "
          f"holdout=b{hold[0]}–b{hold[1]}")
    print(f"网格：psi_low × psi_mid = {len(lows)} × {len(mids)} → "
          f"{len(points)} 个合法点（要求 low<mid）")
    print(f"选阈值阶段最多跑到 b{upto}，**不会读取 b{hold[0]}**\n")

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    # ---- 参照：从不更新（等价静态基线）----
    print("[参照] 从不更新（psi_low 极大）")
    never = run_config(low=1e9, mid=2e9, upto=upto, seed=args.seed,
                       detector=args.detector, dataset=args.dataset,
                       out_dir=os.path.join(args.output, "_ref_never"))
    never_val = summarize(segment(never["rows"], *val), never["decisions"], never["protocol"])
    need_gain = 0.0                     # 硬约束：不能比"从不更新"差

    # ---- 参照：旧阈值 ----
    print("[参照] 旧阈值 1.2/2.5（冻结基线）")
    old = run_config(low=OLD_THRESHOLD[0], mid=OLD_THRESHOLD[1], upto=upto,
                     seed=args.seed, detector=args.detector, dataset=args.dataset,
                     out_dir=os.path.join(args.output, "_ref_old"))
    old_val = summarize(segment(old["rows"], *val), old["decisions"], old["protocol"])

    ref_cost = max(old_val["gpu_sec"], never_val["gpu_sec"], 1e-6)

    # ---- 网格搜索 ----
    candidates = []
    for i, (lo, mi) in enumerate(points, 1):
        out_dir = os.path.join(args.output, f"_grid_{lo}_{mi}")
        print(f"[{i}/{len(points)}] psi_low={lo} psi_mid={mi} ...", end=" ", flush=True)
        m = run_config(low=lo, mid=mi, upto=upto, seed=args.seed,
                       detector=args.detector, dataset=args.dataset, out_dir=out_dir)
        val_seg = summarize(segment(m["rows"], *val), m["decisions"], m["protocol"])
        cal_seg = summarize(segment(m["rows"], *cal), m["decisions"], m["protocol"])
        u = utility(val_seg, lam_cost=args.lambda_cost,
                    lam_rollback=args.lambda_rollback, ref_gpu_sec=ref_cost)
        ok, why = is_feasible(val_seg, max_rollback_rate=args.max_rollback_rate,
                              need_gain=need_gain,
                              max_wasted_ratio=args.max_wasted_ratio)
        candidates.append({
            "psi_low": lo, "psi_mid": mi,
            "seg": val_seg, "calibration_seg": cal_seg,
            "utility": u, "feasible": ok, "reason": why,
            # 留一敏感性/证据分析要用，写报告前会剔除
            "_rows": m["rows"], "_decisions": m["decisions"],
        })
        print(f"val={val_seg['system']:.4f} 增益={val_seg['gain']:+.4f} "
              f"动作={val_seg['actions']} 回滚={val_seg['rollbacks']} "
              f"gpu={val_seg['gpu_sec']:.2f}s utility={u:+.4f} "
              f"{'可行' if ok else '淘汰:' + why}")

        if not args.keep_runs:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

    chosen = select_best(candidates, tol=args.tol)
    lo, mi = chosen["psi_low"], chosen["psi_mid"]
    chosen["_is_candidate"] = True
    chosen["_label"] = "候选"

    # 现役配置（默认值）也放进候选池，供"证据批次"对比用
    active = (args.active_low, args.active_mid)
    active_cand = next((c for c in candidates
                        if (c["psi_low"], c["psi_mid"]) == active), None)
    if active_cand is not None:
        active_cand["_is_active"] = True
        active_cand["_label"] = "现役"
    else:
        # 网格里没有现役配置（例如自定义了网格），单独跑一次当参照
        m_act = run_config(low=active[0], mid=active[1], upto=upto, seed=args.seed,
                           detector=args.detector, dataset=args.dataset,
                           out_dir=os.path.join(args.output, "_ref_active"))
        val_seg = summarize(segment(m_act["rows"], *val), m_act["decisions"],
                            m_act["protocol"])
        active_cand = {"psi_low": active[0], "psi_mid": active[1], "seg": val_seg,
                       "calibration_seg": summarize(segment(m_act["rows"], *cal),
                                                    m_act["decisions"], m_act["protocol"]),
                       "utility": utility(val_seg, lam_cost=args.lambda_cost,
                                          lam_rollback=args.lambda_rollback,
                                          ref_gpu_sec=ref_cost),
                       "feasible": True, "reason": "现役参照",
                       "_rows": m_act["rows"], "_decisions": m_act["decisions"],
                       "_is_active": True, "_label": "现役"}
        candidates.append(active_cand)

    robust = robust_pick(candidates, active=active, tol=args.tol)
    print(f"\n>>> 实验最优（utility）：psi_low={lo}, psi_mid={mi}")
    print(f"    {chosen['selection_note']}")
    print(f">>> 稳健推荐：psi_low={robust['psi_low']}, psi_mid={robust['psi_mid']}"
          f"（{robust['source']}，reason={robust['reason']}）")
    print(f"    {robust['note']}")

    sens = sensitivity(candidates, lo=val[0], hi=val[1])

    # ---- 冻结后：b02–b09 pre-holdout 汇总（仅报告）----
    print(f"\n[冻结] 在 b{PRE_HOLDOUT[0]}–b{PRE_HOLDOUT[1]} 上出 pre-holdout 汇总（不再改阈值）")
    pre = run_config(low=robust["psi_low"], mid=robust["psi_mid"], upto=PRE_HOLDOUT[1],
                     seed=args.seed, detector=args.detector, dataset=args.dataset,
                     out_dir=os.path.join(args.output, "_frozen_pre"))
    segments = {
        "calibration": summarize(segment(pre["rows"], *cal), pre["decisions"], pre["protocol"]),
        "validation": summarize(segment(pre["rows"], *val), pre["decisions"], pre["protocol"]),
        "pre_holdout": summarize(segment(pre["rows"], *PRE_HOLDOUT),
                                 pre["decisions"], pre["protocol"]),
    }
    # 候选配置在 b02–b09 上的对照（不含 b10）
    pre_cand = run_config(low=lo, mid=mi, upto=PRE_HOLDOUT[1], seed=args.seed,
                          detector=args.detector, dataset=args.dataset,
                          out_dir=os.path.join(args.output, "_frozen_pre_cand"))
    segments_candidate = {
        "calibration": summarize(segment(pre_cand["rows"], *cal),
                                 pre_cand["decisions"], pre_cand["protocol"]),
        "validation": summarize(segment(pre_cand["rows"], *val),
                                pre_cand["decisions"], pre_cand["protocol"]),
        "pre_holdout": summarize(segment(pre_cand["rows"], *PRE_HOLDOUT),
                                 pre_cand["decisions"], pre_cand["protocol"]),
    }

    # ---- 一次性 holdout ----
    holdout_run = False
    holdout_carried = None
    if args.run_holdout:
        print(f"\n[holdout] 阈值已冻结，运行一次 b{hold[0]}（不参与任何选择）")
        h = run_config(low=robust["psi_low"], mid=robust["psi_mid"], upto=hold[1],
                       seed=args.seed, detector=args.detector, dataset=args.dataset,
                       out_dir=os.path.join(args.output, "_frozen_holdout"),
                       allow_holdout=True)   # 唯一一次、且不参与任何选择
        segments["holdout"] = summarize(segment(h["rows"], *hold),
                                        h["decisions"], h["protocol"])
        holdout_run = True
    elif args.carry_holdout_from:
        # 结转上次已记录的 holdout，**不重跑 b10**
        try:
            with open(args.carry_holdout_from, encoding="utf-8") as f:
                prev = json.load(f)
            prev_h = (prev.get("segments") or {}).get("holdout")
            prev_cfg = (prev.get("chosen") or {})
            if prev_h:
                segments["holdout"] = prev_h
                holdout_carried = {
                    "from": args.carry_holdout_from,
                    "config": {"psi_low": prev_cfg.get("psi_low"),
                               "psi_mid": prev_cfg.get("psi_mid")},
                    "note": ("该 holdout 结果属于**上一次记录的配置**，本次未重跑 b10；"
                             "不得用于本次阈值选择，也不得据此调参"),
                }
                print(f"\n[holdout] 结转上次结果（不重跑 b10）："
                      f"配置 {prev_cfg.get('psi_low')}/{prev_cfg.get('psi_mid')}")
        except Exception as exc:                    # noqa: BLE001
            print(f"\n[holdout] 结转失败（{exc}），本次无 holdout 结果")
    else:
        print("\n[holdout] 未运行（加 --run-holdout 才会跑一次 b10）")

    # 网格饱和：不同网格点是否产生真正不同的结果？
    outcomes = {(round(c["seg"]["system"], 4), round(c["seg"]["gpu_sec"], 2),
                 c["seg"]["rollbacks"]) for c in candidates}
    saturation = {"n_points": len(points), "distinct_outcomes": len(outcomes)}

    report = {
        "dataset": args.dataset,
        "detector": args.detector,
        "seed": args.seed,
        "split": {"calibration": list(cal), "validation": list(val),
                  "holdout": list(hold), "pre_holdout": list(PRE_HOLDOUT)},
        "grid": {"psi_low": lows, "psi_mid": mids, "n_points": len(points)},
        "objective": {"utility": "gain - lambda_cost*normalized_cost - lambda_rollback*rollback_rate",
                      "lambda_cost": args.lambda_cost,
                      "lambda_rollback": args.lambda_rollback,
                      "normalized_cost_ref_gpu_sec": ref_cost,
                      "tol": args.tol,
                      "constraints": {"violations": 0, "gain_min": need_gain,
                                      "max_rollback_rate": args.max_rollback_rate,
                                      "max_wasted_ratio": args.max_wasted_ratio}},
        "candidates": [{k: v for k, v in c.items()
                        if k != "calibration_seg" and not k.startswith("_")}
                       for c in candidates],
        "chosen": {"psi_low": lo, "psi_mid": mi,
                   "selection_note": chosen["selection_note"],
                   "validation": chosen["seg"]},
        # ---- 稳健推荐（默认值）----
        "active_thresholds": {"psi_low": robust["psi_low"],
                              "psi_mid": robust["psi_mid"]},
        "calibrated_candidate": {"psi_low": lo, "psi_mid": mi,
                                 "validation": chosen["seg"],
                                 "segments_b02_b09": segments_candidate},
        "robust_choice": {"psi_low": robust["psi_low"], "psi_mid": robust["psi_mid"],
                          "source": robust["source"], "reason": robust["reason"],
                          "evidence_batches": robust["evidence_batches"],
                          "note": robust["note"]},
        "fallback_reason": robust["reason"] if robust["reason"] != "multi_batch_evidence"
                           else None,
        "changed_default": (robust["psi_low"], robust["psi_mid"]) != active,
        "sensitivity": sens,
        "holdout_carried": holdout_carried,
        "references": {
            "从不更新(静态基线)": {"psi_low": 1e9, "psi_mid": 2e9, "validation": never_val},
            "旧阈值 1.2/2.5": {"psi_low": OLD_THRESHOLD[0], "psi_mid": OLD_THRESHOLD[1],
                              "validation": old_val},
            "稳健推荐": {"psi_low": robust["psi_low"], "psi_mid": robust["psi_mid"],
                        "validation": segments["validation"]},
            "实验候选": {"psi_low": lo, "psi_mid": mi,
                        "validation": segments_candidate["validation"]},
        },
        "segments": segments,
        "grid_saturation": saturation,
        "holdout_run": holdout_run,
        "holdout_used_for_selection": False,
        "elapsed_sec": round(time.time() - t0, 1),
        "limitations": [
            "只有 10 个批次：validation 仅 4 个批次，标定结果**统计上不稳定**，"
            "不能称为最终最优阈值",
            "这是 Gas 公开数据上的时间顺序标定结果，**不具备跨数据集泛化能力**",
            "网格是离散的，真实最优可能落在网格点之间",
            "未做多随机种子重复；单次结果可能受个别批次的偶然波动影响",
            "holdout 只在 --run-holdout 时运行一次，不得据此回头调参",
            f"网格饱和：{saturation['n_points']} 个点只有 "
            f"{saturation['distinct_outcomes']} 种不同结果，阈值区分度低",
            "holdout 上未做旧阈值对照（遵守只跑一次的纪律），因此"
            "\"标定阈值 vs 旧阈值\"的优势只在 validation 上成立，未在 holdout 上复现",
            f"实验候选与现役配置的动作差异只出现在 "
            f"{len(robust['evidence_batches'])} 个批次"
            f"（{robust['evidence_batches']}）——单批次证据，不足以替换默认值",
            "敏感性分析只是统计层面的留一重算，不是新的独立实验，"
            "不得作为\"已验证的最终测试成绩\"",
        ],
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
    }

    out_json = os.path.join(args.output, "calibration_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    out_md = os.path.join(args.output, "calibration_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    # 可直接被 run_pipeline 读取的配置快照
    out_cfg = os.path.join(args.output, "thresholds.json")
    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump({
            "active_thresholds": report["active_thresholds"],
            "calibrated_candidate": {"psi_low": lo, "psi_mid": mi},
            "fallback_reason": report["fallback_reason"],
            "changed_default": report["changed_default"],
            "run_pipeline_args": ["--psi-low", str(report["active_thresholds"]["psi_low"]),
                                  "--psi-mid", str(report["active_thresholds"]["psi_mid"])],
            "note": "默认仍用 active_thresholds；实验候选需显式传参才会生效",
        }, f, ensure_ascii=False, indent=2)
    print(f"配置快照：{out_cfg}")

    print(f"\n报告：{out_json}")
    print(f"      {out_md}")
    if segments.get("holdout"):
        h = segments["holdout"]
        print(f"\n一次性 holdout（b{hold[0]}）：系统 {h['system']:.4f} vs 基线 "
              f"{h['baseline']:.4f}（增益 {h['gain']:+.4f}）")
    return 0


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:                               # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationError as exc:
        print(f"[!] 标定终止：{exc}", file=sys.stderr)
        raise SystemExit(2)

# -*- coding: utf-8 -*-
"""多种子 + 自助法：把「单次运行的数字」变成「带区间的结论」。

为什么需要：
    现在所有结论都是跑一次得到的，比如「气体数据 0.783」。
    但换一个随机种子，它可能是 0.75 也可能是 0.82 —— 没有区间，
    这个数字就没有说服力，答辩被问「做过几次」也答不上来。

做两件事：
    1) 多种子：换 random_state 重跑，看结果对种子有多敏感。
       对 logreg 这类确定性模型，种子变化主要影响数据划分/采样；
       对 rf / mlp / gbdt 影响权重初始化。
    2) 自助法（bootstrap）：对**逐批分数**重采样，得到均值的 95% 置信区间。
       这衡量的是"批次抽样"带来的不确定性，和种子方差是两回事，互补。

用法：
    python scripts/sweep.py --dataset gas --seeds 5
    python scripts/sweep.py --dataset synthetic --seeds 5 --regenerate
    python scripts/sweep.py --dataset gas --seeds 5 --model gbdt --llm pydantic-sim
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

import numpy as np

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(args_list: list[str], out: str) -> dict | None:
    cmd = [PY, "-X", "utf8", os.path.join(ROOT, "scripts", "run_pipeline.py"),
           *args_list, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", cwd=ROOT)
    p = os.path.join(ROOT, out, "metrics.json")
    if not os.path.exists(p):
        tail = (r.stderr or "").strip().splitlines()[-3:]
        print(f"    [!] 失败：{' | '.join(tail)}")
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    rows = d["rows"]
    used = d["ledger"]["used"]
    base = float(np.mean([r["f1_baseline"] for r in rows]))
    sysf = float(np.mean([r["f1_system"] for r in rows]))
    return {"base": base, "system": sysf, "gain": sysf - base,
            "cost": float(used.get("gpu_sec", 0.0)),
            "model": d.get("model"),
            "per_batch": [float(r["f1_system"]) for r in rows],
            "n_batches": len(rows)}


def bootstrap_ci(values: list[float], n_boot: int = 4000,
                 alpha: float = 0.05, seed: int = 0):
    """对均值的自助法置信区间。"""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=v.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# 给定数据与 random_state 下，这些模型的结果完全确定：
# 换 seed 不会改变模型本身，因此"seed 方差 = 0"是**算法性质**，不是"结论稳健"的证据。
DETERMINISTIC_MODELS = frozenset({
    "logreg",       # lbfgs，确定性
    "gnb",          # 闭式解
    "ridge",        # 闭式解
})

# 成本的标准差 / 均值 超过这个比例，就认为"单位成本增益"不稳定
COST_CV_WARN = 0.20
# 系统分数随 seed 的标准差超过这个值，就认为结论对随机性敏感
SEED_STD_WARN = 0.03


def variance_report(*, model_kind: str, n_seeds: int, system: list[float],
                    gain: list[float], cost: list[float],
                    ci: tuple[float, float], pooled_n: int) -> dict:
    """把不确定性拆成三类，分别给结论。

    ① 模型方差（换 seed）：只有随机模型才有
    ② 批次不确定性（bootstrap）：衡量"换一批数据分数会不会变"
    ③ 运行成本方差：即使分数稳定，成本可能不稳，但它直接决定客观分

    绝不因为"std == 0"就说"结论稳健"——确定性模型的 std 天然为 0。
    """
    def _sd(xs):
        return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0

    sys_sd = _sd(system)
    gain_sd = _sd(gain)
    cost_sd = _sd(cost)
    cost_mean = float(statistics.mean(cost)) if cost else 0.0
    cost_cv = (cost_sd / cost_mean) if cost_mean > 1e-9 else 0.0

    deterministic = model_kind in DETERMINISTIC_MODELS
    seed_varies = (not deterministic) and sys_sd > 1e-9
    cost_unstable = cost_cv > COST_CV_WARN

    lines = ["", "  不确定性拆成三类（不要只看一个 std）："]
    # ①
    if deterministic:
        lines.append(f"  ① 模型方差（seed）：**不适用** —— {model_kind} 是确定性算法，"
                     f"换 seed 不会改变模型。std={sys_sd:.4f} 不代表结论稳健。")
    elif seed_varies:
        tag = "  [!] 超过 %.2f，结论对随机性敏感，引用时必须带区间" % SEED_STD_WARN \
            if sys_sd > SEED_STD_WARN else ""
        lines.append(f"  ① 模型方差（seed）：系统 std={sys_sd:.4f}，"
                     f"增益 std={gain_sd:.4f}{tag}")
    else:
        lines.append(f"  ① 模型方差（seed）：系统 std={sys_sd:.4f}，"
                     f"{model_kind} 虽非确定性算法，但在这份数据上 seed 未造成差异")
    # ②
    lines.append(f"  ② 批次不确定性（bootstrap，{pooled_n} 个批次样本）："
                 f"系统分数 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    # ③
    if cost_unstable:
        lines.append(f"  ③ 运行成本方差：{cost_mean:.4f}s ± {cost_sd:.4f}s"
                     f"（CV={cost_cv:.0%}）—— **成本不稳定，单位成本增益不可靠**")
    else:
        lines.append(f"  ③ 运行成本方差：{cost_mean:.4f}s ± {cost_sd:.4f}s"
                     f"（CV={cost_cv:.0%}），成本波动在可接受范围")

    # 结论：三类合起来说，而不是单看 std
    if deterministic and cost_unstable:
        verdict = ("  结论：分数可复现（确定性模型），但**成本**波动大，"
                   "单位成本增益只能给区间，不能给单一数字。")
    elif deterministic:
        verdict = ("  结论：该模型结果可复现；剩余不确定性主要来自批次抽样，"
                   "见 ② 的置信区间。")
    elif seed_varies and cost_unstable:
        verdict = "  结论：分数与成本都对随机性敏感，两个都要带区间引用。"
    elif seed_varies:
        verdict = "  结论：分数随 seed 波动，引用时必须带上 ② 与 ① 的区间。"
    else:
        verdict = "  结论：seed 未造成差异，不确定性主要来自批次抽样（②）。"
    lines.append(verdict)

    return {
        "model_kind": model_kind,
        "deterministic_model": deterministic,
        "seed_varies_model": seed_varies,
        "system_std": sys_sd,
        "gain_std": gain_sd,
        "cost_std": cost_sd,
        "cost_cv": cost_cv,
        "cost_unstable": cost_unstable,
        "n_seeds": n_seeds,
        "ci": ci,
        "pooled_n": pooled_n,
        "lines": lines,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gas")
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm", default="none")
    ap.add_argument("--seeds", type=int, default=5, help="跑几个种子（0..N-1）")
    ap.add_argument("--regenerate", action="store_true",
                    help="数据集随种子变化时先重新造数据（仅 synthetic 有意义）")
    ap.add_argument("--extra", default="", help="透传给 run_pipeline 的额外参数")
    ap.add_argument("--out", default="out/sweep")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    results = []

    print(f"数据集 {args.dataset} · 模型 {args.model or '默认'} · "
          f"决策层 {args.llm} · 种子 {len(seeds)} 个")
    print("-" * 64)

    for s in seeds:
        if args.regenerate:
            subprocess.run([PY, "-X", "utf8",
                            os.path.join(ROOT, "scripts", "make_data.py"),
                            "--seed", str(s)], capture_output=True,
                           cwd=ROOT)
        extra = args.extra.split() if args.extra else []
        cli = ["--dataset", args.dataset, "--seed", str(s), "--llm", args.llm,
               *extra]
        if args.model:
            cli += ["--model", args.model]
        res = run_one(cli, f"{args.out}/seed_{s}")
        if res is None:
            continue
        results.append((s, res))
        print(f"  seed {s:<3} 基线 {res['base']:.3f}  系统 {res['system']:.3f}  "
              f"增益 {res['gain']:+.3f}  成本 {res['cost']:6.2f}s")

    if not results:
        print("\n没有成功的运行。")
        return

    base = [r["base"] for _, r in results]
    sysf = [r["system"] for _, r in results]
    gain = [r["gain"] for _, r in results]
    cost = [r["cost"] for _, r in results]
    # 实际用的模型名（args.model 为空时由 run_pipeline 决定，取第一次运行的结果）
    model_kind = results[0][1].get("model") or args.model or "unknown"

    def stat(xs):
        if len(xs) == 1:
            return xs[0], 0.0, xs[0], xs[0]
        return (statistics.mean(xs), statistics.stdev(xs), min(xs), max(xs))

    print("-" * 64)
    for name, xs in (("基线", base), ("系统", sysf),
                     ("增益", gain), ("成本", cost)):
        m, sd, lo, hi = stat(xs)
        unit = "s" if name == "成本" else ""
        print(f"  {name:<4} {m:8.4f}{unit}  ± {sd:.4f}   "
              f"区间 [{lo:.4f}, {hi:.4f}]")

    # 汇总：把所有批次的分数合在一起做自助法（衡量批次抽样不确定性）
    pooled = [v for _, r in results for v in r["per_batch"]]
    lo, hi = bootstrap_ci(pooled)
    print(f"\n  系统分数的 95% 置信区间（自助法，{len(pooled)} 个批次样本）："
          f"[{lo:.4f}, {hi:.4f}]")

    m_gain, sd_gain, _, _ = stat(gain)
    m_cost, sd_cost, _, _ = stat(cost)
    if m_cost:
        print(f"  单位成本增益：{m_gain / m_cost:.4f}"
              f"（增益 {m_gain:.4f} ÷ 成本 {m_cost:.4f}s）")

    # 【修复】把三类不确定性分开报告，并写进 JSON，避免下游只看到一个 std 就下结论
    extra = {}
    if args.seeds > 1:
        rep = variance_report(model_kind=model_kind, n_seeds=len(results),
                              system=sysf, gain=gain, cost=cost,
                              ci=(lo, hi), pooled_n=len(pooled))
        for line in rep["lines"]:
            print(line)
        extra = {
            "model_kind": rep["model_kind"],
            "deterministic_model": rep["deterministic_model"],
            "seed_varies_model": rep["seed_varies_model"],
            "system_std_seed": round(rep["system_std"], 6),
            "cost_std_seed": round(rep["cost_std"], 6),
            "cost_cv": round(rep["cost_cv"], 4),
            "cost_unstable": rep["cost_unstable"],
            "bootstrap_n": rep["pooled_n"],
        }

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    out_json = os.path.join(ROOT, args.out, "sweep.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset, "model": model_kind, "llm": args.llm,
            "seeds": [s for s, _ in results],
            "rows": [{"seed": s, **{k: v for k, v in r.items()
                                    if k != "per_batch"}}
                     for s, r in results],
            "summary": {
                "base_mean": stat(base)[0], "base_std": stat(base)[1],
                "system_mean": stat(sysf)[0], "system_std": stat(sysf)[1],
                "gain_mean": m_gain, "gain_std": sd_gain,
                "cost_mean": m_cost, "cost_std": sd_cost,
                "bootstrap_ci": [lo, hi],
                **extra,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  明细 → {os.path.relpath(out_json, ROOT)}")


if __name__ == "__main__":
    main()

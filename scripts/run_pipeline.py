"""主流程：把「静态基线」和「自进化闭环」放在同一条漂移流上跑，对比两条曲线。

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --dataset gas --model gbdt

产出（全部写到 out/）：
    metrics.json      每批分数，用于画曲线
    decisions.jsonl   每一次决策的完整记录（门槛③的证据）
    curve.png         静态基线 vs 我们，两条线
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolver.adapters import dataset as ds          # noqa: E402
from evolver.adapters import metric as mt          # noqa: E402
from evolver.adapters import model as md           # noqa: E402
from evolver.config import Config                  # noqa: E402
from evolver.engine import llm_agent as la         # noqa: E402
from evolver.engine import policy                  # noqa: E402
from evolver.ledger.ledger import Ledger           # noqa: E402
from evolver.loop import decide as dc              # noqa: E402
from evolver.loop import diagnose as dg            # noqa: E402
from evolver.loop import evaluate as ev            # noqa: E402
from evolver.loop import release as rl             # noqa: E402
from evolver.loop.monitor import Monitor, label_psi  # noqa: E402
from evolver.memory import ExperienceTable  # noqa: E402
from evolver.protocol import LocalEvaluator, ProtocolViolation, Submission  # noqa: E402


def recent_window(ev, start: int, end: int, tau: float = 0.5):
    """取最近几批拼起来，用于增量重训。

    只从评测器**已下发**的批次里取——这是协议强制的，
    直接去 adapter 捞未来批次会被 LocalEvaluator 拦下。

    关键：越新的批次权重越高（近似指数衰减）。
    否则历史里的旧规则会把模型往回拽，重训反而更差。
    """
    Xs, ys, ws = [], [], []
    for b, batch in ev.history(start, end):
        w = float(np.exp(-(end - b) / tau))
        Xs.append(batch.X)
        ys.append(batch.y)
        ws.append(np.full(len(batch.y), w))
    if not Xs:
        raise ProtocolViolation(f"没有可用的历史批次（{start}..{end}）")
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def plot(rows: list[dict], path: str, concept_start: int | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    bs = [r["batch"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    ax.plot(bs, [r["f1_baseline"] for r in rows], "o--", lw=2,
            color="#DC2626", label="静态基线（部署后不再更新）")
    ax.plot(bs, [r["f1_system"] for r in rows], "o-", lw=2.6,
            color="#2563EB", label="我们的系统（自进化闭环）")
    ax.set_xlabel("数据批次（时间推进）")
    ax.set_ylabel("F1（macro）")
    _t = "静态基线 vs 自进化闭环"
    if concept_start is not None:
        _t += f"（第 {concept_start} 批判定标准改变）"
    ax.set_title(_t, fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower left", framealpha=0.9)

    if concept_start is not None:
        ax.axvline(concept_start, color="#D97706", ls="--", lw=1.2, alpha=0.8)

    ax2 = ax.twinx()
    ax2.plot(bs, [r["psi_mean"] for r in rows], ":", lw=1.4, color="#94A3B8",
             label="PSI（漂移强度，右轴）")
    ax2.set_ylabel("PSI", color="#94A3B8")
    ax2.tick_params(axis="y", colors="#94A3B8")

    for r in rows:
        if r.get("decision") == "accept":
            ax.annotate("▲", (r["batch"], r["f1_system"]), color="#059669",
                        fontsize=9, ha="center", va="bottom")
        elif r.get("decision") == "rollback":
            ax.annotate("×", (r["batch"], r["f1_system"]), color="#DC2626",
                        fontsize=9, ha="center", va="bottom")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "gas"])
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbdt", "rf"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--psi-low", type=float, default=None,
                    help="稳定阈值（低于它不动）。不填用 config 默认值")
    ap.add_argument("--no-memory", action="store_true",
                    help="关闭经验表（用于 A/B 对照）")
    ap.add_argument("--psi-mid", type=float, default=None,
                    help="重型阈值（超过它直接重训）。不填用 config 默认值")
    ap.add_argument("--llm", default="none",
                    choices=["none", "mock", "mock_bad", "pytrio", "ollama"],
                    help="决策层后端：mock 验证链路，pytrio/ollama 接真模型")
    args = ap.parse_args()

    cfg = Config()
    # 阈值必须按数据集标定：合成数据 PSI 0.1~0.3，真实气体传感器 1~4，
    # 用同一套绝对值必然失效（要么从不动作，要么永远走最贵那档）。
    if args.psi_low is not None:
        cfg.psi_low = args.psi_low
    if args.psi_mid is not None:
        cfg.psi_mid = args.psi_mid
    adapter = ds.build(args.dataset)
    metric = mt.build("f1_macro")
    os.makedirs(args.out, exist_ok=True)

    print(f"数据集：{adapter.name}  批次：{adapter.n_batches}  模型：{args.model}")

    # ── 评测协议：一问一答 ──
    # 没交卷就拿不到下一批；任何"偷看未来"的动作都会被直接拦下。
    proto = LocalEvaluator(adapter, strict=True)
    first = proto.next_batch()
    base = md.build(args.model, random_state=cfg.random_state).fit(first.X, first.y)

    baseline = base.clone()            # 冻结，永不更新
    rel = rl.ReleaseManager(base.clone(), 0.0, cfg.gate_margin)

    ref_score = metric.score(first.y, rel.current.predict(first.X))
    rel.current_score = ref_score

    agent = la.build(args.llm)
    if agent.enabled:
        print(f"决策层：{args.llm}（{getattr(agent, 'model', '-')}）")
    else:
        print("决策层：规则引擎（未接大模型）")

    memory = None if args.no_memory else ExperienceTable()
    monitor = Monitor(first.X, cfg.psi_bins)
    ledger = Ledger(os.path.join(args.out, "decisions.jsonl"),
                    cfg.budget_gpu_sec, cfg.budget_labels, cfg.budget_tokens)

    rows = []
    best = ref_score
    last_update = 0
    last_action, last_gain = None, 0.0

    # 第 1 轮：冷启动训练完，也要交一份判断
    proto.submit(Submission(
        round_no=1, batch_id=first.batch_id, decision="deploy",
        model_version=rel.version,
        reason="冷启动：以首批数据训练初始模型并部署",
        predictions=rel.current.predict(first.X),
        self_reported={"f1_system": round(ref_score, 4)},
    ))

    while True:
        batch = proto.next_batch()
        if batch is None:
            break
        b = batch.batch_id
        t0 = time.time()

        # 对照组：静态基线
        f1_base = metric.score(batch.y, baseline.predict(batch.X))

        # ① 监测
        rep = monitor.check(batch.X)
        lshift = label_psi(first.y, batch.y)

        # ② 归因
        diag = dg.diagnose(rep, lshift, cfg)

        # 用实验证据反过来修正归因：便宜手段试过没用，说明不是单纯尺度问题
        if last_action == "refresh_normalization" and last_gain < cfg.min_gain:
            diag = {**diag, "kind": "concept_inferred",
                    "reason": f"标准化校准仅带来 {last_gain:+.3f} 增益，反推不是单纯分布平移"}

        # ③ 决策
        state = dc.State(
            batch_id=b, psi_mean=rep["psi_mean"],
            current_score=rel.current_score, best_score=best,
            steps_since_update=b - last_update,
            last_action=last_action, last_gain=last_gain,
            warmup=b <= cfg.warmup_batches,
            history=[{"psi_mean": m["psi_mean"]} for m in monitor.history[:-1]],
        )
        history_read = state.read_history()
        dec = policy.decide(state, diag, cfg, agent, memory)

        # ④ 执行 + 影子评估
        cand, shadow = None, None
        if dec["action"] == "refresh_normalization":
            cand = rel.current.clone().refresh_normalization(batch.X)
        elif dec["action"] == "refit_model":
            # 只能用 b-1 及更早的数据训练（标签滞后），再到第 b 批上检验。
            # 用第 b 批的标签训练、又在第 b 批上验收 = 数据泄漏，分数会虚高。
            Xw, yw, ww = recent_window(proto, max(1, b - 4), b - 1)
            # 续训而不是重建：把“之前学到的”带过来。
            # 但必须同时刷新标准化——否则权重配着旧尺度，反而被拽住（实测更差）。
            cand = rel.current.clone().fit_continue(Xw, yw, ww, refresh_scaler=True)

        if cand is not None:
            shadow = ev.shadow(rel.current, cand, batch, metric)
            rel_rec = rel.try_release(cand, shadow["after"], shadow["before"])
            gain = shadow["gain"]
        else:
            rel_rec = {"decision": "noop", "reason": dec["reason"]}
            gain = 0.0

        # ⑤ 本批最终分数
        f1_sys = metric.score(batch.y, rel.current.predict(batch.X))
        best = max(best, f1_sys)

        if dec["action"] != "do_nothing":
            ledger.charge(gpu_sec=dec["cost_sec"],
                          labels=len(batch.y) if dec["action"] == "refit_model" else 0)
        if agent.enabled and isinstance(dec.get("llm"), dict):
            ledger.charge(tokens=int(dec["llm"].get("tokens", 0)))

        # 门槛③ 的证据：一条完整记录
        ledger.log({
            "cycle": len([r for r in rows if r.get("action") not in (None, "do_nothing")]) + 1,
            "batch": b,
            "history_read": history_read,
            "hypothesis": f"诊断={diag['kind']}；{dec['hypothesis']}",
            "experiment": {"action": dec["action"], "cost_sec": round(dec["cost_sec"], 3),
                           "diagnosis": diag["kind"],
                           "psi_mean": round(rep["psi_mean"], 4),
                           "decider": dec.get("source", "rule")},
            "llm": ({"action": dec["llm"].get("action"),
                      "reason": dec["llm"].get("reason"),
                      "model": dec["llm"].get("model"),
                      "latency": dec["llm"].get("latency")}
                     if isinstance(dec.get("llm"), dict) and "action" in dec["llm"] else None),
            "note": dec.get("note"),
            "result": {"f1_baseline": round(f1_base, 4),
                       "f1_before": round(shadow["before"], 4) if shadow else None,
                       "f1_after": round(shadow["after"], 4) if shadow else None},
            "decision": rel_rec["decision"],
            "reason": rel_rec["reason"],
            "elapsed_sec": round(time.time() - t0, 3),
        })

        rows.append({
            "batch": b, "f1_baseline": round(f1_base, 4), "f1_system": round(f1_sys, 4),
            "psi_max": round(rep["psi_max"], 4), "psi_mean": round(rep["psi_mean"], 4),
            "label_shift": round(lshift, 4), "diagnosis": diag["kind"],
            "action": dec["action"], "decision": rel_rec["decision"],
        })

        flag = {"accept": "[OK] 上线", "rollback": "[X] 回滚", "noop": "-"}[rel_rec["decision"]]
        src = {"rule": "规则", "rule+llm": "规则+LLM", "llm": "LLM",
               "memory": "记忆", "memory+llm": "记忆+LLM"}.get(
                   dec.get("source", "rule"), dec.get("source", "?"))
        print(f"  b{b:02d} | 基线 {f1_base:.3f} | 系统 {f1_sys:.3f} | "
              f"PSI {rep['psi_mean']:.2f} | {diag['kind']:9s} | "
              f"{dec['action']:20s} | {flag} | {src}")

        # ── 交卷：本批的判断必须交出去，才拿得到下一批 ──
        proto.submit(Submission(
            round_no=len(rows) + 1,
            batch_id=b,
            decision="deploy" if rel_rec["decision"] == "accept" else "keep",
            model_version=rel.version,
            reason=f"{dec['action']} -> {rel_rec['decision']}",
            predictions=rel.current.predict(batch.X),
            self_reported={"f1_system": round(f1_sys, 4),
                           "f1_baseline": round(f1_base, 4)},
        ))

        # 把这次的战绩写进经验表（下一轮同类漂移可以直接查）
        if memory is not None:
            memory.record(diag["kind"], dec["action"], gain, dec["cost_sec"],
                          rel_rec["decision"])

        last_action, last_gain = dec["action"], gain
        if rel_rec["decision"] == "accept":
            last_update = b
            # 模型真正换代后，把漂移参照系挪到“当前已知良好状态”，
            # 避免参照系永远钉在部署那天、导致反复做无用更新。
            if dec["action"] == "refit_model":
                monitor.reset_reference(batch.X)

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": adapter.name, "model": args.model,
                   "rows": rows, "ledger": ledger.summary(),
                   "protocol": proto.summary()}, f,
                  ensure_ascii=False, indent=2)

    concept_start = None
    for b in range(1, adapter.n_batches + 1):
        try:
            if float(adapter.get(b).meta.get("concept_flag", 0.0)) > 0:
                concept_start = b
                break
        except Exception:
            break
    plot(rows, os.path.join(args.out, "curve.png"), concept_start)

    n_acc = sum(1 for r in rows if r["decision"] == "accept")
    n_roll = sum(1 for r in rows if r["decision"] == "rollback")
    n_act = sum(1 for r in rows if r["action"] != "do_nothing")
    print(f"\n平均分  基线 {np.mean([r['f1_baseline'] for r in rows]):.3f}  "
          f"系统 {np.mean([r['f1_system'] for r in rows]):.3f}")
    print(f"决策    动作 {n_act} 次 | 上线 {n_acc} 次 | 回滚 {n_roll} 次")
    print(f"账本    {ledger.summary()}")
    if agent.enabled:
        print(f"决策层  {agent.stats()}")
    if memory is not None:
        memory.print_summary()
        memory.to_json(os.path.join(args.out, "experience.json"))
        print()

    ps = proto.summary()
    print(f"协议    {ps['rounds']} 轮 | 部署 {ps['deploy']} | 维持 {ps['keep']} | "
          f"违规 {ps['violations']}")
    print(f"产出    {args.out}/metrics.json  decisions.jsonl  curve.png")
    if n_act < 3:
        print("[!] 有效动作不足 3 次，门槛③可能过不了——加大漂移强度或调低阈值")


if __name__ == "__main__":
    main()

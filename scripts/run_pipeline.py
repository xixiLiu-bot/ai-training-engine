"""主流程：把「静态基线」和「自进化闭环」放在同一条漂移流上跑，对比两条曲线。

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --dataset gas --model gbdt
    python scripts/run_pipeline.py --dataset synthetic_rul          # 回归任务
    python scripts/run_pipeline.py --detector all                   # 同时算 PSI/KS/W1
    python scripts/run_pipeline.py --gate-window 3 --gate-bootstrap 500

产出（全部写到 out/）：
    metrics.json      每批分数，用于画曲线
    decisions.jsonl   每一次决策的完整记录（门槛③的证据）
    curve.png         静态基线 vs 我们，两条线
    run.jsonl         实验记录（门槛②；可选 SwanLab）

【本轮整改】
  A1 主循环加兜底：任何异常都必须输出结果，不能整轮 0 分
  A2 预算超限（BudgetExceeded）降级为"本批不动"，而不是崩溃
  A4 按数据集任务类型分派指标（分类 F1 / 回归 MAE）
  B1 --detector 选择 PSI / KS / Wasserstein / all
  B2 --gate-window / --gate-bootstrap 多批窗口 + Bootstrap 置信下界
  B4 每批写实验记录（本地 JSONL，SwanLab 可选）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 【配置】先读项目根目录的 .env（存在才读），再解析命令行参数。
# 已有环境变量优先，不被 .env 覆盖。
from evolver.envfile import load_env, summary as env_summary  # noqa: E402

_ENV = load_env()

from evolver.adapters import dataset as ds          # noqa: E402
from evolver.adapters import metric as mt          # noqa: E402
from evolver.adapters import model as md           # noqa: E402
from evolver.config import Config                  # noqa: E402
from evolver.engine import llm_agent as la         # noqa: E402
from evolver.engine import policy                  # noqa: E402
from evolver.ledger.ledger import BudgetExceeded, Ledger  # noqa: E402
from evolver.logs.recorder import Recorder         # noqa: E402
from evolver.loop import decide as dc              # noqa: E402
from evolver.loop import diagnose as dg
from evolver.loop.calibrate import ThresholdCalibrator
from evolver.memory import ExperienceTable            # noqa: E402
from evolver.loop import evaluate as ev            # noqa: E402
from evolver.loop import release as rl             # noqa: E402
from evolver.loop.monitor import Monitor, label_drift  # noqa: E402
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

    valid = [r for r in rows if r.get("f1_system") is not None]
    bs = [r["batch"] for r in valid]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    ax.plot(bs, [r["f1_baseline"] for r in valid], "o--", lw=2,
            color="#DC2626", label="静态基线（部署后不再更新）")
    ax.plot(bs, [r["f1_system"] for r in valid], "o-", lw=2.6,
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
    ax2.plot(bs, [r["psi_mean"] for r in valid], ":", lw=1.4, color="#94A3B8",
             label="PSI（漂移强度，右轴）")
    ax2.set_ylabel("PSI", color="#94A3B8")
    ax2.tick_params(axis="y", colors="#94A3B8")

    for r in valid:
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


# ─────────────────────────────────────────────────────────────
# 【A1/A2】兜底工具：任何一步失败，都要能交出一份判断
# ─────────────────────────────────────────────────────────────
def safe_predict(model, X):
    """预测失败不能拖垮流程——宁可交 None。"""
    try:
        return model.predict(X)
    except Exception:                              # noqa: BLE001
        return None


def safe_score(metric, y_true, pred):
    if pred is None:
        return None
    try:
        return float(metric.score(y_true, pred))
    except Exception:                              # noqa: BLE001
        return None


def submit_fallback(proto, rel, batch, round_no, metric, reason: str):
    """本批处理失败时的保底交卷：维持现役模型，照样交出预测。"""
    pred = safe_predict(rel.current, batch.X)
    score = safe_score(metric, batch.y, pred)
    try:
        proto.submit(Submission(
            round_no=round_no, batch_id=batch.batch_id, decision="keep",
            model_version=rel.version, reason=reason, predictions=pred,
            self_reported={"f1_system": None if score is None else round(score, 4)},
        ))
    except Exception:                              # noqa: BLE001
        pass                                       # 连交卷都失败就只能放弃本批
    return score


def _guard_identity(adapter, args, cfg) -> None:
    """按数据身份决定能不能跑。

    - public_substitute / synthetic / validation / public_holdout：照常跑
    - official_public：**拒绝**，只能走 scripts/validate_official.py
    - official_test：必须 --frozen-config 且指纹一致，并申请一次性凭据
    """
    from evolver.adapters import identities
    from evolver import evaluation as evalmod

    ident = getattr(adapter, "data_identity", identities.UNKNOWN)
    try:
        identities.normalize(ident)
    except identities.DataIdentityError as exc:
        print(f"[!] 数据身份异常，拒绝运行：{exc}")
        raise SystemExit(2)

    if ident == identities.OFFICIAL_PUBLIC:
        print("[!] 官方公开集**不允许**进入闭环运行（只允许格式体检）。")
        print("    请执行：python scripts/validate_official.py --root data/official")
        raise SystemExit(2)

    if ident != identities.OFFICIAL_TEST:
        return

    frozen_path = getattr(args, "frozen_config", None)
    if not frozen_path:
        print("[!] official_test 必须提供已冻结的配置：--frozen-config <thresholds.json>")
        print("    官方测试流只允许在参数冻结后正式评估一次。")
        raise SystemExit(2)
    try:
        frozen = evalmod.assert_frozen(
            frozen_path, psi_low=cfg.psi_low, psi_mid=cfg.psi_mid,
            detector=cfg.detector, gate_margin=cfg.gate_margin,
            margin_mode=getattr(cfg, "margin_mode", "absolute"))
    except evalmod.FreezeError as exc:
        print(f"[!] 冻结校验失败：{exc}")
        raise SystemExit(2)

    ticket_path = getattr(args, "eval_ticket", None) or os.path.join(
        os.path.dirname(os.path.abspath(frozen_path)), "eval_ticket.json")
    try:
        evalmod.acquire_ticket(
            ticket_path, dataset=getattr(adapter, "name", "official_test"),
            identity=ident, config_sha=frozen["config_sha"],
            frozen_at=frozen.get("frozen_at"))
    except evalmod.TicketError as exc:
        print(f"[!] 一次性凭据校验失败：{exc}")
        raise SystemExit(2)
    print(f"[OK] 冻结配置校验通过（config_sha={frozen['config_sha'][:12]}…），"
          f"一次性凭据：{ticket_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "gas", "gas_drift", "synthetic_rul",
                             "bearing", "rul"])
    ap.add_argument("--model", default=None,
                    choices=["logreg", "gbdt", "rf", "ridge"],
                    help="不填则按任务类型自动选：分类=logreg，回归=ridge")
    ap.add_argument("--out", default="out")
    ap.add_argument("--psi-low", type=float, default=None,
                    help="稳定阈值（低于它不动）。不填用 config 默认值")
    ap.add_argument("--psi-mid", type=float, default=None,
                    help="重型阈值（超过它直接重训）。不填用 config 默认值")
    ap.add_argument("--task-type", default=None,
                    choices=["classification", "regression"],
                    help="不填则用数据集声明的任务类型")
    ap.add_argument("--detector", default="psi",
                    choices=["psi", "ks", "wasserstein", "all"],
                    help="漂移检测方法")
    ap.add_argument("--gate-window", type=int, default=1,
                    help="门控用最近几批增益（1=只看当前批，旧行为）")
    ap.add_argument("--gate-bootstrap", type=int, default=0,
                    help="Bootstrap 重采样次数（0=关闭）")
    ap.add_argument("--margin-mode", default=None,
                    choices=["absolute", "relative"],
                    help="门控 margin 量纲。不填则回归用 relative、分类用 absolute")
    ap.add_argument("--enable-output-calibration", action="store_true",
                    help="启用最便宜的一档动作 calibrate_output（改输出）。"
                         "默认关闭以保持既有结果不变")
    ap.add_argument("--no-recorder", action="store_true",
                    help="不写 run.jsonl 实验记录")
    ap.add_argument("--frozen-config", default=None,
                    help="已冻结的 thresholds.json（official_test 必填）")
    ap.add_argument("--eval-ticket", default=None,
                    help="一次性运行凭据路径（默认与 frozen-config 同目录）")
    ap.add_argument("--swanlab", action="store_true",
                    help="强制启用 SwanLab 云端上传（等价于 SWANLAB_ENABLED=true）")
    ap.add_argument("--experiment", default=None,
                    help="SwanLab 实验名（也可用 SWANLAB_EXPERIMENT 环境变量）")
    ap.add_argument("--no-memory", action="store_true",
                    help="关闭经验表（A/B 对照用）")
    ap.add_argument("--load-experience", default=None,
                    help="加载一份已有经验表（跨数据集迁移实验用）")
    ap.add_argument("--start", type=int, default=None,
                    help="只跑第 start 批之后（训练/测试切分用）")
    ap.add_argument("--end", type=int, default=None,
                    help="只跑到第 end 批为止")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="关闭阈值自标定（用手工 --psi-low/--psi-mid）")
    ap.add_argument("--calib-batches", type=int, default=2,
                    help="用前几批观察 PSI 来标定阈值")
    ap.add_argument("--ratio-th", type=float, default=None,
                    help="超标比例阈值（默认 0.85）")
    ap.add_argument("--llm", default="none",
                    choices=["none", "mock", "mock_bad", "pytrio", "qwen4b", "ollama"],
                    help="决策层后端：mock 验证链路，pytrio/qwen4b/ollama 接真模型")
    args = ap.parse_args()

    # 只打印"读了哪些 key"，不打印任何 value（.env 里可能有密钥）
    if not _ENV.get("missing"):
        print(f"配置    {env_summary(_ENV)}")

    cfg = Config()
    # 阈值必须按数据集标定：合成数据 PSI 0.1~0.3，真实气体传感器 1~4，
    # 用同一套绝对值必然失效（要么从不动作，要么永远走最贵那档）。
    if args.psi_low is not None:
        cfg.psi_low = args.psi_low
    if args.psi_mid is not None:
        cfg.psi_mid = args.psi_mid
    if args.ratio_th is not None:
        cfg.psi_ratio_th = args.ratio_th
    cfg.detector = args.detector
    cfg.gate_window = args.gate_window
    cfg.gate_bootstrap = args.gate_bootstrap
    cfg.enable_output_calibration = bool(args.enable_output_calibration)

    adapter = ds.build(args.dataset)
    # 训练/测试切分：只跑其中一段（迁移实验用）
    if args.start is not None or args.end is not None:
        adapter = ds.SubsetAdapter(adapter, args.start or 1, args.end)

    # ── 数据身份闸门 ──────────────────────────────────────────────
    # official_public → 只能做格式体检，不许进闭环
    # official_test   → 必须已冻结配置 + 一次性凭据，且只允许正式评估
    data_identity = getattr(adapter, "data_identity", "unknown")
    _guard_identity(adapter, args, cfg)

    # 【A4】按数据集声明的任务类型分派指标，避免把 RMSE 当 F1 算
    task_type = args.task_type or getattr(adapter, "task_type", cfg.task_type)
    cfg.task_type = task_type
    metric = mt.build_for(task_type, cfg.metric_name or None)

    # 【A4】模型也要按任务类型选：回归任务用分类器拟合连续标签，结果是错的
    model_kind = args.model or md.default_for(task_type)
    task_warn = md.check_task(model_kind, task_type)
    # 动作清单（这个模型能做什么、各多少钱）。
    # 用 try 兜底：测试里会用假模型模块模拟故障，它未必提供 action_catalog。
    try:
        catalog = md.action_catalog(model_kind)
    except Exception:                              # noqa: BLE001
        catalog = None

    if task_warn:
        print(f"[!] {task_warn}")

    # 【R1】margin 归一化：回归指标（MAE/RMSE）值域是"天"，绝对阈值 0.02 无意义，
    # 自动切到 relative（相对 baseline 的比例改进）。分类保持 absolute 以不改既有结果。
    margin_mode = args.margin_mode or ("relative" if task_type == "regression"
                                       else cfg.margin_mode)
    cfg.margin_mode = margin_mode
    os.makedirs(args.out, exist_ok=True)

    print(f"数据集：{adapter.name}  批次：{adapter.n_batches}  模型：{model_kind}")
    print(f"动作清单：{catalog}")
    print(f"任务类型：{task_type}  指标：{metric.name}"
          f"（{getattr(metric, 'direction', 'higher')}）  检测器：{cfg.detector}")
    print(f"门控：margin={cfg.gate_margin}  模式={margin_mode}"
          f"  窗口={cfg.gate_window}  Bootstrap={cfg.gate_bootstrap}")

    recorder = None
    if not args.no_recorder:
        # 【SwanLab】配置项写入 init(config=...)，在网页端"配置"面板可见。
        # 注意：只放可公开的实验配置，绝不放 API Key / 标签 / 路径。
        rec_cfg = {
            "dataset": adapter.name,
            "task_type": task_type,
            "metric": metric.name,
            "detector": cfg.detector,
            "margin_mode": margin_mode,
            "psi_low": cfg.psi_low,
            "psi_mid": cfg.psi_mid,
            "gate_margin": cfg.gate_margin,
            "gate_window": cfg.gate_window,
            "gate_bootstrap": cfg.gate_bootstrap,
        }
        recorder = Recorder(
            os.path.join(args.out, "run.jsonl"),
            # None = 由 SWANLAB_ENABLED 决定；--swanlab 可强制开启
            use_swanlab=True if args.swanlab else None,
            project=None,                 # SWANLAB_PROJECT 环境变量优先
            experiment=args.experiment,   # SWANLAB_EXPERIMENT 环境变量优先
            config=rec_cfg,
        )
        st = recorder.status()
        if st["swanlab_enabled"]:
            print(f"实验记录：SwanLab {st['swanlab_cloud']}")
            print(f"           project={st['swanlab_project']} "
                  f"experiment={st['swanlab_experiment']} mode={st['swanlab_mode']}")
            if st["swanlab_error"]:
                print(f"           [!] 云端不可用，已降级为本地记录：{st['swanlab_error']}")
        else:
            print(f"实验记录：本地 {recorder.path}（SwanLab 未启用）")

    rows: list[dict] = []
    errors: list[dict] = []

    # ── 经验表（策略层自改进）──
    if args.no_memory:
        memory = None
    elif args.load_experience:
        memory = ExperienceTable.load_json(args.load_experience)
        print(f"经验表  {args.load_experience}（{len(memory.records)} 条记录）")
    else:
        memory = ExperienceTable()

    # ── 阈值自标定：不手工给 --psi-low 时，用前几批的 PSI 自己定 ──
    calib = None
    if not args.no_calibrate and args.psi_low is None:
        calib = ThresholdCalibrator(warmup=args.calib_batches)

    # ── 冷启动：取第 1 批、训练初始模型 ──
    # 【A1】这一段原来在 try 之外：第 1 批取不到/训练失败会直接崩、连 metrics.json
    # 都不会生成。现在失败也要留下可查的记录，并以"已产出结果"的方式退出。
    try:
        # 评测协议：一问一答。没交卷就拿不到下一批；偷看未来会被直接拦下。
        proto = LocalEvaluator(adapter, strict=True)
        first = proto.next_batch()
        base = md.build(model_kind, random_state=cfg.random_state).fit(first.X, first.y)

        baseline = base.clone()            # 冻结，永不更新
        rel = rl.ReleaseManager(base.clone(), 0.0, cfg.gate_margin,
                               max_snapshots=cfg.max_snapshots,
                               gate_window=cfg.gate_window,
                               n_boot=cfg.gate_bootstrap,
                               seed=cfg.random_state,
                               margin_mode=margin_mode,
                               metric_name=metric.name,
                               metric_direction=getattr(metric, "direction", "higher"),
                               eps=cfg.margin_relative_floor)

        ref_score = metric.score(first.y, rel.current.predict(first.X))
        rel.current_score = ref_score

        agent = la.build(args.llm)
        if agent.enabled:
            print(f"决策层：{args.llm}（{getattr(agent, 'model', '-')}）")
        else:
            print("决策层：规则引擎（未接大模型）")

        monitor = Monitor(first.X, cfg.psi_bins, detector=cfg.detector)
        ledger = Ledger(os.path.join(args.out, "decisions.jsonl"),
                        cfg.budget_gpu_sec, cfg.budget_labels, cfg.budget_tokens)

        # 第 1 轮：冷启动训练完，也要交一份判断
        proto.submit(Submission(
            round_no=1, batch_id=first.batch_id, decision="deploy",
            model_version=rel.version,
            reason="冷启动：以首批数据训练初始模型并部署",
            predictions=rel.current.predict(first.X),
            self_reported={"f1_system": round(ref_score, 4)},
        ))
    except Exception as exc:                       # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        errors.append({"batch": None, "stage": "cold_start", "error": msg})
        print(f"[!] 冷启动失败，无法继续：{msg}")
        # 最常见的失败是运行时依赖没装，直接给可执行的修复命令，别让人猜
        if isinstance(exc, ModuleNotFoundError):
            print("[!] 缺少运行时依赖，请执行：")
            print("        pip install -r requirements.lock")
            print("    （只装 requirements-dev.txt 是不够的：那只含 swanlab/pytest）")
            print(f"    当前 Python：{sys.executable}")
        fatal = {"dataset": args.dataset, "model": model_kind,
                 "task_type": task_type, "metric": metric.name,
                 "fatal": True, "rows": [], "errors": errors,
                 "protocol": {"rounds": 0, "violations": 0},
                 "ledger": {"used": {"gpu_sec": 0, "labels": 0, "tokens": 0}}}
        with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(fatal, f, ensure_ascii=False, indent=2)
        if recorder is not None:
            recorder.finish({"fatal": True, "error": msg})
        return

    best = ref_score
    last_update = 0
    last_action, last_gain = None, 0.0

    # 【A1】整个循环包一层：连 next_batch 都可能抛异常（协议/数据问题）
    try:
        while True:
            batch = proto.next_batch()
            if batch is None:
                break
            b = batch.batch_id
            t0 = time.time()
            submitted = False

            try:
                # 对照组：静态基线
                f1_base = metric.score(batch.y, baseline.predict(batch.X))

                # ① 监测
                rep = monitor.check(batch.X, cfg.psi_mid)   # 超标比例按“重型阈值”算

                # 阈值自标定：预热期只观察，采够了就用数据自己定阈值
                if calib is not None and not calib.ready:
                    calib.observe(rep["psi_mean"])
                    if calib.ready:
                        cfg.psi_low, cfg.psi_mid = calib.calibrate()
                        print(f"  阈值标定：{calib.explain()}")
                lshift = label_drift(first.y, batch.y, task_type)

                # ② 归因
                diag = dg.diagnose(rep, lshift, cfg)

                # 用实验证据反过来修正归因：便宜手段试过没用，说明不是单纯尺度问题
                if last_action == "refresh_normalization" and last_gain < cfg.min_gain:
                    diag = {**diag, "kind": "concept_inferred",
                            "reason": f"标准化校准仅带来 {last_gain:+.3f} 增益，反推不是单纯平移"}

                # ③ 决策
                state = dc.State(
                    batch_id=b, psi_mean=rep["psi_mean"],
                    psi_ratio=rep["psi_ratio"], psi_top=rep["psi_max"],
                    current_score=rel.current_score, best_score=best,
                    steps_since_update=b - last_update,
                    last_action=last_action, last_gain=last_gain,
                    warmup=b <= cfg.warmup_batches,
                    history=[{"psi_mean": m["psi_mean"]} for m in monitor.history[:-1]],
                )
                history_read = state.read_history()
                dec = policy.decide(state, diag, cfg, agent, memory, catalog)

                # ④ 执行 + 影子评估
                cand, shadow = None, None
                if dec["action"] == "calibrate_output":
                    # 【B3】只用 b-1 及更早的**已标注**批次拟合校准参数，
                    # 绝不使用当前批标签（否则就是拿验收数据调参）。
                    Xw, yw, _ww = recent_window(proto, max(1, b - 4), b - 1)
                    try:
                        cand = rel.current.clone().calibrate_output(Xw, yw)
                    except Exception as exc:          # noqa: BLE001
                        # 不支持 / 无 predict_proba / 回归任务 → 明确降级并留痕
                        cand = None
                        msg = f"输出校准不可用，本批降级为 do_nothing：{exc}"
                        errors.append({"batch": b, "stage": "calibration",
                                       "error": str(exc)})
                        print(f"  b{b:02d} | [!] {msg}")
                        dec = {**dec, "action": "do_nothing",
                               "reason": msg, "note": msg, "cost_sec": 0.0}
                elif dec["action"] == "refresh_normalization":
                    cand = rel.current.clone().refresh_normalization(batch.X)
                elif dec["action"] == "refit_model":
                    # 只能用 b-1 及更早的数据训练（标签滞后），再到第 b 批上检验。
                    # 用第 b 批的标签训练、又在第 b 批上验收 = 数据泄漏，分数会虚高。
                    Xw, yw, ww = recent_window(proto, max(1, b - 4), b - 1)
                    # 续训：在当前权重上接着训，并刷新标准化（实测最优：
                    #   冷启动 128ms/0.785 → 续训+刷尺子 75ms/0.805）
                    # 只续训不刷尺子反而更差（254ms/0.795），所以必须 refresh_scaler=True
                    try:
                        cand = rel.current.clone().fit_continue(
                            Xw, yw, ww, refresh_scaler=True)
                    except Exception:                 # noqa: BLE001
                        cand = md.build(model_kind, random_state=cfg.random_state
                                        ).fit(Xw, yw, ww)

                if cand is not None:
                    shadow = ev.shadow(rel.current, cand, batch, metric)
                    rel_rec = rel.try_release(cand, shadow["after"], shadow["before"])
                    gain = shadow["gain"]
                else:
                    rel_rec = {"decision": "noop", "reason": dec["reason"]}
                    gain = 0.0

                # ⑥ 经验表记一笔：哪类漂移、用了哪招、赚了还是亏了
                if memory is not None:
                    memory.record(diag["kind"], dec["action"], gain,
                                  dec["cost_sec"], rel_rec["decision"])

                # ⑤ 本批最终分数
                f1_sys = metric.score(batch.y, rel.current.predict(batch.X))
                best = max(best, f1_sys)

                # 逐批计费：先算出本批实际记账多少，再 charge + 留痕。
                # 【为什么】标定脚本要按"批次区间"拆成本，只靠账本累计值拆不出来。
                chg_gpu = 0.0
                chg_labels = 0
                chg_tokens = 0
                if dec["action"] != "do_nothing":
                    chg_gpu = float(dec["cost_sec"])
                    chg_labels = len(batch.y) if dec["action"] == "refit_model" else 0
                if agent.enabled and isinstance(dec.get("llm"), dict):
                    chg_tokens = int(dec["llm"].get("tokens", 0) or 0)
                if chg_gpu or chg_labels or chg_tokens:
                    ledger.charge(gpu_sec=chg_gpu, labels=chg_labels, tokens=chg_tokens)

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
                    # 【R1】门控证据：量纲、方向、原始值、归一化增益全部留痕，
                    # 事后可以复盘"为什么这次没上线"，而不是只看一句话。
                    "gate": {k: rel_rec.get(k) for k in (
                        "metric_name", "metric_direction", "baseline_score",
                        "candidate_score", "raw_gain", "normalized_gain",
                        "margin", "margin_mode", "margin_mode_used",
                        "release_decision")},
                    # 本批实际计费（供按区间拆成本，例：标定/验证分段统计）
                    "charge": {"gpu_sec": round(chg_gpu, 4),
                               "labels": int(chg_labels),
                               "tokens": int(chg_tokens)},
                    "elapsed_sec": round(time.time() - t0, 3),
                })

                row = {
                    "batch": b, "f1_baseline": round(f1_base, 4),
                    "f1_system": round(f1_sys, 4),
                    "psi_max": round(rep["psi_max"], 4), "psi_mean": round(rep["psi_mean"], 4),
                    "label_shift": round(lshift, 4), "diagnosis": diag["kind"],
                    "action": dec["action"], "decision": rel_rec["decision"],
                }
                for key in ("ks_mean", "wasserstein_mean"):
                    if key in rep:
                        row[key] = round(rep[key], 4)
                rows.append(row)

                if recorder is not None:
                    # 【SwanLab】逐批上传字段（全部在 recorder.UPLOAD_KEYS 白名单内）。
                    # 不含任何标签数组、路径或密钥。
                    recorder.log({
                        "stage": "batch", **row,
                        "batch_id": b,          # 仅用于 SwanLab 横轴 step，不当曲线
                        "dataset": adapter.name, "task_type": task_type,
                        "metric": metric.name, "detector": cfg.detector,
                        "margin_mode": margin_mode,
                        # 注：不再单独传 drift_score —— 它与 psi_mean 完全重复
                        "action": dec["action"],
                        "accepted_action": (dec["action"]
                                            if rel_rec["decision"] == "accept" else None),
                        "baseline_score": rel_rec.get("baseline_score"),
                        "candidate_score": rel_rec.get("candidate_score"),
                        "raw_gain": rel_rec.get("raw_gain"),
                        "normalized_gain": rel_rec.get("normalized_gain"),
                        "release_decision": rel_rec.get("release_decision"),
                        "latency": round(time.time() - t0, 3),
                        "token_usage": (int(dec["llm"].get("tokens", 0) or 0)
                                        if isinstance(dec.get("llm"), dict) else 0),
                        "cost_sec": round(dec["cost_sec"], 3),
                    })

                flag = {"accept": "[OK] 上线", "rollback": "[X] 回滚",
                        "noop": "-"}.get(rel_rec["decision"], rel_rec["decision"])
                src = {"rule": "规则", "rule+llm": "规则+LLM",
                       "llm": "LLM"}.get(dec.get("source", "rule"), "规则")
                print(f"  b{b:02d} | 基线 {f1_base:.3f} | 系统 {f1_sys:.3f} | "
                      f"PSI {rep['psi_mean']:.3f} | {diag['kind']:9s} | "
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
                submitted = True

                last_action, last_gain = dec["action"], gain
                if rel_rec["decision"] == "accept":
                    last_update = b
                    # 模型真正换代后，把漂移参照系挪到“当前已知良好状态”，
                    # 避免参照系永远钉在部署那天、导致反复做无用更新。
                    if dec["action"] == "refit_model":
                        monitor.reset_reference(batch.X)

            except BudgetExceeded as exc:
                # 【A2】预算超限不是崩溃，是"这一档动作不允许" → 降级为不动
                msg = f"预算超限，本批降级为 do_nothing：{exc}"
                errors.append({"batch": b, "stage": "budget", "error": str(exc)})
                print(f"  b{b:02d} | [!] {msg}")
                try:
                    ledger.log({"cycle": len(rows) + 1, "batch": b,
                                "history_read": "—", "hypothesis": "预算已超限",
                                "experiment": {"action": "do_nothing", "cost_sec": 0.0,
                                               "decider": "budget_guard"},
                                "decision": "noop", "reason": msg, "error": str(exc),
                                "result": {"f1_baseline": None, "f1_before": None,
                                           "f1_after": None}})
                except Exception:                  # noqa: BLE001
                    pass

            except Exception as exc:
                # 【A1】任何意外都要留下证据，并保证本批照样交卷
                msg = f"{type(exc).__name__}: {exc}"
                errors.append({"batch": b, "stage": "loop", "error": msg})
                print(f"  b{b:02d} | [!] 处理异常，已降级：{msg}")
                try:
                    ledger.log({"cycle": len(rows) + 1, "batch": b,
                                "history_read": "—", "hypothesis": "处理异常",
                                "experiment": {"action": "do_nothing", "cost_sec": 0.0,
                                               "decider": "error_guard"},
                                "decision": "noop", "reason": msg, "error": msg,
                                "result": {"f1_baseline": None, "f1_before": None,
                                           "f1_after": None}})
                except Exception:                  # noqa: BLE001
                    pass

            finally:
                if not submitted:
                    score = submit_fallback(
                        proto, rel, batch, len(rows) + 1, metric,
                        reason="保底：本批处理异常，维持现役模型")
                    if not any(r.get("batch") == b for r in rows):
                        rows.append({
                            "batch": b, "f1_baseline": None, "f1_system": score,
                            "psi_max": None, "psi_mean": None, "label_shift": None,
                            "diagnosis": "error", "action": "do_nothing",
                            "decision": "keep", "failed": True,
                        })
                    if recorder is not None:
                        recorder.log({"stage": "batch", "batch": b, "failed": True,
                                      "f1_system": score})

    except Exception as exc:                       # noqa: BLE001
        # 循环本身挂了（连 next_batch 都失败）：记录后继续走收尾流程
        msg = f"{type(exc).__name__}: {exc}"
        errors.append({"batch": None, "stage": "driver", "error": msg})
        print(f"[!] 主循环中断：{msg}")

    # ── 收尾：无论前面发生什么，都要写出结果 ──
    summary = {"dataset": adapter.name, "model": model_kind,
               "task_type": task_type, "metric": metric.name,
               "rows": rows, "ledger": ledger.summary(),
               "protocol": proto.summary(),
               "errors": errors,
               "config": {"detector": cfg.detector, "psi_low": cfg.psi_low,
                          "psi_mid": cfg.psi_mid, "gate_margin": cfg.gate_margin,
                          "gate_window": cfg.gate_window,
                          "gate_bootstrap": cfg.gate_bootstrap},
               "release": {"version": rel.version,
                           "snapshots": rel.snapshot_versions()}}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    concept_start = None
    for b in range(1, adapter.n_batches + 1):
        try:
            if float(adapter.get(b).meta.get("concept_flag", 0.0)) > 0:
                concept_start = b
                break
        except Exception:                          # noqa: BLE001
            break
    try:
        plot(rows, os.path.join(args.out, "curve.png"), concept_start)
    except Exception as exc:                       # noqa: BLE001
        errors.append({"batch": None, "stage": "plot", "error": str(exc)})
        print(f"[!] 绘图失败（不影响结果）：{exc}")

    valid = [r for r in rows if r.get("f1_system") is not None]
    n_acc = sum(1 for r in rows if r["decision"] == "accept")
    n_roll = sum(1 for r in rows if r["decision"] == "rollback")
    n_act = sum(1 for r in rows if r["action"] != "do_nothing")
    avg_base = float(np.mean([r["f1_baseline"] for r in valid
                              if r.get("f1_baseline") is not None])) if valid else float("nan")
    avg_sys = float(np.mean([r["f1_system"] for r in valid])) if valid else float("nan")

    print(f"\n平均分  基线 {avg_base:.3f}  系统 {avg_sys:.3f}")
    print(f"决策    动作 {n_act} 次 | 上线 {n_acc} 次 | 回滚 {n_roll} 次")
    print(f"账本    {ledger.summary()}")
    if agent.enabled:
        print(f"决策层  {agent.stats()}")
    ps = proto.summary()
    print(f"协议    {ps['rounds']} 轮 | 部署 {ps['deploy']} | 维持 {ps['keep']} | "
          f"违规 {ps['violations']}")
    if errors:
        print(f"异常    {len(errors)} 条（已降级，详见 metrics.json.errors）")
    if recorder is not None:
        used = ledger.used
        recorder.finish({
            "avg_baseline": avg_base, "avg_system": avg_sys,
            "n_acc": n_acc, "n_roll": n_roll, "n_act": n_act,
            "errors": len(errors),
            # 总成本（客观分的分母）
            "gpu_sec": round(float(used.get("gpu_sec", 0.0)), 4),
            "labels": int(used.get("labels", 0)),
            "tokens": int(used.get("tokens", 0)),
        })
        st = recorder.status()
        print(f"记录    本地 {st['path']}（{st['n_records']} 条）")
        if st["swanlab_enabled"]:
            print(f"        SwanLab project={st['swanlab_project']} "
                  f"experiment={st['swanlab_experiment']}")
            if st["swanlab_error"]:
                print(f"        [!] 云端上传被阻塞，仅本地可用：{st['swanlab_error']}")
            else:
                print(f"        SwanLab 状态：{st['swanlab_cloud']}")
                print("        （是否已在网页端可见尚未核实，请自行打开网页确认）")
    # ── 经验表收尾：打印 + 存盘（迁移实验可带着这份去新数据集）──
    if memory is not None:
        print()
        memory.print_summary()
        memory.to_json(os.path.join(args.out, "experience.json"))
        print(f"        经验表已存 {args.out}/experience.json（提交包务必不带这个文件）")
    if calib is not None and calib.ready:
        print(f"阈值    {calib.explain()}")

    print(f"产出    {args.out}/metrics.json  decisions.jsonl  curve.png  run.jsonl")
    if n_act < 3:
        print("[!] 有效动作不足 3 次，门槛③可能过不了——加大漂移强度或调低阈值")


if __name__ == "__main__":
    main()

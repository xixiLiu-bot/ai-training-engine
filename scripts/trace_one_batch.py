# -*- coding: utf-8 -*-
"""单批追踪：把「一批数据进来之后，程序内部到底算了什么」全部打出来。

这是个教学脚本，不是主流程。主流程看 scripts/run_pipeline.py。

    python scripts/trace_one_batch.py
    python scripts/trace_one_batch.py --dataset gas --batch 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolver.adapters import dataset as ds
from evolver.adapters import model as md
from evolver.config import Config
from evolver.loop import decide as dc
from evolver.loop import diagnose as dg
from evolver.loop.monitor import Monitor, label_psi

LINE = "=" * 74


def title(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gas", choices=["synthetic", "gas"])
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    cfg = Config()
    adapter = ds.build(args.dataset)

    # ────────────────────────────────────────────────────────
    title(f"STEP 0  第 1 批进来：训练模型 + 记住参照分布")
    ref = adapter.get(1)
    print(f"  数据形状      X = {ref.X.shape}   y = {ref.y.shape}")
    print(f"  特征数        {ref.X.shape[1]}")
    print(f"  类别          共 {len(np.unique(ref.y))} 类，分布 {np.bincount(ref.y).tolist()}")
    print(f"  第一行前 5 个特征值  {np.round(ref.X[0, :5], 2).tolist()}")
    print(f"  （注意量级差很大：{ref.X.min():.1f} ~ {ref.X.max():.1f}）")

    model = md.build("logreg").fit(ref.X, ref.y)
    monitor = Monitor(ref.X, cfg.psi_bins)
    print(f"\n  → 模型训练完成（逻辑回归，输入先做标准化）")
    print(f"  → 参照系记下：128 个特征各自的分布，各切成 {cfg.psi_bins} 个箱")

    # ────────────────────────────────────────────────────────
    title(f"STEP 1  第 {args.batch} 批进来：先看数据长什么样")
    cur = adapter.get(args.batch)
    print(f"  数据形状      X = {cur.X.shape}   y = {cur.y.shape}")
    print(f"  类别分布      {np.bincount(cur.y).tolist()}")
    print(f"  第一行前 5 个特征值  {np.round(cur.X[0, :5], 2).tolist()}")
    print(f"\n  对比参照系第 1 号特征：")
    print(f"    参照系中位数 {np.median(ref.X[:, 0]):12.1f}")
    print(f"    当前批中位数 {np.median(cur.X[:, 0]):12.1f}")

    # ────────────────────────────────────────────────────────
    title("STEP 2  算 PSI：逐个特征比较『分布搬没搬家』")
    rep = monitor.check(cur.X)
    per = np.array(rep["per_feature"])
    print(f"  128 个特征各算出一个 PSI：")
    print(f"    最小的 5 个   {np.round(np.sort(per)[:5], 3).tolist()}")
    print(f"    最大的 5 个   {np.round(np.sort(per)[-5:], 3).tolist()}")
    print(f"    中位数        {np.median(per):.3f}")
    print(f"\n  psi_mean（128 个的平均，决策就用这个）= {rep['psi_mean']:.3f}")
    print(f"  psi_max （最严重的那一个，只作参考）   = {rep['psi_max']:.3f}")

    # ────────────────────────────────────────────────────────
    title("STEP 3  算标签漂移：『哪些类出现了』变了没")
    ls = label_psi(ref.y, cur.y)
    print(f"  参照系类别分布  {np.bincount(ref.y).tolist()}")
    print(f"  当前批类别分布  {np.bincount(cur.y).tolist()}")
    print(f"  label_psi = {ls:.3f}")
    print(f"  （判断阈值 {cfg.label_psi_th}，超过算『标签也变了』）")

    # ────────────────────────────────────────────────────────
    title("STEP 4  归因：到底是『数据变了』还是『规则变了』")
    diag = dg.diagnose(rep, ls, cfg)
    print(f"  输入：psi_mean = {diag['psi_mean']:.3f}, label_psi = {diag['label_shift']:.3f}")
    print(f"  判断：psi_mean {'≥' if diag['psi_mean'] >= cfg.psi_low else '<'} {cfg.psi_low} ?  "
          f"→ {'分布有漂移' if diag['psi_mean'] >= cfg.psi_low else '分布稳定'}")
    print(f"  结论：{diag['kind']}  —— {diag['reason']}")
    print(f"  建议动作：{diag['suggest']}")

    # ────────────────────────────────────────────────────────
    title("STEP 5  组装状态：把『历史』也交给决策器")
    ref_score = f1_score(ref.y, model.predict(ref.X), average="macro")
    state = dc.State(
        batch_id=args.batch,
        psi_mean=rep["psi_mean"],
        current_score=ref_score,
        best_score=ref_score,
        steps_since_update=99,
        last_action=None,
        last_gain=0.0,
        history=[{"psi_mean": rep["psi_mean"]}],
    )
    print(f"  当前分数（模型在第 1 批上的 F1）= {state.current_score:.3f}")
    print(f"  历史最好                        = {state.best_score:.3f}")
    print(f"  距上次更新                      = {state.steps_since_update} 批")
    print(f"\n  给决策器看的『读历史』摘要：")
    print(f"    {state.read_history()}")

    # ────────────────────────────────────────────────────────
    title("STEP 6  决策：6 条规则逐条过，命中就停")
    dec = dc.decide(state, diag, cfg)
    checks = [
        ("规则1 预热期？", state.batch_id <= cfg.warmup_batches),
        ("规则2 分布稳定？(psi < {:.2f})".format(cfg.psi_low), state.psi_mean < cfg.psi_low),
        ("规则3 距上次更新太近？", state.steps_since_update < cfg.min_gap),
        ("规则4 上次校准没用？", state.last_action == "refresh_normalization"
         and state.last_gain < cfg.min_gain),
        ("规则5 漂移很猛？(psi >= {:.2f})".format(cfg.psi_mid), state.psi_mean >= cfg.psi_mid),
    ]
    for name, hit in checks:
        print(f"  {name:32s} {'命中 → 停' if hit else '不命中，往下走'}")
    print(f"\n  最终动作：{dec['action']}   （成本 {dec['cost_sec']} 秒）")
    print(f"  假设：  {dec['hypothesis']}")
    print(f"  理由：  {dec['reason']}")

    # ────────────────────────────────────────────────────────
    title("STEP 7  执行动作，得到一个『候选模型』")
    if dec["action"] == "do_nothing":
        print("  什么都不做 → 没有候选模型，流程直接结束")
        return
    if dec["action"] == "refresh_normalization":
        cand = model.clone().refresh_normalization(cur.X)
        print("  做法：复制当前模型，只重算标准化的均值方差，权重不动")
        print(f"  参照系均值（前 3 个特征） {np.round(model.scaler.mean_[:3], 2).tolist()}")
        print(f"  当前批均值（前 3 个特征） {np.round(cand.scaler.mean_[:3], 2).tolist()}")
    else:
        # 只能用「已经下发过的、且早于当前批」的数据训练。
        # 用第 N 批自己训练、又在第 N 批上验收 = 数据泄漏，分数会虚高到 1.0。
        lo = max(1, args.batch - 4)
        Xs, ys, ws = [], [], []
        for k in range(lo, args.batch):
            h = adapter.get(k)
            w = float(np.exp(-(args.batch - 1 - k) / 0.5))   # 越新权重越高
            Xs.append(h.X)
            ys.append(h.y)
            ws.append(np.full(len(h.y), w))
        Xw, yw, ww = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
        print(f"  做法：用第 {lo}~{args.batch - 1} 批重新训练（越新权重越高）")
        print(f"        不能用第 {args.batch} 批自己训练 —— 那是数据泄漏，分数会虚高到 1.0")
        print(f"        训练样本共 {len(yw)} 条")
        cand = md.build("logreg").fit(Xw, yw, ww)
    print("  → 候选模型生成完毕")

    # ────────────────────────────────────────────────────────
    title("STEP 8  影子评估：候选模型和当前模型比一比")
    before = f1_score(cur.y, model.predict(cur.X), average="macro")
    after = f1_score(cur.y, cand.predict(cur.X), average="macro")
    print(f"  同一批数据上打分：")
    print(f"    当前模型  F1 = {before:.4f}")
    print(f"    候选模型  F1 = {after:.4f}")
    print(f"    增益          {after - before:+.4f}")
    print(f"\n  静态基线（永不更新）在同一批上 F1 = {f1_score(cur.y, model.predict(cur.X), average='macro'):.4f}")

    # ────────────────────────────────────────────────────────
    title("STEP 9  门控：够不够格上线？")
    gain = after - before
    ok = gain >= cfg.gate_margin
    print(f"  增益 {gain:+.4f}  {'≥' if ok else '<'} 门控 {cfg.gate_margin}")
    if ok:
        print(f"  → [OK] 上线。当前模型换成候选模型，版本号 +1")
        print(f"  → 同时把参照系挪到当前批（否则以后每批都显得『漂移严重』）")
    else:
        print(f"  → [X] 回滚。什么都不换，继续用旧模型")
        print(f"  （这一步是『不低于静态基线』的保命绳）")

    # ────────────────────────────────────────────────────────
    title("STEP 10  交卷，然后才能拿下一批")
    print(f'  submit({{')
    print(f'      "round":       {args.batch - 1},')
    print(f'      "batch":       {args.batch},')
    print(f'      "decision":    "{"deploy" if ok else "keep"}",')
    print(f'      "model_version": {"+1" if ok else "不变"},')
    print(f'      "reason":      "{dec["action"]} -> {"accept" if ok else "rollback"}"')
    print(f'  }})')
    print(f"\n  交卷成功 → 评测器才会下发第 {args.batch + 1} 批")


if __name__ == "__main__":
    main()

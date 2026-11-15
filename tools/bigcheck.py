# -*- coding: utf-8 -*-
"""全量体检 v2：四数据集 × 两决策层，修正增益方向与决策来源统计。"""
import json
import os
import shutil
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_pipeline.py")

CASES = [
    ("合成漂移流", "分类", "synthetic", []),
    ("气体传感器", "分类", "gas", []),
    ("轴承退化", "分类", "bearing", ["--scaler", "auto"]),
    ("C-MAPSS", "回归", "rul", []),
]
LAYERS = [("纯规则", ["--llm", "none"]), ("真27B", ["--llm", "pytrio"])]


def run(dataset, extra):
    out = os.path.join(ROOT, "out_bigchk")
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([PY, "-X", "utf8", SCRIPT, "--dataset", dataset, *extra,
                    "--out", out], capture_output=True, text=True,
                   encoding="utf-8", errors="ignore", cwd=ROOT)
    mp = os.path.join(out, "metrics.json")
    if not os.path.exists(mp):
        return None
    with open(mp, encoding="utf-8") as f:
        d = json.load(f)
    rows = d["rows"]
    if not rows:
        return None
    used = d["ledger"]["used"]
    base = sum(r["f1_baseline"] for r in rows) / len(rows)
    sysf = sum(r["f1_system"] for r in rows) / len(rows)
    lower_better = sysf < 0            # 回归内部取了负号
    if lower_better:
        base, sysf = -base, -sysf
    gain = (base - sysf) if lower_better else (sysf - base)

    # 决策来源要看 decisions.jsonl（metrics.json 的 rows 里没有这层）
    srcs = {}
    dp = os.path.join(out, "decisions.jsonl")
    if os.path.exists(dp):
        for line in open(dp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            e = r.get("experiment")
            if isinstance(e, dict) and "decider" in e:
                k = e["decider"]
                srcs[k] = srcs.get(k, 0) + 1

    res = {
        "model": d.get("model"), "metric": d.get("metric"),
        "lower_better": lower_better, "base": base, "system": sysf,
        "gain": gain, "gpu": float(used.get("gpu_sec", 0.0)),
        "tokens": int(used.get("tokens", 0)),
        "n_act": sum(1 for r in rows if r.get("action") != "do_nothing"),
        "n_acc": sum(1 for r in rows if r.get("decision") == "accept"),
        "n_roll": sum(1 for r in rows if r.get("decision") == "rollback"),
        "rounds": d["protocol"]["rounds"],
        "violations": d["protocol"]["violations"],
        "deciders": srcs,
    }
    shutil.rmtree(out, ignore_errors=True)
    return res


def main():
    results = []
    print("跑批中（%d 次真实运行）……\n" % (len(CASES) * len(LAYERS)))
    for task, ttype, ds, extra in CASES:
        for layer, largs in LAYERS:
            r = run(ds, extra + largs)
            if r is None:
                print(f"  {task} / {layer} 失败")
                continue
            r.update(task=task, ttype=ttype, layer=layer)
            results.append(r)
            print(f"  {task:<12}{layer:<8}系统 {r['system']:>8.3f}  "
                  f"算力 {r['gpu']:>5.2f}s  tokens {r['tokens']:>5}  "
                  f"动作 {r['n_act']:>2}  {r['deciders']}")

    print("\n" + "=" * 94)
    h = (f"{'数据集':<12}{'任务':<6}{'决策层':<8}{'指标':<11}"
         f"{'基线':>8}{'系统':>9}{'增益':>8}{'算力':>7}{'tokens':>7}{'性价比':>9}")
    print(h)
    print("-" * 94)
    for r in results:
        arrow = "↓" if r["lower_better"] else "↑"
        eff = r["gain"] / max(r["gpu"], 1e-9)
        print(f"{r['task']:<12}{r['ttype']:<6}{r['layer']:<8}"
              f"{r['metric']+arrow:<11}{r['base']:>8.3f}{r['system']:>9.3f}"
              f"{r['gain']:>+8.3f}{r['gpu']:>7.2f}{r['tokens']:>7}{eff:>9.4f}")

    print("\n" + "=" * 94)
    print("接大模型的边际效果（同数据集对比）")
    print(f"  {'数据集':<12}{'分数变化':>10}{'算力变化':>11}{'新增tokens':>11}"
          f"{'结论':>10}")
    for task, ttype, ds, extra in CASES:
        a = next((x for x in results if x["task"] == task and x["layer"] == "纯规则"), None)
        b = next((x for x in results if x["task"] == task and x["layer"] == "真27B"), None)
        if not (a and b):
            continue
        ds_ = b["system"] - a["system"]
        if b["lower_better"]:
            ds_ = -ds_          # MAE 降低才是好处
        verdict = "有帮助" if ds_ > 0.01 else ("没差别" if ds_ > -0.01 else "**更差**")
        print(f"  {task:<12}{ds_:>+10.3f}{b['gpu']-a['gpu']:>+10.2f}s"
              f"{b['tokens']:>11}{verdict:>10}")

    out = os.path.join(ROOT, "out", "bigcheck.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n明细 → {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()

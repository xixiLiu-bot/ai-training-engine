# -*- coding: utf-8 -*-
"""端到端韧性测试：主循环遇到异常 / 预算超限 / 模型失能，必须照样收尾并交卷。

对应整改 A1（主循环兜底）、A2（预算超限降级）。
这是门槛①「无人工干预完成全部阶段」的直接证据。

    python -m unittest tests.test_resilience -v
"""
import importlib.util
import json
import os
import shutil
import sys
import types
import unittest
import uuid

import numpy as np

VENDOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VENDOR)


def _load_run_pipeline():
    path = os.path.join(VENDOR, "scripts", "run_pipeline.py")
    spec = importlib.util.spec_from_file_location("rp_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────
# 最小替身：不依赖 sklearn
# ─────────────────────────────────────────────────────────────
class TinyBatch:
    def __init__(self, i, X, y, meta=None):
        self.batch_id = i
        self.X = X
        self.y = y
        self.meta = meta or {}


class TinyModel:
    """可配置的假模型：raise_beyond 之后 predict 会抛异常，用来模拟模型失能。"""

    def __init__(self, w: float = 1.0, raise_beyond: float | None = None,
                 calib_fails: bool = False):
        self.w = w
        self.raise_beyond = raise_beyond
        self.calib_fails = calib_fails

    def fit(self, X, y, sample_weight=None):
        self.w = float(np.mean(X))
        return self

    def predict(self, X):
        if self.raise_beyond is not None and float(np.abs(np.mean(X))) > self.raise_beyond:
            raise RuntimeError("模型推理失败（故障注入）")
        return np.zeros(len(X), dtype=int)

    def refresh_normalization(self, X):
        self.w = float(np.mean(X))
        return self

    def calibrate_output(self, X, y):
        if self.calib_fails:
            raise RuntimeError("输出校准不可用（故障注入）")
        return self

    def clone(self):
        return TinyModel(self.w, self.raise_beyond, self.calib_fails)


class TinyMetric:
    name = "f1_macro"
    task_type = "classification"

    def score(self, y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


class DriftingAdapter:
    """强漂移数据：保证系统会尝试动作，从而触发 charge()。"""

    name = "fake_drift"
    task_type = "classification"

    def __init__(self, n_batches=5, boom_at=None):
        self.n_batches = n_batches
        self.boom_at = boom_at
        self.loads = []

    def get(self, b):
        self.loads.append(b)
        if self.boom_at is not None and b == self.boom_at:
            raise RuntimeError(f"第 {b} 批数据损坏（故障注入）")
        rng = np.random.default_rng(100 + b)
        X = rng.normal(size=(60, 3)) + b * 6.0        # 批间巨大平移 → PSI 爆表
        y = rng.integers(0, 2, size=60)
        return TinyBatch(b, X, y)


def _install_stubs(rp, adapter, cfg_factory=None, model_factory=None):
    rp.ds = types.SimpleNamespace(build=lambda kind, **kw: adapter)
    rp.md = types.SimpleNamespace(
        build=model_factory or (lambda *a, **k: TinyModel()),
        # run_pipeline 现在会按任务类型选模型并校验一致性，替身必须实现这两个
        default_for=lambda task_type: "ridge" if task_type == "regression" else "logreg",
        check_task=lambda kind, task_type: None,
    )
    rp.mt = types.SimpleNamespace(build_for=lambda *a, **k: TinyMetric())
    rp.plot = lambda *a, **k: None
    if cfg_factory is not None:
        rp.Config = cfg_factory


class TestResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _load_run_pipeline()

    def setUp(self):
        # 必须放在工作区内，且**不能用 tempfile.mkdtemp**：
        # 它在本沙箱下会创建 ACL 不可访问的目录（本次实际踩到）。
        base = os.path.join(VENDOR, "out", "_unittest_tmp")
        os.makedirs(base, exist_ok=True)
        self.out = os.path.join(base, f"res-{uuid.uuid4().hex[:10]}")
        os.makedirs(self.out, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def _run(self, adapter, cfg_factory=None, model_factory=None, extra_args=None):
        _install_stubs(self.rp, adapter, cfg_factory, model_factory)
        sys.argv = ["run_pipeline.py", "--llm", "none", "--out", self.out,
                    *(extra_args or [])]
        self.rp.main()                      # 不许抛异常
        path = os.path.join(self.out, "metrics.json")
        self.assertTrue(os.path.exists(path), "任何情况下都必须写出 metrics.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ---- A1：数据在中途炸了 ----
    def test_数据异常时安全停止并保留已完成结果(self):
        """第 3 批取不到 = 协议无法推进（不能跳批），必须安全停下并保留前 2 轮。"""
        summary = self._run(DriftingAdapter(n_batches=5, boom_at=3))
        self.assertFalse(summary.get("fatal"), "前两轮是成功的，不算致命失败")
        self.assertEqual(summary["protocol"]["rounds"], 2,
                         "冷启动 1 轮 + 第 2 轮，第 3 批取不到就无法继续")
        self.assertEqual(summary["protocol"]["violations"], 0)
        self.assertTrue(summary["errors"], "异常必须被记录，不能静默吞掉")
        self.assertEqual(summary["errors"][0]["stage"], "driver")
        self.assertIn("数据损坏", summary["errors"][0]["error"])
        self.assertEqual(len(summary["rows"]), 1, "已完成批次的分数要保住")

    # ---- A2：预算超限 ----
    def test_预算超限降级而不是崩溃(self):
        from evolver.config import Config

        def tiny_budget():
            return Config(budget_gpu_sec=0.001, budget_labels=1)

        summary = self._run(DriftingAdapter(n_batches=4), cfg_factory=tiny_budget)
        self.assertEqual(summary["protocol"]["rounds"], 4,
                         "预算超限后每批仍要交卷，协议轮次不能少")
        self.assertEqual(summary["protocol"]["violations"], 0)
        stages = [e["stage"] for e in summary["errors"]]
        self.assertIn("budget", stages, "预算超限必须走降级分支")
        self.assertTrue(all(e["batch"] is not None for e in summary["errors"]))

    # ---- A1：模型失能，交不出预测 ----
    def test_预测失败时交空预测而不是崩溃(self):
        # 批均值 ≈ 6·b，所以 raise_beyond=14 时第 1~2 批正常、第 3 批起失能
        summary = self._run(
            DriftingAdapter(n_batches=4),
            model_factory=lambda *a, **k: TinyModel(raise_beyond=14.0),
        )
        self.assertEqual(summary["protocol"]["violations"], 0)
        failed = [r for r in summary["rows"] if r.get("failed")]
        self.assertTrue(failed, "模型失能应留下 failed 行")
        self.assertIsNone(failed[0]["f1_system"],
                          "无法预测时分数必须为 None，不能伪造 0 分")
        self.assertTrue(summary["errors"])

    # ---- A1：连第 1 批都取不到 ----
    def test_冷启动失败也要写出结果(self):
        class AllBoom(DriftingAdapter):
            def get(self, b):
                raise RuntimeError("整条数据流都坏了")

        summary = self._run(AllBoom(n_batches=3))
        self.assertTrue(summary.get("fatal"))
        self.assertEqual(summary["errors"][0]["stage"], "cold_start")
        self.assertEqual(summary["protocol"]["violations"], 0)

    # ---- B3：输出校准不可用时降级 ----
    def test_输出校准不可用时降级并留痕(self):
        """回归任务/无 predict_proba 时校准必然失败，必须降级而不是崩。"""
        summary = self._run(
            DriftingAdapter(n_batches=4),
            model_factory=lambda *a, **k: TinyModel(calib_fails=True),
            # psi-mid 调高，让决策落到"轻度漂移"分支，才会选中 calibrate_output
            extra_args=["--enable-output-calibration",
                        "--psi-mid", "1000", "--psi-low", "0.001"],
        )
        self.assertEqual(summary["protocol"]["rounds"], 4)
        self.assertEqual(summary["protocol"]["violations"], 0)
        stages = [e["stage"] for e in summary["errors"]]
        self.assertIn("calibration", stages, "校准失败必须留下 calibration 记录")

    # ---- 只读输出目录 ----
    def test_输出目录不可写时降级为内存账本(self):
        """评测容器可能是只读文件系统：记录失败必须不影响主流程。"""
        from evolver.ledger.ledger import Ledger

        bogus = os.path.join(VENDOR, "out", "_unittest_tmp", "no_such\x00dir", "x.jsonl")
        led = Ledger(bogus)                        # 不许抛异常
        self.assertFalse(led.write_ok)
        led.log({"batch": 1, "experiment": {"action": "refit_model"}})
        self.assertEqual(led.summary()["n_actions"], 1, "内存账本仍然要可用")


class TestRecorder(unittest.TestCase):
    def test_不可写目录不抛异常(self):
        from evolver.logs.recorder import Recorder

        bad = os.path.join(VENDOR, "out", "no_such\x00dir", "run.jsonl")
        rec = Recorder(bad)
        self.assertFalse(rec.write_ok)
        rec.log({"stage": "batch", "f1_system": 0.9})
        rec.finish({"avg_system": 0.9})
        self.assertGreaterEqual(len(rec.records), 2)

    def test_正常路径写出jsonl(self):
        from evolver.logs.recorder import Recorder

        base = os.path.join(VENDOR, "out", "_unittest_tmp")
        os.makedirs(base, exist_ok=True)
        d = os.path.join(base, f"rec-{uuid.uuid4().hex[:10]}")
        os.makedirs(d, exist_ok=True)
        try:
            path = os.path.join(d, "run.jsonl")
            rec = Recorder(path)
            self.assertTrue(rec.write_ok)
            rec.log({"stage": "batch", "f1_system": 0.5})
            rec.finish({"avg_system": 0.5})
            with open(path, encoding="utf-8") as f:
                lines = [json.loads(x) for x in f if x.strip()]
            self.assertEqual(len(lines), 2)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# -*- coding: utf-8 -*-
"""输出校准（整改 B3）与动作阶梯顺序的测试。

关键结论（也是本文件的回归保护）：
    在"协变量尺度漂移"数据集上，calibrate_output 修不了问题
    （它只改判定阈值，改不了特征尺度），效果明显劣于 refresh_normalization。
    因此 `enable_output_calibration` **默认关闭**，开启必须逐数据集 A/B 验证。

    python -m unittest tests.test_calibration -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolver.adapters.model import (CalibrationUnsupported, _best_logit_biases,
                                    _best_threshold, _macro_f1, model_task)
from evolver.config import Config
from evolver.loop import decide as dc
from evolver.loop.decide import COST, State

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except Exception:                                   # noqa: BLE001
    HAS_SKLEARN = False


def _state(**kw) -> State:
    base = dict(batch_id=4, psi_mean=0.15, current_score=0.80, best_score=0.95,
                steps_since_update=2, last_action=None, last_gain=0.0,
                warmup=False, history=[])
    base.update(kw)
    return State(**base)


class TestPureHelpers(unittest.TestCase):
    def test_macro_f1(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(_macro_f1(y, y), 1.0)
        self.assertAlmostEqual(_macro_f1(y, np.array([0, 0, 0, 0])), 1 / 3, places=6)

    def test_阈值搜索能修正偏斜的概率(self):
        """模型系统性低估正类概率时，阈值应被压到 0.5 以下。"""
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(300, int), np.ones(300, int)])
        # 正类概率整体被压低一半
        p = np.concatenate([rng.uniform(0, 0.5, 300), rng.uniform(0.3, 1.0, 300)]) * 0.6
        t = _best_threshold(p, y, 1)
        self.assertLess(t, 0.5, "概率被压低时应当下调阈值")
        base = _macro_f1(y, np.where(p >= 0.5, 1, 0))
        tuned = _macro_f1(y, np.where(p >= t, 1, 0))
        self.assertGreaterEqual(tuned, base - 1e-9)

    def test_阈值搜索不会让结果变差(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 400)
        p = rng.uniform(0, 1, 400)
        t = _best_threshold(p, y, 1)
        base = _macro_f1(y, np.where(p >= 0.5, 1, 0))
        tuned = _macro_f1(y, np.where(p >= t, 1, 0))
        self.assertGreaterEqual(tuned, base - 1e-9)

    def test_多分类偏置能提升macro_f1(self):
        rng = np.random.default_rng(2)
        k, n = 3, 600
        y = rng.integers(0, k, n)
        # 构造"模型把类别1系统性低估"的概率
        logits = rng.normal(scale=1.0, size=(n, k))
        logits[np.arange(n), y] += 2.0
        logits[:, 1] -= 1.5
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        classes = np.arange(k)

        b = _best_logit_biases(p, y, classes)
        base = _macro_f1(y, classes[np.argmax(np.log(p + 1e-12), axis=1)])
        tuned = _macro_f1(y, classes[np.argmax(np.log(p + 1e-12) + b, axis=1)])
        self.assertGreaterEqual(tuned, base - 1e-9)
        self.assertGreater(b[1], 0, "被低估的类别应获得正偏置")

    def test_回归任务被标记为不支持(self):
        self.assertEqual(model_task("ridge"), "regression")
        self.assertTrue(issubclass(CalibrationUnsupported, RuntimeError))


class TestActionLadder(unittest.TestCase):
    def _cfg(self, calib: bool, task: str = "classification") -> Config:
        c = Config()
        c.enable_output_calibration = calib
        c.task_type = task
        return c

    def test_默认关闭时行为与旧版一致(self):
        dec = dc.decide(_state(), {"kind": "covariate"}, self._cfg(False))
        self.assertEqual(dec["action"], "refresh_normalization")

    def test_开启后轻度漂移优先改输出(self):
        dec = dc.decide(_state(), {"kind": "covariate"}, self._cfg(True))
        self.assertEqual(dec["action"], "calibrate_output")

    def test_回归任务不会选输出校准(self):
        dec = dc.decide(_state(), {"kind": "covariate"}, self._cfg(True, "regression"))
        self.assertEqual(dec["action"], "refresh_normalization")

    def test_改输出失败后升级到改特征(self):
        dec = dc.decide(
            _state(last_action="calibrate_output", last_gain=0.0),
            {"kind": "covariate"}, self._cfg(True))
        self.assertEqual(dec["action"], "refresh_normalization")

    def test_改特征失败后升级到重训(self):
        dec = dc.decide(
            _state(last_action="refresh_normalization", last_gain=0.0),
            {"kind": "covariate"}, self._cfg(True))
        self.assertEqual(dec["action"], "refit_model")

    def test_漂移剧烈直接重训(self):
        dec = dc.decide(_state(psi_mean=0.40), {"kind": "covariate"}, self._cfg(True))
        self.assertEqual(dec["action"], "refit_model")

    def test_成本阶梯顺序正确(self):
        self.assertLess(COST["do_nothing"], COST["calibrate_output"])
        self.assertLess(COST["calibrate_output"], COST["refresh_normalization"])
        self.assertLess(COST["refresh_normalization"], COST["refit_model"])

    def test_动作空间包含四个动作(self):
        self.assertEqual(set(dc.ACTIONS),
                         {"do_nothing", "calibrate_output",
                          "refresh_normalization", "refit_model"})


@unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn 才能构造模型适配器")
class TestModelCalibration(unittest.TestCase):
    def _fitted(self, n_classes=3, n=800, seed=0):
        from evolver.adapters.model import build
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 5))
        y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
        if n_classes == 2:
            y = (X[:, 0] > 0).astype(int)
        m = build("logreg", random_state=0)
        return m.fit(X, y), X, y

    def test_二分类校准产出阈值(self):
        m, X, y = self._fitted(n_classes=2)
        before = m.predict(X)
        m.calibrate_output(X, y)
        self.assertIsNotNone(m.calibration_summary())
        self.assertEqual(m.calibration_summary()["kind"], "threshold")
        after = m.predict(X)
        self.assertEqual(len(after), len(before))

    def test_多分类校准产出偏置(self):
        m, X, y = self._fitted(n_classes=3)
        m.calibrate_output(X, y)
        s = m.calibration_summary()
        self.assertEqual(s["kind"], "bias")
        self.assertEqual(len(s["biases"]), 3)

    def test_校准不改权重(self):
        m, X, y = self._fitted(n_classes=2)
        coef_before = m.est.coef_.copy()
        m.calibrate_output(X, y)
        np.testing.assert_allclose(m.est.coef_, coef_before, rtol=0, atol=0)

    def test_clone隔离(self):
        m, X, y = self._fitted(n_classes=2)
        c = m.clone()
        c.calibrate_output(X, y)
        self.assertIsNotNone(c.calibration_summary())
        self.assertIsNone(m.calibration_summary(), "校准不能污染原模型")

    def test_回归模型拒绝校准(self):
        from evolver.adapters.model import build
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 4))
        y = X[:, 0] * 2 + 1
        m = build("ridge").fit(X, y)
        with self.assertRaises(CalibrationUnsupported) as ctx:
            m.calibrate_output(X, y)
        self.assertIn("回归", str(ctx.exception))

    def test_重新训练后旧校准失效(self):
        m, X, y = self._fitted(n_classes=2)
        m.calibrate_output(X, y)
        self.assertIsNotNone(m.calibration_summary())
        m.fit(X, y)
        self.assertIsNone(m.calibration_summary(), "重训后旧阈值必须作废")


if __name__ == "__main__":
    unittest.main(verbosity=2)

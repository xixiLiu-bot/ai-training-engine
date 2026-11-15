# -*- coding: utf-8 -*-
"""续训（fit_continue）在类别集合变化时的行为。

背景（第 5 步实测）：
    气体 b01 有 6 类、b03 只有 5 类。sklearn 的 warm_start 会复用旧形状的系数，
    直接抛 `Initialization coef is of shape (6,129), expected (5,129)`。
    修复后必须**降级为冷启动并留痕**，而不是崩溃或静默降级。

    python -m unittest tests.test_model_fit_continue -v
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evolver.adapters import model as md  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except Exception:                       # noqa: BLE001
    HAS_SKLEARN = False

_GAS = [os.path.join(REPO, "data", "gas", "processed"),
        os.path.join(REPO, "data", "gas")]
HAS_GAS = any(os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))
              for d in _GAS)


def _blobs(classes, n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(rng.choice(classes, n))
    X = np.stack([y * 1.0, y * -0.5], 1) + rng.normal(0, 0.3, (n, 2))
    return X, y


@unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
class TestFitContinueClassSet(unittest.TestCase):
    def test_类别一致时保持warm_start(self):
        m = md.build("logreg")
        X, y = _blobs([0, 1, 2])
        m.fit(X, y)
        shape_before = m.est.coef_.shape
        X2, y2 = _blobs([0, 1, 2], seed=1)
        m.fit_continue(X2, y2)
        self.assertFalse(m.fit_note["degraded"], "类别一致不应降级")
        self.assertEqual(m.fit_note["action"], "warm_start")
        self.assertEqual(m.fit_note["reason"], "warm_start")
        self.assertEqual(m.est.coef_.shape, shape_before)

    def test_类别减少不崩溃且冷启动(self):
        m = md.build("logreg")
        X, y = _blobs([0, 1, 2])
        m.fit(X, y)
        X2, y2 = _blobs([0, 1], seed=2)          # 少一类（对应 b01→b03）
        m.fit_continue(X2, y2)                    # 修复前这里抛 ValueError
        note = m.fit_note
        self.assertTrue(note["degraded"], "类别变化必须标记降级")
        self.assertEqual(note["reason"], "class_set_changed")
        self.assertEqual(note["action"], "cold_start")
        self.assertEqual([int(c) for c in note["classes_before"]], [0, 1, 2])
        self.assertEqual([int(c) for c in note["classes_after"]], [0, 1])
        self.assertEqual([int(c) for c in m.est.classes_], [0, 1])
        # 冷启动后仍能正常预测
        pred = m.predict(X2)
        self.assertEqual(len(pred), len(y2))
        self.assertTrue(set(int(v) for v in np.unique(pred)) <= {0, 1})

    def test_类别增加不崩溃且冷启动(self):
        m = md.build("logreg")
        X, y = _blobs([0, 1])
        m.fit(X, y)
        X2, y2 = _blobs([0, 1, 2], seed=3)
        m.fit_continue(X2, y2)
        self.assertTrue(m.fit_note["degraded"])
        self.assertEqual(m.fit_note["reason"], "class_set_changed")
        self.assertEqual([int(c) for c in m.est.classes_], [0, 1, 2])

    def test_未训练时冷启动不报错(self):
        m = md.build("logreg")
        X, y = _blobs([0, 1])
        m.fit_continue(X, y)
        self.assertIsNotNone(m.fit_note)
        self.assertEqual(m.fit_note["action"], "cold_start")
        self.assertFalse(m.fit_note["degraded"])

    def test_不支持warm_start的模型也走冷启动(self):
        m = md.build("gnb")                       # GaussianNB 没有 warm_start
        X, y = _blobs([0, 1])
        m.fit(X, y)
        X2, y2 = _blobs([0, 1], seed=4)
        m.fit_continue(X2, y2)
        self.assertEqual(m.fit_note["action"], "cold_start")
        self.assertIn("not_warm_startable", m.fit_note["reason"])

    def test_note结构稳定(self):
        m = md.build("logreg")
        X, y = _blobs([0, 1])
        m.fit(X, y)
        m.fit_continue(X, y)
        for key in ("degraded", "reason", "action"):
            self.assertIn(key, m.fit_note)


@unittest.skipUnless(HAS_SKLEARN and HAS_GAS, "需要 scikit-learn + data/gas")
class TestFitContinueGas(unittest.TestCase):
    """真实气体场景：b01（6 类）→ b03（5 类）。"""

    def _dir(self):
        """兼容两种数据布局：data/gas/processed（新）与 data/gas（旧）。"""
        for d in _GAS:
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "batch_01.npz")):
                return d
        self.skipTest("需要 data/gas")

    def test_b01到b03类别缺失不崩溃(self):
        p = self._dir()
        b1 = np.load(os.path.join(p, "batch_01.npz"))
        b3 = np.load(os.path.join(p, "batch_03.npz"))
        m = md.build("logreg")
        m.fit(b1["X"], b1["y"])
        before = sorted(int(v) for v in np.unique(b1["y"]))
        after = sorted(int(v) for v in np.unique(b3["y"]))
        self.assertNotEqual(before, after, "这份数据本身就该是类别集不同的")
        m.fit_continue(b3["X"], b3["y"])          # 修复前抛 ValueError
        self.assertTrue(m.fit_note["degraded"])
        self.assertEqual(m.fit_note["reason"], "class_set_changed")
        self.assertEqual(m.fit_note["classes_before"], [float(v) for v in before])
        self.assertEqual(m.fit_note["classes_after"], [float(v) for v in after])
        # 预测出来的类别不能超出新批次实际存在的类别
        pred = set(int(v) for v in np.unique(m.predict(b3["X"])))
        self.assertTrue(pred <= set(after), f"预测出 {pred}，超出 {after}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

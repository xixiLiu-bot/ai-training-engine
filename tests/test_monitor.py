# -*- coding: utf-8 -*-
"""漂移检测三方法 + 标签漂移分派（对应整改 B1 + A4）。

    python -m unittest tests.test_monitor -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolver.loop.monitor import (_ks, _psi, _wasserstein, Monitor,
                                  label_drift, label_psi)


class TestDistances(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.a = rng.normal(0, 1, size=2000)
        self.b = rng.normal(0, 1, size=2000)          # 同分布
        self.c = rng.normal(3, 1, size=2000)          # 平移 3 个标准差

    def test_同分布_三种距离都接近0(self):
        self.assertLess(_psi(self.a, self.b), 0.02)
        self.assertLess(_ks(self.a, self.b), 0.06)
        self.assertLess(_wasserstein(self.a, self.b), 0.10)

    def test_平移后_三种距离都显著变大(self):
        self.assertGreater(_psi(self.a, self.c), 1.0)
        self.assertGreater(_ks(self.a, self.c), 0.5)
        self.assertGreater(_wasserstein(self.a, self.c), 1.0)

    def test_KS值域在0到1(self):
        for other in (self.b, self.c):
            v = _ks(self.a, other)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_Wasserstein可解释为标准化后的平移量(self):
        # 平移 3 个标准差 → 归一化后的 W1 应当接近 3
        self.assertAlmostEqual(_wasserstein(self.a, self.c), 3.0, delta=0.4)

    def test_空输入不崩(self):
        empty = np.array([])
        self.assertEqual(_psi(empty, self.a), 0.0)
        self.assertEqual(_ks(empty, self.a), 0.0)
        self.assertEqual(_wasserstein(empty, self.a), 0.0)


class TestMonitorDetector(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.ref = rng.normal(0, 1, size=(500, 4))
        self.cur = rng.normal(2, 1, size=(500, 4))

    def test_默认仍是PSI_键名保持不变(self):
        rep = Monitor(self.ref).check(self.cur)
        self.assertEqual(rep["detector"], "psi")
        self.assertIn("psi_mean", rep)
        self.assertIn("psi_max", rep)
        self.assertIn("per_feature", rep)

    def test_all模式同时报告三种(self):
        rep = Monitor(self.ref, detector="all").check(self.cur)
        for key in ("psi_mean", "ks_mean", "wasserstein_mean",
                    "psi_max", "ks_max", "wasserstein_max"):
            self.assertIn(key, rep)

    def test_ks模式的主字段是KS值(self):
        rep = Monitor(self.ref, detector="ks").check(self.cur)
        self.assertEqual(rep["detector"], "ks")
        self.assertLessEqual(rep["psi_mean"], 1.0)   # 旧键名里装的是 KS，值域仍 [0,1]

    def test_未知检测器报错(self):
        with self.assertRaises(ValueError):
            Monitor(self.ref, detector="mmd")

    def test_重置参照系后漂移归零(self):
        mon = Monitor(self.ref)
        mon.check(self.cur)
        mon.reset_reference(self.cur)
        rep = mon.check(self.cur)
        self.assertLess(rep["psi_mean"], 0.05)


class TestLabelDrift(unittest.TestCase):
    def test_分类任务走PSI(self):
        y1 = np.array([0] * 50 + [1] * 50)
        y2 = np.array([0] * 20 + [1] * 80)
        self.assertGreater(label_drift(y1, y2, "classification"), 0.0)
        self.assertAlmostEqual(label_drift(y1, y2, "classification"),
                               label_psi(y1, y2))

    def test_回归任务走KS_不会退化成0(self):
        """原实现对连续标签会失效，这里守住修复后不会。"""
        rng = np.random.default_rng(2)
        y1 = rng.normal(50, 10, size=500)
        y2 = rng.normal(80, 10, size=500)
        v = label_drift(y1, y2, "regression")
        self.assertGreater(v, 0.5, "连续标签的漂移必须能被测出来")
        self.assertLessEqual(v, 1.0)

    def test_回归任务同分布时接近0(self):
        rng = np.random.default_rng(3)
        y1 = rng.normal(50, 10, size=500)
        y2 = rng.normal(50, 10, size=500)
        self.assertLess(label_drift(y1, y2, "regression"), 0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestSmoothingAndTopK(unittest.TestCase):
    """空箱平滑 + top-k（对应 docs/漂移检测-问题与文献.docx 的问题一/三）。"""

    def setUp(self):
        rng = np.random.default_rng(7)
        # 参照批集中在一侧，当前批偏到另一侧 → 必然出现空箱
        self.ref = np.concatenate([rng.normal(0, 0.3, 180),
                                   rng.normal(10, 0.1, 20)])
        self.cur = np.concatenate([rng.normal(0, 0.3, 40),
                                   rng.normal(10, 0.1, 160)])

    def test_默认平滑与eps无关(self):
        """问题一的验收标准：改 eps 不应改变结果。"""
        vals = [_psi(self.ref, self.cur, bins=10, eps=e)
                for e in (1e-2, 1e-6, 1e-12)]
        self.assertLess(max(vals) - min(vals), 1e-12,
                        f"Laplace 平滑不应依赖 eps，实际 {vals}")

    def test_旧的clip平滑会随eps漂移(self):
        """反向测试：证明修复确实起作用，防止以后被改回去。"""
        vals = [_psi(self.ref, self.cur, bins=10, eps=e, smoothing="clip")
                for e in (1e-2, 1e-12)]
        self.assertGreater(max(vals) - min(vals), 0.1,
                           "旧行为本来就该不稳定，否则这条测试没有意义")

    def test_完全分离时仍给出显著值(self):
        """两个分布完全不重叠，PSI 必须很大——不能被任何平滑压成 0。"""
        rng = np.random.default_rng(11)
        v = _psi(rng.normal(0, 1, 500), rng.normal(8, 1, 500), bins=10)
        self.assertGreater(v, 1.0, "完全不重叠却给出接近 0，说明平滑方式有缺陷")

    def test_topk夹在均值与最大值之间(self):
        m = Monitor(self.ref.reshape(-1, 1))
        rep = m.check(self.cur.reshape(-1, 1), threshold=0.25)
        self.assertIn("psi_topk", rep)
        self.assertGreaterEqual(rep["psi_topk"], rep["psi_mean"] - 1e-9)
        self.assertLessEqual(rep["psi_topk"], rep["psi_max"] + 1e-9)

    def test_三个信号互补_数值关系正确(self):
        """mean <= topk <= max：三个指标从不同角度刻画同一批漂移。"""
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, size=(500, 60))
        cur = rng.normal(0, 1, size=(500, 60))
        cur[:, :10] = rng.normal(0.6, 1, size=(500, 10))     # 10 个特征轻漂
        rep = Monitor(ref).check(cur, threshold=0.25)
        self.assertLessEqual(rep["psi_mean"], rep["psi_topk"] + 1e-9)
        self.assertLessEqual(rep["psi_topk"], rep["psi_max"] + 1e-9)
        # 均值被多数正常特征稀释，topk 才看得出「有一批特征在漂」
        self.assertGreater(rep["psi_topk"], rep["psi_mean"] * 2)

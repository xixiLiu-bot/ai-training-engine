# -*- coding: utf-8 -*-
"""指标分派（整改 A4）。

    python -m unittest tests.test_metric -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolver.adapters import metric as mt

try:                                    # 无 sklearn 的环境（如评测容器外的最小环境）
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except Exception:                       # noqa: BLE001
    HAS_SKLEARN = False


class TestBuildFor(unittest.TestCase):
    def test_分类默认F1(self):
        self.assertEqual(mt.build_for("classification").name, "f1_macro")

    def test_回归默认MAE(self):
        self.assertEqual(mt.build_for("regression").name, "mae")

    def test_显式指标优先(self):
        self.assertEqual(mt.build_for("regression", "rmse").name, "rmse")

    def test_指标与任务不匹配时报错(self):
        """拿 F1 去评回归任务，必须立刻报错，而不是算出一个假分数。"""
        with self.assertRaises(ValueError):
            mt.build_for("regression", "f1_macro")
        with self.assertRaises(ValueError):
            mt.build_for("classification", "mae")

    def test_未知任务报错(self):
        with self.assertRaises(ValueError):
            mt.build_for("ranking")

    def test_旧接口仍然可用(self):
        self.assertEqual(mt.build("f1_macro").name, "f1_macro")
        self.assertEqual(mt.build("accuracy").name, "accuracy")


class TestSignConvention(unittest.TestCase):
    """所有指标必须"越大越好"，否则门控方向会反。"""

    @unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn 才能算指标数值")
    def test_MAE取负号(self):
        m = mt.build("mae")
        perfect = m.score([1.0, 2.0], [1.0, 2.0])
        worse = m.score([1.0, 2.0], [1.0, 5.0])
        self.assertAlmostEqual(perfect, 0.0)
        self.assertLess(worse, perfect, "误差大反而分数高 = 门控方向反了")

    @unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn 才能算指标数值")
    def test_RMSE取负号(self):
        m = mt.build("rmse")
        self.assertLess(m.score([1.0], [3.0]), 0.0)

    def test_任务类型标注正确(self):
        self.assertEqual(mt.build("f1_macro").task_type, "classification")
        self.assertEqual(mt.build("accuracy").task_type, "classification")
        self.assertEqual(mt.build("mae").task_type, "regression")
        self.assertEqual(mt.build("rmse").task_type, "regression")


if __name__ == "__main__":
    unittest.main(verbosity=2)

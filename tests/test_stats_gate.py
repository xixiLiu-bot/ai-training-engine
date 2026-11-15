# -*- coding: utf-8 -*-
"""门控的统计检验：锁定行为，防止以后被改回"比大小"。"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evolver.loop.stats import (                       # noqa: E402
    macro_f1_local, mcnemar, paired_bootstrap_test, required_margin_for,
)


class TestMcNemar(unittest.TestCase):
    def test_完全一致时不显著(self):
        y = np.array([0, 1, 2] * 10)
        r = mcnemar(y, y, y)
        self.assertEqual(r["n"], 0)
        self.assertFalse(r["significant"])
        self.assertEqual(r["p_value"], 1.0)

    def test_候选全对现役全错时显著(self):
        y = np.array([0, 1, 2] * 10)
        b = (y + 1) % 3                      # 现役全错
        r = mcnemar(y, y, b)
        self.assertEqual(r["b"], len(y))
        self.assertTrue(r["significant"])

    def test_分歧太少时不显著(self):
        y = np.array([0] * 40)
        a = y.copy()
        b = y.copy()
        b[:2] = 1                            # 现役错 2 条、候选全对 → b=2, c=0
        r = mcnemar(y, a, b)
        self.assertEqual(r["b"], 2)
        self.assertEqual(r["c"], 0)
        self.assertFalse(r["significant"], "分歧 2:0 的 p=0.5，不该显著")


class TestMacroF1(unittest.TestCase):
    def test_与sklearn一致(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 200)
        p = rng.integers(0, 3, 200)
        self.assertAlmostEqual(macro_f1_local(y, p),
                               f1_score(y, p, average="macro"), places=9)


class TestPairedBootstrap(unittest.TestCase):
    def test_候选普遍更好时显著(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 60 + [1] * 60 + [2] * 60)
        b = y.copy()
        idx = rng.choice(len(y), 30, replace=False)
        b[idx] = (b[idx] + 1) % 3
        r = paired_bootstrap_test(y, y, b)
        self.assertGreater(r["observed_diff"], 0)
        self.assertTrue(r["significant"])
        self.assertGreater(r["ci_low"], 0)

    def test_准确率不变但macro_f1改善时能识别(self):
        """这是 McNemar 的盲点，也是换检验的原因。"""
        y = np.array([0] * 90 + [1] * 15 + [2] * 15)
        a = y.copy()                          # 候选：少数类全对，多数类错 12
        b = y.copy()                          # 现役：少数类全错
        minor = np.where(y == 1)[0]
        major = np.where(y == 0)[0]
        b[minor] = 2
        a[major[:12]] = 1

        mc = mcnemar(y, a, b)
        bs = paired_bootstrap_test(y, a, b)
        self.assertGreater(bs["observed_diff"], 0.2)
        self.assertTrue(bs["significant"], "macro-F1 差异巨大，必须显著")
        self.assertFalse(mc["significant"],
                         "McNemar 在这里会误判——正因如此门控不用它")

    def test_两模型相同时不显著(self):
        y = np.array([0, 1, 2] * 20)
        r = paired_bootstrap_test(y, y, y)
        self.assertEqual(r["observed_diff"], 0)
        self.assertFalse(r["significant"])


class TestRequiredMargin(unittest.TestCase):
    def test_样本越少要求越大(self):
        self.assertGreater(required_margin_for(14), required_margin_for(846))

    def test_量级合理(self):
        # 每类 14 条时，理论噪声约 0.2（0.095×2）
        self.assertTrue(0.1 < required_margin_for(14) < 0.4)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""sweep.py 的不确定性报告：三类必须分开，且不得凭 std==0 说"稳健"。

背景（第 5 步实测）：
    logreg 是确定性算法，换 seed 结果完全一样（std=0），脚本却打印
    "种子间差异很小，这个数字比较稳"；而 rf 的成本 std 达 2.66s（CV≈57%）时
    仍然给出"稳"的结论。

    python -m unittest tests.test_sweep_variance -v
"""
import importlib.util
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    "sweep_under_test", os.path.join(REPO, "scripts", "sweep.py"))
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)


class TestVarianceReport(unittest.TestCase):
    def _report(self, **kw):
        base = dict(model_kind="logreg", n_seeds=3,
                    system=[0.864, 0.864, 0.864],
                    gain=[0.365, 0.365, 0.365],
                    cost=[1.15, 1.15, 1.15],
                    ci=(0.786, 0.926), pooled_n=27)
        base.update(kw)
        return sw.variance_report(**base)

    def test_确定性模型被明确标记(self):
        r = self._report(model_kind="logreg")
        self.assertTrue(r["deterministic_model"])
        self.assertFalse(r["seed_varies_model"])
        text = "\n".join(r["lines"])
        self.assertIn("确定性", text)
        self.assertIn("不代表结论稳健", text)

    def test_确定性模型不输出比较稳(self):
        r = self._report(model_kind="logreg")
        text = "\n".join(r["lines"])
        self.assertNotIn("比较稳", text, "std=0 不得被解释成'稳'")

    def test_随机模型的种子方差会报告(self):
        r = self._report(model_kind="rf",
                         system=[0.54, 0.545, 0.538],
                         gain=[0.03, 0.034, 0.027],
                         cost=[1.6, 6.2, 6.2])
        self.assertFalse(r["deterministic_model"])
        self.assertTrue(r["seed_varies_model"])
        self.assertGreater(r["system_std"], 0)

    def test_成本方差大时提示成本不稳定(self):
        r = self._report(model_kind="rf",
                         system=[0.54, 0.545, 0.538],
                         gain=[0.03, 0.034, 0.027],
                         cost=[1.6, 6.2, 6.2])
        self.assertTrue(r["cost_unstable"])
        text = "\n".join(r["lines"])
        self.assertIn("成本不稳定", text)
        self.assertIn("CV=", text)

    def test_确定性且成本不稳时结论指向成本(self):
        r = self._report(model_kind="logreg",
                         system=[0.864, 0.864, 0.864],
                         gain=[0.365, 0.365, 0.365],
                         cost=[1.0, 3.0, 9.0])
        self.assertTrue(r["cost_unstable"])
        last = r["lines"][-1]
        self.assertIn("成本", last)
        self.assertIn("区间", last)

    def test_三类不确定性都出现在报告里(self):
        r = self._report()
        text = "\n".join(r["lines"])
        self.assertIn("① 模型方差", text)
        self.assertIn("② 批次不确定性", text)
        self.assertIn("③ 运行成本方差", text)

    def test_bootstrap区间被原样带出(self):
        r = self._report(ci=(0.70, 0.91), pooled_n=30)
        self.assertEqual(r["ci"], (0.70, 0.91))
        self.assertEqual(r["pooled_n"], 30)
        self.assertIn("0.7000", "\n".join(r["lines"]))

    def test_单种子不炸(self):
        r = self._report(n_seeds=1, system=[0.5], gain=[0.1], cost=[1.0])
        self.assertEqual(r["system_std"], 0.0)
        self.assertEqual(r["cost_cv"], 0.0)


class TestDeterministicRegistry(unittest.TestCase):
    def test_已知确定性模型集合(self):
        for k in ("logreg", "gnb", "ridge"):
            self.assertIn(k, sw.DETERMINISTIC_MODELS)
        for k in ("rf", "mlp", "sgd"):
            self.assertNotIn(k, sw.DETERMINISTIC_MODELS,
                             f"{k} 受 random_state 影响，不应算确定性")

    def test_bootstrap_ci仍可用(self):
        lo, hi = sw.bootstrap_ci([0.5] * 10)
        self.assertAlmostEqual(lo, 0.5, places=6)
        self.assertAlmostEqual(hi, 0.5, places=6)
        lo, hi = sw.bootstrap_ci([0.1, 0.9, 0.5, 0.3])
        self.assertLess(lo, hi)


if __name__ == "__main__":
    unittest.main(verbosity=2)

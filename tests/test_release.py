# -*- coding: utf-8 -*-
"""门控 / 版本快照 / 回滚 的回归测试（对应整改 A3 + B2）。

    python -m unittest tests.test_release -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolver.loop.release import ReleaseManager, bootstrap_lower_mean


class ToyModel:
    """带状态的假模型：用来验证快照是否真的被隔离。"""

    def __init__(self, tag: str, w: float = 1.0):
        self.tag = tag
        self.w = w

    def clone(self):
        return ToyModel(self.tag, self.w)

    def predict(self, X):                       # pragma: no cover
        return X


class TestSnapshot(unittest.TestCase):
    def test_上线后保留快照(self):
        rel = ReleaseManager(ToyModel("v0"))
        ok = rel.try_release(ToyModel("v1"), 0.90, 0.50)
        self.assertEqual(ok["decision"], "accept")
        self.assertEqual(rel.version, 1)
        self.assertEqual(rel.snapshot_versions(), [0, 1])
        self.assertEqual(rel.current.tag, "v1")

    def test_回滚到更早版本(self):
        rel = ReleaseManager(ToyModel("v0"))
        rel.try_release(ToyModel("v1"), 0.90, 0.50)
        rel.try_release(ToyModel("v2"), 0.95, 0.90)
        self.assertEqual(rel.version, 2)

        res = rel.rollback_to(1)
        self.assertTrue(res["ok"])
        self.assertEqual(rel.version, 1)
        self.assertEqual(rel.current.tag, "v1")
        # 回滚后，v2 的快照应当被丢弃
        self.assertEqual(rel.snapshot_versions(), [0, 1])

    def test_回滚到不存在的版本不抛异常(self):
        rel = ReleaseManager(ToyModel("v0"))
        res = rel.rollback_to(99)
        self.assertFalse(res["ok"])
        self.assertEqual(rel.version, 0)          # 状态不变
        self.assertEqual(rel.current.tag, "v0")

    def test_快照是副本不是引用(self):
        """关键：上线后修改现役模型，不能把快照里的旧模型也改掉。"""
        rel = ReleaseManager(ToyModel("v0"))
        rel.try_release(ToyModel("v1", w=2.0), 0.90, 0.50)
        rel.current.w = 99.0                       # 模拟后续训练改了权重
        rel.rollback_to(0)
        self.assertEqual(rel.current.w, 1.0, "快照里的旧模型被污染了")

    def test_拒绝即维持(self):
        rel = ReleaseManager(ToyModel("v0"))
        rec = rel.try_release(ToyModel("bad"), 0.40, 0.50)
        self.assertEqual(rec["decision"], "rollback")
        self.assertEqual(rel.current.tag, "v0")
        self.assertEqual(rel.version, 0)

    def test_快照数量上限(self):
        rel = ReleaseManager(ToyModel("v0"), max_snapshots=3)
        for i in range(1, 7):
            rel.try_release(ToyModel(f"v{i}"), 0.5 + i * 0.1, 0.5 + (i - 1) * 0.1)
        self.assertLessEqual(len(rel.snapshots), 3)
        self.assertIn(0, rel.snapshot_versions(), "version 0 必须一直保留")


class TestWindowGate(unittest.TestCase):
    def test_默认只看当前批_保持旧行为(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02)
        self.assertTrue(rel.gate(0.60, 0.50)[0])
        self.assertFalse(rel.gate(0.51, 0.50)[0])

    def test_窗口未满时退回单批判据(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, gate_window=3)
        ok, why = rel.gate(0.60, 0.50)           # 只有 1 批，窗口未满
        self.assertTrue(ok)
        self.assertIn("同批对比", why)

    def test_窗口均值判据(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, gate_window=3)
        rel.gate(0.60, 0.50)                     # +0.10
        rel.gate(0.50, 0.50)                     # +0.00
        ok, why = rel.gate(0.48, 0.50)           # −0.02 → 均值 +0.027 ≥ 0.02
        self.assertTrue(ok, why)
        self.assertIn("窗口均值", why)

    def test_窗口判据能拒绝噪声(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, gate_window=3)
        rel.gate(0.51, 0.50)                     # +0.01
        rel.gate(0.49, 0.50)                     # −0.01
        ok, _ = rel.gate(0.51, 0.50)             # +0.01 → 均值 +0.0033 < 0.02
        self.assertFalse(ok)

    def test_bootstrap_下界比均值更保守(self):
        gains = [0.05, 0.05, 0.05, 0.05]
        low = bootstrap_lower_mean(gains, n_boot=300, seed=0)
        self.assertLessEqual(low, 0.05 + 1e-9)

    def test_bootstrap_样本太少退回均值(self):
        self.assertAlmostEqual(bootstrap_lower_mean([0.1, 0.2], n_boot=100), 0.15)


class TestMarginNormalization(unittest.TestCase):
    """整改 R1：margin 必须按指标量纲归一化。

    约定：内部一律用 utility（越大越好）。MAE 的 utility = −MAE。
    """

    def test_absolute模式保持旧行为(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="absolute")
        self.assertTrue(rel.gate(0.52, 0.50)[0])     # +0.02
        self.assertFalse(rel.gate(0.51, 0.50)[0])    # +0.01

    def test_relative模式按baseline归一(self):
        # baseline 0.50，增益 0.02 → 4% ≥ 2%
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="relative")
        self.assertTrue(rel.gate(0.52, 0.50)[0])
        # 增益 0.005 → 1% < 2%
        self.assertFalse(rel.gate(0.505, 0.50)[0])

    def test_回归越低越好_改善被正确识别(self):
        # MAE utility：-22 → -20 表示 MAE 从 22 降到 20，改善 2，相对 2/22 ≈ 9%
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="relative",
                             metric_name="mae", metric_direction="lower")
        ok, why = rel.gate(-20.0, -22.0)
        self.assertTrue(ok, why)

    def test_回归方向不会反过来(self):
        # MAE 变差：-22 → -24，归一化增益为负，必须拒绝
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="relative",
                             metric_name="mae", metric_direction="lower")
        ok, _ = rel.gate(-24.0, -22.0)
        self.assertFalse(ok)

    def test_回归微小改善不再被0_02卡住(self):
        """原缺陷：MAE≈20 时绝对阈值 0.02 等价于'几乎任何正增益都放行'，
        但方向相反的情形（0.02 天以内）又被拒绝得很随意。这里验证量纲对齐。"""
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="relative",
                             metric_name="mae", metric_direction="lower")
        # 改善 0.2 天 → 相对 1% < 2% → 拒绝（旧实现会因 0.2>0.02 而放行）
        self.assertFalse(rel.gate(-21.8, -22.0)[0])

    def test_baseline接近0时退回absolute不除零(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="relative")
        ok, why = rel.gate(0.5, 0.0)                 # baseline=0 → 不能除
        self.assertTrue(ok)
        self.assertEqual(rel.last_mode_used, "absolute_fallback")
        self.assertIn("退回 absolute", why)

    def test_负基线用绝对值归一(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="relative")
        # baseline -0.5，candidate -0.4 → raw_gain 0.1，|−0.5| → 20% ≥ 2%
        self.assertTrue(rel.gate(-0.4, -0.5)[0])

    def test_相对模式下的日志字段(self):
        rel = ReleaseManager(ToyModel("v0"), margin=0.02, margin_mode="relative",
                             metric_name="mae", metric_direction="lower")
        rec = rel.try_release(ToyModel("v1"), -20.0, -22.0)
        for key in ("metric_name", "metric_direction", "baseline_score",
                    "candidate_score", "raw_gain", "normalized_gain",
                    "margin", "margin_mode", "release_decision"):
            self.assertIn(key, rec, f"缺少门控留痕字段 {key}")
        self.assertEqual(rec["metric_name"], "mae")
        self.assertEqual(rec["metric_direction"], "lower")
        self.assertEqual(rec["baseline_score"], 22.0, "日志里应是人类可读的 MAE")
        self.assertEqual(rec["candidate_score"], 20.0)
        self.assertAlmostEqual(rec["raw_gain"], 2.0)
        self.assertAlmostEqual(rec["normalized_gain"], 2.0 / 22.0, places=6)
        self.assertEqual(rec["release_decision"], "accept")

    def test_非法margin_mode报错(self):
        with self.assertRaises(ValueError):
            ReleaseManager(ToyModel("v0"), margin_mode="ratio")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# -*- coding: utf-8 -*-
"""协议层的回归测试：确保「偷看未来」和「不交卷」真的会被拦。

    D:\\anaconda\\python.exe -m unittest tests.test_protocol -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolver.protocol import LocalEvaluator, ProtocolViolation, Submission


class FakeBatch:
    def __init__(self, i):
        self.batch_id = i
        self.X = np.zeros((3, 2))
        self.y = np.zeros(3, dtype=int)
        self.meta = {}


class FakeAdapter:
    name = "fake"
    n_batches = 5

    def __init__(self):
        self.loads = []

    def get(self, i):
        self.loads.append(i)          # 记录到底加载了哪些批次
        return FakeBatch(i)


def sub(batch_id, decision="deploy", round_no=1):
    return Submission(round_no=round_no, batch_id=batch_id,
                      decision=decision, model_version=1, reason="test")


class TestProtocol(unittest.TestCase):
    def test_必须交卷才能拿下一批(self):
        ev = LocalEvaluator(FakeAdapter())
        ev.next_batch()
        with self.assertRaises(ProtocolViolation):
            ev.next_batch()           # 没交卷就要下一批 → 拦下
        self.assertEqual(len(ev.violations), 1)

    def test_交卷之后才能继续(self):
        ev = LocalEvaluator(FakeAdapter())
        b1 = ev.next_batch()
        ev.submit(sub(b1.batch_id))
        b2 = ev.next_batch()
        self.assertEqual(b2.batch_id, 2)

    def test_不许偷看未来(self):
        ev = LocalEvaluator(FakeAdapter())
        ev.next_batch()               # 只下发到第 1 批
        with self.assertRaises(ProtocolViolation):
            ev.history(1, 4)          # 想看第 4 批 → 拦下
        self.assertTrue(any("尚未下发" in v for v in ev.violations))

    def test_只加载真正下发过的批次(self):
        adapter = FakeAdapter()
        ev = LocalEvaluator(adapter)
        ev.next_batch()
        ev.submit(sub(1))
        ev.next_batch()
        # 只应该加载过 1、2 两批，绝不能提前加载 3/4/5
        self.assertEqual(adapter.loads, [1, 2])

    def test_决策只能是deploy或keep(self):
        with self.assertRaises(ProtocolViolation):
            sub(1, decision="maybe")

    def test_提交批次必须匹配(self):
        ev = LocalEvaluator(FakeAdapter())
        ev.next_batch()
        with self.assertRaises(ProtocolViolation):
            ev.submit(sub(3))         # 当前待处理是 1，却提交 3


if __name__ == "__main__":
    unittest.main(verbosity=2)

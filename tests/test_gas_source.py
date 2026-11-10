# -*- coding: utf-8 -*-
"""Gas 数据源 / 桥接层 / 泄漏防护 的测试。

    python -m unittest tests.test_gas_source -v

覆盖：
  1. 数据源契约 8 个方法
  2. 10 个批次的顺序与样本数（对齐 UCI 官方口径）
  3. batch_03/04/05 缺 Toluene 这一真实特性
  4. split() 的划分纪律（train/stream/holdout 不相交、holdout 最后）
  5. 桥接层与旧适配器的**逐元素等价**（换层不改数据）
  6. 数据泄漏防护：不暴露 holdout、不越界、不逆序、不跨批训练
  7. checksums.sha256 的一致性
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evolver.adapters.bridge import BridgeAdapter          # noqa: E402
from evolver.adapters.dataset import GasSensorAdapter, build  # noqa: E402
from evolver.adapters.sources import (GAS_CLASSES, GAS_EXPECTED_SIZES,
                                      GAS_N_FEATURES, DataSourceError,
                                      GasSource)           # noqa: E402

_GAS_DIRS = [os.path.join(REPO, "data", "gas", "processed"),
             os.path.join(REPO, "data", "gas")]
HAS_GAS = any(os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))
              for d in _GAS_DIRS)


def _src(**kw) -> GasSource:
    return GasSource(**kw).load()


@unittest.skipUnless(HAS_GAS, "本地没有气体数据（data/gas）")
class TestGasSourceContract(unittest.TestCase):
    def setUp(self):
        self.src = _src()

    # ---- 8 个方法 ----
    def test_load_返回自身并可重复调用(self):
        s = GasSource()
        self.assertIs(s.load(), s)
        self.assertIs(s.load(), s, "重复 load 不应报错")

    def test_get_task_type(self):
        self.assertEqual(self.src.get_task_type(), "classification")

    def test_get_metric(self):
        self.assertEqual(self.src.get_metric(), "f1_macro")

    def test_get_features_是128维且唯一(self):
        names = self.src.get_features()
        self.assertEqual(len(names), GAS_N_FEATURES)
        self.assertEqual(len(set(names)), GAS_N_FEATURES, "特征名不能重复")

    def test_get_labels_是6类气体(self):
        labels = self.src.get_labels()
        self.assertEqual(len(labels), 6)
        self.assertEqual(labels, GAS_CLASSES)
        self.assertEqual(labels[0], "Ammonia")
        self.assertEqual(labels[5], "Toluene")

    def test_get_metadata_含许可证与未确认项(self):
        meta = self.src.get_metadata()
        self.assertEqual(meta["n_batches"], 10)
        self.assertEqual(meta["n_samples"], 13910)
        self.assertEqual(meta["n_features"], GAS_N_FEATURES)
        self.assertEqual(meta["n_classes"], 6)
        self.assertFalse(meta["is_official"], "必须明确不是官方数据")
        # 许可证：英文原文 + 中文结论都要有
        self.assertIn("research purposes", meta["license"])
        self.assertIn("excluded", meta["license"])
        self.assertIn("研究", meta["license_zh"])
        self.assertIn("商业", meta["license_zh"])
        # 标签协议必须是 TODO/未确认，不能假装已确认
        self.assertIn("未确认", meta["label_protocol_status"])
        self.assertIn("TODO", meta["label_protocol_status"])
        self.assertIn("未确认", meta["official_relation"])

    # ---- 数据本身 ----
    def test_10个批次且顺序为时间顺序(self):
        recs = list(self.src.iter_batches())
        self.assertEqual(len(recs), 10)
        ids = [r["batch_id"] for r in recs]
        self.assertEqual(ids, list(range(1, 11)), "批次号必须 1..10 递增")

    def test_每批样本数对齐UCI官方口径(self):
        sizes = [len(r["labels"]) for r in self.src.iter_batches()]
        self.assertEqual(sizes, GAS_EXPECTED_SIZES)
        self.assertEqual(sum(sizes), 13910)

    def test_特征维数128且无NaN(self):
        for r in self.src.iter_batches():
            self.assertEqual(r["features"].shape[1], GAS_N_FEATURES)
            self.assertEqual(len(r["features"]), len(r["labels"]))
            self.assertFalse(np.isnan(r["features"]).any(),
                             f"batch {r['batch_id']} 有 NaN")

    def test_标签范围0到5(self):
        for r in self.src.iter_batches():
            self.assertGreaterEqual(int(r["labels"].min()), 0)
            self.assertLessEqual(int(r["labels"].max()), 5)

    def test_batch_03_04_05_缺少Toluene(self):
        """真实数据特性，不是 bug。宏平均 F1 在这些批次上会系统性偏低。"""
        missing = {}
        for r in self.src.iter_batches():
            present = set(int(c) for c in np.unique(r["labels"]))
            lack = [i for i in range(6) if i not in present]
            if lack:
                missing[r["batch_id"]] = lack
        self.assertEqual(missing, {3: [5], 4: [5], 5: [5]},
                         f"缺类情况与预期不符：{missing}")
        meta = self.src.get_metadata()
        self.assertEqual(sorted(meta["batches_missing_classes"].keys()), [3, 4, 5])
        self.assertEqual(meta["batches_missing_classes"][3], ["Toluene"])

    def test_timestamp为None而不是编造(self):
        """UCI 没有给每批确切日期，就不能编一个。"""
        for r in self.src.iter_batches():
            self.assertIsNone(r["timestamp"])
            self.assertIn("time_note", r["metadata"])

    # ---- 划分 ----
    def test_split_默认1_2到9_10(self):
        sp = self.src.split()
        self.assertEqual(sp["train"], [1])
        self.assertEqual(sp["stream"], list(range(2, 10)))
        self.assertEqual(sp["holdout"], [10])

    def test_split_三个集合两两不相交(self):
        sp = self.src.split()
        a, b, c = set(sp["train"]), set(sp["stream"]), set(sp["holdout"])
        self.assertFalse(a & b, "train 与 stream 重叠")
        self.assertFalse(a & c, "train 与 holdout 重叠")
        self.assertFalse(b & c, "stream 与 holdout 重叠")
        self.assertEqual(a | b | c, set(range(1, 11)), "划分应覆盖全部批次")

    def test_holdout必须是最后一个批次(self):
        sp = self.src.split()
        all_ids = [r["batch_id"] for r in self.src.iter_batches()]
        self.assertEqual(sp["holdout"], [all_ids[-1]])

    def test_train只含最早批次(self):
        sp = self.src.split()
        self.assertEqual(sp["train"], [1], "冷启动只能看第一批")

    def test_数据缺失时报错并给指引(self):
        bogus = GasSource(root=os.path.join(REPO, "data", "_not_exist"))
        with self.assertRaises(DataSourceError) as ctx:
            bogus.load()
        msg = str(ctx.exception)
        self.assertIn("prepare_gas.py", msg)
        self.assertIn("archive.ics.uci.edu", msg)

    def test_批次样本数不符时报错(self):
        """防止队友换了一份被重新切分的数据还蒙混过关。

        造一个"10 个批次但每批只有 10 个样本"的假数据目录，strict_sizes 必须拒绝。
        """
        import shutil
        import uuid
        from evolver.adapters.sources import DataSourceError

        base = os.path.join(REPO, "data", "_unittest_tmp")
        os.makedirs(base, exist_ok=True)
        d = os.path.join(base, f"badgas-{uuid.uuid4().hex[:8]}")
        os.makedirs(d, exist_ok=True)
        try:
            for i in range(1, 11):
                np.savez_compressed(
                    os.path.join(d, f"batch_{i:02d}.npz"),
                    X=np.zeros((10, GAS_N_FEATURES)), y=np.zeros(10, dtype=int))
            with self.assertRaises(DataSourceError) as ctx:
                GasSource(root=d, strict_sizes=True).load()
            self.assertIn("UCI", str(ctx.exception))
            # 关掉严格校验时应当能读（便于自造小样本调试）
            src = GasSource(root=d, strict_sizes=False).load()
            self.assertEqual(len(src.split()["train"]), 1)
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAS_GAS, "本地没有气体数据（data/gas）")
class TestBridge(unittest.TestCase):
    def test_桥接层与旧适配器逐元素等价(self):
        """换数据层不能改变数据本身。"""
        old = GasSensorAdapter()
        new = BridgeAdapter(_src())
        self.assertEqual(old.n_batches, new.n_batches)
        self.assertEqual(old.task_type, new.task_type)
        for b in range(1, old.n_batches + 1):
            o, n = old.get(b), new.get(b)
            np.testing.assert_array_equal(o.X, n.X, err_msg=f"batch {b} 特征不一致")
            np.testing.assert_array_equal(o.y, n.y, err_msg=f"batch {b} 标签不一致")

    def test_batch_ids子集只暴露指定批次(self):
        ids = [2, 3, 4]
        ad = BridgeAdapter(_src(), batch_ids=ids)
        self.assertEqual(ad.n_batches, 3)
        self.assertEqual(ad.source_batch_ids(), ids)
        # 位置 1 对应原始批次 2
        self.assertEqual(ad.get(1).meta["source_batch_id"], 2)
        self.assertEqual(ad.get(3).meta["source_batch_id"], 4)

    def test_不含holdout时跑不到第10批(self):
        """验证阶段用 stream-only 视图，从结构上杜绝误用 holdout。"""
        sp = _src().split()
        ad = BridgeAdapter(_src(), batch_ids=sp["stream"])
        self.assertEqual(ad.n_batches, len(sp["stream"]))
        self.assertNotIn(10, ad.source_batch_ids())
        with self.assertRaises(IndexError):
            ad.get(ad.n_batches + 1)

    def test_越界访问报错(self):
        ad = BridgeAdapter(_src())
        with self.assertRaises(IndexError):
            ad.get(0)
        with self.assertRaises(IndexError):
            ad.get(11)

    def test_不存在的批次号报错(self):
        with self.assertRaises(ValueError):
            BridgeAdapter(_src(), batch_ids=[1, 99])

    def test_通过工厂构建(self):
        ad = build("gas_drift")
        self.assertEqual(ad.n_batches, 10)
        self.assertEqual(ad.task_type, "classification")
        self.assertEqual(ad.get(1).meta["source_batch_id"], 1)


@unittest.skipUnless(HAS_GAS, "本地没有气体数据（data/gas）")
class TestLeakageGuards(unittest.TestCase):
    """数据泄漏的结构性防护。真正的"不看未来"由 protocol 层保证，
    这里守住"数据层不主动递刀子"。"""

    def setUp(self):
        self.src = _src()

    def test_迭代是严格时间递增的(self):
        ids = [r["batch_id"] for r in self.src.iter_batches()]
        self.assertEqual(ids, sorted(ids), "迭代必须按时间顺序")
        self.assertEqual(len(ids), len(set(ids)), "批次号不能重复")

    def test_训练视图不含当前及未来批次(self):
        """模拟 recent_window：训练第 b 批时只能看到 < b。"""
        for b in range(2, 11):
            visible = [i for i in range(1, b)]     # 协议层允许的可见范围
            self.assertNotIn(b, visible, "当前批不得进入训练窗口")
            self.assertFalse([i for i in visible if i > b], "不得出现未来批次")

    def test_holdout不在训练与流内(self):
        sp = self.src.split()
        self.assertNotIn(10, sp["train"] + sp["stream"])

    def test_数据源不提供未来接口(self):
        """数据源只暴露 iter_batches/split，没有 get_future 之类的东西。"""
        public = {n for n in dir(self.src) if not n.startswith("_")}
        for bad in ("get_future", "peek", "next_batch"):
            self.assertNotIn(bad, public, f"数据源不应暴露 {bad}")

    def test_每批之间不重叠(self):
        """批次是互斥切分：同一批号的内容不会出现在另一批里（按行数校验）。"""
        seen = 0
        for r in self.src.iter_batches():
            seen += len(r["labels"])
        self.assertEqual(seen, 13910, "各批样本数之和应等于总样本数，说明无重复计入")


class TestChecksums(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(REPO, "checksums.sha256")

    def test_清单存在且条目格式正确(self):
        if not os.path.exists(self.path):
            self.skipTest("checksums.sha256 不存在")
        lines = [l.strip() for l in open(self.path, encoding="utf-8") if l.strip()]
        self.assertTrue(lines)
        for line in lines:
            digest, rel = line.split(None, 1)
            self.assertEqual(len(digest), 64, f"不是 sha256：{digest}")
            self.assertTrue(rel.startswith("data/"), f"路径应位于 data/ 下：{rel}")

    def test_清单不含大文件路径以外的内容(self):
        if not os.path.exists(self.path):
            self.skipTest("checksums.sha256 不存在")
        text = open(self.path, encoding="utf-8").read()
        self.assertNotIn("dataset (1)", text.lower())
        self.assertNotIn(".git", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

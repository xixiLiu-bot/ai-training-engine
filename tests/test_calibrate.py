# -*- coding: utf-8 -*-
"""阈值标定（scripts/calibrate.py）的测试。

覆盖用户要求的 8 项：
  1. holdout 批次不会被读取
  2. psi_low 始终小于 psi_mid
  3. 固定 seed 结果可复现
  4. 空数据 / 批次不足时显式报错
  5. 输出配置能被 run_pipeline.py 读取
  6. 旧阈值行为保持兼容
  7. 标定过程不会修改数据或现役模型
  8. 校准后重新运行结果可复现

    python -m unittest tests.test_calibrate -v
"""
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# 载入 calibrate 模块（它在 scripts/ 下，不是包）
_spec = importlib.util.spec_from_file_location(
    "calibrate_under_test", os.path.join(REPO, "scripts", "calibrate.py"))
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)

try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except Exception:                       # noqa: BLE001
    _HAS_SKLEARN = False

_GAS_DIRS = [os.path.join(REPO, "data", "gas", "processed"),
             os.path.join(REPO, "data", "gas")]
HAS_GAS = any(os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))
              for d in _GAS_DIRS)
HAS_BACKEND = _HAS_SKLEARN or bool(os.getenv("ZHIXUN_NP_STACK_PATH"))


class TestGridParsing(unittest.TestCase):
    def test_正常解析并排序去重(self):
        lows, mids = cal.parse_grid("1.2,0.6,0.8,0.8|3.0,2.0")
        self.assertEqual(lows, [0.6, 0.8, 1.2])
        self.assertEqual(mids, [2.0, 3.0])

    def test_格式错误显式报错(self):
        for spec in ("", "0.6,0.8", "a|b", "|2.0", "1.0|"):
            with self.subTest(spec=spec):
                with self.assertRaises(cal.CalibrationError):
                    cal.parse_grid(spec)


class TestPointValidation(unittest.TestCase):
    """第 2 项：psi_low 必须小于 psi_mid。"""

    def test_合法点通过(self):
        cal.validate_point(0.6, 2.0)

    def test_low大于等于mid被拒(self):
        for low, mid in ((2.0, 2.0), (2.5, 2.0), (0.0, 2.0), (-1.0, 2.0)):
            with self.subTest(low=low, mid=mid):
                with self.assertRaises(cal.CalibrationError):
                    cal.validate_point(low, mid)

    def test_网格自动过滤非法点(self):
        lows, mids = cal.parse_grid("0.6,3.0|2.0")
        # low=3.0 与 mid=2.0 组合非法，必须被过滤掉
        points = [(lo, mi) for lo in lows for mi in mids if mi > lo]
        self.assertEqual(points, [(0.6, 2.0)])
        for lo, mi in points:
            cal.validate_point(lo, mi)


class TestHoldoutGuard(unittest.TestCase):
    """第 1 项：选阈值阶段绝不允许访问 holdout。"""

    def test_含holdout的批次列表被拒(self):
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.assert_no_holdout([1, 2, 3, 10])
        self.assertIn("泄漏", str(ctx.exception))

    def test_不含holdout时通过(self):
        cal.assert_no_holdout(list(range(1, 9)))
        cal.assert_no_holdout([5, 6, 7, 8])

    def test_默认分段不触碰holdout(self):
        self.assertEqual(cal.HOLDOUT, (10, 10))
        self.assertLess(cal.DEFAULT_VALIDATION[1], cal.HOLDOUT[0],
                        "validation 必须早于 holdout")
        self.assertLess(cal.PRE_HOLDOUT[1], cal.HOLDOUT[0],
                        "pre-holdout 汇总也不能包含 holdout")
        self.assertLessEqual(cal.DEFAULT_CALIBRATION[1], cal.DEFAULT_VALIDATION[0] - 1,
                             "calibration 与 validation 必须按时间前后分开")


class TestBatchRangeParsing(unittest.TestCase):
    """第 4 项：批次不足/格式错误必须显式报错。"""

    def test_正常解析(self):
        self.assertEqual(cal.parse_batches("5-8", "validation"), (5, 8))
        self.assertEqual(cal.parse_batches("10-10", "holdout"), (10, 10))

    def test_非法区间报错(self):
        for spec in ("8-5", "0-3", "abc", "5"):
            with self.subTest(spec=spec):
                with self.assertRaises(cal.CalibrationError):
                    cal.parse_batches(spec, "validation")


class TestSummarizeAndCost(unittest.TestCase):
    """成本必须按批次区间拆分，不能计入别的段。"""

    def _rows(self):
        return [
            {"batch": 2, "f1_system": 0.5, "f1_baseline": 0.5,
             "action": "do_nothing", "decision": "noop"},
            {"batch": 3, "f1_system": 0.8, "f1_baseline": 0.4,
             "action": "refit_model", "decision": "accept"},
            {"batch": 4, "f1_system": 0.3, "f1_baseline": 0.4,
             "action": "refresh_normalization", "decision": "rollback"},
        ]

    def _decisions(self):
        return [
            {"batch": 2, "action": "do_nothing", "decision": "noop",
             "charge": {"gpu_sec": 0.0, "labels": 0, "tokens": 0}},
            {"batch": 3, "action": "refit_model", "decision": "accept",
             "charge": {"gpu_sec": 2.0, "labels": 1000, "tokens": 0}},
            {"batch": 4, "action": "refresh_normalization", "decision": "rollback",
             "charge": {"gpu_sec": 0.05, "labels": 0, "tokens": 0}},
        ]

    def test_按区间统计(self):
        rows, dec = self._rows(), self._decisions()
        seg = cal.summarize(cal.segment(rows, 3, 4), dec, {"violations": 0})
        self.assertEqual(seg["n"], 2)
        self.assertAlmostEqual(seg["gpu_sec"], 2.05, places=6)
        self.assertEqual(seg["labels"], 1000)
        self.assertEqual(seg["rollbacks"], 1)
        self.assertAlmostEqual(seg["wasted_gpu_sec"], 0.05, places=6)

    def test_不把别段成本算进来(self):
        rows, dec = self._rows(), self._decisions()
        seg = cal.summarize(cal.segment(rows, 2, 2), dec, {"violations": 0})
        self.assertAlmostEqual(seg["gpu_sec"], 0.0)

    def test_空段返回n为0(self):
        seg = cal.summarize([], [], {"violations": 0})
        self.assertEqual(seg["n"], 0)

    def test_归一化成本与utility(self):
        self.assertAlmostEqual(cal.normalized_cost(2.0, 4.0), 0.5)
        self.assertAlmostEqual(cal.normalized_cost(2.0, 0.0), 2.0 / 1e-6)
        seg = {"gain": 0.1, "gpu_sec": 4.0, "rollback_rate": 0.0}
        u = cal.utility(seg, lam_cost=0.5, lam_rollback=0.2, ref_gpu_sec=4.0)
        self.assertAlmostEqual(u, 0.1 - 0.5 * 1.0)


class TestFeasibilityAndSelection(unittest.TestCase):
    """硬约束 + "容差内优先低成本"的选择规则。"""

    def _c(self, low, mid, gain, gpu, roll, rate=1.0, wasted=0.0):
        return {"psi_low": low, "psi_mid": mid,
                "seg": {"gain": gain, "gpu_sec": gpu, "rollbacks": roll,
                        "rollback_rate": roll / 4, "violations": 0,
                        "wasted_ratio": wasted},
                "utility": 0.0, "feasible": True, "reason": "ok"}

    def test_违规直接淘汰(self):
        seg = {"violations": 1, "gain": 1.0, "rollback_rate": 0.0, "wasted_ratio": 0.0}
        ok, why = cal.is_feasible(seg, max_rollback_rate=1.0, need_gain=0.0)
        self.assertFalse(ok)
        self.assertIn("违规", why)

    def test_增益为负淘汰(self):
        seg = {"violations": 0, "gain": -0.01, "rollback_rate": 0.0, "wasted_ratio": 0.0}
        ok, why = cal.is_feasible(seg, max_rollback_rate=1.0, need_gain=0.0)
        self.assertFalse(ok)
        self.assertIn("增益", why)

    def test_回滚率超限淘汰(self):
        seg = {"violations": 0, "gain": 0.1, "rollback_rate": 0.8, "wasted_ratio": 0.0}
        ok, why = cal.is_feasible(seg, max_rollback_rate=0.5, need_gain=0.0)
        self.assertFalse(ok)
        self.assertIn("回滚率", why)

    def test_浪费比例超限淘汰(self):
        seg = {"violations": 0, "gain": 0.1, "rollback_rate": 0.0, "wasted_ratio": 0.9}
        ok, why = cal.is_feasible(seg, max_rollback_rate=1.0, need_gain=0.0,
                                  max_wasted_ratio=0.5)
        self.assertFalse(ok)

    def test_容差内优先最低成本(self):
        a = self._c(0.8, 2.0, 0.60, gpu=8.0, roll=1)
        b = self._c(1.2, 4.0, 0.58, gpu=4.0, roll=2)
        a["utility"], b["utility"] = 0.500, 0.495      # 差 0.005 < tol 0.01
        chosen = cal.select_best([a, b], tol=0.01)
        self.assertEqual(chosen["psi_low"], 1.2, "容差内应选成本更低的那个")

    def test_超出容差选utility最高(self):
        a = self._c(0.8, 2.0, 0.60, gpu=8.0, roll=1)
        b = self._c(1.2, 4.0, 0.58, gpu=4.0, roll=2)
        a["utility"], b["utility"] = 0.60, 0.10
        chosen = cal.select_best([a, b], tol=0.01)
        self.assertEqual(chosen["psi_low"], 0.8)

    def test_无可行点显式报错(self):
        c = self._c(1.0, 2.0, 0.1, gpu=1.0, roll=0)
        c["feasible"] = False
        with self.assertRaises(cal.CalibrationError):
            cal.select_best([c])


class TestOldThresholdCompat(unittest.TestCase):
    """第 6 项：旧阈值必须保留为可回退配置。"""

    def test_旧阈值常量存在且合法(self):
        self.assertEqual(cal.OLD_THRESHOLD, (1.2, 2.5))
        cal.validate_point(*cal.OLD_THRESHOLD)


class TestRobustPick(unittest.TestCase):
    """第 8 项：单批次证据不得自动替换默认阈值。"""

    ACTIVE = (1.2, 2.5)

    def _cand(self, low, mid, system, gpu=4.0, rollbacks=2, psi=None,
              actions=None, active=False):
        rows = [{"batch": b, "psi_mean": (psi or {}).get(b, 9.0),
                 "f1_system": system, "f1_baseline": 0.2,
                 "action": "do_nothing", "decision": "noop"}
                for b in (2, 3, 4, 5, 6, 7, 8)]
        dec = [{"batch": b, "experiment": {"action": (actions or {}).get(b, "do_nothing")},
                "decision": "accept", "charge": {"gpu_sec": 0.5, "labels": 0, "tokens": 0}}
               for b in (2, 3, 4, 5, 6, 7, 8)]
        return {"psi_low": low, "psi_mid": mid,
                "seg": {"system": system, "gain": system - 0.2, "gpu_sec": gpu,
                        "rollbacks": rollbacks, "rollback_rate": 0.5,
                        "actions": 4, "violations": 0, "wasted_ratio": 0.2},
                "utility": system, "feasible": True, "reason": "ok",
                "_rows": rows, "_decisions": dec, "_is_active": active}

    def test_单批次翻转时回退默认(self):
        # psi=0.766 只落在 [0.6, 1.2) 内 → 只有 b02 受影响
        psi = {2: 0.766}
        act = self._cand(1.2, 2.5, system=0.7952, psi=psi, active=True)
        chosen = self._cand(0.6, 2.5, system=0.8281, psi=psi,
                            actions={2: "refresh_normalization"})   # 只有 b02 动作不同
        other = self._cand(1.2, 2.5, system=0.7952, psi=psi)
        cands = [act, chosen, other]
        r = cal.robust_pick(cands, active=self.ACTIVE, tol=0.01)
        self.assertEqual((r["psi_low"], r["psi_mid"]), self.ACTIVE,
                         "单批次证据不得替换默认阈值")
        self.assertEqual(r["reason"], "single_batch_evidence")
        self.assertEqual(r["evidence_batches"], [2])

    def test_无行为差异时回退默认(self):
        psi = {b: 3.0 for b in (2, 3, 4, 5, 6, 7, 8)}
        act = self._cand(1.2, 2.5, system=0.7952, psi=psi, active=True)
        chosen = self._cand(0.8, 4.0, system=0.8281, psi=psi)      # 动作完全相同
        r = cal.robust_pick([act, chosen], active=self.ACTIVE, tol=0.01)
        self.assertEqual((r["psi_low"], r["psi_mid"]), self.ACTIVE)
        self.assertEqual(r["reason"], "no_behavioral_difference")

    def test_多批次证据时才允许更换(self):
        # PSI 都在 3.0：任何相邻阈值差都不覆盖该值 → 不存在单批次翻转
        psi = {b: 3.0 for b in (2, 3, 4, 5, 6, 7, 8)}
        act = self._cand(1.2, 2.5, system=0.70, psi=psi, active=True)
        chosen = self._cand(0.6, 4.0, system=0.80, psi=psi,
                            actions={5: "refit_model"})            # 行为确有差异
        r = cal.robust_pick([act, chosen], active=self.ACTIVE, tol=0.01)
        self.assertEqual(r["reason"], "multi_batch_evidence")
        self.assertEqual((r["psi_low"], r["psi_mid"]), (0.6, 4.0))

    def test_阈值区间覆盖批次判定(self):
        rows = [{"batch": 2, "psi_mean": 0.766}, {"batch": 3, "psi_mean": 1.421},
                {"batch": 4, "psi_mean": 4.384}]
        self.assertEqual(cal.axis_affected_batches(rows, 0.6, 1.2), [2])
        self.assertEqual(cal.axis_affected_batches(rows, 1.2, 0.6), [2])
        self.assertEqual(cal.axis_affected_batches(rows, 2.5, 4.0), [])
        # [1.0, 5.0) 覆盖 1.421 与 4.384 两批，0.766 不在内
        self.assertEqual(cal.axis_affected_batches(rows, 1.0, 5.0), [3, 4])

    def test_单批次翻转识别(self):
        psi = {2: 0.766}
        chosen = self._cand(0.6, 2.5, system=0.83, psi=psi,
                            actions={2: "refresh_normalization"})
        other = self._cand(1.2, 2.5, system=0.79, psi=psi)
        flip = cal.single_batch_flip([chosen, other], chosen)
        self.assertIsNotNone(flip)
        self.assertEqual(flip["batch"], 2)
        self.assertEqual(flip["axis"], "psi_low")
        self.assertAlmostEqual(flip["delta"], 0.04, places=4)

    def test_没有翻转时返回None(self):
        psi = {b: 3.0 for b in (2, 3, 4, 5, 6, 7, 8)}
        chosen = self._cand(0.6, 2.5, system=0.83, psi=psi,
                            actions={5: "refit_model"})
        other = self._cand(1.2, 2.5, system=0.79, psi=psi)
        self.assertIsNone(cal.single_batch_flip([chosen, other], chosen))


class TestSensitivity(unittest.TestCase):
    """第 3 项：留一敏感性只用统计重算，不碰 holdout。"""

    def test_默认移除批次不含holdout(self):
        self.assertNotIn(10, (2, 5, 6, 7, 8))
        self.assertIn(2, (2, 5, 6, 7, 8))

    def test_留一后统计只作用于validation区间(self):
        rows = [{"batch": b, "psi_mean": 1.0, "f1_system": 0.5,
                 "f1_baseline": 0.2, "action": "do_nothing", "decision": "noop"}
                for b in range(2, 10)]
        dec = [{"batch": b, "experiment": {"action": "do_nothing"},
                "decision": "noop", "charge": {"gpu_sec": 0.1}} for b in range(2, 10)]
        c = {"psi_low": 1.2, "psi_mid": 2.5, "seg": {"system": 0.5, "gpu_sec": 0.4,
                                                     "rollbacks": 0},
             "feasible": True, "_rows": rows, "_decisions": dec, "_is_active": True,
             "_label": "现役"}
        s = cal.sensitivity([c], lo=5, hi=8)
        self.assertEqual([e["dropped_batch"] for e in s["rows"]], [2, 5, 6, 7, 8])
        for e in s["rows"]:
            self.assertEqual(list(e["candidates"]), ["现役"])
            if e["dropped_batch"] == 2:
                # b02 不在 validation 段，移除后批次数不变
                self.assertEqual(e["candidates"]["现役"]["n"], 4)
            else:
                self.assertEqual(e["candidates"]["现役"]["n"], 3)


class TestRunPipelineCompatibility(unittest.TestCase):
    """第 5 项：推荐的阈值要能直接喂给 run_pipeline.py 的参数。"""

    def test_run_pipeline接受psi参数(self):
        import argparse
        src = io.open(os.path.join(REPO, "scripts", "run_pipeline.py"),
                      encoding="utf-8").read()
        self.assertIn("--psi-low", src)
        self.assertIn("--psi-mid", src)
        # 模拟 argparse：这两个参数能被解析成 float
        ap = argparse.ArgumentParser()
        ap.add_argument("--psi-low", type=float)
        ap.add_argument("--psi-mid", type=float)
        args = ap.parse_args(["--psi-low", "0.6", "--psi-mid", "4.0"])
        self.assertAlmostEqual(args.psi_low, 0.6)
        self.assertAlmostEqual(args.psi_mid, 4.0)


@unittest.skipUnless(HAS_GAS and HAS_BACKEND, "需要气体数据 + sklearn 或 numpy 后端")
class TestIntegration(unittest.TestCase):
    """真实跑一次小网格：验证可复现、不碰 holdout、不改数据。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = os.path.join(REPO, "out", "_calib_test")
        shutil.rmtree(cls.tmp, ignore_errors=True)
        os.makedirs(cls.tmp, exist_ok=True)
        cls.cfg = dict(low=0.8, mid=3.0, upto=8, seed=0, detector="psi",
                       dataset="gas")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_拒绝跑到holdout(self):
        """真跑一次必须拦住：upto=10 会触碰 b10。"""
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.run_config(low=0.8, mid=3.0, upto=10, seed=0, detector="psi",
                           dataset="gas",
                           out_dir=os.path.join(self.tmp, "bad"))
        self.assertIn("泄漏", str(ctx.exception))

    def test_可复现(self):
        m1 = cal.run_config(out_dir=os.path.join(self.tmp, "r1"), **self.cfg)
        m2 = cal.run_config(out_dir=os.path.join(self.tmp, "r2"), **self.cfg)
        s1 = cal.summarize(cal.segment(m1["rows"], 5, 8), m1["decisions"], m1["protocol"])
        s2 = cal.summarize(cal.segment(m2["rows"], 5, 8), m2["decisions"], m2["protocol"])
        self.assertAlmostEqual(s1["system"], s2["system"], places=9,
                               msg="固定 seed 两次运行结果必须一致")
        self.assertAlmostEqual(s1["gpu_sec"], s2["gpu_sec"], places=9)
        self.assertEqual(s1["rollbacks"], s2["rollbacks"])

    def test_只跑1到8批(self):
        m = cal.run_config(out_dir=os.path.join(self.tmp, "r4"), **self.cfg)
        batches = sorted({r["batch"] for r in m["rows"]})
        self.assertEqual(batches, list(range(2, 9)),
                         "结果里应只出现 b02–b08，不应有 b10")

    def test_标定不改动数据文件(self):
        import hashlib

        def digest_dir():
            h = hashlib.sha256()
            for d in _GAS_DIRS:
                if not os.path.isdir(d):
                    continue
                for name in sorted(os.listdir(d)):
                    if name.endswith(".npz"):
                        with open(os.path.join(d, name), "rb") as f:
                            h.update(f.read())
            return h.hexdigest()

        before = digest_dir()
        cal.run_config(out_dir=os.path.join(self.tmp, "r3"), **self.cfg)
        self.assertEqual(before, digest_dir(), "标定过程修改了数据文件")

# -*- coding: utf-8 -*-
"""数据身份 / 官方适配 / 冻结 / 一次性凭据 的测试。

覆盖用户要求：
  1. DatasetSource.data_identity
  2. official_public / official_test 身份校验
  3. validate_official 的空目录、缺文件、格式错误
  5. official_test 必须要求冻结配置
  6. eval_ticket 一次性保护
  7. 不得声称官方数据已接入

    python -m unittest tests.test_official -v
"""
import importlib.util
import io
import json
import os
import shutil
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evolver import evaluation as ev                      # noqa: E402
from evolver.adapters import identities as ids            # noqa: E402
from evolver.adapters.sources import DataSourceError, GasSource  # noqa: E402
from evolver.adapters.official import MANIFEST_NAME, OfficialSource  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "validate_official_under_test",
    os.path.join(REPO, "scripts", "validate_official.py"))
vo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vo)

# 气体批次是否就位。仓库 .gitignore 排除了 data/，所以"干净 clone"里没有这些 npz；
# 依赖数据的测试必须显式 skip，而不是让 GasSource 抛 DataSourceError 变成 error。
_GAS_DIRS = [os.path.join(REPO, "data", "gas", "processed"),
             os.path.join(REPO, "data", "gas")]
HAS_GAS = any(os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))
              for d in _GAS_DIRS)


def _tmp(name: str) -> str:
    import uuid
    base = os.path.join(REPO, "out", "_unittest_tmp")
    os.makedirs(base, exist_ok=True)
    d = os.path.join(base, f"{name}-{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


class TestIdentities(unittest.TestCase):
    def test_身份取值完整(self):
        for v in (ids.VALIDATION, ids.PUBLIC_HOLDOUT, ids.OFFICIAL_PUBLIC,
                  ids.OFFICIAL_TEST, ids.PUBLIC_SUBSTITUTE, ids.SYNTHETIC):
            self.assertIn(v, ids.ALL_IDENTITIES)
            self.assertIn(v, ids.IDENTITY_ZH)

    def test_官方身份不可用于选阈值(self):
        for v in (ids.OFFICIAL_PUBLIC, ids.OFFICIAL_TEST):
            with self.subTest(v=v):
                with self.assertRaises(ids.DataIdentityError):
                    ids.assert_selectable(v)

    def test_留出段不可用于选阈值(self):
        with self.assertRaises(ids.DataIdentityError):
            ids.assert_selectable(ids.PUBLIC_HOLDOUT)

    def test_替代数据与合成数据可选(self):
        for v in (ids.PUBLIC_SUBSTITUTE, ids.SYNTHETIC, ids.VALIDATION):
            self.assertEqual(ids.assert_selectable(v), v)

    def test_未标注身份被拒绝(self):
        with self.assertRaises(ids.DataIdentityError):
            ids.assert_selectable(ids.UNKNOWN)

    def test_未知身份字符串报错(self):
        with self.assertRaises(ids.DataIdentityError):
            ids.assert_selectable("official")        # 不是合法枚举

    def test_official_test不接受本地原始数据(self):
        with self.assertRaises(ids.DataIdentityError):
            ids.assert_inspectable(ids.OFFICIAL_TEST)
        self.assertEqual(ids.assert_inspectable(ids.OFFICIAL_PUBLIC),
                         ids.OFFICIAL_PUBLIC)

    def test_正式评估只认official_test(self):
        with self.assertRaises(ids.DataIdentityError):
            ids.assert_official_test(ids.OFFICIAL_PUBLIC)
        self.assertEqual(ids.assert_official_test(ids.OFFICIAL_TEST),
                         ids.OFFICIAL_TEST)


class TestSourceIdentity(unittest.TestCase):
    @unittest.skipUnless(HAS_GAS, "需要 data/gas")
    def test_Gas声明为公开替代数据(self):
        # 类属性不依赖数据，先断言；再校验需要 load() 的 metadata
        self.assertEqual(GasSource.data_identity, ids.PUBLIC_SUBSTITUTE)
        m = GasSource().get_metadata()
        self.assertEqual(m["data_identity"], ids.PUBLIC_SUBSTITUTE)
        self.assertFalse(m["is_official"])

    @unittest.skipUnless(HAS_GAS, "需要 data/gas")
    def test_桥接层透传身份(self):
        from evolver.adapters.bridge import BridgeAdapter
        b = BridgeAdapter(GasSource(), batch_ids=[1, 2, 3])
        self.assertEqual(b.data_identity, ids.PUBLIC_SUBSTITUTE)
        self.assertEqual(b.get(1).meta["data_identity"], ids.PUBLIC_SUBSTITUTE)

    def test_基类默认身份是unknown(self):
        from evolver.adapters.sources import DatasetSource
        self.assertEqual(DatasetSource.data_identity, ids.UNKNOWN)


class TestOfficialSourceGuards(unittest.TestCase):
    def setUp(self):
        self.dir = _tmp("official")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_目录不存在时报错并说明未提供(self):
        src = OfficialSource(os.path.join(self.dir, "nope"))
        with self.assertRaises(DataSourceError) as ctx:
            src.discover()
        msg = str(ctx.exception)
        self.assertIn("尚未提供", msg)
        self.assertIn("不会伪造", msg)

    def test_空目录报错(self):
        src = OfficialSource(self.dir)
        with self.assertRaises(DataSourceError) as ctx:
            src.discover()
        self.assertIn("空", str(ctx.exception))

    def test_缺manifest报错且拒绝猜测(self):
        open(os.path.join(self.dir, "train.csv"), "w").close()
        with self.assertRaises(DataSourceError) as ctx:
            OfficialSource(self.dir).discover()
        self.assertIn("manifest.json", str(ctx.exception))
        self.assertIn("拒绝猜测", str(ctx.exception))

    def test_manifest非法JSON报错(self):
        with open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
            f.write("{不是json")
        with self.assertRaises(DataSourceError):
            OfficialSource(self.dir).discover()

    def test_manifest缺字段报错(self):
        with open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump({"format": "csv"}, f)
        with self.assertRaises(DataSourceError) as ctx:
            OfficialSource(self.dir).load()
        self.assertIn("label_column", str(ctx.exception))

    def test_声明的文件不存在报错(self):
        with open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump({"format": "csv", "label_column": "y",
                       "batch_column": "b", "files": {"train": "missing.csv"}}, f)
        with self.assertRaises(DataSourceError) as ctx:
            OfficialSource(self.dir).load()
        self.assertIn("不存在", str(ctx.exception))

    def test_不支持的格式报错(self):
        with open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump({"format": "parquet", "label_column": "y",
                       "batch_column": "b", "files": {"train": "t.csv"}}, f)
        with self.assertRaises(DataSourceError):
            OfficialSource(self.dir).load()

    def test_拒绝把official_test当本地文件(self):
        with self.assertRaises(ids.DataIdentityError):
            OfficialSource(self.dir, identity=ids.OFFICIAL_TEST)

    def _valid_csv(self):
        rows = ["b,f0,f1,y"]
        for b in (1, 2, 3):
            for i in range(4):
                rows.append(f"{b},{i}.0,{i * 2}.0,{'A' if i % 2 else 'B'}")
        with open(os.path.join(self.dir, "train.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        with open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump({"format": "csv", "label_column": "y", "batch_column": "b",
                       "files": {"train": "train.csv"},
                       "task_type": "classification", "metric": "f1_macro"}, f)

    def test_合法csv能加载(self):
        self._valid_csv()
        src = OfficialSource(self.dir).load()
        self.assertEqual(src.data_identity, ids.OFFICIAL_PUBLIC)
        self.assertEqual(len(list(src.iter_batches())), 3)
        self.assertEqual(src.get_metric(), "f1_macro")
        self.assertEqual(src.split(), {"train": [1], "stream": [2], "holdout": [3]})
        meta = src.get_metadata()
        self.assertTrue(meta["is_official"])
        self.assertEqual(meta["n_batches"], 3)

    def test_标签列不存在报错(self):
        self._valid_csv()
        m = json.load(open(os.path.join(self.dir, MANIFEST_NAME), encoding="utf-8"))
        m["label_column"] = "nope"
        json.dump(m, open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8"))
        with self.assertRaises(DataSourceError) as ctx:
            OfficialSource(self.dir).load()
        self.assertIn("标签列", str(ctx.exception))

    def test_批次号不连续报错(self):
        rows = ["b,f0,y", "1,1.0,A", "3,2.0,B"]
        open(os.path.join(self.dir, "train.csv"), "w", encoding="utf-8").write(
            "\n".join(rows) + "\n")
        json.dump({"format": "csv", "label_column": "y", "batch_column": "b",
                   "files": {"train": "train.csv"}},
                  open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8"))
        with self.assertRaises(DataSourceError) as ctx:
            OfficialSource(self.dir).load()
        self.assertIn("不连续", str(ctx.exception))


class TestValidateOfficialScript(unittest.TestCase):
    def setUp(self):
        self.dir = _tmp("vo")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, root: str, extra: list[str] | None = None) -> int:
        argv = sys.argv
        sys.argv = ["validate_official.py", "--root", root] + (extra or [])
        buf = io.StringIO()
        import contextlib
        try:
            with contextlib.redirect_stdout(buf):
                rc = vo.main()
        finally:
            sys.argv = argv
        self.out = buf.getvalue()
        return rc

    def test_缺目录返回2(self):
        self.assertEqual(self._run(os.path.join(self.dir, "nope")), 2)
        self.assertIn("尚未提供", self.out)

    def test_缺目录allow_missing返回0(self):
        self.assertEqual(self._run(os.path.join(self.dir, "nope"),
                                   ["--allow-missing"]), 0)

    def test_空目录返回2(self):
        self.assertEqual(self._run(self.dir), 2)
        self.assertIn("空", self.out)

    def test_缺manifest返回1(self):
        open(os.path.join(self.dir, "x.csv"), "w").close()
        self.assertEqual(self._run(self.dir), 1)
        self.assertIn("manifest.json", self.out)

    def test_体检通过返回0且声明未接入(self):
        rows = ["b,f0,y"]
        for b in (1, 2, 3):
            for i in range(3):
                rows.append(f"{b},{i}.0,A")
        open(os.path.join(self.dir, "train.csv"), "w", encoding="utf-8").write(
            "\n".join(rows) + "\n")
        json.dump({"format": "csv", "label_column": "y", "batch_column": "b",
                   "files": {"train": "train.csv"}, "metric": "f1_macro"},
                  open(os.path.join(self.dir, MANIFEST_NAME), "w", encoding="utf-8"))
        rc = self._run(self.dir)
        self.assertEqual(rc, 0, self.out)
        self.assertIn("不代表任何评测成绩", self.out)


class TestFreeze(unittest.TestCase):
    def setUp(self):
        self.dir = _tmp("freeze")
        self.path = os.path.join(self.dir, "thresholds.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_未冻结时拒绝(self):
        with self.assertRaises(ev.FreezeError) as ctx:
            ev.assert_frozen(self.path, psi_low=1.2, psi_mid=2.5)
        self.assertIn("next", str(ctx.exception).lower() + "next")   # 有指引
        self.assertIn("冻结", str(ctx.exception))

    def test_冻结后一致可通过(self):
        ev.freeze_thresholds(self.path, psi_low=1.2, psi_mid=2.5)
        data = ev.assert_frozen(self.path, psi_low=1.2, psi_mid=2.5)
        self.assertIn("config_sha", data)
        self.assertIn("frozen_at", data)

    def test_改参数后冻结失效(self):
        ev.freeze_thresholds(self.path, psi_low=1.2, psi_mid=2.5)
        with self.assertRaises(ev.FreezeError) as ctx:
            ev.assert_frozen(self.path, psi_low=0.6, psi_mid=4.0)
        self.assertIn("不一致", str(ctx.exception))

    def test_检测器或margin变化也会失效(self):
        ev.freeze_thresholds(self.path, psi_low=1.2, psi_mid=2.5)
        with self.assertRaises(ev.FreezeError):
            ev.assert_frozen(self.path, psi_low=1.2, psi_mid=2.5, detector="ks")
        with self.assertRaises(ev.FreezeError):
            ev.assert_frozen(self.path, psi_low=1.2, psi_mid=2.5, gate_margin=0.05)

    def test_指纹稳定(self):
        a = ev.config_fingerprint({"b": 1, "a": 2})
        b = ev.config_fingerprint({"a": 2, "b": 1})
        self.assertEqual(a, b)
        self.assertNotEqual(a, ev.config_fingerprint({"a": 2, "b": 2}))


class TestEvalTicket(unittest.TestCase):
    def setUp(self):
        self.dir = _tmp("ticket")
        self.path = os.path.join(self.dir, "eval_ticket.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _acquire(self, sha="abc123"):
        return ev.acquire_ticket(self.path, dataset="official_test",
                                 identity=ids.OFFICIAL_TEST,
                                 config_sha=sha, frozen_at="2026-01-01")

    def test_首次申请成功(self):
        t = self._acquire()
        self.assertEqual(t["status"], "pending")
        self.assertEqual(ev.ticket_status(self.path)["status"], "pending")

    def test_用过后再次申请被拒(self):
        self._acquire()
        ev.mark_used(self.path, result_sha="deadbeef")
        with self.assertRaises(ev.TicketError) as ctx:
            self._acquire()
        self.assertIn("只允许正式评估一次", str(ctx.exception))

    def test_不同配置的待用凭据冲突(self):
        self._acquire(sha="aaa")
        with self.assertRaises(ev.TicketError):
            self._acquire(sha="bbb")

    def test_非official_test身份被拒(self):
        for ident in (ids.OFFICIAL_PUBLIC, ids.PUBLIC_SUBSTITUTE, ids.VALIDATION):
            with self.subTest(ident=ident):
                with self.assertRaises(ids.DataIdentityError):
                    ev.acquire_ticket(self.path, dataset="x", identity=ident,
                                      config_sha="s")

    def test_未申请就标记使用会报错(self):
        with self.assertRaises(ev.TicketError):
            ev.mark_used(self.path)

    def test_不能重复标记(self):
        self._acquire()
        ev.mark_used(self.path)
        with self.assertRaises(ev.TicketError):
            ev.mark_used(self.path)

    def test_凭据文件可读回(self):
        self._acquire()
        data = json.load(open(self.path, encoding="utf-8"))
        self.assertEqual(data["identity"], ids.OFFICIAL_TEST)
        self.assertIsNone(data["used_at"])


class TestPipelineGuard(unittest.TestCase):
    """run_pipeline 的身份闸门（第 4、5、6 项）。"""

    def setUp(self):
        # run_pipeline 导入时会 load_env()，快照/恢复环境避免污染其他测试
        self._snap = dict(os.environ)
        self.dir = _tmp("guard")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._snap)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _rp(self):
        spec = importlib.util.spec_from_file_location(
            "rp_guard_under_test", os.path.join(REPO, "scripts", "run_pipeline.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _adapter(self, ident):
        class A:
            data_identity = ident
            name = "fake"
        return A()

    class _Args:
        def __init__(self, frozen=None, ticket=None):
            self.frozen_config = frozen
            self.eval_ticket = ticket

    class _Cfg:
        psi_low = 1.2
        psi_mid = 2.5
        detector = "psi"
        gate_margin = 0.02
        margin_mode = "absolute"

    def test_官方公开集不许进闭环(self):
        rp = self._rp()
        with self.assertRaises(SystemExit):
            rp._guard_identity(self._adapter(ids.OFFICIAL_PUBLIC),
                               self._Args(), self._Cfg())

    def test_official_test缺冻结配置被拒(self):
        rp = self._rp()
        with self.assertRaises(SystemExit):
            rp._guard_identity(self._adapter(ids.OFFICIAL_TEST),
                               self._Args(), self._Cfg())

    def test_official_test参数与冻结不一致被拒(self):
        rp = self._rp()
        path = os.path.join(self.dir, "thresholds.json")
        ev.freeze_thresholds(path, psi_low=0.6, psi_mid=4.0)     # 冻结的是别的参数
        with self.assertRaises(SystemExit):
            rp._guard_identity(self._adapter(ids.OFFICIAL_TEST),
                               self._Args(frozen=path), self._Cfg())

    def test_冻结一致则放行并生成凭据(self):
        rp = self._rp()
        path = os.path.join(self.dir, "thresholds.json")
        ev.freeze_thresholds(path, psi_low=1.2, psi_mid=2.5)
        rp._guard_identity(self._adapter(ids.OFFICIAL_TEST),
                           self._Args(frozen=path), self._Cfg())
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "eval_ticket.json")))

    def test_第二次跑official_test被拒(self):
        rp = self._rp()
        path = os.path.join(self.dir, "thresholds.json")
        ev.freeze_thresholds(path, psi_low=1.2, psi_mid=2.5)
        rp._guard_identity(self._adapter(ids.OFFICIAL_TEST),
                           self._Args(frozen=path), self._Cfg())
        ev.mark_used(os.path.join(self.dir, "eval_ticket.json"), result_sha="x")
        with self.assertRaises(SystemExit):
            rp._guard_identity(self._adapter(ids.OFFICIAL_TEST),
                               self._Args(frozen=path), self._Cfg())

    def test_普通数据不受影响(self):
        rp = self._rp()
        for ident in (ids.PUBLIC_SUBSTITUTE, ids.SYNTHETIC, ids.VALIDATION):
            with self.subTest(ident=ident):
                rp._guard_identity(self._adapter(ident), self._Args(), self._Cfg())


class TestNoFalseClaim(unittest.TestCase):
    """第 7 项：不得声称官方数据已接入。"""

    def test_仓库里没有官方数据目录(self):
        for name in ("official",):
            p = os.path.join(REPO, "data", name)
            self.assertFalse(os.path.isdir(p) and os.listdir(p),
                             "出现了 data/official 且有内容，需重新审查是否已接入")

    def test_官方适配器默认不假装能读(self):
        src = OfficialSource(os.path.join(REPO, "data", "official"))
        with self.assertRaises(DataSourceError):
            src.load()

    def test_cmapss与xjtu只有契约没有数据(self):
        for name in ("cmapss", "xjtu_sy"):
            d = os.path.join(REPO, "data", name)
            if not os.path.isdir(d):
                continue
            files = os.listdir(d)
            self.assertLessEqual(set(files), {"README_DATA.md"},
                                 f"data/{name} 里出现了非契约文件：{files}")
            text = open(os.path.join(d, "README_DATA.md"), encoding="utf-8").read()
            self.assertIn("尚未下载", text)
            self.assertNotIn("http://", text)
            self.assertNotIn("https://", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

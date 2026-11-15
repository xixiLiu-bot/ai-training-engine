# -*- coding: utf-8 -*-
"""绘图是可选的：缺 matplotlib 只能是 warning，不能变成运行 error。

背景（最终验收前实测）：
    `requirements*.txt` 里声明了 matplotlib，但它只用于生成 `curve.png`，
    闭环本身不需要。原来绘图失败会被 append 进 `errors`，于是
    "没装 matplotlib" 会让一次完全正常的运行显示成"异常 1 条"。

    现在：绘图失败 → `warnings`（不进 errors），并在控制台明确说"跳过绘图"。

    python -m unittest tests.test_plot_optional -v
"""
import importlib.util
import io
import json
import os
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# 这些测试要跑一次完整合成闭环，需要 data/synthetic。
# 仓库 .gitignore 排除了 data/，干净 clone 里没有 → 必须 skip 而不是报错。
_SYN = os.path.join(REPO, "data", "synthetic")
HAS_SYNTHETIC = os.path.isdir(_SYN) and any(
    f.endswith(".npz") for f in os.listdir(_SYN))


def _tmp(name: str) -> str:
    import uuid
    base = os.path.join(REPO, "out", "_unittest_tmp")
    os.makedirs(base, exist_ok=True)
    d = os.path.join(base, f"{name}-{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


@unittest.skipUnless(HAS_SYNTHETIC, "需要 data/synthetic（先跑 scripts/make_data.py）")
class TestPlotIsOptional(unittest.TestCase):
    def setUp(self):
        # run_pipeline 导入时会 load_env()，快照环境避免污染其它测试
        self._env = dict(os.environ)
        self.dir = _tmp("plot")
        self.out = os.path.join(self.dir, "out")
        spec = importlib.util.spec_from_file_location(
            "rp_plot_under_test", os.path.join(REPO, "scripts", "run_pipeline.py"))
        self.rp = importlib.util.module_from_spec(spec)
        sys.modules["rp_plot_under_test"] = self.rp
        spec.loader.exec_module(self.rp)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self):
        import contextlib
        argv = sys.argv
        buf = io.StringIO()
        sys.argv = ["run_pipeline.py", "--dataset", "synthetic",
                    "--llm", "none", "--out", self.out, "--no-recorder"]
        try:
            with contextlib.redirect_stdout(buf):
                self.rp.main()
        finally:
            sys.argv = argv
        self.text = buf.getvalue()
        with open(os.path.join(self.out, "metrics.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_缺matplotlib只产生warning(self):
        # 模拟"没装 matplotlib"：plot 抛 ImportError
        def boom(*a, **k):
            raise ImportError("No module named 'matplotlib'")
        with mock.patch.object(self.rp, "plot", boom):
            m = self._run()
        self.assertEqual(m.get("errors"), [], "缺 matplotlib 不得算 error")
        warns = m.get("warnings") or []
        self.assertEqual(len(warns), 1, f"应恰好 1 条 warning，实际 {warns}")
        self.assertEqual(warns[0]["warning"], "matplotlib_missing")
        self.assertIn("跳过绘图", self.text)
        self.assertIn("警告", self.text)
        self.assertNotIn("异常", self.text)
        # 没画出来就不该出现在产出清单里
        self.assertNotIn("curve.png", self.text.split("产出")[-1])

    def test_绘图成功时没有warning且产出含curve(self):
        def fake_plot(rows, path, concept_start=None):
            with open(path, "wb") as f:
                f.write(b"\x89PNG fake")
        with mock.patch.object(self.rp, "plot", fake_plot):
            m = self._run()
        self.assertEqual(m.get("errors"), [])
        self.assertEqual(m.get("warnings"), [])
        self.assertIn("curve.png", self.text.split("产出")[-1])
        self.assertNotIn("警告", self.text)

    def test_绘图抛其它异常也只是warning(self):
        def boom(*a, **k):
            raise RuntimeError("字体渲染失败")
        with mock.patch.object(self.rp, "plot", boom):
            m = self._run()
        self.assertEqual(m.get("errors"), [])
        warns = m.get("warnings") or []
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0]["warning"], "plot_failed")

    def test_结论数字不受绘图影响(self):
        def boom(*a, **k):
            raise ImportError("no matplotlib")
        with mock.patch.object(self.rp, "plot", boom):
            m1 = self._run()
        with mock.patch.object(self.rp, "plot", lambda *a, **k: None):
            m2 = self._run()
        self.assertEqual(len(m1["rows"]), len(m2["rows"]))
        self.assertEqual(m1["ledger"]["used"], m2["ledger"]["used"])
        self.assertEqual(m1["protocol"], m2["protocol"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

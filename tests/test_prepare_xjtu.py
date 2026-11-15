# -*- coding: utf-8 -*-
"""prepare_xjtu.py 的输入校验。

背景（第 5 步实测）：
    输入目录存在但没有可用 CSV 时，脚本 exit 0 并写出 0 个批次 —— "静默成功"。
    修复后必须**非零退出**，且不得留下空的"成功产物"（npz / index.txt / 空目录）。

    python -m unittest tests.test_prepare_xjtu -v
"""
import importlib.util
import io
import os
import shutil
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    "prepare_xjtu_under_test", os.path.join(REPO, "scripts", "prepare_xjtu.py"))
px = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(px)

COND = "35Hz12kN"


def _tmp(name: str) -> str:
    import uuid
    base = os.path.join(REPO, "out", "_unittest_tmp")
    os.makedirs(base, exist_ok=True)
    d = os.path.join(base, f"{name}-{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


class TestPrepareXjtuGuards(unittest.TestCase):
    def setUp(self):
        self.dir = _tmp("xjtu")
        self.src = os.path.join(self.dir, "src")
        self.out = os.path.join(self.dir, "out")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, extra=None):
        argv = sys.argv
        sys.argv = ["prepare_xjtu.py", "--src", self.src, "--out", self.out,
                    "--stages", "3"] + (extra or [])
        buf = io.StringIO()
        self.exit_msg = ""
        import contextlib
        try:
            # SystemExit("消息") 的文本在异常对象里（被我们捕获后不会自动打印）
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                px.main()
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            self.exit_msg = e.code if isinstance(e.code, str) else ""
        finally:
            sys.argv = argv
        self.out_text = buf.getvalue()
        return code

    def _bearing(self, bear, n_files, cond=COND, bad=False):
        d = os.path.join(self.src, cond, bear)
        os.makedirs(d, exist_ok=True)
        for i in range(1, n_files + 1):
            p = os.path.join(d, f"{i}.csv")
            if bad:
                open(p, "w", encoding="utf-8").write("a,b\nx,y\n")
            else:
                sig = np.random.default_rng(i).normal(0, 1, (64, 2))
                np.savetxt(p, sig, delimiter=",", fmt="%.5f")
        return d

    # ---- 失败路径 ----
    def test_源目录不存在时非零退出(self):
        code = self._run()
        self.assertNotEqual(code, 0, "源目录不存在必须非零退出")
        self.assertIn("没找到", self.exit_msg)

    def test_目录存在但没有工况子目录时非零退出(self):
        os.makedirs(self.src, exist_ok=True)
        code = self._run()
        self.assertNotEqual(code, 0)
        self.assertIn("没有产出任何批次", self.out_text)
        self.assertFalse(os.path.exists(os.path.join(self.out, "index.txt")),
                         "不得生成空的 index.txt")
        self.assertFalse(os.path.isdir(self.out) and os.listdir(self.out),
                         "不得留下任何产物")

    def test_轴承目录里没有csv时非零退出(self):
        os.makedirs(os.path.join(self.src, COND, "Bearing1_1"), exist_ok=True)
        code = self._run()
        self.assertNotEqual(code, 0)
        self.assertIn("没有产出任何批次", self.out_text)

    def test_csv数量不足时非零退出(self):
        self._bearing("Bearing1_1", 3)          # stages=3 需要 >=6 个
        code = self._run()
        self.assertNotEqual(code, 0)
        self.assertIn("没有产出任何批次", self.out_text)

    def test_无效csv被跳过并最终非零退出(self):
        self._bearing("Bearing1_1", 8, bad=True)
        code = self._run()
        self.assertNotEqual(code, 0)
        self.assertIn("读取失败", self.out_text)
        self.assertIn("没有产出任何批次", self.out_text)

    def test_单列csv也被判为无效(self):
        d = os.path.join(self.src, COND, "Bearing1_1")
        os.makedirs(d, exist_ok=True)
        for i in range(1, 9):
            np.savetxt(os.path.join(d, f"{i}.csv"),
                       np.random.default_rng(i).normal(0, 1, (64, 1)),
                       delimiter=",", fmt="%.5f")
        code = self._run()
        self.assertNotEqual(code, 0)
        self.assertIn("两列数值", self.out_text)

    # ---- 成功路径 ----
    def test_合法输入正常产出(self):
        self._bearing("Bearing1_1", 8)
        code = self._run()
        self.assertEqual(code, 0, self.out_text)
        npz = os.path.join(self.out, "batch_01.npz")
        self.assertTrue(os.path.isfile(npz))
        d = np.load(npz)
        self.assertEqual(d["X"].shape[0], 8)
        self.assertEqual(len(np.unique(d["y"])), 3)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "index.txt")))

    def test_一个坏一个好后仍能产出且提示坏的那个(self):
        self._bearing("Bearing1_1", 8)
        self._bearing("Bearing1_2", 8, bad=True)
        code = self._run()
        self.assertEqual(code, 0, self.out_text)
        self.assertIn("读取失败", self.out_text)
        files = [f for f in os.listdir(self.out) if f.endswith(".npz")]
        self.assertEqual(len(files), 1, "只应有 1 个成功批次")


if __name__ == "__main__":
    unittest.main(verbosity=2)

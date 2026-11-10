# -*- coding: utf-8 -*-
"""`.env` 加载器测试（整改：让配置文件真正生效）。

重点：
  1. `.env` 不存在时**静默跳过**，不报错
  2. 已存在的环境变量**优先**（不被 .env 覆盖）
  3. 返回摘要里**只有 key 名，没有 value**（防止密钥被顺手打印）
  4. 各种写法都能解析：export / 引号 / 行尾注释 / 空行 / 非法行

    python -m unittest tests.test_envfile -v
"""
import os
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evolver.envfile import load_env, parse_text, summary  # noqa: E402


class TestParse(unittest.TestCase):
    def test_基本键值(self):
        self.assertEqual(parse_text("A=1\nB=hello\n"), [("A", "1"), ("B", "hello")])

    def test_忽略注释与空行(self):
        text = "# 注释\n\nA=1\n   \n# B=2\nC=3\n"
        self.assertEqual(parse_text(text), [("A", "1"), ("C", "3")])

    def test_export前缀(self):
        self.assertEqual(parse_text("export A=1\n"), [("A", "1")])

    def test_引号与行尾注释(self):
        text = 'A="hello world"\nB=\'x,y\'\nC=value  # 尾注\nD="keep # inside"\n'
        got = dict(parse_text(text))
        self.assertEqual(got["A"], "hello world")
        self.assertEqual(got["B"], "x,y")
        self.assertEqual(got["C"], "value")
        self.assertEqual(got["D"], "keep # inside")

    def test_值里含等号(self):
        self.assertEqual(dict(parse_text("A=a=b\n"))["A"], "a=b")

    def test_空值合法(self):
        self.assertEqual(dict(parse_text("A=\n"))["A"], "")

    def test_非法行被跳过(self):
        text = "没有等号\n=空key\nA=1\n"
        self.assertEqual(parse_text(text), [("A", "1")])


class TestLoad(unittest.TestCase):
    def setUp(self):
        import uuid
        self.dir = os.path.join(REPO, "out", "_unittest_tmp",
                                f"env-{uuid.uuid4().hex[:8]}")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, ".env")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, text: str):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_文件不存在时静默跳过(self):
        res = load_env(os.path.join(self.dir, "nope.env"))
        self.assertTrue(res["missing"])
        self.assertEqual(res["applied"], [])
        self.assertIn("未找到", summary(res))

    def test_正常载入(self):
        self._write("ZS_TEST_A=1\nZS_TEST_B=2\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZS_TEST_A", None)
            os.environ.pop("ZS_TEST_B", None)
            res = load_env(self.path)
            self.assertEqual(sorted(res["applied"]), ["ZS_TEST_A", "ZS_TEST_B"])
            self.assertEqual(os.environ["ZS_TEST_A"], "1")
        os.environ.pop("ZS_TEST_A", None)
        os.environ.pop("ZS_TEST_B", None)

    def test_已有环境变量优先(self):
        self._write("ZS_TEST_C=from_file\n")
        with mock.patch.dict(os.environ, {"ZS_TEST_C": "from_shell"}, clear=False):
            res = load_env(self.path)
            self.assertEqual(os.environ["ZS_TEST_C"], "from_shell")
            self.assertIn("ZS_TEST_C", res["kept"])
            self.assertNotIn("ZS_TEST_C", res["applied"])

    def test_override可强制覆盖(self):
        self._write("ZS_TEST_D=from_file\n")
        with mock.patch.dict(os.environ, {"ZS_TEST_D": "from_shell"}, clear=False):
            load_env(self.path, override=True)
            self.assertEqual(os.environ["ZS_TEST_D"], "from_file")
        os.environ.pop("ZS_TEST_D", None)

    def test_返回摘要不含value(self):
        secret = "swanlab-secret-value-xyz"
        self._write(f"ZS_TEST_SECRET={secret}\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZS_TEST_SECRET", None)
            res = load_env(self.path)
            blob = repr(res) + summary(res)
            self.assertNotIn(secret, blob, "返回值/摘要里泄漏了值")
            self.assertIn("ZS_TEST_SECRET", res["applied"])
        os.environ.pop("ZS_TEST_SECRET", None)

    def test_带BOM也能读(self):
        with open(self.path, "w", encoding="utf-8-sig") as f:
            f.write("ZS_TEST_E=1\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZS_TEST_E", None)
            load_env(self.path)
            self.assertEqual(os.environ.get("ZS_TEST_E"), "1")
        os.environ.pop("ZS_TEST_E", None)


class TestRepoDefaults(unittest.TestCase):
    def test_默认路径指向项目根目录(self):
        from evolver.envfile import repo_root
        self.assertTrue(os.path.isfile(os.path.join(repo_root(), "README.md")))
        self.assertTrue(os.path.isdir(os.path.join(repo_root(), "evolver")))

    def test_仓库里有env示例文件(self):
        from evolver.envfile import repo_root
        example = os.path.join(repo_root(), ".env.example")
        self.assertTrue(os.path.isfile(example), "缺少 .env.example 模板")
        text = open(example, encoding="utf-8").read()
        self.assertIn("SWANLAB_ENABLED", text)
        # 示例文件里不能出现真实 key 的样子
        self.assertNotIn("trio_", text)
        self.assertNotIn("sk-", text)

    def test_env示例的值都带双引号(self):
        """回归保护：swanlab 用 pydantic-settings 解析同一份 .env，
        裸值会报 `SettingsError: error parsing value for field "project"`。
        所以示例里的值必须统一用双引号包裹。"""
        from evolver.envfile import repo_root
        example = os.path.join(repo_root(), ".env.example")
        checked = 0
        for raw in open(example, encoding="utf-8"):
            line = raw.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if not key.startswith("SWANLAB_"):
                continue
            value = value.strip()
            self.assertTrue(
                len(value) >= 2 and value[0] == '"' and value[-1] == '"',
                f'{key} 的值必须用双引号包裹，当前是：{value!r}')
            checked += 1
        self.assertGreater(checked, 0, "示例里没有找到 SWANLAB_ 配置项")

    def test_带引号的值被正确剥离(self):
        """双引号是给第三方解析器看的；本项目读到时应是干净值。"""
        self.assertEqual(dict(parse_text('A="zhixun-engine"\n'))["A"], "zhixun-engine")
        self.assertEqual(dict(parse_text('A=""\n'))["A"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

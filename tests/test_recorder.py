# -*- coding: utf-8 -*-
"""实验记录 / SwanLab 集成测试。

重点验证三件事：
  1. 云端不可用时**必须如实报阻塞**，不能假装成功
  2. API Key **绝不能**出现在本地记录里
  3. 上传字段白名单生效（嵌套数据、标签数组、路径不会被上传）

    python -m unittest tests.test_recorder -v
"""
import json
import os
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evolver.logs.recorder import (ALLOWED_MODES, UPLOAD_KEYS, Recorder,
                                  _metric_only, _scalar_only, normalize_mode,
                                  unsafe_dotenv_hits)  # noqa: E402


def _tmpdir(name: str) -> str:
    import uuid
    base = os.path.join(REPO, "out", "_unittest_tmp")
    os.makedirs(base, exist_ok=True)
    d = os.path.join(base, f"{name}-{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


class TestLocalOnly(unittest.TestCase):
    def setUp(self):
        self.dir = _tmpdir("rec")
        self.path = os.path.join(self.dir, "run.jsonl")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_未启用时不尝试上传(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWANLAB_ENABLED", None)
            r = Recorder(self.path)
        self.assertFalse(r.status()["swanlab_enabled"])
        self.assertIn("disabled", r.status()["swanlab_cloud"])
        self.assertFalse(r.status()["verified_on_web"])

    def test_本地记录完整(self):
        r = Recorder(self.path)
        r.log({"stage": "batch", "batch_id": 1, "f1_system": 0.5})
        r.finish({"avg_system": 0.5})
        self.assertTrue(r.write_ok)
        lines = [json.loads(x) for x in open(self.path, encoding="utf-8") if x.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["batch_id"], 1)
        self.assertEqual(lines[1]["stage"], "summary")

    def test_不可写目录不抛异常(self):
        bad = os.path.join(REPO, "out", "no_such\x00dir", "run.jsonl")
        r = Recorder(bad)
        self.assertFalse(r.write_ok)
        r.log({"stage": "batch"})
        r.finish({"avg_system": 0.1})
        self.assertGreaterEqual(len(r.records), 2)


class TestCloudBlocked(unittest.TestCase):
    """缺 SDK / 缺 Key 时，必须报 blocked 而不是 uploading。"""

    def setUp(self):
        self.dir = _tmpdir("recblk")
        self.path = os.path.join(self.dir, "run.jsonl")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_缺SDK时报阻塞(self):
        env = {"SWANLAB_ENABLED": "true", "SWANLAB_API_KEY": "dummy-key-for-test"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.dict(sys.modules, {"swanlab": None}):
                r = Recorder(self.path)
        st = r.status()
        self.assertTrue(st["swanlab_enabled"])
        self.assertIn("blocked", st["swanlab_cloud"])
        self.assertTrue(st["swanlab_error"])
        self.assertFalse(st["verified_on_web"], "没有人工确认就不能声称已验证")

    def test_缺Key时报阻塞且提示获取方式(self):
        if "swanlab" not in sys.modules:
            try:
                import swanlab  # noqa: F401
            except Exception:                       # noqa: BLE001
                self.skipTest("swanlab 未安装，无法单独测'缺 key'分支")
        with mock.patch.dict(os.environ, {"SWANLAB_ENABLED": "true"}, clear=False):
            os.environ.pop("SWANLAB_API_KEY", None)
            r = Recorder(self.path)
        st = r.status()
        self.assertIn("blocked", st["swanlab_cloud"])
        self.assertIn("SWANLAB_API_KEY", st["swanlab_error"])

    def test_项目名与实验名从环境变量读(self):
        # 用 *_NAME（swanlab 原生写法）。注意：其它测试可能已经 import 了
        # run_pipeline，它启动时会 load_env() 把 .env 写进 os.environ，
        # 所以这里显式覆盖两个变量，避免受环境残留影响。
        env = {"SWANLAB_ENABLED": "true", "SWANLAB_PROJECT_NAME": "proj-x",
               "SWANLAB_EXPERIMENT_NAME": "exp-y", "SWANLAB_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.dict(sys.modules, {"swanlab": None}):
                r = Recorder(self.path)
        st = r.status()
        self.assertEqual(st["swanlab_project"], "proj-x")
        self.assertEqual(st["swanlab_experiment"], "exp-y")

    def test_兼容旧的PROJECT变量名(self):
        """旧写法 SWANLAB_PROJECT 仍可用（仅在未设 *_NAME 时生效）。"""
        saved = {k: os.environ.pop(k, None)
                 for k in ("SWANLAB_PROJECT_NAME", "SWANLAB_EXPERIMENT_NAME")}
        try:
            env = {"SWANLAB_ENABLED": "true", "SWANLAB_PROJECT": "legacy-p",
                   "SWANLAB_EXPERIMENT": "legacy-e", "SWANLAB_API_KEY": "k"}
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch.dict(sys.modules, {"swanlab": None}):
                    r = Recorder(self.path)
            st = r.status()
            self.assertEqual(st["swanlab_project"], "legacy-p")
            self.assertEqual(st["swanlab_experiment"], "legacy-e")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class TestSecretHygiene(unittest.TestCase):
    """密钥绝不能落盘。"""

    def setUp(self):
        self.dir = _tmpdir("recsec")
        self.path = os.path.join(self.dir, "run.jsonl")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_api_key不出现在记录里(self):
        secret = "sk-super-secret-abc123456789"
        env = {"SWANLAB_ENABLED": "true", "SWANLAB_API_KEY": secret}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.dict(sys.modules, {"swanlab": None}):
                r = Recorder(self.path, config={"dataset": "gas", "api_key": secret})
                r.log({"stage": "batch", "batch_id": 1, "f1_system": 0.5})
                r.finish({"avg_system": 0.5})
        text = open(self.path, encoding="utf-8").read()
        self.assertNotIn(secret, text, "API Key 泄漏到本地记录")
        self.assertNotIn("api_key", text.lower())

    def test_配置里的敏感字段不会被上传(self):
        # config 里即便混进 api_key，_scalar_only 也会因为不在白名单而丢掉
        cfg = {"dataset": "gas", "api_key": "xxx", "token": "yyy",
               "task_type": "classification"}
        out = _scalar_only(cfg)
        self.assertIn("dataset", out)
        self.assertIn("task_type", out)
        self.assertNotIn("api_key", out)
        self.assertNotIn("token", out)


class TestSwanlabCompat(unittest.TestCase):
    """针对 swanlab 0.10.x 真实约束的回归测试（读源码确认的行为）。"""

    def test_mode只允许四个值(self):
        self.assertEqual(ALLOWED_MODES, {"disabled", "online", "local", "offline"})

    def test_cloud被归一化为online(self):
        self.assertEqual(normalize_mode("cloud"), "online")
        self.assertEqual(normalize_mode("CLOUD"), "online")
        self.assertEqual(normalize_mode("remote"), "online")

    def test_非法mode退回online(self):
        self.assertEqual(normalize_mode("whatever"), "online")
        self.assertEqual(normalize_mode(None), "online")
        self.assertEqual(normalize_mode(""), "online")

    def test_合法mode原样保留(self):
        for m in ("online", "local", "offline", "disabled"):
            self.assertEqual(normalize_mode(m), m)

    def test_检测到裸PROJECT变量名(self):
        import uuid
        base = os.path.join(REPO, "out", "_unittest_tmp")
        os.makedirs(base, exist_ok=True)
        d = os.path.join(base, f"env-{uuid.uuid4().hex[:8]}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, ".env")
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write('SWANLAB_PROJECT="x"\nSWANLAB_EXPERIMENT="y"\n')
            hits = unsafe_dotenv_hits(p)
            self.assertEqual(len(hits), 2)
            self.assertTrue(any("SWANLAB_PROJECT_NAME" in h for h in hits))
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_检测到非法mode(self):
        import uuid
        base = os.path.join(REPO, "out", "_unittest_tmp")
        os.makedirs(base, exist_ok=True)
        d = os.path.join(base, f"env-{uuid.uuid4().hex[:8]}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, ".env")
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write('SWANLAB_MODE="cloud"\n')
            hits = unsafe_dotenv_hits(p)
            self.assertEqual(len(hits), 1)
            self.assertIn("online", hits[0])
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_合规env不报警(self):
        import uuid
        base = os.path.join(REPO, "out", "_unittest_tmp")
        os.makedirs(base, exist_ok=True)
        d = os.path.join(base, f"env-{uuid.uuid4().hex[:8]}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, ".env")
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write('SWANLAB_ENABLED="true"\n'
                        'SWANLAB_PROJECT_NAME="zhixun-engine"\n'
                        'SWANLAB_EXPERIMENT_NAME="gas"\n'
                        'SWANLAB_MODE="online"\n')
            self.assertEqual(unsafe_dotenv_hits(p), [])
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_仓库示例文件必须合规(self):
        hits = unsafe_dotenv_hits(os.path.join(REPO, ".env.example"))
        self.assertEqual(hits, [], f".env.example 里有不兼容变量：{hits}")


class _FakeSwanlab:
    """假 swanlab 模块：记录 init/log/finish 的调用参数。"""

    def __init__(self):
        self.init_calls: list[dict] = []
        self.log_calls: list[dict] = []
        self.finish_calls = 0
        self.log_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self

    def log(self, **kwargs):
        self.log_calls.append(kwargs)

    def finish(self):
        self.finish_calls += 1


class TestSwanlabUpload(unittest.TestCase):
    """用假 swanlab 模块验证真实上传时的调用形态（无需联网、无需 Key）。"""

    def setUp(self):
        import uuid
        base = os.path.join(REPO, "out", "_unittest_tmp")
        os.makedirs(base, exist_ok=True)
        self.dir = os.path.join(base, f"sw-{uuid.uuid4().hex[:8]}")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "run.jsonl")
        self.fake = _FakeSwanlab()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _make(self, **cfg):
        env = {"SWANLAB_ENABLED": "true", "SWANLAB_API_KEY": "test-key",
               "SWANLAB_PROJECT_NAME": "zhixun-engine",
               "SWANLAB_EXPERIMENT_NAME": "gas-validation-v2",
               "SWANLAB_MODE": "online"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.dict(sys.modules, {"swanlab": self.fake}):
                return Recorder(self.path, config=cfg or {"dataset": "gas"})

    def test_init参数与码表(self):
        self._make(dataset="gas_sensor_drift", detector="psi")
        kw = self.fake.init_calls[0]
        self.assertEqual(kw["project"], "zhixun-engine")
        self.assertEqual(kw["experiment_name"], "gas-validation-v2")
        self.assertEqual(kw["mode"], "online")
        cfg = kw["config"]
        self.assertEqual(cfg["dataset"], "gas_sensor_drift")
        # 码表必须完整
        book = cfg["categorical_codebook"]
        for key in ("action_code", "accepted_action_code", "release_decision_code"):
            self.assertIn(key, book)

    def test_step使用batch_id(self):
        r = self._make()
        r.log({"stage": "batch", "batch_id": 3, "psi_mean": 1.42,
               "action": "refit_model", "release_decision": "accept"})
        r.log({"stage": "batch", "batch_id": 10, "psi_mean": 1.35,
               "action": "do_nothing", "release_decision": "noop"})
        steps = [c["step"] for c in self.fake.log_calls]
        self.assertEqual(steps, [3, 10], "横轴必须是真实批次号")

    def test_载荷里没有batch_id与drift_score(self):
        r = self._make()
        r.log({"stage": "batch", "batch_id": 4, "batch": 4,
               "drift_score": 2.3, "psi_mean": 2.3, "f1_system": 0.7})
        data = self.fake.log_calls[0]["data"]
        self.assertNotIn("batch_id", data)
        self.assertNotIn("batch", data)
        self.assertNotIn("drift_score", data, "重复字段不应再上传")
        self.assertIn("psi_mean", data)

    def test_载荷全部是数值(self):
        r = self._make()
        r.log({"stage": "batch", "batch_id": 5, "dataset": "gas",
               "task_type": "classification", "metric": "f1_macro",
               "detector": "psi", "margin_mode": "absolute",
               "psi_mean": 2.377, "action": "refresh_normalization",
               "accepted_action": "refresh_normalization",
               "release_decision": "accept", "f1_system": 0.98,
               "raw_gain": 0.1, "normalized_gain": 0.1, "latency": 0.02,
               "token_usage": 0, "cost_sec": 0.05})
        data = self.fake.log_calls[0]["data"]
        for k, v in data.items():
            self.assertIsInstance(v, float, f"{k} 不是数值：{v!r}")
            # 字符串绝不允许出现（SwanLab 会报 Unsupported scalar string value）
            self.assertNotIsInstance(v, str)
        self.assertEqual(data["action_code"], 2.0)
        self.assertEqual(data["accepted_action_code"], 2.0)
        self.assertEqual(data["release_decision_code"], 1.0)

    def test_summary无batch_id时不传step(self):
        r = self._make()
        r.finish({"avg_system": 0.716, "n_acc": 4})
        last = self.fake.log_calls[-1]
        self.assertNotIn("step", last, "没有批次号时不应硬塞 step")

    def test_finish被调用且状态为uploaded(self):
        r = self._make()
        r.log({"stage": "batch", "batch_id": 2, "f1_system": 0.6})
        r.finish({"avg_system": 0.716})
        self.assertEqual(self.fake.finish_calls, 1)
        st = r.status()
        self.assertTrue(st["swanlab_enabled"])
        self.assertIn("uploaded", st["swanlab_cloud"])
        self.assertFalse(st["verified_on_web"], "网页可见性只能人工确认")

    def test_本地日志保留字符串字段(self):
        r = self._make()
        r.log({"stage": "batch", "batch_id": 3, "action": "refit_model",
               "accepted_action": "refit_model", "release_decision": "accept",
               "dataset": "gas_sensor_drift", "drift_score": 1.42})
        line = json.loads(open(self.path, encoding="utf-8").read().strip())
        # 本地必须完整保留字符串与重复字段（只在云端做裁剪）
        self.assertEqual(line["action"], "refit_model")
        self.assertEqual(line["accepted_action"], "refit_model")
        self.assertEqual(line["release_decision"], "accept")
        self.assertEqual(line["dataset"], "gas_sensor_drift")
        self.assertEqual(line["drift_score"], 1.42)
        self.assertEqual(line["batch_id"], 3)


class TestUploadWhitelist(unittest.TestCase):
    def test_白名单剔除嵌套与数组(self):
        rec = {
            "batch_id": 3,
            "drift_score": 1.42,
            "action": "refresh_normalization",
            "labels": [0, 1, 1, 0],          # 标签数组绝不能上传
            "features": [[1.0, 2.0]],
            "metadata": {"secret": "x"},
            "api_key": "k",
        }
        out = _scalar_only(rec)
        for k in ("batch_id", "drift_score"):
            self.assertIn(k, out)
        for k in ("labels", "features", "metadata", "api_key"):
            self.assertNotIn(k, out, f"{k} 不应出现在上传载荷里")


class TestMetricPayload(unittest.TestCase):
    """SwanLab 的 log() 只吃数值标量；字符串必须编码或丢弃，否则报错刷屏。"""

    def test_字符串被丢弃而不是上传(self):
        from evolver.logs.recorder import _metric_only
        out = _metric_only({"dataset": "gas_sensor_drift", "task_type": "classification",
                            "stage": "batch", "batch_id": 3, "f1_system": 0.7})
        # batch_id 只作横轴 step，不上曲线；字符串一律丢弃
        self.assertEqual(set(out), {"f1_system"})
        for v in out.values():
            self.assertIsInstance(v, float)

    def test_分类字段被编码成数字(self):
        from evolver.logs.recorder import _metric_only
        out = _metric_only({"action": "refresh_normalization",
                            "accepted_action": "refit_model",
                            "release_decision": "rollback",
                            "batch_id": 4})
        self.assertEqual(out["action_code"], 2.0)
        self.assertEqual(out["accepted_action_code"], 3.0)
        self.assertEqual(out["release_decision_code"], 2.0)
        # 原始字符串字段不能出现
        for k in ("action", "accepted_action", "release_decision"):
            self.assertNotIn(k, out)

    def test_未采纳时accepted_action被跳过(self):
        from evolver.logs.recorder import _metric_only
        out = _metric_only({"action": "do_nothing", "accepted_action": None})
        self.assertEqual(out, {"action_code": 0.0})

    def test_未知分类值被跳过(self):
        from evolver.logs.recorder import _metric_only
        out = _metric_only({"action": "some_new_action"})
        self.assertEqual(out, {})

    def test_码表覆盖全部动作(self):
        from evolver.logs.recorder import ACTION_CODES, CODEBOOK
        from evolver.loop.decide import ACTIONS
        self.assertEqual(set(ACTION_CODES), set(ACTIONS))
        for code, name in CODEBOOK["action_code"].items():
            if code.isdigit():
                self.assertIn(name, ACTIONS)

    def test_必需字段都在白名单里(self):
        need = ["dataset", "task_type", "detector", "margin_mode", "batch_id",
                "drift_score", "action", "accepted_action", "baseline_score",
                "candidate_score", "raw_gain", "normalized_gain",
                "release_decision", "latency", "token_usage",
                "avg_baseline", "avg_system", "n_acc", "n_roll",
                "gpu_sec", "labels", "tokens"]
        missing = [k for k in need if k not in UPLOAD_KEYS]
        self.assertEqual(missing, [], f"这些必需字段不在上传白名单：{missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

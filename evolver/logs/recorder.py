"""实验记录：SwanLab 云端 + 本地 JSONL 双层。

设计目标
--------
1. **云端优先**：SwanLab 可用时上传到网页端，能在浏览器里看曲线
2. **本地兜底**：SwanLab 不可用（未装/未配 key/断网）时，本地 JSONL 照样完整
3. **绝不假装成功**：任何一步失败都记录成 `swanlab_error`，并在 `status()` 里如实报告
4. **密钥零泄漏**：API Key 只从环境变量读，**不写入任何记录、不打印、不回显**

环境变量
--------
    SWANLAB_ENABLED     true/1/yes/on 才启用云端上传（默认关闭）
    SWANLAB_PROJECT     项目名，默认 zhixun-engine
    SWANLAB_EXPERIMENT  实验名，默认 run-<时间戳>
    SWANLAB_API_KEY     **只从环境变量读取**，不落盘
    SWANLAB_MODE        覆盖模式：cloud（默认）/ local / disabled

上传字段白名单见 `UPLOAD_KEYS`：只有明确列出的标量字段会上传，
杜绝把嵌套的私有数据、标签数组或密钥顺带传上去。
"""
from __future__ import annotations

import json
import os
import time

# 允许上传到 SwanLab 的字段（白名单）。不在名单里的一律不上传。
UPLOAD_KEYS = {
    # 实验配置（首批写入 config）
    "dataset", "task_type", "metric", "detector", "margin_mode",
    "psi_low", "psi_mid", "gate_margin", "gate_window", "gate_bootstrap",
    # 逐批指标
    "batch_id", "drift_score", "action", "accepted_action",
    "baseline_score", "candidate_score", "raw_gain", "normalized_gain",
    "release_decision", "latency", "token_usage", "cost_sec",
    "f1_baseline", "f1_system", "psi_mean", "ks_mean", "wasserstein_mean",
    # 汇总
    "avg_baseline", "avg_system", "n_acc", "n_roll", "n_act",
    "gpu_sec", "labels", "tokens", "errors",
}

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}

# SwanLab 0.10.x 的 ModeType 只接受这四个值（源码 swanlab/sdk/typings/run/__init__.py）
ALLOWED_MODES = {"disabled", "online", "local", "offline"}
MODE_ALIASES = {
    "cloud": "online",      # 常见误写：本项目早期文档也用过 cloud
    "remote": "online",
    "local": "local",
    "online": "online",
    "offline": "offline",
    "disabled": "disabled",
    "off": "disabled",
    "none": "disabled",
}

# 这些变量名会与 SwanLab 内部 Settings 的**嵌套模型**冲突：
#   Settings.project    是 ProjectSettings（子字段 name）
#   Settings.experiment 是 ExperimentSettings（子字段 name）
# 且 model_config 里 env_nested_delimiter="_"，所以
#   SWANLAB_PROJECT=xxx  → pydantic 试图把 "xxx" 当 JSON 解析嵌套模型 → SettingsError
# 正确写法：SWANLAB_PROJECT_NAME / SWANLAB_EXPERIMENT_NAME
UNSAFE_ENV_KEYS = ("SWANLAB_PROJECT", "SWANLAB_EXPERIMENT", "SWANLAB_WORKSPACE")


def normalize_mode(raw: str | None) -> str:
    """把用户写法收敛到 SwanLab 允许的枚举；非法值退回 online。"""
    m = (raw or "").strip().lower()
    if not m:
        return "online"
    return MODE_ALIASES.get(m, m if m in ALLOWED_MODES else "online")


def unsafe_dotenv_hits(path: str = ".env") -> list[str]:
    """扫描当前目录 .env，找出会让 swanlab 解析失败的变量名。

    SwanLab 自己也会读这个文件（env_file=".env"），所以哪怕我们用显式参数传值，
    文件里的裸 SWANLAB_PROJECT 仍然会让 init 抛 SettingsError。
    """
    import os as _os
    if not _os.path.isfile(path):
        return []
    hits: list[str] = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in UNSAFE_ENV_KEYS:
                    hits.append(f"{key}（应改为 {key}_NAME）")
                elif key == "SWANLAB_MODE" and value.lower() not in ALLOWED_MODES:
                    hits.append(f'SWANLAB_MODE="{value}"（只允许 disabled/online/local/offline）')
    except Exception:                           # noqa: BLE001
        return []
    return hits


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


def _scalar_only(record: dict) -> dict:
    """只保留白名单内的标量字段（config 用，允许字符串）。"""
    out = {}
    for k, v in record.items():
        if k not in UPLOAD_KEYS:
            continue
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float, str)):
            out[k] = v
        elif v is None:
            continue
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    return out


# ─────────────────────────────────────────────────────────────
# 逐批指标：SwanLab 的 log() 只接受**数值标量**，字符串会报
#   TypeError: Unsupported scalar string value: 'do_nothing'
# 所以分类字段一律编码成整数，并把码表写进 init(config=...) 供网页端对照。
# ─────────────────────────────────────────────────────────────
ACTION_CODES = {
    "do_nothing": 0,
    "calibrate_output": 1,
    "refresh_normalization": 2,
    "refit_model": 3,
}
DECISION_CODES = {
    "noop": 0,
    "accept": 1,
    "rollback": 2,
    "rollback_to": 3,
}
CATEGORICAL = {
    "action": ACTION_CODES,
    "accepted_action": ACTION_CODES,
    "release_decision": DECISION_CODES,
}

# 写进 config，网页端"配置"面板能看到每个数字代表什么
CODEBOOK = {
    "action_code": {"0": "do_nothing", "1": "calibrate_output",
                    "2": "refresh_normalization", "3": "refit_model"},
    "accepted_action_code": {"0": "do_nothing", "1": "calibrate_output",
                             "2": "refresh_normalization", "3": "refit_model",
                             "none": "本批未采纳（回滚/不动）"},
    "release_decision_code": {"0": "noop", "1": "accept", "2": "rollback"},
}


# 只作为横轴（step）使用，不作为指标曲线上传
STEP_FIELDS = ("batch_id", "batch")
# 与 psi_mean 完全重复的别名字段，不再上传（本地 jsonl 仍保留）
DUPLICATE_FIELDS = ("drift_score",)


def _metric_only(record: dict) -> dict:
    """转成 SwanLab 能接受的**纯数值**载荷。

    - 数值字段原样保留
    - 分类字段转成 `<name>_code` 整数
    - `batch_id` / `batch` 用作横轴 step，**不作为曲线**
    - `drift_score` 与 `psi_mean` 完全重复，**不再重复上传**
    - 其它字符串（dataset/task_type/stage 等）丢弃，它们走 config 面板
    """
    out: dict = {}
    for k, v in record.items():
        if k in STEP_FIELDS or k in DUPLICATE_FIELDS:
            continue
        if k in CATEGORICAL:
            if v is None:
                continue
            code = CATEGORICAL[k].get(str(v))
            if code is not None:
                out[f"{k}_code"] = float(code)
            continue
        if k not in UPLOAD_KEYS:
            continue
        if isinstance(v, bool):
            out[k] = float(int(v))
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
            continue
        if v is None:
            continue
        # 其余（字符串/列表/字典）一律不上传，避免 SwanLab 报错刷屏
    return out


def _step_of(record: dict) -> int | None:
    """取本条的横轴 step：优先 batch_id，其次 batch。

    SwanLab 要求 step 是非负整数，类型不对会被忽略整条 log。
    """
    for key in STEP_FIELDS:
        v = record.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v >= 0:
            return v
        if isinstance(v, float) and v.is_integer() and v >= 0:
            return int(v)
    return None


class Recorder:
    """一次运行一个 recorder。

    use_swanlab 参数保留向后兼容；实际是否上传由 SWANLAB_ENABLED 或该参数决定。
    """

    def __init__(self, path: str, use_swanlab: bool | None = None,
                 project: str | None = None, experiment: str | None = None,
                 config: dict | None = None):
        self.path = path
        self.records: list[dict] = []
        self.write_ok = False
        self.write_error: str | None = None

        # ---- 本地日志：无论云端能否用，都要先保住 ----
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "w", encoding="utf-8").close()
            self.write_ok = True
        except Exception as exc:                   # noqa: BLE001
            self.write_error = f"{type(exc).__name__}: {exc}"

        # ---- 云端配置 ----
        self.enabled = use_swanlab if use_swanlab is not None else _env_flag("SWANLAB_ENABLED")
        # 优先读 swanlab 原生的 *_NAME 变量；旧的 SWANLAB_PROJECT 仅作兜底，
        # 且会在 init 前被临时移出环境（否则 swanlab 会把它当嵌套模型解析）。
        self.project = (project
                        or os.getenv("SWANLAB_PROJECT_NAME")
                        or os.getenv("SWANLAB_PROJECT")
                        or "zhixun-engine")
        self.experiment = (experiment
                           or os.getenv("SWANLAB_EXPERIMENT_NAME")
                           or os.getenv("SWANLAB_EXPERIMENT")
                           or f"run-{time.strftime('%Y%m%d-%H%M%S')}")
        self.mode = normalize_mode(os.getenv("SWANLAB_MODE"))
        self.config = dict(config or {})

        self.swan = None
        self.swanlab_error: str | None = None
        self.finished = False
        self.uploaded_logs = 0

        if self.enabled:
            self._init_swanlab()

    # ------------------------------------------------------------------
    def _init_swanlab(self) -> None:
        """初始化 SwanLab。失败只记录原因，绝不影响主流程。"""
        try:
            import swanlab                        # type: ignore
        except Exception as exc:                   # noqa: BLE001
            self.swanlab_error = (f"swanlab 未安装（{type(exc).__name__}: {exc}）。"
                                  f"先执行：pip install -r requirements-dev.txt")
            return

        # 先体检 .env：swanlab 自己也会读这个文件，坏变量名会让 init 直接抛 SettingsError。
        # 与其让它报一句看不懂的 pydantic 错误，不如这里给出可执行的修改方法。
        hits = unsafe_dotenv_hits()
        if hits:
            self.swanlab_error = (
                "检测到 .env 里有 SwanLab 无法解析的变量：\n"
                "        " + "\n        ".join(hits) +
                "\n      原因：swanlab 的 project/experiment 是嵌套模型（子字段 name），"
                "裸变量名会被当成 JSON 解析。\n"
                "      改法：SWANLAB_PROJECT_NAME=... / SWANLAB_EXPERIMENT_NAME=...，"
                "SWANLAB_MODE 用 online")
            return

        # 密钥只从环境变量读，且**不打印、不写日志**
        api_key = os.getenv("SWANLAB_API_KEY") or None
        if api_key is None and self.mode in ("online",):
            self.swanlab_error = (
                "未设置 SWANLAB_API_KEY，无法上传到网页端。"
                "请先登录 https://swanlab.cn 获取 API Key，"
                "然后 `set SWANLAB_API_KEY=...`（Windows）/ `export SWANLAB_API_KEY=...`")
            return

        kwargs = {
            "project": self.project,
            "experiment_name": self.experiment,
            "mode": self.mode,
        }
        # config 面板：字符串类信息都放这里（log() 不接受字符串）
        cfg = _scalar_only(self.config)
        if cfg:
            cfg = {**cfg, "categorical_codebook": CODEBOOK}
            kwargs["config"] = cfg

        # 把会污染 SwanLab 内部 Settings 的变量临时移出环境，再用显式参数传值；
        # 结束后原样恢复，保证我们自己的代码/测试仍能看到这些变量。
        saved: dict[str, str] = {}
        for key in UNSAFE_ENV_KEYS + ("SWANLAB_MODE",):
            if key in os.environ:
                saved[key] = os.environ.pop(key)
        try:
            self.swan = swanlab.init(**kwargs)
        except Exception as exc:                   # noqa: BLE001
            self.swanlab_error = f"swanlab.init 失败（{type(exc).__name__}: {exc}）"
            self.swan = None
        finally:
            os.environ.update(saved)

    # ------------------------------------------------------------------
    def log(self, record: dict) -> None:
        """写一条记录。本地必写；云端尽力上传。永不抛异常。"""
        rec = {**record, "ts": time.time()}
        self.records.append(rec)

        if self.write_ok:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            except Exception as exc:               # noqa: BLE001
                self.write_ok = False              # 只告警一次，之后纯内存
                self.write_error = self.write_error or f"本地写入失败: {exc}"

        if self.swan is None:
            return
        payload = _metric_only(rec)
        if not payload:
            return
        step = _step_of(rec)
        try:
            # 用 batch_id 当横轴：网页端 x 轴直接是真实批次号（2..10），
            # 而不是 SwanLab 自增的 0..8。
            if step is not None:
                self.swan.log(data=payload, step=step)
            else:
                self.swan.log(data=payload)
            self.uploaded_logs += 1
        except Exception as exc:                   # noqa: BLE001
            self.swanlab_error = f"swanlab.log 失败（{type(exc).__name__}: {exc}）"
            self.swan = None                       # 失败一次就停用，避免每批重试

    # ------------------------------------------------------------------
    def finish(self, summary: dict) -> None:
        """实验结束：补一条 summary，然后关闭 SwanLab run。"""
        self.log({"stage": "summary", **summary})
        if self.swan is not None:
            try:
                self.swan.finish()
                self.finished = True
            except Exception as exc:               # noqa: BLE001
                self.swanlab_error = f"swanlab.finish 失败（{type(exc).__name__}: {exc}）"
            finally:
                self.swan = None

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """如实报告状态——调用方据此决定能不能宣称"已上传"。"""
        if not self.enabled:
            cloud = "disabled（SWANLAB_ENABLED 未开启）"
        elif self.swanlab_error:
            cloud = f"blocked（{self.swanlab_error}）"
        elif self.swan is not None:
            cloud = (f"uploading（project={self.project}, "
                     f"experiment={self.experiment}, mode={self.mode}）")
        elif self.finished:
            cloud = (f"uploaded（project={self.project}, experiment={self.experiment}，"
                     f"{self.uploaded_logs} 次 log 已提交）")
        else:
            cloud = "unknown"
        return {
            "path": self.path,
            "n_records": len(self.records),
            "write_ok": self.write_ok,
            "write_error": self.write_error,
            "swanlab_enabled": self.enabled,
            "swanlab_project": self.project,
            "swanlab_experiment": self.experiment,
            "swanlab_mode": self.mode,
            "swanlab_cloud": cloud,
            "swanlab_error": self.swanlab_error,
            "swanlab_finished": self.finished,
            "swanlab_logged": self.uploaded_logs,
            # 明确：是否真的在网页端确认过，只有人工去浏览器看过才能置 True
            "verified_on_web": False,
        }

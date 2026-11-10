"""实验记录（SwanLab）：旁路记录，失败绝不影响主流程。

设计原则：
1. **不是主流程**。闭环能不能跑完，跟它无关。
2. **失败静默**。没装、断网、上传失败 → 打一行提示，继续跑。
3. **默认离线**。评测环境无外网，用 SWAN_MODE=offline 写本地 swanlog/，
   赛后在有网机器上 `swanlab sync ./swanlog/run-xxx` 上传。

用法：
    swan = SwanLogger(experiment_name="gas-logreg", config={"dataset": "gas"})
    swan.log({"f1_system": 0.9}, step=2)
    swan.finish()
"""
from __future__ import annotations

import os


class SwanLogger:
    def __init__(self, enabled: bool = True, project: str | None = None,
                 experiment_name: str | None = None,
                 config: dict | None = None, mode: str | None = None):
        self.run = None
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        try:
            import swanlab

            self.run = swanlab.init(
                project=project or os.environ.get("SWAN_PROJECT", "zhixun-engine"),
                experiment_name=experiment_name,
                mode=mode or os.environ.get("SWAN_MODE", "offline"),
                logdir=os.environ.get("SWAN_LOGDIR", "swanlog"),
                config=config or {},
                on_online_error="offline",   # 网络抖动时自动降级，不打断训练
            )
        except Exception as exc:              # noqa: BLE001
            self.enabled = False
            print(f"[swanlab] 未启用：{type(exc).__name__}: {exc}")

    # ---- 记录一批指标 ----
    def log(self, metrics: dict, step: int | None = None):
        if self.run is None:
            return
        try:
            if step is None:
                self.run.log(metrics)
            else:
                self.run.log(metrics, step=step)
        except Exception:                     # noqa: BLE001
            pass

    # ---- 记录一张图（比如 curve.png） ----
    def log_image(self, path: str, key: str = "curve"):
        if self.run is None or not os.path.exists(path):
            return
        try:
            import swanlab

            self.run.log({key: swanlab.Image(path)})
        except Exception:                     # noqa: BLE001
            pass

    # ---- 收尾 ----
    def finish(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:                     # noqa: BLE001
            pass

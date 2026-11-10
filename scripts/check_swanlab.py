# -*- coding: utf-8 -*-
"""SwanLab 就绪检查。

    python scripts/check_swanlab.py

只做检查，不上传。用来回答三个问题：
    1. SDK 装了吗？
    2. API Key 配了吗？（只报告"是否设置"，绝不回显内容）
    3. 能连上 swanlab.cn 吗？

任何一项不满足，都说明**网页端验证一定会阻塞**，不要对外声称"已上传"。
"""
from __future__ import annotations

import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from evolver.envfile import load_env, summary as env_summary  # noqa: E402

_ENV = load_env()


def check_sdk() -> tuple[bool, str]:
    try:
        import swanlab                        # type: ignore
        return True, f"已安装，版本 {getattr(swanlab, '__version__', '未知')}"
    except Exception as exc:                   # noqa: BLE001
        return False, (f"未安装（{type(exc).__name__}: {exc}）→ "
                       f"运行：pip install -r requirements-dev.txt")


def check_runtime() -> tuple[bool, str]:
    """跑闭环还需要运行时依赖（requirements.lock），很多人只装了 dev 那份。"""
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("sklearn", "scikit-learn"),
                     ("matplotlib", "matplotlib")):
        try:
            __import__(mod)
        except Exception:                      # noqa: BLE001
            missing.append(pkg)
    if missing:
        return False, (f"缺少 {', '.join(missing)} → 运行：pip install -r requirements.lock")
    return True, "numpy / scikit-learn / matplotlib 齐备"


def check_key() -> tuple[bool, str]:
    raw = os.getenv("SWANLAB_API_KEY")
    if not raw:
        return False, "未设置 SWANLAB_API_KEY → 无法上传（登录 https://swanlab.cn 获取）"
    # 只报告长度与前缀掩码，绝不回显
    masked = "*" * max(0, len(raw) - 4) + raw[-4:] if len(raw) > 4 else "****"
    return True, f"已设置（掩码 {masked}，长度 {len(raw)}）"


def check_net() -> tuple[bool, str]:
    for url in ("https://swanlab.cn", "https://api.swanlab.cn"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"},
                                         method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as r:
                return True, f"{url} 可达（HTTP {r.status}）"
        except Exception as exc:               # noqa: BLE001
            last = f"{url} 不可达（{type(exc).__name__}: {exc}）"
    return False, last


def check_env() -> list[str]:
    lines = []
    for name in ("SWANLAB_ENABLED", "SWANLAB_PROJECT_NAME", "SWANLAB_EXPERIMENT_NAME",
                 "SWANLAB_MODE"):
        v = os.getenv(name)
        lines.append(f"{name} = {v if v else '<未设置，使用默认值>'}")
    # 常见错误写法提醒
    warn = []
    for bad, good in (("SWANLAB_PROJECT", "SWANLAB_PROJECT_NAME"),
                      ("SWANLAB_EXPERIMENT", "SWANLAB_EXPERIMENT_NAME")):
        if os.getenv(bad):
            warn.append(f"检测到 {bad}，请改用 {good}（否则 swanlab 会报 SettingsError）")
    mode = (os.getenv("SWANLAB_MODE") or "").strip().lower()
    if mode and mode not in ("disabled", "online", "local", "offline"):
        warn.append(f'SWANLAB_MODE="{mode}" 非法，只允许 disabled/online/local/offline')
    lines.extend(f"[!] {w}" for w in warn)
    return lines


def main() -> int:
    print("=" * 62)
    print("SwanLab 就绪检查")
    print("=" * 62)
    print(f"[--] 配置   : {env_summary(_ENV)}")
    if _ENV.get("missing"):
        print("            （在项目根目录放一个 .env，或直接设环境变量）")
    ok_sdk, msg_sdk = check_sdk()
    ok_key, msg_key = check_key()
    ok_net, msg_net = check_net()
    ok_rt, msg_rt = check_runtime()
    print(f"[{'OK' if ok_rt else '!!'}] 运行时 : {msg_rt}")
    print(f"[{'OK' if ok_sdk else '!!'}] SDK    : {msg_sdk}")
    print(f"[{'OK' if ok_key else '!!'}] API Key: {msg_key}")
    print(f"[{'OK' if ok_net else '!!'}] 网络   : {msg_net}")
    print("-" * 62)
    for line in check_env():
        print(f"     {line}")
    print("-" * 62)
    if ok_sdk and ok_key and ok_net and ok_rt:
        print("结论：四项齐备，可以尝试上传。")
        print("      上传后**必须**打开网页确认实验是否出现，才能说'已验证'。")
        return 0
    print("结论：**网页端验证会被阻塞**，请先解决上面标 !! 的项。")
    print("      本地 run.jsonl 仍然可用，不影响跑实验。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

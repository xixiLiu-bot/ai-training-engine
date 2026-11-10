"""极简 .env 加载器（纯标准库，零依赖）。

为什么自己写：项目承诺"运行时零第三方依赖"（评测镜像里只有 numpy/sklearn），
引入 python-dotenv 会破坏这一点。而 .env 的语法简单到不值得加依赖。

行为约定
--------
1. 只读，不写；文件不存在就**静默跳过**（.env 是可选的）
2. **不覆盖**已存在的环境变量（命令行 export 优先），可用 override=True 强制
3. 支持 `KEY=VALUE`、`export KEY=VALUE`、单/双引号包裹、`#` 注释、空行
4. **返回值不含 value**：只返回"应用了哪些 key"，避免调用方把密钥顺手打出来

安全提醒：.env 里会有 API Key，必须确保它被 .gitignore 忽略，且绝不进镜像。
"""
from __future__ import annotations

import os

DEFAULT_NAME = ".env"


def repo_root() -> str:
    """evolver/envfile.py → 上溯两级 = 代码包根目录。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _strip_comment(line: str) -> str:
    """去掉行尾注释。只处理不在引号里的 #。"""
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_text(text: str) -> list[tuple[str, str]]:
    """把 .env 文本解析成 (key, value) 列表。坏行直接跳过，不抛异常。"""
    pairs: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue                                  # 非法变量名，跳过
        value = _unquote(_strip_comment(value))
        pairs.append((key, value))
    return pairs


def load_env(path: str | None = None, override: bool = False) -> dict:
    """加载 .env 到 os.environ。

    返回 {'path': ..., 'applied': [key...], 'kept': [key...], 'missing': bool}
    —— **不返回任何 value**，防止密钥被顺手打印。
    """
    target = path or os.path.join(repo_root(), DEFAULT_NAME)
    result = {"path": target, "applied": [], "kept": [], "missing": False}
    if not os.path.isfile(target):
        result["missing"] = True
        return result
    try:
        with open(target, encoding="utf-8-sig") as f:
            text = f.read()
    except Exception:                                  # noqa: BLE001
        result["missing"] = True                       # 读不了就当没有，不影响主流程
        return result

    for key, value in parse_text(text):
        if not override and os.getenv(key) is not None:
            result["kept"].append(key)                 # 已有环境变量优先
            continue
        os.environ[key] = value
        result["applied"].append(key)
    return result


def summary(res: dict) -> str:
    """给人看的一行摘要（只有 key 名，没有 value）。"""
    if res.get("missing"):
        return f".env 未找到（{res['path']}），跳过"
    parts = [f"{len(res['applied'])} 项已载入"]
    if res["kept"]:
        parts.append(f"{len(res['kept'])} 项因已存在而保留原值")
    return f".env 已读取（{res['path']}）：" + "，".join(parts)

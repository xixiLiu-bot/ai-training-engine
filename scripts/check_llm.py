# -*- coding: utf-8 -*-
"""连大模型之前的自检：配没配、通不通、真的能回话吗。

设计原则（和 check_swanlab.py 一致）：
    三个问题分开答，任何一项不满足都明确报阻塞，**不假装成功**。
    API Key 只显示掩码，绝不打印全文、绝不写进日志。

用法：
    python scripts/check_llm.py                # 默认查 qwen4b（4B）
    python scripts/check_llm.py --backend pytrio   # 查 27B
    python scripts/check_llm.py --backend ollama   # 查本地 ollama

退出码：0 = 全通；1 = 有阻塞（方便脚本判断）
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evolver.envfile import load_env                     # noqa: E402
from evolver.engine.qwen4b import compat as q4b          # noqa: E402

# 和 run_pipeline 一致：启动时读根目录 .env（已 gitignore，不会提交）
_ENV = load_env()
from evolver.engine.qwen4b.client import (                 # noqa: E402
    AuthError, CapabilityError, LLMError, ServerError, TransportError,
)

OK = "[OK]"
NO = "[X]"
WARN = "[!]"


def mask(s: str) -> str:
    if not s:
        return "(空)"
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "*" * (len(s) - 8) + s[-4:]


def line(title: str, ok: bool | None, detail: str):
    mark = OK if ok is True else (NO if ok is False else WARN)
    print(f"  {mark:<5}{title:<22}{detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen4b",
                    choices=["qwen4b", "pytrio", "ollama"])
    args = ap.parse_args()

    if _ENV.get("applied"):
        print(f"  已读取 .env：{len(_ENV['applied'])} 项（{_ENV.get('path', '')}）")
    elif _ENV.get("missing"):
        print("  提示：没找到 .env，仅使用当前环境变量")
    print(f"=== 大模型接入自检（backend = {args.backend}）===\n")

    # ── 一、配置 ──
    url_env, model_env, default_model = q4b._BACKENDS[args.backend]
    base_url = os.getenv(url_env, "")
    if not base_url and args.backend == "ollama":
        base_url = q4b._OLLAMA_DEFAULT
    if not base_url and args.backend in ("qwen4b", "pytrio"):
        base_url = os.getenv("PYTRIO_BASE_URL", q4b._PYTRIO_DEFAULT)
    api_key = os.getenv("QWEN_API_KEY", "")
    model = os.getenv(model_env, default_model)

    print("一、配置")
    line("base_url", bool(base_url), base_url or "未设置")
    line("model", bool(model), model)
    # ollama 不需要真 Key（本地服务），其余后端必须要
    has_key = (bool(api_key) and api_key != "EMPTY") or args.backend == "ollama"
    line("QWEN_API_KEY", has_key,
         mask(api_key) if api_key else "未设置 ← 没它连不上")
    line("关闭思考模式", True, "enable_thinking=false（默认开启，PyTRiO 必须关）")

    if not has_key:
        print(f"\n{NO} 阻塞：没有 API Key。")
        print("     PowerShell 临时设置（只对当前窗口生效）：")
        print('       $env:QWEN_API_KEY = "从平台复制的 key"')
        print('       python scripts/check_llm.py --backend %s' % args.backend)
        print("     或者写进 .env（已在 .gitignore 里，不会提交）：")
        print('       QWEN_API_KEY="..."')
        return 1

    # ── 二、连通性 + 真实回话 ──
    print("\n二、调用")
    decider = q4b.build(args.backend)
    if not getattr(decider, "enabled", False):
        print(f"  {NO} 决策器未启用（build 返回了 DisabledDecider）")
        return 1

    state = {"batch_id": 3, "psi_mean": 1.2, "current_score": 0.70,
             "best_score": 0.90, "steps_since_update": 2,
             "last_action": "无", "last_gain": 0.0}
    diag = {"kind": "covariate", "reason": "特征整体平移，标签分布稳定"}
    catalog = {"do_nothing": {"cost_sec": 0.0},
               "refresh_normalization": {"cost_sec": 0.05},
               "refit_model": {"cost_sec": 0.4}}

    try:
        sug = decider.suggest(state, diag, catalog, "refresh_normalization")
    except AuthError as exc:
        line("鉴权", False, f"Key 无效或被拒：{str(exc)[:80]}")
        print(f"\n{NO} 阻塞：Key 不对。去平台确认，或换一个 Key。")
        return 1
    except CapabilityError as exc:
        line("协议", False, f"平台拒绝了请求参数：{str(exc)[:100]}")
        print(f"\n{NO} 阻塞：参数不被支持（可能需要调整 extra_body）。")
        return 1
    except TransportError as exc:
        line("网络", False, f"连不上：{str(exc)[:100]}")
        print(f"\n{NO} 阻塞：网络问题。检查 base_url、代理、防火墙。")
        return 1
    except ServerError as exc:
        line("服务端", False, f"5xx：{str(exc)[:100]}")
        return 1
    except LLMError as exc:
        line("调用", False, f"{type(exc).__name__}: {str(exc)[:100]}")
        return 1

    if not sug or "action" not in sug:
        err = ""
        if isinstance(sug, dict):
            err = str(sug.get("error") or sug.get("note") or "") or str(sug)
        st = decider.stats()
        line("回话", False, f"没拿到合法动作：{err[:120]}")
        print(f"  统计：{st}")

        # 按错误内容给准确诊断，别把鉴权问题说成"max_tokens 太小"
        low = err.lower()
        if "401" in low or "403" in low or "unauthorized" in low or "autherror" in low:
            print(f"\n{NO} 阻塞：**Key 无效或被拒**（HTTP 401/403）。")
            print("    去平台确认这个 Key 还有效、且没抄错（前后空格也算错）。")
        elif "timeout" in low or "transport" in low or "urlerror" in low:
            print(f"\n{NO} 阻塞：**网络连不上**。检查 base_url / 代理 / 防火墙。")
        elif "400" in low or "capabilit" in low:
            print(f"\n{NO} 阻塞：**平台拒绝了请求参数**。可能需要调 extra_body。")
        elif "json" in low or "parse" in low or "截断" in err:
            print(f"\n{NO} 阻塞：**模型输出格式不对或被截断**。")
            print("   4B 把 max_tokens 设小一点（默认 96）；27B 要大一些。")
        else:
            print(f"\n{NO} 阻塞：模型没返回可用的决策。原始错误见上方。")
        return 1

    line("回话", True, f"action={sug.get('action')}  "
                       f"reason={str(sug.get('reason'))[:40]}")
    st = decider.stats()
    line("耗时", True, f"{st.get('last_latency', sug.get('latency', '-'))} s")
    line("tokens", True, f"{st.get('tokens', 0)}")
    line("熔断状态", not st.get("tripped"), "已熔断" if st.get("tripped") else "正常")

    print(f"\n{OK} 全部通过。可以正式跑了：")
    print(f"     python scripts/run_pipeline.py --dataset gas --llm {args.backend}")
    print("   （注意：决策层会额外花 token，账本里的 tokens 会计入成本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

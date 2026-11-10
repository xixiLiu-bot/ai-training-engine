"""实验记录层。

    Recorder —— 本地 JSONL + 可选 SwanLab

对应官方《实验记录规范》与门槛②（记录完整）。断网、无账号时本地记录仍然可用。
"""
from .recorder import Recorder

__all__ = ["Recorder"]

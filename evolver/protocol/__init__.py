"""评测协议层：本地模拟「一问一答」，与企业赛接口同构。

    LocalEvaluator   —— 本地开发用，从装好的数据里逐批发放
    RemoteEvaluator  —— 企业赛用，接官方接口（占位）
"""
from .batch_stream import (LocalEvaluator, ProtocolViolation,
                           RemoteEvaluator, Submission)

__all__ = ["LocalEvaluator", "ProtocolViolation", "RemoteEvaluator", "Submission"]

"""桥接层：把 DatasetSource 适配成闭环认识的 DatasetAdapter 协议。

契约对照：

    DatasetSource（丰富）              DatasetAdapter（闭环只需要这些）
    ─────────────────────────         ─────────────────────────────
    name                               name
    get_task_type()                    task_type
    get_features()                     feature_names
    iter_batches()                     n_batches / get(batch_id)
    get_metadata()                     （塞进 Batch.meta）

这样 run_pipeline.py、protocol/batch_stream.py、loop/ 全部**一行都不用改**。
"""
from __future__ import annotations

from .dataset import Batch, DatasetAdapter
from .sources import DatasetSource


class BridgeAdapter(DatasetAdapter):
    """把数据源包成旧协议。

    参数
    ----
    source      DatasetSource 实例
    batch_ids   只暴露这些原始批次（按给出的顺序）。默认暴露全部。
                用途：验证阶段只跑 stream（不含 holdout），避免误用 test。
    """

    def __init__(self, source: DatasetSource, batch_ids: list[int] | None = None):
        self.source = source
        records = list(source.iter_batches())
        if batch_ids is not None:
            allow = list(batch_ids)
            by_id = {r["batch_id"]: r for r in records}
            missing = [i for i in allow if i not in by_id]
            if missing:
                raise ValueError(f"数据源 {source.name} 没有这些批次：{missing}")
            records = [by_id[i] for i in allow]
        self._records = records
        self._source_ids = [r["batch_id"] for r in records]

        self.name = f"{source.name}_drift"
        self.task_type = source.get_task_type()
        self.feature_names = list(source.get_features())
        self.n_batches = len(records)
        # 身份随数据源透传：上游（标定/评估）据此决定能不能用
        self.data_identity = getattr(source, "data_identity", "unknown")

    # ------------------------------------------------------------------
    def get(self, batch_id: int) -> Batch:
        """1-based 位置索引（与旧适配器一致），映射到数据源的原始批次号。"""
        if not (1 <= batch_id <= self.n_batches):
            raise IndexError(
                f"batch_id={batch_id} 超出范围 1..{self.n_batches}")
        rec = self._records[batch_id - 1]
        meta = {
            **rec["metadata"],
            "dataset": rec["dataset"],
            "source_batch_id": rec["batch_id"],
            "timestamp": rec["timestamp"],
            "data_identity": self.data_identity,
        }
        return Batch(batch_id, rec["features"], rec["labels"], meta)

    def source_batch_ids(self) -> list[int]:
        """暴露出来的原始批次号（按时间顺序），便于日志与核验。"""
        return list(self._source_ids)

    def __repr__(self):
        return (f"<BridgeAdapter source={self.source.name} "
                f"n_batches={self.n_batches} ids={self._source_ids}>")

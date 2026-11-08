"""④ 评估 + 影子对比：新方案先在旁边跑，不直接影响线上。"""
from __future__ import annotations


def evaluate(model, batch, metric) -> dict:
    """在一个有标签的批次上评估模型。"""
    pred = model.predict(batch.X)
    return {"score": metric.score(batch.y, pred), "n": int(len(batch.y))}


def shadow(current_model, candidate_model, batch, metric) -> dict:
    """影子评估：当前模型 vs 候选模型，同一批数据上比。"""
    before = evaluate(current_model, batch, metric)["score"]
    after = evaluate(candidate_model, batch, metric)["score"]
    return {
        "before": before,
        "after": after,
        "gain": after - before,
        "metric": metric.name,
    }

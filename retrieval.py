from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalMetrics:
    hit_at_k: float
    mrr: float
    ndcg_at_k: float
    case_count: int


def evaluate_rankings(
    rankings: list[list[str]], relevant_items: list[set[str]], k: int = 5
) -> RetrievalMetrics:
    if len(rankings) != len(relevant_items):
        raise ValueError("rankings 与 relevant_items 数量必须一致")
    if not rankings:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0)

    hits = 0.0
    reciprocal_ranks = 0.0
    ndcg = 0.0
    for ranking, relevant in zip(rankings, relevant_items, strict=True):
        top = ranking[:k]
        ranks = [index for index, item in enumerate(top, start=1) if item in relevant]
        if ranks:
            hits += 1
            reciprocal_ranks += 1 / ranks[0]
        dcg = sum(1 / math.log2(index + 1) for index in ranks)
        ideal_hits = min(len(relevant), k)
        ideal_dcg = sum(1 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        ndcg += dcg / ideal_dcg if ideal_dcg else 0
    count = len(rankings)
    return RetrievalMetrics(hits / count, reciprocal_ranks / count, ndcg / count, count)

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[T, float]]],
    rank_constant: int = 60,
) -> list[tuple[T, float]]:
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, (item_id, _raw_score) in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

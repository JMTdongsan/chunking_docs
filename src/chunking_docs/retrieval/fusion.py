from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedHit:
    item_id: str
    rank: int
    score: float
    source: str


def reciprocal_rank_fusion(
    result_sets: list[list[RankedHit]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[str, float, list[str]]]:
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)
    for hits in result_sets:
        for hit in hits:
            rank = max(hit.rank, 1)
            scores[hit.item_id] += 1.0 / (k + rank)
            sources[hit.item_id].add(hit.source)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(item_id, score, sorted(sources[item_id])) for item_id, score in ranked]

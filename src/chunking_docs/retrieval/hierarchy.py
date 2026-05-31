from __future__ import annotations

from collections import defaultdict

from chunking_docs.models import DocumentChunk
from chunking_docs.retrieval.fusion import RankedHit


def canonical_chunk_id(
    item_id: str,
    chunk_by_id: dict[str, DocumentChunk],
    collapse_hierarchical: bool,
) -> str:
    if not collapse_hierarchical:
        return item_id
    chunk = chunk_by_id.get(item_id)
    if chunk is None:
        return item_id
    parent_id = chunk.metadata.get("hierarchical_parent_chunk_id")
    if isinstance(parent_id, str) and parent_id in chunk_by_id:
        return parent_id
    return item_id


def collapse_ranked_hits(
    hits: list[RankedHit],
    chunk_by_id: dict[str, DocumentChunk],
    collapse_hierarchical: bool,
) -> tuple[list[RankedHit], dict[str, list[str]]]:
    if not collapse_hierarchical:
        return hits, {}

    best_by_item: dict[str, RankedHit] = {}
    evidence_by_item: dict[str, set[str]] = defaultdict(set)
    for hit in hits:
        canonical_id = canonical_chunk_id(hit.item_id, chunk_by_id, collapse_hierarchical=True)
        if canonical_id != hit.item_id:
            evidence_by_item[canonical_id].add(hit.item_id)
        existing = best_by_item.get(canonical_id)
        if existing is None or (hit.rank, -hit.score) < (existing.rank, -existing.score):
            best_by_item[canonical_id] = RankedHit(
                item_id=canonical_id,
                rank=hit.rank,
                score=hit.score,
                source=hit.source,
            )

    return (
        sorted(best_by_item.values(), key=lambda hit: hit.rank),
        {item_id: sorted(evidence_ids) for item_id, evidence_ids in evidence_by_item.items()},
    )


def merge_evidence_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for evidence_map in maps:
        for item_id, evidence_ids in evidence_map.items():
            merged[item_id].update(evidence_ids)
    return {item_id: sorted(evidence_ids) for item_id, evidence_ids in merged.items()}

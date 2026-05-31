from __future__ import annotations

from typing import Any

from chunking_docs.models import DocumentChunk, GraphTriple


def triple_asset_ids(triple: GraphTriple) -> set[str]:
    qualifiers = triple.qualifiers or {}
    asset_ids: set[str] = set()
    for key in ("asset_id", "source_asset_id", "visual_asset_id"):
        asset_ids.update(string_values(qualifiers.get(key)))
    for key in ("asset_ids", "source_asset_ids", "visual_asset_ids"):
        asset_ids.update(string_values(qualifiers.get(key)))
    for value in string_values(qualifiers.get("source_ref")):
        asset_ids.update(asset_ids_from_ref(value))
    for value in string_values(qualifiers.get("source_refs")):
        asset_ids.update(asset_ids_from_ref(value))
    return asset_ids


def chunk_asset_ids(chunk: DocumentChunk) -> list[str]:
    asset_ids = list(chunk.asset_ids)
    for ref in chunk.source_refs:
        asset_ids.extend(asset_ids_from_ref(ref))
    return ordered_unique(asset_ids)


def chunk_id_alias_map(chunks: list[DocumentChunk]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for chunk in chunks:
        aliases.setdefault(chunk.chunk_id, chunk.chunk_id)
        for key in ("source_chunk_id", "parent_chunk_id"):
            value = chunk.metadata.get(key)
            if isinstance(value, str) and value:
                aliases.setdefault(value, chunk.chunk_id)
    return aliases


def chunk_ids_by_asset_id(chunks: list[DocumentChunk]) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = {}
    for chunk in chunks:
        for asset_id in chunk_asset_ids(chunk):
            indexed.setdefault(asset_id, []).append(chunk.chunk_id)
    return {asset_id: ordered_unique(chunk_ids) for asset_id, chunk_ids in indexed.items()}


def triple_resolved_chunk_ids(
    triple: GraphTriple,
    chunk_ids: set[str],
    chunk_id_by_alias: dict[str, str],
    chunk_ids_by_asset: dict[str, list[str]],
) -> list[str]:
    candidates = []
    chunk_id = chunk_id_by_alias.get(triple.chunk_id, triple.chunk_id)
    if chunk_id in chunk_ids:
        candidates.append(chunk_id)
    for asset_id in sorted(triple_asset_ids(triple)):
        candidates.extend(chunk_id for chunk_id in chunk_ids_by_asset.get(asset_id, []) if chunk_id in chunk_ids)
    return ordered_unique(candidates)


def string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, str) and item]
    return []


def asset_ids_from_ref(ref: str) -> set[str]:
    if ref.startswith("asset:") and len(ref) > len("asset:"):
        return {ref.removeprefix("asset:")}
    return set()


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result

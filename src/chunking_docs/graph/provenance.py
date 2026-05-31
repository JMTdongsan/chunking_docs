from __future__ import annotations

from typing import Any

from chunking_docs.models import GraphTriple


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

from __future__ import annotations

from typing import Any


QDRANT_RECORD_FILES = {
    "text_dense": "qdrant_text_records.jsonl",
    "image_dense": "qdrant_image_records.jsonl",
    "caption_dense": "qdrant_caption_records.jsonl",
    "object_dense": "qdrant_object_records.jsonl",
    "triple_dense": "qdrant_triple_records.jsonl",
}

QDRANT_PAYLOAD_INDEXES = [
    {"field": "doc_id", "schema": "keyword"},
    {"field": "chunk_id", "schema": "keyword"},
    {"field": "asset_id", "schema": "keyword"},
    {"field": "object_id", "schema": "keyword"},
    {"field": "triple_id", "schema": "keyword"},
    {"field": "record_kind", "schema": "keyword"},
    {"field": "kind", "schema": "keyword"},
    {"field": "label", "schema": "keyword"},
    {"field": "bbox_region", "schema": "keyword"},
    {"field": "source_key", "schema": "keyword"},
    {"field": "visual_feature_type", "schema": "keyword"},
    {"field": "predicate", "schema": "keyword"},
    {"field": "chunking_strategy", "schema": "keyword"},
    {"field": "retrieval_role", "schema": "keyword"},
    {"field": "parent_chunk_id", "schema": "keyword"},
    {"field": "source_chunk_id", "schema": "keyword"},
    {"field": "hierarchical_parent_chunk_id", "schema": "keyword"},
    {"field": "visual_asset_unlinked", "schema": "bool"},
    {"field": "text_quality", "schema": "keyword"},
    {"field": "text_quality_reasons", "schema": "keyword"},
    {"field": "requires_ocr", "schema": "bool"},
    {"field": "requires_vlm", "schema": "bool"},
    {"field": "asset_scope", "schema": "keyword"},
    {"field": "parent_asset_id", "schema": "keyword"},
    {"field": "tile_index", "schema": "integer"},
    {"field": "tile_row", "schema": "integer"},
    {"field": "tile_col", "schema": "integer"},
    {"field": "tile_rows", "schema": "integer"},
    {"field": "tile_cols", "schema": "integer"},
    {"field": "control_char_count", "schema": "integer"},
    {"field": "control_char_ratio", "schema": "float"},
    {"field": "letter_or_number_ratio", "schema": "float"},
    {"field": "cjk_char_ratio", "schema": "float"},
    {"field": "image_block_count", "schema": "integer"},
    {"field": "embedded_image_count", "schema": "integer"},
    {"field": "drawing_count", "schema": "integer"},
    {"field": "tile_overlap_ratio", "schema": "float"},
    {"field": "page_no", "schema": "integer"},
    {"field": "page_start", "schema": "integer"},
    {"field": "page_end", "schema": "integer"},
    {"field": "section.chapter", "schema": "keyword"},
    {"field": "section.issue", "schema": "keyword"},
]


def qdrant_payload_index_fields(
    payload_indexes: list[str | dict[str, Any]] | None = None,
) -> set[str]:
    fields: set[str] = set()
    indexes = QDRANT_PAYLOAD_INDEXES if payload_indexes is None else payload_indexes
    for index in indexes:
        if isinstance(index, str):
            fields.add(index)
        else:
            field = index.get("field") or index.get("field_name")
            if field:
                fields.add(str(field))
    return fields


def qdrant_payload_index_schemas(
    payload_indexes: list[str | dict[str, Any]] | None = None,
) -> dict[str, str]:
    schemas: dict[str, str] = {}
    indexes = QDRANT_PAYLOAD_INDEXES if payload_indexes is None else payload_indexes
    for index in indexes:
        if isinstance(index, str):
            field = index.strip()
            if field:
                schemas[field] = default_payload_schema(field)
            continue
        field = str(index.get("field") or index.get("field_name") or "").strip()
        if not field:
            continue
        schemas[field] = normalize_payload_schema(
            str(index.get("schema") or default_payload_schema(field))
        )
    return dict(sorted(schemas.items()))


def default_payload_schema(field_name: str) -> str:
    integer_fields = {
        "page_no",
        "page_start",
        "page_end",
        "tile_index",
        "tile_row",
        "tile_col",
        "tile_rows",
        "tile_cols",
        "control_char_count",
        "image_block_count",
        "embedded_image_count",
        "drawing_count",
    }
    float_fields = {
        "control_char_ratio",
        "letter_or_number_ratio",
        "cjk_char_ratio",
        "tile_overlap_ratio",
    }
    bool_fields = {"visual_asset_unlinked", "requires_ocr", "requires_vlm"}
    if field_name in integer_fields:
        return "integer"
    if field_name in float_fields:
        return "float"
    if field_name in bool_fields:
        return "bool"
    return "keyword"


def normalize_payload_schema(value: str) -> str:
    normalized = value.strip().lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    aliases = {
        "int": "integer",
        "int64": "integer",
        "uint64": "integer",
        "bool": "bool",
        "boolean": "bool",
        "float64": "float",
        "double": "float",
    }
    return aliases.get(normalized, normalized)

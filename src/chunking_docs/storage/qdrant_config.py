from __future__ import annotations

from typing import Any


QDRANT_RECORD_FILES = {
    "text_dense": "qdrant_text_records.jsonl",
    "image_dense": "qdrant_image_records.jsonl",
    "caption_dense": "qdrant_caption_records.jsonl",
    "triple_dense": "qdrant_triple_records.jsonl",
}

QDRANT_PAYLOAD_INDEXES = [
    {"field": "doc_id", "schema": "keyword"},
    {"field": "chunk_id", "schema": "keyword"},
    {"field": "asset_id", "schema": "keyword"},
    {"field": "triple_id", "schema": "keyword"},
    {"field": "kind", "schema": "keyword"},
    {"field": "predicate", "schema": "keyword"},
    {"field": "chunking_strategy", "schema": "keyword"},
    {"field": "retrieval_role", "schema": "keyword"},
    {"field": "parent_chunk_id", "schema": "keyword"},
    {"field": "source_chunk_id", "schema": "keyword"},
    {"field": "hierarchical_parent_chunk_id", "schema": "keyword"},
    {"field": "visual_asset_unlinked", "schema": "bool"},
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

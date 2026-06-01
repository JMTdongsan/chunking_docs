from __future__ import annotations

import uuid
from typing import Any

from chunking_docs.embeddings.interfaces import DenseTextEmbedder
from chunking_docs.embeddings.interfaces import DenseImageEmbedder
from chunking_docs.graph.provenance import chunk_asset_ids, triple_asset_ids
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.storage.records import EmbeddingRecord

VISUAL_OBJECT_METADATA_KEYS = (
    "objects",
    "detected_objects",
    "visual_objects",
    "detections",
    "regions",
    "areas",
)


def point_id(chunk_id: str, vector_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunking-docs:{chunk_id}:{vector_name}"))


def make_text_embedding_records(
    chunks: list[DocumentChunk],
    embedder: DenseTextEmbedder,
    vector_name: str = "text_dense",
    batch_size: int = 32,
) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_texts([chunk.text for chunk in batch])
        for chunk, vector in zip(batch, vectors):
            asset_ids = chunk_asset_ids(chunk)
            records.append(
                EmbeddingRecord(
                    point_id=point_id(chunk.chunk_id, vector_name),
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    vector_name=vector_name,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "kind": chunk.kind,
                        "section": chunk.section.model_dump(),
                        "asset_id": asset_ids,
                        "asset_ids": asset_ids,
                        "source_refs": chunk.source_refs,
                        "text": chunk.text,
                        **chunk.metadata,
                    },
                )
            )
    return records


def make_image_embedding_records(
    assets: list[VisualAsset],
    embedder: DenseImageEmbedder,
    vector_name: str = "image_dense",
    batch_size: int = 16,
) -> list[EmbeddingRecord]:
    image_assets = [asset for asset in assets if asset.path is not None]
    records: list[EmbeddingRecord] = []
    for start in range(0, len(image_assets), batch_size):
        batch = image_assets[start : start + batch_size]
        vectors = embedder.embed_images([asset.path for asset in batch if asset.path is not None])
        for asset, vector in zip(batch, vectors):
            records.append(
                EmbeddingRecord(
                    point_id=point_id(asset.asset_id, vector_name),
                    chunk_id=asset.asset_id,
                    doc_id=asset.doc_id,
                    vector_name=vector_name,
                    vector=vector,
                    payload={
                        "asset_id": asset.asset_id,
                        "doc_id": asset.doc_id,
                        "page_no": asset.page_no,
                        "kind": asset.kind,
                        "caption": asset.caption,
                        "ocr_text": asset.ocr_text,
                        "vlm_summary": asset.vlm_summary,
                        **asset.metadata,
                    },
                )
            )
    return records


def make_caption_embedding_records(
    assets: list[VisualAsset],
    embedder: DenseTextEmbedder,
    vector_name: str = "caption_dense",
    batch_size: int = 32,
) -> list[EmbeddingRecord]:
    caption_assets = [asset for asset in assets if asset_text(asset)]
    records: list[EmbeddingRecord] = []
    for start in range(0, len(caption_assets), batch_size):
        batch = caption_assets[start : start + batch_size]
        texts = [asset_text(asset) for asset in batch]
        vectors = embedder.embed_texts(texts)
        for asset, text, vector in zip(batch, texts, vectors):
            records.append(
                EmbeddingRecord(
                    point_id=point_id(asset.asset_id, vector_name),
                    chunk_id=asset.asset_id,
                    doc_id=asset.doc_id,
                    vector_name=vector_name,
                    vector=vector,
                    payload={
                        "asset_id": asset.asset_id,
                        "doc_id": asset.doc_id,
                        "page_no": asset.page_no,
                        "kind": asset.kind,
                        "text": text,
                        "caption": asset.caption,
                        **asset.metadata,
                    },
                )
            )
    return records


def make_triple_embedding_records(
    triples: list[GraphTriple],
    embedder: DenseTextEmbedder,
    vector_name: str = "triple_dense",
    batch_size: int = 32,
) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for start in range(0, len(triples), batch_size):
        batch = triples[start : start + batch_size]
        texts = [triple_text(triple) for triple in batch]
        vectors = embedder.embed_texts(texts)
        for triple, text, vector in zip(batch, texts, vectors):
            asset_ids = sorted(triple_asset_ids(triple))
            records.append(
                EmbeddingRecord(
                    point_id=point_id(triple.triple_id, vector_name),
                    chunk_id=triple.chunk_id,
                    doc_id=triple.doc_id,
                    vector_name=vector_name,
                    vector=vector,
                    payload={
                        "triple_id": triple.triple_id,
                        "chunk_id": triple.chunk_id,
                        "doc_id": triple.doc_id,
                        "kind": "graph_triple",
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "object": triple.object,
                        "text": text,
                        "asset_id": asset_ids,
                        "asset_ids": asset_ids,
                        "confidence": triple.confidence,
                        "qualifiers": triple.qualifiers,
                    },
                )
            )
    return records


def triple_text(triple: GraphTriple) -> str:
    predicate = str(triple.predicate).replace("_", " ")
    parts = [triple.subject, predicate, triple.object]
    source_field = triple.qualifiers.get("source_field")
    if source_field:
        parts.append(f"source field {source_field}")
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def asset_text(asset: VisualAsset) -> str:
    return "\n".join(asset_text_parts(asset))


def asset_text_parts(asset: VisualAsset) -> list[str]:
    return deduplicate_text_parts(
        [
            asset.caption or "",
            asset.ocr_text or "",
            asset.vlm_summary or "",
            *asset_metadata_text_parts(asset.metadata),
        ]
    )


def asset_metadata_text_parts(metadata: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    page_type = metadata_text_value(metadata.get("page_type"))
    if page_type:
        parts.append(f"Page type: {page_type}")
    for key, label, limit in [
        ("key_points", "Key points", 6),
        ("entities", "Entities", 8),
        ("visual_elements", "Visual elements", 8),
    ]:
        values = metadata_text_items(metadata.get(key), limit=limit)
        if values:
            parts.append(f"{label}: {'; '.join(values)}")
    objects = dedupe_metadata_values(
        [
            object_text
            for key in VISUAL_OBJECT_METADATA_KEYS
            for object_text in metadata_object_items(metadata.get(key), limit=8)
        ],
        limit=8,
    )
    if objects:
        parts.append(f"Objects: {'; '.join(objects)}")
    return parts


def metadata_text_items(value: Any, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, dict):
                label = metadata_first_string(item, ["label", "name", "title", "entity", "object", "type"])
                description = metadata_first_string(item, ["description", "summary", "text"])
                if label and description and description != label:
                    values.append(f"{label}: {description}")
                elif label:
                    values.append(label)
                elif description:
                    values.append(description)
            else:
                values.append(str(item))
    else:
        values = [str(value)]
    return dedupe_metadata_values(values, limit=limit)


def metadata_object_items(value: Any, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    elif isinstance(value, dict):
        if metadata_first_string(value, ["label", "name", "title", "object", "type", "category"]):
            candidates = [value]
        else:
            candidates = metadata_object_mapping_items(value)
    else:
        candidates = [value]

    objects: list[str] = []
    for item in candidates:
        if isinstance(item, dict):
            label = metadata_first_string(item, ["label", "name", "title", "object", "type", "category"])
            if not label:
                continue
            attributes = metadata_text_items(
                item.get("attributes") or item.get("features") or item.get("descriptors"),
                limit=6,
            )
            description = metadata_first_string(item, ["description", "summary", "text"])
            if description and description not in attributes:
                attributes.append(description)
            location = metadata_first_string(item, ["location", "position", "region"])
            if location and location not in attributes:
                attributes.append(location)
            if attributes:
                objects.append(f"{label}: {', '.join(attributes[:6])}")
            else:
                objects.append(label)
        else:
            value_text = metadata_text_value(item)
            if value_text:
                objects.append(value_text)
    return dedupe_metadata_values(objects, limit=limit)


def metadata_object_mapping_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for label, details in value.items():
        label_text = metadata_text_value(label)
        if not label_text:
            continue
        if isinstance(details, dict):
            item = {**details}
            item.setdefault("label", label_text)
        elif isinstance(details, str):
            item = {"label": label_text, "description": details}
        else:
            item = {"label": label_text}
        items.append(item)
    return items


def metadata_first_string(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = metadata_text_value(payload.get(key))
        if value:
            return value
    return ""


def metadata_text_value(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text.strip()


def dedupe_metadata_values(values: list[str], limit: int) -> list[str]:
    selected = []
    seen = set()
    for value in values:
        normalized = metadata_text_value(value)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        selected.append(normalized)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def deduplicate_text_parts(parts: list[str]) -> list[str]:
    selected = []
    seen = set()
    for part in parts:
        normalized = metadata_text_value(part)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        selected.append(part.strip())
        seen.add(key)
    return selected

from __future__ import annotations

import uuid
from typing import Any

from chunking_docs.embeddings.interfaces import DenseTextEmbedder
from chunking_docs.embeddings.interfaces import DenseImageEmbedder
from chunking_docs.graph.provenance import (
    chunk_asset_ids,
    chunk_id_alias_map,
    chunk_ids_by_asset_id,
    triple_asset_ids,
    triple_resolved_chunk_ids,
)
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.spatial import bbox_region_from_bbox
from chunking_docs.vision.spatial import normalize_bbox

VISUAL_OBJECT_METADATA_KEYS = (
    "objects",
    "detected_objects",
    "visual_objects",
    "detections",
    "regions",
    "areas",
)
VISUAL_FEATURE_METADATA_KEYS = ("visual_elements",)
VISUAL_OBJECT_LABEL_KEYS = ["label", "name", "title", "object", "type", "category"]
VISUAL_OBJECT_BBOX_KEYS = ["bbox", "box", "bounding_box", "boundingBox", "bounds"]


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


def make_object_embedding_records(
    assets: list[VisualAsset],
    embedder: DenseTextEmbedder,
    vector_name: str = "object_dense",
    batch_size: int = 32,
) -> list[EmbeddingRecord]:
    object_items = visual_object_embedding_items(assets)
    records: list[EmbeddingRecord] = []
    for start in range(0, len(object_items), batch_size):
        batch = object_items[start : start + batch_size]
        texts = [str(item["text"]) for item in batch]
        vectors = embedder.embed_texts(texts)
        for item, vector in zip(batch, vectors):
            object_id = str(item["object_id"])
            payload = visual_object_payload(item)
            records.append(
                EmbeddingRecord(
                    point_id=point_id(object_id, vector_name),
                    chunk_id=str(item["asset_id"]),
                    doc_id=str(item["doc_id"]),
                    vector_name=vector_name,
                    vector=vector,
                    payload=payload,
                )
            )
    return records


def make_triple_embedding_records(
    triples: list[GraphTriple],
    embedder: DenseTextEmbedder,
    vector_name: str = "triple_dense",
    batch_size: int = 32,
    chunks: list[DocumentChunk] | None = None,
) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for start in range(0, len(triples), batch_size):
        batch = triples[start : start + batch_size]
        texts = [triple_text(triple) for triple in batch]
        vectors = embedder.embed_texts(texts)
        for triple, text, vector in zip(batch, texts, vectors):
            chunk = resolved_triple_chunk(triple, chunks or [])
            payload = triple_record_payload(triple, text, chunk)
            record_chunk_id = payload.get("chunk_id")
            records.append(
                EmbeddingRecord(
                    point_id=point_id(triple.triple_id, vector_name),
                    chunk_id=str(record_chunk_id or triple.chunk_id),
                    doc_id=triple.doc_id,
                    vector_name=vector_name,
                    vector=vector,
                    payload=payload,
                )
            )
    return records


def resolved_triple_chunk(triple: GraphTriple, chunks: list[DocumentChunk]) -> DocumentChunk | None:
    if not chunks:
        return None
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    resolved_ids = triple_resolved_chunk_ids(
        triple,
        set(chunk_by_id),
        chunk_id_alias_map(chunks),
        chunk_ids_by_asset_id(chunks),
    )
    return chunk_by_id.get(resolved_ids[0]) if resolved_ids else None


def triple_record_payload(
    triple: GraphTriple,
    text: str,
    chunk: DocumentChunk | None = None,
) -> dict[str, Any]:
    chunk_asset_id_values = chunk_asset_ids(chunk) if chunk is not None else []
    asset_ids = ordered_unique([*sorted(triple_asset_ids(triple)), *chunk_asset_id_values])
    payload: dict[str, Any] = {}
    if chunk is not None:
        payload.update(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "kind": chunk.kind,
                "section": chunk.section.model_dump(),
                "asset_id": asset_ids,
                "asset_ids": asset_ids,
                "source_refs": chunk.source_refs,
                **chunk.metadata,
            }
        )
    else:
        payload.update(
            {
                "chunk_id": triple.chunk_id,
                "doc_id": triple.doc_id,
                "kind": "graph_triple",
                "asset_id": asset_ids,
                "asset_ids": asset_ids,
            }
        )
        for key in ("page_start", "page_end", "page_no"):
            if key in triple.qualifiers:
                payload[key] = triple.qualifiers[key]
    payload.update(
        {
            "record_kind": "graph_triple",
            "triple_id": triple.triple_id,
            "source_triple_chunk_id": triple.chunk_id,
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "text": text,
            "confidence": triple.confidence,
            "qualifiers": triple.qualifiers,
        }
    )
    return payload


def triple_text(triple: GraphTriple) -> str:
    predicate = str(triple.predicate).replace("_", " ")
    parts = [triple.subject, predicate, triple.object]
    parts.extend(triple_qualifier_text_parts(triple.qualifiers))
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def triple_qualifier_text_parts(qualifiers: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    evidence = metadata_text_value(qualifiers.get("evidence"))
    if evidence:
        parts.append(f"evidence {evidence}")
    attributes = metadata_text_items(qualifiers.get("attributes"), limit=6)
    if attributes:
        parts.append(f"attributes {' '.join(attributes)}")
    description = metadata_text_value(qualifiers.get("description"))
    if description and description not in attributes:
        parts.append(f"description {description}")
    location = metadata_first_string(qualifiers, ["location", "position", "region"])
    if location:
        parts.append(f"location {location}")
    bbox_region = metadata_first_string(qualifiers, ["bbox_region", "spatial_region"])
    if not bbox_region:
        bbox_region = bbox_region_from_bbox(
            qualifiers.get("bbox")
            or qualifiers.get("box")
            or qualifiers.get("bounding_box")
            or qualifiers.get("boundingBox")
        ) or ""
    if bbox_region and not location:
        parts.append(f"bbox region {bbox_region}")
    source_field = metadata_text_value(qualifiers.get("source_field"))
    if source_field:
        parts.append(f"source field {source_field}")
    source_key = metadata_text_value(qualifiers.get("source_key"))
    if source_key and source_key != source_field:
        parts.append(f"source key {source_key}")
    return parts


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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


def visual_object_embedding_items(
    assets: list[VisualAsset],
    limit_per_asset: int = 32,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for asset in assets:
        asset_objects: list[dict[str, Any]] = []
        for source_key in VISUAL_OBJECT_METADATA_KEYS:
            asset_objects.extend(
                metadata_object_records(
                    asset.metadata.get(source_key),
                    source_key=source_key,
                    limit=limit_per_asset,
                )
            )
        for source_key in VISUAL_FEATURE_METADATA_KEYS:
            for item in metadata_object_records(
                asset.metadata.get(source_key),
                source_key=source_key,
                limit=limit_per_asset,
            ):
                asset_objects.append(
                    {
                        **item,
                        "visual_feature_type": source_key.removesuffix("s"),
                    }
                )
        for object_index, visual_object in enumerate(
            dedupe_visual_object_records(asset_objects, limit=limit_per_asset)
        ):
            text = visual_object_text(asset, visual_object)
            if not text:
                continue
            item = {
                **asset_object_base_payload(asset),
                **visual_object,
                "object_id": f"{asset.asset_id}:object:{object_index}",
                "object_index": object_index,
                "text": text,
            }
            items.append(item)
    return items


def metadata_object_records(
    value: Any,
    source_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    elif isinstance(value, dict):
        if metadata_first_string(value, VISUAL_OBJECT_LABEL_KEYS):
            candidates = [value]
        else:
            candidates = metadata_object_mapping_items(value)
    else:
        candidates = [value]

    objects: list[dict[str, Any]] = []
    for item in candidates:
        normalized = normalize_metadata_object_record(item, source_key=source_key)
        if normalized:
            objects.append(normalized)
        if len(objects) >= limit:
            break
    return objects


def normalize_metadata_object_record(item: Any, source_key: str) -> dict[str, Any]:
    if isinstance(item, dict):
        label = metadata_first_string(item, VISUAL_OBJECT_LABEL_KEYS)
        if not label:
            return {}
        normalized: dict[str, Any] = {
            "label": label,
            "source_key": metadata_text_value(item.get("source_key")) or source_key,
        }
        attributes = metadata_text_items(
            item.get("attributes") or item.get("features") or item.get("descriptors"),
            limit=6,
        )
        description = metadata_first_string(item, ["description", "summary", "text"])
        if description:
            normalized["description"] = description
            if description not in attributes:
                attributes.append(description)
        if attributes:
            normalized["attributes"] = attributes[:6]
        location = metadata_first_string(item, ["location", "position", "region"])
        if location:
            normalized["location"] = location
        bbox = normalize_bbox(first_present(item, VISUAL_OBJECT_BBOX_KEYS))
        if bbox is not None:
            normalized["bbox"] = bbox
            bbox_region = bbox_region_from_bbox(bbox)
            if bbox_region:
                normalized["bbox_region"] = bbox_region
        else:
            bbox_region = metadata_first_string(item, ["bbox_region", "spatial_region"])
            if bbox_region:
                normalized["bbox_region"] = bbox_region
        confidence = metadata_confidence(item.get("confidence", item.get("score")))
        if confidence is not None:
            normalized["confidence"] = confidence
        return normalized

    label = metadata_text_value(item)
    if not label:
        return {}
    return {"label": label, "source_key": source_key}


def first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def metadata_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().removesuffix("%").strip()
    else:
        text = value
    try:
        confidence = float(text)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    if 0.0 <= confidence <= 1.0:
        return round(confidence, 6)
    return None


def dedupe_visual_object_records(objects: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for item in objects:
        key = (
            metadata_text_value(item.get("label")).casefold(),
            tuple(
                metadata_text_value(value).casefold()
                for value in item.get("attributes", [])
            ),
            metadata_text_value(item.get("description")).casefold(),
            metadata_text_value(item.get("location")).casefold(),
            metadata_text_value(item.get("bbox_region")).casefold(),
            metadata_text_value(item.get("source_key")).casefold(),
            metadata_text_value(item.get("visual_feature_type")).casefold(),
            tuple(item.get("bbox", [])),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def visual_object_text(asset: VisualAsset, visual_object: dict[str, Any]) -> str:
    attributes = metadata_text_items(visual_object.get("attributes"), limit=6)
    feature_type = metadata_text_value(visual_object.get("visual_feature_type"))
    label_prefix = "Visual feature" if feature_type else "Object"
    parts = [
        f"{label_prefix}: {metadata_text_value(visual_object.get('label'))}",
    ]
    if feature_type:
        parts.append(f"Feature type: {feature_type}")
    if attributes:
        parts.append(f"Attributes: {'; '.join(attributes)}")
    description = metadata_text_value(visual_object.get("description"))
    if description and description not in attributes:
        parts.append(f"Description: {description}")
    location = metadata_text_value(visual_object.get("location"))
    if location:
        parts.append(f"Location: {location}")
    bbox_region = metadata_text_value(visual_object.get("bbox_region"))
    if bbox_region and bbox_region != location:
        parts.append(f"Bbox region: {bbox_region}")
    source_key = metadata_text_value(visual_object.get("source_key"))
    if source_key:
        parts.append(f"Source field: {source_key}")
    page_type = metadata_text_value(asset.metadata.get("page_type"))
    if page_type:
        parts.append(f"Page type: {page_type}")
    if asset.caption:
        parts.append(f"Caption: {asset.caption}")
    if asset.vlm_summary:
        parts.append(f"VLM summary: {asset.vlm_summary}")
    return "\n".join(deduplicate_text_parts(parts))


def asset_object_base_payload(asset: VisualAsset) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "doc_id": asset.doc_id,
        "page_no": asset.page_no,
        "kind": asset.kind,
        "caption": asset.caption,
        "vlm_summary": asset.vlm_summary,
        **asset.metadata,
    }


def visual_object_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in item.items()
        if key not in {"attributes"} or value
    }
    payload["record_kind"] = "visual_object"
    payload["asset_ids"] = [payload["asset_id"]]
    return payload


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
            bbox_region = metadata_first_string(item, ["bbox_region", "spatial_region"])
            if not bbox_region:
                bbox_region = bbox_region_from_bbox(
                    item.get("bbox") or item.get("box") or item.get("bounding_box") or item.get("boundingBox")
                ) or ""
            if bbox_region and not location and bbox_region not in attributes:
                attributes.append(bbox_region)
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
        elif isinstance(details, list) and len(details) == 4:
            item = {"label": label_text, "bbox": details}
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

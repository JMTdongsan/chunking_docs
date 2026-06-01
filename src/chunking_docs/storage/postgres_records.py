from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chunking_docs.embeddings.records import visual_object_embedding_items
from chunking_docs.graph.provenance import asset_ids_from_ref, chunk_asset_ids
from chunking_docs.io import read_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, SourceDocument, VisualAsset
from chunking_docs.storage.qdrant_config import qdrant_payload_index_fields
from chunking_docs.storage.qdrant_config import qdrant_payload_index_schemas
from chunking_docs.storage.records import EmbeddingRecord

BM25_TOKEN_MANIFEST = "bm25_tokens.json"
VISUAL_OBJECT_ROW_KEYS = {
    "object_id",
    "doc_id",
    "asset_id",
    "page_no",
    "kind",
    "object_index",
    "label",
    "source_key",
    "visual_feature_type",
    "bbox",
    "bbox_region",
    "attributes",
    "description",
    "location",
    "confidence",
    "text",
}


def document_row(document: SourceDocument) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "source_url": document.source_url,
        "local_path": str(document.local_path),
        "metadata": document.metadata,
    }


def page_row(profile: PageProfile) -> dict[str, Any]:
    return {
        "doc_id": profile.doc_id,
        "page_no": profile.page_no,
        "width": profile.width,
        "height": profile.height,
        "text_quality": profile.text_quality,
        "profile": profile.model_dump(mode="json"),
    }


def chunk_row(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "kind": chunk.kind,
        "section": chunk.section.model_dump(mode="json"),
        "text": chunk.text,
        "metadata": {
            **chunk.metadata,
            "asset_ids": chunk_asset_ids(chunk),
            "source_refs": chunk.source_refs,
        },
    }


def chunk_lexical_token_rows(
    doc_id: str,
    chunks: list[DocumentChunk],
    package_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if package_dir is None:
        return []
    path = package_dir / BM25_TOKEN_MANIFEST
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = payload.get("tokenizer") or {}
    entries = payload.get("chunks") or []
    chunk_doc_ids = {chunk.chunk_id: chunk.doc_id for chunk in chunks}
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        if chunk_id not in chunk_doc_ids:
            continue
        tokens = normalized_tokens(entry.get("tokens") or [])
        rows.append(
            {
                "chunk_id": chunk_id,
                "doc_id": chunk_doc_ids.get(chunk_id) or doc_id,
                "tokenizer": tokenizer,
                "text_char_count": int(entry.get("text_char_count") or 0),
                "token_count": len(tokens),
                "tokens": tokens,
                "metadata": {
                    "manifest_file": path.name,
                    "manifest_chunk_count": len(entries),
                },
            }
        )
    return rows


def asset_row(asset: VisualAsset, base_dir: Path | None = None) -> dict[str, Any]:
    path = asset.path
    if path is not None and base_dir is not None:
        try:
            path = path.relative_to(base_dir)
        except ValueError:
            pass
    return {
        "asset_id": asset.asset_id,
        "doc_id": asset.doc_id,
        "page_no": asset.page_no,
        "kind": asset.kind,
        "path": str(path) if path else None,
        "bbox": list(asset.bbox) if asset.bbox else None,
        "caption": asset.caption,
        "ocr_text": asset.ocr_text,
        "vlm_summary": asset.vlm_summary,
        "metadata": asset.metadata,
    }


def visual_object_rows(assets: list[VisualAsset]) -> list[dict[str, Any]]:
    rows = []
    for item in visual_object_embedding_items(assets):
        rows.append(visual_object_row(item))
    return rows


def visual_object_row(item: dict[str, Any]) -> dict[str, Any]:
    bbox = item.get("bbox")
    metadata = {
        key: value
        for key, value in item.items()
        if key not in VISUAL_OBJECT_ROW_KEYS and value not in (None, "", [], {})
    }
    metadata["record_kind"] = "visual_object"
    return {
        "object_id": str(item["object_id"]),
        "doc_id": str(item["doc_id"]),
        "asset_id": str(item["asset_id"]),
        "page_no": int(item["page_no"]),
        "kind": str(item["kind"]),
        "object_index": int(item["object_index"]),
        "label": str(item["label"]),
        "source_key": item.get("source_key"),
        "visual_feature_type": item.get("visual_feature_type"),
        "bbox": list(bbox) if isinstance(bbox, list) else None,
        "bbox_region": item.get("bbox_region"),
        "attributes": item.get("attributes") or [],
        "description": item.get("description"),
        "location": item.get("location"),
        "confidence": item.get("confidence"),
        "text": str(item["text"]),
        "metadata": metadata,
    }


def chunk_asset_link_rows(
    chunks: list[DocumentChunk],
    valid_asset_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        for asset_id in chunk_asset_ids(chunk):
            if valid_asset_ids is not None and asset_id not in valid_asset_ids:
                continue
            sources = chunk_asset_link_sources(chunk, asset_id)
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "asset_id": asset_id,
                    "doc_id": chunk.doc_id,
                    "source": "+".join(sources) if sources else "unknown",
                    "metadata": {
                        "sources": sources,
                        "source_refs": [
                            ref for ref in chunk.source_refs if asset_id in asset_ids_from_ref(ref)
                        ],
                        "visual_asset_unlinked": bool(chunk.metadata.get("visual_asset_unlinked")),
                        "chunking_strategy": chunk.metadata.get("chunking_strategy"),
                    },
                }
            )
    return rows


def chunk_asset_link_sources(chunk: DocumentChunk, asset_id: str) -> list[str]:
    sources = []
    if asset_id in chunk.asset_ids:
        sources.append("asset_ids")
    if any(asset_id in asset_ids_from_ref(ref) for ref in chunk.source_refs):
        sources.append("source_refs")
    if chunk.metadata.get("visual_asset_unlinked"):
        sources.append("standalone_visual_chunk")
    return sources


def triple_row(triple: GraphTriple) -> dict[str, Any]:
    return {
        "triple_id": triple.triple_id,
        "doc_id": triple.doc_id,
        "chunk_id": triple.chunk_id,
        "subject": triple.subject,
        "predicate": triple.predicate,
        "object": triple.object,
        "qualifiers": triple.qualifiers,
        "confidence": triple.confidence,
    }


def normalized_tokens(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    tokens = []
    for value in values:
        token = str(value).strip()
        if token:
            tokens.append(token)
    return tokens


def embedding_artifact_rows(doc_id: str, package_dir: Path | None = None) -> list[dict[str, Any]]:
    if package_dir is None:
        return []
    manifest_path = package_dir / "embedding_manifest.json"
    if not manifest_path.exists():
        return []

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = str(payload.get("collection") or "document_chunks")
    payload_indexes = payload.get("payload_indexes") or []
    vectors = payload.get("vectors") or {}

    rows = []
    for vector_name, vector in sorted(vectors.items()):
        if not isinstance(vector, dict):
            continue
        row_keys = {"file", "record_count", "dimension", "distance", "note", "bytes", "sha256", "exists"}
        metadata = {key: value for key, value in vector.items() if key not in row_keys}
        metadata["exists"] = bool(vector.get("exists", False))
        metadata["manifest_file"] = manifest_path.name
        metadata["payload_indexes"] = payload_indexes
        metadata["payload_index_fields"] = sorted(qdrant_payload_index_fields(payload_indexes))
        metadata["payload_index_schemas"] = qdrant_payload_index_schemas(payload_indexes)
        rows.append(
            {
                "doc_id": doc_id,
                "vector_name": str(vector_name),
                "collection": collection,
                "file": str(vector.get("file") or ""),
                "record_count": int(vector.get("record_count") or 0),
                "dimension": int(vector.get("dimension") or 0),
                "distance": str(vector.get("distance") or "Cosine"),
                "note": vector.get("note"),
                "bytes": int(vector.get("bytes") or 0),
                "sha256": vector.get("sha256"),
                "metadata": metadata,
            }
        )
    return rows


def embedding_record_rows(doc_id: str, package_dir: Path | None = None) -> list[dict[str, Any]]:
    if package_dir is None:
        return []
    manifest_path = package_dir / "embedding_manifest.json"
    if not manifest_path.exists():
        return []

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = str(payload.get("collection") or "document_chunks")
    vectors = payload.get("vectors") or {}
    if not isinstance(vectors, dict):
        return []

    rows: list[dict[str, Any]] = []
    for vector_name, vector in sorted(vectors.items()):
        if not isinstance(vector, dict):
            continue
        filename = str(vector.get("file") or "").strip()
        if not filename:
            continue
        record_path = package_record_path(package_dir, filename)
        if record_path is None:
            continue
        if not record_path.exists():
            continue
        manifest_dimension = int(vector.get("dimension") or 0)
        manifest_record_count = int(vector.get("record_count") or 0)
        for record in read_jsonl(record_path, EmbeddingRecord):
            target_kind, target_id = embedding_record_target(record)
            rows.append(
                {
                    "point_id": record.point_id,
                    "doc_id": record.doc_id or doc_id,
                    "vector_name": record.vector_name or str(vector_name),
                    "target_id": target_id,
                    "target_kind": target_kind,
                    "vector": record.vector,
                    "dimension": len(record.vector),
                    "payload": record.payload,
                    "metadata": {
                        "collection": collection,
                        "manifest_file": manifest_path.name,
                        "record_file": record_path.relative_to(package_dir.resolve()).as_posix(),
                        "manifest_vector_name": str(vector_name),
                        "manifest_dimension": manifest_dimension,
                        "manifest_record_count": manifest_record_count,
                    },
                }
            )
    return rows


def package_record_path(package_dir: Path, filename: str) -> Path | None:
    package_root = package_dir.resolve()
    record_path = (package_root / filename).resolve()
    try:
        record_path.relative_to(package_root)
    except ValueError:
        return None
    return record_path


def embedding_record_target(record: EmbeddingRecord) -> tuple[str, str]:
    payload = record.payload
    if payload.get("record_kind") == "graph_triple" or payload.get("triple_id"):
        return "triple", payload_id(payload.get("triple_id")) or record.chunk_id
    if record.vector_name == "object_dense" or payload.get("object_id"):
        return "object", payload_id(payload.get("object_id")) or record.chunk_id
    if record.vector_name in {"image_dense", "caption_dense"}:
        return "asset", payload_id(payload.get("asset_id")) or record.chunk_id
    if payload.get("chunk_id"):
        return "chunk", payload_id(payload.get("chunk_id")) or record.chunk_id
    return "record", record.chunk_id


def payload_id(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list) and len(value) == 1:
        return payload_id(value[0])
    return None

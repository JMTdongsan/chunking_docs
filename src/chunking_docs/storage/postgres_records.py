from __future__ import annotations

from pathlib import Path
from typing import Any

from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, SourceDocument, VisualAsset


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
            "asset_ids": chunk.asset_ids,
            "source_refs": chunk.source_refs,
        },
    }


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

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset


class RAGContextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    kind: str
    text: str
    section: str = ""
    asset_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    score: float | None = None
    sources: list[str] = Field(default_factory=list)
    role: str = "hit"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGContextAsset(BaseModel):
    asset_id: str
    page_no: int
    kind: str
    caption: str | None = None
    ocr_text: str | None = None
    vlm_summary: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGContextTriple(BaseModel):
    triple_id: str
    chunk_id: str
    subject: str
    predicate: str
    object: str
    confidence: float | None = None
    qualifiers: dict[str, Any] = Field(default_factory=dict)


class RAGContextBundle(BaseModel):
    query: str
    chunks: list[RAGContextChunk]
    assets: list[RAGContextAsset] = Field(default_factory=list)
    triples: list[RAGContextTriple] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_context_bundle(
    query: str,
    hits,
    chunks: list[DocumentChunk] | None = None,
    assets: list[VisualAsset] | None = None,
    triples: list[GraphTriple] | None = None,
    max_chars_per_chunk: int = 1400,
    max_chars_per_asset_text: int = 1400,
    include_evidence: bool = True,
    neighbor_window: int = 0,
    include_assets: bool = True,
    include_triples: bool = True,
) -> RAGContextBundle:
    context_chunks: list[RAGContextChunk] = []
    seen_chunks = set()
    hit_chunk_ids = set()

    for rank, hit in enumerate(hits, start=1):
        chunk = getattr(hit, "chunk", None)
        if chunk is None:
            continue
        hit_chunk_ids.add(chunk.chunk_id)
        if chunk.chunk_id not in seen_chunks:
            context_chunks.append(
                context_chunk(
                    chunk,
                    role="hit",
                    max_chars=max_chars_per_chunk,
                    score=getattr(hit, "score", None),
                    sources=list(getattr(hit, "sources", [])),
                    rank=rank,
                    payloads=list(getattr(hit, "payloads", [])),
                )
            )
            seen_chunks.add(chunk.chunk_id)

        if not include_evidence:
            continue
        for evidence_chunk in getattr(hit, "evidence_chunks", []):
            hit_chunk_ids.add(evidence_chunk.chunk_id)
            if evidence_chunk.chunk_id in seen_chunks:
                continue
            context_chunks.append(
                context_chunk(
                    evidence_chunk,
                    role="evidence",
                    max_chars=max_chars_per_chunk,
                    score=getattr(hit, "score", None),
                    sources=list(getattr(hit, "sources", [])),
                    rank=rank,
                    parent_chunk_id=chunk.chunk_id,
                )
            )
            seen_chunks.add(evidence_chunk.chunk_id)

    if chunks and neighbor_window > 0:
        add_neighbor_chunks(
            context_chunks,
            seen_chunks=seen_chunks,
            source_chunk_ids=hit_chunk_ids,
            chunks=chunks,
            neighbor_window=neighbor_window,
            max_chars_per_chunk=max_chars_per_chunk,
        )

    selected_chunk_ids = {chunk.chunk_id for chunk in context_chunks}
    selected_assets = []
    if include_assets and assets:
        selected_assets = context_assets(
            assets,
            context_chunks,
            max_chars_per_asset_text=max_chars_per_asset_text,
        )

    selected_triples = []
    if include_triples and triples:
        selected_triples = context_triples(triples, selected_chunk_ids)

    return RAGContextBundle(
        query=query,
        chunks=context_chunks,
        assets=selected_assets,
        triples=selected_triples,
        metadata=context_bundle_metadata(
            context_chunks,
            selected_assets,
            selected_triples,
            max_chars_per_chunk=max_chars_per_chunk,
            max_chars_per_asset_text=max_chars_per_asset_text,
            neighbor_window=neighbor_window,
        ),
    )


def context_chunk(
    chunk: DocumentChunk,
    role: str,
    max_chars: int,
    score: float | None,
    sources: list[str],
    rank: int,
    parent_chunk_id: str | None = None,
    payloads: list[dict[str, Any]] | None = None,
) -> RAGContextChunk:
    metadata = {
        "rank": rank,
        **chunk.metadata,
    }
    if parent_chunk_id:
        metadata["retrieved_parent_chunk_id"] = parent_chunk_id
    retrieval_refs = retrieval_payload_refs(payloads or [])
    retrieved_asset_ids = sorted(
        {ref["asset_id"] for ref in retrieval_refs if "asset_id" in ref}
    )
    if retrieval_refs:
        metadata["retrieval_payload_refs"] = retrieval_refs
    if retrieved_asset_ids:
        metadata["retrieved_asset_ids"] = retrieved_asset_ids
    return RAGContextChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        kind=str(chunk.kind),
        text=trim_text(chunk.text, max_chars),
        section=chunk.section.label(),
        asset_ids=chunk.asset_ids,
        source_refs=chunk.source_refs,
        score=score,
        sources=sources,
        role=role,
        metadata=metadata,
    )


def add_neighbor_chunks(
    context_chunks: list[RAGContextChunk],
    seen_chunks: set[str],
    source_chunk_ids: set[str],
    chunks: list[DocumentChunk],
    neighbor_window: int,
    max_chars_per_chunk: int,
):
    ordered = sorted(chunks, key=chunk_order_key)
    positions = {chunk.chunk_id: index for index, chunk in enumerate(ordered)}
    for source_chunk_id in sorted(source_chunk_ids, key=lambda chunk_id: positions.get(chunk_id, -1)):
        position = positions.get(source_chunk_id)
        if position is None:
            continue
        for offset in range(-neighbor_window, neighbor_window + 1):
            if offset == 0:
                continue
            neighbor_index = position + offset
            if neighbor_index < 0 or neighbor_index >= len(ordered):
                continue
            neighbor = ordered[neighbor_index]
            source = ordered[position]
            if neighbor.doc_id != source.doc_id or neighbor.chunk_id in seen_chunks:
                continue
            context_chunks.append(
                context_chunk(
                    neighbor,
                    role="neighbor",
                    max_chars=max_chars_per_chunk,
                    score=None,
                    sources=["neighbor"],
                    rank=0,
                    parent_chunk_id=source_chunk_id,
                )
            )
            context_chunks[-1].metadata.pop("retrieved_parent_chunk_id", None)
            context_chunks[-1].metadata["neighbor_source_chunk_id"] = source_chunk_id
            context_chunks[-1].metadata["neighbor_offset"] = offset
            seen_chunks.add(neighbor.chunk_id)


def chunk_order_key(chunk: DocumentChunk):
    return (
        chunk.doc_id,
        chunk.page_start,
        chunk.page_end,
        str(chunk.kind),
        safe_int(chunk.metadata.get("subchunk_index", 0)),
        chunk.chunk_id,
    )


def safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def context_assets(
    assets: list[VisualAsset],
    chunks: list[RAGContextChunk],
    max_chars_per_asset_text: int,
) -> list[RAGContextAsset]:
    asset_ids = {asset_id for chunk in chunks for asset_id in chunk.asset_ids}
    selected = []
    seen = set()
    for asset in assets:
        if asset.asset_id not in asset_ids or asset.asset_id in seen:
            continue
        seen.add(asset.asset_id)
        caption, ocr_text, vlm_summary, text_metadata = context_asset_text_fields(
            asset,
            max_chars_per_asset_text=max_chars_per_asset_text,
        )
        selected.append(
            RAGContextAsset(
                asset_id=asset.asset_id,
                page_no=asset.page_no,
                kind=str(asset.kind),
                caption=caption,
                ocr_text=ocr_text,
                vlm_summary=vlm_summary,
                path=str(asset.path) if asset.path else None,
                metadata={**asset.metadata, "context_text": text_metadata},
            )
        )
    return selected


def context_triples(triples: list[GraphTriple], chunk_ids: set[str]) -> list[RAGContextTriple]:
    selected = []
    seen = set()
    for triple in triples:
        if triple.chunk_id not in chunk_ids or triple.triple_id in seen:
            continue
        seen.add(triple.triple_id)
        selected.append(
            RAGContextTriple(
                triple_id=triple.triple_id,
                chunk_id=triple.chunk_id,
                subject=triple.subject,
                predicate=triple.predicate,
                object=triple.object,
                confidence=triple.confidence,
                qualifiers=triple.qualifiers,
            )
        )
    return selected


def context_bundle_metadata(
    chunks: list[RAGContextChunk],
    assets: list[RAGContextAsset],
    triples: list[RAGContextTriple],
    max_chars_per_chunk: int,
    max_chars_per_asset_text: int,
    neighbor_window: int,
) -> dict[str, Any]:
    role_counts = count_values(chunk.role for chunk in chunks)
    source_counts = count_values(source for chunk in chunks for source in chunk.sources)
    source_family_counts = count_values(
        context_source_family(source) for chunk in chunks for source in chunk.sources
    )
    kind_counts = count_values(chunk.kind for chunk in chunks)
    pages = context_pages(chunks)
    asset_text = context_asset_text_summary(assets)
    retrieved_asset_ids = sorted(
        {
            asset_id
            for chunk in chunks
            for asset_id in chunk.metadata.get("retrieved_asset_ids", [])
        }
    )
    return {
        "hit_count": len(chunks),
        "chunk_count": len(chunks),
        "hit_chunk_count": role_counts.get("hit", 0),
        "evidence_chunk_count": role_counts.get("evidence", 0),
        "neighbor_chunk_count": role_counts.get("neighbor", 0),
        "asset_count": len(assets),
        "retrieved_asset_count": len(retrieved_asset_ids),
        "retrieved_asset_ids": retrieved_asset_ids,
        "triple_count": len(triples),
        "page_count": len(pages),
        "pages": pages,
        "page_ranges": context_page_ranges(chunks),
        "doc_ids": sorted({chunk.doc_id for chunk in chunks}),
        "role_counts": role_counts,
        "kind_counts": kind_counts,
        "source_counts": source_counts,
        "source_family_counts": source_family_counts,
        "has_dense_text_context": source_family_counts.get("dense_text", 0) > 0,
        "has_lexical_context": source_family_counts.get("lexical", 0) > 0,
        "has_visual_context": bool(assets) or source_family_counts.get("visual", 0) > 0,
        "has_graph_context": bool(triples) or source_family_counts.get("graph", 0) > 0,
        "max_chars_per_chunk": max_chars_per_chunk,
        "max_chars_per_asset_text": max_chars_per_asset_text,
        "asset_text_char_count": asset_text["char_count"],
        "asset_context_char_count": asset_text["context_char_count"],
        "asset_text_truncated_count": asset_text["truncated_count"],
        "asset_text_truncated_fields": asset_text["truncated_fields"],
        "neighbor_window": max(0, neighbor_window),
    }


def context_source_family(source: str) -> str:
    normalized = source.strip().lower()
    if "caption_dense" in normalized or "image_dense" in normalized:
        return "visual"
    if normalized == "dense" or "text_dense" in normalized:
        return "dense_text"
    if normalized.startswith("rerank:"):
        return "reranker"
    if normalized == "bm25" or "lexical" in normalized:
        return "lexical"
    if normalized == "graph":
        return "graph"
    return normalized.split(":", 1)[0] if normalized else "unknown"


def context_pages(chunks: list[RAGContextChunk]) -> list[int]:
    pages = set()
    for chunk in chunks:
        start = min(chunk.page_start, chunk.page_end)
        end = max(chunk.page_start, chunk.page_end)
        pages.update(range(start, end + 1))
    return sorted(pages)


def context_page_ranges(chunks: list[RAGContextChunk]) -> list[dict[str, int | str]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
        }
        for chunk in chunks
    ]


def count_values(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def retrieval_payload_refs(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    seen = set()
    for payload in payloads:
        ref = retrieval_payload_ref(payload)
        if not ref:
            continue
        key = tuple(sorted(ref.items()))
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    return refs


def retrieval_payload_ref(payload: dict[str, Any]) -> dict[str, Any]:
    ref = {}
    for key in (
        "asset_id",
        "chunk_id",
        "doc_id",
        "page_no",
        "page_start",
        "page_end",
        "kind",
    ):
        value = payload.get(key)
        if value is not None:
            ref[key] = value
    return ref


def context_asset_text_fields(
    asset: VisualAsset,
    max_chars_per_asset_text: int,
) -> tuple[str | None, str | None, str | None, dict[str, Any]]:
    raw_values = {
        "caption": asset.caption,
        "ocr_text": asset.ocr_text,
        "vlm_summary": asset.vlm_summary,
    }
    trimmed_values = {
        field: trim_optional_text(value, max_chars_per_asset_text)
        for field, value in raw_values.items()
    }
    char_counts = {
        field: len(normalize_text(value))
        for field, value in raw_values.items()
        if normalize_text(value)
    }
    context_char_counts = {
        field: len(normalize_text(value))
        for field, value in trimmed_values.items()
        if normalize_text(value)
    }
    truncated_fields = [
        field
        for field, value in raw_values.items()
        if is_text_truncated(value, max_chars_per_asset_text)
    ]
    return (
        trimmed_values["caption"],
        trimmed_values["ocr_text"],
        trimmed_values["vlm_summary"],
        {
            "max_chars_per_field": max_chars_per_asset_text,
            "char_counts": char_counts,
            "context_char_counts": context_char_counts,
            "truncated_fields": truncated_fields,
        },
    )


def context_asset_text_summary(assets: list[RAGContextAsset]) -> dict[str, Any]:
    char_count = 0
    context_char_count = 0
    truncated_count = 0
    truncated_fields = Counter()
    for asset in assets:
        text_metadata = asset.metadata.get("context_text", {})
        char_count += sum(text_metadata.get("char_counts", {}).values())
        context_char_count += sum(text_metadata.get("context_char_counts", {}).values())
        fields = text_metadata.get("truncated_fields", [])
        if fields:
            truncated_count += 1
        truncated_fields.update(fields)
    return {
        "char_count": char_count,
        "context_char_count": context_char_count,
        "truncated_count": truncated_count,
        "truncated_fields": dict(sorted(truncated_fields.items())),
    }


def trim_optional_text(text: str | None, max_chars: int) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    return trim_text(normalized, max_chars)


def is_text_truncated(text: str | None, max_chars: int) -> bool:
    normalized = normalize_text(text)
    return max_chars > 0 and len(normalized) > max_chars


def normalize_text(text: str | None) -> str:
    return text.strip() if text else ""


def trim_text(text: str, max_chars: int) -> str:
    normalized = text.strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return "." * max_chars
    return normalized[: max_chars - 3].rstrip() + "..."

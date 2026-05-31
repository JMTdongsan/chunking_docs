from __future__ import annotations

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
    assets: list[VisualAsset] | None = None,
    triples: list[GraphTriple] | None = None,
    max_chars_per_chunk: int = 1400,
    include_evidence: bool = True,
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

    selected_assets = []
    if include_assets and assets:
        selected_assets = context_assets(assets, context_chunks)

    selected_triples = []
    if include_triples and triples:
        selected_triples = context_triples(triples, hit_chunk_ids)

    return RAGContextBundle(
        query=query,
        chunks=context_chunks,
        assets=selected_assets,
        triples=selected_triples,
        metadata={
            "hit_count": len(context_chunks),
            "asset_count": len(selected_assets),
            "triple_count": len(selected_triples),
            "max_chars_per_chunk": max_chars_per_chunk,
        },
    )


def context_chunk(
    chunk: DocumentChunk,
    role: str,
    max_chars: int,
    score: float | None,
    sources: list[str],
    rank: int,
    parent_chunk_id: str | None = None,
) -> RAGContextChunk:
    metadata = {
        "rank": rank,
        **chunk.metadata,
    }
    if parent_chunk_id:
        metadata["retrieved_parent_chunk_id"] = parent_chunk_id
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


def context_assets(assets: list[VisualAsset], chunks: list[RAGContextChunk]) -> list[RAGContextAsset]:
    asset_ids = {asset_id for chunk in chunks for asset_id in chunk.asset_ids}
    selected = []
    seen = set()
    for asset in assets:
        if asset.asset_id not in asset_ids or asset.asset_id in seen:
            continue
        seen.add(asset.asset_id)
        selected.append(
            RAGContextAsset(
                asset_id=asset.asset_id,
                page_no=asset.page_no,
                kind=str(asset.kind),
                caption=asset.caption,
                ocr_text=asset.ocr_text,
                vlm_summary=asset.vlm_summary,
                path=str(asset.path) if asset.path else None,
                metadata=asset.metadata,
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


def trim_text(text: str, max_chars: int) -> str:
    normalized = text.strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "..."

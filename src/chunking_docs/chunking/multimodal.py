from __future__ import annotations

import hashlib
from typing import Literal

from chunking_docs.chunking.hierarchical import build_hierarchical_chunks, chunk_visual_context
from chunking_docs.chunking.semantic_splitter import semantic_subchunks
from chunking_docs.embeddings.records import asset_text
from chunking_docs.graph.provenance import chunk_asset_ids
from chunking_docs.models import ChunkKind, DocumentChunk, VisualAsset

ChunkStrategy = Literal["page", "semantic", "multimodal", "hierarchical"]


def build_strategy_chunks(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    strategy: ChunkStrategy,
    max_chars: int = 1600,
    overlap_chars: int = 180,
    min_chars: int = 180,
    include_context_prefix: bool = True,
    parent_max_chars: int = 900,
    visual_context_chars: int = 700,
) -> list[DocumentChunk]:
    if strategy == "page":
        return contextualize_chunks(chunks, include_context_prefix=include_context_prefix)
    if strategy == "semantic":
        split = semantic_subchunks(
            contextualize_chunks(chunks, include_context_prefix=include_context_prefix),
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )
        return split
    if strategy == "multimodal":
        contextualized = contextualize_chunks(chunks, include_context_prefix=include_context_prefix)
        visual_contextualized = add_visual_context_to_chunks(
            contextualized,
            assets,
            max_chars=visual_context_chars,
        )
        base = semantic_subchunks(
            visual_contextualized,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )
        return base + visual_asset_chunks(chunks, assets, include_context_prefix=include_context_prefix)
    if strategy == "hierarchical":
        return build_hierarchical_chunks(
            contextualize_chunks(chunks, include_context_prefix=include_context_prefix),
            assets,
            split_fn=semantic_subchunks,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
            parent_max_chars=parent_max_chars,
            visual_context_chars=visual_context_chars,
        )
    raise ValueError(f"Unsupported chunk strategy: {strategy}")


def contextualize_chunks(
    chunks: list[DocumentChunk],
    include_context_prefix: bool = True,
) -> list[DocumentChunk]:
    if not include_context_prefix:
        return list(chunks)
    updated = []
    for chunk in chunks:
        prefix = context_prefix(chunk)
        if not prefix or chunk.text.startswith(prefix):
            updated.append(chunk)
            continue
        updated.append(
            chunk.model_copy(
                update={
                    "text": f"{prefix}\n{chunk.text}".strip(),
                    "metadata": {
                        **chunk.metadata,
                        "context_prefix_added": True,
                    },
                }
            )
        )
    return updated


def add_visual_context_to_chunks(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    max_chars: int = 700,
) -> list[DocumentChunk]:
    if max_chars <= 0 or not assets:
        return list(chunks)
    assets_by_id = {asset.asset_id: asset for asset in assets}
    updated = []
    for chunk in chunks:
        visual_context = chunk_visual_context(chunk, assets_by_id, max_chars=max_chars)
        if not visual_context:
            updated.append(chunk)
            continue
        block = f"Visual context:\n{visual_context}"
        if block in chunk.text:
            updated.append(chunk)
            continue
        updated.append(
            chunk.model_copy(
                update={
                    "text": "\n\n".join([chunk.text.rstrip(), block]).strip(),
                    "metadata": {
                        **chunk.metadata,
                        "visual_context_added": True,
                    },
                }
            )
        )
    return updated


def visual_asset_chunks(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    include_context_prefix: bool = True,
) -> list[DocumentChunk]:
    chunk_by_asset_id = {
        asset_id: chunk
        for chunk in chunks
        for asset_id in chunk_asset_ids(chunk)
    }
    results = []
    for asset in assets:
        text = asset_text(asset)
        if not text:
            continue
        parent = chunk_by_asset_id.get(asset.asset_id)
        if parent is None:
            continue
        prefix = context_prefix(parent) if include_context_prefix else ""
        body = "\n".join(
            part
            for part in [
                prefix,
                f"Visual asset kind: {asset.kind}",
                f"Asset page: {asset.page_no}",
                text,
            ]
            if part
        )
        metadata = {
            **parent.metadata,
            "parent_chunk_id": parent.chunk_id,
            "asset_id": asset.asset_id,
            "asset_kind": str(asset.kind),
            "chunking_strategy": "visual_asset_text",
        }
        results.append(
            DocumentChunk(
                chunk_id=visual_chunk_id(parent.chunk_id, asset.asset_id),
                doc_id=parent.doc_id,
                page_start=asset.page_no,
                page_end=asset.page_no,
                kind=asset_chunk_kind(asset),
                text=body,
                section=parent.section,
                asset_ids=[asset.asset_id],
                source_refs=[f"asset:{asset.asset_id}"],
                metadata=metadata,
            )
        )
    return results


def context_prefix(chunk: DocumentChunk) -> str:
    parts = [f"Page range: {chunk.page_start}-{chunk.page_end}"]
    section_label = chunk.section.label() or chunk.metadata.get("section_label")
    if section_label:
        parts.insert(0, f"Section: {section_label}")
    return "\n".join(parts)


def asset_chunk_kind(asset: VisualAsset) -> ChunkKind:
    mapping = {
        "table": ChunkKind.TABLE,
        "figure": ChunkKind.FIGURE,
        "map": ChunkKind.MAP,
        "chart": ChunkKind.FIGURE,
    }
    return mapping.get(str(asset.kind), ChunkKind.PAGE_SUMMARY)


def visual_chunk_id(parent_chunk_id: str, asset_id: str) -> str:
    raw = f"{parent_chunk_id}:visual:{asset_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]

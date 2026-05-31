from __future__ import annotations

import hashlib

from chunking_docs.embeddings.records import asset_text
from chunking_docs.models import ChunkKind, DocumentChunk, VisualAsset


def build_hierarchical_chunks(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    split_fn,
    max_chars: int = 1200,
    overlap_chars: int = 160,
    min_chars: int = 140,
    parent_max_chars: int = 900,
    visual_context_chars: int = 700,
) -> list[DocumentChunk]:
    """Build coarse parent chunks plus fine child chunks for retrieval experiments."""

    assets_by_id = {asset.asset_id: asset for asset in assets}
    parent_chunks = [
        parent_context_chunk(
            chunk,
            assets_by_id=assets_by_id,
            parent_max_chars=parent_max_chars,
            visual_context_chars=visual_context_chars,
        )
        for chunk in chunks
    ]
    child_sources = [
        child_source_chunk(chunk, assets_by_id=assets_by_id, visual_context_chars=visual_context_chars)
        for chunk in chunks
    ]
    children = split_fn(child_sources, max_chars=max_chars, overlap_chars=overlap_chars, min_chars=min_chars)
    child_chunks = [normalize_child_chunk(child) for child in children]
    return parent_chunks + child_chunks


def parent_context_chunk(
    chunk: DocumentChunk,
    assets_by_id: dict[str, VisualAsset],
    parent_max_chars: int,
    visual_context_chars: int,
) -> DocumentChunk:
    source_text = trim_text(chunk.text, parent_max_chars)
    visual_context = chunk_visual_context(chunk, assets_by_id, visual_context_chars)
    body = "\n\n".join(
        part
        for part in [
            source_text,
            f"Visual context:\n{visual_context}" if visual_context else "",
        ]
        if part
    )
    return chunk.model_copy(
        update={
            "chunk_id": hierarchical_parent_id(chunk.chunk_id),
            "kind": ChunkKind.PAGE_SUMMARY if chunk.kind == ChunkKind.TEXT else chunk.kind,
            "text": body,
            "metadata": {
                **chunk.metadata,
                "source_chunk_id": chunk.chunk_id,
                "chunking_strategy": "hierarchical_parent",
                "retrieval_role": "parent",
            },
        }
    )


def child_source_chunk(
    chunk: DocumentChunk,
    assets_by_id: dict[str, VisualAsset],
    visual_context_chars: int,
) -> DocumentChunk:
    visual_context = chunk_visual_context(chunk, assets_by_id, visual_context_chars)
    body = "\n\n".join(
        part
        for part in [
            chunk.text,
            f"Visual context:\n{visual_context}" if visual_context else "",
        ]
        if part
    )
    return chunk.model_copy(
        update={
            "text": body,
            "metadata": {
                **chunk.metadata,
                "source_chunk_id": chunk.chunk_id,
                "hierarchical_parent_chunk_id": hierarchical_parent_id(chunk.chunk_id),
            },
        }
    )


def normalize_child_chunk(chunk: DocumentChunk) -> DocumentChunk:
    source_chunk_id = str(
        chunk.metadata.get("source_chunk_id")
        or chunk.metadata.get("parent_chunk_id")
        or chunk.chunk_id
    )
    subchunk_index = int(chunk.metadata.get("subchunk_index", 0))
    parent_id = hierarchical_parent_id(source_chunk_id)
    return chunk.model_copy(
        update={
            "chunk_id": hierarchical_child_id(source_chunk_id, subchunk_index),
            "metadata": {
                **chunk.metadata,
                "source_chunk_id": source_chunk_id,
                "parent_chunk_id": source_chunk_id,
                "hierarchical_parent_chunk_id": parent_id,
                "chunking_strategy": "hierarchical_child",
                "retrieval_role": "child",
            },
            "source_refs": stable_refs([*chunk.source_refs, f"parent:{parent_id}"]),
        }
    )


def chunk_visual_context(
    chunk: DocumentChunk,
    assets_by_id: dict[str, VisualAsset],
    max_chars: int,
) -> str:
    entries = []
    for asset_id in chunk.asset_ids:
        asset = assets_by_id.get(asset_id)
        if asset is None:
            continue
        text = asset_text(asset)
        if not text:
            continue
        entries.append(f"- {asset.kind} page {asset.page_no}: {text}")
    return trim_text("\n".join(entries), max_chars)


def trim_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "..."


def hierarchical_parent_id(source_chunk_id: str) -> str:
    raw = f"{source_chunk_id}:hierarchical:parent".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def hierarchical_child_id(source_chunk_id: str, index: int) -> str:
    raw = f"{source_chunk_id}:hierarchical:child:{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def stable_refs(refs: list[str]) -> list[str]:
    seen = set()
    result = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        result.append(ref)
    return result

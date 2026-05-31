from __future__ import annotations

import uuid

from chunking_docs.embeddings.interfaces import DenseTextEmbedder
from chunking_docs.embeddings.interfaces import DenseImageEmbedder
from chunking_docs.models import DocumentChunk, VisualAsset
from chunking_docs.storage.records import EmbeddingRecord


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
                        "asset_ids": chunk.asset_ids,
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


def asset_text(asset: VisualAsset) -> str:
    return "\n".join(
        part
        for part in [asset.caption or "", asset.ocr_text or "", asset.vlm_summary or ""]
        if part.strip()
    )

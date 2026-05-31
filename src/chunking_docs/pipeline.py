from __future__ import annotations

import json
from pathlib import Path

from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.embeddings.bm25 import BM25LexicalIndex
from chunking_docs.embeddings.interfaces import HashingImageEmbedder, HashingTextEmbedder
from chunking_docs.embeddings.records import (
    make_caption_embedding_records,
    make_image_embedding_records,
    make_text_embedding_records,
)
from chunking_docs.graph.heuristics import section_triples
from chunking_docs.ingest.pdf_loader import load_source_document
from chunking_docs.io import write_jsonl
from chunking_docs.models import ProcessingManifest
from chunking_docs.vision.assets import attach_assets_to_chunks, build_page_assets


def build_processing_package(
    pdf_path: Path,
    output_dir: Path,
    source_url: str | None = None,
    title: str | None = None,
    render_zoom: float = 1.5,
    dry_run_embeddings: bool = True,
) -> ProcessingManifest:
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_source_document(pdf_path, title=title, source_url=source_url)
    profiles = profile_pdf(pdf_path, source.doc_id)
    chunks = page_level_chunks(pdf_path, source.doc_id, profiles)
    assets = build_page_assets(
        pdf_path=pdf_path,
        doc_id=source.doc_id,
        profiles=profiles,
        output_dir=output_dir / "assets",
        zoom=render_zoom,
    )
    chunks = attach_assets_to_chunks(chunks, assets)
    triples = section_triples(chunks)

    manifest = ProcessingManifest(
        doc=source,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        metadata={
            "profile_summary": summarize_profiles(profiles),
            "embedding_mode": "hashing_dry_run" if dry_run_embeddings else "external",
        },
    )
    write_package(output_dir, manifest, dry_run_embeddings=dry_run_embeddings)
    return manifest


def write_package(
    output_dir: Path,
    manifest: ProcessingManifest,
    dry_run_embeddings: bool = True,
) -> None:
    write_jsonl(output_dir / "pages.jsonl", manifest.profiles)
    write_jsonl(output_dir / "chunks.jsonl", manifest.chunks)
    write_jsonl(output_dir / "assets.jsonl", manifest.assets)
    write_jsonl(output_dir / "triples.jsonl", manifest.triples)
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    bm25 = BM25LexicalIndex(manifest.chunks)
    bm25.dump_manifest(output_dir / "bm25_tokens.json")

    if dry_run_embeddings:
        embedder = HashingTextEmbedder()
        image_embedder = HashingImageEmbedder(embedding_dim=embedder.embedding_dim)
        records = make_text_embedding_records(manifest.chunks, embedder)
        image_records = make_image_embedding_records(manifest.assets, image_embedder)
        caption_records = make_caption_embedding_records(manifest.assets, embedder)
        write_jsonl(output_dir / "qdrant_text_records.jsonl", records)
        write_jsonl(output_dir / "qdrant_image_records.jsonl", image_records)
        write_jsonl(output_dir / "qdrant_caption_records.jsonl", caption_records)
        (output_dir / "qdrant_collection.json").write_text(
            json.dumps(
                {
                    "collection": "planning_chunks",
                    "named_vectors": {
                        "text_dense": {
                            "size": embedder.embedding_dim,
                            "distance": "Cosine",
                            "note": "HashingTextEmbedder dry-run dimension. Replace with real dense model dimension.",
                        },
                        "image_dense": {
                            "size": image_embedder.embedding_dim,
                            "distance": "Cosine",
                            "note": "HashingImageEmbedder dry-run dimension. Replace with real image model dimension.",
                        },
                        "caption_dense": {
                            "size": embedder.embedding_dim,
                            "distance": "Cosine",
                            "note": "Caption text dry-run dimension. Replace with real dense model dimension.",
                        },
                    },
                    "payload_indexes": [
                        "doc_id",
                        "chunk_id",
                        "asset_id",
                        "kind",
                        "page_no",
                        "page_start",
                        "page_end",
                        "section.chapter",
                        "section.issue",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def rebuild_search_artifacts(output_dir: Path, chunks) -> None:
    bm25 = BM25LexicalIndex(chunks)
    bm25.dump_manifest(output_dir / "bm25_tokens.json")

    embedder = HashingTextEmbedder()
    records = make_text_embedding_records(chunks, embedder)
    write_jsonl(output_dir / "qdrant_text_records.jsonl", records)

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.chunking.section_map import SectionRange
from chunking_docs.chunking.semantic_splitter import semantic_subchunks
from chunking_docs.embeddings.bm25 import BM25LexicalIndex, chunk_lexical_texts
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.embeddings.interfaces import (
    DenseImageEmbedder,
    DenseTextEmbedder,
    HashingImageEmbedder,
    HashingTextEmbedder,
)
from chunking_docs.embeddings.records import (
    make_caption_embedding_records,
    make_image_embedding_records,
    make_object_embedding_records,
    make_text_embedding_records,
    make_triple_embedding_records,
)
from chunking_docs.graph.heuristics import section_triples
from chunking_docs.ingest.pdf_loader import load_source_document
from chunking_docs.ingest.tables import extract_pdf_tables, visual_table_chunks_from_assets
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import (
    DocumentChunk,
    GraphTriple,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    VisualAsset,
)
from chunking_docs.storage.qdrant_config import QDRANT_PAYLOAD_INDEXES, QDRANT_RECORD_FILES
from chunking_docs.vision.assets import attach_assets_to_chunks, build_page_assets, merge_visual_assets


def build_processing_package(
    pdf_path: Path,
    output_dir: Path,
    source_url: str | None = None,
    title: str | None = None,
    render_zoom: float = 1.5,
    dry_run_embeddings: bool = True,
    section_ranges: list[SectionRange] | None = None,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    extract_tables: bool = True,
) -> ProcessingManifest:
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_source_document(pdf_path, title=title, source_url=source_url)
    profiles = profile_pdf(pdf_path, source.doc_id)
    page_chunks = page_level_chunks(pdf_path, source.doc_id, profiles, section_ranges=section_ranges)
    page_assets = build_page_assets(
        pdf_path=pdf_path,
        doc_id=source.doc_id,
        profiles=profiles,
        output_dir=output_dir / "assets",
        zoom=render_zoom,
        section_ranges=section_ranges,
    )
    table_assets, table_chunks = ([], [])
    if extract_tables:
        table_assets, table_chunks = extract_pdf_tables(
            pdf_path=pdf_path,
            doc_id=source.doc_id,
            output_dir=output_dir / "assets",
            section_ranges=section_ranges,
            zoom=render_zoom,
        )
    assets = merge_visual_assets(page_assets, table_assets)
    visual_table_chunks = visual_table_chunks_from_assets(
        assets,
        existing_chunks=table_chunks,
        section_ranges=section_ranges,
    )
    chunks = [*attach_assets_to_chunks(page_chunks, assets), *table_chunks, *visual_table_chunks]
    triples = section_triples(chunks)

    manifest = ProcessingManifest(
        doc=source,
        profiles=profiles,
        chunks=chunks,
        assets=assets,
        triples=triples,
        metadata=package_metadata(
            pdf_path=pdf_path,
            profiles=profiles,
            render_zoom=render_zoom,
            dry_run_embeddings=dry_run_embeddings,
            section_ranges=section_ranges,
            tokenizer_config=tokenizer_config,
            extract_tables=extract_tables,
            table_count=len(table_chunks) + len(visual_table_chunks),
            table_asset_count=table_asset_count(assets),
        ),
    )
    write_package(
        output_dir,
        manifest,
        dry_run_embeddings=dry_run_embeddings,
        tokenizer_config=tokenizer_config,
    )
    return manifest


def write_package(
    output_dir: Path,
    manifest: ProcessingManifest,
    dry_run_embeddings: bool = True,
    tokenizer_config: LexicalTokenizerConfig | None = None,
) -> None:
    write_jsonl(output_dir / "pages.jsonl", manifest.profiles)
    write_jsonl(output_dir / "chunks.jsonl", manifest.chunks)
    write_jsonl(output_dir / "assets.jsonl", manifest.assets)
    write_jsonl(output_dir / "triples.jsonl", manifest.triples)
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    bm25 = BM25LexicalIndex(
        manifest.chunks,
        tokenizer_config=tokenizer_config,
        texts=chunk_lexical_texts(manifest.chunks, manifest.assets),
    )
    bm25.dump_manifest(output_dir / "bm25_tokens.json")

    if dry_run_embeddings:
        embedder = HashingTextEmbedder()
        image_embedder = HashingImageEmbedder(embedding_dim=embedder.embedding_dim)
        write_embedding_artifacts(
            output_dir=output_dir,
            chunks=manifest.chunks,
            assets=manifest.assets,
            triples=manifest.triples,
            text_embedder=embedder,
            image_embedder=image_embedder,
            caption_embedder=embedder,
            object_embedder=embedder,
            triple_embedder=embedder,
            vector_notes=hashing_vector_notes(include_visual=True, include_triples=True),
            vector_metadata=hashing_vector_metadata(
                include_visual=True,
                include_triples=True,
                embedding_dim=embedder.embedding_dim,
            ),
        )


def package_metadata(
    pdf_path: Path,
    profiles: list[PageProfile],
    render_zoom: float,
    dry_run_embeddings: bool,
    section_ranges: list[SectionRange] | None,
    tokenizer_config: LexicalTokenizerConfig | None,
    extract_tables: bool,
    table_count: int,
    table_asset_count: int,
) -> dict[str, Any]:
    tokenizer = tokenizer_config or LexicalTokenizerConfig()
    return {
        "profile_summary": summarize_profiles(profiles),
        "source_file": source_file_metadata(pdf_path),
        "package_config": {
            "base_chunking_strategy": "page",
            "render_zoom": render_zoom,
            "dry_run_embeddings": dry_run_embeddings,
            "section_map_count": len(section_ranges or []),
            "extract_tables": extract_tables,
            "lexical_tokenizer": tokenizer.model_dump(),
        },
        "embedding_mode": "hashing_dry_run" if dry_run_embeddings else "external",
        "section_map_count": len(section_ranges or []),
        "table_count": table_count,
        "table_asset_count": table_asset_count,
    }


def refresh_package_metadata(
    output_dir: Path,
    manifest: ProcessingManifest | None = None,
    pdf_path: Path | None = None,
    render_zoom: float | None = None,
    dry_run_embeddings: bool | None = None,
    section_map_count: int | None = None,
    extract_tables: bool | None = None,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    base_chunking_strategy: str | None = None,
) -> ProcessingManifest:
    """Refresh reproducibility metadata for an existing package."""
    manifest = manifest or load_processing_package(output_dir)
    existing_metadata = dict(manifest.metadata)
    existing_config = existing_metadata.get("package_config")
    existing_config = existing_config if isinstance(existing_config, dict) else {}
    source_path = pdf_path or manifest.doc.local_path
    table_count = current_table_count(manifest)
    asset_table_count = table_asset_count(manifest.assets)
    refreshed_metadata = {
        **existing_metadata,
        "profile_summary": summarize_profiles(manifest.profiles),
        "source_file": source_file_metadata(source_path),
        "package_config": {
            "base_chunking_strategy": (
                base_chunking_strategy
                or existing_config.get("base_chunking_strategy")
                or "page"
            ),
            "render_zoom": render_zoom
            if render_zoom is not None
            else float(existing_config.get("render_zoom") or 1.5),
            "dry_run_embeddings": infer_dry_run_embeddings(
                output_dir,
                existing_config=existing_config,
                override=dry_run_embeddings,
            ),
            "section_map_count": section_map_count
            if section_map_count is not None
            else int(existing_config.get("section_map_count") or 0),
            "extract_tables": extract_tables
            if extract_tables is not None
            else bool(existing_config.get("extract_tables", table_count > 0)),
            "lexical_tokenizer": (
                tokenizer_config
                or package_tokenizer_config(output_dir)
                or existing_tokenizer_config(existing_config)
                or LexicalTokenizerConfig()
            ).model_dump(),
        },
        "embedding_mode": (
            "hashing_dry_run"
            if infer_dry_run_embeddings(
                output_dir,
                existing_config=existing_config,
                override=dry_run_embeddings,
            )
            else "external"
        ),
        "section_map_count": section_map_count
        if section_map_count is not None
        else int(existing_config.get("section_map_count") or 0),
        "table_count": table_count,
        "table_asset_count": asset_table_count,
    }
    updated = manifest.model_copy(update={"metadata": refreshed_metadata})
    (output_dir / "manifest.json").write_text(
        updated.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return updated


def source_file_metadata(path: Path) -> dict[str, Any]:
    summary = file_summary(path)
    return {
        "name": path.name,
        "bytes": summary["bytes"],
        "sha256": summary["sha256"],
    }


def current_table_count(manifest: ProcessingManifest) -> int:
    return sum(1 for chunk in manifest.chunks if str(chunk.kind) == "table")


def table_asset_count(assets: list[VisualAsset]) -> int:
    return sum(1 for asset in assets if str(asset.kind) == "table")


def package_tokenizer_config(output_dir: Path) -> LexicalTokenizerConfig | None:
    path = output_dir / "bm25_tokens.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokenizer_payload = payload.get("tokenizer")
        if isinstance(tokenizer_payload, dict):
            return LexicalTokenizerConfig.model_validate(tokenizer_payload)
    except (OSError, ValueError):
        return None
    return None


def existing_tokenizer_config(existing_config: dict[str, Any]) -> LexicalTokenizerConfig | None:
    tokenizer_payload = existing_config.get("lexical_tokenizer")
    if not isinstance(tokenizer_payload, dict):
        return None
    try:
        return LexicalTokenizerConfig.model_validate(tokenizer_payload)
    except ValueError:
        return None


def infer_dry_run_embeddings(
    output_dir: Path,
    existing_config: dict[str, Any],
    override: bool | None = None,
) -> bool:
    if override is not None:
        return override
    existing_value = existing_config.get("dry_run_embeddings")
    if isinstance(existing_value, bool):
        return existing_value
    manifest_path = output_dir / "embedding_manifest.json"
    if not manifest_path.exists():
        return True
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    manifest_text = json.dumps(payload, ensure_ascii=False).lower()
    if "hashingtextembedder" in manifest_text or '"backend": "hashing"' in manifest_text:
        return True
    return False


def rebuild_search_artifacts(
    output_dir: Path,
    chunks,
    assets=None,
    triples=None,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    rebuild_embeddings: bool = False,
) -> None:
    bm25 = BM25LexicalIndex(
        chunks,
        tokenizer_config=tokenizer_config,
        texts=chunk_lexical_texts(chunks, assets),
    )
    bm25.dump_manifest(output_dir / "bm25_tokens.json")

    if not rebuild_embeddings:
        return

    embedder = HashingTextEmbedder()
    write_embedding_artifacts(
        output_dir=output_dir,
        chunks=chunks,
        assets=assets or [],
        triples=triples or [],
        text_embedder=embedder,
        image_embedder=HashingImageEmbedder(embedding_dim=embedder.embedding_dim)
        if assets is not None
        else None,
        caption_embedder=embedder if assets is not None else None,
        object_embedder=embedder if assets is not None else None,
        triple_embedder=embedder if triples is not None else None,
        collection=existing_collection_name(output_dir),
        vector_notes=hashing_vector_notes(
            include_visual=assets is not None,
            include_triples=triples is not None,
        ),
        vector_metadata=hashing_vector_metadata(
            include_visual=assets is not None,
            include_triples=triples is not None,
            embedding_dim=embedder.embedding_dim,
        ),
    )


def refresh_package_indexes(
    output_dir: Path,
    manifest: ProcessingManifest | None = None,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    rebuild_dry_run_embeddings: bool = False,
    clear_stale_embeddings: bool = True,
) -> dict[str, Any]:
    """Refresh lexical artifacts and invalidate vector artifacts that may be stale."""
    manifest = manifest or load_processing_package(output_dir)
    tokenizer = (
        tokenizer_config
        or package_tokenizer_config(output_dir)
        or existing_tokenizer_config(
            manifest.metadata.get("package_config")
            if isinstance(manifest.metadata.get("package_config"), dict)
            else {}
        )
        or LexicalTokenizerConfig()
    )
    rebuild_search_artifacts(
        output_dir,
        manifest.chunks,
        assets=manifest.assets,
        triples=manifest.triples,
        tokenizer_config=tokenizer,
        rebuild_embeddings=rebuild_dry_run_embeddings,
    )
    cleared_embedding_artifacts = []
    if clear_stale_embeddings and not rebuild_dry_run_embeddings:
        cleared_embedding_artifacts = clear_embedding_artifacts(output_dir)
    return {
        "bm25_tokens": str(output_dir / "bm25_tokens.json"),
        "chunk_count": len(manifest.chunks),
        "asset_count": len(manifest.assets),
        "triple_count": len(manifest.triples),
        "tokenizer": tokenizer.model_dump(),
        "rebuilt_dry_run_embeddings": rebuild_dry_run_embeddings,
        "cleared_embedding_artifacts": cleared_embedding_artifacts,
    }


def clear_embedding_artifacts(output_dir: Path) -> list[str]:
    """Remove vector artifacts that become stale when chunk IDs or text change."""
    removed = []
    for path in embedding_artifact_paths(output_dir):
        if not path.exists():
            continue
        path.unlink()
        removed.append(str(path))
    return removed


def embedding_artifact_paths(output_dir: Path) -> list[Path]:
    return [
        *(output_dir / filename for filename in QDRANT_RECORD_FILES.values()),
        output_dir / "qdrant_collection.json",
        output_dir / "embedding_manifest.json",
    ]


def write_split_chunks(output_dir: Path, chunks, max_chars: int = 1600, overlap_chars: int = 180):
    split_chunks = semantic_subchunks(chunks, max_chars=max_chars, overlap_chars=overlap_chars)
    write_jsonl(output_dir / "chunks.split.jsonl", split_chunks)
    return split_chunks


def write_embedding_artifacts(
    output_dir: Path,
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple] | None = None,
    text_embedder: DenseTextEmbedder | None = None,
    image_embedder: DenseImageEmbedder | None = None,
    caption_embedder: DenseTextEmbedder | None = None,
    object_embedder: DenseTextEmbedder | None = None,
    triple_embedder: DenseTextEmbedder | None = None,
    collection: str = "document_chunks",
    text_batch_size: int = 32,
    image_batch_size: int = 16,
    caption_batch_size: int = 32,
    object_batch_size: int = 32,
    triple_batch_size: int = 32,
    vector_notes: dict[str, str] | None = None,
    vector_metadata: dict[str, dict[str, Any]] | None = None,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """Write Qdrant record files from concrete embedders.

    This is used both for deterministic dry-runs and for local GPU-backed model
    experiments. Clearing known record files prevents stale vector files from
    being upserted after a vector family is intentionally disabled.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        clear_embedding_artifacts(output_dir)

    notes = vector_notes or {}
    metadata = {
        name: dict(value)
        for name, value in (vector_metadata or {}).items()
    }
    named_vectors: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}

    if text_embedder is not None:
        text_records = make_text_embedding_records(
            chunks,
            text_embedder,
            batch_size=text_batch_size,
        )
        write_jsonl(output_dir / QDRANT_RECORD_FILES["text_dense"], text_records)
        named_vectors["text_dense"] = vector_config(text_embedder.embedding_dim, notes.get("text_dense"))
        counts["text_dense"] = len(text_records)
        metadata.setdefault("text_dense", {}).setdefault("batch_size", text_batch_size)

    if image_embedder is not None:
        image_records = make_image_embedding_records(
            assets,
            image_embedder,
            batch_size=image_batch_size,
        )
        write_jsonl(output_dir / QDRANT_RECORD_FILES["image_dense"], image_records)
        named_vectors["image_dense"] = vector_config(
            image_embedder.embedding_dim,
            notes.get("image_dense"),
        )
        counts["image_dense"] = len(image_records)
        metadata.setdefault("image_dense", {}).setdefault("batch_size", image_batch_size)

    if caption_embedder is not None:
        caption_records = make_caption_embedding_records(
            assets,
            caption_embedder,
            batch_size=caption_batch_size,
        )
        write_jsonl(output_dir / QDRANT_RECORD_FILES["caption_dense"], caption_records)
        named_vectors["caption_dense"] = vector_config(
            caption_embedder.embedding_dim,
            notes.get("caption_dense"),
        )
        counts["caption_dense"] = len(caption_records)
        metadata.setdefault("caption_dense", {}).setdefault("batch_size", caption_batch_size)

    if object_embedder is not None:
        object_records = make_object_embedding_records(
            assets,
            object_embedder,
            batch_size=object_batch_size,
        )
        write_jsonl(output_dir / QDRANT_RECORD_FILES["object_dense"], object_records)
        named_vectors["object_dense"] = vector_config(
            object_embedder.embedding_dim,
            notes.get("object_dense"),
        )
        counts["object_dense"] = len(object_records)
        metadata.setdefault("object_dense", {}).setdefault("batch_size", object_batch_size)

    if triple_embedder is not None:
        triple_records = make_triple_embedding_records(
            triples or [],
            triple_embedder,
            batch_size=triple_batch_size,
            chunks=chunks,
        )
        write_jsonl(output_dir / QDRANT_RECORD_FILES["triple_dense"], triple_records)
        named_vectors["triple_dense"] = vector_config(
            triple_embedder.embedding_dim,
            notes.get("triple_dense"),
        )
        counts["triple_dense"] = len(triple_records)
        metadata.setdefault("triple_dense", {}).setdefault("batch_size", triple_batch_size)

    write_qdrant_collection_config(output_dir, collection, named_vectors)
    write_embedding_manifest(
        output_dir=output_dir,
        collection=collection,
        named_vectors=named_vectors,
        record_counts=counts,
        vector_notes=notes,
        vector_metadata=metadata,
    )
    return {
        "collection": collection,
        "records": counts,
        "named_vectors": {name: config["size"] for name, config in named_vectors.items()},
        "embedding_manifest": str(output_dir / "embedding_manifest.json"),
    }


def hashing_vector_notes(include_visual: bool = True, include_triples: bool = False) -> dict[str, str]:
    notes = {
        "text_dense": "HashingTextEmbedder dry-run dimension. Replace with real dense model dimension.",
    }
    if include_visual:
        notes.update(
            {
                "image_dense": "HashingImageEmbedder dry-run dimension. Replace with real image model dimension.",
                "caption_dense": "Caption text dry-run dimension. Replace with real dense model dimension.",
                "object_dense": "VLM object text dry-run dimension. Replace with real dense model dimension.",
            }
        )
    if include_triples:
        notes["triple_dense"] = "Graph triple text dry-run dimension. Replace with real dense model dimension."
    return notes


def hashing_vector_metadata(
    include_visual: bool = True,
    include_triples: bool = False,
    embedding_dim: int = 384,
) -> dict[str, dict[str, Any]]:
    metadata = {
        "text_dense": {
            "backend": "hashing",
            "model": "HashingTextEmbedder",
            "dimension": embedding_dim,
            "deterministic": True,
        }
    }
    if include_visual:
        metadata.update(
            {
                "image_dense": {
                    "backend": "hashing",
                    "model": "HashingImageEmbedder",
                    "dimension": embedding_dim,
                    "deterministic": True,
                },
                "caption_dense": {
                    "backend": "hashing",
                    "model": "HashingTextEmbedder",
                    "dimension": embedding_dim,
                    "deterministic": True,
                    "same_as": "text_dense",
                },
                "object_dense": {
                    "backend": "hashing",
                    "model": "HashingTextEmbedder",
                    "dimension": embedding_dim,
                    "deterministic": True,
                    "same_as": "text_dense",
                },
            }
        )
    if include_triples:
        metadata["triple_dense"] = {
            "backend": "hashing",
            "model": "HashingTextEmbedder",
            "dimension": embedding_dim,
            "deterministic": True,
            "same_as": "text_dense",
        }
    return metadata


def existing_collection_name(output_dir: Path, default: str = "document_chunks") -> str:
    path = output_dir / "qdrant_collection.json"
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    collection = payload.get("collection")
    return str(collection) if collection else default


def vector_config(size: int, note: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {"size": int(size), "distance": "Cosine"}
    if note:
        config["note"] = note
    return config


def write_qdrant_collection_config(
    output_dir: Path,
    collection: str,
    named_vectors: dict[str, dict[str, Any]],
) -> None:
    (output_dir / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": collection,
                "named_vectors": named_vectors,
                "payload_indexes": QDRANT_PAYLOAD_INDEXES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_embedding_manifest(
    output_dir: Path,
    collection: str,
    named_vectors: dict[str, dict[str, Any]],
    record_counts: dict[str, int],
    vector_notes: dict[str, str],
    vector_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    metadata = {
        name: dict(value)
        for name, value in (vector_metadata or {}).items()
    }
    vectors = {}
    for vector_name, config in sorted(named_vectors.items()):
        filename = QDRANT_RECORD_FILES[vector_name]
        path = output_dir / filename
        vector_payload: dict[str, Any] = {
            "file": filename,
            "record_count": record_counts.get(vector_name, 0),
            "dimension": int(config["size"]),
            "distance": config.get("distance", "Cosine"),
            "note": vector_notes.get(vector_name) or config.get("note"),
            **file_summary(path),
        }
        if metadata.get(vector_name):
            vector_payload["embedding"] = metadata[vector_name]
        vectors[vector_name] = {
            **vector_payload,
        }
    (output_dir / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": collection,
                "vectors": vectors,
                "payload_indexes": QDRANT_PAYLOAD_INDEXES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "sha256": None}
    content = path.read_bytes()
    return {
        "exists": True,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def load_processing_package(package_dir: Path) -> ProcessingManifest:
    manifest_payload = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    return ProcessingManifest(
        doc=SourceDocument.model_validate(manifest_payload["doc"]),
        profiles=read_jsonl(package_dir / "pages.jsonl", PageProfile),
        chunks=read_jsonl(package_dir / "chunks.jsonl", DocumentChunk),
        assets=read_jsonl(package_dir / "assets.jsonl", VisualAsset),
        triples=read_jsonl(package_dir / "triples.jsonl", GraphTriple),
        metadata=manifest_payload.get("metadata", {}),
    )

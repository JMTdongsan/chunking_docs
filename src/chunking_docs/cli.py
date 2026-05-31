from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer
from rich import print

from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles, write_profile_outputs
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.ingest.pdf_loader import load_source_document, render_pages
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.pipeline import build_processing_package, rebuild_search_artifacts, write_split_chunks
from chunking_docs.retrieval.local_hybrid import LocalHybridSearcher
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.annotate import annotate_assets, merge_asset_annotations_into_chunks
from chunking_docs.vision.manual_annotations import AssetAnnotation, apply_asset_annotations

app = typer.Typer(help="Document chunking utilities.")


@app.command()
def download(url: str, output: Path):
    """Download a source document."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with output.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    print(f"Downloaded {output}")


@app.command()
def profile(pdf: Path, output_dir: Path = Path("outputs/profile")):
    """Profile PDF text and visual density."""
    document = load_source_document(pdf)
    profiles = profile_pdf(pdf, document.doc_id)
    write_profile_outputs(profiles, output_dir)
    print_json(summarize_profiles(profiles))


@app.command()
def render(pdf: Path, output_dir: Path = Path("outputs/renders"), pages: str = ""):
    """Render selected PDF pages to PNG."""
    page_numbers = [int(item) for item in pages.split(",") if item.strip()] or None
    rendered = render_pages(pdf, output_dir, pages=page_numbers)
    print(f"Rendered {len(rendered)} pages into {output_dir}")


@app.command()
def chunk(pdf: Path, output: Path = Path("outputs/chunks.jsonl")):
    """Create page-level starter chunks."""
    source = load_source_document(pdf)
    profiles = profile_pdf(pdf, source.doc_id)
    chunks = page_level_chunks(pdf, source.doc_id, profiles)
    write_jsonl(output, chunks)
    print(f"Wrote {len(chunks)} chunks to {output}")


@app.command(name="package")
def package_pdf(
    pdf: Path,
    output_dir: Path = Path("outputs/package"),
    source_url: str = "",
    title: str = "",
    render_zoom: float = 1.5,
):
    """Build the full local processing package for DB ingestion."""
    manifest = build_processing_package(
        pdf_path=pdf,
        output_dir=output_dir,
        source_url=source_url or None,
        title=title or None,
        render_zoom=render_zoom,
    )
    print(
        {
            "doc_id": manifest.doc.doc_id,
            "pages": len(manifest.profiles),
            "chunks": len(manifest.chunks),
            "assets": len(manifest.assets),
            "triples": len(manifest.triples),
            "output_dir": str(output_dir),
        }
    )


@app.command(name="qdrant-upsert")
def qdrant_upsert(
    records: Path = Path("outputs/package/qdrant_text_records.jsonl"),
    url: str = "http://localhost:6333",
    collection: str = "planning_chunks",
    vector_name: str = "text_dense",
    vector_size: int = 384,
    location: str = "",
    path: str = "",
):
    """Create a Qdrant collection if needed and upsert embedding records."""
    from chunking_docs.storage.qdrant_store import QdrantChunkStore

    parsed = [
        EmbeddingRecord.model_validate_json(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    store = QdrantChunkStore(
        url=url,
        collection_name=collection,
        location=location or None,
        path=path or None,
    )
    store.ensure_collection({vector_name: vector_size})
    result = store.upsert(parsed)
    print({**result.model_dump(), "stored_count": store.count()})


@app.command(name="qdrant-upsert-package")
def qdrant_upsert_package(
    package_dir: Path = Path("outputs/package"),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
):
    """Upsert all qdrant_*_records.jsonl files from a processing package."""
    from chunking_docs.storage.qdrant_store import QdrantChunkStore

    collection_config = json.loads((package_dir / "qdrant_collection.json").read_text(encoding="utf-8"))
    collection_name = collection or collection_config["collection"]
    named_vectors = {
        name: int(config["size"])
        for name, config in collection_config.get("named_vectors", {}).items()
    }
    store = QdrantChunkStore(
        url=url,
        collection_name=collection_name,
        location=location or None,
        path=path or None,
    )
    store.ensure_collection(named_vectors)

    total = 0
    files = sorted(package_dir.glob("qdrant_*_records.jsonl"))
    for record_file in files:
        records = read_jsonl(record_file, EmbeddingRecord)
        result = store.upsert(records)
        total += result.count

    print(
        {
            "collection": collection_name,
            "files": [str(file.name) for file in files],
            "upserted": total,
            "stored_count": store.count(),
            "named_vectors": sorted(named_vectors),
        }
    )


@app.command(name="annotate-assets")
def annotate_assets_command(
    package_dir: Path = Path("outputs/package"),
    pages: str = "",
    limit: int | None = None,
    ocr: str = "none",
    vlm: str = "none",
    vlm_model: str = "",
    in_place: bool = False,
    rebuild_search: bool = True,
):
    """Annotate rendered assets with OCR/VLM output and merge it into chunks."""
    selected_pages = {int(item) for item in pages.split(",") if item.strip()} or None
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)

    ocr_backend = None
    if ocr == "tesseract":
        from chunking_docs.vision.tesseract_ocr import TesseractOCRBackend

        ocr_backend = TesseractOCRBackend()
    elif ocr != "none":
        raise typer.BadParameter("ocr must be one of: none, tesseract")

    vlm_backend = None
    if vlm == "hf":
        if not vlm_model:
            raise typer.BadParameter("--vlm-model is required when --vlm hf")
        from chunking_docs.vision.hf_vlm import HuggingFaceVLMBackend

        vlm_backend = HuggingFaceVLMBackend(model_name=vlm_model)
    elif vlm != "none":
        raise typer.BadParameter("vlm must be one of: none, hf")

    annotated_assets = annotate_assets(
        assets,
        ocr_backend=ocr_backend,
        vlm_backend=vlm_backend,
        pages=selected_pages,
        limit=limit,
    )
    annotated_chunks = merge_asset_annotations_into_chunks(chunks, annotated_assets)

    asset_output = package_dir / ("assets.jsonl" if in_place else "assets.annotated.jsonl")
    chunk_output = package_dir / ("chunks.jsonl" if in_place else "chunks.annotated.jsonl")
    write_jsonl(asset_output, annotated_assets)
    write_jsonl(chunk_output, annotated_chunks)
    if rebuild_search and in_place:
        rebuild_search_artifacts(package_dir, annotated_chunks)

    annotated_count = sum(
        1 for asset in annotated_assets if asset.ocr_text is not None or asset.vlm_summary is not None
    )
    print(
        {
            "assets": len(annotated_assets),
            "chunks": len(annotated_chunks),
            "annotated_assets": annotated_count,
            "asset_output": str(asset_output),
            "chunk_output": str(chunk_output),
            "rebuilt_search": bool(rebuild_search and in_place),
        }
    )


@app.command(name="apply-annotations")
def apply_annotations_command(
    annotations: Path,
    package_dir: Path = Path("outputs/package"),
    in_place: bool = True,
    rebuild_search: bool = True,
):
    """Apply manual or external VLM annotations from JSONL to package assets/chunks/triples."""
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    parsed_annotations = read_jsonl(annotations, AssetAnnotation)

    updated_assets, updated_chunks, updated_triples = apply_asset_annotations(
        assets,
        chunks,
        parsed_annotations,
        existing_triples=triples,
    )

    suffix = "" if in_place else ".annotated"
    write_jsonl(package_dir / f"assets{suffix}.jsonl", updated_assets)
    write_jsonl(package_dir / f"chunks{suffix}.jsonl", updated_chunks)
    write_jsonl(package_dir / f"triples{suffix}.jsonl", updated_triples)

    if in_place and rebuild_search:
        rebuild_search_artifacts(package_dir, updated_chunks, assets=updated_assets)

    print(
        {
            "annotations": len(parsed_annotations),
            "assets": len(updated_assets),
            "chunks": len(updated_chunks),
            "triples": len(updated_triples),
            "in_place": in_place,
            "rebuilt_search": bool(in_place and rebuild_search),
        }
    )


@app.command(name="split-chunks")
def split_chunks_command(
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    max_chars: int = 1600,
    overlap_chars: int = 180,
    in_place: bool = False,
    rebuild_search: bool = True,
):
    """Create semantic subchunks from page chunks, usually after OCR/VLM annotation."""
    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    split_chunks = write_split_chunks(
        package_dir,
        chunks,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    output = package_dir / "chunks.split.jsonl"
    if in_place:
        write_jsonl(package_dir / "chunks.jsonl", split_chunks)
        if rebuild_search:
            rebuild_search_artifacts(package_dir, split_chunks)
    print(
        {
            "source": str(package_dir / chunks_file),
            "output": str(output),
            "input_chunks": len(chunks),
            "split_chunks": len(split_chunks),
            "in_place": in_place,
            "rebuilt_search": bool(in_place and rebuild_search),
        }
    )


@app.command(name="search-local")
def search_local(
    query: str,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    top_k: int = 5,
):
    """Run local dry-run hybrid search over package chunks using hashing dense + BM25."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder())
    hits = searcher.search(query, top_k=top_k)
    print(
        [
            {
                "rank": index + 1,
                "chunk_id": hit.chunk.chunk_id,
                "page": [hit.chunk.page_start, hit.chunk.page_end],
                "score": round(hit.score, 6),
                "sources": hit.sources,
                "section": hit.chunk.section.label(),
                "preview": hit.chunk.text[:180],
            }
            for index, hit in enumerate(hits)
        ]
    )


def print_json(payload: dict):
    print(json.dumps(payload, ensure_ascii=False, indent=2))

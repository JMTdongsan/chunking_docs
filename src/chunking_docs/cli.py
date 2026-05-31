from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer
from rich import print

from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles, write_profile_outputs
from chunking_docs.chunking.section_map import load_section_ranges
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.evaluation.audit import audit_package
from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.retrieval import evaluate_retrieval, load_retrieval_cases
from chunking_docs.graph.repair import remap_triples_to_available_chunks
from chunking_docs.ingest.pdf_loader import load_source_document, render_pages
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.pipeline import (
    build_processing_package,
    load_processing_package,
    rebuild_search_artifacts,
    write_embedding_artifacts,
    write_split_chunks,
)
from chunking_docs.retrieval.local_hybrid import LocalHybridSearcher
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.annotate import annotate_assets, merge_asset_annotations_into_chunks
from chunking_docs.vision.interfaces import OCRBackend, VLMBackend
from chunking_docs.vision.jobs import (
    VisualAnnotationJob,
    completed_annotations,
    plan_visual_jobs,
    run_visual_jobs,
)
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
def chunk(
    pdf: Path,
    output: Path = Path("outputs/chunks.jsonl"),
    section_map: Path | None = None,
):
    """Create page-level starter chunks."""
    source = load_source_document(pdf)
    profiles = profile_pdf(pdf, source.doc_id)
    chunks = page_level_chunks(
        pdf,
        source.doc_id,
        profiles,
        section_ranges=load_section_ranges(section_map),
    )
    write_jsonl(output, chunks)
    print(f"Wrote {len(chunks)} chunks to {output}")


@app.command(name="package")
def package_pdf(
    pdf: Path,
    output_dir: Path = Path("outputs/package"),
    source_url: str = "",
    title: str = "",
    render_zoom: float = 1.5,
    section_map: Path | None = None,
):
    """Build the full local processing package for DB ingestion."""
    manifest = build_processing_package(
        pdf_path=pdf,
        output_dir=output_dir,
        source_url=source_url or None,
        title=title or None,
        render_zoom=render_zoom,
        section_ranges=load_section_ranges(section_map),
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


@app.command(name="embed-package")
def embed_package_command(
    package_dir: Path = Path("outputs/package"),
    collection: str = "planning_chunks",
    text_backend: str = "sentence-transformers",
    caption_backend: str = "same-as-text",
    image_backend: str = "clip",
    text_model: str = "BAAI/bge-m3",
    caption_model: str = "",
    image_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    text_device: str = "",
    image_device: str = "",
    hashing_dim: int = 384,
    text_batch_size: int = 16,
    caption_batch_size: int = 16,
    image_batch_size: int = 8,
):
    """Rebuild Qdrant records with concrete dense/image embedding models."""
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)

    text_embedder, text_note = build_text_embedder(
        backend=text_backend,
        model_name=text_model,
        device=text_device or device,
        hashing_dim=hashing_dim,
        vector_name="text_dense",
    )
    caption_embedder, caption_note = build_caption_embedder(
        backend=caption_backend,
        model_name=caption_model or text_model,
        device=text_device or device,
        hashing_dim=hashing_dim,
        text_embedder=text_embedder,
        text_note=text_note,
    )
    image_embedder, image_note = build_image_embedder(
        backend=image_backend,
        model_name=image_model,
        device=image_device or device,
        hashing_dim=hashing_dim,
    )

    if text_embedder is None and caption_embedder is None and image_embedder is None:
        raise typer.BadParameter("At least one backend must be enabled")

    notes = {}
    if text_note:
        notes["text_dense"] = text_note
    if caption_note:
        notes["caption_dense"] = caption_note
    if image_note:
        notes["image_dense"] = image_note

    result = write_embedding_artifacts(
        output_dir=package_dir,
        chunks=chunks,
        assets=assets,
        text_embedder=text_embedder,
        caption_embedder=caption_embedder,
        image_embedder=image_embedder,
        collection=collection,
        text_batch_size=text_batch_size,
        caption_batch_size=caption_batch_size,
        image_batch_size=image_batch_size,
        vector_notes=notes,
    )
    print(
        {
            **result,
            "package_dir": str(package_dir),
            "text_backend": text_backend,
            "caption_backend": caption_backend,
            "image_backend": image_backend,
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

    ocr_backend, _ = build_ocr_backend(ocr)
    vlm_backend, _ = build_vlm_backend(vlm, vlm_model)

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
        rebuild_search_artifacts(package_dir, annotated_chunks, assets=annotated_assets)

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


@app.command(name="plan-visual-jobs")
def plan_visual_jobs_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    pages: str = "",
    limit: int | None = None,
    include_ocr: bool = True,
    include_vlm: bool = True,
):
    """Plan prioritized OCR/VLM jobs for rendered visual assets."""
    selected_pages = {int(item) for item in pages.split(",") if item.strip()} or None
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    jobs = plan_visual_jobs(
        assets,
        pages=selected_pages,
        include_ocr=include_ocr,
        include_vlm=include_vlm,
        limit=limit,
    )
    output_path = output or package_dir / "visual_jobs.jsonl"
    write_jsonl(output_path, jobs)
    print(
        {
            "jobs": len(jobs),
            "output": str(output_path),
            "top_pages": [job.page_no for job in jobs[:10]],
            "operation_counts": operation_counts(jobs),
            "kind_counts": kind_counts(jobs),
        }
    )


@app.command(name="run-visual-jobs")
def run_visual_jobs_command(
    package_dir: Path = Path("outputs/package"),
    jobs: Path | None = None,
    annotations_output: Path | None = None,
    results_output: Path | None = None,
    limit: int | None = None,
    ocr: str = "none",
    vlm: str = "none",
    vlm_model: str = "",
    ocr_language: str = "kor+eng",
    apply: bool = False,
    rebuild_search: bool = True,
):
    """Run planned OCR/VLM jobs and optionally apply the resulting annotations."""
    job_path = jobs or package_dir / "visual_jobs.jsonl"
    annotation_path = annotations_output or package_dir / "visual_annotations.jsonl"
    result_path = results_output or package_dir / "visual_job_results.jsonl"
    planned_jobs = read_jsonl(job_path, VisualAnnotationJob)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)

    ocr_backend, ocr_name = build_ocr_backend(ocr)
    vlm_backend, vlm_name = build_vlm_backend(vlm, vlm_model)
    results = run_visual_jobs(
        planned_jobs,
        assets,
        ocr_backend=ocr_backend,
        vlm_backend=vlm_backend,
        limit=limit,
        ocr_language=ocr_language,
        ocr_backend_name=ocr_name,
        vlm_backend_name=vlm_name,
    )
    annotations = completed_annotations(results)
    write_jsonl(result_path, results)
    write_jsonl(annotation_path, annotations)

    applied = False
    if apply and annotations:
        chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
        triples_path = package_dir / "triples.jsonl"
        triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
        updated_assets, updated_chunks, updated_triples = apply_asset_annotations(
            assets,
            chunks,
            annotations,
            existing_triples=triples,
        )
        write_jsonl(package_dir / "assets.jsonl", updated_assets)
        write_jsonl(package_dir / "chunks.jsonl", updated_chunks)
        write_jsonl(package_dir / "triples.jsonl", updated_triples)
        if rebuild_search:
            rebuild_search_artifacts(package_dir, updated_chunks, assets=updated_assets)
        applied = True

    print(
        {
            "jobs": len(planned_jobs),
            "completed": sum(1 for result in results if result.status == "completed"),
            "failed": sum(1 for result in results if result.status == "failed"),
            "skipped": sum(1 for result in results if result.status == "skipped"),
            "annotations": len(annotations),
            "annotations_output": str(annotation_path),
            "results_output": str(result_path),
            "applied": applied,
            "rebuilt_search": bool(applied and rebuild_search),
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
        triples_path = package_dir / "triples.jsonl"
        remapped_triples = []
        if triples_path.exists():
            triples = read_jsonl(triples_path, GraphTriple)
            remapped_triples = remap_triples_to_available_chunks(triples, split_chunks)
            write_jsonl(triples_path, remapped_triples)
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
            "remapped_triples": len(remapped_triples) if in_place else 0,
        }
    )


@app.command(name="search-local")
def search_local(
    query: str,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    top_k: int = 5,
    graph_expand: bool = False,
):
    """Run local dry-run hybrid search over package chunks using hashing dense + BM25."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    searcher = LocalHybridSearcher(chunks, HashingTextEmbedder(), triples=triples)
    hits = searcher.search(query, top_k=top_k, graph_expand=graph_expand)
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


@app.command(name="export-graph")
def export_graph_command(package_dir: Path = Path("outputs/package")):
    """Export triples as graph nodes and edges JSONL."""
    from chunking_docs.graph.export import export_graph

    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    triples = read_jsonl(package_dir / "triples.jsonl", GraphTriple)
    nodes, edges = export_graph(triples, chunks=chunks)
    write_jsonl(package_dir / "graph_nodes.jsonl", nodes)
    write_jsonl(package_dir / "graph_edges.jsonl", edges)
    print(
        {
            "nodes": len(nodes),
            "edges": len(edges),
            "nodes_output": str(package_dir / "graph_nodes.jsonl"),
            "edges_output": str(package_dir / "graph_edges.jsonl"),
        }
    )


@app.command(name="postgres-upsert")
def postgres_upsert(
    dsn: str,
    package_dir: Path = Path("outputs/package"),
    apply_schema: bool = True,
):
    """Apply schema and upsert package metadata into PostgreSQL."""
    from chunking_docs.storage.postgres_store import PostgresDocumentStore

    manifest = load_processing_package(package_dir)
    store = PostgresDocumentStore(dsn)
    if apply_schema:
        store.apply_schema()
    result = store.upsert_manifest(manifest, base_dir=package_dir)
    print(result)


@app.command(name="postgres-rows")
def postgres_rows(package_dir: Path = Path("outputs/package")):
    """Validate and summarize rows that would be sent to PostgreSQL."""
    from chunking_docs.storage.postgres_store import manifest_rows

    manifest = load_processing_package(package_dir)
    rows = manifest_rows(manifest, base_dir=package_dir)
    print(
        {
            "documents": 1,
            "pages": len(rows["pages"]),
            "chunks": len(rows["chunks"]),
            "assets": len(rows["assets"]),
            "triples": len(rows["triples"]),
            "first_document": rows["document"],
        }
    )


@app.command(name="audit-package")
def audit_package_command(
    package_dir: Path = Path("outputs/package"),
    require_annotations_for_visual_pages: bool = False,
):
    """Audit package completeness and remaining OCR/VLM work."""
    manifest = load_processing_package(package_dir)
    audit = audit_package(
        manifest.profiles,
        manifest.chunks,
        manifest.assets,
        manifest.triples,
        require_annotations_for_visual_pages=require_annotations_for_visual_pages,
    )
    print(audit.model_dump())


@app.command(name="eval-retrieval")
def eval_retrieval_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    top_k: int = 5,
):
    """Evaluate local hybrid retrieval against JSONL seed cases."""
    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    evaluation = evaluate_retrieval(
        chunks=chunks,
        triples=triples,
        cases=load_retrieval_cases(cases),
        top_k=top_k,
    )
    print(evaluation.model_dump())


@app.command(name="eval-chunking")
def eval_chunking_command(
    package_dir: Path = Path("outputs/package"),
    cases: Path | None = None,
    top_k: int = 5,
    min_chars: int = 120,
    max_chars: int = 1800,
):
    """Evaluate chunking quality and optional retrieval performance."""
    manifest = load_processing_package(package_dir)
    retrieval_cases = load_retrieval_cases(cases) if cases is not None else None
    report = evaluate_chunking_quality(
        chunks=manifest.chunks,
        profiles=manifest.profiles,
        assets=manifest.assets,
        triples=manifest.triples,
        retrieval_cases=retrieval_cases,
        top_k=top_k,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    print(report.model_dump())


def print_json(payload: dict):
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_text_embedder(
    backend: str,
    model_name: str,
    device: str,
    hashing_dim: int,
    vector_name: str,
):
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None, ""
    if normalized == "hashing":
        from chunking_docs.embeddings.interfaces import HashingTextEmbedder

        return (
            HashingTextEmbedder(embedding_dim=hashing_dim),
            f"HashingTextEmbedder deterministic fallback for {vector_name}.",
        )
    if normalized in {"sentence-transformers", "sentence_transformers", "st"}:
        from chunking_docs.embeddings.sentence_transformers import SentenceTransformerTextEmbedder

        return (
            SentenceTransformerTextEmbedder(model_name=model_name, device=device or None),
            f"SentenceTransformerTextEmbedder model={model_name} device={device or 'auto'}.",
        )
    raise typer.BadParameter(
        "text backend must be one of: none, hashing, sentence-transformers"
    )


def build_caption_embedder(
    backend: str,
    model_name: str,
    device: str,
    hashing_dim: int,
    text_embedder,
    text_note: str,
):
    normalized = normalize_backend(backend)
    if normalized in {"same-as-text", "same_as_text", "same"}:
        if text_embedder is None:
            raise typer.BadParameter("--caption-backend same-as-text requires text backend")
        return text_embedder, f"Same model as text_dense. {text_note}"
    return build_text_embedder(
        backend=backend,
        model_name=model_name,
        device=device,
        hashing_dim=hashing_dim,
        vector_name="caption_dense",
    )


def build_image_embedder(
    backend: str,
    model_name: str,
    device: str,
    hashing_dim: int,
):
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None, ""
    if normalized == "hashing":
        from chunking_docs.embeddings.interfaces import HashingImageEmbedder

        return (
            HashingImageEmbedder(embedding_dim=hashing_dim),
            "HashingImageEmbedder deterministic fallback for image_dense.",
        )
    if normalized in {"clip", "transformers"}:
        from chunking_docs.embeddings.clip import TransformersImageEmbedder

        return (
            TransformersImageEmbedder(model_name=model_name, device=device),
            f"TransformersImageEmbedder model={model_name} device={device}.",
        )
    raise typer.BadParameter("image backend must be one of: none, hashing, clip")


def normalize_backend(value: str) -> str:
    return value.strip().lower()


def build_ocr_backend(backend: str) -> tuple[OCRBackend | None, str]:
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None, ""
    if normalized == "tesseract":
        from chunking_docs.vision.tesseract_ocr import TesseractOCRBackend

        return TesseractOCRBackend(), "tesseract"
    raise typer.BadParameter("ocr must be one of: none, tesseract")


def build_vlm_backend(backend: str, model_name: str) -> tuple[VLMBackend | None, str]:
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None, ""
    if normalized == "hf":
        if not model_name:
            raise typer.BadParameter("--vlm-model is required when --vlm hf")
        from chunking_docs.vision.hf_vlm import HuggingFaceVLMBackend

        return HuggingFaceVLMBackend(model_name=model_name), f"hf:{model_name}"
    raise typer.BadParameter("vlm must be one of: none, hf")


def operation_counts(jobs: list[VisualAnnotationJob]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        for operation in job.operations:
            counts[operation] = counts.get(operation, 0) + 1
    return counts


def kind_counts(jobs: list[VisualAnnotationJob]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        kind = str(job.kind)
        counts[kind] = counts.get(kind, 0) + 1
    return counts

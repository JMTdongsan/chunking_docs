from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer
from rich import print

from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles, write_profile_outputs
from chunking_docs.chunking.multimodal import ChunkStrategy, build_strategy_chunks
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.chunking.section_map import load_section_ranges
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig, TokenizerStrategy
from chunking_docs.evaluation.audit import audit_package
from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.compare import compare_chunking_reports
from chunking_docs.evaluation.experiment import build_experiment_report
from chunking_docs.evaluation.ablation import evaluate_retrieval_ablation, parse_ablation_modes
from chunking_docs.evaluation.retrieval import evaluate_retrieval, load_retrieval_cases
from chunking_docs.evaluation.sweep import run_chunking_sweep
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
from chunking_docs.retrieval.context import build_context_bundle
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.annotate import annotate_assets, merge_asset_annotations_into_chunks
from chunking_docs.vision.interfaces import OCRBackend, VLMBackend
from chunking_docs.vision.jobs import (
    VisualAnnotationJob,
    VisualJobRunResult,
    completed_annotations,
    plan_visual_jobs,
    run_visual_jobs,
)
from chunking_docs.vision.manual_annotations import AssetAnnotation, apply_asset_annotations
from chunking_docs.vision.report import summarize_visual_results

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
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Build the full local processing package for DB ingestion."""
    manifest = build_processing_package(
        pdf_path=pdf,
        output_dir=output_dir,
        source_url=source_url or None,
        title=title or None,
        render_zoom=render_zoom,
        section_ranges=load_section_ranges(section_map),
        tokenizer_config=build_tokenizer_config(
            lexical_tokenizer,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            ngram_cjk_only=ngram_cjk_only,
        ),
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
    collection: str = "document_chunks",
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


@app.command(name="qdrant-search-package")
def qdrant_search_package(
    query: str,
    package_dir: Path = Path("outputs/package"),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    vector_name: str = "text_dense",
    top_k: int = 5,
    text_backend: str = "hashing",
    text_model: str = "BAAI/bge-m3",
    device: str = "cuda",
    hashing_dim: int = 384,
    doc_id: str = "",
):
    """Upsert package records and run a Qdrant named-vector text query."""
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
    upserted = upsert_package_records(store, package_dir)

    embedder, _ = build_text_embedder(
        backend=text_backend,
        model_name=text_model,
        device=device,
        hashing_dim=hashing_dim,
        vector_name=vector_name,
    )
    if embedder is None:
        raise typer.BadParameter("text backend must not be none for qdrant-search-package")
    vector = embedder.embed_texts([query])[0]
    filters = {"doc_id": doc_id} if doc_id else None
    hits = store.query_vector(
        vector=vector,
        vector_name=vector_name,
        top_k=top_k,
        must_payload=filters,
    )
    print(
        {
            "collection": collection_name,
            "vector_name": vector_name,
            "upserted": upserted,
            "stored_count": store.count(),
            "hits": [
                {
                    "rank": index + 1,
                    "score": hit.score,
                    "point_id": hit.point_id,
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "page": hit.payload.get("page_no")
                    or [hit.payload.get("page_start"), hit.payload.get("page_end")],
                    "kind": hit.payload.get("kind"),
                    "preview": str(hit.payload.get("text") or hit.payload.get("caption") or "")[:180],
                }
                for index, hit in enumerate(hits)
            ],
        }
    )


@app.command(name="qdrant-hybrid-search")
def qdrant_hybrid_search(
    query: str,
    package_dir: Path = Path("outputs/package"),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    vector_names: str = "text_dense,caption_dense",
    top_k: int = 5,
    graph_expand: bool = False,
    collapse_hierarchical: bool = False,
    text_backend: str = "hashing",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "none",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    doc_id: str = "",
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Run Qdrant named-vector + BM25 + optional graph hybrid retrieval."""
    from chunking_docs.retrieval.qdrant_hybrid import QdrantHybridSearcher
    from chunking_docs.storage.qdrant_store import QdrantChunkStore

    collection_config = json.loads((package_dir / "qdrant_collection.json").read_text(encoding="utf-8"))
    collection_name = collection or collection_config["collection"]
    named_vectors = {
        name: int(config["size"])
        for name, config in collection_config.get("named_vectors", {}).items()
    }
    selected_vectors = [item.strip() for item in vector_names.split(",") if item.strip()]
    store = QdrantChunkStore(
        url=url,
        collection_name=collection_name,
        location=location or None,
        path=path or None,
    )
    store.ensure_collection(named_vectors)
    upserted = upsert_package_records(store, package_dir)

    embedder, _ = build_text_embedder(
        backend=text_backend,
        model_name=text_model,
        device=device,
        hashing_dim=hashing_dim,
        vector_name=selected_vectors[0] if selected_vectors else "text_dense",
    )
    if embedder is None:
        raise typer.BadParameter("text backend must not be none for qdrant-hybrid-search")
    vector_embedders = build_qdrant_query_embedders(
        selected_vectors=selected_vectors,
        vector_sizes=named_vectors,
        default_embedder=embedder,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
        device=device,
        hashing_dim=hashing_dim,
    )

    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    searcher = QdrantHybridSearcher(
        store=store,
        chunks=chunks,
        assets=assets,
        embedder=embedder,
        vector_embedders=vector_embedders,
        triples=triples,
        tokenizer_config=build_tokenizer_config(
            lexical_tokenizer,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            ngram_cjk_only=ngram_cjk_only,
        ),
    )
    hits = searcher.search(
        query=query,
        vector_names=selected_vectors,
        top_k=top_k,
        graph_expand=graph_expand,
        doc_id=doc_id or None,
        collapse_hierarchical=collapse_hierarchical,
    )
    print(
        {
            "collection": collection_name,
            "vector_names": selected_vectors,
            "query_encoders": {
                name: (
                    "default_text"
                    if vector_embedders.get(name, embedder) is embedder
                    else image_query_backend
                )
                for name in selected_vectors
            },
            "upserted": upserted,
            "stored_count": store.count(),
            "hits": [
                {
                    "rank": index + 1,
                    "score": hit.score,
                    "item_id": hit.item_id,
                    "sources": hit.sources,
                    "page": [hit.chunk.page_start, hit.chunk.page_end] if hit.chunk else None,
                    "kind": str(hit.chunk.kind) if hit.chunk else None,
                    "preview": hit.chunk.text[:180] if hit.chunk else "",
                    "evidence_chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "page": [chunk.page_start, chunk.page_end],
                            "preview": chunk.text[:120],
                        }
                        for chunk in hit.evidence_chunks[:3]
                    ],
                    "qdrant_payloads": hit.payloads[:2],
                }
                for index, hit in enumerate(hits)
            ],
        }
    )


@app.command(name="embed-package")
def embed_package_command(
    package_dir: Path = Path("outputs/package"),
    collection: str = "document_chunks",
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


@app.command(name="summarize-visual-results")
def summarize_visual_results_command(
    results: Path = Path("outputs/package/visual_job_results.jsonl"),
    output: Path | None = None,
):
    """Summarize OCR/VLM job results by status, backend, latency, and output volume."""
    parsed_results = read_jsonl(results, VisualJobRunResult)
    summary = summarize_visual_results(parsed_results)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        print({"output": str(output), **summary.model_dump()})
        return
    print(summary.model_dump())


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


@app.command(name="build-chunk-strategy")
def build_chunk_strategy_command(
    package_dir: Path = Path("outputs/package"),
    strategy: ChunkStrategy = "multimodal",
    output: Path | None = None,
    max_chars: int = 1600,
    overlap_chars: int = 180,
    min_chars: int = 180,
    context_prefix: bool = True,
    parent_max_chars: int = 900,
    visual_context_chars: int = 700,
):
    """Build an alternate chunk file for a named chunking strategy."""
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    strategy_chunks = build_strategy_chunks(
        chunks,
        assets,
        strategy=strategy,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
        include_context_prefix=context_prefix,
        parent_max_chars=parent_max_chars,
        visual_context_chars=visual_context_chars,
    )
    output_path = output or package_dir / f"chunks.{strategy}.jsonl"
    write_jsonl(output_path, strategy_chunks)
    print(
        {
            "strategy": strategy,
            "source_chunks": len(chunks),
            "strategy_chunks": len(strategy_chunks),
            "output": str(output_path),
        }
    )


@app.command(name="search-local")
def search_local(
    query: str,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    top_k: int = 5,
    graph_expand: bool = False,
    collapse_hierarchical: bool = False,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Run local dry-run hybrid search over package chunks using hashing dense + BM25."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    searcher = LocalHybridSearcher(
        chunks,
        HashingTextEmbedder(),
        triples=triples,
        tokenizer_config=build_tokenizer_config(
            lexical_tokenizer,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            ngram_cjk_only=ngram_cjk_only,
        ),
    )
    hits = searcher.search(
        query,
        top_k=top_k,
        graph_expand=graph_expand,
        collapse_hierarchical=collapse_hierarchical,
    )
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
                "evidence_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "page": [chunk.page_start, chunk.page_end],
                        "preview": chunk.text[:120],
                    }
                    for chunk in hit.evidence_chunks[:3]
                ],
            }
            for index, hit in enumerate(hits)
        ]
    )


@app.command(name="build-rag-context")
def build_rag_context_command(
    query: str,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    output: Path | None = None,
    top_k: int = 5,
    graph_expand: bool = False,
    collapse_hierarchical: bool = False,
    max_chars_per_chunk: int = 1400,
    include_evidence: bool = True,
    include_assets: bool = True,
    include_triples: bool = True,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Build a citation-ready RAG context bundle from local hybrid search hits."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    assets_path = package_dir / "assets.jsonl"
    triples_path = package_dir / "triples.jsonl"
    assets = read_jsonl(assets_path, VisualAsset) if assets_path.exists() else []
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    searcher = LocalHybridSearcher(
        chunks,
        HashingTextEmbedder(),
        triples=triples,
        tokenizer_config=build_tokenizer_config(
            lexical_tokenizer,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            ngram_cjk_only=ngram_cjk_only,
        ),
    )
    hits = searcher.search(
        query,
        top_k=top_k,
        graph_expand=graph_expand,
        collapse_hierarchical=collapse_hierarchical,
    )
    bundle = build_context_bundle(
        query=query,
        hits=hits,
        assets=assets,
        triples=triples,
        max_chars_per_chunk=max_chars_per_chunk,
        include_evidence=include_evidence,
        include_assets=include_assets,
        include_triples=include_triples,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        print({"output": str(output), **bundle.metadata})
        return
    print(bundle.model_dump())


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
    collapse_hierarchical: bool = False,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
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
        collapse_hierarchical=collapse_hierarchical,
        tokenizer_config=build_tokenizer_config(
            lexical_tokenizer,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            ngram_cjk_only=ngram_cjk_only,
        ),
    )
    print(evaluation.model_dump())


@app.command(name="eval-retrieval-ablation")
def eval_retrieval_ablation_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    output: Path | None = None,
    modes: str = "dense,bm25,hybrid,graph,hybrid_graph",
    top_k: int = 5,
    collapse_hierarchical: bool = False,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Compare dense, BM25, graph, and fused retrieval on the same cases."""
    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    try:
        parsed_modes = parse_ablation_modes(modes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = evaluate_retrieval_ablation(
        chunks=chunks,
        triples=triples,
        cases=load_retrieval_cases(cases),
        modes=parsed_modes,
        top_k=top_k,
        collapse_hierarchical=collapse_hierarchical,
        tokenizer_config=build_tokenizer_config(
            lexical_tokenizer,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            ngram_cjk_only=ngram_cjk_only,
        ),
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "best_by_recall": report.best_by_recall,
            "best_by_mrr": report.best_by_mrr,
            "rows": [
                {
                    "mode": row.mode.name,
                    "recall_at_k": row.evaluation.recall_at_k,
                    "mrr": row.evaluation.mrr,
                    "hit_rate": row.evaluation.hit_rate,
                    "failed_queries": row.evaluation.failed_queries,
                }
                for row in report.rows
            ],
        }
    print(payload)


@app.command(name="eval-chunking")
def eval_chunking_command(
    package_dir: Path = Path("outputs/package"),
    cases: Path | None = None,
    top_k: int = 5,
    min_chars: int = 120,
    max_chars: int = 1800,
    collapse_hierarchical: bool = False,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
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
        collapse_hierarchical=collapse_hierarchical,
        tokenizer_config=build_tokenizer_config(
            lexical_tokenizer,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            ngram_cjk_only=ngram_cjk_only,
        ),
    )
    print(report.model_dump())


@app.command(name="compare-chunking")
def compare_chunking_command(
    package_dir: Path = Path("outputs/package"),
    candidates: list[str] = typer.Option(
        None,
        "--candidate",
        help="Candidate in name=path form. Repeat for multiple chunk files.",
    ),
    cases: Path | None = None,
    top_k: int = 5,
    min_chars: int = 120,
    max_chars: int = 1800,
    collapse_hierarchical: bool = False,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Compare multiple chunk files with the same quality and retrieval metrics."""
    manifest = load_processing_package(package_dir)
    retrieval_cases = load_retrieval_cases(cases) if cases is not None else None
    parsed_candidates = parse_candidates(candidates, package_dir)
    reports = {}
    for name, path in parsed_candidates.items():
        chunks = read_jsonl(path, DocumentChunk)
        reports[name] = evaluate_chunking_quality(
            chunks=chunks,
            profiles=manifest.profiles,
            assets=manifest.assets,
            triples=manifest.triples,
            retrieval_cases=retrieval_cases,
            top_k=top_k,
            min_chars=min_chars,
            max_chars=max_chars,
            collapse_hierarchical=collapse_hierarchical,
            tokenizer_config=build_tokenizer_config(
                lexical_tokenizer,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
                ngram_cjk_only=ngram_cjk_only,
            ),
        )
    print(compare_chunking_reports(reports).model_dump())


@app.command(name="sweep-chunking")
def sweep_chunking_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    candidates_dir: Path | None = None,
    strategies: str = "semantic,multimodal,hierarchical",
    max_chars: list[int] = typer.Option(
        None,
        "--max-chars",
        help="Repeat to evaluate multiple max chunk sizes.",
    ),
    overlap_chars: list[int] = typer.Option(
        None,
        "--overlap-chars",
        help="Repeat to evaluate multiple overlap sizes.",
    ),
    min_chars: int = 180,
    parent_max_chars: list[int] = typer.Option(
        None,
        "--parent-max-chars",
        help="Repeat to evaluate hierarchical parent summary sizes.",
    ),
    visual_context_chars: list[int] = typer.Option(
        None,
        "--visual-context-chars",
        help="Repeat to evaluate hierarchical visual context sizes.",
    ),
    cases: Path | None = None,
    top_k: int = 5,
    collapse_hierarchical: bool = True,
    write_candidates: bool = True,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Generate and evaluate a grid of chunking strategy candidates."""
    manifest = load_processing_package(package_dir)
    retrieval_cases = load_retrieval_cases(cases) if cases is not None else None
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    candidate_output_dir = candidates_dir or package_dir / "chunking_sweep"
    report = run_chunking_sweep(
        chunks=manifest.chunks,
        assets=manifest.assets,
        profiles=manifest.profiles,
        triples=manifest.triples,
        strategies=parse_strategy_list(strategies),
        max_chars_values=max_chars or [1200, 1600],
        overlap_chars_values=overlap_chars or [120, 180],
        min_chars=min_chars,
        parent_max_chars_values=parent_max_chars or [700, 900],
        visual_context_chars_values=visual_context_chars or [500, 700],
        retrieval_cases=retrieval_cases,
        top_k=top_k,
        tokenizer_config=tokenizer_config,
        collapse_hierarchical=collapse_hierarchical,
        output_dir=candidate_output_dir,
        write_candidates=write_candidates,
    )
    output_path = output or package_dir / "chunking_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        {
            "output": str(output_path),
            "candidate_files_dir": str(candidate_output_dir) if write_candidates else None,
            "candidate_count": len(report.candidates),
            "best_by_quality": report.comparison.best_by_quality,
            "best_by_retrieval": report.comparison.best_by_retrieval,
            "top_candidates": [
                {
                    "name": candidate.name,
                    "quality_score": round(candidate.report.quality_score, 6),
                    "recall_at_k": candidate.report.retrieval.recall_at_k
                    if candidate.report.retrieval
                    else None,
                    "mrr": candidate.report.retrieval.mrr if candidate.report.retrieval else None,
                    "chunks_file": candidate.chunks_file,
                }
                for candidate in report.candidates[:5]
            ],
        }
    )


@app.command(name="write-experiment-report")
def write_experiment_report_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    candidates: list[str] = typer.Option(
        None,
        "--candidate",
        help="Candidate in name=path form. Repeat for multiple chunk files.",
    ),
    cases: Path | None = None,
    top_k: int = 5,
    min_chars: int = 120,
    max_chars: int = 1800,
    collapse_hierarchical: bool = False,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Write a reproducible experiment report for package artifacts and chunk candidates."""
    manifest = load_processing_package(package_dir)
    retrieval_cases = load_retrieval_cases(cases) if cases is not None else None
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    parsed_candidates = parse_candidates(candidates, package_dir)
    report = build_experiment_report(
        package_dir=package_dir,
        manifest=manifest,
        candidates=parsed_candidates,
        retrieval_cases=retrieval_cases,
        top_k=top_k,
        min_chars=min_chars,
        max_chars=max_chars,
        tokenizer_config=tokenizer_config,
        collapse_hierarchical=collapse_hierarchical,
        config={
            "top_k": top_k,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "collapse_hierarchical": collapse_hierarchical,
            "retrieval_cases": str(cases) if cases else None,
            "lexical_tokenizer": tokenizer_config.model_dump(),
        },
    )
    output_path = output or package_dir / "experiment_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        {
            "output": str(output_path),
            "candidates": list(parsed_candidates),
            "best_by_quality": report.comparison.best_by_quality if report.comparison else None,
            "best_by_retrieval": report.comparison.best_by_retrieval if report.comparison else None,
        }
    )


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


def build_qdrant_query_embedders(
    selected_vectors: list[str],
    vector_sizes: dict[str, int],
    default_embedder,
    image_query_backend: str,
    image_query_model: str,
    device: str,
    hashing_dim: int,
):
    if "image_dense" not in selected_vectors:
        return {}
    normalized = normalize_backend(image_query_backend)
    if normalized == "none":
        image_size = vector_sizes.get("image_dense")
        if image_size is not None and image_size != default_embedder.embedding_dim:
            raise typer.BadParameter(
                "--image-query-backend is required when image_dense size differs from the text query encoder"
            )
        return {}
    if normalized == "same-as-text":
        return {"image_dense": default_embedder}
    if normalized == "hashing":
        from chunking_docs.embeddings.interfaces import HashingTextEmbedder

        return {"image_dense": HashingTextEmbedder(embedding_dim=hashing_dim)}
    if normalized in {"clip", "transformers"}:
        from chunking_docs.embeddings.clip import TransformersCLIPTextEmbedder

        return {"image_dense": TransformersCLIPTextEmbedder(model_name=image_query_model, device=device)}
    raise typer.BadParameter("image query backend must be one of: none, same-as-text, hashing, clip")


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


def parse_candidates(values: list[str] | None, package_dir: Path) -> dict[str, Path]:
    if not values:
        return {"current": package_dir / "chunks.jsonl"}
    parsed = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter("--candidate must be in name=path form")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter("--candidate name must not be empty")
        candidate_path = Path(path)
        if not candidate_path.is_absolute() and not candidate_path.exists():
            candidate_path = package_dir / candidate_path
        parsed[name] = candidate_path
    return parsed


def parse_strategy_list(value: str) -> list[ChunkStrategy]:
    allowed = {"page", "semantic", "multimodal", "hierarchical"}
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    if not strategies:
        raise typer.BadParameter("--strategies must include at least one strategy")
    invalid = sorted(set(strategies) - allowed)
    if invalid:
        raise typer.BadParameter(f"Unsupported strategies: {', '.join(invalid)}")
    return strategies


def upsert_package_records(store, package_dir: Path) -> int:
    total = 0
    for record_file in sorted(package_dir.glob("qdrant_*_records.jsonl")):
        records = read_jsonl(record_file, EmbeddingRecord)
        result = store.upsert(records)
        total += result.count
    return total


def build_tokenizer_config(
    strategy: TokenizerStrategy,
    ngram_min: int,
    ngram_max: int,
    ngram_cjk_only: bool,
) -> LexicalTokenizerConfig:
    return LexicalTokenizerConfig(
        strategy=strategy,
        min_n=ngram_min,
        max_n=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )

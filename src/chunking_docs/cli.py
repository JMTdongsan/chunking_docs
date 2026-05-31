from __future__ import annotations

import builtins
import json
from pathlib import Path
from time import perf_counter

import httpx
import typer
from rich import print

from chunking_docs.analysis.characterize import characterize_package
from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles, write_profile_outputs
from chunking_docs.chunking.multimodal import ChunkStrategy, build_strategy_chunks
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.chunking.section_map import load_section_ranges
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig, TokenizerStrategy
from chunking_docs.evaluation.audit import audit_package
from chunking_docs.evaluation.ablation import (
    QdrantVectorAblationReport,
    QdrantVectorAblationRow,
    build_qdrant_vector_ablation_report,
    evaluate_retrieval_ablation,
    gate_qdrant_vector_ablation,
    parse_ablation_modes,
    parse_qdrant_vector_ablation_modes,
    qdrant_vector_names_for_modes,
)
from chunking_docs.evaluation.casegen import generate_retrieval_case_skeleton
from chunking_docs.evaluation.case_audit import audit_retrieval_cases
from chunking_docs.evaluation.chunking_gate import (
    chunking_gate_summary_payload,
    gate_chunking_comparison,
    load_chunking_comparison,
)
from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.compare import compare_chunking_reports
from chunking_docs.evaluation.diagnostics import (
    analyze_retrieval_evaluation,
    load_retrieval_evaluation,
)
from chunking_docs.evaluation.delta import compare_processing_packages
from chunking_docs.evaluation.experiment import build_experiment_report
from chunking_docs.evaluation.gate import gate_retrieval_evaluation, gate_summary_payload
from chunking_docs.evaluation.readiness import build_ingestion_readiness_report
from chunking_docs.evaluation.retrieval import (
    evaluate_retrieval,
    evaluate_search_results,
    load_retrieval_cases,
)
from chunking_docs.evaluation.sweep import run_chunking_sweep
from chunking_docs.graph.heuristics import section_triples
from chunking_docs.graph.quality import normalize_graph_triples
from chunking_docs.graph.repair import remap_triples_to_available_chunks
from chunking_docs.ingest.pdf_loader import load_source_document, render_pages
from chunking_docs.ingest.tables import extract_pdf_tables
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import AssetKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.pipeline import (
    build_processing_package,
    load_processing_package,
    rebuild_search_artifacts,
    write_embedding_artifacts,
    write_split_chunks,
)
from chunking_docs.retrieval.local_hybrid import LocalHybridSearcher
from chunking_docs.retrieval.context import build_context_bundle
from chunking_docs.retrieval.rerank import (
    LexicalOverlapReranker,
    SentenceTransformerCrossEncoderReranker,
)
from chunking_docs.runtime import inspect_runtime
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.annotate import annotate_assets, merge_asset_annotations_into_chunks
from chunking_docs.vision.assets import (
    attach_assets_to_chunks,
    build_page_tile_assets,
    merge_visual_assets,
)
from chunking_docs.vision.compare import compare_visual_runs
from chunking_docs.vision.experiments import build_vlm_experiment_plan, parse_profile_list
from chunking_docs.vision.interfaces import OCRBackend, VLMBackend
from chunking_docs.vision.jobs import (
    VisualAnnotationJob,
    VisualJobRunResult,
    completed_annotations,
    plan_visual_jobs,
    run_visual_jobs,
)
from chunking_docs.vision.manual_annotations import AssetAnnotation, apply_asset_annotations
from chunking_docs.vision.quality import evaluate_visual_results, visual_results_from_assets
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


@app.command(name="doctor")
def doctor_command(
    output: Path | None = None,
    require_gpu: bool = False,
    require_qdrant: bool = False,
    require_postgres: bool = False,
    require_embeddings: bool = False,
    require_ocr: bool = False,
    require_vision: bool = False,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when required runtime checks fail.",
    ),
):
    """Inspect optional runtime dependencies for storage, embedding, OCR, VLM, and GPU work."""
    report = inspect_runtime(
        require_gpu=require_gpu,
        require_qdrant=require_qdrant,
        require_postgres=require_postgres,
        require_embeddings=require_embeddings,
        require_ocr=require_ocr,
        require_vision=require_vision,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "gpu_count": len(report.gpus),
            "failed_checks": [check.name for check in report.checks if not check.passed],
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


@app.command()
def render(pdf: Path, output_dir: Path = Path("outputs/renders"), pages: str = ""):
    """Render selected PDF pages to PNG."""
    page_numbers = sorted(parse_page_numbers(pages)) or None
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
    extract_tables: bool = True,
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
        extract_tables=extract_tables,
    )
    print(
        {
            "doc_id": manifest.doc.doc_id,
            "pages": len(manifest.profiles),
            "chunks": len(manifest.chunks),
            "assets": len(manifest.assets),
            "triples": len(manifest.triples),
            "tables": manifest.metadata.get("table_count", 0),
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
    payload_indexes = collection_config.get("payload_indexes", [])
    store.ensure_collection(named_vectors, payload_indexes=payload_indexes)

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
            "payload_indexes": payload_indexes,
        }
    )


@app.command(name="qdrant-check-collection")
def qdrant_check_collection(
    package_dir: Path = Path("outputs/package"),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    output: Path | None = None,
    allow_missing: bool = False,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when the existing Qdrant collection does not match the package contract.",
    ),
):
    """Validate an existing Qdrant collection against package named vectors and payload indexes."""
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
    report = store.check_collection_contract(
        named_vectors,
        payload_indexes=collection_config.get("payload_indexes", []),
        allow_missing=allow_missing,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "exists": report.exists,
            "collection": report.collection,
            "failed_checks": report.failed_checks,
            "missing_vectors": report.missing_vectors,
            "mismatched_vectors": report.mismatched_vectors,
            "missing_payload_indexes": report.missing_payload_indexes,
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


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
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
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
    store.ensure_collection(named_vectors, payload_indexes=collection_config.get("payload_indexes", []))
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
    filters = build_payload_filter(doc_id=doc_id, filter_specs=payload_filter)
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
            "filters": filters,
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
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as bm25=1.3, graph=0.8, qdrant:caption_dense=1.5.",
    ),
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Run Qdrant named-vector + BM25 + optional graph hybrid retrieval."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    parsed_reranker = build_reranker(
        reranker,
        model_name=reranker_model,
        device=reranker_device,
        max_length=reranker_max_length,
        tokenizer_config=tokenizer_config,
    )
    prepared = prepare_qdrant_hybrid_search(
        package_dir=package_dir,
        url=url,
        collection=collection,
        location=location,
        path=path,
        vector_names=vector_names,
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
        device=device,
        hashing_dim=hashing_dim,
        lexical_tokenizer=lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    hits = prepared["searcher"].search(
        query=query,
        vector_names=prepared["selected_vectors"],
        top_k=top_k,
        graph_expand=graph_expand,
        doc_id=doc_id or None,
        payload_filter=build_payload_filter(filter_specs=payload_filter),
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
        reranker=parsed_reranker,
        rerank_top_k=rerank_top_k,
    )
    print(
        {
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "query_encoders": prepared["query_encoders"],
            "upserted": prepared["upserted"],
            "stored_count": prepared["store"].count(),
            "filters": build_payload_filter(doc_id=doc_id, filter_specs=payload_filter),
            "fusion_weights": fusion_weights,
            "reranker": parsed_reranker.source if parsed_reranker else None,
            "rerank_top_k": rerank_top_k if parsed_reranker else None,
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


@app.command(name="qdrant-rag-context")
def qdrant_rag_context_command(
    query: str,
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    vector_names: str = "text_dense,caption_dense",
    top_k: int = 5,
    graph_expand: bool = False,
    collapse_hierarchical: bool = False,
    max_chars_per_chunk: int = 1400,
    include_evidence: bool = True,
    neighbor_window: int = 0,
    include_assets: bool = True,
    include_triples: bool = True,
    text_backend: str = "hashing",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "none",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    doc_id: str = "",
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as bm25=1.3, graph=0.8, qdrant:caption_dense=1.5.",
    ),
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Build a citation-ready RAG context bundle from Qdrant hybrid search hits."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    parsed_reranker = build_reranker(
        reranker,
        model_name=reranker_model,
        device=reranker_device,
        max_length=reranker_max_length,
        tokenizer_config=tokenizer_config,
    )
    prepared = prepare_qdrant_hybrid_search(
        package_dir=package_dir,
        url=url,
        collection=collection,
        location=location,
        path=path,
        vector_names=vector_names,
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
        device=device,
        hashing_dim=hashing_dim,
        lexical_tokenizer=lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    hits = prepared["searcher"].search(
        query=query,
        vector_names=prepared["selected_vectors"],
        top_k=top_k,
        graph_expand=graph_expand,
        doc_id=doc_id or None,
        payload_filter=build_payload_filter(filter_specs=payload_filter),
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
        reranker=parsed_reranker,
        rerank_top_k=rerank_top_k,
    )
    bundle = build_context_bundle(
        query=query,
        hits=hits,
        chunks=prepared["chunks"],
        assets=prepared["assets"],
        triples=prepared["triples"],
        max_chars_per_chunk=max_chars_per_chunk,
        include_evidence=include_evidence,
        neighbor_window=neighbor_window,
        include_assets=include_assets,
        include_triples=include_triples,
    )
    bundle.metadata.update(
        {
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "query_encoders": prepared["query_encoders"],
            "upserted": prepared["upserted"],
            "stored_count": prepared["store"].count(),
            "filters": build_payload_filter(doc_id=doc_id, filter_specs=payload_filter),
            "fusion_weights": fusion_weights,
            "reranker": parsed_reranker.source if parsed_reranker else None,
            "rerank_top_k": rerank_top_k if parsed_reranker else None,
        }
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        print({"output": str(output), **bundle.metadata})
        return
    print(bundle.model_dump())


@app.command(name="eval-qdrant-retrieval")
def eval_qdrant_retrieval_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    vector_names: str = "text_dense,caption_dense",
    top_k: int = 5,
    repeat: int = 1,
    collapse_hierarchical: bool = False,
    text_backend: str = "hashing",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "none",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    doc_id: str = "",
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as bm25=1.3, graph=0.8, qdrant:caption_dense=1.5.",
    ),
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Evaluate Qdrant hybrid retrieval against JSONL benchmark cases."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    parsed_reranker = build_reranker(
        reranker,
        model_name=reranker_model,
        device=reranker_device,
        max_length=reranker_max_length,
        tokenizer_config=tokenizer_config,
    )
    prepare_start = perf_counter()
    prepared = prepare_qdrant_hybrid_search(
        package_dir=package_dir,
        url=url,
        collection=collection,
        location=location,
        path=path,
        vector_names=vector_names,
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
        device=device,
        hashing_dim=hashing_dim,
        lexical_tokenizer=lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    index_build_ms = (perf_counter() - prepare_start) * 1000
    evaluation = evaluate_search_results(
        cases=load_retrieval_cases(cases),
        search_fn=lambda case, graph_expand: prepared["searcher"].search(
            query=case.query,
            vector_names=prepared["selected_vectors"],
            top_k=top_k,
            graph_expand=graph_expand,
            doc_id=doc_id or None,
            payload_filter=build_payload_filter(filter_specs=payload_filter),
            collapse_hierarchical=collapse_hierarchical,
            fusion_weights=fusion_weights,
            reranker=parsed_reranker,
            rerank_top_k=rerank_top_k,
        ),
        top_k=top_k,
        repeat=repeat,
        index_build_ms=index_build_ms,
        triples=prepared["triples"],
    )
    evaluation.metadata.update(
        {
            "backend": "qdrant_hybrid",
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "query_encoders": prepared["query_encoders"],
            "upserted": prepared["upserted"],
            "stored_count": prepared["store"].count(),
            "filters": build_payload_filter(doc_id=doc_id, filter_specs=payload_filter),
            "fusion_weights": fusion_weights,
            "reranker": parsed_reranker.source if parsed_reranker else None,
            "rerank_top_k": rerank_top_k if parsed_reranker else None,
        }
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
        print(
            {
                "output": str(output),
                "case_count": evaluation.case_count,
                "recall_at_k": evaluation.recall_at_k,
                "mrr": evaluation.mrr,
                "target_coverage_at_k": evaluation.target_coverage_at_k,
                "mean_target_ndcg_at_k": evaluation.mean_target_ndcg_at_k,
                "mean_precision_at_k": evaluation.mean_precision_at_k,
                "mean_latency_ms": evaluation.mean_latency_ms,
                "p95_latency_ms": evaluation.p95_latency_ms,
                "target_metrics": retrieval_target_metrics_payload(evaluation),
                "source_family_metrics": retrieval_source_family_metrics_payload(evaluation),
                **evaluation.metadata,
            }
        )
        return
    print(evaluation.model_dump())


@app.command(name="eval-qdrant-vector-ablation")
def eval_qdrant_vector_ablation_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    modes: str = "text,caption,text_caption,text_caption_graph",
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    top_k: int = 5,
    repeat: int = 1,
    collapse_hierarchical: bool = False,
    text_backend: str = "hashing",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "none",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    doc_id: str = "",
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as bm25=1.3, graph=0.8, qdrant:caption_dense=1.5.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Compare Qdrant text, visual caption, image, and graph retrieval signals."""
    try:
        parsed_modes = parse_qdrant_vector_ablation_modes(modes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    fusion_weights = parse_fusion_weights(fusion_weight)
    selected_vectors = qdrant_vector_names_for_modes(parsed_modes)
    prepare_start = perf_counter()
    prepared = prepare_qdrant_hybrid_search(
        package_dir=package_dir,
        url=url,
        collection=collection,
        location=location,
        path=path,
        vector_names=",".join(selected_vectors),
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
        device=device,
        hashing_dim=hashing_dim,
        lexical_tokenizer=lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    index_build_ms = (perf_counter() - prepare_start) * 1000
    retrieval_cases = load_retrieval_cases(cases)
    filters = build_payload_filter(filter_specs=payload_filter)
    metadata_filters = build_payload_filter(doc_id=doc_id, filter_specs=payload_filter)

    rows = []
    for mode in parsed_modes:
        evaluation = evaluate_search_results(
            cases=retrieval_cases,
            search_fn=lambda case, graph_expand, mode=mode: prepared["searcher"].search(
                query=case.query,
                vector_names=mode.vector_names,
                top_k=top_k,
                graph_expand=graph_expand,
                doc_id=doc_id or None,
                payload_filter=filters,
                collapse_hierarchical=collapse_hierarchical,
                fusion_weights=fusion_weights,
            ),
            top_k=top_k,
            repeat=repeat,
            index_build_ms=index_build_ms,
            graph_expand_override=mode.graph_expand,
            triples=prepared["triples"],
        )
        evaluation.metadata.update(
            {
                "backend": "qdrant_hybrid",
                "mode": mode.name,
                "collection": prepared["collection_name"],
                "vector_names": mode.vector_names,
                "graph_expand": mode.graph_expand,
                "query_encoders": {
                    name: prepared["query_encoders"].get(name, "unknown")
                    for name in mode.vector_names
                },
                "upserted": prepared["upserted"],
                "stored_count": prepared["store"].count(),
                "filters": metadata_filters,
                "fusion_weights": fusion_weights,
            }
        )
        rows.append(QdrantVectorAblationRow(mode=mode, evaluation=evaluation))

    report = build_qdrant_vector_ablation_report(rows)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(
            {
                "output": str(output),
                "best_by_recall": report.best_by_recall,
                "best_by_target_coverage": report.best_by_target_coverage,
                "best_by_target_ndcg": report.best_by_target_ndcg,
                "best_by_mrr": report.best_by_mrr,
                "fastest_by_mean_latency": report.fastest_by_mean_latency,
                "rows": [
                    {
                        "mode": row.mode.name,
                        "vector_names": row.mode.vector_names,
                        "graph_expand": row.mode.graph_expand,
                        "recall_at_k": row.evaluation.recall_at_k,
                        "mrr": row.evaluation.mrr,
                        "hit_rate": row.evaluation.hit_rate,
                        "target_coverage_at_k": row.evaluation.target_coverage_at_k,
                        "mean_target_ndcg_at_k": row.evaluation.mean_target_ndcg_at_k,
                        "mean_precision_at_k": row.evaluation.mean_precision_at_k,
                        "repeat": row.evaluation.repeat,
                        "mean_latency_ms": row.evaluation.mean_latency_ms,
                        "p95_latency_ms": row.evaluation.p95_latency_ms,
                        "target_metrics": retrieval_target_metrics_payload(row.evaluation),
                        "source_family_metrics": retrieval_source_family_metrics_payload(
                            row.evaluation
                        ),
                        "failed_queries": row.evaluation.failed_queries,
                    }
                    for row in report.rows
                ],
            }
        )
        return
    print(report.model_dump())


@app.command(name="gate-qdrant-vector-ablation")
def gate_qdrant_vector_ablation_command(
    report: Path,
    mode: str = typer.Option(
        ...,
        "--mode",
        help="Ablation mode to gate, such as image or caption_image.",
    ),
    output: Path | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-family-target-coverage",
        help="Require source-family target coverage such as visual=0.8. Repeat for multiple families.",
    ),
    require_best_by_recall: bool = False,
    require_best_by_target_coverage: bool = False,
    require_best_by_target_ndcg: bool = False,
    require_fastest_by_mean_latency: bool = False,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when the vector ablation gate fails.",
    ),
):
    """Fail a Qdrant vector ablation mode when retrieval metrics are below thresholds."""
    parsed_report = QdrantVectorAblationReport.model_validate_json(
        report.read_text(encoding="utf-8")
    )
    source_family_thresholds = parse_named_float_thresholds(
        min_source_family_target_coverage,
        "source family target coverage",
    )
    try:
        gate_report = gate_qdrant_vector_ablation(
            parsed_report,
            mode=mode,
            min_recall_at_k=min_recall_at_k,
            min_target_coverage_at_k=min_target_coverage_at_k,
            min_target_ndcg_at_k=min_target_ndcg_at_k,
            min_mrr=min_mrr,
            min_precision_at_k=min_precision_at_k,
            max_failed_queries=max_failed_queries,
            max_mean_latency_ms=max_mean_latency_ms,
            max_p95_latency_ms=max_p95_latency_ms,
            min_source_family_target_coverage=source_family_thresholds,
            require_best_by_recall=require_best_by_recall,
            require_best_by_target_coverage=require_best_by_target_coverage,
            require_best_by_target_ndcg=require_best_by_target_ndcg,
            require_fastest_by_mean_latency=require_fastest_by_mean_latency,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = gate_report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(gate_report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": gate_report.passed,
            "mode": gate_report.mode,
            "vector_names": gate_report.vector_names,
            "failed_checks": gate_report.failed_checks,
            "metrics": gate_report.metrics,
            "source_family_metrics": gate_report.source_family_metrics,
        }
    print(payload)
    if fail and not gate_report.passed:
        raise typer.Exit(1)


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


@app.command(name="build-tile-assets")
def build_tile_assets_command(
    package_dir: Path = Path("outputs/package"),
    pdf: Path | None = None,
    pages: str = "",
    rows: int = 2,
    cols: int = 2,
    overlap_ratio: float = 0.08,
    zoom: float = 2.0,
    in_place: bool = True,
    rebuild_search: bool = True,
):
    """Create overlapping page-tile visual assets for OCR/VLM processing."""
    manifest = load_processing_package(package_dir)
    pdf_path = pdf or manifest.doc.local_path
    selected_pages = parse_page_numbers(pages) or None
    try:
        tile_assets = build_page_tile_assets(
            pdf_path=pdf_path,
            doc_id=manifest.doc.doc_id,
            profiles=manifest.profiles,
            output_dir=package_dir / "assets",
            pages=selected_pages,
            rows=rows,
            cols=cols,
            overlap_ratio=overlap_ratio,
            zoom=zoom,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    tile_assets = apply_chunk_section_labels(tile_assets, manifest.chunks)

    existing_assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    merged_assets = merge_visual_assets(existing_assets, tile_assets)
    updated_chunks = attach_assets_to_chunks(manifest.chunks, merged_assets)

    asset_output = package_dir / ("assets.jsonl" if in_place else "assets.tiled.jsonl")
    chunk_output = package_dir / ("chunks.jsonl" if in_place else "chunks.tiled.jsonl")
    write_jsonl(asset_output, merged_assets)
    write_jsonl(chunk_output, updated_chunks)
    if in_place and rebuild_search:
        rebuild_search_artifacts(package_dir, updated_chunks, assets=merged_assets)

    print(
        {
            "tile_assets": len(tile_assets),
            "total_assets": len(merged_assets),
            "asset_output": str(asset_output),
            "chunk_output": str(chunk_output),
            "rows": rows,
            "cols": cols,
            "overlap_ratio": overlap_ratio,
            "rebuilt_search": bool(in_place and rebuild_search),
        }
    )


@app.command(name="extract-tables")
def extract_tables_command(
    package_dir: Path = Path("outputs/package"),
    pdf: Path | None = None,
    pages: str = "",
    section_map: Path | None = None,
    min_rows: int = 2,
    min_cols: int = 2,
    zoom: float = 2.0,
    in_place: bool = True,
    rebuild_search: bool = True,
):
    """Extract PDF table assets and table chunks from a processing package."""
    if min_rows <= 0:
        raise typer.BadParameter("--min-rows must be positive")
    if min_cols <= 0:
        raise typer.BadParameter("--min-cols must be positive")
    manifest = load_processing_package(package_dir)
    pdf_path = pdf or manifest.doc.local_path
    selected_pages = parse_page_numbers(pages) or None
    table_assets, table_chunks = extract_pdf_tables(
        pdf_path=pdf_path,
        doc_id=manifest.doc.doc_id,
        output_dir=package_dir / "assets",
        section_ranges=load_section_ranges(section_map),
        pages=selected_pages,
        min_rows=min_rows,
        min_cols=min_cols,
        zoom=zoom,
    )

    existing_assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    merged_assets = merge_visual_assets(existing_assets, table_assets)
    base_chunks = [
        chunk
        for chunk in manifest.chunks
        if chunk.metadata.get("source") != "pdf_table_detection"
    ]
    updated_chunks = sorted(
        [*attach_assets_to_chunks(base_chunks, merged_assets), *table_chunks],
        key=lambda chunk: (chunk.page_start, chunk.page_end, str(chunk.kind), chunk.chunk_id),
    )

    asset_output = package_dir / ("assets.jsonl" if in_place else "assets.tables.jsonl")
    chunk_output = package_dir / ("chunks.jsonl" if in_place else "chunks.tables.jsonl")
    triple_output = package_dir / ("triples.jsonl" if in_place else "triples.tables.jsonl")
    triples_path = package_dir / "triples.jsonl"
    existing_triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    updated_triples = normalize_graph_triples(
        [*existing_triples, *section_triples(table_chunks)]
    )
    write_jsonl(asset_output, merged_assets)
    write_jsonl(chunk_output, updated_chunks)
    write_jsonl(triple_output, updated_triples)
    if in_place and rebuild_search:
        rebuild_search_artifacts(package_dir, updated_chunks, assets=merged_assets)

    print(
        {
            "table_assets": len(table_assets),
            "table_chunks": len(table_chunks),
            "total_assets": len(merged_assets),
            "total_chunks": len(updated_chunks),
            "triples": len(updated_triples),
            "asset_output": str(asset_output),
            "chunk_output": str(chunk_output),
            "triple_output": str(triple_output),
            "rebuilt_search": bool(in_place and rebuild_search),
        }
    )


@app.command(name="annotate-assets")
def annotate_assets_command(
    package_dir: Path = Path("outputs/package"),
    pages: str = "",
    limit: int | None = None,
    ocr: str = "none",
    ocr_model_lang: str = "korean",
    ocr_device: str = "",
    ocr_engine: str = "",
    ocr_min_confidence: float = 0.0,
    ocr_use_gpu: bool = False,
    ocr_enable_mkldnn: bool = False,
    vlm: str = "none",
    vlm_profile: str = "",
    vlm_model: str = "",
    vlm_model_class: str = "auto",
    vlm_device_map: str = "auto",
    vlm_torch_dtype: str = "auto",
    vlm_max_new_tokens: int = 768,
    vlm_attn_implementation: str = "",
    in_place: bool = False,
    rebuild_search: bool = True,
):
    """Annotate rendered assets with OCR/VLM output and merge it into chunks."""
    selected_pages = parse_page_numbers(pages) or None
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)

    ocr_backend, _ = build_ocr_backend(
        ocr,
        model_lang=ocr_model_lang,
        device=ocr_device,
        engine=ocr_engine,
        min_confidence=ocr_min_confidence,
        use_gpu=ocr_use_gpu,
        enable_mkldnn=ocr_enable_mkldnn,
    )
    vlm_backend, _ = build_vlm_backend(
        vlm,
        vlm_model,
        profile=vlm_profile,
        model_class=vlm_model_class,
        device_map=vlm_device_map,
        torch_dtype=vlm_torch_dtype,
        max_new_tokens=vlm_max_new_tokens,
        attn_implementation=vlm_attn_implementation,
    )

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
    kinds: list[AssetKind] = typer.Option(
        None,
        "--kind",
        help="Restrict jobs to asset kinds such as map, table, figure, or page_image.",
    ),
    include_ocr: bool = True,
    include_vlm: bool = True,
):
    """Plan prioritized OCR/VLM jobs for rendered visual assets."""
    selected_pages = parse_page_numbers(pages) or None
    selected_kinds = set(kinds) if kinds else None
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    jobs = plan_visual_jobs(
        assets,
        pages=selected_pages,
        kinds=selected_kinds,
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
            "filtered_kinds": [str(kind) for kind in sorted(selected_kinds)] if selected_kinds else [],
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
    ocr_model_lang: str = "korean",
    ocr_device: str = "",
    ocr_engine: str = "",
    ocr_min_confidence: float = 0.0,
    ocr_use_gpu: bool = False,
    ocr_enable_mkldnn: bool = False,
    vlm: str = "none",
    vlm_profile: str = "",
    vlm_model: str = "",
    vlm_model_class: str = "auto",
    vlm_device_map: str = "auto",
    vlm_torch_dtype: str = "auto",
    vlm_max_new_tokens: int = 768,
    vlm_attn_implementation: str = "",
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

    ocr_backend, ocr_name = build_ocr_backend(
        ocr,
        model_lang=ocr_model_lang,
        device=ocr_device,
        engine=ocr_engine,
        min_confidence=ocr_min_confidence,
        use_gpu=ocr_use_gpu,
        enable_mkldnn=ocr_enable_mkldnn,
    )
    vlm_backend, vlm_name = build_vlm_backend(
        vlm,
        vlm_model,
        profile=vlm_profile,
        model_class=vlm_model_class,
        device_map=vlm_device_map,
        torch_dtype=vlm_torch_dtype,
        max_new_tokens=vlm_max_new_tokens,
        attn_implementation=vlm_attn_implementation,
    )
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


@app.command(name="summarize-visual-assets")
def summarize_visual_assets_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    required_only: bool = typer.Option(
        True,
        "--required-only/--all-assets",
        help="Summarize only assets marked as requiring OCR/VLM, or every asset with visual text.",
    ),
):
    """Summarize the final OCR/VLM annotation state currently stored in assets.jsonl."""
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    results = visual_results_from_assets(assets, required_only=required_only)
    summary = summarize_visual_results(results)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        print({"output": str(output), **summary.model_dump()})
        return
    print(summary.model_dump())


@app.command(name="compare-visual-runs")
def compare_visual_runs_command(
    runs: list[str] = typer.Option(
        None,
        "--run",
        help="Visual run in name=results.jsonl form. Repeat for multiple OCR/VLM runs.",
    ),
    output: Path | None = None,
):
    """Compare multiple OCR/VLM result files by coverage, parse rate, triples, and latency."""
    parsed_runs = parse_visual_run_inputs(runs)
    comparison = compare_visual_runs(
        {
            name: read_jsonl(path, VisualJobRunResult)
            for name, path in parsed_runs.items()
        }
    )
    payload = comparison.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "run_count": len(comparison.rows),
            "best_by_quality": comparison.best_by_quality,
            "fastest_by_total_latency": comparison.fastest_by_total_latency,
            "best_by_triple_density": comparison.best_by_triple_density,
        }
    print(payload)


@app.command(name="plan-vlm-experiments")
def plan_vlm_experiments_command(
    package_dir: Path = Path("outputs/package"),
    jobs: Path | None = None,
    profiles: str = "qwen2_5_vl_7b,qwen2_vl_7b,llava_next_7b",
    output: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    ocr: str = "paddleocr",
    ocr_model_lang: str = "korean",
    ocr_device: str = "cpu",
    ocr_min_confidence: float = 0.3,
    ocr_use_gpu: bool = False,
    ocr_enable_mkldnn: bool = False,
    vlm_device_map: str = "auto",
    vlm_torch_dtype: str = "auto",
    vlm_max_new_tokens: int | None = None,
    vlm_attn_implementation: str = "",
):
    """Write a reproducible command plan for comparing multiple Hugging Face VLM profiles."""
    jobs_path = jobs or package_dir / "visual_jobs.priority.jsonl"
    try:
        plan = build_vlm_experiment_plan(
            package_dir=package_dir,
            jobs_file=jobs_path,
            profiles=parse_profile_list(profiles),
            output_dir=output_dir,
            limit=limit,
            ocr=ocr,
            ocr_model_lang=ocr_model_lang,
            ocr_device=ocr_device,
            ocr_min_confidence=ocr_min_confidence,
            ocr_use_gpu=ocr_use_gpu,
            ocr_enable_mkldnn=ocr_enable_mkldnn,
            vlm_device_map=vlm_device_map,
            vlm_torch_dtype=vlm_torch_dtype,
            vlm_max_new_tokens=vlm_max_new_tokens,
            vlm_attn_implementation=vlm_attn_implementation,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = output or package_dir / "vlm_experiment_plan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    print(
        {
            "output": str(output_path),
            "profile_count": len(plan.profiles),
            "profiles": plan.profiles,
            "jobs_file": plan.jobs_file,
            "compare_command": plan.compare_command,
        }
    )


@app.command(name="gate-visual-results")
def gate_visual_results_command(
    results: Path = Path("outputs/package/visual_job_results.jsonl"),
    output: Path | None = None,
    min_completion_rate: float = 0.0,
    min_annotation_rate: float = 0.0,
    min_ocr_text_coverage: float = 0.0,
    min_vlm_summary_coverage: float = 0.0,
    min_vlm_json_parse_rate: float = 0.0,
    min_triples_per_vlm_job: float = 0.0,
    min_mean_ocr_text_chars: float = 0.0,
    min_mean_vlm_summary_chars: float = 0.0,
    max_failed_count: int | None = None,
    max_skipped_count: int | None = None,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when visual quality checks fail.",
    ),
):
    """Fail an OCR/VLM run when usable visual annotation quality is too low."""
    parsed_results = read_jsonl(results, VisualJobRunResult)
    report = evaluate_visual_results(
        parsed_results,
        min_completion_rate=min_completion_rate,
        min_annotation_rate=min_annotation_rate,
        min_ocr_text_coverage=min_ocr_text_coverage,
        min_vlm_summary_coverage=min_vlm_summary_coverage,
        min_vlm_json_parse_rate=min_vlm_json_parse_rate,
        min_triples_per_vlm_job=min_triples_per_vlm_job,
        min_mean_ocr_text_chars=min_mean_ocr_text_chars,
        min_mean_vlm_summary_chars=min_mean_vlm_summary_chars,
        max_failed_count=max_failed_count,
        max_skipped_count=max_skipped_count,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "failed_checks": report.failed_checks,
            "completion_rate": report.completion_rate,
            "annotation_rate": report.annotation_rate,
            "ocr_text_coverage": report.ocr_text_coverage,
            "vlm_summary_coverage": report.vlm_summary_coverage,
            "vlm_json_parse_rate": report.vlm_json_parse_rate,
            "triples_per_vlm_job": report.triples_per_vlm_job,
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


@app.command(name="gate-visual-assets")
def gate_visual_assets_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    required_only: bool = typer.Option(
        True,
        "--required-only/--all-assets",
        help="Gate only assets marked as requiring OCR/VLM, or every asset with visual text.",
    ),
    min_completion_rate: float = 0.0,
    min_annotation_rate: float = 0.0,
    min_ocr_text_coverage: float = 0.0,
    min_vlm_summary_coverage: float = 0.0,
    min_vlm_json_parse_rate: float = 0.0,
    min_mean_ocr_text_chars: float = 0.0,
    min_mean_vlm_summary_chars: float = 0.0,
    max_failed_count: int | None = None,
    max_skipped_count: int | None = None,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when visual asset quality checks fail.",
    ),
):
    """Fail a package when the applied visual annotations in assets.jsonl are incomplete."""
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    report = evaluate_visual_results(
        visual_results_from_assets(assets, required_only=required_only),
        min_completion_rate=min_completion_rate,
        min_annotation_rate=min_annotation_rate,
        min_ocr_text_coverage=min_ocr_text_coverage,
        min_vlm_summary_coverage=min_vlm_summary_coverage,
        min_vlm_json_parse_rate=min_vlm_json_parse_rate,
        min_mean_ocr_text_chars=min_mean_ocr_text_chars,
        min_mean_vlm_summary_chars=min_mean_vlm_summary_chars,
        max_failed_count=max_failed_count,
        max_skipped_count=max_skipped_count,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "failed_checks": report.failed_checks,
            "completion_rate": report.completion_rate,
            "annotation_rate": report.annotation_rate,
            "ocr_text_coverage": report.ocr_text_coverage,
            "vlm_summary_coverage": report.vlm_summary_coverage,
            "vlm_json_parse_rate": report.vlm_json_parse_rate,
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


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
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Run local dry-run hybrid search over package chunks using hashing dense + BM25."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    parsed_reranker = build_reranker(
        reranker,
        model_name=reranker_model,
        device=reranker_device,
        max_length=reranker_max_length,
        tokenizer_config=tokenizer_config,
    )
    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    searcher = LocalHybridSearcher(
        chunks,
        HashingTextEmbedder(),
        triples=triples,
        tokenizer_config=tokenizer_config,
    )
    hits = searcher.search(
        query,
        top_k=top_k,
        graph_expand=graph_expand,
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
        reranker=parsed_reranker,
        rerank_top_k=rerank_top_k,
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
    neighbor_window: int = 0,
    include_assets: bool = True,
    include_triples: bool = True,
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Build a citation-ready RAG context bundle from local hybrid search hits."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    parsed_reranker = build_reranker(
        reranker,
        model_name=reranker_model,
        device=reranker_device,
        max_length=reranker_max_length,
        tokenizer_config=tokenizer_config,
    )
    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    assets_path = package_dir / "assets.jsonl"
    triples_path = package_dir / "triples.jsonl"
    assets = read_jsonl(assets_path, VisualAsset) if assets_path.exists() else []
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    searcher = LocalHybridSearcher(
        chunks,
        HashingTextEmbedder(),
        triples=triples,
        tokenizer_config=tokenizer_config,
    )
    hits = searcher.search(
        query,
        top_k=top_k,
        graph_expand=graph_expand,
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
        reranker=parsed_reranker,
        rerank_top_k=rerank_top_k,
    )
    bundle = build_context_bundle(
        query=query,
        hits=hits,
        chunks=chunks,
        assets=assets,
        triples=triples,
        max_chars_per_chunk=max_chars_per_chunk,
        include_evidence=include_evidence,
        neighbor_window=neighbor_window,
        include_assets=include_assets,
        include_triples=include_triples,
    )
    bundle.metadata["fusion_weights"] = fusion_weights
    if parsed_reranker is not None:
        bundle.metadata["reranker"] = parsed_reranker.source
        bundle.metadata["rerank_top_k"] = rerank_top_k
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


@app.command(name="audit-graph-triples")
def audit_graph_triples_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    max_issues: int = 200,
):
    """Audit graph triples for normalization, duplicate, and provenance issues."""
    from chunking_docs.graph.quality import audit_graph_triples

    triples_path = package_dir / "triples.jsonl"
    chunks_path = package_dir / "chunks.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    chunks = read_jsonl(chunks_path, DocumentChunk) if chunks_path.exists() else None
    report = audit_graph_triples(triples, chunks=chunks, max_issues=max_issues)
    payload = report.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_json(payload)


@app.command(name="normalize-graph-triples")
def normalize_graph_triples_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    in_place: bool = False,
    dedupe: bool = True,
    export_graph_artifacts: bool = typer.Option(False, "--export-graph"),
):
    """Normalize graph triple labels and optionally rebuild graph JSONL artifacts."""
    from chunking_docs.graph.export import export_graph
    from chunking_docs.graph.quality import (
        audit_graph_triples,
        graph_triple_has_required_fields,
        normalize_graph_triple,
        normalize_graph_triples,
    )

    triples_path = package_dir / "triples.jsonl"
    if not triples_path.exists():
        raise typer.BadParameter(f"Missing triples file: {triples_path}")
    chunks_path = package_dir / "chunks.jsonl"
    chunks = read_jsonl(chunks_path, DocumentChunk) if chunks_path.exists() else []
    triples = read_jsonl(triples_path, GraphTriple)
    before = audit_graph_triples(triples, chunks=chunks if chunks_path.exists() else None)
    valid_normalized = [
        triple
        for triple in (normalize_graph_triple(triple) for triple in triples)
        if graph_triple_has_required_fields(triple)
    ]
    normalized = normalize_graph_triples(triples, dedupe=dedupe)
    output_path = package_dir / "triples.jsonl" if in_place else output
    if output_path is None:
        output_path = package_dir / "triples.normalized.jsonl"
    write_jsonl(output_path, normalized)

    graph_nodes_output = None
    graph_edges_output = None
    if export_graph_artifacts:
        nodes, edges = export_graph(normalized, chunks=chunks)
        graph_nodes_output = package_dir / "graph_nodes.jsonl"
        graph_edges_output = package_dir / "graph_edges.jsonl"
        write_jsonl(graph_nodes_output, nodes)
        write_jsonl(graph_edges_output, edges)

    print_json(
        {
            "source": str(triples_path),
            "output": str(output_path),
            "input_triples": len(triples),
            "output_triples": len(normalized),
            "normalized_triples": before.normalized_count,
            "removed_invalid": len(triples) - len(valid_normalized),
            "removed_duplicates": len(valid_normalized) - len(normalized),
            "dedupe": dedupe,
            "graph_nodes_output": str(graph_nodes_output) if graph_nodes_output else None,
            "graph_edges_output": str(graph_edges_output) if graph_edges_output else None,
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


@app.command(name="postgres-check-schema")
def postgres_check_schema(
    dsn: str,
    output: Path | None = None,
    apply_schema: bool = False,
    require_pgvector: bool = True,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when the PostgreSQL schema contract check fails.",
    ),
):
    """Validate PostgreSQL tables, columns, types, and pgvector extension before metadata upsert."""
    from chunking_docs.storage.postgres_store import PostgresDocumentStore

    store = PostgresDocumentStore(dsn)
    if apply_schema:
        store.apply_schema()
    report = store.check_schema(require_pgvector=require_pgvector)
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "failed_checks": report.failed_checks,
            "missing_extensions": report.missing_extensions,
            "missing_tables": report.missing_tables,
            "missing_columns": report.missing_columns,
            "type_mismatches": report.type_mismatches,
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


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
            "embedding_artifacts": len(rows["embedding_artifacts"]),
            "first_document": rows["document"],
        }
    )


@app.command(name="audit-package")
def audit_package_command(
    package_dir: Path = Path("outputs/package"),
    require_annotations_for_visual_pages: bool = False,
    require_qdrant_records: bool = False,
):
    """Audit package completeness and remaining OCR/VLM work."""
    manifest = load_processing_package(package_dir)
    audit = audit_package(
        manifest.profiles,
        manifest.chunks,
        manifest.assets,
        manifest.triples,
        require_annotations_for_visual_pages=require_annotations_for_visual_pages,
        package_dir=package_dir,
        require_qdrant_records=require_qdrant_records,
    )
    print(audit.model_dump())


@app.command(name="ingestion-readiness")
def ingestion_readiness_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    require_qdrant_records: bool = True,
    require_bm25: bool = True,
    require_embedding_manifest: bool = True,
    require_postgres_rows: bool = True,
    require_visual_annotations: bool = False,
    visual_results: Path | None = None,
    require_visual_quality: bool = False,
    min_visual_completion_rate: float = 0.0,
    min_ocr_text_coverage: float = 0.0,
    min_vlm_summary_coverage: float = 0.0,
    min_vlm_json_parse_rate: float = 0.0,
    max_visual_failed_count: int | None = None,
    retrieval_cases: Path | None = None,
    require_retrieval_cases: bool = False,
    min_case_count: int = 1,
    min_page_cases: int = 0,
    min_chunk_cases: int = 0,
    min_asset_cases: int = 0,
    min_triple_cases: int = 0,
    max_duplicate_queries: int = 0,
    retrieval_evaluation: Path | None = None,
    require_retrieval_evaluation: bool = False,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_p95_latency_ms: float | None = None,
    min_retrieval_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-source-family-target-coverage",
        help="Require retrieval source-family target coverage such as lexical=0.8.",
    ),
    chunking_comparison: Path | None = None,
    require_chunking_comparison: bool = False,
    chunking_candidate: str | None = None,
    baseline_chunking_candidate: str | None = None,
    min_chunking_quality_score: float = 0.0,
    min_chunking_recall_at_k: float | None = None,
    min_chunking_target_coverage_at_k: float | None = None,
    min_chunking_target_ndcg_at_k: float | None = None,
    max_chunking_failed_queries: int | None = 0,
    max_chunking_recall_drop: float | None = None,
    max_chunking_mean_latency_ratio: float | None = None,
    qdrant_vector_ablation: Path | None = None,
    require_qdrant_vector_ablation: bool = False,
    qdrant_vector_mode: str | None = None,
    min_qdrant_vector_recall_at_k: float = 0.0,
    min_qdrant_vector_target_coverage_at_k: float = 0.0,
    min_qdrant_vector_target_ndcg_at_k: float = 0.0,
    min_qdrant_vector_mrr: float = 0.0,
    min_qdrant_vector_precision_at_k: float = 0.0,
    max_qdrant_vector_failed_queries: int | None = None,
    max_qdrant_vector_mean_latency_ms: float | None = None,
    max_qdrant_vector_p95_latency_ms: float | None = None,
    min_qdrant_vector_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-vector-source-family-target-coverage",
        help="Require selected Qdrant vector source-family coverage such as visual=0.8.",
    ),
    require_qdrant_vector_best_by_recall: bool = False,
    require_qdrant_vector_best_by_target_coverage: bool = False,
    require_qdrant_vector_best_by_target_ndcg: bool = False,
    require_qdrant_vector_fastest_by_mean_latency: bool = False,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when ingestion readiness checks fail.",
    ),
):
    """Check whether a package is ready for Qdrant/PostgreSQL ingestion and RAG evaluation."""
    manifest = load_processing_package(package_dir)
    parsed_visual_results = read_jsonl(visual_results, VisualJobRunResult) if visual_results else None
    parsed_retrieval_cases = load_retrieval_cases(retrieval_cases) if retrieval_cases else None
    parsed_retrieval = load_retrieval_evaluation(retrieval_evaluation) if retrieval_evaluation else None
    parsed_chunking_comparison = load_chunking_comparison(chunking_comparison) if chunking_comparison else None
    parsed_qdrant_vector_ablation = (
        QdrantVectorAblationReport.model_validate_json(
            qdrant_vector_ablation.read_text(encoding="utf-8")
        )
        if qdrant_vector_ablation
        else None
    )
    qdrant_vector_source_family_thresholds = parse_named_float_thresholds(
        min_qdrant_vector_source_family_target_coverage,
        "Qdrant vector source family target coverage",
    )
    retrieval_source_family_thresholds = parse_named_float_thresholds(
        min_retrieval_source_family_target_coverage,
        "retrieval source family target coverage",
    )
    report = build_ingestion_readiness_report(
        package_dir=package_dir,
        manifest=manifest,
        require_qdrant_records=require_qdrant_records,
        require_bm25=require_bm25,
        require_embedding_manifest=require_embedding_manifest,
        require_postgres_rows=require_postgres_rows,
        require_visual_annotations=require_visual_annotations,
        visual_results=parsed_visual_results,
        require_visual_quality=require_visual_quality,
        visual_quality_options={
            "min_completion_rate": min_visual_completion_rate,
            "min_ocr_text_coverage": min_ocr_text_coverage,
            "min_vlm_summary_coverage": min_vlm_summary_coverage,
            "min_vlm_json_parse_rate": min_vlm_json_parse_rate,
            "max_failed_count": max_visual_failed_count,
        },
        retrieval_cases=parsed_retrieval_cases,
        require_retrieval_cases=require_retrieval_cases,
        retrieval_case_options={
            "min_case_count": min_case_count,
            "min_page_cases": min_page_cases,
            "min_chunk_cases": min_chunk_cases,
            "min_asset_cases": min_asset_cases,
            "min_triple_cases": min_triple_cases,
            "max_duplicate_queries": max_duplicate_queries,
        },
        retrieval_evaluation=parsed_retrieval,
        require_retrieval_evaluation=require_retrieval_evaluation,
        retrieval_gate_options={
            "min_recall_at_k": min_recall_at_k,
            "min_target_coverage_at_k": min_target_coverage_at_k,
            "min_target_ndcg_at_k": min_target_ndcg_at_k,
            "min_precision_at_k": min_precision_at_k,
            "max_p95_latency_ms": max_p95_latency_ms,
            "min_source_family_target_coverage": retrieval_source_family_thresholds,
        },
        chunking_comparison=parsed_chunking_comparison,
        require_chunking_comparison=require_chunking_comparison,
        chunking_gate_options={
            "candidate": chunking_candidate,
            "baseline_candidate": baseline_chunking_candidate,
            "min_quality_score": min_chunking_quality_score,
            "min_recall_at_k": min_chunking_recall_at_k,
            "min_target_coverage_at_k": min_chunking_target_coverage_at_k,
            "min_target_ndcg_at_k": min_chunking_target_ndcg_at_k,
            "max_failed_queries": max_chunking_failed_queries,
            "max_recall_drop": max_chunking_recall_drop,
            "max_mean_latency_ratio": max_chunking_mean_latency_ratio,
        },
        qdrant_vector_ablation=parsed_qdrant_vector_ablation,
        require_qdrant_vector_ablation=require_qdrant_vector_ablation,
        qdrant_vector_ablation_mode=qdrant_vector_mode,
        qdrant_vector_ablation_gate_options={
            "min_recall_at_k": min_qdrant_vector_recall_at_k,
            "min_target_coverage_at_k": min_qdrant_vector_target_coverage_at_k,
            "min_target_ndcg_at_k": min_qdrant_vector_target_ndcg_at_k,
            "min_mrr": min_qdrant_vector_mrr,
            "min_precision_at_k": min_qdrant_vector_precision_at_k,
            "max_failed_queries": max_qdrant_vector_failed_queries,
            "max_mean_latency_ms": max_qdrant_vector_mean_latency_ms,
            "max_p95_latency_ms": max_qdrant_vector_p95_latency_ms,
            "min_source_family_target_coverage": qdrant_vector_source_family_thresholds,
            "require_best_by_recall": require_qdrant_vector_best_by_recall,
            "require_best_by_target_coverage": require_qdrant_vector_best_by_target_coverage,
            "require_best_by_target_ndcg": require_qdrant_vector_best_by_target_ndcg,
            "require_fastest_by_mean_latency": require_qdrant_vector_fastest_by_mean_latency,
        },
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "failed_components": report.failed_components,
            "package_counts": report.package_counts,
            "postgres_row_counts": report.postgres_row_counts,
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


@app.command(name="characterize-package")
def characterize_package_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    max_pages: int = 25,
):
    """Summarize document characteristics that guide chunking, visual, and retrieval strategy."""
    manifest = load_processing_package(package_dir)
    report = characterize_package(
        profiles=manifest.profiles,
        chunks=manifest.chunks,
        assets=manifest.assets,
        triples=manifest.triples,
        package_dir=package_dir,
        max_pages=max_pages,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "page_count": report.text_layer.page_count,
            "degraded_or_empty_ratio": report.text_layer.degraded_or_empty_ratio,
            "asset_kind_counts": report.visual.asset_kind_counts,
            "observations": [observation.code for observation in report.observations],
            "recommendations": [recommendation.code for recommendation in report.recommendations],
        }
    print(payload)


@app.command(name="compare-packages")
def compare_packages_command(
    before_dir: Path,
    after_dir: Path,
    output: Path | None = None,
    max_ids: int = 50,
):
    """Compare two processing packages before and after annotation or strategy changes."""
    before = load_processing_package(before_dir)
    after = load_processing_package(after_dir)
    report = compare_processing_packages(
        before,
        after,
        before_dir=before_dir,
        after_dir=after_dir,
        max_ids=max_ids,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "count_delta": report.count_delta,
            "changed_ids": report.changed_ids,
            "qdrant_record_count_delta": report.qdrant_record_count_delta,
            "observations": [observation.code for observation in report.observations],
        }
    print(payload)


@app.command(name="eval-retrieval")
def eval_retrieval_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    output: Path | None = None,
    top_k: int = 5,
    repeat: int = 1,
    collapse_hierarchical: bool = False,
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Evaluate local hybrid retrieval against JSONL seed cases."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
    )
    parsed_reranker = build_reranker(
        reranker,
        model_name=reranker_model,
        device=reranker_device,
        max_length=reranker_max_length,
        tokenizer_config=tokenizer_config,
    )
    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    evaluation = evaluate_retrieval(
        chunks=chunks,
        triples=triples,
        cases=load_retrieval_cases(cases),
        top_k=top_k,
        repeat=repeat,
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
        reranker=parsed_reranker,
        rerank_top_k=rerank_top_k,
        tokenizer_config=tokenizer_config,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
        print(
            {
                "output": str(output),
                "case_count": evaluation.case_count,
                "recall_at_k": evaluation.recall_at_k,
                "mrr": evaluation.mrr,
                "target_coverage_at_k": evaluation.target_coverage_at_k,
                "mean_target_ndcg_at_k": evaluation.mean_target_ndcg_at_k,
                "mean_precision_at_k": evaluation.mean_precision_at_k,
                "mean_latency_ms": evaluation.mean_latency_ms,
                "p95_latency_ms": evaluation.p95_latency_ms,
                "target_metrics": retrieval_target_metrics_payload(evaluation),
                "source_family_metrics": retrieval_source_family_metrics_payload(evaluation),
            }
        )
        return
    print(evaluation.model_dump())


@app.command(name="generate-retrieval-cases")
def generate_retrieval_cases_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    chunks: Path | None = None,
    page_limit: int = 20,
    asset_limit: int = 20,
    triple_limit: int = 20,
    include_pages: bool = True,
    include_assets: bool = True,
    include_triples: bool = True,
    include_todo: bool = False,
    query_max_chars: int = 120,
    query_mode: str = "snippet",
    selection_strategy: str = "document_order",
    min_query_terms: int = 3,
    max_query_terms: int = 8,
    dedupe_queries: bool = True,
):
    """Generate retrieval benchmark JSONL drafts from package chunks, assets, and triples."""
    manifest = load_processing_package(package_dir)
    case_chunks = read_jsonl(chunks, DocumentChunk) if chunks is not None else manifest.chunks
    try:
        cases = generate_retrieval_case_skeleton(
            chunks=case_chunks,
            assets=manifest.assets,
            triples=manifest.triples,
            page_limit=page_limit,
            asset_limit=asset_limit,
            triple_limit=triple_limit,
            include_pages=include_pages,
            include_assets=include_assets,
            include_triples=include_triples,
            include_todo=include_todo,
            query_max_chars=query_max_chars,
            query_mode=query_mode,
            selection_strategy=selection_strategy,
            min_query_terms=min_query_terms,
            max_query_terms=max_query_terms,
            dedupe_queries=dedupe_queries,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = output or package_dir / "retrieval_cases.skeleton.jsonl"
    write_jsonl(output_path, cases)
    print(
        {
            "output": str(output_path),
            "case_count": len(cases),
            "page_limit": page_limit,
            "asset_limit": asset_limit,
            "triple_limit": triple_limit,
            "include_todo": include_todo,
            "query_mode": query_mode,
            "selection_strategy": selection_strategy,
            "chunks": str(chunks) if chunks is not None else str(package_dir / "chunks.jsonl"),
            "dedupe_queries": dedupe_queries,
        }
    )


@app.command(name="audit-retrieval-cases")
def audit_retrieval_cases_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    min_case_count: int = 1,
    min_page_cases: int = 0,
    min_chunk_cases: int = 0,
    min_asset_cases: int = 0,
    min_triple_cases: int = 0,
    max_duplicate_queries: int = 0,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when retrieval case audit fails.",
    ),
):
    """Validate retrieval benchmark cases against package targets and target-family coverage."""
    manifest = load_processing_package(package_dir)
    parsed_cases = load_retrieval_cases(cases)
    report = audit_retrieval_cases(
        parsed_cases,
        profiles=manifest.profiles,
        chunks=manifest.chunks,
        assets=manifest.assets,
        triples=manifest.triples,
        min_case_count=min_case_count,
        min_page_cases=min_page_cases,
        min_chunk_cases=min_chunk_cases,
        min_asset_cases=min_asset_cases,
        min_triple_cases=min_triple_cases,
        max_duplicate_queries=max_duplicate_queries,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "case_count": report.case_count,
            "target_counts": report.target_counts,
            "missing_target_counts": report.missing_target_counts,
            "failed_checks": report.failed_checks,
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


@app.command(name="eval-retrieval-ablation")
def eval_retrieval_ablation_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    chunks_file: str = "chunks.jsonl",
    output: Path | None = None,
    modes: str = "dense,bm25,hybrid,graph,hybrid_graph",
    top_k: int = 5,
    repeat: int = 1,
    collapse_hierarchical: bool = False,
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Compare dense, BM25, graph, and fused retrieval on the same cases."""
    fusion_weights = parse_fusion_weights(fusion_weight)
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
        repeat=repeat,
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
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
            "best_by_target_coverage": report.best_by_target_coverage,
            "best_by_target_ndcg": report.best_by_target_ndcg,
            "best_by_mrr": report.best_by_mrr,
            "fastest_by_mean_latency": report.fastest_by_mean_latency,
            "rows": [
                {
                    "mode": row.mode.name,
                    "recall_at_k": row.evaluation.recall_at_k,
                    "mrr": row.evaluation.mrr,
                    "hit_rate": row.evaluation.hit_rate,
                    "target_coverage_at_k": row.evaluation.target_coverage_at_k,
                    "mean_target_ndcg_at_k": row.evaluation.mean_target_ndcg_at_k,
                    "mean_precision_at_k": row.evaluation.mean_precision_at_k,
                    "repeat": row.evaluation.repeat,
                    "mean_latency_ms": row.evaluation.mean_latency_ms,
                    "p95_latency_ms": row.evaluation.p95_latency_ms,
                    "target_metrics": retrieval_target_metrics_payload(row.evaluation),
                    "source_family_metrics": retrieval_source_family_metrics_payload(row.evaluation),
                    "failed_queries": row.evaluation.failed_queries,
                }
                for row in report.rows
            ],
        }
    print(payload)


@app.command(name="diagnose-retrieval")
def diagnose_retrieval_command(
    evaluation: Path,
    output: Path | None = None,
    precision_floor: float = 0.2,
    target_ndcg_floor: float = 0.7,
    include_passed: bool = False,
):
    """Analyze retrieval evaluation failures and partial target coverage."""
    if precision_floor < 0.0 or precision_floor > 1.0:
        raise typer.BadParameter("--precision-floor must be between 0.0 and 1.0")
    if target_ndcg_floor < 0.0 or target_ndcg_floor > 1.0:
        raise typer.BadParameter("--target-ndcg-floor must be between 0.0 and 1.0")
    parsed_evaluation = load_retrieval_evaluation(evaluation)
    report = analyze_retrieval_evaluation(
        parsed_evaluation,
        precision_floor=precision_floor,
        target_ndcg_floor=target_ndcg_floor,
        include_passed=include_passed,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "case_count": report.case_count,
            "failed_count": report.failed_count,
            "partial_count": report.partial_count,
            "no_hit_count": report.no_hit_count,
            "low_precision_count": report.low_precision_count,
            "low_target_ndcg_count": report.low_target_ndcg_count,
            "reason_counts": report.reason_counts,
            "missing_target_type_counts": report.missing_target_type_counts,
        }
    print(payload)


@app.command(name="gate-retrieval")
def gate_retrieval_command(
    evaluation: Path,
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="Optional baseline retrieval evaluation JSON for regression checks.",
    ),
    output: Path | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-family-target-coverage",
        help="Require source-family target coverage such as lexical=0.8. Repeat for multiple families.",
    ),
    max_recall_drop: float | None = None,
    max_target_coverage_drop: float | None = None,
    max_target_ndcg_drop: float | None = None,
    max_precision_drop: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when any gate check fails.",
    ),
):
    """Fail a retrieval run when absolute metrics or baseline regression limits are missed."""
    parsed_evaluation = load_retrieval_evaluation(evaluation)
    parsed_baseline = load_retrieval_evaluation(baseline) if baseline else None
    source_family_thresholds = parse_named_float_thresholds(
        min_source_family_target_coverage,
        "source family target coverage",
    )
    report = gate_retrieval_evaluation(
        parsed_evaluation,
        baseline=parsed_baseline,
        min_recall_at_k=min_recall_at_k,
        min_target_coverage_at_k=min_target_coverage_at_k,
        min_target_ndcg_at_k=min_target_ndcg_at_k,
        min_mrr=min_mrr,
        min_precision_at_k=min_precision_at_k,
        max_mean_latency_ms=max_mean_latency_ms,
        max_p95_latency_ms=max_p95_latency_ms,
        min_source_family_target_coverage=source_family_thresholds,
        max_recall_drop=max_recall_drop,
        max_target_coverage_drop=max_target_coverage_drop,
        max_target_ndcg_drop=max_target_ndcg_drop,
        max_precision_drop=max_precision_drop,
        max_mean_latency_ratio=max_mean_latency_ratio,
        max_p95_latency_ratio=max_p95_latency_ratio,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {"output": str(output), **gate_summary_payload(report)}
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


@app.command(name="eval-chunking")
def eval_chunking_command(
    package_dir: Path = Path("outputs/package"),
    cases: Path | None = None,
    top_k: int = 5,
    retrieval_repeat: int = 1,
    min_chars: int = 120,
    max_chars: int = 1800,
    collapse_hierarchical: bool = False,
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Evaluate chunking quality and optional retrieval performance."""
    manifest = load_processing_package(package_dir)
    fusion_weights = parse_fusion_weights(fusion_weight)
    retrieval_cases = load_retrieval_cases(cases) if cases is not None else None
    report = evaluate_chunking_quality(
        chunks=manifest.chunks,
        profiles=manifest.profiles,
        assets=manifest.assets,
        triples=manifest.triples,
        retrieval_cases=retrieval_cases,
        top_k=top_k,
        retrieval_repeat=retrieval_repeat,
        min_chars=min_chars,
        max_chars=max_chars,
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
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
    output: Path | None = None,
    candidates: list[str] = typer.Option(
        None,
        "--candidate",
        help="Candidate in name=path form. Repeat for multiple chunk files.",
    ),
    cases: Path | None = None,
    top_k: int = 5,
    retrieval_repeat: int = 1,
    min_chars: int = 120,
    max_chars: int = 1800,
    collapse_hierarchical: bool = False,
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Compare multiple chunk files with the same quality and retrieval metrics."""
    manifest = load_processing_package(package_dir)
    fusion_weights = parse_fusion_weights(fusion_weight)
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
            retrieval_repeat=retrieval_repeat,
            min_chars=min_chars,
            max_chars=max_chars,
            collapse_hierarchical=collapse_hierarchical,
            fusion_weights=fusion_weights,
            tokenizer_config=build_tokenizer_config(
                lexical_tokenizer,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
                ngram_cjk_only=ngram_cjk_only,
            ),
        )
    comparison = compare_chunking_reports(reports)
    payload = comparison.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "candidate_count": len(comparison.rows),
            "best_by_quality": comparison.best_by_quality,
            "best_by_retrieval": comparison.best_by_retrieval,
            "fastest_by_mean_latency": comparison.fastest_by_mean_latency,
        }
    print(payload)


@app.command(name="gate-chunking-comparison")
def gate_chunking_comparison_command(
    comparison: Path,
    output: Path | None = None,
    candidate: str | None = None,
    baseline_candidate: str | None = None,
    require_retrieval: bool = typer.Option(
        False,
        "--require-retrieval/--allow-quality-only",
        help="Require the selected candidate to contain retrieval metrics.",
    ),
    min_quality_score: float = 0.0,
    min_page_coverage_ratio: float = 1.0,
    min_visual_annotation_ratio: float | None = None,
    min_recall_at_k: float | None = None,
    min_target_coverage_at_k: float | None = None,
    min_target_ndcg_at_k: float | None = None,
    min_mrr: float | None = None,
    min_precision_at_k: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_failed_queries: int | None = 0,
    max_chunks_under_min_chars: int | None = None,
    max_chunks_over_max_chars: int | None = None,
    max_quality_drop: float | None = None,
    max_recall_drop: float | None = None,
    max_target_coverage_drop: float | None = None,
    max_target_ndcg_drop: float | None = None,
    max_precision_drop: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when any gate check fails.",
    ),
):
    """Fail a chunking strategy comparison when quality, retrieval, or latency checks are missed."""
    parsed_comparison = load_chunking_comparison(comparison)
    report = gate_chunking_comparison(
        parsed_comparison,
        candidate=candidate,
        baseline_candidate=baseline_candidate,
        require_retrieval=require_retrieval,
        min_quality_score=min_quality_score,
        min_page_coverage_ratio=min_page_coverage_ratio,
        min_visual_annotation_ratio=min_visual_annotation_ratio,
        min_recall_at_k=min_recall_at_k,
        min_target_coverage_at_k=min_target_coverage_at_k,
        min_target_ndcg_at_k=min_target_ndcg_at_k,
        min_mrr=min_mrr,
        min_precision_at_k=min_precision_at_k,
        max_mean_latency_ms=max_mean_latency_ms,
        max_p95_latency_ms=max_p95_latency_ms,
        max_failed_queries=max_failed_queries,
        max_chunks_under_min_chars=max_chunks_under_min_chars,
        max_chunks_over_max_chars=max_chunks_over_max_chars,
        max_quality_drop=max_quality_drop,
        max_recall_drop=max_recall_drop,
        max_target_coverage_drop=max_target_coverage_drop,
        max_target_ndcg_drop=max_target_ndcg_drop,
        max_precision_drop=max_precision_drop,
        max_mean_latency_ratio=max_mean_latency_ratio,
        max_p95_latency_ratio=max_p95_latency_ratio,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {"output": str(output), **chunking_gate_summary_payload(report)}
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


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
        help="Repeat to evaluate multimodal and hierarchical visual context sizes.",
    ),
    cases: Path | None = None,
    top_k: int = 5,
    retrieval_repeat: int = 1,
    collapse_hierarchical: bool = True,
    write_candidates: bool = True,
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Generate and evaluate a grid of chunking strategy candidates."""
    manifest = load_processing_package(package_dir)
    fusion_weights = parse_fusion_weights(fusion_weight)
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
        retrieval_repeat=retrieval_repeat,
        fusion_weights=fusion_weights,
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
            "fastest_by_mean_latency": report.comparison.fastest_by_mean_latency,
            "top_candidates": [
                {
                    "name": candidate.name,
                    "quality_score": round(candidate.report.quality_score, 6),
                    "recall_at_k": candidate.report.retrieval.recall_at_k
                    if candidate.report.retrieval
                    else None,
                    "target_coverage_at_k": candidate.report.retrieval.target_coverage_at_k
                    if candidate.report.retrieval
                    else None,
                    "mean_target_ndcg_at_k": candidate.report.retrieval.mean_target_ndcg_at_k
                    if candidate.report.retrieval
                    else None,
                    "mrr": candidate.report.retrieval.mrr if candidate.report.retrieval else None,
                    "mean_latency_ms": candidate.report.retrieval.mean_latency_ms
                    if candidate.report.retrieval
                    else None,
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
    retrieval_repeat: int = 1,
    min_chars: int = 120,
    max_chars: int = 1800,
    collapse_hierarchical: bool = False,
    fusion_weight: list[str] = typer.Option(
        None,
        "--fusion-weight",
        help="RRF source weight such as dense=1.0, bm25=1.3, graph=0.8.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
):
    """Write a reproducible experiment report for package artifacts and chunk candidates."""
    manifest = load_processing_package(package_dir)
    fusion_weights = parse_fusion_weights(fusion_weight)
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
        retrieval_repeat=retrieval_repeat,
        min_chars=min_chars,
        max_chars=max_chars,
        tokenizer_config=tokenizer_config,
        collapse_hierarchical=collapse_hierarchical,
        fusion_weights=fusion_weights,
        config={
            "top_k": top_k,
            "retrieval_repeat": retrieval_repeat,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "collapse_hierarchical": collapse_hierarchical,
            "retrieval_cases": str(cases) if cases else None,
            "lexical_tokenizer": tokenizer_config.model_dump(),
            "fusion_weights": fusion_weights,
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
            "fastest_by_mean_latency": report.comparison.fastest_by_mean_latency
            if report.comparison
            else None,
        }
    )


def print_json(payload: dict):
    builtins.print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_page_numbers(value: str) -> set[int]:
    pages: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0:
                raise typer.BadParameter("page numbers must be positive")
            if end < start:
                raise typer.BadParameter("page ranges must be ascending")
            pages.update(range(start, end + 1))
        else:
            page = int(item)
            if page <= 0:
                raise typer.BadParameter("page numbers must be positive")
            pages.add(page)
    return pages


def apply_chunk_section_labels(
    assets: list[VisualAsset],
    chunks: list[DocumentChunk],
) -> list[VisualAsset]:
    labels = section_labels_by_page(chunks)
    updated_assets = []
    for asset in assets:
        label = labels.get(asset.page_no)
        if not label:
            updated_assets.append(asset)
            continue
        updated_assets.append(
            asset.model_copy(update={"metadata": {**asset.metadata, "section_label": label}})
        )
    return updated_assets


def section_labels_by_page(chunks: list[DocumentChunk]) -> dict[int, str]:
    labels = {}
    for chunk in chunks:
        label = chunk.section.label() or str(chunk.metadata.get("section_label", ""))
        if not label:
            continue
        for page_no in range(chunk.page_start, chunk.page_end + 1):
            labels.setdefault(page_no, label)
    return labels


def retrieval_target_metrics_payload(evaluation) -> dict:
    return {
        name: metric.model_dump()
        for name, metric in getattr(evaluation, "target_metrics", {}).items()
    }


def retrieval_source_family_metrics_payload(evaluation) -> dict:
    return {
        name: metric.model_dump()
        for name, metric in getattr(evaluation, "source_family_metrics", {}).items()
    }


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


def build_payload_filter(
    doc_id: str = "",
    filter_specs: list[str] | None = None,
) -> dict:
    filters = {"doc_id": doc_id} if doc_id else {}
    for spec in filter_specs or []:
        key, operator, value = parse_payload_filter(spec)
        if operator == "match":
            filters[key] = value
            continue
        existing = filters.get(key)
        range_filter = existing if isinstance(existing, dict) else {}
        range_filter[operator] = value
        filters[key] = range_filter
    return filters


def parse_fusion_weights(specs: list[str] | None = None) -> dict[str, float]:
    weights: dict[str, float] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise typer.BadParameter("fusion weights must use source=weight")
        source, raw_weight = spec.split("=", 1)
        source = source.strip()
        if not source:
            raise typer.BadParameter("fusion weight source must not be empty")
        try:
            weight = float(raw_weight.strip())
        except ValueError as exc:
            raise typer.BadParameter(f"fusion weight for {source} must be numeric") from exc
        if weight < 0:
            raise typer.BadParameter(f"fusion weight for {source} must be non-negative")
        weights[source] = weight
    return weights


def parse_named_float_thresholds(
    specs: list[str] | None = None,
    label: str = "threshold",
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise typer.BadParameter(f"{label} must use name=value")
        name, raw_value = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter(f"{label} name must not be empty")
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise typer.BadParameter(f"{label} for {name} must be numeric") from exc
        if value < 0:
            raise typer.BadParameter(f"{label} for {name} must be non-negative")
        thresholds[name] = value
    return thresholds


def build_reranker(
    backend: str,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cuda",
    max_length: int = 0,
    tokenizer_config: LexicalTokenizerConfig | None = None,
):
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None
    if normalized == "lexical":
        return LexicalOverlapReranker(tokenizer_config=tokenizer_config)
    if normalized in {"cross-encoder", "cross_encoder", "sentence-transformers"}:
        return SentenceTransformerCrossEncoderReranker(
            model_name=model_name,
            device=device,
            max_length=max_length or None,
        )
    raise typer.BadParameter("reranker must be one of: none, lexical, cross-encoder")


def parse_payload_filter(spec: str) -> tuple[str, str, object]:
    for token, operator in [
        (">=", "gte"),
        ("<=", "lte"),
        (">", "gt"),
        ("<", "lt"),
        ("=", "match"),
    ]:
        if token in spec:
            key, raw_value = spec.split(token, 1)
            key = key.strip()
            if not key:
                break
            return key, operator, parse_payload_filter_value(raw_value.strip())
    raise typer.BadParameter(
        "payload filters must use key=value, key>=value, key<=value, key>value, or key<value"
    )


def parse_payload_filter_value(value: str):
    if not value:
        raise typer.BadParameter("payload filter value must not be empty")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def prepare_qdrant_hybrid_search(
    package_dir: Path,
    url: str,
    collection: str,
    location: str,
    path: str,
    vector_names: str,
    text_backend: str,
    text_model: str,
    image_query_backend: str,
    image_query_model: str,
    device: str,
    hashing_dim: int,
    lexical_tokenizer: TokenizerStrategy,
    ngram_min: int,
    ngram_max: int,
    ngram_cjk_only: bool,
):
    from chunking_docs.retrieval.qdrant_hybrid import QdrantHybridSearcher
    from chunking_docs.storage.qdrant_store import QdrantChunkStore

    collection_config = json.loads((package_dir / "qdrant_collection.json").read_text(encoding="utf-8"))
    collection_name = collection or collection_config["collection"]
    named_vectors = {
        name: int(config["size"])
        for name, config in collection_config.get("named_vectors", {}).items()
    }
    selected_vectors = [item.strip() for item in vector_names.split(",") if item.strip()]
    unknown_vectors = sorted(set(selected_vectors) - set(named_vectors))
    if unknown_vectors:
        raise typer.BadParameter(
            f"Unknown Qdrant named vectors for this package: {', '.join(unknown_vectors)}"
        )
    store = QdrantChunkStore(
        url=url,
        collection_name=collection_name,
        location=location or None,
        path=path or None,
    )
    store.ensure_collection(named_vectors, payload_indexes=collection_config.get("payload_indexes", []))
    upserted = upsert_package_records(store, package_dir)
    embedder, _ = build_text_embedder(
        backend=text_backend,
        model_name=text_model,
        device=device,
        hashing_dim=hashing_dim,
        vector_name=selected_vectors[0] if selected_vectors else "text_dense",
    )
    if embedder is None:
        raise typer.BadParameter("text backend must not be none for Qdrant hybrid search")
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
    return {
        "searcher": searcher,
        "store": store,
        "collection_name": collection_name,
        "selected_vectors": selected_vectors,
        "query_encoders": query_encoder_names(
            selected_vectors,
            vector_embedders=vector_embedders,
            default_embedder=embedder,
            image_query_backend=image_query_backend,
        ),
        "upserted": upserted,
        "chunks": chunks,
        "assets": assets,
        "triples": triples,
    }


def query_encoder_names(
    selected_vectors: list[str],
    vector_embedders: dict,
    default_embedder,
    image_query_backend: str,
) -> dict[str, str]:
    return {
        name: "default_text" if vector_embedders.get(name, default_embedder) is default_embedder else image_query_backend
        for name in selected_vectors
    }


def normalize_backend(value: str) -> str:
    return value.strip().lower()


def build_ocr_backend(
    backend: str,
    model_lang: str = "korean",
    device: str = "",
    engine: str = "",
    min_confidence: float = 0.0,
    use_gpu: bool = False,
    enable_mkldnn: bool = False,
) -> tuple[OCRBackend | None, str]:
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None, ""
    if normalized == "tesseract":
        from chunking_docs.vision.tesseract_ocr import TesseractOCRBackend

        return TesseractOCRBackend(), "tesseract"
    if normalized in {"paddle", "paddleocr"}:
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise typer.BadParameter("--ocr-min-confidence must be between 0.0 and 1.0")
        from chunking_docs.vision.paddle_ocr import PaddleOCRBackend

        return (
            PaddleOCRBackend(
                lang=model_lang,
                device=device,
                engine=engine,
                min_confidence=min_confidence,
                enable_mkldnn=enable_mkldnn,
                use_gpu=True if use_gpu else None,
            ),
            f"paddleocr:{model_lang}",
        )
    raise typer.BadParameter("ocr must be one of: none, tesseract, paddleocr")


def build_vlm_backend(
    backend: str,
    model_name: str,
    profile: str = "",
    model_class: str = "auto",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    max_new_tokens: int = 768,
    attn_implementation: str = "",
) -> tuple[VLMBackend | None, str]:
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None, ""
    if normalized == "hf":
        profile_name = ""
        if profile:
            from chunking_docs.vision.hf_vlm import get_vlm_model_profile

            try:
                model_profile = get_vlm_model_profile(profile)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            profile_name = model_profile.name
            model_name = model_name or model_profile.model_name
            model_class = model_class if model_class != "auto" else model_profile.model_class
            device_map = device_map if device_map != "auto" else model_profile.device_map
            torch_dtype = torch_dtype if torch_dtype != "auto" else model_profile.torch_dtype
            max_new_tokens = max_new_tokens if max_new_tokens != 768 else model_profile.max_new_tokens
            attn_implementation = attn_implementation or model_profile.attn_implementation
        if not model_name:
            raise typer.BadParameter("--vlm-model is required when --vlm hf")
        if max_new_tokens <= 0:
            raise typer.BadParameter("--vlm-max-new-tokens must be positive")
        from chunking_docs.vision.hf_vlm import HuggingFaceVLMBackend

        return (
            HuggingFaceVLMBackend(
                model_name=model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                max_new_tokens=max_new_tokens,
                attn_implementation=attn_implementation,
                model_class=model_class,
                profile=profile_name,
            ),
            f"hf:{profile_name or model_name}",
        )
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


def parse_visual_run_inputs(values: list[str] | None) -> dict[str, Path]:
    if not values:
        raise typer.BadParameter("At least one --run name=results.jsonl value is required")
    parsed = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter("--run must be in name=results.jsonl form")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter("--run name must not be empty")
        parsed[name] = Path(path)
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

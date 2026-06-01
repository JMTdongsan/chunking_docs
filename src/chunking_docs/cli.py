from __future__ import annotations

import builtins
import json
import shlex
from pathlib import Path
from time import perf_counter

import httpx
import typer
from rich import print

from chunking_docs.analysis.characterize import characterize_package
from chunking_docs.analysis.workflow import build_ingestion_workflow_plan
from chunking_docs.analysis.pdf_profile import profile_pdf, summarize_profiles, write_profile_outputs
from chunking_docs.chunking.multimodal import ChunkStrategy, build_strategy_chunks
from chunking_docs.chunking.page_chunker import page_level_chunks
from chunking_docs.chunking.section_map import load_section_ranges
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig, TokenizerStrategy
from chunking_docs.evaluation.audit import audit_package, audit_public_artifacts
from chunking_docs.evaluation.ablation import (
    QdrantRerankerAblationReport,
    QdrantRerankerAblationRow,
    QdrantVectorAblationReport,
    QdrantVectorAblationRow,
    RetrievalAblationReport,
    build_qdrant_reranker_ablation_report,
    build_qdrant_vector_ablation_report,
    evaluate_retrieval_ablation,
    gate_qdrant_reranker_ablation,
    gate_retrieval_ablation,
    gate_qdrant_vector_ablation,
    parse_ablation_modes,
    parse_qdrant_reranker_ablation_modes,
    parse_qdrant_vector_ablation_modes,
    qdrant_vector_names_for_modes,
)
from chunking_docs.evaluation.casegen import generate_retrieval_case_skeleton
from chunking_docs.evaluation.case_audit import (
    audit_retrieval_cases,
    count_case_groups,
    count_case_group_distinct_targets,
    count_retrieval_case_distinct_excluded_targets,
    count_retrieval_case_distinct_targets,
    count_retrieval_case_excluded_targets,
    count_retrieval_case_max_target_mentions,
    count_retrieval_case_targets,
    count_visual_image_probes,
    count_visual_object_probes,
)
from chunking_docs.evaluation.chunking_gate import (
    chunking_gate_summary_payload,
    gate_chunking_comparison,
    load_chunking_comparison,
)
from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.compare import compare_chunking_reports
from chunking_docs.evaluation.context_quality import (
    gate_rag_context_evaluation,
    load_rag_context_evaluation,
    evaluate_rag_contexts,
    rag_context_gate_summary_payload,
)
from chunking_docs.evaluation.diagnostics import (
    analyze_retrieval_evaluation,
    load_retrieval_evaluation,
)
from chunking_docs.evaluation.delta import compare_processing_packages
from chunking_docs.evaluation.experiment import build_experiment_report
from chunking_docs.evaluation.fusion_sweep import (
    QdrantFusionSweepCandidate,
    QdrantFusionSweepReport,
    build_fusion_weight_grid,
    build_qdrant_fusion_sweep_report,
    fusion_weight_candidate_name,
)
from chunking_docs.evaluation.gate import gate_retrieval_evaluation, gate_summary_payload
from chunking_docs.evaluation.readiness import build_ingestion_readiness_report
from chunking_docs.evaluation.retrieval import (
    RetrievalCase,
    evaluate_retrieval,
    evaluate_search_results,
    load_retrieval_cases,
)
from chunking_docs.evaluation.retrieval_config import (
    QdrantRetrievalConfig,
    apply_qdrant_retrieval_route_preset,
    build_qdrant_retrieval_config_from_fusion_sweep,
    qdrant_retrieval_config_vector_names,
    select_qdrant_retrieval_route,
)
from chunking_docs.evaluation.sweep import (
    ChunkingSweepCandidate,
    ChunkingSweepReport,
    run_chunking_sweep,
)
from chunking_docs.graph.heuristics import section_triples
from chunking_docs.graph.quality import normalize_graph_triples
from chunking_docs.graph.repair import (
    remap_triples_to_available_chunks,
    repair_visual_derived_triples,
)
from chunking_docs.ingest.pdf_loader import load_source_document, render_pages
from chunking_docs.ingest.tables import extract_pdf_tables
from chunking_docs.io import read_jsonl, write_jsonl
from chunking_docs.models import AssetKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.pipeline import (
    build_processing_package,
    clear_embedding_artifacts,
    load_processing_package,
    refresh_package_metadata,
    refresh_package_indexes,
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
from chunking_docs.runtime import RuntimeReport, inspect_runtime
from chunking_docs.storage.records import EmbeddingRecord
from chunking_docs.vision.annotate import (
    annotate_assets,
    merge_asset_annotations_into_chunks,
    repair_visual_text_chunks,
)
from chunking_docs.vision.assets import (
    attach_assets_to_chunks,
    build_page_tile_assets,
    merge_visual_assets,
)
from chunking_docs.vision.compare import VisualRunComparison, compare_visual_runs
from chunking_docs.vision.experiment_gate import gate_vlm_experiment_plan
from chunking_docs.vision.experiments import build_vlm_experiment_plan, parse_profile_list
from chunking_docs.vision.interfaces import OCRBackend, VLMBackend
from chunking_docs.vision.jobs import (
    VisualAnnotationJob,
    VisualJobRunResult,
    completed_annotations,
    merge_visual_job_results,
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
    require_ocr_gpu: bool = typer.Option(
        False,
        "--require-ocr-gpu",
        help="Require PaddlePaddle CUDA support for GPU OCR.",
    ),
    require_vision: bool = False,
    vlm_profile: list[str] = typer.Option(
        None,
        "--vlm-profile",
        help="Check GPU memory against a VLM profile such as qwen2_5_vl_7b.",
    ),
    vlm_memory_margin_ratio: float = typer.Option(
        0.0,
        "--vlm-memory-margin-ratio",
        help="Warn when a VLM profile does not have this extra GPU memory margin.",
    ),
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
        require_ocr_gpu=require_ocr_gpu,
        require_vision=require_vision,
        vlm_profiles=vlm_profile,
        vlm_memory_margin_ratio=vlm_memory_margin_ratio,
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
    deduplicate_tokens: bool = False,
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
            deduplicate_tokens=deduplicate_tokens,
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


@app.command(name="refresh-package-metadata")
def refresh_package_metadata_command(
    package_dir: Path = Path("outputs/package"),
    pdf: Path | None = None,
    render_zoom: float | None = None,
    section_map_count: int | None = None,
    base_chunking_strategy: str = "",
    embedding_mode: str = "auto",
    extract_tables: str = "auto",
    lexical_tokenizer: str = "auto",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
    deduplicate_tokens: bool = False,
):
    """Refresh source checksum and package config metadata for an existing package."""
    tokenizer_config = None
    if lexical_tokenizer != "auto":
        try:
            tokenizer_config = LexicalTokenizerConfig(
                strategy=lexical_tokenizer,
                min_n=ngram_min,
                max_n=ngram_max,
                ngram_cjk_only=ngram_cjk_only,
                deduplicate=deduplicate_tokens,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    manifest = refresh_package_metadata(
        output_dir=package_dir,
        pdf_path=pdf,
        render_zoom=render_zoom,
        dry_run_embeddings=parse_embedding_mode(embedding_mode),
        section_map_count=section_map_count,
        extract_tables=parse_auto_bool(extract_tables, option_name="--extract-tables"),
        tokenizer_config=tokenizer_config,
        base_chunking_strategy=base_chunking_strategy or None,
    )
    print(
        {
            "package_dir": str(package_dir),
            "source_file": manifest.metadata.get("source_file", {}),
            "package_config": manifest.metadata.get("package_config", {}),
            "embedding_mode": manifest.metadata.get("embedding_mode"),
            "profile_page_count": manifest.metadata.get("profile_summary", {}).get("page_count"),
        }
    )


@app.command(name="refresh-package-indexes")
def refresh_package_indexes_command(
    package_dir: Path = Path("outputs/package"),
    lexical_tokenizer: str = "auto",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
    deduplicate_tokens: bool = False,
    rebuild_dry_run_embeddings: bool = False,
    clear_stale_embeddings: bool = True,
):
    """Refresh BM25 tokens and invalidate stale vector artifacts for an existing package."""
    tokenizer_config = None
    if lexical_tokenizer != "auto":
        try:
            tokenizer_config = LexicalTokenizerConfig(
                strategy=lexical_tokenizer,
                min_n=ngram_min,
                max_n=ngram_max,
                ngram_cjk_only=ngram_cjk_only,
                deduplicate=deduplicate_tokens,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    payload = refresh_package_indexes(
        output_dir=package_dir,
        tokenizer_config=tokenizer_config,
        rebuild_dry_run_embeddings=rebuild_dry_run_embeddings,
        clear_stale_embeddings=clear_stale_embeddings,
    )
    print({"package_dir": str(package_dir), **payload})


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
            "mismatched_payload_indexes": report.mismatched_payload_indexes,
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
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
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
    deduplicate_tokens: bool = False,
):
    """Run Qdrant named-vector + BM25 + optional graph hybrid retrieval."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
        deduplicate_tokens=deduplicate_tokens,
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
            "query_encoder_details": prepared.get("query_encoder_details", {}),
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
    max_chars_per_asset_text: int = 1400,
    include_evidence: bool = True,
    neighbor_window: int = 0,
    include_assets: bool = True,
    include_triples: bool = True,
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
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
    deduplicate_tokens: bool = False,
):
    """Build a citation-ready RAG context bundle from Qdrant hybrid search hits."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
        deduplicate_tokens=deduplicate_tokens,
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
        max_chars_per_asset_text=max_chars_per_asset_text,
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
            "query_encoder_details": prepared.get("query_encoder_details", {}),
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


@app.command(name="qdrant-rag-context-config")
def qdrant_rag_context_config_command(
    config: Path,
    query: str,
    output: Path | None = None,
    package_dir: Path | None = typer.Option(
        None,
        "--package-dir",
        help="Override package_dir from the retrieval config.",
    ),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    max_chars_per_chunk: int = 1400,
    max_chars_per_asset_text: int = 1400,
    include_evidence: bool = True,
    neighbor_window: int = 0,
    include_assets: bool = True,
    include_triples: bool = True,
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    reranker_device: str = "cuda",
    doc_id: str = "",
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
):
    """Build a RAG context bundle using an exported Qdrant retrieval config."""
    retrieval_config = read_qdrant_retrieval_config(config)
    tokenizer_options = retrieval_config_tokenizer_options(retrieval_config)
    effective_package_dir = package_dir or Path(
        retrieval_config.package_dir or "outputs/package"
    )
    effective_collection = collection or retrieval_config.collection_name or ""
    prepared_vector_names = qdrant_retrieval_config_vector_names(retrieval_config)
    vector_names = ",".join(prepared_vector_names)
    parsed_reranker = build_retrieval_config_reranker(
        retrieval_config,
        tokenizer_options=tokenizer_options,
        device=reranker_device,
    )
    effective_rerank_top_k = retrieval_config_rerank_top_k(
        retrieval_config,
        parsed_reranker,
    )
    query_backend_options = resolve_qdrant_query_backend_options(
        package_dir=effective_package_dir,
        selected_vectors=prepared_vector_names,
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
    )
    filters = build_payload_filter(filter_specs=payload_filter)
    metadata_filters = build_payload_filter(doc_id=doc_id, filter_specs=payload_filter)

    prepared = prepare_qdrant_hybrid_search(
        package_dir=effective_package_dir,
        url=url,
        collection=effective_collection,
        location=location,
        path=path,
        vector_names=vector_names,
        text_backend=query_backend_options["text_backend"],
        text_model=query_backend_options["text_model"],
        image_query_backend=query_backend_options["image_query_backend"],
        image_query_model=query_backend_options["image_query_model"],
        device=device,
        hashing_dim=hashing_dim,
        lexical_tokenizer=tokenizer_options["strategy"],
        ngram_min=tokenizer_options["min_n"],
        ngram_max=tokenizer_options["max_n"],
        ngram_cjk_only=tokenizer_options["ngram_cjk_only"],
        deduplicate_tokens=tokenizer_options["deduplicate"],
    )
    route_decision = select_qdrant_retrieval_route(retrieval_config, query)
    hits = prepared["searcher"].search(
        query=query,
        vector_names=route_decision.vector_names,
        top_k=retrieval_config.top_k,
        graph_expand=route_decision.graph_expand,
        doc_id=doc_id or None,
        payload_filter=filters,
        collapse_hierarchical=retrieval_config.collapse_hierarchical,
        fusion_weights=route_decision.fusion_weights,
        reranker=parsed_reranker,
        rerank_top_k=effective_rerank_top_k,
    )
    bundle = build_context_bundle(
        query=query,
        hits=hits,
        chunks=prepared["chunks"],
        assets=prepared["assets"],
        triples=prepared["triples"],
        max_chars_per_chunk=max_chars_per_chunk,
        max_chars_per_asset_text=max_chars_per_asset_text,
        include_evidence=include_evidence,
        neighbor_window=neighbor_window,
        include_assets=include_assets,
        include_triples=include_triples,
    )
    bundle.metadata.update(
        {
            "backend": "qdrant_hybrid_config",
            "config": str(config),
            "config_selection": retrieval_config.selection.model_dump(),
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "base_vector_names": retrieval_config.vector_names,
            "graph_expand": retrieval_config.graph_expand,
            "selected_route": route_decision.model_dump(),
            "query_encoders": prepared["query_encoders"],
            "query_encoder_details": prepared.get("query_encoder_details", {}),
            "upserted": prepared["upserted"],
            "stored_count": prepared["store"].count(),
            "filters": metadata_filters,
            "fusion_weights": retrieval_config.fusion_weights,
            "selected_fusion_weights": route_decision.fusion_weights,
            "collapse_hierarchical": retrieval_config.collapse_hierarchical,
            "reranker": parsed_reranker.source if parsed_reranker else None,
            "rerank_top_k": effective_rerank_top_k if parsed_reranker else None,
            "lexical_tokenizer": tokenizer_options,
            "routing_enabled": bool(retrieval_config.routes),
            "routes": [route.model_dump() for route in retrieval_config.routes],
        }
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        print({"output": str(output), **bundle.metadata})
        return
    print(bundle.model_dump())


@app.command(name="eval-qdrant-rag-context-config")
def eval_qdrant_rag_context_config_command(
    config: Path,
    cases: Path,
    output: Path | None = None,
    contexts_output: Path | None = typer.Option(
        None,
        "--contexts-output",
        help="Optional JSONL output containing the generated context bundle for each case.",
    ),
    package_dir: Path | None = typer.Option(
        None,
        "--package-dir",
        help="Override package_dir from the retrieval config.",
    ),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    max_chars_per_chunk: int = 1400,
    max_chars_per_asset_text: int = 1400,
    include_evidence: bool = True,
    neighbor_window: int = 0,
    include_assets: bool = True,
    include_triples: bool = True,
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    reranker_device: str = "cuda",
    doc_id: str = "",
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
):
    """Evaluate final RAG context bundles built from an exported Qdrant config."""
    retrieval_config = read_qdrant_retrieval_config(config)
    tokenizer_options = retrieval_config_tokenizer_options(retrieval_config)
    effective_package_dir = package_dir or Path(
        retrieval_config.package_dir or "outputs/package"
    )
    effective_collection = collection or retrieval_config.collection_name or ""
    prepared_vector_names = qdrant_retrieval_config_vector_names(retrieval_config)
    vector_names = ",".join(prepared_vector_names)
    parsed_reranker = build_retrieval_config_reranker(
        retrieval_config,
        tokenizer_options=tokenizer_options,
        device=reranker_device,
    )
    effective_rerank_top_k = retrieval_config_rerank_top_k(
        retrieval_config,
        parsed_reranker,
    )
    query_backend_options = resolve_qdrant_query_backend_options(
        package_dir=effective_package_dir,
        selected_vectors=prepared_vector_names,
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
    )
    filters = build_payload_filter(filter_specs=payload_filter)
    metadata_filters = build_payload_filter(doc_id=doc_id, filter_specs=payload_filter)

    prepare_start = perf_counter()
    prepared = prepare_qdrant_hybrid_search(
        package_dir=effective_package_dir,
        url=url,
        collection=effective_collection,
        location=location,
        path=path,
        vector_names=vector_names,
        text_backend=query_backend_options["text_backend"],
        text_model=query_backend_options["text_model"],
        image_query_backend=query_backend_options["image_query_backend"],
        image_query_model=query_backend_options["image_query_model"],
        device=device,
        hashing_dim=hashing_dim,
        lexical_tokenizer=tokenizer_options["strategy"],
        ngram_min=tokenizer_options["min_n"],
        ngram_max=tokenizer_options["max_n"],
        ngram_cjk_only=tokenizer_options["ngram_cjk_only"],
        deduplicate_tokens=tokenizer_options["deduplicate"],
    )
    index_build_ms = (perf_counter() - prepare_start) * 1000
    loaded_cases = load_retrieval_cases(cases)
    route_decisions = [
        select_qdrant_retrieval_route(
            retrieval_config,
            case.query,
            case_metadata=case.metadata,
            graph_expand=retrieval_config.graph_expand,
        )
        for case in loaded_cases
    ]
    routed_cases = retrieval_cases_with_route_metadata(loaded_cases, route_decisions)
    bundles = []
    latencies_ms = []
    for case, route_decision in zip(routed_cases, route_decisions):
        case_start = perf_counter()
        hits = prepared["searcher"].search(
            query=case.query,
            vector_names=route_decision.vector_names,
            top_k=retrieval_config.top_k,
            graph_expand=route_decision.graph_expand,
            doc_id=doc_id or None,
            payload_filter=filters,
            collapse_hierarchical=retrieval_config.collapse_hierarchical,
            fusion_weights=route_decision.fusion_weights,
            reranker=parsed_reranker,
            rerank_top_k=effective_rerank_top_k,
        )
        bundle = build_context_bundle(
            query=case.query,
            hits=hits,
            chunks=prepared["chunks"],
            assets=prepared["assets"],
            triples=prepared["triples"],
            max_chars_per_chunk=max_chars_per_chunk,
            max_chars_per_asset_text=max_chars_per_asset_text,
            include_evidence=include_evidence,
            neighbor_window=neighbor_window,
            include_assets=include_assets,
            include_triples=include_triples,
        )
        latencies_ms.append((perf_counter() - case_start) * 1000)
        bundle.metadata.update(
            {
                "backend": "qdrant_hybrid_config",
                "config": str(config),
                "config_selection": retrieval_config.selection.model_dump(),
                "collection": prepared["collection_name"],
                "vector_names": prepared["selected_vectors"],
                "base_vector_names": retrieval_config.vector_names,
                "graph_expand": retrieval_config.graph_expand,
                "selected_route": route_decision.model_dump(),
                "query_encoders": prepared["query_encoders"],
                "query_encoder_details": prepared.get("query_encoder_details", {}),
                "filters": metadata_filters,
                "fusion_weights": retrieval_config.fusion_weights,
                "selected_fusion_weights": route_decision.fusion_weights,
                "collapse_hierarchical": retrieval_config.collapse_hierarchical,
                "reranker": parsed_reranker.source if parsed_reranker else None,
                "rerank_top_k": effective_rerank_top_k if parsed_reranker else None,
                "lexical_tokenizer": tokenizer_options,
                "case_metadata": case.metadata,
                "routing_enabled": bool(retrieval_config.routes),
                "routes": [route.model_dump() for route in retrieval_config.routes],
            }
        )
        bundles.append(bundle)
    evaluation = evaluate_rag_contexts(
        cases=routed_cases,
        bundles=bundles,
        latencies_ms=latencies_ms,
    )
    evaluation.metadata.update(
        {
            "backend": "qdrant_rag_context_config",
            "config": str(config),
            "config_selection": retrieval_config.selection.model_dump(),
            "cases": str(cases),
            "contexts_output": str(contexts_output) if contexts_output is not None else None,
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "base_vector_names": retrieval_config.vector_names,
            "graph_expand": retrieval_config.graph_expand,
            "query_encoders": prepared["query_encoders"],
            "query_encoder_details": prepared.get("query_encoder_details", {}),
            "upserted": prepared["upserted"],
            "stored_count": prepared["store"].count(),
            "filters": metadata_filters,
            "fusion_weights": retrieval_config.fusion_weights,
            "collapse_hierarchical": retrieval_config.collapse_hierarchical,
            "reranker": parsed_reranker.source if parsed_reranker else None,
            "rerank_top_k": effective_rerank_top_k if parsed_reranker else None,
            "lexical_tokenizer": tokenizer_options,
            "index_build_ms": index_build_ms,
            "routing_enabled": bool(retrieval_config.routes),
            "routes": [route.model_dump() for route in retrieval_config.routes],
            "route_decisions": [decision.model_dump() for decision in route_decisions],
            "route_usage": qdrant_route_usage(route_decisions),
            "context_options": {
                "max_chars_per_chunk": max_chars_per_chunk,
                "max_chars_per_asset_text": max_chars_per_asset_text,
                "include_evidence": include_evidence,
                "neighbor_window": neighbor_window,
                "include_assets": include_assets,
                "include_triples": include_triples,
            },
        }
    )
    if contexts_output is not None:
        write_jsonl(contexts_output, bundles)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    print(
        {
            "output": str(output) if output is not None else None,
            "contexts_output": str(contexts_output) if contexts_output is not None else None,
            "config": str(config),
            "case_count": evaluation.case_count,
            "passed_count": evaluation.passed_count,
            "failed_count": evaluation.failed_count,
            "target_coverage": evaluation.target_coverage,
            "excluded_target_hit_rate": evaluation.excluded_target_hit_rate,
            "mean_context_char_count": evaluation.mean_context_char_count,
            "max_context_char_count": evaluation.max_context_char_count,
            "mean_latency_ms": evaluation.mean_latency_ms,
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "fusion_weights": retrieval_config.fusion_weights,
            "route_usage": qdrant_route_usage(route_decisions),
            "config_selection": retrieval_config.selection.model_dump(),
        }
    )


@app.command(name="gate-rag-context")
def gate_rag_context_command(
    evaluation: Path,
    output: Path | None = None,
    min_case_count: int = typer.Option(
        0,
        "--min-case-count",
        help="Require at least this many context benchmark cases.",
    ),
    min_expected_case_count: int = typer.Option(
        0,
        "--min-expected-case-count",
        help="Require at least this many cases with expected targets.",
    ),
    min_expected_target_count: int = typer.Option(
        0,
        "--min-expected-target-count",
        help="Require at least this many expected page/chunk/asset/triple targets.",
    ),
    min_passed_case_count: int = typer.Option(
        0,
        "--min-passed-case-count",
        help="Require at least this many context cases to pass.",
    ),
    max_failed_cases: int | None = typer.Option(
        None,
        "--max-failed-cases",
        help="Limit context benchmark cases that miss expected evidence or include hard negatives.",
    ),
    min_hit_rate: float = 0.0,
    min_target_coverage: float = 0.0,
    max_excluded_target_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-target-hit-rate",
        help="Limit the fraction of explicit excluded targets present in final context.",
    ),
    max_excluded_query_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-query-hit-rate",
        help="Limit hard-negative cases whose final context includes any excluded target.",
    ),
    max_excluded_hit_query_count: int | None = typer.Option(
        None,
        "--max-excluded-hit-query-count",
        help="Limit hard-negative cases whose final context includes any excluded target.",
    ),
    max_mean_latency_ms: float | None = None,
    max_mean_context_char_count: float | None = None,
    max_context_char_count: int | None = None,
    max_mean_chunk_count: float | None = None,
    max_mean_asset_count: float | None = None,
    max_mean_triple_count: float | None = None,
    min_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-target-type-coverage",
        help="Require final-context target coverage such as asset=1.0 or triple=1.0.",
    ),
    min_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-case-group-target-coverage",
        help="Require case metadata group coverage such as case_source:visual_object_probe=0.8.",
    ),
    max_case_group_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-case-group-excluded-target-hit-rate",
        help="Limit case metadata group hard-negative rate such as case_source:visual_image_probe=0.0.",
    ),
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when any context gate check fails.",
    ),
):
    """Fail a RAG context evaluation when final context quality gates are missed."""
    parsed_evaluation = load_rag_context_evaluation(evaluation)
    target_type_thresholds = parse_named_float_thresholds(
        min_target_type_coverage,
        "target type coverage",
    )
    case_group_thresholds = parse_named_float_thresholds(
        min_case_group_target_coverage,
        "case group target coverage",
    )
    case_group_excluded_thresholds = parse_named_float_thresholds(
        max_case_group_excluded_target_hit_rate,
        "case group excluded-target hit rate",
    )
    report = gate_rag_context_evaluation(
        parsed_evaluation,
        min_case_count=min_case_count,
        min_expected_case_count=min_expected_case_count,
        min_expected_target_count=min_expected_target_count,
        min_passed_case_count=min_passed_case_count,
        max_failed_case_count=max_failed_cases,
        min_hit_rate=min_hit_rate,
        min_target_coverage=min_target_coverage,
        max_excluded_target_hit_rate=max_excluded_target_hit_rate,
        max_excluded_query_hit_rate=max_excluded_query_hit_rate,
        max_excluded_hit_query_count=max_excluded_hit_query_count,
        max_mean_latency_ms=max_mean_latency_ms,
        max_mean_context_char_count=max_mean_context_char_count,
        max_context_char_count=max_context_char_count,
        max_mean_chunk_count=max_mean_chunk_count,
        max_mean_asset_count=max_mean_asset_count,
        max_mean_triple_count=max_mean_triple_count,
        min_target_type_coverage=target_type_thresholds,
        min_case_group_target_coverage=case_group_thresholds,
        max_case_group_excluded_target_hit_rate=case_group_excluded_thresholds,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {"output": str(output), **rag_context_gate_summary_payload(report)}
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


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
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
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
    deduplicate_tokens: bool = False,
):
    """Evaluate Qdrant hybrid retrieval against JSONL benchmark cases."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
        deduplicate_tokens=deduplicate_tokens,
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
            "query_encoder_details": prepared.get("query_encoder_details", {}),
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
                "excluded_query_count": evaluation.excluded_query_count,
                "excluded_hit_query_count": evaluation.excluded_hit_query_count,
                "excluded_query_hit_rate": evaluation.excluded_query_hit_rate,
                "excluded_target_hit_rate": evaluation.excluded_target_hit_rate,
                "mean_latency_ms": evaluation.mean_latency_ms,
                "p95_latency_ms": evaluation.p95_latency_ms,
                "unstable_result_count": evaluation.unstable_result_count,
                "result_stability_rate": evaluation.result_stability_rate,
                "target_metrics": retrieval_target_metrics_payload(evaluation),
                "source_metrics": retrieval_source_metrics_payload(evaluation),
                "source_family_metrics": retrieval_source_family_metrics_payload(evaluation),
                "chunk_strategy_metrics": retrieval_chunk_strategy_metrics_payload(evaluation),
                "retrieval_role_metrics": retrieval_role_metrics_payload(evaluation),
                "case_group_metrics": retrieval_case_group_metrics_payload(evaluation),
                **evaluation.metadata,
            }
        )
        return
    print(evaluation.model_dump())


@app.command(name="eval-qdrant-retrieval-config")
def eval_qdrant_retrieval_config_command(
    config: Path,
    cases: Path,
    output: Path | None = None,
    package_dir: Path | None = typer.Option(
        None,
        "--package-dir",
        help="Override package_dir from the retrieval config.",
    ),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    repeat: int = typer.Option(
        0,
        "--repeat",
        help="Repeat count for latency/stability sampling. Defaults to the config metadata value.",
    ),
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    reranker_device: str = "cuda",
    doc_id: str = "",
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
):
    """Evaluate benchmark cases using an exported Qdrant retrieval config."""
    retrieval_config = read_qdrant_retrieval_config(config)
    tokenizer_options = retrieval_config_tokenizer_options(retrieval_config)
    effective_package_dir = package_dir or Path(
        retrieval_config.package_dir or "outputs/package"
    )
    effective_repeat = repeat or int(retrieval_config.metadata.get("repeat") or 1)
    effective_collection = collection or retrieval_config.collection_name or ""
    prepared_vector_names = qdrant_retrieval_config_vector_names(retrieval_config)
    vector_names = ",".join(prepared_vector_names)
    parsed_reranker = build_retrieval_config_reranker(
        retrieval_config,
        tokenizer_options=tokenizer_options,
        device=reranker_device,
    )
    effective_rerank_top_k = retrieval_config_rerank_top_k(
        retrieval_config,
        parsed_reranker,
    )
    query_backend_options = resolve_qdrant_query_backend_options(
        package_dir=effective_package_dir,
        selected_vectors=prepared_vector_names,
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
    )
    filters = build_payload_filter(filter_specs=payload_filter)
    metadata_filters = build_payload_filter(doc_id=doc_id, filter_specs=payload_filter)

    prepare_start = perf_counter()
    prepared = prepare_qdrant_hybrid_search(
        package_dir=effective_package_dir,
        url=url,
        collection=effective_collection,
        location=location,
        path=path,
        vector_names=vector_names,
        text_backend=query_backend_options["text_backend"],
        text_model=query_backend_options["text_model"],
        image_query_backend=query_backend_options["image_query_backend"],
        image_query_model=query_backend_options["image_query_model"],
        device=device,
        hashing_dim=hashing_dim,
        lexical_tokenizer=tokenizer_options["strategy"],
        ngram_min=tokenizer_options["min_n"],
        ngram_max=tokenizer_options["max_n"],
        ngram_cjk_only=tokenizer_options["ngram_cjk_only"],
        deduplicate_tokens=tokenizer_options["deduplicate"],
    )
    index_build_ms = (perf_counter() - prepare_start) * 1000
    loaded_cases = load_retrieval_cases(cases)
    route_decisions = [
        select_qdrant_retrieval_route(
            retrieval_config,
            case.query,
            case_metadata=case.metadata,
            graph_expand=retrieval_config.graph_expand,
        )
        for case in loaded_cases
    ]
    routed_cases = retrieval_cases_with_route_metadata(loaded_cases, route_decisions)

    def search_with_config(case, graph_expand):
        decision = select_qdrant_retrieval_route(
            retrieval_config,
            case.query,
            case_metadata=case.metadata,
            graph_expand=graph_expand,
        )
        return prepared["searcher"].search(
            query=case.query,
            vector_names=decision.vector_names,
            top_k=retrieval_config.top_k,
            graph_expand=decision.graph_expand,
            doc_id=doc_id or None,
            payload_filter=filters,
            collapse_hierarchical=retrieval_config.collapse_hierarchical,
            fusion_weights=decision.fusion_weights,
            reranker=parsed_reranker,
            rerank_top_k=effective_rerank_top_k,
        )

    evaluation = evaluate_search_results(
        cases=routed_cases,
        search_fn=search_with_config,
        top_k=retrieval_config.top_k,
        repeat=effective_repeat,
        index_build_ms=index_build_ms,
        graph_expand_override=retrieval_config.graph_expand,
        triples=prepared["triples"],
    )
    evaluation.metadata.update(
        {
            "backend": "qdrant_hybrid_config",
            "config": str(config),
            "config_selection": retrieval_config.selection.model_dump(),
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "base_vector_names": retrieval_config.vector_names,
            "graph_expand": retrieval_config.graph_expand,
            "query_encoders": prepared["query_encoders"],
            "query_encoder_details": prepared.get("query_encoder_details", {}),
            "upserted": prepared["upserted"],
            "stored_count": prepared["store"].count(),
            "filters": metadata_filters,
            "fusion_weights": retrieval_config.fusion_weights,
            "collapse_hierarchical": retrieval_config.collapse_hierarchical,
            "reranker": parsed_reranker.source if parsed_reranker else None,
            "rerank_top_k": effective_rerank_top_k if parsed_reranker else None,
            "lexical_tokenizer": tokenizer_options,
            "routing_enabled": bool(retrieval_config.routes),
            "routes": [route.model_dump() for route in retrieval_config.routes],
            "route_decisions": [decision.model_dump() for decision in route_decisions],
            "route_usage": qdrant_route_usage(route_decisions),
        }
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    print(
        {
            "output": str(output) if output is not None else None,
            "config": str(config),
            "case_count": evaluation.case_count,
            "recall_at_k": evaluation.recall_at_k,
            "mrr": evaluation.mrr,
            "target_coverage_at_k": evaluation.target_coverage_at_k,
            "mean_target_ndcg_at_k": evaluation.mean_target_ndcg_at_k,
            "mean_precision_at_k": evaluation.mean_precision_at_k,
            "excluded_query_count": evaluation.excluded_query_count,
            "excluded_hit_query_count": evaluation.excluded_hit_query_count,
            "excluded_query_hit_rate": evaluation.excluded_query_hit_rate,
            "excluded_target_hit_rate": evaluation.excluded_target_hit_rate,
            "mean_latency_ms": evaluation.mean_latency_ms,
            "p95_latency_ms": evaluation.p95_latency_ms,
            "unstable_result_count": evaluation.unstable_result_count,
            "result_stability_rate": evaluation.result_stability_rate,
            "collection": prepared["collection_name"],
            "vector_names": prepared["selected_vectors"],
            "fusion_weights": retrieval_config.fusion_weights,
            "route_usage": qdrant_route_usage(route_decisions),
            "config_selection": retrieval_config.selection.model_dump(),
        }
    )


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
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
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
    deduplicate_tokens: bool = False,
):
    """Compare Qdrant text, visual caption, object, image, and graph retrieval signals."""
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
        deduplicate_tokens=deduplicate_tokens,
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
                "query_encoder_details": {
                    name: prepared.get("query_encoder_details", {}).get(name, {})
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
                "case_group_best_modes": report.case_group_best_modes,
                "pairwise": [comparison.model_dump() for comparison in report.pairwise],
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
                        "excluded_query_count": row.evaluation.excluded_query_count,
                        "excluded_hit_query_count": row.evaluation.excluded_hit_query_count,
                        "excluded_query_hit_rate": row.evaluation.excluded_query_hit_rate,
                        "excluded_target_count": row.evaluation.excluded_target_count,
                        "excluded_matched_target_count": (
                            row.evaluation.excluded_matched_target_count
                        ),
                        "excluded_target_hit_rate": row.evaluation.excluded_target_hit_rate,
                        "repeat": row.evaluation.repeat,
                        "mean_latency_ms": row.evaluation.mean_latency_ms,
                        "p95_latency_ms": row.evaluation.p95_latency_ms,
                        "unstable_result_count": row.evaluation.unstable_result_count,
                        "result_stability_rate": row.evaluation.result_stability_rate,
                        "target_metrics": retrieval_target_metrics_payload(row.evaluation),
                        "source_metrics": retrieval_source_metrics_payload(row.evaluation),
                        "source_family_metrics": retrieval_source_family_metrics_payload(
                            row.evaluation
                        ),
                        "chunk_strategy_metrics": retrieval_chunk_strategy_metrics_payload(
                            row.evaluation
                        ),
                        "retrieval_role_metrics": retrieval_role_metrics_payload(row.evaluation),
                        "case_group_metrics": retrieval_case_group_metrics_payload(row.evaluation),
                        "failed_queries": row.evaluation.failed_queries,
                    }
                    for row in report.rows
                ],
            }
        )
        return
    print(report.model_dump())


@app.command(name="eval-qdrant-reranker-ablation")
def eval_qdrant_reranker_ablation_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    modes: str = "none,lexical",
    vector_names: str = "text_dense,caption_dense",
    graph_expand: bool = typer.Option(False, "--graph-expand/--no-graph-expand"),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    top_k: int = 5,
    repeat: int = 1,
    collapse_hierarchical: bool = False,
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
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
    deduplicate_tokens: bool = False,
):
    """Compare Qdrant hybrid retrieval before and after reranking."""
    try:
        parsed_modes = parse_qdrant_reranker_ablation_modes(modes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    parsed_vector_names = [item.strip() for item in vector_names.split(",") if item.strip()]
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
        deduplicate_tokens=deduplicate_tokens,
    )
    index_build_ms = (perf_counter() - prepare_start) * 1000
    retrieval_cases = load_retrieval_cases(cases)
    filters = build_payload_filter(filter_specs=payload_filter)
    metadata_filters = build_payload_filter(doc_id=doc_id, filter_specs=payload_filter)

    rows = []
    for mode in parsed_modes:
        parsed_reranker = build_reranker(
            mode.reranker,
            model_name=reranker_model,
            device=reranker_device,
            max_length=reranker_max_length,
            tokenizer_config=tokenizer_config,
        )
        effective_rerank_top_k = (mode.rerank_top_k or rerank_top_k) if parsed_reranker else None
        effective_mode = mode.model_copy(
            update={"rerank_top_k": effective_rerank_top_k or 0}
        )

        def search_for_case(case, case_graph_expand):
            return prepared["searcher"].search(
                query=case.query,
                vector_names=parsed_vector_names,
                top_k=top_k,
                graph_expand=case_graph_expand,
                doc_id=doc_id or None,
                payload_filter=filters,
                collapse_hierarchical=collapse_hierarchical,
                fusion_weights=fusion_weights,
                reranker=parsed_reranker,
                rerank_top_k=effective_rerank_top_k,
            )

        evaluation = evaluate_search_results(
            cases=retrieval_cases,
            search_fn=search_for_case,
            top_k=top_k,
            repeat=repeat,
            index_build_ms=index_build_ms,
            graph_expand_override=graph_expand,
            triples=prepared["triples"],
        )
        evaluation.metadata.update(
            {
                "backend": "qdrant_hybrid",
                "mode": mode.name,
                "collection": prepared["collection_name"],
                "vector_names": parsed_vector_names,
                "graph_expand": graph_expand,
                "query_encoders": {
                    name: prepared["query_encoders"].get(name, "unknown")
                    for name in parsed_vector_names
                },
                "query_encoder_details": {
                    name: prepared.get("query_encoder_details", {}).get(name, {})
                    for name in parsed_vector_names
                },
                "upserted": prepared["upserted"],
                "stored_count": prepared["store"].count(),
                "filters": metadata_filters,
                "fusion_weights": fusion_weights,
                "collapse_hierarchical": collapse_hierarchical,
                "reranker": parsed_reranker.source if parsed_reranker else None,
                "reranker_model": reranker_model if parsed_reranker else "",
                "reranker_max_length": reranker_max_length if parsed_reranker else 0,
                "rerank_top_k": effective_rerank_top_k or 0,
                "lexical_tokenizer": {
                    "strategy": lexical_tokenizer,
                    "min_n": ngram_min,
                    "max_n": ngram_max,
                    "ngram_cjk_only": ngram_cjk_only,
                    "deduplicate": deduplicate_tokens,
                },
            }
        )
        rows.append(QdrantRerankerAblationRow(mode=effective_mode, evaluation=evaluation))

    report = build_qdrant_reranker_ablation_report(rows)
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
                "case_group_best_modes": report.case_group_best_modes,
                "pairwise": [comparison.model_dump() for comparison in report.pairwise],
                "rows": [
                    {
                        "mode": row.mode.name,
                        "reranker": row.evaluation.metadata.get("reranker"),
                        "rerank_top_k": row.evaluation.metadata.get("rerank_top_k"),
                        "recall_at_k": row.evaluation.recall_at_k,
                        "mrr": row.evaluation.mrr,
                        "hit_rate": row.evaluation.hit_rate,
                        "target_coverage_at_k": row.evaluation.target_coverage_at_k,
                        "mean_target_ndcg_at_k": row.evaluation.mean_target_ndcg_at_k,
                        "mean_precision_at_k": row.evaluation.mean_precision_at_k,
                        "excluded_query_count": row.evaluation.excluded_query_count,
                        "excluded_hit_query_count": row.evaluation.excluded_hit_query_count,
                        "excluded_query_hit_rate": row.evaluation.excluded_query_hit_rate,
                        "excluded_target_count": row.evaluation.excluded_target_count,
                        "excluded_matched_target_count": (
                            row.evaluation.excluded_matched_target_count
                        ),
                        "excluded_target_hit_rate": row.evaluation.excluded_target_hit_rate,
                        "repeat": row.evaluation.repeat,
                        "mean_latency_ms": row.evaluation.mean_latency_ms,
                        "p95_latency_ms": row.evaluation.p95_latency_ms,
                        "unstable_result_count": row.evaluation.unstable_result_count,
                        "result_stability_rate": row.evaluation.result_stability_rate,
                        "target_metrics": retrieval_target_metrics_payload(row.evaluation),
                        "source_metrics": retrieval_source_metrics_payload(row.evaluation),
                        "source_family_metrics": retrieval_source_family_metrics_payload(
                            row.evaluation
                        ),
                        "chunk_strategy_metrics": retrieval_chunk_strategy_metrics_payload(
                            row.evaluation
                        ),
                        "retrieval_role_metrics": retrieval_role_metrics_payload(row.evaluation),
                        "case_group_metrics": retrieval_case_group_metrics_payload(row.evaluation),
                        "failed_queries": row.evaluation.failed_queries,
                    }
                    for row in report.rows
                ],
            }
        )
        return
    print(report.model_dump())


@app.command(name="sweep-qdrant-fusion")
def sweep_qdrant_fusion_command(
    cases: Path,
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    vector_names: str = "text_dense,caption_dense",
    graph_expand: bool = typer.Option(False, "--graph-expand/--no-graph-expand"),
    url: str = "http://localhost:6333",
    collection: str = "",
    location: str = "",
    path: str = "",
    top_k: int = 5,
    repeat: int = 1,
    collapse_hierarchical: bool = False,
    text_backend: str = "auto",
    text_model: str = "BAAI/bge-m3",
    image_query_backend: str = "auto",
    image_query_model: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    hashing_dim: int = 384,
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda",
    reranker_max_length: int = 0,
    rerank_top_k: int = 20,
    doc_id: str = "",
    payload_filter: list[str] = typer.Option(
        None,
        "--filter",
        help="Payload filter such as kind=map, page_no=12, page_start<=12. Repeat for multiple filters.",
    ),
    weight_grid: list[str] = typer.Option(
        None,
        "--weight-grid",
        help=(
            "Fusion weight grid as source=v1,v2. Sources include bm25, graph, qdrant, "
            "or exact sources such as qdrant:caption_dense. Repeat for multiple sources."
        ),
    ),
    fixed_fusion_weight: list[str] = typer.Option(
        None,
        "--fixed-fusion-weight",
        help="Fusion weight fixed for every candidate, as source=value. Repeat for multiple sources.",
    ),
    include_fixed_candidate: bool = True,
    max_candidates: int = 200,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = typer.Option(
        None,
        "--max-p95-latency-ms",
        help="Reject fusion candidates whose p95 query latency is above this value.",
    ),
    max_excluded_target_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-target-hit-rate",
        help="Reject fusion candidates whose explicit excluded target hit rate is above this value.",
    ),
    max_excluded_query_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-query-hit-rate",
        help="Reject fusion candidates whose hard-negative query hit rate is above this value.",
    ),
    max_excluded_hit_query_count: int | None = typer.Option(
        None,
        "--max-excluded-hit-query-count",
        help="Reject fusion candidates with more hard-negative hit queries than this count.",
    ),
    max_source_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-excluded-target-hit-rate",
        help=(
            "Reject fusion candidates whose exact-source excluded-target hit rate exceeds "
            "name=value, such as qdrant:image_dense=0.0."
        ),
    ),
    max_source_family_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-family-excluded-target-hit-rate",
        help=(
            "Reject fusion candidates whose source-family excluded-target hit rate exceeds "
            "name=value, such as visual=0.0."
        ),
    ),
    max_chunk_strategy_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-chunk-strategy-excluded-target-hit-rate",
        help=(
            "Reject fusion candidates whose chunking-strategy excluded-target hit rate "
            "exceeds name=value, such as visual_asset_text=0.0."
        ),
    ),
    max_retrieval_role_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-retrieval-role-excluded-target-hit-rate",
        help=(
            "Reject fusion candidates whose retrieval-role excluded-target hit rate exceeds "
            "name=value, such as child=0.0."
        ),
    ),
    recall_weight: float = 1.0,
    target_coverage_weight: float = 2.0,
    target_ndcg_weight: float = 1.0,
    mrr_weight: float = 1.0,
    precision_weight: float = 0.5,
    failed_query_penalty: float = 0.02,
    excluded_query_hit_penalty: float = typer.Option(
        1.0,
        "--excluded-query-hit-penalty",
        help="Selection-score penalty applied to hard-negative query hit rate.",
    ),
    excluded_target_hit_penalty: float = typer.Option(
        1.0,
        "--excluded-target-hit-penalty",
        help="Selection-score penalty applied to explicit excluded target hit rate.",
    ),
    source_excluded_target_hit_penalty: float = typer.Option(
        0.0,
        "--source-excluded-target-hit-penalty",
        help="Selection-score penalty applied to the worst exact-source excluded-target hit rate.",
    ),
    source_family_excluded_target_hit_penalty: float = typer.Option(
        0.0,
        "--source-family-excluded-target-hit-penalty",
        help="Selection-score penalty applied to the worst source-family excluded-target hit rate.",
    ),
    chunk_strategy_excluded_target_hit_penalty: float = typer.Option(
        0.0,
        "--chunk-strategy-excluded-target-hit-penalty",
        help=(
            "Selection-score penalty applied to the worst chunking-strategy "
            "excluded-target hit rate."
        ),
    ),
    retrieval_role_excluded_target_hit_penalty: float = typer.Option(
        0.0,
        "--retrieval-role-excluded-target-hit-penalty",
        help="Selection-score penalty applied to the worst retrieval-role excluded-target hit rate.",
    ),
    latency_weight: float = 0.05,
    p95_latency_weight: float = typer.Option(
        0.0,
        "--p95-latency-weight",
        help="Selection-score penalty applied to p95 query latency in seconds.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
    deduplicate_tokens: bool = False,
    summary_limit: int = 10,
    pairwise_top_k: int = typer.Option(
        10,
        "--pairwise-top-k",
        help="Compute query-paired candidate comparisons among the top N ranked fusion candidates.",
    ),
):
    """Sweep Qdrant/BM25/graph fusion weights and recommend a retrieval configuration."""
    grid = parse_fusion_weight_grid(weight_grid)
    fixed_weights = parse_fusion_weights(fixed_fusion_weight)
    source_excluded_thresholds = parse_named_float_thresholds(
        max_source_excluded_target_hit_rate,
        "--max-source-excluded-target-hit-rate",
    )
    source_family_excluded_thresholds = parse_named_float_thresholds(
        max_source_family_excluded_target_hit_rate,
        "--max-source-family-excluded-target-hit-rate",
    )
    chunk_strategy_excluded_thresholds = parse_named_float_thresholds(
        max_chunk_strategy_excluded_target_hit_rate,
        "--max-chunk-strategy-excluded-target-hit-rate",
    )
    retrieval_role_excluded_thresholds = parse_named_float_thresholds(
        max_retrieval_role_excluded_target_hit_rate,
        "--max-retrieval-role-excluded-target-hit-rate",
    )
    try:
        candidate_weights = build_fusion_weight_grid(
            grid,
            fixed_weights=fixed_weights,
            include_fixed_candidate=include_fixed_candidate,
            max_candidates=max_candidates,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    parsed_vector_names = [item.strip() for item in vector_names.split(",") if item.strip()]
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
    )
    parsed_reranker = build_reranker(
        reranker,
        model_name=reranker_model,
        device=reranker_device,
        max_length=reranker_max_length,
        tokenizer_config=tokenizer_config,
    )
    effective_rerank_top_k = rerank_top_k if parsed_reranker else None
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
        deduplicate_tokens=deduplicate_tokens,
    )
    index_build_ms = (perf_counter() - prepare_start) * 1000
    retrieval_cases = load_retrieval_cases(cases)
    filters = build_payload_filter(filter_specs=payload_filter)
    metadata_filters = build_payload_filter(doc_id=doc_id, filter_specs=payload_filter)

    candidates = []
    for weights in candidate_weights:
        evaluation = evaluate_search_results(
            cases=retrieval_cases,
            search_fn=lambda case, case_graph_expand, weights=weights: prepared[
                "searcher"
            ].search(
                query=case.query,
                vector_names=parsed_vector_names,
                top_k=top_k,
                graph_expand=case_graph_expand,
                doc_id=doc_id or None,
                payload_filter=filters,
                collapse_hierarchical=collapse_hierarchical,
                fusion_weights=weights,
                reranker=parsed_reranker,
                rerank_top_k=effective_rerank_top_k,
            ),
            top_k=top_k,
            repeat=repeat,
            index_build_ms=index_build_ms,
            graph_expand_override=graph_expand,
            triples=prepared["triples"],
        )
        evaluation.metadata.update(
            {
                "backend": "qdrant_hybrid",
                "collection": prepared["collection_name"],
                "vector_names": parsed_vector_names,
                "graph_expand": graph_expand,
                "query_encoders": {
                    name: prepared["query_encoders"].get(name, "unknown")
                    for name in parsed_vector_names
                },
                "query_encoder_details": {
                    name: prepared.get("query_encoder_details", {}).get(name, {})
                    for name in parsed_vector_names
                },
                "upserted": prepared["upserted"],
                "stored_count": prepared["store"].count(),
                "filters": metadata_filters,
                "fusion_weights": weights,
                "reranker": parsed_reranker.source if parsed_reranker else None,
                "rerank_top_k": effective_rerank_top_k if parsed_reranker else None,
            }
        )
        candidates.append(
            QdrantFusionSweepCandidate(
                name=fusion_weight_candidate_name(weights),
                fusion_weights=weights,
                evaluation=evaluation,
            )
        )

    report = build_qdrant_fusion_sweep_report(
        candidates,
        vector_names=parsed_vector_names,
        graph_expand=graph_expand,
        min_recall_at_k=min_recall_at_k,
        min_target_coverage_at_k=min_target_coverage_at_k,
        min_target_ndcg_at_k=min_target_ndcg_at_k,
        min_mrr=min_mrr,
        max_failed_queries=max_failed_queries,
        max_mean_latency_ms=max_mean_latency_ms,
        max_p95_latency_ms=max_p95_latency_ms,
        max_excluded_target_hit_rate=max_excluded_target_hit_rate,
        max_excluded_query_hit_rate=max_excluded_query_hit_rate,
        max_excluded_hit_query_count=max_excluded_hit_query_count,
        max_source_excluded_target_hit_rate=source_excluded_thresholds,
        max_source_family_excluded_target_hit_rate=source_family_excluded_thresholds,
        max_chunk_strategy_excluded_target_hit_rate=chunk_strategy_excluded_thresholds,
        max_retrieval_role_excluded_target_hit_rate=retrieval_role_excluded_thresholds,
        recall_weight=recall_weight,
        target_coverage_weight=target_coverage_weight,
        target_ndcg_weight=target_ndcg_weight,
        mrr_weight=mrr_weight,
        precision_weight=precision_weight,
        failed_query_penalty=failed_query_penalty,
        excluded_query_hit_penalty=excluded_query_hit_penalty,
        excluded_target_hit_penalty=excluded_target_hit_penalty,
        source_excluded_target_hit_penalty=source_excluded_target_hit_penalty,
        source_family_excluded_target_hit_penalty=source_family_excluded_target_hit_penalty,
        chunk_strategy_excluded_target_hit_penalty=chunk_strategy_excluded_target_hit_penalty,
        retrieval_role_excluded_target_hit_penalty=retrieval_role_excluded_target_hit_penalty,
        latency_weight=latency_weight,
        p95_latency_weight=p95_latency_weight,
        pairwise_top_k=pairwise_top_k,
        metadata={
            "weight_grid": grid,
            "fixed_fusion_weights": fixed_weights,
            "include_fixed_candidate": include_fixed_candidate,
            "candidate_count": len(candidate_weights),
            "package_dir": str(package_dir),
            "collection": prepared["collection_name"],
            "bm25_tokens_path": str(package_dir / "bm25_tokens.json"),
            "top_k": top_k,
            "repeat": repeat,
            "collapse_hierarchical": collapse_hierarchical,
            "query_encoders": prepared["query_encoders"],
            "reranker": normalize_backend(reranker),
            "reranker_model": reranker_model if parsed_reranker else "",
            "reranker_max_length": reranker_max_length if parsed_reranker else 0,
            "rerank_top_k": effective_rerank_top_k or 0,
            "max_mean_latency_ms": max_mean_latency_ms,
            "max_p95_latency_ms": max_p95_latency_ms,
            "max_excluded_target_hit_rate": max_excluded_target_hit_rate,
            "max_excluded_query_hit_rate": max_excluded_query_hit_rate,
            "max_excluded_hit_query_count": max_excluded_hit_query_count,
            "max_source_excluded_target_hit_rate": source_excluded_thresholds,
            "max_source_family_excluded_target_hit_rate": source_family_excluded_thresholds,
            "max_chunk_strategy_excluded_target_hit_rate": chunk_strategy_excluded_thresholds,
            "max_retrieval_role_excluded_target_hit_rate": retrieval_role_excluded_thresholds,
            "excluded_query_hit_penalty": excluded_query_hit_penalty,
            "excluded_target_hit_penalty": excluded_target_hit_penalty,
            "source_excluded_target_hit_penalty": source_excluded_target_hit_penalty,
            "source_family_excluded_target_hit_penalty": source_family_excluded_target_hit_penalty,
            "chunk_strategy_excluded_target_hit_penalty": chunk_strategy_excluded_target_hit_penalty,
            "retrieval_role_excluded_target_hit_penalty": retrieval_role_excluded_target_hit_penalty,
            "latency_weight": latency_weight,
            "p95_latency_weight": p95_latency_weight,
            "pairwise_top_k": pairwise_top_k,
            "lexical_tokenizer": {
                "strategy": lexical_tokenizer,
                "min_n": ngram_min,
                "max_n": ngram_max,
                "ngram_cjk_only": ngram_cjk_only,
                "deduplicate": deduplicate_tokens,
            },
        },
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        {
            "output": str(output) if output is not None else None,
            "candidate_count": report.candidate_count,
            "eligible_count": report.eligible_count,
            "recommended": report.recommended,
            "best_by_recall": report.best_by_recall,
            "best_by_target_coverage": report.best_by_target_coverage,
            "best_by_target_ndcg": report.best_by_target_ndcg,
            "best_by_mrr": report.best_by_mrr,
            "fastest_by_mean_latency": report.fastest_by_mean_latency,
            "case_group_recommendations": {
                group_name: {
                    group_value: {
                        "recommended": recommendation.recommended,
                        "best_by_target_coverage": recommendation.best_by_target_coverage,
                        "best_by_target_ndcg": recommendation.best_by_target_ndcg,
                        "eligible_count": recommendation.eligible_count,
                        "top_candidates": [
                            candidate.name
                            for candidate in recommendation.top_candidates
                        ],
                    }
                    for group_value, recommendation in group_values.items()
                }
                for group_name, group_values in report.case_group_recommendations.items()
            },
            "candidates": [
                {
                    "rank": candidate.rank,
                    "name": candidate.name,
                    "fusion_weights": candidate.fusion_weights,
                    "selection_score": candidate.selection_score,
                    "eligible": candidate.eligible,
                    "eligibility_failures": candidate.eligibility_failures,
                    "recall_at_k": candidate.evaluation.recall_at_k,
                    "target_coverage_at_k": candidate.evaluation.target_coverage_at_k,
                    "mean_target_ndcg_at_k": candidate.evaluation.mean_target_ndcg_at_k,
                    "mrr": candidate.evaluation.mrr,
                    "excluded_query_hit_rate": candidate.evaluation.excluded_query_hit_rate,
                    "excluded_target_hit_rate": candidate.evaluation.excluded_target_hit_rate,
                    "excluded_hit_query_count": candidate.evaluation.excluded_hit_query_count,
                    "max_source_excluded_target_hit_rate": (
                        candidate.max_source_excluded_target_hit_rate
                    ),
                    "max_source_excluded_target_hit_rate_name": (
                        candidate.max_source_excluded_target_hit_rate_name
                    ),
                    "max_source_family_excluded_target_hit_rate": (
                        candidate.max_source_family_excluded_target_hit_rate
                    ),
                    "max_source_family_excluded_target_hit_rate_name": (
                        candidate.max_source_family_excluded_target_hit_rate_name
                    ),
                    "max_chunk_strategy_excluded_target_hit_rate": (
                        candidate.max_chunk_strategy_excluded_target_hit_rate
                    ),
                    "max_chunk_strategy_excluded_target_hit_rate_name": (
                        candidate.max_chunk_strategy_excluded_target_hit_rate_name
                    ),
                    "max_retrieval_role_excluded_target_hit_rate": (
                        candidate.max_retrieval_role_excluded_target_hit_rate
                    ),
                    "max_retrieval_role_excluded_target_hit_rate_name": (
                        candidate.max_retrieval_role_excluded_target_hit_rate_name
                    ),
                    "mean_latency_ms": candidate.evaluation.mean_latency_ms,
                    "p95_latency_ms": candidate.evaluation.p95_latency_ms,
                    "failed_query_count": len(candidate.evaluation.failed_queries),
                }
                for candidate in report.candidates[: max(summary_limit, 0)]
            ],
            "pairwise_against_recommended": [
                {
                    "baseline": comparison.baseline,
                    "shared_query_count": comparison.shared_query_count,
                    "candidate_win_rate": comparison.candidate_win_rate,
                    "mean_target_coverage_delta": comparison.mean_target_coverage_delta,
                    "mean_target_ndcg_delta": comparison.mean_target_ndcg_delta,
                    "mean_target_rank_delta": comparison.mean_target_rank_delta,
                    "mean_latency_delta_ms": comparison.mean_latency_delta_ms,
                }
                for comparison in report.pairwise
                if comparison.candidate == report.recommended
            ][: max(summary_limit, 0)],
        }
    )


@app.command(name="export-qdrant-retrieval-config")
def export_qdrant_retrieval_config_command(
    report: Path,
    output: Path | None = None,
    candidate: str = typer.Option(
        "",
        "--candidate",
        help="Exact fusion sweep candidate name to export. Overrides --case-group selection.",
    ),
    case_group: str = typer.Option(
        "",
        "--case-group",
        help="Case-group recommendation to export, such as case_source:visual_object_probe.",
    ),
    route_preset: str = typer.Option(
        "",
        "--route-preset",
        help="Attach a retrieval route preset such as adaptive or visual-object-graph.",
    ),
):
    """Export a service-ready Qdrant retrieval config from a fusion sweep report."""
    try:
        sweep_report = QdrantFusionSweepReport.model_validate_json(
            report.read_text(encoding="utf-8")
        )
        config = build_qdrant_retrieval_config_from_fusion_sweep(
            sweep_report,
            candidate_name=candidate or None,
            case_group=case_group or None,
            source_report=str(report),
        )
        config = apply_qdrant_retrieval_route_preset(config, route_preset)
    except OSError as exc:
        raise typer.BadParameter(f"Could not read fusion sweep report: {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    print(
        {
            "output": str(output) if output is not None else None,
            "backend": config.backend,
            "vector_names": config.vector_names,
            "graph_expand": config.graph_expand,
            "top_k": config.top_k,
            "fusion_weights": config.fusion_weights,
            "reranker": config.reranker,
            "rerank_top_k": config.rerank_top_k,
            "routes": [route.name for route in config.routes],
            "selection": config.selection.model_dump(),
        }
    )


@app.command(name="gate-qdrant-vector-ablation")
def gate_qdrant_vector_ablation_command(
    report: Path,
    mode: str = typer.Option(
        ...,
        "--mode",
        help="Ablation mode to gate, such as image or caption_image.",
    ),
    baseline_mode: str | None = typer.Option(
        None,
        "--baseline-mode",
        help="Optional baseline mode for query-paired comparison metrics.",
    ),
    output: Path | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_excluded_target_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-target-hit-rate",
        help="Limit selected mode explicit excluded page/chunk/asset/triple target hit rate.",
    ),
    max_excluded_query_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-query-hit-rate",
        help="Limit selected mode hard-negative query hit rate.",
    ),
    max_excluded_hit_query_count: int | None = typer.Option(
        None,
        "--max-excluded-hit-query-count",
        help="Limit selected mode hard-negative hit query count.",
    ),
    min_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-target-type-coverage",
        help="Require target-type coverage such as asset=1.0. Repeat for multiple types.",
    ),
    min_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-target-coverage",
        help=(
            "Require exact retrieval-source target coverage such as "
            "qdrant:image_dense=0.5. Repeat for multiple sources."
        ),
    ),
    min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-family-target-coverage",
        help="Require source-family target coverage such as visual=0.8. Repeat for multiple families.",
    ),
    max_source_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-excluded-target-hit-rate",
        help=(
            "Limit selected mode exact-source excluded-target hit rate such as "
            "qdrant:image_dense=0.0."
        ),
    ),
    max_source_family_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-family-excluded-target-hit-rate",
        help="Limit selected mode source-family excluded-target hit rate such as visual=0.0.",
    ),
    max_chunk_strategy_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-chunk-strategy-excluded-target-hit-rate",
        help=(
            "Limit selected mode chunking-strategy excluded-target hit rate such as "
            "visual_asset_text=0.0."
        ),
    ),
    max_retrieval_role_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-retrieval-role-excluded-target-hit-rate",
        help="Limit selected mode retrieval-role excluded-target hit rate such as child=0.0.",
    ),
    min_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-case-group-target-coverage",
        help=(
            "Require metadata case-group target coverage such as "
            "case_source:visual_object_probe=0.7."
        ),
    ),
    min_pairwise_shared_queries: int | None = None,
    min_pairwise_win_rate: float | None = None,
    min_pairwise_target_coverage_lift: float | None = None,
    min_pairwise_target_ndcg_lift: float | None = None,
    min_pairwise_mrr_lift: float | None = None,
    min_pairwise_precision_lift: float | None = None,
    min_pairwise_target_coverage_ci_low: float | None = None,
    min_pairwise_target_ndcg_ci_low: float | None = None,
    min_pairwise_mrr_ci_low: float | None = None,
    min_pairwise_precision_ci_low: float | None = None,
    max_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_pairwise_mean_target_rank_delta: float | None = None,
    max_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_pairwise_target_rank_delta_ci_high: float | None = None,
    max_pairwise_mean_latency_delta_ms: float | None = None,
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
    source_thresholds = parse_named_float_thresholds(
        min_source_target_coverage,
        "source target coverage",
    )
    source_excluded_thresholds = parse_named_float_thresholds(
        max_source_excluded_target_hit_rate,
        "source excluded-target hit rate",
    )
    source_family_excluded_thresholds = parse_named_float_thresholds(
        max_source_family_excluded_target_hit_rate,
        "source family excluded-target hit rate",
    )
    chunk_strategy_excluded_thresholds = parse_named_float_thresholds(
        max_chunk_strategy_excluded_target_hit_rate,
        "chunk strategy excluded-target hit rate",
    )
    retrieval_role_excluded_thresholds = parse_named_float_thresholds(
        max_retrieval_role_excluded_target_hit_rate,
        "retrieval role excluded-target hit rate",
    )
    target_type_thresholds = parse_named_float_thresholds(
        min_target_type_coverage,
        "target type coverage",
    )
    case_group_thresholds = parse_named_float_thresholds(
        min_case_group_target_coverage,
        "case group target coverage",
    )
    try:
        gate_report = gate_qdrant_vector_ablation(
            parsed_report,
            mode=mode,
            baseline_mode=baseline_mode,
            min_recall_at_k=min_recall_at_k,
            min_target_coverage_at_k=min_target_coverage_at_k,
            min_target_ndcg_at_k=min_target_ndcg_at_k,
            min_mrr=min_mrr,
            min_precision_at_k=min_precision_at_k,
            max_failed_queries=max_failed_queries,
            max_mean_first_relevant_rank=max_mean_first_relevant_rank,
            max_p95_first_relevant_rank=max_p95_first_relevant_rank,
            max_mean_target_rank=max_mean_target_rank,
            max_p95_target_rank=max_p95_target_rank,
            max_mean_latency_ms=max_mean_latency_ms,
            max_p95_latency_ms=max_p95_latency_ms,
            max_excluded_target_hit_rate=max_excluded_target_hit_rate,
            max_excluded_query_hit_rate=max_excluded_query_hit_rate,
            max_excluded_hit_query_count=max_excluded_hit_query_count,
            min_target_type_coverage=target_type_thresholds,
            min_source_target_coverage=source_thresholds,
            min_source_family_target_coverage=source_family_thresholds,
            max_source_excluded_target_hit_rate=source_excluded_thresholds,
            max_source_family_excluded_target_hit_rate=source_family_excluded_thresholds,
            max_chunk_strategy_excluded_target_hit_rate=chunk_strategy_excluded_thresholds,
            max_retrieval_role_excluded_target_hit_rate=retrieval_role_excluded_thresholds,
            min_case_group_target_coverage=case_group_thresholds,
            min_pairwise_shared_queries=min_pairwise_shared_queries,
            min_pairwise_win_rate=min_pairwise_win_rate,
            min_pairwise_target_coverage_lift=min_pairwise_target_coverage_lift,
            min_pairwise_target_ndcg_lift=min_pairwise_target_ndcg_lift,
            min_pairwise_mrr_lift=min_pairwise_mrr_lift,
            min_pairwise_precision_lift=min_pairwise_precision_lift,
            min_pairwise_target_coverage_ci_low=min_pairwise_target_coverage_ci_low,
            min_pairwise_target_ndcg_ci_low=min_pairwise_target_ndcg_ci_low,
            min_pairwise_mrr_ci_low=min_pairwise_mrr_ci_low,
            min_pairwise_precision_ci_low=min_pairwise_precision_ci_low,
            max_pairwise_mean_first_relevant_rank_delta=(
                max_pairwise_mean_first_relevant_rank_delta
            ),
            max_pairwise_mean_target_rank_delta=max_pairwise_mean_target_rank_delta,
            max_pairwise_first_relevant_rank_delta_ci_high=(
                max_pairwise_first_relevant_rank_delta_ci_high
            ),
            max_pairwise_target_rank_delta_ci_high=max_pairwise_target_rank_delta_ci_high,
            max_pairwise_mean_latency_delta_ms=max_pairwise_mean_latency_delta_ms,
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
            "baseline_mode": gate_report.baseline_mode,
            "vector_names": gate_report.vector_names,
            "failed_checks": gate_report.failed_checks,
            "metrics": gate_report.metrics,
            "baseline_metrics": gate_report.baseline_metrics,
            "target_metrics": gate_report.target_metrics,
            "source_metrics": gate_report.source_metrics,
            "source_family_metrics": gate_report.source_family_metrics,
            "chunk_strategy_metrics": gate_report.chunk_strategy_metrics,
            "retrieval_role_metrics": gate_report.retrieval_role_metrics,
            "case_group_metrics": gate_report.case_group_metrics,
            "pairwise_metrics": gate_report.pairwise_metrics,
            "case_group_best_modes": gate_report.case_group_best_modes,
        }
    print(payload)
    if fail and not gate_report.passed:
        raise typer.Exit(1)


@app.command(name="gate-qdrant-reranker-ablation")
def gate_qdrant_reranker_ablation_command(
    report: Path,
    mode: str = typer.Option(
        ...,
        "--mode",
        help="Reranker ablation mode to gate, such as lexical or cross_encoder.",
    ),
    baseline_mode: str | None = typer.Option(
        None,
        "--baseline-mode",
        help="Optional baseline mode for query-paired comparison metrics.",
    ),
    output: Path | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_excluded_target_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-target-hit-rate",
        help="Limit selected mode explicit excluded page/chunk/asset/triple target hit rate.",
    ),
    max_excluded_query_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-query-hit-rate",
        help="Limit selected mode hard-negative query hit rate.",
    ),
    max_excluded_hit_query_count: int | None = typer.Option(
        None,
        "--max-excluded-hit-query-count",
        help="Limit selected mode hard-negative hit query count.",
    ),
    min_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-target-type-coverage",
        help="Require target-type coverage such as asset=1.0. Repeat for multiple types.",
    ),
    min_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-target-coverage",
        help=(
            "Require exact retrieval-source target coverage such as "
            "rerank:lexical=0.8. Repeat for multiple sources."
        ),
    ),
    min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-family-target-coverage",
        help="Require source-family target coverage such as lexical=0.8. Repeat for multiple families.",
    ),
    max_source_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-excluded-target-hit-rate",
        help=(
            "Limit selected mode exact-source excluded-target hit rate such as "
            "rerank:lexical=0.0."
        ),
    ),
    max_source_family_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-family-excluded-target-hit-rate",
        help="Limit selected mode source-family excluded-target hit rate such as lexical=0.0.",
    ),
    max_chunk_strategy_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-chunk-strategy-excluded-target-hit-rate",
        help=(
            "Limit selected mode chunking-strategy excluded-target hit rate such as "
            "visual_asset_text=0.0."
        ),
    ),
    max_retrieval_role_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-retrieval-role-excluded-target-hit-rate",
        help="Limit selected mode retrieval-role excluded-target hit rate such as child=0.0.",
    ),
    min_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-case-group-target-coverage",
        help=(
            "Require metadata case-group target coverage such as "
            "case_source:visual_object_probe=0.7."
        ),
    ),
    min_pairwise_shared_queries: int | None = None,
    min_pairwise_win_rate: float | None = None,
    min_pairwise_target_coverage_lift: float | None = None,
    min_pairwise_target_ndcg_lift: float | None = None,
    min_pairwise_mrr_lift: float | None = None,
    min_pairwise_precision_lift: float | None = None,
    min_pairwise_target_coverage_ci_low: float | None = None,
    min_pairwise_target_ndcg_ci_low: float | None = None,
    min_pairwise_mrr_ci_low: float | None = None,
    min_pairwise_precision_ci_low: float | None = None,
    max_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_pairwise_mean_target_rank_delta: float | None = None,
    max_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_pairwise_target_rank_delta_ci_high: float | None = None,
    max_pairwise_mean_latency_delta_ms: float | None = None,
    require_best_by_recall: bool = False,
    require_best_by_target_coverage: bool = False,
    require_best_by_target_ndcg: bool = False,
    require_fastest_by_mean_latency: bool = False,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when the reranker ablation gate fails.",
    ),
):
    """Fail a Qdrant reranker mode when retrieval lift or latency is outside thresholds."""
    parsed_report = QdrantRerankerAblationReport.model_validate_json(
        report.read_text(encoding="utf-8")
    )
    source_family_thresholds = parse_named_float_thresholds(
        min_source_family_target_coverage,
        "source family target coverage",
    )
    source_thresholds = parse_named_float_thresholds(
        min_source_target_coverage,
        "source target coverage",
    )
    source_excluded_thresholds = parse_named_float_thresholds(
        max_source_excluded_target_hit_rate,
        "source excluded-target hit rate",
    )
    source_family_excluded_thresholds = parse_named_float_thresholds(
        max_source_family_excluded_target_hit_rate,
        "source family excluded-target hit rate",
    )
    chunk_strategy_excluded_thresholds = parse_named_float_thresholds(
        max_chunk_strategy_excluded_target_hit_rate,
        "chunk strategy excluded-target hit rate",
    )
    retrieval_role_excluded_thresholds = parse_named_float_thresholds(
        max_retrieval_role_excluded_target_hit_rate,
        "retrieval role excluded-target hit rate",
    )
    target_type_thresholds = parse_named_float_thresholds(
        min_target_type_coverage,
        "target type coverage",
    )
    case_group_thresholds = parse_named_float_thresholds(
        min_case_group_target_coverage,
        "case group target coverage",
    )
    try:
        gate_report = gate_qdrant_reranker_ablation(
            parsed_report,
            mode=mode,
            baseline_mode=baseline_mode,
            min_recall_at_k=min_recall_at_k,
            min_target_coverage_at_k=min_target_coverage_at_k,
            min_target_ndcg_at_k=min_target_ndcg_at_k,
            min_mrr=min_mrr,
            min_precision_at_k=min_precision_at_k,
            max_failed_queries=max_failed_queries,
            max_mean_first_relevant_rank=max_mean_first_relevant_rank,
            max_p95_first_relevant_rank=max_p95_first_relevant_rank,
            max_mean_target_rank=max_mean_target_rank,
            max_p95_target_rank=max_p95_target_rank,
            max_mean_latency_ms=max_mean_latency_ms,
            max_p95_latency_ms=max_p95_latency_ms,
            max_excluded_target_hit_rate=max_excluded_target_hit_rate,
            max_excluded_query_hit_rate=max_excluded_query_hit_rate,
            max_excluded_hit_query_count=max_excluded_hit_query_count,
            min_target_type_coverage=target_type_thresholds,
            min_source_target_coverage=source_thresholds,
            min_source_family_target_coverage=source_family_thresholds,
            max_source_excluded_target_hit_rate=source_excluded_thresholds,
            max_source_family_excluded_target_hit_rate=source_family_excluded_thresholds,
            max_chunk_strategy_excluded_target_hit_rate=chunk_strategy_excluded_thresholds,
            max_retrieval_role_excluded_target_hit_rate=retrieval_role_excluded_thresholds,
            min_case_group_target_coverage=case_group_thresholds,
            min_pairwise_shared_queries=min_pairwise_shared_queries,
            min_pairwise_win_rate=min_pairwise_win_rate,
            min_pairwise_target_coverage_lift=min_pairwise_target_coverage_lift,
            min_pairwise_target_ndcg_lift=min_pairwise_target_ndcg_lift,
            min_pairwise_mrr_lift=min_pairwise_mrr_lift,
            min_pairwise_precision_lift=min_pairwise_precision_lift,
            min_pairwise_target_coverage_ci_low=min_pairwise_target_coverage_ci_low,
            min_pairwise_target_ndcg_ci_low=min_pairwise_target_ndcg_ci_low,
            min_pairwise_mrr_ci_low=min_pairwise_mrr_ci_low,
            min_pairwise_precision_ci_low=min_pairwise_precision_ci_low,
            max_pairwise_mean_first_relevant_rank_delta=(
                max_pairwise_mean_first_relevant_rank_delta
            ),
            max_pairwise_mean_target_rank_delta=max_pairwise_mean_target_rank_delta,
            max_pairwise_first_relevant_rank_delta_ci_high=(
                max_pairwise_first_relevant_rank_delta_ci_high
            ),
            max_pairwise_target_rank_delta_ci_high=max_pairwise_target_rank_delta_ci_high,
            max_pairwise_mean_latency_delta_ms=max_pairwise_mean_latency_delta_ms,
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
            "baseline_mode": gate_report.baseline_mode,
            "reranker": gate_report.reranker,
            "rerank_top_k": gate_report.rerank_top_k,
            "failed_checks": gate_report.failed_checks,
            "metrics": gate_report.metrics,
            "baseline_metrics": gate_report.baseline_metrics,
            "target_metrics": gate_report.target_metrics,
            "source_metrics": gate_report.source_metrics,
            "source_family_metrics": gate_report.source_family_metrics,
            "chunk_strategy_metrics": gate_report.chunk_strategy_metrics,
            "retrieval_role_metrics": gate_report.retrieval_role_metrics,
            "case_group_metrics": gate_report.case_group_metrics,
            "pairwise_metrics": gate_report.pairwise_metrics,
            "case_group_best_modes": gate_report.case_group_best_modes,
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
    object_backend: str = "same-as-caption",
    image_backend: str = "clip",
    triple_backend: str = "same-as-text",
    text_model: str = "BAAI/bge-m3",
    caption_model: str = "",
    object_model: str = "",
    image_model: str = "openai/clip-vit-large-patch14",
    triple_model: str = "",
    device: str = "cuda",
    text_device: str = "",
    image_device: str = "",
    hashing_dim: int = 384,
    text_batch_size: int = 16,
    caption_batch_size: int = 16,
    object_batch_size: int = 16,
    image_batch_size: int = 8,
    triple_batch_size: int = 16,
):
    """Rebuild Qdrant records with concrete dense/image embedding models."""
    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    assets = read_jsonl(package_dir / "assets.jsonl", VisualAsset)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []

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
    object_embedder, object_note = build_object_embedder(
        backend=object_backend,
        model_name=object_model or caption_model or text_model,
        device=text_device or device,
        hashing_dim=hashing_dim,
        text_embedder=text_embedder,
        text_note=text_note,
        caption_embedder=caption_embedder,
        caption_note=caption_note,
    )
    image_embedder, image_note = build_image_embedder(
        backend=image_backend,
        model_name=image_model,
        device=image_device or device,
        hashing_dim=hashing_dim,
    )
    triple_embedder, triple_note = build_caption_embedder(
        backend=triple_backend,
        model_name=triple_model or text_model,
        device=text_device or device,
        hashing_dim=hashing_dim,
        text_embedder=text_embedder,
        text_note=text_note,
        vector_name="triple_dense",
        option_name="triple",
    )

    if (
        text_embedder is None
        and caption_embedder is None
        and object_embedder is None
        and image_embedder is None
        and triple_embedder is None
    ):
        raise typer.BadParameter("At least one backend must be enabled")

    notes = {}
    if text_note:
        notes["text_dense"] = text_note
    if caption_note:
        notes["caption_dense"] = caption_note
    if object_note:
        notes["object_dense"] = object_note
    if image_note:
        notes["image_dense"] = image_note
    if triple_note:
        notes["triple_dense"] = triple_note
    vector_metadata = embedding_vector_metadata(
        text_backend=text_backend,
        caption_backend=caption_backend,
        object_backend=object_backend,
        image_backend=image_backend,
        triple_backend=triple_backend,
        text_model=text_model,
        caption_model=caption_model or text_model,
        object_model=object_model or caption_model or text_model,
        image_model=image_model,
        triple_model=triple_model or text_model,
        text_device=text_device or device,
        image_device=image_device or device,
        text_batch_size=text_batch_size,
        caption_batch_size=caption_batch_size,
        object_batch_size=object_batch_size,
        image_batch_size=image_batch_size,
        triple_batch_size=triple_batch_size,
        hashing_dim=hashing_dim,
        include_text=text_embedder is not None,
        include_caption=caption_embedder is not None,
        include_object=object_embedder is not None,
        include_image=image_embedder is not None,
        include_triple=triple_embedder is not None,
    )

    result = write_embedding_artifacts(
        output_dir=package_dir,
        chunks=chunks,
        assets=assets,
        triples=triples,
        text_embedder=text_embedder,
        caption_embedder=caption_embedder,
        object_embedder=object_embedder,
        image_embedder=image_embedder,
        triple_embedder=triple_embedder,
        collection=collection,
        text_batch_size=text_batch_size,
        caption_batch_size=caption_batch_size,
        object_batch_size=object_batch_size,
        image_batch_size=image_batch_size,
        triple_batch_size=triple_batch_size,
        vector_notes=notes,
        vector_metadata=vector_metadata,
    )
    print(
        {
            **result,
            "package_dir": str(package_dir),
            "text_backend": text_backend,
            "caption_backend": caption_backend,
            "object_backend": object_backend,
            "image_backend": image_backend,
            "triple_backend": triple_backend,
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
    offset: int = 0,
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
        offset=offset,
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
            "offset": max(0, offset),
            "limit": limit,
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


@app.command(name="merge-visual-results")
def merge_visual_results_command(
    results: list[Path] = typer.Option(
        None,
        "--results",
        help="Visual job results JSONL file to merge. Repeat for multiple batch files.",
    ),
    output: Path = Path("outputs/package/visual_job_results.jsonl"),
    annotations_output: Path | None = None,
):
    """Merge batch visual job result files into one run-level result file."""
    if not results:
        raise typer.BadParameter("At least one --results file is required.")
    parsed_results: list[VisualJobRunResult] = []
    for result_path in results:
        parsed_results.extend(read_jsonl(result_path, VisualJobRunResult))
    merged_results = merge_visual_job_results(parsed_results)
    annotations = completed_annotations(merged_results)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, merged_results)
    if annotations_output is not None:
        annotations_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(annotations_output, annotations)
    print(
        {
            "input_file_count": len(results),
            "input_result_count": len(parsed_results),
            "output": str(output),
            "merged_result_count": len(merged_results),
            "completed": sum(1 for result in merged_results if result.status == "completed"),
            "failed": sum(1 for result in merged_results if result.status == "failed"),
            "skipped": sum(1 for result in merged_results if result.status == "skipped"),
            "annotations": len(annotations),
            "annotations_output": str(annotations_output) if annotations_output else None,
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
    retrieval_evals: list[str] = typer.Option(
        None,
        "--retrieval-eval",
        help=(
            "Optional retrieval evaluation in name=retrieval_eval.json form. "
            "Names should match --run names."
        ),
    ),
    output: Path | None = None,
    require_same_jobs: bool = typer.Option(
        False,
        "--require-same-jobs/--no-require-same-jobs",
        help="Exit with status 1 when compared runs were not produced from the same job IDs.",
    ),
):
    """Compare multiple OCR/VLM result files by coverage, parse rate, triples, and latency."""
    parsed_runs = parse_visual_run_inputs(runs)
    parsed_retrieval_evals = parse_retrieval_eval_inputs(retrieval_evals)
    comparison = compare_visual_runs(
        {
            name: read_jsonl(path, VisualJobRunResult)
            for name, path in parsed_runs.items()
        },
        retrieval_evaluations={
            name: load_retrieval_evaluation(path) for name, path in parsed_retrieval_evals.items()
        },
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
            "best_by_retrieval": comparison.best_by_retrieval,
            "retrieval_evaluation_run_count": comparison.retrieval_evaluation_run_count,
            "missing_retrieval_evaluation_runs": comparison.missing_retrieval_evaluation_runs,
            "job_set_mismatch": comparison.job_set_mismatch,
            "union_job_count": comparison.union_job_count,
            "shared_job_count": comparison.shared_job_count,
            "missing_job_ids_by_run": comparison.missing_job_ids_by_run,
        }
    print(payload)
    if require_same_jobs and comparison.job_set_mismatch:
        raise typer.Exit(1)


@app.command(name="plan-vlm-experiments")
def plan_vlm_experiments_command(
    package_dir: Path = Path("outputs/package"),
    jobs: Path | None = None,
    profiles: str = "qwen2_5_vl_7b,qwen2_vl_7b,llava_next_7b",
    output: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
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
    vlm_memory_margin_ratio: float = 0.1,
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
            batch_size=batch_size,
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
            vlm_memory_margin_ratio=vlm_memory_margin_ratio,
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
            "total_job_count": plan.job_summary.total_job_count,
            "selected_job_count": plan.job_summary.selected_job_count,
            "batch_size": plan.batch_size,
            "batch_count": len(plan.batches),
            "operation_counts": plan.job_summary.operation_counts,
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
    min_vlm_object_coverage: float = 0.0,
    min_objects_per_vlm_job: float = 0.0,
    min_object_bbox_coverage: float = 0.0,
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
        min_vlm_object_coverage=min_vlm_object_coverage,
        min_objects_per_vlm_job=min_objects_per_vlm_job,
        min_object_bbox_coverage=min_object_bbox_coverage,
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
            "vlm_object_coverage": report.vlm_object_coverage,
            "objects_per_vlm_job": report.objects_per_vlm_job,
            "object_bbox_coverage": report.object_bbox_coverage,
            "triples_per_vlm_job": report.triples_per_vlm_job,
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


@app.command(name="gate-vlm-experiment-plan")
def gate_vlm_experiment_plan_command(
    plan: Path = Path("outputs/package/vlm_experiment_plan.json"),
    output: Path | None = None,
    min_profile_count: int = 1,
    require_doctor_outputs: bool = False,
    require_results: bool = False,
    require_annotations: bool = False,
    min_completed_result_profiles: int = 0,
    require_same_result_jobs: bool = typer.Option(
        False,
        "--require-same-result-jobs/--no-require-same-result-jobs",
        help="Require existing profile result files to contain the same visual job IDs.",
    ),
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when VLM experiment plan checks fail.",
    ),
):
    """Gate VLM experiment plans against runtime and visual-result output coverage."""
    report = gate_vlm_experiment_plan(
        plan,
        min_profile_count=min_profile_count,
        require_doctor_outputs=require_doctor_outputs,
        require_results=require_results,
        require_annotations=require_annotations,
        min_completed_result_profiles=min_completed_result_profiles,
        require_same_result_jobs=require_same_result_jobs,
    )
    payload = report.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "failed_checks": report.failed_checks,
            "profile_count": report.profile_count,
            "recipe_count": report.recipe_count,
            "existing_doctor_output_count": report.existing_doctor_output_count,
            "existing_results_output_count": report.existing_results_output_count,
            "completed_result_profile_count": report.completed_result_profile_count,
            "existing_annotations_output_count": report.existing_annotations_output_count,
            "job_set_mismatch": report.job_set_mismatch,
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
    min_vlm_object_coverage: float = 0.0,
    min_objects_per_vlm_job: float = 0.0,
    min_object_bbox_coverage: float = 0.0,
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
        min_vlm_object_coverage=min_vlm_object_coverage,
        min_objects_per_vlm_job=min_objects_per_vlm_job,
        min_object_bbox_coverage=min_object_bbox_coverage,
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
            "vlm_object_coverage": report.vlm_object_coverage,
            "objects_per_vlm_job": report.objects_per_vlm_job,
            "object_bbox_coverage": report.object_bbox_coverage,
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
    deduplicate_tokens: bool = False,
):
    """Run local dry-run hybrid search over package chunks using hashing dense + BM25."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
        assets=assets,
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
    max_chars_per_asset_text: int = 1400,
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
    deduplicate_tokens: bool = False,
):
    """Build a citation-ready RAG context bundle from local hybrid search hits."""
    from chunking_docs.embeddings.interfaces import HashingTextEmbedder

    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
        assets=assets,
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
        max_chars_per_asset_text=max_chars_per_asset_text,
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
    from chunking_docs.graph.export import export_graph, summarize_graph

    chunks = read_jsonl(package_dir / "chunks.jsonl", DocumentChunk)
    triples = read_jsonl(package_dir / "triples.jsonl", GraphTriple)
    nodes, edges = export_graph(triples, chunks=chunks)
    summary = summarize_graph(nodes, edges)
    write_jsonl(package_dir / "graph_nodes.jsonl", nodes)
    write_jsonl(package_dir / "graph_edges.jsonl", edges)
    (package_dir / "graph_summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(
        {
            "nodes": len(nodes),
            "edges": len(edges),
            "connected_components": summary.connected_component_count,
            "largest_component_nodes": summary.largest_component_node_count,
            "nodes_output": str(package_dir / "graph_nodes.jsonl"),
            "edges_output": str(package_dir / "graph_edges.jsonl"),
            "summary_output": str(package_dir / "graph_summary.json"),
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
    from chunking_docs.graph.export import export_graph, summarize_graph
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
    graph_summary_output = None
    if export_graph_artifacts:
        nodes, edges = export_graph(normalized, chunks=chunks)
        summary = summarize_graph(nodes, edges)
        graph_nodes_output = package_dir / "graph_nodes.jsonl"
        graph_edges_output = package_dir / "graph_edges.jsonl"
        graph_summary_output = package_dir / "graph_summary.json"
        write_jsonl(graph_nodes_output, nodes)
        write_jsonl(graph_edges_output, edges)
        graph_summary_output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

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
            "graph_summary_output": str(graph_summary_output) if graph_summary_output else None,
        }
    )


@app.command(name="repair-visual-triples")
def repair_visual_triples_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    in_place: bool = False,
    export_graph_artifacts: bool = typer.Option(False, "--export-graph"),
    clear_stale_embeddings: bool = True,
):
    """Repair missing graph triples derived from stored VLM asset metadata."""
    from dataclasses import asdict

    from chunking_docs.graph.export import export_graph, summarize_graph

    chunks_path = package_dir / "chunks.jsonl"
    assets_path = package_dir / "assets.jsonl"
    triples_path = package_dir / "triples.jsonl"
    if not chunks_path.exists():
        raise typer.BadParameter(f"Missing chunks file: {chunks_path}")
    if not assets_path.exists():
        raise typer.BadParameter(f"Missing assets file: {assets_path}")
    if not triples_path.exists():
        raise typer.BadParameter(f"Missing triples file: {triples_path}")

    chunks = read_jsonl(chunks_path, DocumentChunk)
    assets = read_jsonl(assets_path, VisualAsset)
    triples = read_jsonl(triples_path, GraphTriple)
    repaired, report = repair_visual_derived_triples(assets, chunks, triples)

    output_path = package_dir / "triples.jsonl" if in_place else output
    if output_path is None:
        output_path = package_dir / "triples.visual_repaired.jsonl"
    write_jsonl(output_path, repaired)

    graph_nodes_output = None
    graph_edges_output = None
    graph_summary_output = None
    if in_place and export_graph_artifacts:
        nodes, edges = export_graph(repaired, chunks=chunks)
        summary = summarize_graph(nodes, edges)
        graph_nodes_output = package_dir / "graph_nodes.jsonl"
        graph_edges_output = package_dir / "graph_edges.jsonl"
        graph_summary_output = package_dir / "graph_summary.json"
        write_jsonl(graph_nodes_output, nodes)
        write_jsonl(graph_edges_output, edges)
        graph_summary_output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    cleared_embedding_artifacts = []
    changed = report.added_triples > 0 or report.updated_triples > 0
    if in_place and clear_stale_embeddings and changed:
        cleared_embedding_artifacts = clear_embedding_artifacts(package_dir)

    print_json(
        {
            **asdict(report),
            "source": str(triples_path),
            "output": str(output_path),
            "in_place": in_place,
            "graph_nodes_output": str(graph_nodes_output) if graph_nodes_output else None,
            "graph_edges_output": str(graph_edges_output) if graph_edges_output else None,
            "graph_summary_output": str(graph_summary_output) if graph_summary_output else None,
            "cleared_embedding_artifacts": cleared_embedding_artifacts,
            "requires_embedding_rebuild": bool(in_place and changed),
            "next_embedding_command": (
                f"chunking-docs embed-package --package-dir {shlex.quote(package_dir.as_posix())}"
                if in_place and changed
                else None
            ),
        }
    )


@app.command(name="repair-visual-text")
def repair_visual_text_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    in_place: bool = False,
    rebuild_search: bool = True,
    clear_stale_embeddings: bool = True,
):
    """Append missing structured visual asset text to linked chunks."""
    from dataclasses import asdict

    manifest = load_processing_package(package_dir)
    repaired_chunks, report = repair_visual_text_chunks(manifest.chunks, manifest.assets)
    output_path = package_dir / "chunks.jsonl" if in_place else output
    if output_path is None:
        output_path = package_dir / "chunks.visual_text_repaired.jsonl"
    write_jsonl(output_path, repaired_chunks)

    cleared_embedding_artifacts = []
    changed = report.updated_chunks > 0
    if in_place and rebuild_search:
        rebuild_search_artifacts(
            package_dir,
            repaired_chunks,
            assets=manifest.assets,
            triples=manifest.triples,
            tokenizer_config=manifest_tokenizer_config(manifest),
        )
    if in_place and clear_stale_embeddings and changed:
        cleared_embedding_artifacts = clear_embedding_artifacts(package_dir)

    print_json(
        {
            **asdict(report),
            "source": str(package_dir / "chunks.jsonl"),
            "output": str(output_path),
            "in_place": in_place,
            "rebuilt_search": bool(in_place and rebuild_search),
            "cleared_embedding_artifacts": cleared_embedding_artifacts,
            "requires_embedding_rebuild": bool(in_place and changed),
            "next_embedding_command": (
                f"chunking-docs embed-package --package-dir {shlex.quote(package_dir.as_posix())}"
                if in_place and changed
                else None
            ),
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


@app.command(name="postgres-schema")
def postgres_schema(output: Path | None = None):
    """Print or write the PostgreSQL schema SQL used by the package writer."""
    from chunking_docs.storage.postgres_store import postgres_schema_sql

    schema = postgres_schema_sql()
    if output is None:
        builtins.print(schema)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(schema, encoding="utf-8")
    print_json({"output": str(output), "bytes": len(schema.encode("utf-8"))})


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
            "missing_indexes": report.missing_indexes,
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
            "chunk_lexical_tokens": len(rows["chunk_lexical_tokens"]),
            "assets": len(rows["assets"]),
            "visual_objects": len(rows["visual_objects"]),
            "chunk_asset_links": len(rows["chunk_asset_links"]),
            "triples": len(rows["triples"]),
            "embedding_artifacts": len(rows["embedding_artifacts"]),
            "embedding_records": len(rows["embedding_records"]),
            "first_document": rows["document"],
        }
    )


@app.command(name="audit-package")
def audit_package_command(
    package_dir: Path = Path("outputs/package"),
    require_annotations_for_visual_pages: bool = False,
    require_qdrant_records: bool = False,
    require_visual_derived_triples: bool = typer.Option(
        False,
        "--require-visual-derived-triples",
        help=(
            "Fail when VLM entity, object, or visual-element metadata is not represented "
            "by graph triples with visual asset provenance."
        ),
    ),
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
        require_visual_derived_triples=require_visual_derived_triples,
    )
    print(audit.model_dump())


@app.command(name="audit-publication")
def audit_publication_command(
    root: Path = typer.Argument(Path(".")),
    output: Path | None = None,
    forbidden_pattern: list[str] = typer.Option(
        None,
        "--forbidden-pattern",
        help="Fail when public text files contain this case-insensitive pattern.",
    ),
    include_glob: list[str] = typer.Option(
        None,
        "--include-glob",
        help="Limit the scan to files matching this glob.",
    ),
    exclude_glob: list[str] = typer.Option(
        None,
        "--exclude-glob",
        help="Exclude files matching this glob in addition to generated-artifact defaults.",
    ),
    blocked_extension: list[str] = typer.Option(
        None,
        "--blocked-extension",
        help="Fail when a scanned public file has this extension, such as .pdf.",
    ),
    max_file_bytes: int = 2_000_000,
    max_text_scan_bytes: int = 512_000,
    required_gitignore_pattern: list[str] = typer.Option(
        None,
        "--required-gitignore-pattern",
        help="Require a .gitignore pattern for generated or private artifacts.",
    ),
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when publication audit checks fail.",
    ),
):
    """Audit public repository files for forbidden text and accidental artifacts."""
    report = audit_public_artifacts(
        root=root,
        forbidden_patterns=forbidden_pattern,
        include_globs=include_glob,
        exclude_globs=exclude_glob,
        blocked_extensions=blocked_extension,
        max_file_bytes=max_file_bytes,
        max_text_scan_bytes=max_text_scan_bytes,
        required_gitignore_patterns=required_gitignore_pattern,
    )
    payload = {"passed": report.passed, **report.model_dump()}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "passed": report.passed,
            "scanned_file_count": report.scanned_file_count,
            "skipped_file_count": report.skipped_file_count,
            "failed_checks": [issue.code for issue in report.issues if issue.severity == "error"],
        }
    print(payload)
    if fail and not report.passed:
        raise typer.Exit(1)


@app.command(name="ingestion-readiness")
def ingestion_readiness_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    require_qdrant_records: bool = True,
    require_bm25: bool = True,
    require_embedding_manifest: bool = True,
    required_vectors: list[str] = typer.Option(
        None,
        "--required-vector",
        help="Require an embedding vector family such as text_dense, caption_dense, object_dense, or image_dense.",
    ),
    require_derived_vector_coverage: bool = typer.Option(
        False,
        "--require-derived-vector-coverage",
        help=(
            "Fail readiness when chunks, visual text, structured visual objects, or graph "
            "triples exist without matching text/caption/object/triple Qdrant vector artifacts."
        ),
    ),
    require_postgres_rows: bool = True,
    runtime_report: Path | None = None,
    require_runtime_report: bool = typer.Option(
        False,
        "--require-runtime-report",
        help="Fail readiness when no runtime doctor report is supplied.",
    ),
    require_visual_annotations: bool = False,
    require_visual_derived_triples: bool = typer.Option(
        False,
        "--require-visual-derived-triples",
        help=(
            "Fail readiness when VLM entity, object, or visual-element metadata is not represented "
            "by graph triples with visual asset provenance."
        ),
    ),
    min_visual_text_coverage_ratio: float | None = typer.Option(
        None,
        "--min-visual-text-coverage-ratio",
        help="Require this fraction of linked visual assets to have text represented in package chunks.",
    ),
    min_visual_text_part_coverage_ratio: float | None = typer.Option(
        None,
        "--min-visual-text-part-coverage-ratio",
        help=(
            "Require this fraction of linked visual text parts, including OCR, VLM, "
            "and object metadata text, to be represented in package chunks."
        ),
    ),
    visual_results: Path | None = None,
    require_visual_quality: bool = False,
    min_visual_completion_rate: float = 0.0,
    min_ocr_text_coverage: float = 0.0,
    min_vlm_summary_coverage: float = 0.0,
    min_vlm_json_parse_rate: float = 0.0,
    min_vlm_object_coverage: float = 0.0,
    min_objects_per_vlm_job: float = 0.0,
    min_object_bbox_coverage: float = 0.0,
    max_visual_failed_count: int | None = None,
    visual_run_comparison: Path | None = None,
    require_visual_run_comparison: bool = False,
    min_visual_run_count: int = 2,
    require_visual_run_same_jobs: bool = False,
    min_visual_run_shared_jobs: int = 0,
    visual_run_best_by_quality: str | None = None,
    visual_run_best_by_triple_density: str | None = None,
    visual_run_best_by_retrieval: str | None = None,
    retrieval_cases: Path | None = None,
    require_retrieval_cases: bool = False,
    min_case_count: int = 1,
    min_page_cases: int = 0,
    min_chunk_cases: int = 0,
    min_asset_cases: int = 0,
    min_triple_cases: int = 0,
    min_retrieval_distinct_page_targets: int = 0,
    min_retrieval_distinct_chunk_targets: int = 0,
    min_retrieval_distinct_asset_targets: int = 0,
    min_retrieval_distinct_triple_targets: int = 0,
    max_retrieval_page_cases_per_target: int | None = None,
    max_retrieval_chunk_cases_per_target: int | None = None,
    max_retrieval_asset_cases_per_target: int | None = None,
    max_retrieval_triple_cases_per_target: int | None = None,
    min_retrieval_query_terms_per_case: int = 0,
    max_retrieval_target_query_overlap_ratio: float | None = typer.Option(
        None,
        "--max-retrieval-target-query-overlap-ratio",
        help=(
            "Fail retrieval case readiness when eligible query terms overlap expected target text "
            "above this ratio."
        ),
    ),
    max_retrieval_target_query_overlap_terms: int | None = typer.Option(
        None,
        "--max-retrieval-target-query-overlap-terms",
        help=(
            "Fail retrieval case readiness when a query uses too many distinct terms from "
            "expected target text."
        ),
    ),
    min_retrieval_terms_for_target_overlap: int = typer.Option(
        4,
        "--min-retrieval-terms-for-target-overlap",
        help="Only apply retrieval target-query overlap checks to queries with this many terms.",
    ),
    max_retrieval_expected_targets_per_case: int | None = typer.Option(
        None,
        "--max-retrieval-expected-targets-per-case",
        help=(
            "Fail retrieval case readiness when one case declares more expected "
            "page/chunk/asset/triple targets than this ceiling."
        ),
    ),
    min_retrieval_case_group_distinct_targets: list[str] = typer.Option(
        None,
        "--min-retrieval-case-group-distinct-targets",
        help=(
            "Require distinct targets inside a retrieval case group, such as "
            "case_source:visual_object_probe:asset=4."
        ),
    ),
    min_retrieval_case_group_count: list[str] = typer.Option(
        None,
        "--min-retrieval-case-group-count",
        help="Require retrieval case metadata group counts such as case_source:visual_object_probe=5.",
    ),
    require_visual_only_object_probes: bool = typer.Option(
        False,
        "--require-visual-only-object-probes",
        help="Fail readiness when visual_object_probe cases were not generated with visual-only object terms.",
    ),
    max_duplicate_queries: int = 0,
    retrieval_evaluation: Path | None = None,
    require_retrieval_evaluation: bool = False,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_retrieval_failed_queries: int | None = typer.Option(
        None,
        "--max-retrieval-failed-queries",
        help="Limit benchmark queries that failed in the supplied retrieval evaluation.",
    ),
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_result_stability_rate: float = typer.Option(
        0.0,
        "--min-result-stability-rate",
        help="Require repeated retrieval evaluations to keep top-k results stable.",
    ),
    max_unstable_result_count: int | None = typer.Option(
        None,
        "--max-unstable-result-count",
        help="Limit retrieval cases whose top-k result set changes across repeated evaluation.",
    ),
    min_retrieval_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-target-type-coverage",
        help="Require retrieval target-type coverage such as asset=1.0 or triple=1.0.",
    ),
    min_retrieval_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-source-family-target-coverage",
        help="Require retrieval source-family target coverage such as lexical=0.8.",
    ),
    min_retrieval_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-source-target-coverage",
        help="Require retrieval exact-source target coverage such as qdrant:caption_dense=0.8.",
    ),
    min_retrieval_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-case-group-target-coverage",
        help="Require retrieval case group target coverage such as case_source:visual_lexical_probe=0.8.",
    ),
    qdrant_retrieval_config: Path | None = typer.Option(
        None,
        "--qdrant-retrieval-config",
        help="Service Qdrant retrieval config JSON produced by export-qdrant-retrieval-config.",
    ),
    require_qdrant_retrieval_config: bool = typer.Option(
        False,
        "--require-qdrant-retrieval-config",
        help="Fail readiness when no service Qdrant retrieval config is supplied.",
    ),
    rag_context_evaluation: Path | None = typer.Option(
        None,
        "--rag-context-evaluation",
        help="Final RAG context evaluation JSON produced by eval-qdrant-rag-context-config.",
    ),
    require_rag_context_evaluation: bool = typer.Option(
        False,
        "--require-rag-context-evaluation",
        help="Fail readiness when no final RAG context evaluation is supplied.",
    ),
    min_rag_context_case_count: int = 0,
    min_rag_context_hit_rate: float = 0.0,
    min_rag_context_target_coverage: float = 0.0,
    min_rag_context_expected_case_count: int = 0,
    min_rag_context_expected_target_count: int = 0,
    min_rag_context_passed_case_count: int = 0,
    max_rag_context_failed_cases: int | None = None,
    max_rag_context_excluded_target_hit_rate: float | None = None,
    max_rag_context_excluded_query_hit_rate: float | None = None,
    max_rag_context_excluded_hit_query_count: int | None = None,
    max_rag_context_mean_latency_ms: float | None = None,
    max_rag_context_mean_context_char_count: float | None = None,
    max_rag_context_char_count: int | None = None,
    max_rag_context_mean_chunk_count: float | None = None,
    max_rag_context_mean_asset_count: float | None = None,
    max_rag_context_mean_triple_count: float | None = None,
    min_rag_context_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-rag-context-target-type-coverage",
        help="Require final-context target-type coverage such as asset=1.0 or triple=1.0.",
    ),
    min_rag_context_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-rag-context-case-group-target-coverage",
        help="Require final-context case-group coverage such as case_source:visual_object_probe=0.8.",
    ),
    max_rag_context_case_group_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-rag-context-case-group-excluded-target-hit-rate",
        help="Limit final-context case-group hard-negative leakage such as case_source:visual_image_probe=0.0.",
    ),
    chunking_comparison: Path | None = None,
    require_chunking_comparison: bool = False,
    chunking_candidate: str | None = None,
    baseline_chunking_candidate: str | None = None,
    min_chunking_quality_score: float = 0.0,
    min_chunking_recall_at_k: float | None = None,
    min_chunking_target_coverage_at_k: float | None = None,
    min_chunking_target_ndcg_at_k: float | None = None,
    max_chunking_mean_first_relevant_rank: float | None = None,
    max_chunking_p95_first_relevant_rank: float | None = None,
    max_chunking_mean_target_rank: float | None = None,
    max_chunking_p95_target_rank: float | None = None,
    min_chunking_result_stability_rate: float | None = typer.Option(
        None,
        "--min-chunking-result-stability-rate",
        help="Require selected chunking candidate repeated retrieval result stability.",
    ),
    max_chunking_unstable_result_count: int | None = typer.Option(
        None,
        "--max-chunking-unstable-result-count",
        help="Limit selected chunking candidate cases with changing repeated top-k results.",
    ),
    max_chunking_total_chunk_chars: float | None = typer.Option(
        None,
        "--max-chunking-total-chunk-chars",
        help="Limit selected chunking candidate total chunk text characters.",
    ),
    max_chunking_embedding_text_kchars: float | None = typer.Option(
        None,
        "--max-chunking-embedding-text-kchars",
        help="Limit selected chunking candidate embedding text volume in thousands of chars.",
    ),
    min_chunking_visual_text_coverage_ratio: float | None = typer.Option(
        None,
        "--min-chunking-visual-text-coverage-ratio",
        help="Require selected chunking comparison candidate linked visual text coverage.",
    ),
    min_chunking_visual_text_part_coverage_ratio: float | None = typer.Option(
        None,
        "--min-chunking-visual-text-part-coverage-ratio",
        help="Require selected chunking comparison candidate linked visual text part coverage.",
    ),
    min_chunking_retrieval_score_per_embedding_kchar: float | None = typer.Option(
        None,
        "--min-chunking-retrieval-score-per-embedding-kchar",
        help="Require selected chunking aggregate retrieval score per 1k embedding chars.",
    ),
    min_chunking_target_coverage_per_embedding_kchar: float | None = typer.Option(
        None,
        "--min-chunking-target-coverage-per-embedding-kchar",
        help="Require selected chunking target coverage per 1k embedding chars.",
    ),
    min_chunking_target_ndcg_per_embedding_kchar: float | None = typer.Option(
        None,
        "--min-chunking-target-ndcg-per-embedding-kchar",
        help="Require selected chunking target nDCG per 1k embedding chars.",
    ),
    min_chunking_retrieval_score_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--min-chunking-retrieval-score-per-mean-latency-ms",
        help="Require selected chunking aggregate retrieval score per mean latency ms.",
    ),
    min_chunking_target_coverage_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--min-chunking-target-coverage-per-mean-latency-ms",
        help="Require selected chunking target coverage per mean latency ms.",
    ),
    min_chunking_target_ndcg_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--min-chunking-target-ndcg-per-mean-latency-ms",
        help="Require selected chunking target nDCG per mean latency ms.",
    ),
    min_chunking_retrieval_score_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--min-chunking-retrieval-score-per-p95-latency-ms",
        help="Require selected chunking aggregate retrieval score per p95 latency ms.",
    ),
    min_chunking_target_coverage_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--min-chunking-target-coverage-per-p95-latency-ms",
        help="Require selected chunking target coverage per p95 latency ms.",
    ),
    min_chunking_target_ndcg_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--min-chunking-target-ndcg-per-p95-latency-ms",
        help="Require selected chunking target nDCG per p95 latency ms.",
    ),
    max_chunking_failed_queries: int | None = 0,
    max_chunking_recall_drop: float | None = None,
    max_chunking_mean_latency_ratio: float | None = None,
    max_chunking_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_chunking_pairwise_mean_target_rank_delta: float | None = None,
    max_chunking_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_chunking_pairwise_target_rank_delta_ci_high: float | None = None,
    min_chunking_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-chunking-target-type-coverage",
        help="Require selected chunking candidate target-type coverage such as asset=1.0.",
    ),
    min_chunking_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-chunking-source-family-target-coverage",
        help="Require selected chunking candidate source-family target coverage such as lexical=0.8.",
    ),
    min_chunking_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-chunking-case-group-target-coverage",
        help="Require selected chunking candidate case group target coverage such as case_source:visual_lexical_probe=0.8.",
    ),
    retrieval_ablation: Path | None = None,
    require_retrieval_ablation: bool = False,
    retrieval_ablation_mode: str | None = None,
    retrieval_ablation_baseline_mode: str | None = None,
    min_retrieval_ablation_recall_at_k: float = 0.0,
    min_retrieval_ablation_target_coverage_at_k: float = 0.0,
    min_retrieval_ablation_target_ndcg_at_k: float = 0.0,
    min_retrieval_ablation_mrr: float = 0.0,
    min_retrieval_ablation_precision_at_k: float = 0.0,
    max_retrieval_ablation_failed_queries: int | None = None,
    max_retrieval_ablation_mean_first_relevant_rank: float | None = None,
    max_retrieval_ablation_p95_first_relevant_rank: float | None = None,
    max_retrieval_ablation_mean_target_rank: float | None = None,
    max_retrieval_ablation_p95_target_rank: float | None = None,
    max_retrieval_ablation_mean_latency_ms: float | None = None,
    max_retrieval_ablation_p95_latency_ms: float | None = None,
    min_retrieval_ablation_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-ablation-target-type-coverage",
        help="Require selected retrieval ablation target-type coverage such as asset=1.0.",
    ),
    min_retrieval_ablation_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-ablation-source-family-target-coverage",
        help="Require selected retrieval ablation source-family coverage such as lexical=0.8.",
    ),
    min_retrieval_ablation_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-ablation-source-target-coverage",
        help="Require selected retrieval ablation exact-source coverage such as bm25=0.8.",
    ),
    min_retrieval_ablation_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-ablation-case-group-target-coverage",
        help=(
            "Require selected retrieval ablation case-group coverage such as "
            "case_source:visual_object_probe=0.7."
        ),
    ),
    min_retrieval_ablation_recall_lift: float | None = None,
    min_retrieval_ablation_target_coverage_lift: float | None = None,
    min_retrieval_ablation_target_ndcg_lift: float | None = None,
    min_retrieval_ablation_mrr_lift: float | None = None,
    min_retrieval_ablation_precision_lift: float | None = None,
    max_retrieval_ablation_mean_latency_ratio: float | None = None,
    max_retrieval_ablation_p95_latency_ratio: float | None = None,
    min_retrieval_ablation_pairwise_shared_queries: int | None = None,
    min_retrieval_ablation_pairwise_win_rate: float | None = None,
    min_retrieval_ablation_pairwise_target_coverage_lift: float | None = None,
    min_retrieval_ablation_pairwise_target_ndcg_lift: float | None = None,
    min_retrieval_ablation_pairwise_mrr_lift: float | None = None,
    min_retrieval_ablation_pairwise_precision_lift: float | None = None,
    min_retrieval_ablation_pairwise_target_coverage_ci_low: float | None = None,
    min_retrieval_ablation_pairwise_target_ndcg_ci_low: float | None = None,
    min_retrieval_ablation_pairwise_mrr_ci_low: float | None = None,
    min_retrieval_ablation_pairwise_precision_ci_low: float | None = None,
    max_retrieval_ablation_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_retrieval_ablation_pairwise_mean_target_rank_delta: float | None = None,
    max_retrieval_ablation_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_retrieval_ablation_pairwise_target_rank_delta_ci_high: float | None = None,
    max_retrieval_ablation_pairwise_mean_latency_delta_ms: float | None = None,
    require_retrieval_ablation_best_by_recall: bool = False,
    require_retrieval_ablation_best_by_target_coverage: bool = False,
    require_retrieval_ablation_best_by_target_ndcg: bool = False,
    require_retrieval_ablation_fastest_by_mean_latency: bool = False,
    qdrant_vector_ablation: Path | None = None,
    require_qdrant_vector_ablation: bool = False,
    qdrant_vector_mode: str | None = None,
    qdrant_vector_baseline_mode: str | None = None,
    min_qdrant_vector_pairwise_shared_queries: int | None = None,
    min_qdrant_vector_pairwise_win_rate: float | None = None,
    min_qdrant_vector_pairwise_target_coverage_lift: float | None = None,
    min_qdrant_vector_pairwise_target_ndcg_lift: float | None = None,
    min_qdrant_vector_pairwise_mrr_lift: float | None = None,
    min_qdrant_vector_pairwise_precision_lift: float | None = None,
    min_qdrant_vector_pairwise_target_coverage_ci_low: float | None = None,
    min_qdrant_vector_pairwise_target_ndcg_ci_low: float | None = None,
    min_qdrant_vector_pairwise_mrr_ci_low: float | None = None,
    min_qdrant_vector_pairwise_precision_ci_low: float | None = None,
    max_qdrant_vector_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_qdrant_vector_pairwise_mean_target_rank_delta: float | None = None,
    max_qdrant_vector_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_qdrant_vector_pairwise_target_rank_delta_ci_high: float | None = None,
    max_qdrant_vector_pairwise_mean_latency_delta_ms: float | None = None,
    min_qdrant_vector_recall_at_k: float = 0.0,
    min_qdrant_vector_target_coverage_at_k: float = 0.0,
    min_qdrant_vector_target_ndcg_at_k: float = 0.0,
    min_qdrant_vector_mrr: float = 0.0,
    min_qdrant_vector_precision_at_k: float = 0.0,
    max_qdrant_vector_failed_queries: int | None = None,
    max_qdrant_vector_mean_first_relevant_rank: float | None = None,
    max_qdrant_vector_p95_first_relevant_rank: float | None = None,
    max_qdrant_vector_mean_target_rank: float | None = None,
    max_qdrant_vector_p95_target_rank: float | None = None,
    max_qdrant_vector_mean_latency_ms: float | None = None,
    max_qdrant_vector_p95_latency_ms: float | None = None,
    min_qdrant_vector_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-vector-target-type-coverage",
        help="Require selected Qdrant vector target-type coverage such as asset=1.0.",
    ),
    min_qdrant_vector_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-vector-source-family-target-coverage",
        help="Require selected Qdrant vector source-family coverage such as visual=0.8.",
    ),
    min_qdrant_vector_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-vector-source-target-coverage",
        help="Require selected Qdrant vector exact-source coverage such as qdrant:image_dense=0.5.",
    ),
    min_qdrant_vector_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-vector-case-group-target-coverage",
        help=(
            "Require selected Qdrant vector case-group coverage such as "
            "case_source:visual_object_probe=0.7."
        ),
    ),
    require_qdrant_vector_best_by_recall: bool = False,
    require_qdrant_vector_best_by_target_coverage: bool = False,
    require_qdrant_vector_best_by_target_ndcg: bool = False,
    require_qdrant_vector_fastest_by_mean_latency: bool = False,
    qdrant_reranker_ablation: Path | None = None,
    require_qdrant_reranker_ablation: bool = False,
    qdrant_reranker_mode: str | None = None,
    qdrant_reranker_baseline_mode: str | None = None,
    min_qdrant_reranker_pairwise_shared_queries: int | None = None,
    min_qdrant_reranker_pairwise_win_rate: float | None = None,
    min_qdrant_reranker_pairwise_target_coverage_lift: float | None = None,
    min_qdrant_reranker_pairwise_target_ndcg_lift: float | None = None,
    min_qdrant_reranker_pairwise_mrr_lift: float | None = None,
    min_qdrant_reranker_pairwise_precision_lift: float | None = None,
    min_qdrant_reranker_pairwise_target_coverage_ci_low: float | None = None,
    min_qdrant_reranker_pairwise_target_ndcg_ci_low: float | None = None,
    min_qdrant_reranker_pairwise_mrr_ci_low: float | None = None,
    min_qdrant_reranker_pairwise_precision_ci_low: float | None = None,
    max_qdrant_reranker_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_qdrant_reranker_pairwise_mean_target_rank_delta: float | None = None,
    max_qdrant_reranker_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_qdrant_reranker_pairwise_target_rank_delta_ci_high: float | None = None,
    max_qdrant_reranker_pairwise_mean_latency_delta_ms: float | None = None,
    min_qdrant_reranker_recall_at_k: float = 0.0,
    min_qdrant_reranker_target_coverage_at_k: float = 0.0,
    min_qdrant_reranker_target_ndcg_at_k: float = 0.0,
    min_qdrant_reranker_mrr: float = 0.0,
    min_qdrant_reranker_precision_at_k: float = 0.0,
    max_qdrant_reranker_failed_queries: int | None = None,
    max_qdrant_reranker_mean_first_relevant_rank: float | None = None,
    max_qdrant_reranker_p95_first_relevant_rank: float | None = None,
    max_qdrant_reranker_mean_target_rank: float | None = None,
    max_qdrant_reranker_p95_target_rank: float | None = None,
    max_qdrant_reranker_mean_latency_ms: float | None = None,
    max_qdrant_reranker_p95_latency_ms: float | None = None,
    min_qdrant_reranker_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-reranker-target-type-coverage",
        help="Require selected Qdrant reranker target-type coverage such as asset=1.0.",
    ),
    min_qdrant_reranker_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-reranker-source-family-target-coverage",
        help="Require selected Qdrant reranker source-family coverage such as lexical=0.8.",
    ),
    min_qdrant_reranker_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-reranker-source-target-coverage",
        help="Require selected Qdrant reranker exact-source coverage such as rerank:lexical=0.8.",
    ),
    min_qdrant_reranker_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-qdrant-reranker-case-group-target-coverage",
        help=(
            "Require selected Qdrant reranker case-group coverage such as "
            "case_source:visual_object_probe=0.7."
        ),
    ),
    require_qdrant_reranker_best_by_recall: bool = False,
    require_qdrant_reranker_best_by_target_coverage: bool = False,
    require_qdrant_reranker_best_by_target_ndcg: bool = False,
    require_qdrant_reranker_fastest_by_mean_latency: bool = False,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when ingestion readiness checks fail.",
    ),
):
    """Check whether a package is ready for Qdrant/PostgreSQL ingestion and RAG evaluation."""
    manifest = load_processing_package(package_dir)
    parsed_visual_results = read_jsonl(visual_results, VisualJobRunResult) if visual_results else None
    parsed_visual_run_comparison = (
        VisualRunComparison.model_validate_json(
            visual_run_comparison.read_text(encoding="utf-8")
        )
        if visual_run_comparison
        else None
    )
    parsed_retrieval_cases = load_retrieval_cases(retrieval_cases) if retrieval_cases else None
    parsed_retrieval = load_retrieval_evaluation(retrieval_evaluation) if retrieval_evaluation else None
    parsed_qdrant_retrieval_config = (
        read_qdrant_retrieval_config(qdrant_retrieval_config)
        if qdrant_retrieval_config
        else None
    )
    parsed_runtime_report = (
        RuntimeReport.model_validate_json(runtime_report.read_text(encoding="utf-8"))
        if runtime_report
        else None
    )
    parsed_rag_context = (
        load_rag_context_evaluation(rag_context_evaluation)
        if rag_context_evaluation
        else None
    )
    parsed_chunking_comparison = load_chunking_comparison(chunking_comparison) if chunking_comparison else None
    parsed_retrieval_ablation = (
        RetrievalAblationReport.model_validate_json(
            retrieval_ablation.read_text(encoding="utf-8")
        )
        if retrieval_ablation
        else None
    )
    parsed_qdrant_vector_ablation = (
        QdrantVectorAblationReport.model_validate_json(
            qdrant_vector_ablation.read_text(encoding="utf-8")
        )
        if qdrant_vector_ablation
        else None
    )
    parsed_qdrant_reranker_ablation = (
        QdrantRerankerAblationReport.model_validate_json(
            qdrant_reranker_ablation.read_text(encoding="utf-8")
        )
        if qdrant_reranker_ablation
        else None
    )
    qdrant_vector_source_family_thresholds = parse_named_float_thresholds(
        min_qdrant_vector_source_family_target_coverage,
        "Qdrant vector source family target coverage",
    )
    qdrant_vector_source_thresholds = parse_named_float_thresholds(
        min_qdrant_vector_source_target_coverage,
        "Qdrant vector source target coverage",
    )
    qdrant_vector_target_type_thresholds = parse_named_float_thresholds(
        min_qdrant_vector_target_type_coverage,
        "Qdrant vector target type coverage",
    )
    qdrant_vector_case_group_thresholds = parse_named_float_thresholds(
        min_qdrant_vector_case_group_target_coverage,
        "Qdrant vector case group target coverage",
    )
    qdrant_reranker_source_family_thresholds = parse_named_float_thresholds(
        min_qdrant_reranker_source_family_target_coverage,
        "Qdrant reranker source family target coverage",
    )
    qdrant_reranker_source_thresholds = parse_named_float_thresholds(
        min_qdrant_reranker_source_target_coverage,
        "Qdrant reranker source target coverage",
    )
    qdrant_reranker_target_type_thresholds = parse_named_float_thresholds(
        min_qdrant_reranker_target_type_coverage,
        "Qdrant reranker target type coverage",
    )
    qdrant_reranker_case_group_thresholds = parse_named_float_thresholds(
        min_qdrant_reranker_case_group_target_coverage,
        "Qdrant reranker case group target coverage",
    )
    retrieval_source_family_thresholds = parse_named_float_thresholds(
        min_retrieval_source_family_target_coverage,
        "retrieval source family target coverage",
    )
    retrieval_source_thresholds = parse_named_float_thresholds(
        min_retrieval_source_target_coverage,
        "retrieval source target coverage",
    )
    retrieval_target_type_thresholds = parse_named_float_thresholds(
        min_retrieval_target_type_coverage,
        "retrieval target type coverage",
    )
    retrieval_case_group_thresholds = parse_named_float_thresholds(
        min_retrieval_case_group_target_coverage,
        "retrieval case group target coverage",
    )
    rag_context_target_type_thresholds = parse_named_float_thresholds(
        min_rag_context_target_type_coverage,
        "RAG context target type coverage",
    )
    rag_context_case_group_thresholds = parse_named_float_thresholds(
        min_rag_context_case_group_target_coverage,
        "RAG context case group target coverage",
    )
    rag_context_case_group_excluded_thresholds = parse_named_float_thresholds(
        max_rag_context_case_group_excluded_target_hit_rate,
        "RAG context case group excluded-target hit rate",
    )
    retrieval_case_group_count_thresholds = parse_named_int_thresholds(
        min_retrieval_case_group_count,
        "retrieval case group count",
    )
    retrieval_case_group_distinct_thresholds = parse_named_int_thresholds(
        min_retrieval_case_group_distinct_targets,
        "retrieval case group distinct target count",
    )
    chunking_source_family_thresholds = parse_named_float_thresholds(
        min_chunking_source_family_target_coverage,
        "chunking source family target coverage",
    )
    chunking_target_type_thresholds = parse_named_float_thresholds(
        min_chunking_target_type_coverage,
        "chunking target type coverage",
    )
    chunking_case_group_thresholds = parse_named_float_thresholds(
        min_chunking_case_group_target_coverage,
        "chunking case group target coverage",
    )
    retrieval_ablation_source_family_thresholds = parse_named_float_thresholds(
        min_retrieval_ablation_source_family_target_coverage,
        "retrieval ablation source family target coverage",
    )
    retrieval_ablation_source_thresholds = parse_named_float_thresholds(
        min_retrieval_ablation_source_target_coverage,
        "retrieval ablation source target coverage",
    )
    retrieval_ablation_target_type_thresholds = parse_named_float_thresholds(
        min_retrieval_ablation_target_type_coverage,
        "retrieval ablation target type coverage",
    )
    retrieval_ablation_case_group_thresholds = parse_named_float_thresholds(
        min_retrieval_ablation_case_group_target_coverage,
        "retrieval ablation case group target coverage",
    )
    report = build_ingestion_readiness_report(
        package_dir=package_dir,
        manifest=manifest,
        require_qdrant_records=require_qdrant_records,
        require_bm25=require_bm25,
        require_embedding_manifest=require_embedding_manifest,
        required_vectors=required_vectors,
        require_derived_vector_coverage=require_derived_vector_coverage,
        require_postgres_rows=require_postgres_rows,
        runtime_report=parsed_runtime_report,
        require_runtime_report=require_runtime_report,
        require_visual_annotations=require_visual_annotations,
        require_visual_derived_triples=require_visual_derived_triples,
        min_visual_text_coverage_ratio=min_visual_text_coverage_ratio,
        min_visual_text_part_coverage_ratio=min_visual_text_part_coverage_ratio,
        visual_results=parsed_visual_results,
        require_visual_quality=require_visual_quality,
        visual_quality_options={
            "min_completion_rate": min_visual_completion_rate,
            "min_ocr_text_coverage": min_ocr_text_coverage,
            "min_vlm_summary_coverage": min_vlm_summary_coverage,
            "min_vlm_json_parse_rate": min_vlm_json_parse_rate,
            "min_vlm_object_coverage": min_vlm_object_coverage,
            "min_objects_per_vlm_job": min_objects_per_vlm_job,
            "min_object_bbox_coverage": min_object_bbox_coverage,
            "max_failed_count": max_visual_failed_count,
        },
        visual_run_comparison=parsed_visual_run_comparison,
        require_visual_run_comparison=require_visual_run_comparison,
        visual_run_comparison_options={
            "min_run_count": min_visual_run_count,
            "require_same_jobs": require_visual_run_same_jobs,
            "min_shared_job_count": min_visual_run_shared_jobs,
            "expected_best_by_quality": visual_run_best_by_quality,
            "expected_best_by_triple_density": visual_run_best_by_triple_density,
            "expected_best_by_retrieval": visual_run_best_by_retrieval,
        },
        retrieval_cases=parsed_retrieval_cases,
        require_retrieval_cases=require_retrieval_cases,
        retrieval_case_options={
            "min_case_count": min_case_count,
            "min_page_cases": min_page_cases,
            "min_chunk_cases": min_chunk_cases,
            "min_asset_cases": min_asset_cases,
            "min_triple_cases": min_triple_cases,
            "min_distinct_page_targets": min_retrieval_distinct_page_targets,
            "min_distinct_chunk_targets": min_retrieval_distinct_chunk_targets,
            "min_distinct_asset_targets": min_retrieval_distinct_asset_targets,
            "min_distinct_triple_targets": min_retrieval_distinct_triple_targets,
            "max_page_cases_per_target": max_retrieval_page_cases_per_target,
            "max_chunk_cases_per_target": max_retrieval_chunk_cases_per_target,
            "max_asset_cases_per_target": max_retrieval_asset_cases_per_target,
            "max_triple_cases_per_target": max_retrieval_triple_cases_per_target,
            "min_case_group_counts": retrieval_case_group_count_thresholds,
            "min_case_group_distinct_targets": retrieval_case_group_distinct_thresholds,
            "require_visual_only_object_probes": require_visual_only_object_probes,
            "min_query_terms_per_case": min_retrieval_query_terms_per_case,
            "max_target_query_overlap_ratio": max_retrieval_target_query_overlap_ratio,
            "max_target_query_overlap_terms": max_retrieval_target_query_overlap_terms,
            "min_terms_for_target_overlap": min_retrieval_terms_for_target_overlap,
            "max_expected_targets_per_case": max_retrieval_expected_targets_per_case,
            "max_duplicate_queries": max_duplicate_queries,
        },
        retrieval_evaluation=parsed_retrieval,
        require_retrieval_evaluation=require_retrieval_evaluation,
        retrieval_gate_options={
            "min_recall_at_k": min_recall_at_k,
            "min_target_coverage_at_k": min_target_coverage_at_k,
            "min_target_ndcg_at_k": min_target_ndcg_at_k,
            "min_precision_at_k": min_precision_at_k,
            "max_failed_query_count": max_retrieval_failed_queries,
            "max_mean_first_relevant_rank": max_mean_first_relevant_rank,
            "max_p95_first_relevant_rank": max_p95_first_relevant_rank,
            "max_mean_target_rank": max_mean_target_rank,
            "max_p95_target_rank": max_p95_target_rank,
            "max_p95_latency_ms": max_p95_latency_ms,
            "min_result_stability_rate": min_result_stability_rate,
            "max_unstable_result_count": max_unstable_result_count,
            "min_target_type_coverage": retrieval_target_type_thresholds,
            "min_source_target_coverage": retrieval_source_thresholds,
            "min_source_family_target_coverage": retrieval_source_family_thresholds,
            "min_case_group_target_coverage": retrieval_case_group_thresholds,
        },
        qdrant_retrieval_config=parsed_qdrant_retrieval_config,
        require_qdrant_retrieval_config=require_qdrant_retrieval_config,
        rag_context_evaluation=parsed_rag_context,
        require_rag_context_evaluation=require_rag_context_evaluation,
        rag_context_gate_options={
            "min_case_count": min_rag_context_case_count,
            "min_expected_case_count": min_rag_context_expected_case_count,
            "min_expected_target_count": min_rag_context_expected_target_count,
            "min_passed_case_count": min_rag_context_passed_case_count,
            "max_failed_case_count": max_rag_context_failed_cases,
            "min_hit_rate": min_rag_context_hit_rate,
            "min_target_coverage": min_rag_context_target_coverage,
            "max_excluded_target_hit_rate": max_rag_context_excluded_target_hit_rate,
            "max_excluded_query_hit_rate": max_rag_context_excluded_query_hit_rate,
            "max_excluded_hit_query_count": max_rag_context_excluded_hit_query_count,
            "max_mean_latency_ms": max_rag_context_mean_latency_ms,
            "max_mean_context_char_count": max_rag_context_mean_context_char_count,
            "max_context_char_count": max_rag_context_char_count,
            "max_mean_chunk_count": max_rag_context_mean_chunk_count,
            "max_mean_asset_count": max_rag_context_mean_asset_count,
            "max_mean_triple_count": max_rag_context_mean_triple_count,
            "min_target_type_coverage": rag_context_target_type_thresholds,
            "min_case_group_target_coverage": rag_context_case_group_thresholds,
            "max_case_group_excluded_target_hit_rate": (
                rag_context_case_group_excluded_thresholds
            ),
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
            "max_mean_first_relevant_rank": max_chunking_mean_first_relevant_rank,
            "max_p95_first_relevant_rank": max_chunking_p95_first_relevant_rank,
            "max_mean_target_rank": max_chunking_mean_target_rank,
            "max_p95_target_rank": max_chunking_p95_target_rank,
            "min_result_stability_rate": min_chunking_result_stability_rate,
            "max_unstable_result_count": max_chunking_unstable_result_count,
            "max_total_chunk_chars": max_chunking_total_chunk_chars,
            "max_embedding_text_kchars": max_chunking_embedding_text_kchars,
            "min_visual_text_coverage_ratio": min_chunking_visual_text_coverage_ratio,
            "min_visual_text_part_coverage_ratio": min_chunking_visual_text_part_coverage_ratio,
            "min_retrieval_score_per_embedding_kchar": (
                min_chunking_retrieval_score_per_embedding_kchar
            ),
            "min_target_coverage_per_embedding_kchar": (
                min_chunking_target_coverage_per_embedding_kchar
            ),
            "min_target_ndcg_per_embedding_kchar": (
                min_chunking_target_ndcg_per_embedding_kchar
            ),
            "min_retrieval_score_per_mean_latency_ms": (
                min_chunking_retrieval_score_per_mean_latency_ms
            ),
            "min_target_coverage_per_mean_latency_ms": (
                min_chunking_target_coverage_per_mean_latency_ms
            ),
            "min_target_ndcg_per_mean_latency_ms": (
                min_chunking_target_ndcg_per_mean_latency_ms
            ),
            "min_retrieval_score_per_p95_latency_ms": (
                min_chunking_retrieval_score_per_p95_latency_ms
            ),
            "min_target_coverage_per_p95_latency_ms": (
                min_chunking_target_coverage_per_p95_latency_ms
            ),
            "min_target_ndcg_per_p95_latency_ms": (
                min_chunking_target_ndcg_per_p95_latency_ms
            ),
            "max_failed_queries": max_chunking_failed_queries,
            "max_recall_drop": max_chunking_recall_drop,
            "max_mean_latency_ratio": max_chunking_mean_latency_ratio,
            "max_pairwise_mean_first_relevant_rank_delta": (
                max_chunking_pairwise_mean_first_relevant_rank_delta
            ),
            "max_pairwise_mean_target_rank_delta": (
                max_chunking_pairwise_mean_target_rank_delta
            ),
            "max_pairwise_first_relevant_rank_delta_ci_high": (
                max_chunking_pairwise_first_relevant_rank_delta_ci_high
            ),
            "max_pairwise_target_rank_delta_ci_high": (
                max_chunking_pairwise_target_rank_delta_ci_high
            ),
            "min_target_type_coverage": chunking_target_type_thresholds,
            "min_source_family_target_coverage": chunking_source_family_thresholds,
            "min_case_group_target_coverage": chunking_case_group_thresholds,
        },
        retrieval_ablation=parsed_retrieval_ablation,
        require_retrieval_ablation=require_retrieval_ablation,
        retrieval_ablation_mode=retrieval_ablation_mode,
        retrieval_ablation_baseline_mode=retrieval_ablation_baseline_mode,
        retrieval_ablation_gate_options={
            "min_recall_at_k": min_retrieval_ablation_recall_at_k,
            "min_target_coverage_at_k": min_retrieval_ablation_target_coverage_at_k,
            "min_target_ndcg_at_k": min_retrieval_ablation_target_ndcg_at_k,
            "min_mrr": min_retrieval_ablation_mrr,
            "min_precision_at_k": min_retrieval_ablation_precision_at_k,
            "max_failed_queries": max_retrieval_ablation_failed_queries,
            "max_mean_first_relevant_rank": (
                max_retrieval_ablation_mean_first_relevant_rank
            ),
            "max_p95_first_relevant_rank": max_retrieval_ablation_p95_first_relevant_rank,
            "max_mean_target_rank": max_retrieval_ablation_mean_target_rank,
            "max_p95_target_rank": max_retrieval_ablation_p95_target_rank,
            "max_mean_latency_ms": max_retrieval_ablation_mean_latency_ms,
            "max_p95_latency_ms": max_retrieval_ablation_p95_latency_ms,
            "min_target_type_coverage": retrieval_ablation_target_type_thresholds,
            "min_source_target_coverage": retrieval_ablation_source_thresholds,
            "min_source_family_target_coverage": retrieval_ablation_source_family_thresholds,
            "min_case_group_target_coverage": retrieval_ablation_case_group_thresholds,
            "min_recall_lift": min_retrieval_ablation_recall_lift,
            "min_target_coverage_lift": min_retrieval_ablation_target_coverage_lift,
            "min_target_ndcg_lift": min_retrieval_ablation_target_ndcg_lift,
            "min_mrr_lift": min_retrieval_ablation_mrr_lift,
            "min_precision_lift": min_retrieval_ablation_precision_lift,
            "max_mean_latency_ratio": max_retrieval_ablation_mean_latency_ratio,
            "max_p95_latency_ratio": max_retrieval_ablation_p95_latency_ratio,
            "min_pairwise_shared_queries": (
                min_retrieval_ablation_pairwise_shared_queries
            ),
            "min_pairwise_win_rate": min_retrieval_ablation_pairwise_win_rate,
            "min_pairwise_target_coverage_lift": (
                min_retrieval_ablation_pairwise_target_coverage_lift
            ),
            "min_pairwise_target_ndcg_lift": (
                min_retrieval_ablation_pairwise_target_ndcg_lift
            ),
            "min_pairwise_mrr_lift": min_retrieval_ablation_pairwise_mrr_lift,
            "min_pairwise_precision_lift": (
                min_retrieval_ablation_pairwise_precision_lift
            ),
            "min_pairwise_target_coverage_ci_low": (
                min_retrieval_ablation_pairwise_target_coverage_ci_low
            ),
            "min_pairwise_target_ndcg_ci_low": (
                min_retrieval_ablation_pairwise_target_ndcg_ci_low
            ),
            "min_pairwise_mrr_ci_low": min_retrieval_ablation_pairwise_mrr_ci_low,
            "min_pairwise_precision_ci_low": (
                min_retrieval_ablation_pairwise_precision_ci_low
            ),
            "max_pairwise_mean_first_relevant_rank_delta": (
                max_retrieval_ablation_pairwise_mean_first_relevant_rank_delta
            ),
            "max_pairwise_mean_target_rank_delta": (
                max_retrieval_ablation_pairwise_mean_target_rank_delta
            ),
            "max_pairwise_first_relevant_rank_delta_ci_high": (
                max_retrieval_ablation_pairwise_first_relevant_rank_delta_ci_high
            ),
            "max_pairwise_target_rank_delta_ci_high": (
                max_retrieval_ablation_pairwise_target_rank_delta_ci_high
            ),
            "max_pairwise_mean_latency_delta_ms": (
                max_retrieval_ablation_pairwise_mean_latency_delta_ms
            ),
            "require_best_by_recall": require_retrieval_ablation_best_by_recall,
            "require_best_by_target_coverage": (
                require_retrieval_ablation_best_by_target_coverage
            ),
            "require_best_by_target_ndcg": require_retrieval_ablation_best_by_target_ndcg,
            "require_fastest_by_mean_latency": (
                require_retrieval_ablation_fastest_by_mean_latency
            ),
        },
        qdrant_vector_ablation=parsed_qdrant_vector_ablation,
        require_qdrant_vector_ablation=require_qdrant_vector_ablation,
        qdrant_vector_ablation_mode=qdrant_vector_mode,
        qdrant_vector_ablation_gate_options={
            "baseline_mode": qdrant_vector_baseline_mode,
            "min_pairwise_shared_queries": min_qdrant_vector_pairwise_shared_queries,
            "min_pairwise_win_rate": min_qdrant_vector_pairwise_win_rate,
            "min_pairwise_target_coverage_lift": (
                min_qdrant_vector_pairwise_target_coverage_lift
            ),
            "min_pairwise_target_ndcg_lift": (
                min_qdrant_vector_pairwise_target_ndcg_lift
            ),
            "min_pairwise_mrr_lift": min_qdrant_vector_pairwise_mrr_lift,
            "min_pairwise_precision_lift": min_qdrant_vector_pairwise_precision_lift,
            "min_pairwise_target_coverage_ci_low": (
                min_qdrant_vector_pairwise_target_coverage_ci_low
            ),
            "min_pairwise_target_ndcg_ci_low": (
                min_qdrant_vector_pairwise_target_ndcg_ci_low
            ),
            "min_pairwise_mrr_ci_low": min_qdrant_vector_pairwise_mrr_ci_low,
            "min_pairwise_precision_ci_low": min_qdrant_vector_pairwise_precision_ci_low,
            "max_pairwise_mean_first_relevant_rank_delta": (
                max_qdrant_vector_pairwise_mean_first_relevant_rank_delta
            ),
            "max_pairwise_mean_target_rank_delta": (
                max_qdrant_vector_pairwise_mean_target_rank_delta
            ),
            "max_pairwise_first_relevant_rank_delta_ci_high": (
                max_qdrant_vector_pairwise_first_relevant_rank_delta_ci_high
            ),
            "max_pairwise_target_rank_delta_ci_high": (
                max_qdrant_vector_pairwise_target_rank_delta_ci_high
            ),
            "max_pairwise_mean_latency_delta_ms": (
                max_qdrant_vector_pairwise_mean_latency_delta_ms
            ),
            "min_recall_at_k": min_qdrant_vector_recall_at_k,
            "min_target_coverage_at_k": min_qdrant_vector_target_coverage_at_k,
            "min_target_ndcg_at_k": min_qdrant_vector_target_ndcg_at_k,
            "min_mrr": min_qdrant_vector_mrr,
            "min_precision_at_k": min_qdrant_vector_precision_at_k,
            "max_failed_queries": max_qdrant_vector_failed_queries,
            "max_mean_first_relevant_rank": max_qdrant_vector_mean_first_relevant_rank,
            "max_p95_first_relevant_rank": max_qdrant_vector_p95_first_relevant_rank,
            "max_mean_target_rank": max_qdrant_vector_mean_target_rank,
            "max_p95_target_rank": max_qdrant_vector_p95_target_rank,
            "max_mean_latency_ms": max_qdrant_vector_mean_latency_ms,
            "max_p95_latency_ms": max_qdrant_vector_p95_latency_ms,
            "min_target_type_coverage": qdrant_vector_target_type_thresholds,
            "min_source_target_coverage": qdrant_vector_source_thresholds,
            "min_source_family_target_coverage": qdrant_vector_source_family_thresholds,
            "min_case_group_target_coverage": qdrant_vector_case_group_thresholds,
            "require_best_by_recall": require_qdrant_vector_best_by_recall,
            "require_best_by_target_coverage": require_qdrant_vector_best_by_target_coverage,
            "require_best_by_target_ndcg": require_qdrant_vector_best_by_target_ndcg,
            "require_fastest_by_mean_latency": require_qdrant_vector_fastest_by_mean_latency,
        },
        qdrant_reranker_ablation=parsed_qdrant_reranker_ablation,
        require_qdrant_reranker_ablation=require_qdrant_reranker_ablation,
        qdrant_reranker_ablation_mode=qdrant_reranker_mode,
        qdrant_reranker_ablation_gate_options={
            "baseline_mode": qdrant_reranker_baseline_mode,
            "min_pairwise_shared_queries": min_qdrant_reranker_pairwise_shared_queries,
            "min_pairwise_win_rate": min_qdrant_reranker_pairwise_win_rate,
            "min_pairwise_target_coverage_lift": (
                min_qdrant_reranker_pairwise_target_coverage_lift
            ),
            "min_pairwise_target_ndcg_lift": (
                min_qdrant_reranker_pairwise_target_ndcg_lift
            ),
            "min_pairwise_mrr_lift": min_qdrant_reranker_pairwise_mrr_lift,
            "min_pairwise_precision_lift": min_qdrant_reranker_pairwise_precision_lift,
            "min_pairwise_target_coverage_ci_low": (
                min_qdrant_reranker_pairwise_target_coverage_ci_low
            ),
            "min_pairwise_target_ndcg_ci_low": (
                min_qdrant_reranker_pairwise_target_ndcg_ci_low
            ),
            "min_pairwise_mrr_ci_low": min_qdrant_reranker_pairwise_mrr_ci_low,
            "min_pairwise_precision_ci_low": min_qdrant_reranker_pairwise_precision_ci_low,
            "max_pairwise_mean_first_relevant_rank_delta": (
                max_qdrant_reranker_pairwise_mean_first_relevant_rank_delta
            ),
            "max_pairwise_mean_target_rank_delta": (
                max_qdrant_reranker_pairwise_mean_target_rank_delta
            ),
            "max_pairwise_first_relevant_rank_delta_ci_high": (
                max_qdrant_reranker_pairwise_first_relevant_rank_delta_ci_high
            ),
            "max_pairwise_target_rank_delta_ci_high": (
                max_qdrant_reranker_pairwise_target_rank_delta_ci_high
            ),
            "max_pairwise_mean_latency_delta_ms": (
                max_qdrant_reranker_pairwise_mean_latency_delta_ms
            ),
            "min_recall_at_k": min_qdrant_reranker_recall_at_k,
            "min_target_coverage_at_k": min_qdrant_reranker_target_coverage_at_k,
            "min_target_ndcg_at_k": min_qdrant_reranker_target_ndcg_at_k,
            "min_mrr": min_qdrant_reranker_mrr,
            "min_precision_at_k": min_qdrant_reranker_precision_at_k,
            "max_failed_queries": max_qdrant_reranker_failed_queries,
            "max_mean_first_relevant_rank": max_qdrant_reranker_mean_first_relevant_rank,
            "max_p95_first_relevant_rank": max_qdrant_reranker_p95_first_relevant_rank,
            "max_mean_target_rank": max_qdrant_reranker_mean_target_rank,
            "max_p95_target_rank": max_qdrant_reranker_p95_target_rank,
            "max_mean_latency_ms": max_qdrant_reranker_mean_latency_ms,
            "max_p95_latency_ms": max_qdrant_reranker_p95_latency_ms,
            "min_target_type_coverage": qdrant_reranker_target_type_thresholds,
            "min_source_target_coverage": qdrant_reranker_source_thresholds,
            "min_source_family_target_coverage": qdrant_reranker_source_family_thresholds,
            "min_case_group_target_coverage": qdrant_reranker_case_group_thresholds,
            "require_best_by_recall": require_qdrant_reranker_best_by_recall,
            "require_best_by_target_coverage": (
                require_qdrant_reranker_best_by_target_coverage
            ),
            "require_best_by_target_ndcg": require_qdrant_reranker_best_by_target_ndcg,
            "require_fastest_by_mean_latency": (
                require_qdrant_reranker_fastest_by_mean_latency
            ),
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


@app.command(name="plan-ingestion-workflow")
def plan_ingestion_workflow_command(
    package_dir: Path = Path("outputs/package"),
    output: Path | None = None,
    retrieval_cases: Path = Path("examples/retrieval_cases.jsonl"),
    vlm_profiles: str = "qwen2_5_vl_7b,qwen2_vl_7b,llava_next_7b",
    max_pages: int = 25,
):
    """Write an ordered command plan from package characterization to final ingestion readiness."""
    manifest = load_processing_package(package_dir)
    characteristics = characterize_package(
        profiles=manifest.profiles,
        chunks=manifest.chunks,
        assets=manifest.assets,
        triples=manifest.triples,
        package_dir=package_dir,
        max_pages=max_pages,
    )
    plan = build_ingestion_workflow_plan(
        characteristics,
        package_dir=package_dir,
        retrieval_cases=retrieval_cases,
        vlm_profiles=parse_profile_list(vlm_profiles),
    )
    payload = plan.model_dump()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "output": str(output),
            "step_count": len(plan.steps),
            "required_step_count": plan.metadata.get("required_step_count", 0),
            "observation_codes": plan.observation_codes,
            "recommendation_codes": plan.recommendation_codes,
            "step_ids": [step.step_id for step in plan.steps],
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
    deduplicate_tokens: bool = False,
):
    """Evaluate local hybrid retrieval against JSONL seed cases."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
    evaluation = evaluate_retrieval(
        chunks=chunks,
        assets=assets,
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
                "excluded_query_count": evaluation.excluded_query_count,
                "excluded_hit_query_count": evaluation.excluded_hit_query_count,
                "excluded_query_hit_rate": evaluation.excluded_query_hit_rate,
                "excluded_target_hit_rate": evaluation.excluded_target_hit_rate,
                "mean_latency_ms": evaluation.mean_latency_ms,
                "p95_latency_ms": evaluation.p95_latency_ms,
                "unstable_result_count": evaluation.unstable_result_count,
                "result_stability_rate": evaluation.result_stability_rate,
                "target_metrics": retrieval_target_metrics_payload(evaluation),
                "source_metrics": retrieval_source_metrics_payload(evaluation),
                "source_family_metrics": retrieval_source_family_metrics_payload(evaluation),
                "chunk_strategy_metrics": retrieval_chunk_strategy_metrics_payload(evaluation),
                "retrieval_role_metrics": retrieval_role_metrics_payload(evaluation),
                "case_group_metrics": retrieval_case_group_metrics_payload(evaluation),
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
    visual_probe_limit: int = 0,
    image_probe_limit: int = 0,
    object_probe_limit: int = 0,
    object_probe_visual_only: bool = typer.Option(
        True,
        "--object-probe-visual-only/--no-object-probe-visual-only",
        help="Use object probe terms that are not already present in linked chunk text.",
    ),
    max_target_query_overlap_ratio: float | None = typer.Option(
        None,
        "--max-target-query-overlap-ratio",
        help="Skip generated cases whose query terms overlap expected target text above this ratio.",
    ),
    max_target_query_overlap_terms: int | None = typer.Option(
        None,
        "--max-target-query-overlap-terms",
        help="Skip generated cases whose query uses too many distinct terms from expected target text.",
    ),
    min_terms_for_target_overlap: int = typer.Option(
        4,
        "--min-terms-for-target-overlap",
        help="Only apply generated-case target overlap filtering to queries with this many terms.",
    ),
    max_page_cases_per_target: int | None = typer.Option(
        None,
        "--max-page-cases-per-target",
        help="Skip generated cases that would exceed this page-target concentration limit.",
    ),
    max_chunk_cases_per_target: int | None = typer.Option(
        None,
        "--max-chunk-cases-per-target",
        help="Skip generated cases that would exceed this chunk-target concentration limit.",
    ),
    max_asset_cases_per_target: int | None = typer.Option(
        None,
        "--max-asset-cases-per-target",
        help="Skip generated cases that would exceed this visual-asset target concentration limit.",
    ),
    max_triple_cases_per_target: int | None = typer.Option(
        None,
        "--max-triple-cases-per-target",
        help="Skip generated cases that would exceed this graph-triple target concentration limit.",
    ),
    hard_negative_limit: int = typer.Option(
        0,
        "--hard-negative-limit",
        help=(
            "Attach up to this many same-kind similar-but-wrong page/chunk/asset/triple targets "
            "to each generated case as excluded targets."
        ),
    ),
    hard_negative_min_overlap_terms: int = typer.Option(
        2,
        "--hard-negative-min-overlap-terms",
        help="Require this many shared target-text terms before a candidate is used as a hard negative.",
    ),
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
            visual_probe_limit=visual_probe_limit,
            image_probe_limit=image_probe_limit,
            object_probe_limit=object_probe_limit,
            object_probe_visual_only=object_probe_visual_only,
            max_target_query_overlap_ratio=max_target_query_overlap_ratio,
            max_target_query_overlap_terms=max_target_query_overlap_terms,
            min_terms_for_target_overlap=min_terms_for_target_overlap,
            max_page_cases_per_target=max_page_cases_per_target,
            max_chunk_cases_per_target=max_chunk_cases_per_target,
            max_asset_cases_per_target=max_asset_cases_per_target,
            max_triple_cases_per_target=max_triple_cases_per_target,
            hard_negative_limit=hard_negative_limit,
            hard_negative_min_overlap_terms=hard_negative_min_overlap_terms,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = output or package_dir / "retrieval_cases.skeleton.jsonl"
    write_jsonl(output_path, cases)
    visual_object_probe_counts = count_visual_object_probes(cases)
    print(
        {
            "output": str(output_path),
            "case_count": len(cases),
            "target_counts": count_retrieval_case_targets(cases),
            "distinct_target_counts": count_retrieval_case_distinct_targets(cases),
            "excluded_target_counts": count_retrieval_case_excluded_targets(cases),
            "excluded_distinct_target_counts": count_retrieval_case_distinct_excluded_targets(cases),
            "max_cases_per_target": count_retrieval_case_max_target_mentions(cases),
            "case_group_counts": count_case_groups(cases),
            "case_group_distinct_target_counts": count_case_group_distinct_targets(cases),
            "visual_image_probe_count": count_visual_image_probes(cases),
            "visual_object_probe_count": visual_object_probe_counts["total"],
            "visual_only_object_probe_count": visual_object_probe_counts["visual_only"],
            "non_visual_only_object_probe_count": visual_object_probe_counts["non_visual_only"],
            "page_limit": page_limit,
            "asset_limit": asset_limit,
            "triple_limit": triple_limit,
            "visual_probe_limit": visual_probe_limit,
            "image_probe_limit": image_probe_limit,
            "object_probe_limit": object_probe_limit,
            "object_probe_visual_only": object_probe_visual_only,
            "max_target_query_overlap_ratio": max_target_query_overlap_ratio,
            "max_target_query_overlap_terms": max_target_query_overlap_terms,
            "min_terms_for_target_overlap": min_terms_for_target_overlap,
            "max_page_cases_per_target": max_page_cases_per_target,
            "max_chunk_cases_per_target": max_chunk_cases_per_target,
            "max_asset_cases_per_target": max_asset_cases_per_target,
            "max_triple_cases_per_target": max_triple_cases_per_target,
            "hard_negative_limit": hard_negative_limit,
            "hard_negative_min_overlap_terms": hard_negative_min_overlap_terms,
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
    min_distinct_page_targets: int = 0,
    min_distinct_chunk_targets: int = 0,
    min_distinct_asset_targets: int = 0,
    min_distinct_triple_targets: int = 0,
    max_page_cases_per_target: int | None = None,
    max_chunk_cases_per_target: int | None = None,
    max_asset_cases_per_target: int | None = None,
    max_triple_cases_per_target: int | None = None,
    min_query_terms_per_case: int = 0,
    max_target_query_overlap_ratio: float | None = typer.Option(
        None,
        "--max-target-query-overlap-ratio",
        help=(
            "Fail when eligible query terms overlap expected target text above this ratio, "
            "which catches benchmark queries copied too directly from the answer span."
        ),
    ),
    max_target_query_overlap_terms: int | None = typer.Option(
        None,
        "--max-target-query-overlap-terms",
        help=(
            "Fail when a query uses more than this many distinct terms from expected target text."
        ),
    ),
    min_terms_for_target_overlap: int = typer.Option(
        4,
        "--min-terms-for-target-overlap",
        help="Only apply target-query overlap checks to queries with at least this many distinct terms.",
    ),
    max_expected_targets_per_case: int | None = typer.Option(
        None,
        "--max-expected-targets-per-case",
        help=(
            "Fail when one retrieval case declares more expected page/chunk/asset/triple "
            "targets than this ceiling."
        ),
    ),
    min_case_group_distinct_targets: list[str] = typer.Option(
        None,
        "--min-case-group-distinct-targets",
        help=(
            "Require distinct targets inside a case metadata group, such as "
            "case_source:visual_object_probe:asset=4."
        ),
    ),
    min_case_group_count: list[str] = typer.Option(
        None,
        "--min-case-group-count",
        help="Require retrieval case metadata group counts such as case_source:visual_object_probe=5.",
    ),
    require_visual_only_object_probes: bool = typer.Option(
        False,
        "--require-visual-only-object-probes",
        help="Fail when visual_object_probe cases were not generated with visual-only object terms.",
    ),
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
    case_group_thresholds = parse_named_int_thresholds(
        min_case_group_count,
        "case group count",
    )
    case_group_distinct_thresholds = parse_named_int_thresholds(
        min_case_group_distinct_targets,
        "case group distinct target count",
    )
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
        min_distinct_page_targets=min_distinct_page_targets,
        min_distinct_chunk_targets=min_distinct_chunk_targets,
        min_distinct_asset_targets=min_distinct_asset_targets,
        min_distinct_triple_targets=min_distinct_triple_targets,
        max_page_cases_per_target=max_page_cases_per_target,
        max_chunk_cases_per_target=max_chunk_cases_per_target,
        max_asset_cases_per_target=max_asset_cases_per_target,
        max_triple_cases_per_target=max_triple_cases_per_target,
        min_case_group_counts=case_group_thresholds,
        min_case_group_distinct_targets=case_group_distinct_thresholds,
        require_visual_only_object_probes=require_visual_only_object_probes,
        min_query_terms_per_case=min_query_terms_per_case,
        max_target_query_overlap_ratio=max_target_query_overlap_ratio,
        max_target_query_overlap_terms=max_target_query_overlap_terms,
        min_terms_for_target_overlap=min_terms_for_target_overlap,
        max_expected_targets_per_case=max_expected_targets_per_case,
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
            "distinct_target_counts": report.distinct_target_counts,
            "excluded_target_counts": report.excluded_target_counts,
            "excluded_distinct_target_counts": report.excluded_distinct_target_counts,
            "max_cases_per_target": report.max_cases_per_target,
            "case_group_counts": report.case_group_counts,
            "case_group_distinct_target_counts": report.case_group_distinct_target_counts,
            "visual_object_probe_count": report.visual_object_probe_count,
            "visual_only_object_probe_count": report.visual_only_object_probe_count,
            "non_visual_only_object_probe_count": report.non_visual_only_object_probe_count,
            "short_query_count": report.short_query_count,
            "min_query_term_count": report.min_query_term_count,
            "max_query_term_count": report.max_query_term_count,
            "target_query_overlap_count": report.target_query_overlap_count,
            "target_query_overlap_term_count": report.target_query_overlap_term_count,
            "max_target_query_overlap_ratio": report.max_target_query_overlap_ratio,
            "mean_target_query_overlap_ratio": report.mean_target_query_overlap_ratio,
            "max_target_query_overlap_terms": report.max_target_query_overlap_terms,
            "mean_target_query_overlap_terms": report.mean_target_query_overlap_terms,
            "max_expected_targets_per_case": report.max_expected_targets_per_case,
            "oversized_expected_target_case_count": report.oversized_expected_target_case_count,
            "missing_target_counts": report.missing_target_counts,
            "excluded_missing_target_counts": report.excluded_missing_target_counts,
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
    deduplicate_tokens: bool = False,
):
    """Compare dense, BM25, graph, and fused retrieval on the same cases."""
    fusion_weights = parse_fusion_weights(fusion_weight)
    chunks = read_jsonl(package_dir / chunks_file, DocumentChunk)
    triples_path = package_dir / "triples.jsonl"
    triples = read_jsonl(triples_path, GraphTriple) if triples_path.exists() else []
    assets_path = package_dir / "assets.jsonl"
    assets = read_jsonl(assets_path, VisualAsset) if assets_path.exists() else []
    try:
        parsed_modes = parse_ablation_modes(modes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = evaluate_retrieval_ablation(
        chunks=chunks,
        triples=triples,
        cases=load_retrieval_cases(cases),
        assets=assets,
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
            deduplicate_tokens=deduplicate_tokens,
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
            "case_group_best_modes": report.case_group_best_modes,
            "pairwise": [comparison.model_dump() for comparison in report.pairwise],
            "rows": [
                {
                    "mode": row.mode.name,
                    "recall_at_k": row.evaluation.recall_at_k,
                    "mrr": row.evaluation.mrr,
                    "hit_rate": row.evaluation.hit_rate,
                    "target_coverage_at_k": row.evaluation.target_coverage_at_k,
                    "mean_target_ndcg_at_k": row.evaluation.mean_target_ndcg_at_k,
                    "mean_precision_at_k": row.evaluation.mean_precision_at_k,
                    "excluded_query_count": row.evaluation.excluded_query_count,
                    "excluded_hit_query_count": row.evaluation.excluded_hit_query_count,
                    "excluded_query_hit_rate": row.evaluation.excluded_query_hit_rate,
                    "excluded_target_count": row.evaluation.excluded_target_count,
                    "excluded_matched_target_count": (
                        row.evaluation.excluded_matched_target_count
                    ),
                    "excluded_target_hit_rate": row.evaluation.excluded_target_hit_rate,
                    "repeat": row.evaluation.repeat,
                    "mean_latency_ms": row.evaluation.mean_latency_ms,
                    "p95_latency_ms": row.evaluation.p95_latency_ms,
                    "unstable_result_count": row.evaluation.unstable_result_count,
                    "result_stability_rate": row.evaluation.result_stability_rate,
                    "target_metrics": retrieval_target_metrics_payload(row.evaluation),
                    "source_metrics": retrieval_source_metrics_payload(row.evaluation),
                    "source_family_metrics": retrieval_source_family_metrics_payload(row.evaluation),
                    "chunk_strategy_metrics": retrieval_chunk_strategy_metrics_payload(
                        row.evaluation
                    ),
                    "retrieval_role_metrics": retrieval_role_metrics_payload(row.evaluation),
                    "case_group_metrics": retrieval_case_group_metrics_payload(row.evaluation),
                    "failed_queries": row.evaluation.failed_queries,
                }
                for row in report.rows
            ],
        }
    print(payload)


@app.command(name="gate-retrieval-ablation")
def gate_retrieval_ablation_command(
    report: Path,
    mode: str = typer.Option(
        ...,
        "--mode",
        help="Ablation mode to gate, such as bm25_visual or hybrid_graph.",
    ),
    baseline_mode: str | None = typer.Option(
        None,
        "--baseline-mode",
        help="Optional baseline mode for lift or latency-ratio checks.",
    ),
    output: Path | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_excluded_target_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-target-hit-rate",
        help="Limit selected mode explicit excluded page/chunk/asset/triple target hit rate.",
    ),
    max_excluded_query_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-query-hit-rate",
        help="Limit selected mode hard-negative query hit rate.",
    ),
    max_excluded_hit_query_count: int | None = typer.Option(
        None,
        "--max-excluded-hit-query-count",
        help="Limit selected mode hard-negative hit query count.",
    ),
    min_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-target-type-coverage",
        help="Require selected mode target-type coverage such as asset=1.0.",
    ),
    min_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-target-coverage",
        help="Require selected mode exact-source target coverage such as bm25=1.0.",
    ),
    min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-family-target-coverage",
        help="Require selected mode source-family target coverage such as lexical=0.8.",
    ),
    max_source_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-excluded-target-hit-rate",
        help="Limit selected mode exact-source excluded-target hit rate such as bm25=0.0.",
    ),
    max_source_family_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-family-excluded-target-hit-rate",
        help="Limit selected mode source-family excluded-target hit rate such as lexical=0.0.",
    ),
    max_chunk_strategy_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-chunk-strategy-excluded-target-hit-rate",
        help=(
            "Limit selected mode chunking-strategy excluded-target hit rate such as "
            "visual_asset_text=0.0."
        ),
    ),
    max_retrieval_role_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-retrieval-role-excluded-target-hit-rate",
        help="Limit selected mode retrieval-role excluded-target hit rate such as child=0.0.",
    ),
    min_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-case-group-target-coverage",
        help=(
            "Require selected mode metadata case-group target coverage such as "
            "case_source:visual_object_probe=0.7."
        ),
    ),
    min_recall_lift: float | None = None,
    min_target_coverage_lift: float | None = None,
    min_target_ndcg_lift: float | None = None,
    min_mrr_lift: float | None = None,
    min_precision_lift: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
    min_pairwise_shared_queries: int | None = None,
    min_pairwise_win_rate: float | None = None,
    min_pairwise_target_coverage_lift: float | None = None,
    min_pairwise_target_ndcg_lift: float | None = None,
    min_pairwise_mrr_lift: float | None = None,
    min_pairwise_precision_lift: float | None = None,
    min_pairwise_target_coverage_ci_low: float | None = None,
    min_pairwise_target_ndcg_ci_low: float | None = None,
    min_pairwise_mrr_ci_low: float | None = None,
    min_pairwise_precision_ci_low: float | None = None,
    max_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_pairwise_mean_target_rank_delta: float | None = None,
    max_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_pairwise_target_rank_delta_ci_high: float | None = None,
    max_pairwise_mean_latency_delta_ms: float | None = None,
    require_best_by_recall: bool = False,
    require_best_by_target_coverage: bool = False,
    require_best_by_target_ndcg: bool = False,
    require_fastest_by_mean_latency: bool = False,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when the retrieval ablation gate fails.",
    ),
):
    """Fail a retrieval ablation mode when absolute metrics or baseline lift are too weak."""
    parsed_report = RetrievalAblationReport.model_validate_json(report.read_text(encoding="utf-8"))
    source_family_thresholds = parse_named_float_thresholds(
        min_source_family_target_coverage,
        "source family target coverage",
    )
    source_thresholds = parse_named_float_thresholds(
        min_source_target_coverage,
        "source target coverage",
    )
    source_excluded_thresholds = parse_named_float_thresholds(
        max_source_excluded_target_hit_rate,
        "source excluded-target hit rate",
    )
    source_family_excluded_thresholds = parse_named_float_thresholds(
        max_source_family_excluded_target_hit_rate,
        "source family excluded-target hit rate",
    )
    chunk_strategy_excluded_thresholds = parse_named_float_thresholds(
        max_chunk_strategy_excluded_target_hit_rate,
        "chunk strategy excluded-target hit rate",
    )
    retrieval_role_excluded_thresholds = parse_named_float_thresholds(
        max_retrieval_role_excluded_target_hit_rate,
        "retrieval role excluded-target hit rate",
    )
    target_type_thresholds = parse_named_float_thresholds(
        min_target_type_coverage,
        "target type coverage",
    )
    case_group_thresholds = parse_named_float_thresholds(
        min_case_group_target_coverage,
        "case group target coverage",
    )
    try:
        gate_report = gate_retrieval_ablation(
            parsed_report,
            mode=mode,
            baseline_mode=baseline_mode,
            min_recall_at_k=min_recall_at_k,
            min_target_coverage_at_k=min_target_coverage_at_k,
            min_target_ndcg_at_k=min_target_ndcg_at_k,
            min_mrr=min_mrr,
            min_precision_at_k=min_precision_at_k,
            max_failed_queries=max_failed_queries,
            max_mean_first_relevant_rank=max_mean_first_relevant_rank,
            max_p95_first_relevant_rank=max_p95_first_relevant_rank,
            max_mean_target_rank=max_mean_target_rank,
            max_p95_target_rank=max_p95_target_rank,
            max_mean_latency_ms=max_mean_latency_ms,
            max_p95_latency_ms=max_p95_latency_ms,
            max_excluded_target_hit_rate=max_excluded_target_hit_rate,
            max_excluded_query_hit_rate=max_excluded_query_hit_rate,
            max_excluded_hit_query_count=max_excluded_hit_query_count,
            min_target_type_coverage=target_type_thresholds,
            min_source_target_coverage=source_thresholds,
            min_source_family_target_coverage=source_family_thresholds,
            max_source_excluded_target_hit_rate=source_excluded_thresholds,
            max_source_family_excluded_target_hit_rate=source_family_excluded_thresholds,
            max_chunk_strategy_excluded_target_hit_rate=chunk_strategy_excluded_thresholds,
            max_retrieval_role_excluded_target_hit_rate=retrieval_role_excluded_thresholds,
            min_case_group_target_coverage=case_group_thresholds,
            min_recall_lift=min_recall_lift,
            min_target_coverage_lift=min_target_coverage_lift,
            min_target_ndcg_lift=min_target_ndcg_lift,
            min_mrr_lift=min_mrr_lift,
            min_precision_lift=min_precision_lift,
            max_mean_latency_ratio=max_mean_latency_ratio,
            max_p95_latency_ratio=max_p95_latency_ratio,
            min_pairwise_shared_queries=min_pairwise_shared_queries,
            min_pairwise_win_rate=min_pairwise_win_rate,
            min_pairwise_target_coverage_lift=min_pairwise_target_coverage_lift,
            min_pairwise_target_ndcg_lift=min_pairwise_target_ndcg_lift,
            min_pairwise_mrr_lift=min_pairwise_mrr_lift,
            min_pairwise_precision_lift=min_pairwise_precision_lift,
            min_pairwise_target_coverage_ci_low=min_pairwise_target_coverage_ci_low,
            min_pairwise_target_ndcg_ci_low=min_pairwise_target_ndcg_ci_low,
            min_pairwise_mrr_ci_low=min_pairwise_mrr_ci_low,
            min_pairwise_precision_ci_low=min_pairwise_precision_ci_low,
            max_pairwise_mean_first_relevant_rank_delta=(
                max_pairwise_mean_first_relevant_rank_delta
            ),
            max_pairwise_mean_target_rank_delta=max_pairwise_mean_target_rank_delta,
            max_pairwise_first_relevant_rank_delta_ci_high=(
                max_pairwise_first_relevant_rank_delta_ci_high
            ),
            max_pairwise_target_rank_delta_ci_high=max_pairwise_target_rank_delta_ci_high,
            max_pairwise_mean_latency_delta_ms=max_pairwise_mean_latency_delta_ms,
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
            "baseline_mode": gate_report.baseline_mode,
            "failed_checks": gate_report.failed_checks,
            "metrics": gate_report.metrics,
            "baseline_metrics": gate_report.baseline_metrics,
            "target_metrics": gate_report.target_metrics,
            "source_metrics": gate_report.source_metrics,
            "source_family_metrics": gate_report.source_family_metrics,
            "chunk_strategy_metrics": gate_report.chunk_strategy_metrics,
            "retrieval_role_metrics": gate_report.retrieval_role_metrics,
            "case_group_metrics": gate_report.case_group_metrics,
            "pairwise_metrics": gate_report.pairwise_metrics,
            "case_group_best_modes": gate_report.case_group_best_modes,
        }
    print(payload)
    if fail and not gate_report.passed:
        raise typer.Exit(1)


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
            "excluded_source_counts": report.excluded_source_counts,
            "excluded_source_family_counts": report.excluded_source_family_counts,
            "reason_counts_by_case_group": report.reason_counts_by_case_group,
            "missing_target_type_counts_by_case_group": (
                report.missing_target_type_counts_by_case_group
            ),
            "excluded_source_counts_by_case_group": (
                report.excluded_source_counts_by_case_group
            ),
            "excluded_source_family_counts_by_case_group": (
                report.excluded_source_family_counts_by_case_group
            ),
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
    min_case_count: int = typer.Option(
        0,
        "--min-case-count",
        help="Require at least this many benchmark cases in the retrieval evaluation.",
    ),
    min_expected_case_count: int = typer.Option(
        0,
        "--min-expected-case-count",
        help="Require at least this many cases with expected targets.",
    ),
    min_expected_target_count: int = typer.Option(
        0,
        "--min-expected-target-count",
        help="Require at least this many expected page/chunk/asset/triple targets.",
    ),
    min_passed_query_count: int = typer.Option(
        0,
        "--min-passed-query-count",
        help="Require at least this many benchmark queries to pass.",
    ),
    max_failed_queries: int | None = typer.Option(
        None,
        "--max-failed-queries",
        help="Limit benchmark queries that failed to retrieve expected evidence.",
    ),
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_result_stability_rate: float = typer.Option(
        0.0,
        "--min-result-stability-rate",
        help="Require this fraction of repeated retrieval cases to return the same top-k result set.",
    ),
    max_unstable_result_count: int | None = typer.Option(
        None,
        "--max-unstable-result-count",
        help="Limit repeated retrieval cases whose top-k result set changes across repeats.",
    ),
    max_excluded_target_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-target-hit-rate",
        help="Limit the fraction of explicit excluded page/chunk/asset/triple targets retrieved in top-k.",
    ),
    max_excluded_query_hit_rate: float | None = typer.Option(
        None,
        "--max-excluded-query-hit-rate",
        help="Limit the fraction of hard-negative cases that retrieve any explicit excluded target.",
    ),
    max_excluded_hit_query_count: int | None = typer.Option(
        None,
        "--max-excluded-hit-query-count",
        help="Limit hard-negative cases whose top-k contains any explicit excluded target.",
    ),
    min_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-target-type-coverage",
        help="Require target-type coverage such as asset=1.0 or triple=1.0. Repeat for multiple types.",
    ),
    min_source_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-target-coverage",
        help="Require exact retrieval-source target coverage such as qdrant:caption_dense=0.8.",
    ),
    min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-family-target-coverage",
        help="Require source-family target coverage such as lexical=0.8. Repeat for multiple families.",
    ),
    max_source_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-excluded-target-hit-rate",
        help=(
            "Limit exact-source excluded-target hit rate such as "
            "qdrant:image_dense=0.0. Repeat for multiple sources."
        ),
    ),
    max_source_family_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-source-family-excluded-target-hit-rate",
        help=(
            "Limit source-family excluded-target hit rate such as visual=0.0. "
            "Repeat for multiple families."
        ),
    ),
    min_chunk_strategy_target_coverage: list[str] = typer.Option(
        None,
        "--min-chunk-strategy-target-coverage",
        help="Require chunking-strategy target coverage such as visual_asset_text=0.8.",
    ),
    min_retrieval_role_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-role-target-coverage",
        help="Require retrieval-role target coverage such as child=0.8.",
    ),
    max_chunk_strategy_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-chunk-strategy-excluded-target-hit-rate",
        help=(
            "Limit chunking-strategy excluded-target hit rate such as "
            "visual_asset_text=0.0."
        ),
    ),
    max_retrieval_role_excluded_target_hit_rate: list[str] = typer.Option(
        None,
        "--max-retrieval-role-excluded-target-hit-rate",
        help="Limit retrieval-role excluded-target hit rate such as child=0.0.",
    ),
    min_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-case-group-target-coverage",
        help="Require case metadata group target coverage such as case_source:visual_lexical_probe=0.8.",
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
    source_thresholds = parse_named_float_thresholds(
        min_source_target_coverage,
        "source target coverage",
    )
    source_excluded_thresholds = parse_named_float_thresholds(
        max_source_excluded_target_hit_rate,
        "source excluded-target hit rate",
    )
    source_family_excluded_thresholds = parse_named_float_thresholds(
        max_source_family_excluded_target_hit_rate,
        "source family excluded-target hit rate",
    )
    target_type_thresholds = parse_named_float_thresholds(
        min_target_type_coverage,
        "target type coverage",
    )
    chunk_strategy_thresholds = parse_named_float_thresholds(
        min_chunk_strategy_target_coverage,
        "chunk strategy target coverage",
    )
    retrieval_role_thresholds = parse_named_float_thresholds(
        min_retrieval_role_target_coverage,
        "retrieval role target coverage",
    )
    chunk_strategy_excluded_thresholds = parse_named_float_thresholds(
        max_chunk_strategy_excluded_target_hit_rate,
        "chunk strategy excluded-target hit rate",
    )
    retrieval_role_excluded_thresholds = parse_named_float_thresholds(
        max_retrieval_role_excluded_target_hit_rate,
        "retrieval role excluded-target hit rate",
    )
    case_group_thresholds = parse_named_float_thresholds(
        min_case_group_target_coverage,
        "case group target coverage",
    )
    report = gate_retrieval_evaluation(
        parsed_evaluation,
        baseline=parsed_baseline,
        min_case_count=min_case_count,
        min_expected_case_count=min_expected_case_count,
        min_expected_target_count=min_expected_target_count,
        min_passed_query_count=min_passed_query_count,
        max_failed_query_count=max_failed_queries,
        min_recall_at_k=min_recall_at_k,
        min_target_coverage_at_k=min_target_coverage_at_k,
        min_target_ndcg_at_k=min_target_ndcg_at_k,
        min_mrr=min_mrr,
        min_precision_at_k=min_precision_at_k,
        max_mean_first_relevant_rank=max_mean_first_relevant_rank,
        max_p95_first_relevant_rank=max_p95_first_relevant_rank,
        max_mean_target_rank=max_mean_target_rank,
        max_p95_target_rank=max_p95_target_rank,
        max_mean_latency_ms=max_mean_latency_ms,
        max_p95_latency_ms=max_p95_latency_ms,
        min_result_stability_rate=min_result_stability_rate,
        max_unstable_result_count=max_unstable_result_count,
        max_excluded_target_hit_rate=max_excluded_target_hit_rate,
        max_excluded_query_hit_rate=max_excluded_query_hit_rate,
        max_excluded_hit_query_count=max_excluded_hit_query_count,
        min_target_type_coverage=target_type_thresholds,
        min_source_target_coverage=source_thresholds,
        min_source_family_target_coverage=source_family_thresholds,
        max_source_excluded_target_hit_rate=source_excluded_thresholds,
        max_source_family_excluded_target_hit_rate=source_family_excluded_thresholds,
        min_chunk_strategy_target_coverage=chunk_strategy_thresholds,
        min_retrieval_role_target_coverage=retrieval_role_thresholds,
        max_chunk_strategy_excluded_target_hit_rate=chunk_strategy_excluded_thresholds,
        max_retrieval_role_excluded_target_hit_rate=retrieval_role_excluded_thresholds,
        min_case_group_target_coverage=case_group_thresholds,
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
    deduplicate_tokens: bool = False,
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
            deduplicate_tokens=deduplicate_tokens,
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
    deduplicate_tokens: bool = False,
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
                deduplicate_tokens=deduplicate_tokens,
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
    min_visual_text_coverage_ratio: float | None = None,
    min_visual_text_part_coverage_ratio: float | None = None,
    min_recall_at_k: float | None = None,
    min_target_coverage_at_k: float | None = None,
    min_target_ndcg_at_k: float | None = None,
    min_mrr: float | None = None,
    min_precision_at_k: float | None = None,
    max_mean_first_relevant_rank: float | None = None,
    max_p95_first_relevant_rank: float | None = None,
    max_mean_target_rank: float | None = None,
    max_p95_target_rank: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_result_stability_rate: float | None = typer.Option(
        None,
        "--min-result-stability-rate",
        help="Require selected candidate repeated retrieval result stability.",
    ),
    max_unstable_result_count: int | None = typer.Option(
        None,
        "--max-unstable-result-count",
        help="Limit selected candidate cases with changing repeated top-k results.",
    ),
    max_failed_queries: int | None = 0,
    max_total_chunk_chars: float | None = typer.Option(
        None,
        "--max-total-chunk-chars",
        help="Limit total chunk text characters so retrieval gains do not hide embedding/index cost.",
    ),
    max_embedding_text_kchars: float | None = typer.Option(
        None,
        "--max-embedding-text-kchars",
        help="Limit total chunk text measured in thousands of embedding characters.",
    ),
    min_retrieval_score_per_embedding_kchar: float | None = typer.Option(
        None,
        "--min-retrieval-score-per-embedding-kchar",
        help="Require aggregate retrieval score per thousand embedding characters.",
    ),
    min_target_coverage_per_embedding_kchar: float | None = typer.Option(
        None,
        "--min-target-coverage-per-embedding-kchar",
        help="Require target coverage per thousand embedding characters.",
    ),
    min_target_ndcg_per_embedding_kchar: float | None = typer.Option(
        None,
        "--min-target-ndcg-per-embedding-kchar",
        help="Require target nDCG per thousand embedding characters.",
    ),
    min_retrieval_score_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--min-retrieval-score-per-mean-latency-ms",
        help="Require aggregate retrieval score per mean retrieval latency ms.",
    ),
    min_target_coverage_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--min-target-coverage-per-mean-latency-ms",
        help="Require target coverage per mean retrieval latency ms.",
    ),
    min_target_ndcg_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--min-target-ndcg-per-mean-latency-ms",
        help="Require target nDCG per mean retrieval latency ms.",
    ),
    min_retrieval_score_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--min-retrieval-score-per-p95-latency-ms",
        help="Require aggregate retrieval score per p95 retrieval latency ms.",
    ),
    min_target_coverage_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--min-target-coverage-per-p95-latency-ms",
        help="Require target coverage per p95 retrieval latency ms.",
    ),
    min_target_ndcg_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--min-target-ndcg-per-p95-latency-ms",
        help="Require target nDCG per p95 retrieval latency ms.",
    ),
    max_chunks_under_min_chars: int | None = None,
    max_chunks_over_max_chars: int | None = None,
    min_target_type_coverage: list[str] = typer.Option(
        None,
        "--min-target-type-coverage",
        help="Require target-type coverage such as asset=0.9 or triple=0.9. Repeat for multiple types.",
    ),
    min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--min-source-family-target-coverage",
        help="Require source-family target coverage such as lexical=0.8. Repeat for multiple families.",
    ),
    min_chunk_strategy_target_coverage: list[str] = typer.Option(
        None,
        "--min-chunk-strategy-target-coverage",
        help="Require chunking-strategy target coverage such as visual_asset_text=0.8.",
    ),
    min_retrieval_role_target_coverage: list[str] = typer.Option(
        None,
        "--min-retrieval-role-target-coverage",
        help="Require retrieval-role target coverage such as child=0.8.",
    ),
    min_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--min-case-group-target-coverage",
        help="Require case metadata group target coverage such as case_source:visual_lexical_probe=0.8.",
    ),
    max_quality_drop: float | None = None,
    max_recall_drop: float | None = None,
    max_target_coverage_drop: float | None = None,
    max_target_ndcg_drop: float | None = None,
    max_precision_drop: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
    min_pairwise_shared_queries: int | None = None,
    min_pairwise_win_rate: float | None = None,
    min_pairwise_target_coverage_lift: float | None = None,
    min_pairwise_target_ndcg_lift: float | None = None,
    min_pairwise_mrr_lift: float | None = None,
    min_pairwise_precision_lift: float | None = None,
    min_pairwise_target_coverage_ci_low: float | None = None,
    min_pairwise_target_ndcg_ci_low: float | None = None,
    min_pairwise_mrr_ci_low: float | None = None,
    min_pairwise_precision_ci_low: float | None = None,
    max_pairwise_mean_first_relevant_rank_delta: float | None = None,
    max_pairwise_mean_target_rank_delta: float | None = None,
    max_pairwise_first_relevant_rank_delta_ci_high: float | None = None,
    max_pairwise_target_rank_delta_ci_high: float | None = None,
    max_pairwise_mean_latency_delta_ms: float | None = None,
    fail: bool = typer.Option(
        True,
        "--fail/--no-fail",
        help="Exit with status 1 when any gate check fails.",
    ),
):
    """Fail a chunking strategy comparison when quality, retrieval, or latency checks are missed."""
    parsed_comparison = load_chunking_comparison(comparison)
    target_type_thresholds = parse_named_float_thresholds(
        min_target_type_coverage,
        "target type coverage",
    )
    source_family_thresholds = parse_named_float_thresholds(
        min_source_family_target_coverage,
        "source family target coverage",
    )
    chunk_strategy_thresholds = parse_named_float_thresholds(
        min_chunk_strategy_target_coverage,
        "chunk strategy target coverage",
    )
    retrieval_role_thresholds = parse_named_float_thresholds(
        min_retrieval_role_target_coverage,
        "retrieval role target coverage",
    )
    case_group_thresholds = parse_named_float_thresholds(
        min_case_group_target_coverage,
        "case group target coverage",
    )
    report = gate_chunking_comparison(
        parsed_comparison,
        candidate=candidate,
        baseline_candidate=baseline_candidate,
        require_retrieval=require_retrieval,
        min_quality_score=min_quality_score,
        min_page_coverage_ratio=min_page_coverage_ratio,
        min_visual_annotation_ratio=min_visual_annotation_ratio,
        min_visual_text_coverage_ratio=min_visual_text_coverage_ratio,
        min_visual_text_part_coverage_ratio=min_visual_text_part_coverage_ratio,
        min_recall_at_k=min_recall_at_k,
        min_target_coverage_at_k=min_target_coverage_at_k,
        min_target_ndcg_at_k=min_target_ndcg_at_k,
        min_mrr=min_mrr,
        min_precision_at_k=min_precision_at_k,
        max_mean_first_relevant_rank=max_mean_first_relevant_rank,
        max_p95_first_relevant_rank=max_p95_first_relevant_rank,
        max_mean_target_rank=max_mean_target_rank,
        max_p95_target_rank=max_p95_target_rank,
        max_mean_latency_ms=max_mean_latency_ms,
        max_p95_latency_ms=max_p95_latency_ms,
        min_result_stability_rate=min_result_stability_rate,
        max_unstable_result_count=max_unstable_result_count,
        max_failed_queries=max_failed_queries,
        max_total_chunk_chars=max_total_chunk_chars,
        max_embedding_text_kchars=max_embedding_text_kchars,
        min_retrieval_score_per_embedding_kchar=min_retrieval_score_per_embedding_kchar,
        min_target_coverage_per_embedding_kchar=min_target_coverage_per_embedding_kchar,
        min_target_ndcg_per_embedding_kchar=min_target_ndcg_per_embedding_kchar,
        min_retrieval_score_per_mean_latency_ms=(
            min_retrieval_score_per_mean_latency_ms
        ),
        min_target_coverage_per_mean_latency_ms=(
            min_target_coverage_per_mean_latency_ms
        ),
        min_target_ndcg_per_mean_latency_ms=min_target_ndcg_per_mean_latency_ms,
        min_retrieval_score_per_p95_latency_ms=(
            min_retrieval_score_per_p95_latency_ms
        ),
        min_target_coverage_per_p95_latency_ms=min_target_coverage_per_p95_latency_ms,
        min_target_ndcg_per_p95_latency_ms=min_target_ndcg_per_p95_latency_ms,
        max_chunks_under_min_chars=max_chunks_under_min_chars,
        max_chunks_over_max_chars=max_chunks_over_max_chars,
        min_target_type_coverage=target_type_thresholds,
        min_source_family_target_coverage=source_family_thresholds,
        min_chunk_strategy_target_coverage=chunk_strategy_thresholds,
        min_retrieval_role_target_coverage=retrieval_role_thresholds,
        min_case_group_target_coverage=case_group_thresholds,
        max_quality_drop=max_quality_drop,
        max_recall_drop=max_recall_drop,
        max_target_coverage_drop=max_target_coverage_drop,
        max_target_ndcg_drop=max_target_ndcg_drop,
        max_precision_drop=max_precision_drop,
        max_mean_latency_ratio=max_mean_latency_ratio,
        max_p95_latency_ratio=max_p95_latency_ratio,
        min_pairwise_shared_queries=min_pairwise_shared_queries,
        min_pairwise_win_rate=min_pairwise_win_rate,
        min_pairwise_target_coverage_lift=min_pairwise_target_coverage_lift,
        min_pairwise_target_ndcg_lift=min_pairwise_target_ndcg_lift,
        min_pairwise_mrr_lift=min_pairwise_mrr_lift,
        min_pairwise_precision_lift=min_pairwise_precision_lift,
        min_pairwise_target_coverage_ci_low=min_pairwise_target_coverage_ci_low,
        min_pairwise_target_ndcg_ci_low=min_pairwise_target_ndcg_ci_low,
        min_pairwise_mrr_ci_low=min_pairwise_mrr_ci_low,
        min_pairwise_precision_ci_low=min_pairwise_precision_ci_low,
        max_pairwise_mean_first_relevant_rank_delta=max_pairwise_mean_first_relevant_rank_delta,
        max_pairwise_mean_target_rank_delta=max_pairwise_mean_target_rank_delta,
        max_pairwise_first_relevant_rank_delta_ci_high=(
            max_pairwise_first_relevant_rank_delta_ci_high
        ),
        max_pairwise_target_rank_delta_ci_high=max_pairwise_target_rank_delta_ci_high,
        max_pairwise_mean_latency_delta_ms=max_pairwise_mean_latency_delta_ms,
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
    strategies: str = "semantic,multimodal,object_aware,hierarchical",
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
    selection_min_retrieval_recall_at_k: float | None = typer.Option(
        None,
        "--selection-min-retrieval-recall-at-k",
        help="Only recommend sweep candidates at or above this retrieval recall@k.",
    ),
    selection_min_target_coverage_at_k: float | None = typer.Option(
        None,
        "--selection-min-target-coverage-at-k",
        help="Only recommend sweep candidates at or above this target coverage@k.",
    ),
    selection_min_target_ndcg_at_k: float | None = typer.Option(
        None,
        "--selection-min-target-ndcg-at-k",
        help="Only recommend sweep candidates at or above this mean target nDCG@k.",
    ),
    selection_min_precision_at_k: float | None = typer.Option(
        None,
        "--selection-min-precision-at-k",
        help="Only recommend sweep candidates at or above this precision@k.",
    ),
    selection_min_retrieval_score_per_embedding_kchar: float | None = typer.Option(
        None,
        "--selection-min-retrieval-score-per-embedding-kchar",
        help="Only recommend sweep candidates with enough aggregate retrieval score per 1k embedding chars.",
    ),
    selection_min_target_coverage_per_embedding_kchar: float | None = typer.Option(
        None,
        "--selection-min-target-coverage-per-embedding-kchar",
        help="Only recommend sweep candidates with enough target coverage per 1k embedding chars.",
    ),
    selection_min_target_ndcg_per_embedding_kchar: float | None = typer.Option(
        None,
        "--selection-min-target-ndcg-per-embedding-kchar",
        help="Only recommend sweep candidates with enough target nDCG per 1k embedding chars.",
    ),
    selection_min_retrieval_score_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--selection-min-retrieval-score-per-mean-latency-ms",
        help="Only recommend sweep candidates with enough aggregate retrieval score per mean latency ms.",
    ),
    selection_min_target_coverage_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--selection-min-target-coverage-per-mean-latency-ms",
        help="Only recommend sweep candidates with enough target coverage per mean latency ms.",
    ),
    selection_min_target_ndcg_per_mean_latency_ms: float | None = typer.Option(
        None,
        "--selection-min-target-ndcg-per-mean-latency-ms",
        help="Only recommend sweep candidates with enough target nDCG per mean latency ms.",
    ),
    selection_min_retrieval_score_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--selection-min-retrieval-score-per-p95-latency-ms",
        help="Only recommend sweep candidates with enough aggregate retrieval score per p95 latency ms.",
    ),
    selection_min_target_coverage_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--selection-min-target-coverage-per-p95-latency-ms",
        help="Only recommend sweep candidates with enough target coverage per p95 latency ms.",
    ),
    selection_min_target_ndcg_per_p95_latency_ms: float | None = typer.Option(
        None,
        "--selection-min-target-ndcg-per-p95-latency-ms",
        help="Only recommend sweep candidates with enough target nDCG per p95 latency ms.",
    ),
    selection_min_quality_score: float | None = typer.Option(
        None,
        "--selection-min-quality-score",
        help="Only recommend sweep candidates at or above this chunking quality score.",
    ),
    selection_min_visual_text_coverage_ratio: float | None = typer.Option(
        None,
        "--selection-min-visual-text-coverage-ratio",
        help="Only recommend sweep candidates at or above this linked visual text coverage.",
    ),
    selection_min_visual_text_part_coverage_ratio: float | None = typer.Option(
        None,
        "--selection-min-visual-text-part-coverage-ratio",
        help="Only recommend sweep candidates at or above this linked visual text part coverage.",
    ),
    selection_min_result_stability_rate: float | None = typer.Option(
        None,
        "--selection-min-result-stability-rate",
        help="Only recommend sweep candidates with stable repeated retrieval result sets.",
    ),
    selection_min_target_type_coverage: list[str] = typer.Option(
        None,
        "--selection-min-target-type-coverage",
        help="Only recommend sweep candidates with target-type coverage such as asset=0.9.",
    ),
    selection_min_source_family_target_coverage: list[str] = typer.Option(
        None,
        "--selection-min-source-family-target-coverage",
        help="Only recommend sweep candidates with source-family coverage such as visual=0.8.",
    ),
    selection_min_case_group_target_coverage: list[str] = typer.Option(
        None,
        "--selection-min-case-group-target-coverage",
        help="Only recommend sweep candidates with case-group coverage such as case_source:visual_object_probe=0.8.",
    ),
    selection_max_mean_target_rank: float | None = typer.Option(
        None,
        "--selection-max-mean-target-rank",
        help="Only recommend sweep candidates at or below this mean target rank.",
    ),
    selection_max_p95_target_rank: float | None = typer.Option(
        None,
        "--selection-max-p95-target-rank",
        help="Only recommend sweep candidates at or below this p95 target rank.",
    ),
    selection_max_mean_latency_ms: float | None = typer.Option(
        None,
        "--selection-max-mean-latency-ms",
        help="Only recommend sweep candidates at or below this mean retrieval latency.",
    ),
    selection_max_p95_latency_ms: float | None = typer.Option(
        None,
        "--selection-max-p95-latency-ms",
        help="Only recommend sweep candidates at or below this p95 retrieval latency.",
    ),
    selection_max_unstable_result_count: float | None = typer.Option(
        None,
        "--selection-max-unstable-result-count",
        help="Only recommend sweep candidates at or below this repeated retrieval instability count.",
    ),
    selection_max_chunk_count: float | None = typer.Option(
        None,
        "--selection-max-chunk-count",
        help="Only recommend sweep candidates at or below this chunk count.",
    ),
    selection_max_total_chunk_chars: float | None = typer.Option(
        None,
        "--selection-max-total-chunk-chars",
        help="Only recommend sweep candidates at or below this total chunk text size.",
    ),
    selection_max_mean_chunk_chars: float | None = typer.Option(
        None,
        "--selection-max-mean-chunk-chars",
        help="Only recommend sweep candidates at or below this mean chunk text size.",
    ),
    selection_max_p95_chunk_chars: float | None = typer.Option(
        None,
        "--selection-max-p95-chunk-chars",
        help="Only recommend sweep candidates at or below this p95 chunk text size.",
    ),
    selection_max_embedding_text_kchars: float | None = typer.Option(
        None,
        "--selection-max-embedding-text-kchars",
        help="Only recommend sweep candidates at or below this estimated embedding text volume.",
    ),
    selection_max_standalone_visual_chunk_count: float | None = typer.Option(
        None,
        "--selection-max-standalone-visual-chunk-count",
        help="Only recommend sweep candidates at or below this standalone visual chunk count.",
    ),
    selection_max_visual_object_chunk_count: float | None = typer.Option(
        None,
        "--selection-max-visual-object-chunk-count",
        help="Only recommend sweep candidates at or below this visual object chunk count.",
    ),
    lexical_tokenizer: TokenizerStrategy = "mixed",
    ngram_min: int = 2,
    ngram_max: int = 4,
    ngram_cjk_only: bool = True,
    deduplicate_tokens: bool = False,
):
    """Generate and evaluate a grid of chunking strategy candidates."""
    manifest = load_processing_package(package_dir)
    fusion_weights = parse_fusion_weights(fusion_weight)
    selection_target_type_thresholds = parse_named_float_thresholds(
        selection_min_target_type_coverage,
        label="selection target-type coverage",
    )
    selection_source_family_thresholds = parse_named_float_thresholds(
        selection_min_source_family_target_coverage,
        label="selection source-family target coverage",
    )
    selection_case_group_thresholds = parse_named_float_thresholds(
        selection_min_case_group_target_coverage,
        label="selection case-group target coverage",
    )
    selection_constraints: dict[str, float | None] = {
        "min_retrieval_recall_at_k": selection_min_retrieval_recall_at_k,
        "min_target_coverage_at_k": selection_min_target_coverage_at_k,
        "min_target_ndcg_at_k": selection_min_target_ndcg_at_k,
        "min_precision_at_k": selection_min_precision_at_k,
        "min_retrieval_score_per_embedding_kchar": (
            selection_min_retrieval_score_per_embedding_kchar
        ),
        "min_target_coverage_per_embedding_kchar": (
            selection_min_target_coverage_per_embedding_kchar
        ),
        "min_target_ndcg_per_embedding_kchar": (
            selection_min_target_ndcg_per_embedding_kchar
        ),
        "min_retrieval_score_per_mean_latency_ms": (
            selection_min_retrieval_score_per_mean_latency_ms
        ),
        "min_target_coverage_per_mean_latency_ms": (
            selection_min_target_coverage_per_mean_latency_ms
        ),
        "min_target_ndcg_per_mean_latency_ms": (
            selection_min_target_ndcg_per_mean_latency_ms
        ),
        "min_retrieval_score_per_p95_latency_ms": (
            selection_min_retrieval_score_per_p95_latency_ms
        ),
        "min_target_coverage_per_p95_latency_ms": (
            selection_min_target_coverage_per_p95_latency_ms
        ),
        "min_target_ndcg_per_p95_latency_ms": (
            selection_min_target_ndcg_per_p95_latency_ms
        ),
        "min_quality_score": selection_min_quality_score,
        "min_visual_text_coverage_ratio": selection_min_visual_text_coverage_ratio,
        "min_visual_text_part_coverage_ratio": (
            selection_min_visual_text_part_coverage_ratio
        ),
        "min_result_stability_rate": selection_min_result_stability_rate,
        "max_mean_target_rank": selection_max_mean_target_rank,
        "max_p95_target_rank": selection_max_p95_target_rank,
        "max_mean_latency_ms": selection_max_mean_latency_ms,
        "max_p95_latency_ms": selection_max_p95_latency_ms,
        "max_unstable_result_count": selection_max_unstable_result_count,
        "max_chunk_count": selection_max_chunk_count,
        "max_total_chunk_chars": selection_max_total_chunk_chars,
        "max_mean_chunk_chars": selection_max_mean_chunk_chars,
        "max_p95_chunk_chars": selection_max_p95_chunk_chars,
        "max_embedding_text_kchars": selection_max_embedding_text_kchars,
        "max_standalone_visual_chunk_count": (
            selection_max_standalone_visual_chunk_count
        ),
        "max_visual_object_chunk_count": selection_max_visual_object_chunk_count,
    }
    selection_constraints.update(
        {
            f"min_target_type_coverage:{target_type}": threshold
            for target_type, threshold in selection_target_type_thresholds.items()
        }
    )
    selection_constraints.update(
        {
            f"min_source_family_target_coverage:{family}": threshold
            for family, threshold in selection_source_family_thresholds.items()
        }
    )
    selection_constraints.update(
        {
            f"min_case_group_target_coverage:{case_group}": threshold
            for case_group, threshold in selection_case_group_thresholds.items()
        }
    )
    retrieval_cases = load_retrieval_cases(cases) if cases is not None else None
    tokenizer_config = build_tokenizer_config(
        lexical_tokenizer,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate_tokens=deduplicate_tokens,
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
        selection_constraints=selection_constraints,
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
            "recommended": report.selection.recommended,
            "pareto_front": report.selection.pareto_front,
            "eligible_pareto_front": report.selection.eligible_pareto_front,
            "eligible_count": report.selection.eligible_count,
            "rejected_count": report.selection.rejected_count,
            "top_candidates": [
                {
                    "name": candidate.name,
                    "selection_score": next(
                        (
                            round(row.score, 6)
                            for row in report.selection.ranking
                            if row.name == candidate.name
                        ),
                        None,
                    ),
                    "eligible": next(
                        (
                            row.eligible
                            for row in report.selection.ranking
                            if row.name == candidate.name
                        ),
                        None,
                    ),
                    "failed_constraints": next(
                        (
                            row.failed_constraints
                            for row in report.selection.ranking
                            if row.name == candidate.name
                        ),
                        [],
                    ),
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


@app.command(name="apply-chunking-sweep")
def apply_chunking_sweep_command(
    package_dir: Path = Path("outputs/package"),
    report: Path = Path("outputs/package/chunking_sweep.json"),
    candidate: str = "",
    chunks_file: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
    rebuild_search: bool = True,
    rebuild_dry_run_embeddings: bool = False,
    clear_stale_embeddings: bool = True,
):
    """Apply a recommended chunking sweep candidate as the package chunk set."""
    if rebuild_dry_run_embeddings and not rebuild_search:
        raise typer.BadParameter("--rebuild-dry-run-embeddings requires --rebuild-search")

    manifest = load_processing_package(package_dir)
    sweep_report = load_chunking_sweep_report(report)
    selected = select_chunking_sweep_candidate(sweep_report, candidate or None)
    selected_chunks_path = resolve_sweep_chunks_file(selected, chunks_file, report)
    selected_chunks = read_jsonl(selected_chunks_path, DocumentChunk)
    if not selected_chunks:
        raise typer.BadParameter(f"Selected chunk file is empty: {selected_chunks_path}")

    remapped_triples = remap_triples_to_available_chunks(manifest.triples, selected_chunks)
    stale_embedding_artifacts: list[str] = []
    backup_files = {}
    if backup:
        safe_name = safe_filename_part(selected.name)
        backup_files = {
            "chunks": str(package_dir / f"chunks.before-{safe_name}.jsonl"),
            "triples": str(package_dir / f"triples.before-{safe_name}.jsonl"),
        }

    payload = {
        "package_dir": str(package_dir),
        "report": str(report),
        "candidate": selected.name,
        "strategy": selected.strategy,
        "candidate_chunks_file": str(selected_chunks_path),
        "previous_chunk_count": len(manifest.chunks),
        "applied_chunk_count": len(selected_chunks),
        "previous_triple_count": len(manifest.triples),
        "remapped_triple_count": len(remapped_triples),
        "backup_files": backup_files,
        "rebuilt_search": bool(rebuild_search and not dry_run),
        "rebuilt_dry_run_embeddings": bool(rebuild_dry_run_embeddings and not dry_run),
        "cleared_embedding_artifacts": stale_embedding_artifacts,
        "requires_embedding_rebuild": not rebuild_dry_run_embeddings,
        "next_embedding_command": (
            f"chunking-docs embed-package --package-dir {shlex.quote(package_dir.as_posix())}"
        ),
        "dry_run": dry_run,
    }
    if dry_run:
        print(payload)
        return

    if backup:
        write_jsonl(Path(backup_files["chunks"]), manifest.chunks)
        write_jsonl(Path(backup_files["triples"]), manifest.triples)
    write_jsonl(package_dir / "chunks.jsonl", selected_chunks)
    write_jsonl(package_dir / "triples.jsonl", remapped_triples)

    if rebuild_search:
        rebuild_search_artifacts(
            package_dir,
            selected_chunks,
            assets=manifest.assets,
            triples=remapped_triples,
            tokenizer_config=manifest_tokenizer_config(manifest),
            rebuild_embeddings=rebuild_dry_run_embeddings,
        )
    if clear_stale_embeddings and not rebuild_dry_run_embeddings:
        stale_embedding_artifacts = clear_embedding_artifacts(package_dir)
        payload["cleared_embedding_artifacts"] = stale_embedding_artifacts
    update_manifest_chunking_selection(
        package_dir=package_dir,
        report=report,
        candidate=selected,
        chunks_file=selected_chunks_path,
        chunk_count=len(selected_chunks),
    )
    print(payload)


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
    deduplicate_tokens: bool = False,
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
        deduplicate_tokens=deduplicate_tokens,
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


def retrieval_source_metrics_payload(evaluation) -> dict:
    return {
        name: metric.model_dump()
        for name, metric in getattr(evaluation, "source_metrics", {}).items()
    }


def retrieval_source_family_metrics_payload(evaluation) -> dict:
    return {
        name: metric.model_dump()
        for name, metric in getattr(evaluation, "source_family_metrics", {}).items()
    }


def retrieval_chunk_strategy_metrics_payload(evaluation) -> dict:
    return {
        name: metric.model_dump()
        for name, metric in getattr(evaluation, "chunk_strategy_metrics", {}).items()
    }


def retrieval_role_metrics_payload(evaluation) -> dict:
    return {
        name: metric.model_dump()
        for name, metric in getattr(evaluation, "retrieval_role_metrics", {}).items()
    }


def retrieval_case_group_metrics_payload(evaluation) -> dict:
    return {
        group_name: {
            group_value: metric.model_dump()
            for group_value, metric in group_values.items()
        }
        for group_name, group_values in getattr(evaluation, "case_group_metrics", {}).items()
    }


def read_qdrant_retrieval_config(config: Path) -> QdrantRetrievalConfig:
    try:
        return QdrantRetrievalConfig.model_validate_json(config.read_text(encoding="utf-8"))
    except OSError as exc:
        raise typer.BadParameter(f"Could not read retrieval config: {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid retrieval config: {exc}") from exc


def qdrant_route_usage(route_decisions: list) -> dict[str, object]:
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for decision in route_decisions:
        name = decision.name or "default"
        counts[name] = counts.get(name, 0) + 1
        reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
    return {
        "matched_count": sum(1 for decision in route_decisions if decision.matched),
        "counts": counts,
        "reasons": reasons,
    }


def retrieval_cases_with_route_metadata(
    cases: list[RetrievalCase],
    route_decisions: list,
) -> list[RetrievalCase]:
    routed_cases = []
    for case, decision in zip(cases, route_decisions):
        metadata = {
            **case.metadata,
            "retrieval_route": decision.name or "default",
            "retrieval_route_reason": decision.reason,
        }
        routed_cases.append(case.model_copy(update={"metadata": metadata}))
    return routed_cases


def retrieval_config_tokenizer_options(
    config: QdrantRetrievalConfig,
) -> dict[str, object]:
    payload = config.lexical_tokenizer if isinstance(config.lexical_tokenizer, dict) else {}
    strategy = str(payload.get("strategy") or "mixed")
    if strategy not in {"word", "char_ngram", "mixed"}:
        raise typer.BadParameter(f"Unsupported config lexical tokenizer strategy: {strategy}")
    return {
        "strategy": strategy,
        "min_n": int(payload.get("min_n") or 2),
        "max_n": int(payload.get("max_n") or 4),
        "ngram_cjk_only": config_bool(payload.get("ngram_cjk_only"), default=True),
        "deduplicate": config_bool(payload.get("deduplicate"), default=False),
    }


def build_retrieval_config_reranker(
    config: QdrantRetrievalConfig,
    tokenizer_options: dict[str, object],
    device: str,
):
    tokenizer_config = build_tokenizer_config(
        tokenizer_options["strategy"],
        ngram_min=int(tokenizer_options["min_n"]),
        ngram_max=int(tokenizer_options["max_n"]),
        ngram_cjk_only=bool(tokenizer_options["ngram_cjk_only"]),
        deduplicate_tokens=bool(tokenizer_options["deduplicate"]),
    )
    return build_reranker(
        config.reranker,
        model_name=config.reranker_model,
        device=device,
        max_length=config.reranker_max_length,
        tokenizer_config=tokenizer_config,
    )


def retrieval_config_rerank_top_k(config: QdrantRetrievalConfig, reranker) -> int | None:
    if reranker is None:
        return None
    return config.rerank_top_k or config.top_k


def config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
    vector_name: str = "caption_dense",
    option_name: str = "caption",
):
    normalized = normalize_backend(backend)
    if normalized in {"same-as-text", "same_as_text", "same"}:
        if text_embedder is None:
            raise typer.BadParameter(f"--{option_name}-backend same-as-text requires text backend")
        return text_embedder, f"Same model as text_dense. {text_note}"
    return build_text_embedder(
        backend=backend,
        model_name=model_name,
        device=device,
        hashing_dim=hashing_dim,
        vector_name=vector_name,
    )


def build_object_embedder(
    backend: str,
    model_name: str,
    device: str,
    hashing_dim: int,
    text_embedder,
    text_note: str,
    caption_embedder,
    caption_note: str,
):
    normalized = normalize_backend(backend)
    if normalized == "none":
        return None, ""
    if normalized in {"same-as-caption", "same_as_caption", "same"}:
        if caption_embedder is None:
            return None, ""
        return caption_embedder, f"Same model as caption_dense. {caption_note}"
    return build_caption_embedder(
        backend=backend,
        model_name=model_name,
        device=device,
        hashing_dim=hashing_dim,
        text_embedder=text_embedder,
        text_note=text_note,
        vector_name="object_dense",
        option_name="object",
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


def embedding_vector_metadata(
    text_backend: str,
    caption_backend: str,
    object_backend: str,
    image_backend: str,
    triple_backend: str,
    text_model: str,
    caption_model: str,
    object_model: str,
    image_model: str,
    triple_model: str,
    text_device: str,
    image_device: str,
    text_batch_size: int,
    caption_batch_size: int,
    object_batch_size: int,
    image_batch_size: int,
    triple_batch_size: int,
    hashing_dim: int,
    include_text: bool,
    include_caption: bool,
    include_object: bool,
    include_image: bool,
    include_triple: bool,
) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    normalized_text_backend = normalize_backend(text_backend)
    normalized_caption_backend = normalize_backend(caption_backend)
    normalized_object_backend = normalize_backend(object_backend)
    normalized_image_backend = normalize_backend(image_backend)
    normalized_triple_backend = normalize_backend(triple_backend)
    if include_text:
        metadata["text_dense"] = embedding_backend_metadata(
            backend=normalized_text_backend,
            model_name=text_model,
            device=text_device,
            batch_size=text_batch_size,
            hashing_dim=hashing_dim,
        )
    if include_caption:
        if normalized_caption_backend in {"same-as-text", "same_as_text", "same"}:
            metadata["caption_dense"] = {
                **embedding_backend_metadata(
                    backend=normalized_text_backend,
                    model_name=text_model,
                    device=text_device,
                    batch_size=caption_batch_size,
                    hashing_dim=hashing_dim,
                ),
                "same_as": "text_dense",
            }
        else:
            metadata["caption_dense"] = embedding_backend_metadata(
                backend=normalized_caption_backend,
                model_name=caption_model,
                device=text_device,
                batch_size=caption_batch_size,
                hashing_dim=hashing_dim,
            )
    if include_object:
        if normalized_object_backend in {"same-as-caption", "same_as_caption", "same"}:
            metadata["object_dense"] = {
                **dict(metadata.get("caption_dense") or {}),
                "batch_size": object_batch_size,
                "same_as": "caption_dense",
            }
        elif normalized_object_backend in {"same-as-text", "same_as_text"}:
            metadata["object_dense"] = {
                **embedding_backend_metadata(
                    backend=normalized_text_backend,
                    model_name=text_model,
                    device=text_device,
                    batch_size=object_batch_size,
                    hashing_dim=hashing_dim,
                ),
                "same_as": "text_dense",
            }
        else:
            metadata["object_dense"] = embedding_backend_metadata(
                backend=normalized_object_backend,
                model_name=object_model,
                device=text_device,
                batch_size=object_batch_size,
                hashing_dim=hashing_dim,
            )
    if include_image:
        metadata["image_dense"] = embedding_backend_metadata(
            backend=normalized_image_backend,
            model_name=image_model,
            device=image_device,
            batch_size=image_batch_size,
            hashing_dim=hashing_dim,
        )
    if include_triple:
        if normalized_triple_backend in {"same-as-text", "same_as_text", "same"}:
            metadata["triple_dense"] = {
                **embedding_backend_metadata(
                    backend=normalized_text_backend,
                    model_name=text_model,
                    device=text_device,
                    batch_size=triple_batch_size,
                    hashing_dim=hashing_dim,
                ),
                "same_as": "text_dense",
            }
        else:
            metadata["triple_dense"] = embedding_backend_metadata(
                backend=normalized_triple_backend,
                model_name=triple_model,
                device=text_device,
                batch_size=triple_batch_size,
                hashing_dim=hashing_dim,
            )
    return metadata


def embedding_backend_metadata(
    backend: str,
    model_name: str,
    device: str,
    batch_size: int,
    hashing_dim: int,
) -> dict:
    metadata = {
        "backend": backend,
        "batch_size": batch_size,
    }
    if backend == "hashing":
        metadata.update(
            {
                "model": "HashingEmbedder",
                "dimension": hashing_dim,
                "deterministic": True,
            }
        )
    elif backend != "none":
        metadata.update(
            {
                "model": model_name,
                "device": device or "auto",
            }
        )
    return metadata


TEXT_QUERY_VECTOR_PRIORITY = ("text_dense", "caption_dense", "object_dense", "triple_dense")
TEXT_QUERY_BACKENDS = {"sentence-transformers", "sentence_transformers", "st"}
CLIP_QUERY_BACKENDS = {"clip", "transformers"}
SAME_AS_TEXT_BACKENDS = {"same-as-text", "same_as_text", "same"}


def resolve_qdrant_query_backend_options(
    package_dir: Path,
    selected_vectors: list[str],
    text_backend: str,
    text_model: str,
    image_query_backend: str,
    image_query_model: str,
) -> dict[str, str]:
    """Resolve `auto` query encoder settings from embedding_manifest.json when available."""
    vectors = load_embedding_manifest_vectors(package_dir)
    resolved_text_backend = text_backend
    resolved_text_model = text_model
    resolved_image_query_backend = image_query_backend
    resolved_image_query_model = image_query_model

    if normalize_backend(text_backend) == "auto":
        vector_name = select_text_query_vector(selected_vectors, vectors)
        embedding = resolve_embedding_config(vectors, vector_name) if vector_name else {}
        backend = normalize_backend(str(embedding.get("backend") or ""))
        if backend in TEXT_QUERY_BACKENDS:
            resolved_text_backend = "sentence-transformers"
            resolved_text_model = str(embedding.get("model") or text_model)
        elif backend == "hashing":
            resolved_text_backend = "hashing"
        else:
            resolved_text_backend = "hashing"

    if normalize_backend(image_query_backend) == "auto":
        if "image_dense" not in selected_vectors:
            resolved_image_query_backend = "none"
        else:
            embedding = resolve_embedding_config(vectors, "image_dense")
            backend = normalize_backend(str(embedding.get("backend") or ""))
            same_as = str(embedding.get("same_as") or "")
            if backend in CLIP_QUERY_BACKENDS:
                resolved_image_query_backend = "clip"
                resolved_image_query_model = str(embedding.get("model") or image_query_model)
            elif backend == "hashing":
                resolved_image_query_backend = "hashing"
            elif backend in TEXT_QUERY_BACKENDS or normalize_backend(same_as) in TEXT_QUERY_VECTOR_PRIORITY:
                resolved_image_query_backend = "same-as-text"
            elif backend in SAME_AS_TEXT_BACKENDS:
                resolved_image_query_backend = "same-as-text"
            else:
                resolved_image_query_backend = "none"

    return {
        "text_backend": resolved_text_backend,
        "text_model": resolved_text_model,
        "image_query_backend": resolved_image_query_backend,
        "image_query_model": resolved_image_query_model,
    }


def load_embedding_manifest_vectors(package_dir: Path) -> dict[str, dict]:
    manifest_path = package_dir / "embedding_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid embedding manifest JSON: {manifest_path}") from exc
    vectors = manifest.get("vectors") if isinstance(manifest, dict) else None
    return dict(vectors) if isinstance(vectors, dict) else {}


def select_text_query_vector(selected_vectors: list[str], vectors: dict[str, dict]) -> str | None:
    for vector_name in TEXT_QUERY_VECTOR_PRIORITY:
        if vector_name in selected_vectors:
            return vector_name
    for vector_name in selected_vectors:
        if vector_name != "image_dense":
            return vector_name
    if "text_dense" in vectors:
        return "text_dense"
    return None


def resolve_embedding_config(
    vectors: dict[str, dict],
    vector_name: str,
    seen: set[str] | None = None,
) -> dict:
    if not vector_name or vector_name in (seen or set()):
        return {}
    entry = vectors.get(vector_name)
    if not isinstance(entry, dict):
        return {}
    embedding = dict(entry.get("embedding") or {})
    if "dimension" not in embedding and entry.get("dimension") is not None:
        embedding["dimension"] = entry["dimension"]
    same_as = embedding.get("same_as") or entry.get("same_as")
    if not same_as:
        return embedding
    next_seen = set(seen or set())
    next_seen.add(vector_name)
    base = resolve_embedding_config(vectors, str(same_as), next_seen)
    if not base:
        return embedding
    merged = {**base, **{key: value for key, value in embedding.items() if value not in (None, "")}}
    merged["same_as"] = str(same_as)
    return merged


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


def validate_qdrant_query_encoder_dimensions(
    selected_vectors: list[str],
    vector_sizes: dict[str, int],
    default_embedder,
    vector_embedders: dict,
    vector_notes: dict[str, str] | None = None,
    image_query_backend: str = "none",
) -> None:
    mismatches = []
    for vector_name in selected_vectors:
        expected = vector_sizes.get(vector_name)
        embedder = vector_embedders.get(vector_name, default_embedder)
        actual = getattr(embedder, "embedding_dim", None)
        if expected is None or actual is None:
            continue
        if int(actual) != int(expected):
            mismatches.append(
                {
                    "vector_name": vector_name,
                    "expected": int(expected),
                    "actual": int(actual),
                    "encoder": qdrant_query_encoder_label(
                        vector_name,
                        vector_embedders=vector_embedders,
                        default_embedder=default_embedder,
                        image_query_backend=image_query_backend,
                    ),
                    "note": (vector_notes or {}).get(vector_name, ""),
                }
            )

    if not mismatches:
        return

    details = "; ".join(
        (
            f"{item['vector_name']} expects {item['expected']} dimensions, "
            f"but {item['encoder']} produces {item['actual']}"
        )
        for item in mismatches
    )
    notes = "; ".join(
        f"{item['vector_name']} package vector note: {str(item['note']).rstrip('.')}"
        for item in mismatches
        if item["note"]
    )
    guidance = (
        "Use a query encoder that matches the package vectors, for example "
        "--text-backend sentence-transformers --text-model <model-used-for-text_dense>, "
        "or rebuild the package with the selected query encoder."
    )
    if any(item["vector_name"] == "image_dense" for item in mismatches):
        guidance += (
            " For image_dense, set --image-query-backend and --image-query-model to the "
            "text side of the image embedding model."
        )
    message = f"Qdrant query encoder dimension mismatch: {details}. {guidance}"
    if notes:
        message += f" {notes}."
    raise typer.BadParameter(message)


def qdrant_query_encoder_label(
    vector_name: str,
    vector_embedders: dict,
    default_embedder,
    image_query_backend: str,
) -> str:
    if vector_embedders.get(vector_name, default_embedder) is default_embedder:
        return "default text query encoder"
    if vector_name == "image_dense":
        return f"{image_query_backend} image query encoder"
    return f"{vector_name} query encoder"


def qdrant_query_encoder_details(
    selected_vectors: list[str],
    vector_embedders: dict,
    default_embedder,
    text_backend: str,
    text_model: str,
    image_query_backend: str,
    image_query_model: str,
) -> dict[str, dict[str, object]]:
    details = {}
    for vector_name in selected_vectors:
        embedder = vector_embedders.get(vector_name, default_embedder)
        uses_default_text = embedder is default_embedder
        backend = normalize_backend(text_backend) if uses_default_text else normalize_backend(image_query_backend)
        details[vector_name] = {
            "encoder": qdrant_query_encoder_label(
                vector_name,
                vector_embedders=vector_embedders,
                default_embedder=default_embedder,
                image_query_backend=image_query_backend,
            ),
            "backend": backend,
            "model": qdrant_query_encoder_model(
                backend=backend,
                text_model=text_model,
                image_query_model=image_query_model,
                uses_default_text=uses_default_text,
            ),
            "dimension": getattr(embedder, "embedding_dim", None),
        }
    return details


def qdrant_query_encoder_model(
    backend: str,
    text_model: str,
    image_query_model: str,
    uses_default_text: bool,
) -> str | None:
    if uses_default_text:
        if backend in {"sentence-transformers", "sentence_transformers", "st"}:
            return text_model
        return None
    if backend in {"clip", "transformers"}:
        return image_query_model
    return None


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


def parse_fusion_weight_grid(specs: list[str] | None = None) -> dict[str, list[float]]:
    grid: dict[str, list[float]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise typer.BadParameter("fusion weight grids must use source=v1,v2")
        source, raw_values = spec.split("=", 1)
        source = source.strip()
        if not source:
            raise typer.BadParameter("fusion weight grid source must not be empty")
        values = []
        for raw_value in raw_values.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise typer.BadParameter(
                    f"fusion weight grid value for {source} must be numeric"
                ) from exc
            if value < 0:
                raise typer.BadParameter(
                    f"fusion weight grid value for {source} must be non-negative"
                )
            values.append(value)
        if not values:
            raise typer.BadParameter(f"fusion weight grid for {source} must not be empty")
        grid[source] = values
    return grid


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


def parse_named_int_thresholds(
    specs: list[str] | None = None,
    label: str = "threshold",
) -> dict[str, int]:
    thresholds: dict[str, int] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise typer.BadParameter(f"{label} must use name=value")
        name, raw_value = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter(f"{label} name must not be empty")
        try:
            value = int(raw_value.strip())
        except ValueError as exc:
            raise typer.BadParameter(f"{label} for {name} must be an integer") from exc
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
    if normalized in {"lexical", "rerank:lexical"}:
        return LexicalOverlapReranker(tokenizer_config=tokenizer_config)
    if normalized in {
        "cross-encoder",
        "cross_encoder",
        "sentence-transformers",
        "rerank:cross_encoder",
    }:
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
    deduplicate_tokens: bool,
):
    from chunking_docs.retrieval.qdrant_hybrid import QdrantHybridSearcher
    from chunking_docs.storage.qdrant_store import QdrantChunkStore

    collection_config = json.loads((package_dir / "qdrant_collection.json").read_text(encoding="utf-8"))
    collection_name = collection or collection_config["collection"]
    named_vectors = {
        name: int(config["size"])
        for name, config in collection_config.get("named_vectors", {}).items()
    }
    vector_notes = {
        name: str(config.get("note") or "")
        for name, config in collection_config.get("named_vectors", {}).items()
    }
    selected_vectors = [item.strip() for item in vector_names.split(",") if item.strip()]
    unknown_vectors = sorted(set(selected_vectors) - set(named_vectors))
    if unknown_vectors:
        raise typer.BadParameter(
            f"Unknown Qdrant named vectors for this package: {', '.join(unknown_vectors)}"
        )
    query_backend_options = resolve_qdrant_query_backend_options(
        package_dir=package_dir,
        selected_vectors=selected_vectors,
        text_backend=text_backend,
        text_model=text_model,
        image_query_backend=image_query_backend,
        image_query_model=image_query_model,
    )
    text_backend = query_backend_options["text_backend"]
    text_model = query_backend_options["text_model"]
    image_query_backend = query_backend_options["image_query_backend"]
    image_query_model = query_backend_options["image_query_model"]
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
    validate_qdrant_query_encoder_dimensions(
        selected_vectors=selected_vectors,
        vector_sizes=named_vectors,
        default_embedder=embedder,
        vector_embedders=vector_embedders,
        vector_notes=vector_notes,
        image_query_backend=image_query_backend,
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
            deduplicate_tokens=deduplicate_tokens,
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
        "query_encoder_details": qdrant_query_encoder_details(
            selected_vectors,
            vector_embedders=vector_embedders,
            default_embedder=embedder,
            text_backend=text_backend,
            text_model=text_model,
            image_query_backend=image_query_backend,
            image_query_model=image_query_model,
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


def load_chunking_sweep_report(path: Path) -> ChunkingSweepReport:
    if not path.exists():
        raise typer.BadParameter(f"Chunking sweep report does not exist: {path}")
    try:
        return ChunkingSweepReport.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid chunking sweep report: {path}") from exc


def select_chunking_sweep_candidate(
    report: ChunkingSweepReport,
    candidate_name: str | None,
) -> ChunkingSweepCandidate:
    selected_name = candidate_name or report.selection.recommended
    if not selected_name:
        raise typer.BadParameter(
            "No sweep candidate was selected. Pass --candidate or run sweep-chunking "
            "with constraints that leave an eligible recommendation."
        )
    for candidate in report.candidates:
        if candidate.name == selected_name:
            return candidate
    raise typer.BadParameter(f"Unknown sweep candidate: {selected_name}")


def resolve_sweep_chunks_file(
    candidate: ChunkingSweepCandidate,
    chunks_file: Path | None,
    report_path: Path,
) -> Path:
    path = chunks_file or (Path(candidate.chunks_file) if candidate.chunks_file else None)
    if path is None:
        raise typer.BadParameter(
            f"Candidate {candidate.name} does not record a chunks_file. "
            "Pass --chunks-file to apply it."
        )
    resolved = resolve_existing_path(path, base_dir=report_path.parent)
    if not resolved.exists():
        raise typer.BadParameter(f"Selected chunk file does not exist: {path}")
    return resolved


def resolve_existing_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def manifest_tokenizer_config(manifest) -> LexicalTokenizerConfig | None:
    package_config = manifest.metadata.get("package_config")
    if not isinstance(package_config, dict):
        return None
    tokenizer_payload = package_config.get("lexical_tokenizer")
    if not isinstance(tokenizer_payload, dict):
        return None
    try:
        return LexicalTokenizerConfig.model_validate(tokenizer_payload)
    except ValueError:
        return None


def update_manifest_chunking_selection(
    package_dir: Path,
    report: Path,
    candidate: ChunkingSweepCandidate,
    chunks_file: Path,
    chunk_count: int,
) -> None:
    manifest_path = package_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = payload.setdefault("metadata", {})
    package_config = metadata.setdefault("package_config", {})
    package_config["base_chunking_strategy"] = candidate.strategy
    metadata["selected_chunking_candidate"] = {
        "name": candidate.name,
        "strategy": candidate.strategy,
        "config": candidate.config,
        "sweep_report": str(report),
        "chunks_file": str(chunks_file),
        "chunk_count": chunk_count,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def parse_visual_run_inputs(values: list[str] | None) -> dict[str, Path]:
    if not values:
        raise typer.BadParameter("At least one --run name=results.jsonl value is required")
    return parse_named_path_inputs(values, option_name="--run", path_label="results.jsonl")


def parse_retrieval_eval_inputs(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return {}
    return parse_named_path_inputs(
        values,
        option_name="--retrieval-eval",
        path_label="retrieval_eval.json",
    )


def parse_named_path_inputs(
    values: list[str],
    option_name: str,
    path_label: str,
) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter(f"{option_name} must be in name={path_label} form")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter(f"{option_name} name must not be empty")
        parsed[name] = Path(path)
    return parsed


def parse_strategy_list(value: str) -> list[ChunkStrategy]:
    allowed = {"page", "semantic", "multimodal", "object_aware", "hierarchical"}
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


def parse_embedding_mode(value: str) -> bool | None:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return None
    if normalized in {"dry_run", "dryrun", "hashing", "hashing_dry_run"}:
        return True
    if normalized in {"external", "model", "model_backed", "real"}:
        return False
    raise typer.BadParameter("--embedding-mode must be one of: auto, dry_run, external")


def parse_auto_bool(value: str, option_name: str) -> bool | None:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise typer.BadParameter(f"{option_name} must be one of: auto, true, false")


def build_tokenizer_config(
    strategy: TokenizerStrategy,
    ngram_min: int,
    ngram_max: int,
    ngram_cjk_only: bool,
    deduplicate_tokens: bool,
) -> LexicalTokenizerConfig:
    return LexicalTokenizerConfig(
        strategy=strategy,
        min_n=ngram_min,
        max_n=ngram_max,
        ngram_cjk_only=ngram_cjk_only,
        deduplicate=deduplicate_tokens,
    )

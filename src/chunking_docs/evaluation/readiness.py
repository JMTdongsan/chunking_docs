from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from chunking_docs.evaluation.audit import PackageAudit, audit_package
from chunking_docs.evaluation.ablation import (
    QdrantVectorAblationGateReport,
    QdrantVectorAblationReport,
    RetrievalAblationGateReport,
    RetrievalAblationReport,
    gate_retrieval_ablation,
    gate_qdrant_vector_ablation,
)
from chunking_docs.evaluation.case_audit import RetrievalCaseAuditReport, audit_retrieval_cases
from chunking_docs.evaluation.chunking_gate import (
    ChunkingComparisonGateReport,
    gate_chunking_comparison,
)
from chunking_docs.evaluation.chunking_quality import (
    standalone_visual_text_stats,
    visual_text_coverage_stats,
)
from chunking_docs.evaluation.compare import ChunkingComparison
from chunking_docs.evaluation.gate import RetrievalGateReport, gate_retrieval_evaluation
from chunking_docs.evaluation.retrieval import RetrievalCase, RetrievalEvaluation
from chunking_docs.embeddings.bm25 import asset_text_parts, chunk_lexical_texts
from chunking_docs.embeddings.tokenizers import LexicalTokenizer, LexicalTokenizerConfig
from chunking_docs.graph.provenance import chunk_asset_ids
from chunking_docs.models import ProcessingManifest
from chunking_docs.storage.postgres_store import manifest_rows
from chunking_docs.vision.compare import VisualRunComparison
from chunking_docs.vision.jobs import VisualJobRunResult
from chunking_docs.vision.quality import (
    VisualQualityReport,
    evaluate_visual_results,
    visual_results_from_assets,
)


class ReadinessComponent(BaseModel):
    name: str
    passed: bool
    severity: str = "error"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionReadinessReport(BaseModel):
    package_dir: str
    passed: bool
    package_counts: dict[str, int]
    artifact_presence: dict[str, bool]
    postgres_row_counts: dict[str, int] = Field(default_factory=dict)
    audit: PackageAudit
    visual_quality: VisualQualityReport | None = None
    visual_run_comparison: VisualRunComparison | None = None
    retrieval_case_audit: RetrievalCaseAuditReport | None = None
    retrieval_gate: RetrievalGateReport | None = None
    chunking_comparison_gate: ChunkingComparisonGateReport | None = None
    retrieval_ablation_gate: RetrievalAblationGateReport | None = None
    qdrant_vector_ablation_gate: QdrantVectorAblationGateReport | None = None
    components: list[ReadinessComponent] = Field(default_factory=list)
    failed_components: list[str] = Field(default_factory=list)


def build_ingestion_readiness_report(
    package_dir: Path,
    manifest: ProcessingManifest,
    require_qdrant_records: bool = True,
    require_bm25: bool = True,
    require_embedding_manifest: bool = True,
    required_vectors: list[str] | None = None,
    require_postgres_rows: bool = True,
    require_visual_annotations: bool = False,
    min_visual_text_coverage_ratio: float | None = None,
    visual_results: list[VisualJobRunResult] | None = None,
    require_visual_quality: bool = False,
    visual_quality_options: dict[str, Any] | None = None,
    visual_run_comparison: VisualRunComparison | None = None,
    require_visual_run_comparison: bool = False,
    visual_run_comparison_options: dict[str, Any] | None = None,
    retrieval_cases: list[RetrievalCase] | None = None,
    require_retrieval_cases: bool = False,
    retrieval_case_options: dict[str, Any] | None = None,
    retrieval_evaluation: RetrievalEvaluation | None = None,
    require_retrieval_evaluation: bool = False,
    retrieval_gate_options: dict[str, Any] | None = None,
    chunking_comparison: ChunkingComparison | None = None,
    require_chunking_comparison: bool = False,
    chunking_gate_options: dict[str, Any] | None = None,
    retrieval_ablation: RetrievalAblationReport | None = None,
    require_retrieval_ablation: bool = False,
    retrieval_ablation_mode: str | None = None,
    retrieval_ablation_baseline_mode: str | None = None,
    retrieval_ablation_gate_options: dict[str, Any] | None = None,
    qdrant_vector_ablation: QdrantVectorAblationReport | None = None,
    require_qdrant_vector_ablation: bool = False,
    qdrant_vector_ablation_mode: str | None = None,
    qdrant_vector_ablation_gate_options: dict[str, Any] | None = None,
) -> IngestionReadinessReport:
    artifact_presence = package_artifact_presence(package_dir)
    audit = audit_package(
        manifest.profiles,
        manifest.chunks,
        manifest.assets,
        manifest.triples,
        require_annotations_for_visual_pages=require_visual_annotations,
        package_dir=package_dir,
        require_qdrant_records=require_qdrant_records,
    )
    components = [
        ReadinessComponent(
            name="package_audit",
            passed=audit.passed,
            message="Package structure, visual coverage, triples, and Qdrant records are valid.",
            metadata={"issue_codes": [issue.code for issue in audit.issues]},
        )
    ]

    if require_bm25:
        components.append(bm25_tokens_component(package_dir, manifest))
    if require_embedding_manifest:
        components.append(
            required_artifact_component(
                "embedding_manifest",
                artifact_presence,
                "embedding_manifest.json",
            )
        )
    if required_vectors:
        components.append(embedding_vectors_component(package_dir, required_vectors))

    postgres_row_counts: dict[str, int] = {}
    if require_postgres_rows:
        postgres_component, postgres_row_counts = postgres_rows_component(manifest, package_dir)
        components.append(postgres_component)

    if min_visual_text_coverage_ratio is not None:
        components.append(
            visual_text_coverage_component(
                manifest,
                min_coverage_ratio=min_visual_text_coverage_ratio,
            )
        )

    visual_quality = None
    if visual_results is not None or require_visual_quality:
        visual_quality_source = "visual_results"
        evaluated_visual_results = visual_results
        if evaluated_visual_results is None:
            visual_quality_source = "assets"
            evaluated_visual_results = visual_results_from_assets(manifest.assets)
        visual_quality = evaluate_visual_results(
            evaluated_visual_results,
            **(visual_quality_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="visual_quality",
                passed=visual_quality.passed,
                message="Visual OCR/VLM annotations meet configured quality thresholds.",
                metadata={
                    "source": visual_quality_source,
                    "failed_checks": visual_quality.failed_checks,
                    "completion_rate": visual_quality.completion_rate,
                    "ocr_text_coverage": visual_quality.ocr_text_coverage,
                    "vlm_summary_coverage": visual_quality.vlm_summary_coverage,
                    "vlm_json_parse_rate": visual_quality.vlm_json_parse_rate,
                    "vlm_object_coverage": visual_quality.vlm_object_coverage,
                    "objects_per_vlm_job": visual_quality.objects_per_vlm_job,
                    "object_bbox_coverage": visual_quality.object_bbox_coverage,
                },
            )
        )

    if visual_run_comparison is not None:
        components.append(
            visual_run_comparison_component(
                visual_run_comparison,
                **(visual_run_comparison_options or {}),
            )
        )
    elif require_visual_run_comparison:
        components.append(
            ReadinessComponent(
                name="visual_run_comparison",
                passed=False,
                message="Visual run comparison is required but was not supplied.",
            )
        )

    retrieval_case_audit = None
    if retrieval_cases is not None:
        retrieval_case_audit = audit_retrieval_cases(
            retrieval_cases,
            profiles=manifest.profiles,
            chunks=manifest.chunks,
            assets=manifest.assets,
            triples=manifest.triples,
            **(retrieval_case_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="retrieval_case_audit",
                passed=retrieval_case_audit.passed,
                message="Retrieval benchmark cases are valid for the package and configured target coverage.",
                metadata={
                    "failed_checks": retrieval_case_audit.failed_checks,
                    "target_counts": retrieval_case_audit.target_counts,
                    "missing_target_counts": retrieval_case_audit.missing_target_counts,
                },
            )
        )
    elif require_retrieval_cases:
        components.append(
            ReadinessComponent(
                name="retrieval_case_audit",
                passed=False,
                message="Retrieval benchmark cases are required but were not supplied.",
            )
        )

    retrieval_gate = None
    if retrieval_evaluation is not None:
        retrieval_gate = gate_retrieval_evaluation(
            retrieval_evaluation,
            **(retrieval_gate_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="retrieval_gate",
                passed=retrieval_gate.passed,
                message="Retrieval evaluation meets configured quality thresholds.",
                metadata={
                    "failed_checks": retrieval_gate.failed_checks,
                    "metrics": retrieval_gate.metrics,
                    "target_metrics": retrieval_gate.target_metrics,
                    "source_family_metrics": retrieval_gate.source_family_metrics,
                    "case_group_metrics": retrieval_gate.case_group_metrics,
                },
            )
        )
    elif require_retrieval_evaluation:
        components.append(
            ReadinessComponent(
                name="retrieval_gate",
                passed=False,
                message="Retrieval evaluation is required but was not supplied.",
            )
        )

    chunking_comparison_gate = None
    if chunking_comparison is not None:
        chunking_comparison_gate = gate_chunking_comparison(
            chunking_comparison,
            **(chunking_gate_options or {}),
        )
        components.append(
            ReadinessComponent(
                name="chunking_comparison_gate",
                passed=chunking_comparison_gate.passed,
                message="Selected chunking candidate meets configured quality, retrieval, and latency thresholds.",
                metadata={
                    "candidate": chunking_comparison_gate.candidate,
                    "baseline_candidate": chunking_comparison_gate.baseline_candidate,
                    "failed_checks": chunking_comparison_gate.failed_checks,
                    "metrics": chunking_comparison_gate.metrics,
                    "target_metrics": chunking_comparison_gate.target_metrics,
                    "source_family_metrics": chunking_comparison_gate.source_family_metrics,
                    "chunk_strategy_metrics": chunking_comparison_gate.chunk_strategy_metrics,
                    "retrieval_role_metrics": chunking_comparison_gate.retrieval_role_metrics,
                    "case_group_metrics": chunking_comparison_gate.case_group_metrics,
                    "pairwise_metrics": chunking_comparison_gate.pairwise_metrics,
                },
            )
        )
    elif require_chunking_comparison:
        components.append(
            ReadinessComponent(
                name="chunking_comparison_gate",
                passed=False,
                message="Chunking comparison gate is required but no comparison was supplied.",
            )
        )

    retrieval_ablation_gate = None
    if retrieval_ablation is not None:
        if retrieval_ablation_mode is None:
            components.append(
                ReadinessComponent(
                    name="retrieval_ablation_gate",
                    passed=False,
                    message="Retrieval ablation report was supplied but no mode was selected.",
                )
            )
        else:
            try:
                retrieval_ablation_gate = gate_retrieval_ablation(
                    retrieval_ablation,
                    mode=retrieval_ablation_mode,
                    baseline_mode=retrieval_ablation_baseline_mode,
                    **(retrieval_ablation_gate_options or {}),
                )
                components.append(
                    ReadinessComponent(
                        name="retrieval_ablation_gate",
                        passed=retrieval_ablation_gate.passed,
                        message="Selected retrieval ablation mode meets configured thresholds and lift checks.",
                        metadata={
                            "mode": retrieval_ablation_gate.mode,
                            "baseline_mode": retrieval_ablation_gate.baseline_mode,
                            "failed_checks": retrieval_ablation_gate.failed_checks,
                            "metrics": retrieval_ablation_gate.metrics,
                            "baseline_metrics": retrieval_ablation_gate.baseline_metrics,
                            "target_metrics": retrieval_ablation_gate.target_metrics,
                            "source_family_metrics": retrieval_ablation_gate.source_family_metrics,
                            "chunk_strategy_metrics": retrieval_ablation_gate.chunk_strategy_metrics,
                            "retrieval_role_metrics": retrieval_ablation_gate.retrieval_role_metrics,
                            "best_by_recall": retrieval_ablation_gate.best_by_recall,
                            "best_by_target_coverage": (
                                retrieval_ablation_gate.best_by_target_coverage
                            ),
                            "best_by_target_ndcg": retrieval_ablation_gate.best_by_target_ndcg,
                            "fastest_by_mean_latency": (
                                retrieval_ablation_gate.fastest_by_mean_latency
                            ),
                        },
                    )
                )
            except ValueError as exc:
                components.append(
                    ReadinessComponent(
                        name="retrieval_ablation_gate",
                        passed=False,
                        message="Retrieval ablation gate could not be evaluated.",
                        metadata={
                            "error": str(exc),
                            "mode": retrieval_ablation_mode,
                            "baseline_mode": retrieval_ablation_baseline_mode,
                        },
                    )
                )
    elif require_retrieval_ablation:
        components.append(
            ReadinessComponent(
                name="retrieval_ablation_gate",
                passed=False,
                message="Retrieval ablation gate is required but no report was supplied.",
            )
        )

    qdrant_vector_ablation_gate = None
    if qdrant_vector_ablation is not None:
        if qdrant_vector_ablation_mode is None:
            components.append(
                ReadinessComponent(
                    name="qdrant_vector_ablation_gate",
                    passed=False,
                    message="Qdrant vector ablation report was supplied but no mode was selected.",
                )
            )
        else:
            try:
                qdrant_vector_ablation_gate = gate_qdrant_vector_ablation(
                    qdrant_vector_ablation,
                    mode=qdrant_vector_ablation_mode,
                    **(qdrant_vector_ablation_gate_options or {}),
                )
                components.append(
                    ReadinessComponent(
                        name="qdrant_vector_ablation_gate",
                        passed=qdrant_vector_ablation_gate.passed,
                        message="Selected Qdrant vector ablation mode meets configured retrieval thresholds.",
                        metadata={
                            "mode": qdrant_vector_ablation_gate.mode,
                            "vector_names": qdrant_vector_ablation_gate.vector_names,
                            "failed_checks": qdrant_vector_ablation_gate.failed_checks,
                            "metrics": qdrant_vector_ablation_gate.metrics,
                            "target_metrics": qdrant_vector_ablation_gate.target_metrics,
                            "source_family_metrics": (
                                qdrant_vector_ablation_gate.source_family_metrics
                            ),
                            "chunk_strategy_metrics": (
                                qdrant_vector_ablation_gate.chunk_strategy_metrics
                            ),
                            "retrieval_role_metrics": (
                                qdrant_vector_ablation_gate.retrieval_role_metrics
                            ),
                            "best_by_recall": qdrant_vector_ablation_gate.best_by_recall,
                            "best_by_target_coverage": qdrant_vector_ablation_gate.best_by_target_coverage,
                            "best_by_target_ndcg": qdrant_vector_ablation_gate.best_by_target_ndcg,
                            "fastest_by_mean_latency": qdrant_vector_ablation_gate.fastest_by_mean_latency,
                        },
                    )
                )
            except ValueError as exc:
                components.append(
                    ReadinessComponent(
                        name="qdrant_vector_ablation_gate",
                        passed=False,
                        message="Qdrant vector ablation gate could not be evaluated.",
                        metadata={"error": str(exc), "mode": qdrant_vector_ablation_mode},
                    )
                )
    elif require_qdrant_vector_ablation:
        components.append(
            ReadinessComponent(
                name="qdrant_vector_ablation_gate",
                passed=False,
                message="Qdrant vector ablation gate is required but no report was supplied.",
            )
        )

    failed_components = [
        component.name
        for component in components
        if not component.passed and component.severity == "error"
    ]
    return IngestionReadinessReport(
        package_dir=str(package_dir),
        passed=not failed_components,
        package_counts={
            "pages": len(manifest.profiles),
            "chunks": len(manifest.chunks),
            "assets": len(manifest.assets),
            "triples": len(manifest.triples),
        },
        artifact_presence=artifact_presence,
        postgres_row_counts=postgres_row_counts,
        audit=audit,
        visual_quality=visual_quality,
        visual_run_comparison=visual_run_comparison,
        retrieval_case_audit=retrieval_case_audit,
        retrieval_gate=retrieval_gate,
        chunking_comparison_gate=chunking_comparison_gate,
        retrieval_ablation_gate=retrieval_ablation_gate,
        qdrant_vector_ablation_gate=qdrant_vector_ablation_gate,
        components=components,
        failed_components=failed_components,
    )


def package_artifact_presence(package_dir: Path) -> dict[str, bool]:
    names = [
        "manifest.json",
        "pages.jsonl",
        "chunks.jsonl",
        "assets.jsonl",
        "triples.jsonl",
        "bm25_tokens.json",
        "embedding_manifest.json",
        "qdrant_collection.json",
        "qdrant_text_records.jsonl",
        "qdrant_caption_records.jsonl",
        "qdrant_image_records.jsonl",
    ]
    return {name: (package_dir / name).exists() for name in names}


def required_artifact_component(
    name: str,
    artifact_presence: dict[str, bool],
    filename: str,
) -> ReadinessComponent:
    return ReadinessComponent(
        name=name,
        passed=artifact_presence.get(filename, False),
        message=f"Required package artifact exists: {filename}.",
        metadata={"file": filename},
    )


def embedding_vectors_component(
    package_dir: Path,
    required_vectors: list[str],
) -> ReadinessComponent:
    normalized_vectors = sorted({vector.strip() for vector in required_vectors if vector.strip()})
    collection_path = package_dir / "qdrant_collection.json"
    manifest_path = package_dir / "embedding_manifest.json"
    metadata: dict[str, Any] = {
        "required_vectors": normalized_vectors,
        "collection_file": collection_path.name,
        "manifest_file": manifest_path.name,
    }
    if not normalized_vectors:
        return ReadinessComponent(
            name="embedding_vectors",
            passed=True,
            message="No required embedding vectors were configured.",
            metadata=metadata,
        )

    payloads = load_embedding_contract_payloads(collection_path, manifest_path)
    if payloads.get("error"):
        metadata.update(payloads)
        return ReadinessComponent(
            name="embedding_vectors",
            passed=False,
            message="Embedding vector contract files are missing or invalid.",
            metadata=metadata,
        )

    collection = payloads["collection"]
    manifest = payloads["manifest"]
    named_vectors = collection.get("named_vectors")
    manifest_vectors = manifest.get("vectors")
    if not isinstance(named_vectors, dict) or not isinstance(manifest_vectors, dict):
        metadata.update(
            {
                "has_named_vectors": isinstance(named_vectors, dict),
                "has_manifest_vectors": isinstance(manifest_vectors, dict),
            }
        )
        return ReadinessComponent(
            name="embedding_vectors",
            passed=False,
            message="Embedding vector contract files must include named vector and manifest vector objects.",
            metadata=metadata,
        )

    missing_collection_vectors: list[str] = []
    missing_manifest_vectors: list[str] = []
    missing_record_files: list[str] = []
    empty_record_vectors: list[str] = []
    dimension_mismatches: dict[str, dict[str, int | None]] = {}
    required_vector_details: dict[str, dict[str, Any]] = {}
    for vector_name in normalized_vectors:
        collection_vector = named_vectors.get(vector_name)
        manifest_vector = manifest_vectors.get(vector_name)
        if not isinstance(collection_vector, dict):
            missing_collection_vectors.append(vector_name)
            continue
        if not isinstance(manifest_vector, dict):
            missing_manifest_vectors.append(vector_name)
            continue

        collection_dimension = vector_dimension(collection_vector, "size")
        manifest_dimension = vector_dimension(manifest_vector, "dimension")
        if (
            collection_dimension is not None
            and manifest_dimension is not None
            and collection_dimension != manifest_dimension
        ):
            dimension_mismatches[vector_name] = {
                "collection": collection_dimension,
                "manifest": manifest_dimension,
            }

        record_file = str(manifest_vector.get("file") or "")
        record_path = package_dir / record_file if record_file else None
        if record_path is None or not record_path.exists():
            missing_record_files.append(vector_name)
        record_count = manifest_vector.get("record_count")
        if not isinstance(record_count, int) or record_count <= 0:
            empty_record_vectors.append(vector_name)

        required_vector_details[vector_name] = {
            "file": record_file,
            "record_count": record_count,
            "dimension": manifest_dimension,
            "collection_dimension": collection_dimension,
            "distance": manifest_vector.get("distance"),
            "note": manifest_vector.get("note"),
            "embedding": manifest_vector.get("embedding", {}),
            "bytes": manifest_vector.get("bytes"),
            "sha256": manifest_vector.get("sha256"),
        }

    metadata.update(
        {
            "required_vector_details": required_vector_details,
            "missing_collection_vectors": missing_collection_vectors,
            "missing_manifest_vectors": missing_manifest_vectors,
            "missing_record_files": missing_record_files,
            "empty_record_vectors": empty_record_vectors,
            "dimension_mismatches": dimension_mismatches,
        }
    )
    passed = not any(
        [
            missing_collection_vectors,
            missing_manifest_vectors,
            missing_record_files,
            empty_record_vectors,
            dimension_mismatches,
        ]
    )
    return ReadinessComponent(
        name="embedding_vectors",
        passed=passed,
        message=(
            "Required embedding vector families are configured and have record files."
            if passed
            else "Required embedding vector families are missing, empty, or inconsistent."
        ),
        metadata=metadata,
    )


def load_embedding_contract_payloads(collection_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not collection_path.exists():
        return {"error": "missing_qdrant_collection"}
    if not manifest_path.exists():
        return {"error": "missing_embedding_manifest"}
    try:
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": "invalid_json", "detail": str(exc)}
    if not isinstance(collection, dict) or not isinstance(manifest, dict):
        return {"error": "invalid_payload_type"}
    return {"collection": collection, "manifest": manifest}


def vector_dimension(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def visual_run_comparison_component(
    comparison: VisualRunComparison,
    min_run_count: int = 2,
    require_same_jobs: bool = False,
    min_shared_job_count: int = 0,
    expected_best_by_quality: str | None = None,
    expected_best_by_triple_density: str | None = None,
) -> ReadinessComponent:
    failed_checks = []
    run_count = len(comparison.rows)
    if run_count < min_run_count:
        failed_checks.append("min_run_count")
    if require_same_jobs and comparison.job_set_mismatch:
        failed_checks.append("same_job_set")
    if comparison.shared_job_count < min_shared_job_count:
        failed_checks.append("min_shared_job_count")
    if expected_best_by_quality and comparison.best_by_quality != expected_best_by_quality:
        failed_checks.append("expected_best_by_quality")
    if (
        expected_best_by_triple_density
        and comparison.best_by_triple_density != expected_best_by_triple_density
    ):
        failed_checks.append("expected_best_by_triple_density")

    return ReadinessComponent(
        name="visual_run_comparison",
        passed=not failed_checks,
        message="Visual OCR/VLM comparison uses a comparable job set and meets selection checks.",
        metadata={
            "failed_checks": failed_checks,
            "run_count": run_count,
            "min_run_count": min_run_count,
            "best_by_quality": comparison.best_by_quality,
            "expected_best_by_quality": expected_best_by_quality,
            "fastest_by_total_latency": comparison.fastest_by_total_latency,
            "best_by_triple_density": comparison.best_by_triple_density,
            "expected_best_by_triple_density": expected_best_by_triple_density,
            "job_set_mismatch": comparison.job_set_mismatch,
            "require_same_jobs": require_same_jobs,
            "union_job_count": comparison.union_job_count,
            "shared_job_count": comparison.shared_job_count,
            "min_shared_job_count": min_shared_job_count,
            "run_job_counts": comparison.run_job_counts,
            "missing_job_ids_by_run": comparison.missing_job_ids_by_run,
        },
    )


def visual_text_coverage_component(
    manifest: ProcessingManifest,
    min_coverage_ratio: float,
) -> ReadinessComponent:
    stats = visual_text_coverage_stats(manifest.chunks, manifest.assets)
    standalone_stats = standalone_visual_text_stats(manifest.chunks, manifest.assets)
    coverage_ratio = float(stats["coverage_ratio"])
    asset_count = int(stats["asset_count"])
    passed = coverage_ratio >= min_coverage_ratio
    return ReadinessComponent(
        name="visual_text_coverage",
        passed=passed,
        message=(
            "Linked visual asset text is represented in package chunks."
            if passed
            else "Some linked visual asset text is not represented in package chunks."
        ),
        metadata={
            "min_coverage_ratio": min_coverage_ratio,
            "visual_text_asset_count": asset_count,
            "visual_text_covered_asset_count": int(stats["covered_asset_count"]),
            "visual_text_coverage_ratio": coverage_ratio,
            "missing_asset_ids": list(stats["missing_asset_ids"])[:50],
            "standalone_visual_chunk_count": int(standalone_stats["chunk_count"]),
            "standalone_visual_text_asset_count": int(standalone_stats["asset_count"]),
            "standalone_visual_text_asset_ids": list(standalone_stats["asset_ids"])[:50],
        },
    )


def bm25_tokens_component(
    package_dir: Path,
    manifest: ProcessingManifest,
) -> ReadinessComponent:
    path = package_dir / "bm25_tokens.json"
    if not path.exists():
        return ReadinessComponent(
            name="bm25_tokens",
            passed=False,
            message="Required package artifact exists and matches asset-enriched chunk text: bm25_tokens.json.",
            metadata={"file": "bm25_tokens.json", "error": "missing_file"},
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ReadinessComponent(
            name="bm25_tokens",
            passed=False,
            message="bm25_tokens.json is not valid JSON.",
            metadata={"file": "bm25_tokens.json", "error": str(exc)},
        )

    tokenizer_payload = payload.get("tokenizer")
    chunks_payload = payload.get("chunks")
    if not isinstance(tokenizer_payload, dict) or not isinstance(chunks_payload, list):
        return ReadinessComponent(
            name="bm25_tokens",
            passed=False,
            message="bm25_tokens.json must contain tokenizer and chunks entries.",
            metadata={
                "file": "bm25_tokens.json",
                "has_tokenizer": isinstance(tokenizer_payload, dict),
                "has_chunks": isinstance(chunks_payload, list),
            },
        )

    try:
        tokenizer_config = LexicalTokenizerConfig.model_validate(tokenizer_payload)
    except ValidationError as exc:
        return ReadinessComponent(
            name="bm25_tokens",
            passed=False,
            message="bm25_tokens.json tokenizer configuration is invalid.",
            metadata={"file": "bm25_tokens.json", "errors": exc.errors()},
        )

    tokenizer = LexicalTokenizer(tokenizer_config)
    expected_texts = chunk_lexical_texts(manifest.chunks, manifest.assets)
    expected_chunk_ids = {chunk.chunk_id for chunk in manifest.chunks}
    entry_by_chunk_id: dict[str, dict[str, Any]] = {}
    duplicate_chunk_ids: list[str] = []
    invalid_entry_count = 0

    for entry in chunks_payload:
        if not isinstance(entry, dict):
            invalid_entry_count += 1
            continue
        chunk_id = entry.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            invalid_entry_count += 1
            continue
        if chunk_id in entry_by_chunk_id:
            duplicate_chunk_ids.append(chunk_id)
        entry_by_chunk_id[chunk_id] = entry

    observed_chunk_ids = set(entry_by_chunk_id)
    missing_chunk_ids = sorted(expected_chunk_ids - observed_chunk_ids)
    stale_chunk_ids = sorted(observed_chunk_ids - expected_chunk_ids)
    token_mismatch_chunk_ids: list[str] = []
    text_char_count_mismatch_chunk_ids: list[str] = []
    invalid_token_chunk_ids: list[str] = []
    invalid_text_char_count_chunk_ids: list[str] = []

    for chunk, expected_text in zip(manifest.chunks, expected_texts):
        entry = entry_by_chunk_id.get(chunk.chunk_id)
        if entry is None:
            continue
        tokens = entry.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            invalid_token_chunk_ids.append(chunk.chunk_id)
        elif tokens != tokenizer.tokenize(expected_text):
            token_mismatch_chunk_ids.append(chunk.chunk_id)

        text_char_count = entry.get("text_char_count")
        if not isinstance(text_char_count, int):
            invalid_text_char_count_chunk_ids.append(chunk.chunk_id)
        elif text_char_count != len(expected_text):
            text_char_count_mismatch_chunk_ids.append(chunk.chunk_id)

    linked_asset_text_chunk_ids = chunks_with_linked_asset_text(manifest)
    inconsistent_chunk_ids = {
        *missing_chunk_ids,
        *token_mismatch_chunk_ids,
        *text_char_count_mismatch_chunk_ids,
        *invalid_token_chunk_ids,
        *invalid_text_char_count_chunk_ids,
    }
    indexed_linked_asset_text_chunk_ids = sorted(
        chunk_id for chunk_id in linked_asset_text_chunk_ids if chunk_id not in inconsistent_chunk_ids
    )
    missing_linked_asset_text_chunk_ids = sorted(
        set(linked_asset_text_chunk_ids) - set(indexed_linked_asset_text_chunk_ids)
    )

    metadata = {
        "file": "bm25_tokens.json",
        "tokenizer": tokenizer_config.model_dump(),
        "expected_chunk_count": len(manifest.chunks),
        "manifest_chunk_count": len(chunks_payload),
        "missing_chunk_count": len(missing_chunk_ids),
        "stale_chunk_count": len(stale_chunk_ids),
        "duplicate_chunk_count": len(duplicate_chunk_ids),
        "invalid_entry_count": invalid_entry_count,
        "token_mismatch_count": len(token_mismatch_chunk_ids),
        "text_char_count_mismatch_count": len(text_char_count_mismatch_chunk_ids),
        "invalid_token_chunk_count": len(invalid_token_chunk_ids),
        "invalid_text_char_count_chunk_count": len(invalid_text_char_count_chunk_ids),
        "chunks_with_linked_asset_text": len(linked_asset_text_chunk_ids),
        "indexed_linked_asset_text_chunk_count": len(indexed_linked_asset_text_chunk_ids),
        "missing_linked_asset_text_chunk_count": len(missing_linked_asset_text_chunk_ids),
        "missing_chunk_ids": missing_chunk_ids[:50],
        "stale_chunk_ids": stale_chunk_ids[:50],
        "duplicate_chunk_ids": duplicate_chunk_ids[:50],
        "token_mismatch_chunk_ids": token_mismatch_chunk_ids[:50],
        "text_char_count_mismatch_chunk_ids": text_char_count_mismatch_chunk_ids[:50],
        "invalid_token_chunk_ids": invalid_token_chunk_ids[:50],
        "invalid_text_char_count_chunk_ids": invalid_text_char_count_chunk_ids[:50],
        "missing_linked_asset_text_chunk_ids": missing_linked_asset_text_chunk_ids[:50],
    }
    passed = not any(
        [
            missing_chunk_ids,
            stale_chunk_ids,
            duplicate_chunk_ids,
            invalid_entry_count,
            token_mismatch_chunk_ids,
            text_char_count_mismatch_chunk_ids,
            invalid_token_chunk_ids,
            invalid_text_char_count_chunk_ids,
            missing_linked_asset_text_chunk_ids,
        ]
    )
    return ReadinessComponent(
        name="bm25_tokens",
        passed=passed,
        message=(
            "BM25 token manifest covers every chunk and matches asset-enriched lexical text."
            if passed
            else "BM25 token manifest is missing, stale, or does not match asset-enriched chunk text."
        ),
        metadata=metadata,
    )


def chunks_with_linked_asset_text(manifest: ProcessingManifest) -> list[str]:
    asset_by_id = {asset.asset_id: asset for asset in manifest.assets}
    chunk_ids: list[str] = []
    for chunk in manifest.chunks:
        for asset_id in chunk_asset_ids(chunk):
            asset = asset_by_id.get(asset_id)
            if asset is not None and asset_text_parts(asset):
                chunk_ids.append(chunk.chunk_id)
                break
    return chunk_ids


def postgres_rows_component(
    manifest: ProcessingManifest,
    package_dir: Path,
) -> tuple[ReadinessComponent, dict[str, int]]:
    try:
        rows = manifest_rows(manifest, base_dir=package_dir)
    except Exception as exc:
        return (
            ReadinessComponent(
                name="postgres_rows",
                passed=False,
                message="PostgreSQL row conversion failed.",
                metadata={"error": str(exc)},
            ),
            {},
        )
    counts = {
        "documents": 1,
        "pages": len(rows["pages"]),
        "chunks": len(rows["chunks"]),
        "chunk_lexical_tokens": len(rows["chunk_lexical_tokens"]),
        "assets": len(rows["assets"]),
        "chunk_asset_links": len(rows["chunk_asset_links"]),
        "triples": len(rows["triples"]),
        "embedding_artifacts": len(rows["embedding_artifacts"]),
    }
    return (
        ReadinessComponent(
            name="postgres_rows",
            passed=counts["chunks"] > 0 and counts["pages"] > 0,
            message="Package can be converted into PostgreSQL metadata rows.",
            metadata={"row_counts": counts},
        ),
        counts,
    )

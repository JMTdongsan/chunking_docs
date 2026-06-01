from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.compare import ChunkingComparison, compare_chunking_reports
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.io import read_jsonl
from chunking_docs.models import DocumentChunk, ProcessingManifest


class ArtifactSummary(BaseModel):
    path: str
    exists: bool
    bytes: int = 0
    sha256: str | None = None
    record_count: int | None = None


class ValidationSummary(BaseModel):
    path: str
    kind: str
    passed: bool | None = None
    failed_checks: list[str] = Field(default_factory=list)
    failed_components: list[str] = Field(default_factory=list)
    candidate: str | None = None
    mode: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentReport(BaseModel):
    generated_at: str
    package_dir: str
    doc_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    candidate_files: dict[str, str] = Field(default_factory=dict)
    package_counts: dict[str, int] = Field(default_factory=dict)
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    source_file: dict[str, Any] = Field(default_factory=dict)
    package_config: dict[str, Any] = Field(default_factory=dict)
    qdrant_collection: dict[str, Any] = Field(default_factory=dict)
    bm25_tokenizer: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    validation_summaries: list[ValidationSummary] = Field(default_factory=list)
    comparison: ChunkingComparison | None = None


DEFAULT_ARTIFACTS = [
    "manifest.json",
    "pages.jsonl",
    "chunks.jsonl",
    "assets.jsonl",
    "triples.jsonl",
    "bm25_tokens.json",
    "embedding_manifest.json",
    "postgres_schema_contract.json",
    "qdrant_collection.json",
    "qdrant_collection_contract.json",
    "qdrant_text_records.jsonl",
    "qdrant_caption_records.jsonl",
    "qdrant_object_records.jsonl",
    "qdrant_image_records.jsonl",
    "qdrant_triple_records.jsonl",
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "graph_summary.json",
    "visual_job_summary.json",
    "visual_run_comparison.json",
    "visual_quality.json",
    "document_characteristics.json",
    "ingestion_readiness.json",
    "chunking_comparison.json",
    "chunking_comparison_gate.json",
    "chunking_sweep.json",
    "qdrant_retrieval_eval.json",
    "qdrant_vector_ablation.json",
    "qdrant_reranker_ablation.json",
    "qdrant_fusion_sweep.json",
    "qdrant_retrieval_config.json",
    "qdrant_retrieval_config_eval.json",
    "qdrant_rag_context_config_eval.json",
    "qdrant_rag_context_gate.json",
    "retrieval_case_audit.json",
    "retrieval_gate.json",
    "retrieval_ablation.json",
    "rag_context.json",
    "rag_context.qdrant.json",
    "rag_context.config.cases.jsonl",
]

DEFAULT_ARTIFACT_GLOBS = [
    "ingestion_readiness*.json",
    "chunking_comparison*.json",
    "chunking_sweep*.json",
    "chunking_gate*.json",
    "graph_audit*.json",
    "package_delta*.json",
    "qdrant_collection_contract*.json",
    "qdrant_eval*.json",
    "retrieval_gate*.json",
    "retrieval_eval*.json",
    "retrieval_ablation*.json",
    "retrieval_case_audit*.json",
    "retrieval_diagnostics*.json",
    "qdrant_retrieval_eval*.json",
    "qdrant_rag_context_config_eval*.json",
    "qdrant_rag_context_gate*.json",
    "qdrant_vector_ablation*.json",
    "qdrant_vector_ablation_gate*.json",
    "qdrant_reranker_ablation*.json",
    "qdrant_reranker_ablation_gate*.json",
    "qdrant_fusion_sweep*.json",
    "qdrant_retrieval_config*.json",
    "visual_asset_gate*.json",
    "visual_gate*.json",
    "visual_quality*.json",
    "visual_run_comparison*.json",
]

SUMMARY_METRIC_KEYS = {
    "hit_rate",
    "recall_at_k",
    "retrieval_recall_at_k",
    "mrr",
    "retrieval_mrr",
    "target_coverage_at_k",
    "target_coverage",
    "retrieval_target_coverage_at_k",
    "mean_target_ndcg_at_k",
    "retrieval_mean_target_ndcg_at_k",
    "mean_precision_at_k",
    "retrieval_mean_precision_at_k",
    "mean_first_relevant_rank",
    "retrieval_mean_first_relevant_rank",
    "p95_first_relevant_rank",
    "retrieval_p95_first_relevant_rank",
    "mean_target_rank",
    "retrieval_mean_target_rank",
    "p95_target_rank",
    "retrieval_p95_target_rank",
    "ranked_expected_case_count",
    "retrieval_ranked_expected_case_count",
    "ranked_target_count",
    "retrieval_ranked_target_count",
    "mean_latency_ms",
    "retrieval_mean_latency_ms",
    "p95_latency_ms",
    "retrieval_p95_latency_ms",
    "total_query_latency_ms",
    "excluded_target_hit_rate",
    "mean_context_char_count",
    "max_context_char_count",
    "mean_chunk_count",
    "mean_asset_count",
    "mean_triple_count",
    "unstable_result_count",
    "retrieval_unstable_result_count",
    "result_stability_rate",
    "retrieval_result_stability_rate",
    "index_build_ms",
    "case_count",
    "expected_case_count",
    "passed_count",
    "failed_count",
    "partial_count",
    "no_hit_count",
    "low_precision_count",
    "low_target_ndcg_count",
    "failed_query_count",
    "run_count",
    "union_job_count",
    "shared_job_count",
    "job_set_mismatch",
    "best_quality_score",
    "best_triples_per_vlm_job",
    "fastest_total_mean_latency_ms",
    "completion_rate",
    "ocr_text_coverage",
    "vlm_summary_coverage",
    "vlm_json_parse_rate",
    "visual_text_asset_count",
    "visual_text_covered_asset_count",
    "visual_text_coverage_ratio",
    "visual_text_part_count",
    "visual_text_covered_part_count",
    "visual_text_part_coverage_ratio",
    "visual_object_chunk_count",
    "annotation_rate",
    "triple_count",
    "orphan_count",
    "duplicate_count",
    "empty_field_count",
    "invalid_confidence_count",
    "target_type.asset.coverage_at_k",
    "target_type.triple.coverage_at_k",
    "source_family.lexical.target_coverage_at_k",
    "source_family.visual.target_coverage_at_k",
    "source_family.graph.target_coverage_at_k",
    "chunk_strategy.semantic_subchunks.target_coverage_at_k",
    "chunk_strategy.visual_asset_text.target_coverage_at_k",
    "chunk_strategy.visual_object_text.target_coverage_at_k",
    "chunk_strategy.hierarchical_parent.target_coverage_at_k",
    "chunk_strategy.hierarchical_child.target_coverage_at_k",
    "retrieval_role.parent.target_coverage_at_k",
    "retrieval_role.child.target_coverage_at_k",
}


def build_experiment_report(
    package_dir: Path,
    manifest: ProcessingManifest,
    candidates: dict[str, Path] | None = None,
    retrieval_cases: list[RetrievalCase] | None = None,
    top_k: int = 5,
    min_chars: int = 120,
    max_chars: int = 1800,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    retrieval_repeat: int = 1,
    fusion_weights: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> ExperimentReport:
    comparison = None
    artifact_paths = package_artifact_paths(package_dir, candidates or {})
    if candidates:
        reports = {}
        for name, path in candidates.items():
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
                tokenizer_config=tokenizer_config,
                collapse_hierarchical=collapse_hierarchical,
                retrieval_repeat=retrieval_repeat,
                fusion_weights=fusion_weights,
            )
        comparison = compare_chunking_reports(reports)

    return ExperimentReport(
        generated_at=datetime.now(UTC).isoformat(),
        package_dir=str(package_dir),
        doc_id=manifest.doc.doc_id,
        config=config or {},
        candidate_files={name: str(path) for name, path in (candidates or {}).items()},
        package_counts={
            "pages": len(manifest.profiles),
            "chunks": len(manifest.chunks),
            "assets": len(manifest.assets),
            "triples": len(manifest.triples),
        },
        profile_summary=manifest.metadata.get("profile_summary", {}),
        source_file=manifest.metadata.get("source_file", {}),
        package_config=manifest.metadata.get("package_config", {}),
        qdrant_collection=read_json(package_dir / "qdrant_collection.json"),
        bm25_tokenizer=read_json(package_dir / "bm25_tokens.json").get("tokenizer", {}),
        artifacts=[artifact_summary(path, root=package_dir) for path in artifact_paths],
        validation_summaries=validation_artifact_summaries(package_dir, artifact_paths),
        comparison=comparison,
    )


def package_artifact_paths(package_dir: Path, candidates: dict[str, Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for name in DEFAULT_ARTIFACTS:
        append_unique_path(paths, seen, package_dir / name)
    for pattern in DEFAULT_ARTIFACT_GLOBS:
        for path in sorted(package_dir.glob(pattern)):
            append_unique_path(paths, seen, path)
    for path in candidates.values():
        append_unique_path(paths, seen, path)
    return paths


def package_artifact_summaries(package_dir: Path, candidates: dict[str, Path]) -> list[ArtifactSummary]:
    paths = package_artifact_paths(package_dir, candidates)
    return [artifact_summary(path, root=package_dir) for path in paths]


def append_unique_path(paths: list[Path], seen: set[Path], path: Path) -> None:
    normalized = path.resolve() if path.exists() else path
    if normalized in seen:
        return
    seen.add(normalized)
    paths.append(path)


def artifact_summary(path: Path, root: Path | None = None) -> ArtifactSummary:
    display_path = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
    if not path.exists():
        return ArtifactSummary(path=display_path, exists=False)
    content = path.read_bytes()
    return ArtifactSummary(
        path=display_path,
        exists=True,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        record_count=count_records(path),
    )


def count_records(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return None


def validation_artifact_summaries(package_dir: Path, paths: list[Path]) -> list[ValidationSummary]:
    summaries = []
    for path in paths:
        summaries.extend(validation_artifact_summaries_for_path(path, root=package_dir))
    return summaries


def validation_artifact_summaries_for_path(
    path: Path,
    root: Path | None = None,
) -> list[ValidationSummary]:
    if not path.exists() or path.suffix != ".json":
        return []
    payload = read_json(path)
    summaries = []
    if is_visual_run_comparison_payload(payload):
        return [visual_run_comparison_summary(path, payload, root=root)]
    if is_qdrant_fusion_sweep_payload(payload):
        return [qdrant_fusion_sweep_summary(path, payload, root=root)]
    if is_qdrant_retrieval_config_payload(payload):
        return [qdrant_retrieval_config_summary(path, payload, root=root)]
    if is_chunking_sweep_payload(payload):
        return [chunking_sweep_summary(path, payload, root=root)]
    if not is_validation_payload(payload):
        return component_validation_summaries(path, payload, root=root)
    display_path = display_artifact_path(path, root)
    summaries.append(
        ValidationSummary(
            path=display_path,
            kind=validation_kind(path),
            passed=payload.get("passed") if isinstance(payload.get("passed"), bool) else None,
            failed_checks=string_list(payload.get("failed_checks")),
            failed_components=string_list(payload.get("failed_components")),
            candidate=payload.get("candidate") if isinstance(payload.get("candidate"), str) else None,
            mode=payload.get("mode") if isinstance(payload.get("mode"), str) else None,
            metrics=validation_summary_metrics(payload),
        )
    )
    summaries.extend(component_validation_summaries(path, payload, root=root))
    return summaries


def component_validation_summaries(
    path: Path,
    payload: dict[str, Any],
    root: Path | None = None,
) -> list[ValidationSummary]:
    components = payload.get("components")
    if not isinstance(components, list):
        return []
    display_path = display_artifact_path(path, root)
    summaries = []
    for component in components:
        if not isinstance(component, dict):
            continue
        name = component.get("name")
        if not isinstance(name, str) or not name:
            continue
        metadata = component.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        summaries.append(
            ValidationSummary(
                path=f"{display_path}#{name}",
                kind=name,
                passed=component.get("passed") if isinstance(component.get("passed"), bool) else None,
                failed_checks=string_list(metadata.get("failed_checks")),
                failed_components=string_list(metadata.get("failed_components")),
                candidate=metadata.get("candidate") if isinstance(metadata.get("candidate"), str) else None,
                mode=metadata.get("mode") if isinstance(metadata.get("mode"), str) else None,
                metrics=validation_summary_metrics(metadata),
            )
        )
    return summaries


def is_visual_run_comparison_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("rows"), list) and (
        "best_by_quality" in payload
        or "best_by_triple_density" in payload
        or "job_set_mismatch" in payload
    )


def is_chunking_sweep_payload(payload: dict[str, Any]) -> bool:
    selection = payload.get("selection")
    return isinstance(selection, dict) and isinstance(payload.get("candidates"), list)


def is_qdrant_fusion_sweep_payload(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("candidates"), list)
        and isinstance(payload.get("vector_names"), list)
        and "recommended" in payload
        and "eligible_count" in payload
    )


def is_qdrant_retrieval_config_payload(payload: dict[str, Any]) -> bool:
    return (
        payload.get("backend") == "qdrant_hybrid"
        and isinstance(payload.get("selection"), dict)
        and isinstance(payload.get("vector_names"), list)
        and isinstance(payload.get("fusion_weights"), dict)
    )


def qdrant_retrieval_config_summary(
    path: Path,
    payload: dict[str, Any],
    root: Path | None = None,
) -> ValidationSummary:
    selection = payload.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    failed_checks = string_list(selection.get("eligibility_failures"))
    candidate_eligible = selection.get("candidate_eligible") is not False
    candidate = selection.get("candidate")
    candidate = candidate if isinstance(candidate, str) else None
    return ValidationSummary(
        path=display_artifact_path(path, root),
        kind="qdrant_retrieval_config",
        passed=candidate_eligible and not failed_checks if selection else None,
        failed_checks=failed_checks,
        candidate=candidate,
        metrics=qdrant_retrieval_config_metrics(payload, selection),
    )


def qdrant_retrieval_config_metrics(
    payload: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, float]:
    metrics = {
        "vector_count": numeric_metric(list_count(payload.get("vector_names"))),
        "fusion_weight_count": numeric_metric(dict_count(payload.get("fusion_weights"))),
        "top_k": numeric_metric(payload.get("top_k")),
        "graph_expand": 1.0 if payload.get("graph_expand") else 0.0,
        "collapse_hierarchical": 1.0 if payload.get("collapse_hierarchical") else 0.0,
        "candidate_rank": numeric_metric(selection.get("candidate_rank")),
        "candidate_eligible": 1.0 if selection.get("candidate_eligible") is not False else 0.0,
    }
    selected_metrics = selection.get("metrics")
    if isinstance(selected_metrics, dict):
        for key, value in selected_metrics.items():
            numeric_value = optional_numeric_metric(value)
            if numeric_value is not None:
                metrics[key] = numeric_value
    case_group_metrics = selection.get("case_group_metrics")
    if isinstance(case_group_metrics, dict):
        for key, value in case_group_metrics.items():
            numeric_value = optional_numeric_metric(value)
            if numeric_value is not None:
                metrics[f"case_group_selection.{key}"] = numeric_value
    return metrics


def dict_count(value: Any) -> int:
    return len(value) if isinstance(value, dict) else 0


def qdrant_fusion_sweep_summary(
    path: Path,
    payload: dict[str, Any],
    root: Path | None = None,
) -> ValidationSummary:
    recommended = payload.get("recommended")
    recommended = recommended if isinstance(recommended, str) else None
    selected = qdrant_fusion_sweep_selected_candidate(payload, recommended)
    failed_checks = []
    if recommended is None:
        failed_checks = qdrant_fusion_sweep_failed_constraints(payload)
    return ValidationSummary(
        path=display_artifact_path(path, root),
        kind="qdrant_fusion_sweep",
        passed=True if recommended else False,
        failed_checks=failed_checks,
        candidate=recommended,
        metrics=qdrant_fusion_sweep_metrics(payload, selected),
    )


def qdrant_fusion_sweep_selected_candidate(
    payload: dict[str, Any],
    recommended: str | None,
) -> dict[str, Any]:
    candidates = [row for row in payload.get("candidates", []) if isinstance(row, dict)]
    if recommended:
        for row in candidates:
            if row.get("name") == recommended:
                return row
    return candidates[0] if candidates else {}


def qdrant_fusion_sweep_failed_constraints(payload: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    failed: list[str] = []
    for row in payload.get("candidates", []):
        if not isinstance(row, dict):
            continue
        for failure in row.get("eligibility_failures", []):
            if not isinstance(failure, str) or failure in seen:
                continue
            seen.add(failure)
            failed.append(failure)
    return failed


def qdrant_fusion_sweep_metrics(
    payload: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, float]:
    metrics = {
        "candidate_count": numeric_metric(payload.get("candidate_count")),
        "eligible_count": numeric_metric(payload.get("eligible_count")),
        "selection_score": numeric_metric(selected.get("selection_score")),
    }
    evaluation = selected.get("evaluation")
    if isinstance(evaluation, dict):
        for key in [
            "recall_at_k",
            "target_coverage_at_k",
            "mean_target_ndcg_at_k",
            "mrr",
            "mean_precision_at_k",
            "mean_latency_ms",
            "p95_latency_ms",
        ]:
            if (value := optional_numeric_metric(evaluation.get(key))) is not None:
                metrics[key] = value
        failed_queries = evaluation.get("failed_queries")
        if isinstance(failed_queries, list):
            metrics["failed_query_count"] = float(len(failed_queries))
    metrics.update(qdrant_fusion_case_group_recommendation_metrics(payload))
    return metrics


def qdrant_fusion_case_group_recommendation_metrics(payload: dict[str, Any]) -> dict[str, float]:
    recommendations = payload.get("case_group_recommendations")
    if not isinstance(recommendations, dict):
        return {}
    metrics: dict[str, float] = {
        "case_group_recommendation_count": float(
            sum(
                1
                for group_values in recommendations.values()
                if isinstance(group_values, dict)
                for recommendation in group_values.values()
                if isinstance(recommendation, dict)
            )
        )
    }
    for group_name, group_values in recommendations.items():
        if not isinstance(group_name, str) or not isinstance(group_values, dict):
            continue
        for group_value, recommendation in group_values.items():
            if not isinstance(group_value, str) or not isinstance(recommendation, dict):
                continue
            prefix = f"case_group_recommendation.{group_name}.{group_value}"
            for key in ("candidate_count", "eligible_count"):
                if (value := optional_numeric_metric(recommendation.get(key))) is not None:
                    metrics[f"{prefix}.{key}"] = value
            metrics[f"{prefix}.recommended_from_globally_eligible"] = (
                1.0 if recommendation.get("recommended_from_globally_eligible") else 0.0
            )
            top_candidate = first_mapping(recommendation.get("top_candidates"))
            if top_candidate:
                metrics.update(qdrant_fusion_case_group_candidate_metrics(prefix, top_candidate))
    return metrics


def first_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {}
    return next((item for item in value if isinstance(item, dict)), {})


def qdrant_fusion_case_group_candidate_metrics(
    prefix: str,
    candidate: dict[str, Any],
) -> dict[str, float]:
    metrics = {}
    for key in (
        "global_rank",
        "selection_score",
        "case_count",
        "recall_at_k",
        "target_coverage_at_k",
        "ndcg_at_k",
        "mrr",
        "precision_at_k",
        "mean_latency_ms",
        "failed_query_count",
    ):
        if (value := optional_numeric_metric(candidate.get(key))) is not None:
            metrics[f"{prefix}.top_candidate.{key}"] = value
    metrics[f"{prefix}.top_candidate.globally_eligible"] = (
        1.0 if candidate.get("globally_eligible") is not False else 0.0
    )
    return metrics


def chunking_sweep_summary(
    path: Path,
    payload: dict[str, Any],
    root: Path | None = None,
) -> ValidationSummary:
    selection = payload.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    ranking = [row for row in selection.get("ranking", []) if isinstance(row, dict)]
    recommended = selection.get("recommended")
    recommended = recommended if isinstance(recommended, str) else None
    selected_row = sweep_selected_row(ranking, recommended)
    constraints = selection.get("constraints")
    constraints = constraints if isinstance(constraints, dict) else {}
    failed_checks = [] if recommended else sweep_failed_constraints(ranking)
    return ValidationSummary(
        path=display_artifact_path(path, root),
        kind="chunking_sweep",
        passed=True if recommended else False if constraints else None,
        failed_checks=failed_checks,
        candidate=recommended,
        metrics=chunking_sweep_metrics(payload, selection, selected_row),
    )


def sweep_selected_row(
    ranking: list[dict[str, Any]],
    recommended: str | None,
) -> dict[str, Any]:
    if recommended:
        for row in ranking:
            if row.get("name") == recommended:
                return row
    return ranking[0] if ranking else {}


def sweep_failed_constraints(ranking: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    failed: list[str] = []
    for row in ranking:
        constraints = row.get("failed_constraints")
        if not isinstance(constraints, list):
            continue
        for constraint in constraints:
            if not isinstance(constraint, str) or constraint in seen:
                continue
            seen.add(constraint)
            failed.append(constraint)
    return failed


def chunking_sweep_metrics(
    payload: dict[str, Any],
    selection: dict[str, Any],
    selected_row: dict[str, Any],
) -> dict[str, float]:
    metrics = {
        "candidate_count": numeric_metric(list_count(payload.get("candidates"))),
        "eligible_count": numeric_metric(selection.get("eligible_count")),
        "rejected_count": numeric_metric(selection.get("rejected_count")),
        "pareto_front_count": numeric_metric(list_count(selection.get("pareto_front"))),
        "eligible_pareto_front_count": numeric_metric(
            list_count(selection.get("eligible_pareto_front"))
        ),
        "selection_score": numeric_metric(selected_row.get("score")),
    }
    row_metrics = selected_row.get("metrics")
    if isinstance(row_metrics, dict):
        metrics.update(
            {
                key: numeric_value
                for key, metric_value in row_metrics.items()
                if isinstance(key, str)
                and (numeric_value := optional_numeric_metric(metric_value)) is not None
            }
        )
    return metrics


def list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def visual_run_comparison_summary(
    path: Path,
    payload: dict[str, Any],
    root: Path | None = None,
) -> ValidationSummary:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    best_by_quality = payload.get("best_by_quality")
    best_row = next((row for row in rows if row.get("name") == best_by_quality), {})
    best_by_retrieval = payload.get("best_by_retrieval")
    retrieval_row = next((row for row in rows if row.get("name") == best_by_retrieval), {})
    fastest_name = payload.get("fastest_by_total_latency")
    fastest_row = next((row for row in rows if row.get("name") == fastest_name), {})
    mismatch = bool(payload.get("job_set_mismatch", False))
    return ValidationSummary(
        path=display_artifact_path(path, root),
        kind="visual_run_comparison",
        passed=not mismatch,
        failed_checks=["job_set_mismatch"] if mismatch else [],
        candidate=best_by_quality if isinstance(best_by_quality, str) else None,
        metrics={
            "run_count": float(len(rows)),
            "union_job_count": numeric_metric(payload.get("union_job_count")),
            "shared_job_count": numeric_metric(payload.get("shared_job_count")),
            "job_set_mismatch": 1.0 if mismatch else 0.0,
            "best_quality_score": numeric_metric(best_row.get("quality_score")),
            "best_triples_per_vlm_job": numeric_metric(best_row.get("triples_per_vlm_job")),
            "best_retrieval_score": numeric_metric(retrieval_row.get("retrieval_score")),
            "best_retrieval_target_coverage_at_k": numeric_metric(
                retrieval_row.get("retrieval_target_coverage_at_k")
            ),
            "retrieval_evaluation_run_count": numeric_metric(
                payload.get("retrieval_evaluation_run_count")
            ),
            "fastest_total_mean_latency_ms": numeric_metric(
                fastest_row.get("total_mean_latency_ms")
            ),
        },
    )


def is_validation_payload(payload: dict[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "passed",
            "failed_checks",
            "failed_components",
            "metrics",
            "target_metrics",
            "source_family_metrics",
            "chunk_strategy_metrics",
            "retrieval_role_metrics",
            "case_group_metrics",
            "pairwise_metrics",
            *SUMMARY_METRIC_KEYS,
        )
    )


def validation_kind(path: Path) -> str:
    name = path.name
    for prefix in (
        "ingestion_readiness",
        "chunking_gate",
        "graph_audit",
        "retrieval_gate",
        "retrieval_eval",
        "qdrant_eval",
        "qdrant_vector_ablation_gate",
        "qdrant_reranker_ablation_gate",
        "visual_asset_gate",
        "visual_gate",
        "visual_quality",
    ):
        if name.startswith(prefix):
            return prefix
    return path.stem.split(".", 1)[0]


def display_artifact_path(path: Path, root: Path | None = None) -> str:
    return str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def summary_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        key: float(metric)
        for key, metric in value.items()
        if key in SUMMARY_METRIC_KEYS and isinstance(metric, int | float) and not isinstance(metric, bool)
    }


def validation_summary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = summary_metrics(payload)
    nested_metrics = payload.get("metrics")
    metrics.update(summary_metrics(nested_metrics))
    metrics.update(dynamic_metric_payload(payload))
    return metrics


def dynamic_metric_payload(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(flat_metric_group(payload.get("target_metrics"), "target_type"))
    metrics.update(flat_metric_group(payload.get("source_family_metrics"), "source_family"))
    metrics.update(flat_metric_group(payload.get("chunk_strategy_metrics"), "chunk_strategy"))
    metrics.update(flat_metric_group(payload.get("retrieval_role_metrics"), "retrieval_role"))
    metrics.update(flat_case_group_metrics(payload.get("case_group_metrics")))
    metrics.update(flat_count_metrics(payload.get("reason_counts"), "reason"))
    metrics.update(
        flat_count_metrics(payload.get("missing_target_type_counts"), "missing_target_type")
    )
    metrics.update(
        flat_case_group_count_metrics(
            payload.get("reason_counts_by_case_group"),
            "reason",
        )
    )
    metrics.update(
        flat_case_group_count_metrics(
            payload.get("missing_target_type_counts_by_case_group"),
            "missing_target_type",
        )
    )
    metrics.update(numeric_prefixed_metrics(payload.get("pairwise_metrics"), "pairwise_"))
    return metrics


def flat_metric_group(value: Any, prefix: str) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for group, group_metrics in value.items():
        if not isinstance(group_metrics, dict):
            continue
        for metric_name, metric_value in group_metrics.items():
            numeric_value = optional_numeric_metric(metric_value)
            if numeric_value is not None:
                metrics[f"{prefix}.{group}.{metric_name}"] = numeric_value
    return metrics


def flat_case_group_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for group_name, group_values in value.items():
        if not isinstance(group_values, dict):
            continue
        for group_value, group_metrics in group_values.items():
            if not isinstance(group_metrics, dict):
                continue
            for metric_name, metric_value in group_metrics.items():
                numeric_value = optional_numeric_metric(metric_value)
                if numeric_value is not None:
                    metrics[f"case_group.{group_name}.{group_value}.{metric_name}"] = (
                        numeric_value
                    )
    return metrics


def flat_count_metrics(value: Any, prefix: str) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for name, metric_value in value.items():
        numeric_value = optional_numeric_metric(metric_value)
        if numeric_value is not None:
            metrics[f"{prefix}.{name}"] = numeric_value
    return metrics


def flat_case_group_count_metrics(value: Any, metric_prefix: str) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for group_name, group_values in value.items():
        if not isinstance(group_values, dict):
            continue
        for group_value, counts in group_values.items():
            if not isinstance(counts, dict):
                continue
            for count_name, count_value in counts.items():
                numeric_value = optional_numeric_metric(count_value)
                if numeric_value is not None:
                    metrics[f"case_group.{group_name}.{group_value}.{metric_prefix}.{count_name}"] = (
                        numeric_value
                    )
    return metrics


def numeric_prefixed_metrics(value: Any, prefix: str) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        key: numeric_value
        for key, metric_value in value.items()
        if isinstance(key, str)
        and key.startswith(prefix)
        and (numeric_value := optional_numeric_metric(metric_value)) is not None
    }


def optional_numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def numeric_metric(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}

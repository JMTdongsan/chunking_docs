from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.fusion_sweep import (
    QdrantFusionCaseGroupRecommendation,
    QdrantFusionSweepCandidate,
    QdrantFusionSweepReport,
)
from chunking_docs.evaluation.gate import parse_case_group_spec
from chunking_docs.evaluation.retrieval import RetrievalCaseGroupMetric


class QdrantRetrievalConfigSelection(BaseModel):
    candidate: str
    source: str
    source_report: str | None = None
    global_recommended: str | None = None
    case_group: str | None = None
    case_group_recommended_from_globally_eligible: bool | None = None
    candidate_rank: int = 0
    candidate_eligible: bool = True
    eligibility_failures: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    case_group_metrics: dict[str, float] = Field(default_factory=dict)
    pairwise_comparisons: list[dict[str, Any]] = Field(default_factory=list)


class QdrantRetrievalConfig(BaseModel):
    config_version: int = 1
    backend: str = "qdrant_hybrid"
    collection_name: str | None = None
    package_dir: str | None = None
    bm25_tokens_path: str = "bm25_tokens.json"
    vector_names: list[str] = Field(default_factory=list)
    graph_expand: bool = False
    fusion_weights: dict[str, float] = Field(default_factory=dict)
    top_k: int = 5
    collapse_hierarchical: bool = False
    reranker: str = "none"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 0
    rerank_top_k: int = 0
    query_encoders: dict[str, Any] = Field(default_factory=dict)
    lexical_tokenizer: dict[str, Any] = Field(default_factory=dict)
    selection: QdrantRetrievalConfigSelection
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_qdrant_retrieval_config_from_fusion_sweep(
    report: QdrantFusionSweepReport,
    candidate_name: str | None = None,
    case_group: str | None = None,
    source_report: str | None = None,
) -> QdrantRetrievalConfig:
    selected_name = candidate_name.strip() if candidate_name else None
    selection_source = "candidate" if selected_name else "global_recommended"
    case_group_key = None
    group_recommendation = None
    if case_group:
        group_name, group_value = parse_case_group_spec(case_group)
        case_group_key = f"{group_name}:{group_value}"
        group_recommendation = group_recommendation_for(report, group_name, group_value)
    if selected_name is None and group_recommendation is not None:
        selected_name = group_recommendation.recommended
        selection_source = "case_group_recommended"
    if selected_name is None:
        selected_name = report.recommended
    if not selected_name:
        raise ValueError("Fusion sweep report has no recommended candidate.")

    candidate = candidate_by_name(report, selected_name)
    metadata = report.metadata
    query_encoders = metadata.get("query_encoders")
    lexical_tokenizer = metadata.get("lexical_tokenizer")
    return QdrantRetrievalConfig(
        collection_name=collection_name(report, candidate),
        package_dir=metadata_string(metadata, "package_dir"),
        bm25_tokens_path=metadata_string(metadata, "bm25_tokens_path") or "bm25_tokens.json",
        vector_names=report.vector_names,
        graph_expand=report.graph_expand,
        fusion_weights=dict(candidate.fusion_weights),
        top_k=int(metadata.get("top_k") or 5),
        collapse_hierarchical=bool(metadata.get("collapse_hierarchical") or False),
        reranker=metadata_string(metadata, "reranker") or "none",
        reranker_model=(
            metadata_string(metadata, "reranker_model") or "BAAI/bge-reranker-v2-m3"
        ),
        reranker_max_length=int(metadata.get("reranker_max_length") or 0),
        rerank_top_k=int(metadata.get("rerank_top_k") or 0),
        query_encoders=query_encoders if isinstance(query_encoders, dict) else {},
        lexical_tokenizer=lexical_tokenizer if isinstance(lexical_tokenizer, dict) else {},
        selection=QdrantRetrievalConfigSelection(
            candidate=candidate.name,
            source=selection_source,
            source_report=source_report,
            global_recommended=report.recommended,
            case_group=case_group_key,
            case_group_recommended_from_globally_eligible=(
                group_recommendation.recommended_from_globally_eligible
                if group_recommendation is not None
                else None
            ),
            candidate_rank=candidate.rank,
            candidate_eligible=candidate.eligible,
            eligibility_failures=candidate.eligibility_failures,
            metrics=candidate_metrics(candidate),
            case_group_metrics=candidate_case_group_metrics(candidate, case_group_key),
            pairwise_comparisons=selected_pairwise_comparisons(report, candidate.name),
        ),
        metadata=config_metadata(report),
    )


def group_recommendation_for(
    report: QdrantFusionSweepReport,
    group_name: str,
    group_value: str,
) -> QdrantFusionCaseGroupRecommendation:
    group_recommendation = report.case_group_recommendations.get(group_name, {}).get(group_value)
    if group_recommendation is None:
        raise ValueError(f"Fusion sweep report has no case-group recommendation for {group_name}:{group_value}.")
    if not group_recommendation.recommended:
        raise ValueError(f"Case-group recommendation {group_name}:{group_value} has no candidate.")
    return group_recommendation


def candidate_by_name(
    report: QdrantFusionSweepReport,
    candidate_name: str,
) -> QdrantFusionSweepCandidate:
    for candidate in report.candidates:
        if candidate.name == candidate_name:
            return candidate
    raise ValueError(f"Fusion sweep report has no candidate named {candidate_name}.")


def candidate_metrics(candidate: QdrantFusionSweepCandidate) -> dict[str, float]:
    evaluation = candidate.evaluation
    return {
        "selection_score": candidate.selection_score,
        "recall_at_k": evaluation.recall_at_k,
        "target_coverage_at_k": evaluation.target_coverage_at_k,
        "mean_target_ndcg_at_k": evaluation.mean_target_ndcg_at_k,
        "mrr": evaluation.mrr,
        "mean_precision_at_k": evaluation.mean_precision_at_k,
        "excluded_query_hit_rate": evaluation.excluded_query_hit_rate,
        "excluded_target_hit_rate": evaluation.excluded_target_hit_rate,
        "max_source_excluded_target_hit_rate": candidate.max_source_excluded_target_hit_rate,
        "max_source_family_excluded_target_hit_rate": (
            candidate.max_source_family_excluded_target_hit_rate
        ),
        "max_chunk_strategy_excluded_target_hit_rate": (
            candidate.max_chunk_strategy_excluded_target_hit_rate
        ),
        "max_retrieval_role_excluded_target_hit_rate": (
            candidate.max_retrieval_role_excluded_target_hit_rate
        ),
        "mean_latency_ms": evaluation.mean_latency_ms,
        "p95_latency_ms": evaluation.p95_latency_ms,
        "failed_query_count": float(len(evaluation.failed_queries)),
    }


def collection_name(
    report: QdrantFusionSweepReport,
    candidate: QdrantFusionSweepCandidate,
) -> str | None:
    return (
        metadata_string(report.metadata, "collection")
        or metadata_string(report.metadata, "collection_name")
        or metadata_string(candidate.evaluation.metadata, "collection")
        or metadata_string(candidate.evaluation.metadata, "collection_name")
    )


def metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def candidate_case_group_metrics(
    candidate: QdrantFusionSweepCandidate,
    case_group: str | None,
) -> dict[str, float]:
    if not case_group:
        return {}
    group_name, group_value = parse_case_group_spec(case_group)
    metric = candidate.evaluation.case_group_metrics.get(group_name, {}).get(group_value)
    if metric is None:
        return {}
    return retrieval_case_group_metric_payload(metric)


def selected_pairwise_comparisons(
    report: QdrantFusionSweepReport,
    candidate_name: str,
) -> list[dict[str, Any]]:
    comparisons = [
        comparison
        for comparison in report.pairwise
        if comparison.candidate == candidate_name
    ]
    comparisons.sort(
        key=lambda comparison: (
            -comparison.candidate_win_rate,
            -comparison.mean_target_coverage_delta,
            -comparison.mean_target_ndcg_delta,
            comparison.mean_target_rank_delta
            if comparison.mean_target_rank_delta is not None
            else float("inf"),
            comparison.baseline,
        )
    )
    return [comparison.model_dump() for comparison in comparisons]


def retrieval_case_group_metric_payload(metric: RetrievalCaseGroupMetric) -> dict[str, float]:
    return {
        "case_count": float(metric.case_count),
        "passed_count": float(metric.passed_count),
        "failed_count": float(metric.failed_count),
        "recall_at_k": metric.recall_at_k,
        "target_coverage_at_k": metric.target_coverage_at_k,
        "ndcg_at_k": metric.ndcg_at_k,
        "mrr": metric.mrr,
        "precision_at_k": metric.precision_at_k,
        "mean_latency_ms": metric.mean_latency_ms,
    }


def config_metadata(report: QdrantFusionSweepReport) -> dict[str, Any]:
    metadata = dict(report.metadata)
    metadata.update(
        {
            "sweep_candidate_count": report.candidate_count,
            "sweep_eligible_count": report.eligible_count,
            "sweep_best_by_recall": report.best_by_recall,
            "sweep_best_by_target_coverage": report.best_by_target_coverage,
            "sweep_best_by_target_ndcg": report.best_by_target_ndcg,
            "sweep_best_by_mrr": report.best_by_mrr,
            "sweep_fastest_by_mean_latency": report.fastest_by_mean_latency,
        }
    )
    return metadata

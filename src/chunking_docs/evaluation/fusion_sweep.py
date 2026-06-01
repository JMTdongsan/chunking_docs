from __future__ import annotations

from itertools import product
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import RetrievalEvaluation, RetrievalSourceMetric


class QdrantFusionSweepCandidate(BaseModel):
    name: str
    fusion_weights: dict[str, float] = Field(default_factory=dict)
    evaluation: RetrievalEvaluation
    selection_score: float = 0.0
    eligible: bool = True
    eligibility_failures: list[str] = Field(default_factory=list)
    rank: int = 0
    max_source_excluded_target_hit_rate: float = 0.0
    max_source_excluded_target_hit_rate_name: str | None = None
    max_source_family_excluded_target_hit_rate: float = 0.0
    max_source_family_excluded_target_hit_rate_name: str | None = None
    max_chunk_strategy_excluded_target_hit_rate: float = 0.0
    max_chunk_strategy_excluded_target_hit_rate_name: str | None = None
    max_retrieval_role_excluded_target_hit_rate: float = 0.0
    max_retrieval_role_excluded_target_hit_rate_name: str | None = None


class QdrantFusionCaseGroupCandidate(BaseModel):
    name: str
    fusion_weights: dict[str, float] = Field(default_factory=dict)
    global_rank: int = 0
    globally_eligible: bool = True
    selection_score: float = 0.0
    case_count: int = 0
    recall_at_k: float = 0.0
    target_coverage_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    mrr: float = 0.0
    precision_at_k: float = 0.0
    mean_latency_ms: float = 0.0
    failed_query_count: int = 0


class QdrantFusionCaseGroupRecommendation(BaseModel):
    group_name: str
    group_value: str
    candidate_count: int = 0
    eligible_count: int = 0
    recommended: str | None = None
    recommended_from_globally_eligible: bool = False
    best_by_recall: str | None = None
    best_by_target_coverage: str | None = None
    best_by_target_ndcg: str | None = None
    best_by_mrr: str | None = None
    fastest_by_mean_latency: str | None = None
    top_candidates: list[QdrantFusionCaseGroupCandidate] = Field(default_factory=list)


class QdrantFusionSweepReport(BaseModel):
    vector_names: list[str] = Field(default_factory=list)
    graph_expand: bool = False
    candidate_count: int = 0
    eligible_count: int = 0
    recommended: str | None = None
    best_by_recall: str | None = None
    best_by_target_coverage: str | None = None
    best_by_target_ndcg: str | None = None
    best_by_mrr: str | None = None
    fastest_by_mean_latency: str | None = None
    case_group_recommendations: dict[
        str,
        dict[str, QdrantFusionCaseGroupRecommendation],
    ] = Field(default_factory=dict)
    candidates: list[QdrantFusionSweepCandidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_fusion_weight_grid(
    weight_grid: dict[str, list[float]],
    fixed_weights: dict[str, float] | None = None,
    include_fixed_candidate: bool = True,
    max_candidates: int = 200,
) -> list[dict[str, float]]:
    """Build deterministic fusion-weight candidates from a source-to-values grid."""
    fixed_weights = dict(fixed_weights or {})
    grid_sources = sorted(weight_grid)
    candidates: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()

    def add_candidate(weights: dict[str, float]) -> None:
        key = tuple(sorted((source, float(weight)) for source, weight in weights.items()))
        if key in seen:
            return
        seen.add(key)
        candidates.append(dict(weights))

    if include_fixed_candidate or not grid_sources:
        add_candidate(fixed_weights)

    if grid_sources:
        grid_values = [weight_grid[source] for source in grid_sources]
        for values in product(*grid_values):
            add_candidate(
                {
                    **fixed_weights,
                    **{
                        source: float(value)
                        for source, value in zip(grid_sources, values)
                    },
                }
            )

    if len(candidates) > max_candidates:
        raise ValueError(
            f"Fusion weight grid produced {len(candidates)} candidates; "
            f"limit is {max_candidates}."
        )
    return candidates


def fusion_weight_candidate_name(weights: dict[str, float]) -> str:
    if not weights:
        return "default"
    return "__".join(
        f"{safe_weight_source(source)}_{safe_weight_value(weight)}"
        for source, weight in sorted(weights.items())
    )


def safe_weight_source(source: str) -> str:
    return (
        source.strip()
        .replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def safe_weight_value(weight: float) -> str:
    return f"{weight:g}".replace("-", "m").replace(".", "p")


def build_qdrant_fusion_sweep_report(
    candidates: list[QdrantFusionSweepCandidate],
    vector_names: list[str],
    graph_expand: bool = False,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_latency_ms: float | None = None,
    max_excluded_target_hit_rate: float | None = None,
    max_excluded_query_hit_rate: float | None = None,
    max_excluded_hit_query_count: int | None = None,
    max_source_excluded_target_hit_rate: dict[str, float] | None = None,
    max_source_family_excluded_target_hit_rate: dict[str, float] | None = None,
    max_chunk_strategy_excluded_target_hit_rate: dict[str, float] | None = None,
    max_retrieval_role_excluded_target_hit_rate: dict[str, float] | None = None,
    recall_weight: float = 1.0,
    target_coverage_weight: float = 2.0,
    target_ndcg_weight: float = 1.0,
    mrr_weight: float = 1.0,
    precision_weight: float = 0.5,
    failed_query_penalty: float = 0.02,
    excluded_query_hit_penalty: float = 1.0,
    excluded_target_hit_penalty: float = 1.0,
    source_excluded_target_hit_penalty: float = 0.0,
    source_family_excluded_target_hit_penalty: float = 0.0,
    chunk_strategy_excluded_target_hit_penalty: float = 0.0,
    retrieval_role_excluded_target_hit_penalty: float = 0.0,
    latency_weight: float = 0.05,
    case_group_top_k: int = 3,
    metadata: dict[str, Any] | None = None,
) -> QdrantFusionSweepReport:
    ranked = [
        score_fusion_candidate(
            candidate,
            min_recall_at_k=min_recall_at_k,
            min_target_coverage_at_k=min_target_coverage_at_k,
            min_target_ndcg_at_k=min_target_ndcg_at_k,
            min_mrr=min_mrr,
            max_failed_queries=max_failed_queries,
            max_mean_latency_ms=max_mean_latency_ms,
            max_excluded_target_hit_rate=max_excluded_target_hit_rate,
            max_excluded_query_hit_rate=max_excluded_query_hit_rate,
            max_excluded_hit_query_count=max_excluded_hit_query_count,
            max_source_excluded_target_hit_rate=max_source_excluded_target_hit_rate,
            max_source_family_excluded_target_hit_rate=max_source_family_excluded_target_hit_rate,
            max_chunk_strategy_excluded_target_hit_rate=max_chunk_strategy_excluded_target_hit_rate,
            max_retrieval_role_excluded_target_hit_rate=max_retrieval_role_excluded_target_hit_rate,
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
        )
        for candidate in candidates
    ]
    ranked = sorted(ranked, key=fusion_candidate_rank_key, reverse=True)
    ranked = [
        candidate.model_copy(update={"rank": index})
        for index, candidate in enumerate(ranked, start=1)
    ]
    eligible = [candidate for candidate in ranked if candidate.eligible]
    return QdrantFusionSweepReport(
        vector_names=vector_names,
        graph_expand=graph_expand,
        candidate_count=len(ranked),
        eligible_count=len(eligible),
        recommended=eligible[0].name if eligible else None,
        best_by_recall=best_candidate_name(ranked, "recall_at_k"),
        best_by_target_coverage=best_candidate_name(ranked, "target_coverage_at_k"),
        best_by_target_ndcg=best_candidate_name(ranked, "mean_target_ndcg_at_k"),
        best_by_mrr=best_candidate_name(ranked, "mrr"),
        fastest_by_mean_latency=best_candidate_name(
            ranked,
            "mean_latency_ms",
            prefer_lower=True,
        ),
        case_group_recommendations=case_group_recommendations(
            ranked,
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
            top_k=case_group_top_k,
        ),
        candidates=ranked,
        metadata=metadata or {},
    )


def score_fusion_candidate(
    candidate: QdrantFusionSweepCandidate,
    min_recall_at_k: float,
    min_target_coverage_at_k: float,
    min_target_ndcg_at_k: float,
    min_mrr: float,
    max_failed_queries: int | None,
    max_mean_latency_ms: float | None,
    max_excluded_target_hit_rate: float | None,
    max_excluded_query_hit_rate: float | None,
    max_excluded_hit_query_count: int | None,
    max_source_excluded_target_hit_rate: dict[str, float] | None,
    max_source_family_excluded_target_hit_rate: dict[str, float] | None,
    max_chunk_strategy_excluded_target_hit_rate: dict[str, float] | None,
    max_retrieval_role_excluded_target_hit_rate: dict[str, float] | None,
    recall_weight: float,
    target_coverage_weight: float,
    target_ndcg_weight: float,
    mrr_weight: float,
    precision_weight: float,
    failed_query_penalty: float,
    excluded_query_hit_penalty: float,
    excluded_target_hit_penalty: float,
    source_excluded_target_hit_penalty: float,
    source_family_excluded_target_hit_penalty: float,
    chunk_strategy_excluded_target_hit_penalty: float,
    retrieval_role_excluded_target_hit_penalty: float,
    latency_weight: float,
) -> QdrantFusionSweepCandidate:
    evaluation = candidate.evaluation
    source_max_name, source_max_rate = excluded_target_hit_rate_max(evaluation.source_metrics)
    source_family_max_name, source_family_max_rate = excluded_target_hit_rate_max(
        evaluation.source_family_metrics
    )
    chunk_strategy_max_name, chunk_strategy_max_rate = excluded_target_hit_rate_max(
        evaluation.chunk_strategy_metrics
    )
    retrieval_role_max_name, retrieval_role_max_rate = excluded_target_hit_rate_max(
        evaluation.retrieval_role_metrics
    )
    failures = fusion_candidate_failures(
        evaluation,
        min_recall_at_k=min_recall_at_k,
        min_target_coverage_at_k=min_target_coverage_at_k,
        min_target_ndcg_at_k=min_target_ndcg_at_k,
        min_mrr=min_mrr,
        max_failed_queries=max_failed_queries,
        max_mean_latency_ms=max_mean_latency_ms,
        max_excluded_target_hit_rate=max_excluded_target_hit_rate,
        max_excluded_query_hit_rate=max_excluded_query_hit_rate,
        max_excluded_hit_query_count=max_excluded_hit_query_count,
        max_source_excluded_target_hit_rate=max_source_excluded_target_hit_rate,
        max_source_family_excluded_target_hit_rate=max_source_family_excluded_target_hit_rate,
        max_chunk_strategy_excluded_target_hit_rate=max_chunk_strategy_excluded_target_hit_rate,
        max_retrieval_role_excluded_target_hit_rate=max_retrieval_role_excluded_target_hit_rate,
    )
    score = (
        recall_weight * evaluation.recall_at_k
        + target_coverage_weight * evaluation.target_coverage_at_k
        + target_ndcg_weight * evaluation.mean_target_ndcg_at_k
        + mrr_weight * evaluation.mrr
        + precision_weight * evaluation.mean_precision_at_k
        - failed_query_penalty * len(evaluation.failed_queries)
        - excluded_query_hit_penalty * evaluation.excluded_query_hit_rate
        - excluded_target_hit_penalty * evaluation.excluded_target_hit_rate
        - source_excluded_target_hit_penalty * source_max_rate
        - source_family_excluded_target_hit_penalty * source_family_max_rate
        - chunk_strategy_excluded_target_hit_penalty * chunk_strategy_max_rate
        - retrieval_role_excluded_target_hit_penalty * retrieval_role_max_rate
        - latency_weight * (evaluation.mean_latency_ms / 1000.0)
    )
    return candidate.model_copy(
        update={
            "selection_score": score,
            "eligible": not failures,
            "eligibility_failures": failures,
            "max_source_excluded_target_hit_rate": source_max_rate,
            "max_source_excluded_target_hit_rate_name": source_max_name,
            "max_source_family_excluded_target_hit_rate": source_family_max_rate,
            "max_source_family_excluded_target_hit_rate_name": source_family_max_name,
            "max_chunk_strategy_excluded_target_hit_rate": chunk_strategy_max_rate,
            "max_chunk_strategy_excluded_target_hit_rate_name": chunk_strategy_max_name,
            "max_retrieval_role_excluded_target_hit_rate": retrieval_role_max_rate,
            "max_retrieval_role_excluded_target_hit_rate_name": retrieval_role_max_name,
        }
    )


def fusion_candidate_failures(
    evaluation: RetrievalEvaluation,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    max_failed_queries: int | None = None,
    max_mean_latency_ms: float | None = None,
    max_excluded_target_hit_rate: float | None = None,
    max_excluded_query_hit_rate: float | None = None,
    max_excluded_hit_query_count: int | None = None,
    max_source_excluded_target_hit_rate: dict[str, float] | None = None,
    max_source_family_excluded_target_hit_rate: dict[str, float] | None = None,
    max_chunk_strategy_excluded_target_hit_rate: dict[str, float] | None = None,
    max_retrieval_role_excluded_target_hit_rate: dict[str, float] | None = None,
) -> list[str]:
    failures = []
    if evaluation.recall_at_k < min_recall_at_k:
        failures.append("min_recall_at_k")
    if evaluation.target_coverage_at_k < min_target_coverage_at_k:
        failures.append("min_target_coverage_at_k")
    if evaluation.mean_target_ndcg_at_k < min_target_ndcg_at_k:
        failures.append("min_target_ndcg_at_k")
    if evaluation.mrr < min_mrr:
        failures.append("min_mrr")
    if max_failed_queries is not None and len(evaluation.failed_queries) > max_failed_queries:
        failures.append("max_failed_queries")
    if max_mean_latency_ms is not None and evaluation.mean_latency_ms > max_mean_latency_ms:
        failures.append("max_mean_latency_ms")
    if (
        max_excluded_target_hit_rate is not None
        and evaluation.excluded_target_hit_rate > max_excluded_target_hit_rate
    ):
        failures.append("max_excluded_target_hit_rate")
    if (
        max_excluded_query_hit_rate is not None
        and evaluation.excluded_query_hit_rate > max_excluded_query_hit_rate
    ):
        failures.append("max_excluded_query_hit_rate")
    if (
        max_excluded_hit_query_count is not None
        and evaluation.excluded_hit_query_count > max_excluded_hit_query_count
    ):
        failures.append("max_excluded_hit_query_count")
    failures.extend(
        named_excluded_target_hit_rate_failures(
            evaluation.source_metrics,
            max_source_excluded_target_hit_rate,
            "max_source_excluded_target_hit_rate",
        )
    )
    failures.extend(
        named_excluded_target_hit_rate_failures(
            evaluation.source_family_metrics,
            max_source_family_excluded_target_hit_rate,
            "max_source_family_excluded_target_hit_rate",
        )
    )
    failures.extend(
        named_excluded_target_hit_rate_failures(
            evaluation.chunk_strategy_metrics,
            max_chunk_strategy_excluded_target_hit_rate,
            "max_chunk_strategy_excluded_target_hit_rate",
        )
    )
    failures.extend(
        named_excluded_target_hit_rate_failures(
            evaluation.retrieval_role_metrics,
            max_retrieval_role_excluded_target_hit_rate,
            "max_retrieval_role_excluded_target_hit_rate",
        )
    )
    return failures


def named_excluded_target_hit_rate_failures(
    metrics: dict[str, RetrievalSourceMetric],
    thresholds: dict[str, float] | None,
    failure_prefix: str,
) -> list[str]:
    if not thresholds:
        return []
    normalized_metrics = {
        normalize_metric_name(name): metric
        for name, metric in metrics.items()
    }
    failures = []
    for raw_name, threshold in sorted(thresholds.items()):
        name = normalize_metric_name(raw_name)
        metric = normalized_metrics.get(name)
        hit_rate = metric.excluded_target_hit_rate if metric is not None else 0.0
        if hit_rate > threshold:
            failures.append(f"{failure_prefix}:{name}")
    return failures


def excluded_target_hit_rate_max(
    metrics: dict[str, RetrievalSourceMetric],
) -> tuple[str | None, float]:
    if not metrics:
        return None, 0.0
    name, metric = max(
        metrics.items(),
        key=lambda item: (item[1].excluded_target_hit_rate, normalize_metric_name(item[0])),
    )
    return name, metric.excluded_target_hit_rate


def normalize_metric_name(value: str) -> str:
    return str(value).strip().lower()


def fusion_candidate_rank_key(candidate: QdrantFusionSweepCandidate) -> tuple:
    evaluation = candidate.evaluation
    return (
        candidate.eligible,
        candidate.selection_score,
        evaluation.target_coverage_at_k,
        evaluation.mean_target_ndcg_at_k,
        evaluation.mrr,
        evaluation.recall_at_k,
        -evaluation.excluded_query_hit_rate,
        -evaluation.excluded_target_hit_rate,
        -candidate.max_source_excluded_target_hit_rate,
        -candidate.max_source_family_excluded_target_hit_rate,
        -candidate.max_chunk_strategy_excluded_target_hit_rate,
        -candidate.max_retrieval_role_excluded_target_hit_rate,
        -evaluation.excluded_hit_query_count,
        -evaluation.mean_latency_ms,
        candidate.name,
    )


def best_candidate_name(
    candidates: list[QdrantFusionSweepCandidate],
    metric_name: str,
    prefer_lower: bool = False,
) -> str | None:
    if not candidates:
        return None
    if prefer_lower:
        return min(candidates, key=lambda candidate: getattr(candidate.evaluation, metric_name)).name
    return max(candidates, key=lambda candidate: getattr(candidate.evaluation, metric_name)).name


def case_group_recommendations(
    candidates: list[QdrantFusionSweepCandidate],
    recall_weight: float,
    target_coverage_weight: float,
    target_ndcg_weight: float,
    mrr_weight: float,
    precision_weight: float,
    failed_query_penalty: float,
    excluded_query_hit_penalty: float,
    excluded_target_hit_penalty: float,
    source_excluded_target_hit_penalty: float,
    source_family_excluded_target_hit_penalty: float,
    chunk_strategy_excluded_target_hit_penalty: float,
    retrieval_role_excluded_target_hit_penalty: float,
    latency_weight: float,
    top_k: int = 3,
) -> dict[str, dict[str, QdrantFusionCaseGroupRecommendation]]:
    grouped: dict[str, dict[str, list[QdrantFusionCaseGroupCandidate]]] = {}
    for candidate in candidates:
        for group_name, group_values in candidate.evaluation.case_group_metrics.items():
            for group_value, metric in group_values.items():
                group_candidate = QdrantFusionCaseGroupCandidate(
                    name=candidate.name,
                    fusion_weights=candidate.fusion_weights,
                    global_rank=candidate.rank,
                    globally_eligible=candidate.eligible,
                    selection_score=case_group_selection_score(
                        recall_at_k=metric.recall_at_k,
                        target_coverage_at_k=metric.target_coverage_at_k,
                        ndcg_at_k=metric.ndcg_at_k,
                        mrr=metric.mrr,
                        precision_at_k=metric.precision_at_k,
                        failed_query_count=metric.failed_count,
                        mean_latency_ms=metric.mean_latency_ms,
                        recall_weight=recall_weight,
                        target_coverage_weight=target_coverage_weight,
                        target_ndcg_weight=target_ndcg_weight,
                        mrr_weight=mrr_weight,
                        precision_weight=precision_weight,
                        failed_query_penalty=failed_query_penalty,
                        latency_weight=latency_weight,
                    )
                    - fusion_candidate_excluded_hit_penalty(
                        candidate,
                        excluded_query_hit_penalty=excluded_query_hit_penalty,
                        excluded_target_hit_penalty=excluded_target_hit_penalty,
                        source_excluded_target_hit_penalty=source_excluded_target_hit_penalty,
                        source_family_excluded_target_hit_penalty=source_family_excluded_target_hit_penalty,
                        chunk_strategy_excluded_target_hit_penalty=chunk_strategy_excluded_target_hit_penalty,
                        retrieval_role_excluded_target_hit_penalty=retrieval_role_excluded_target_hit_penalty,
                    ),
                    case_count=metric.case_count,
                    recall_at_k=metric.recall_at_k,
                    target_coverage_at_k=metric.target_coverage_at_k,
                    ndcg_at_k=metric.ndcg_at_k,
                    mrr=metric.mrr,
                    precision_at_k=metric.precision_at_k,
                    mean_latency_ms=metric.mean_latency_ms,
                    failed_query_count=metric.failed_count,
                )
                grouped.setdefault(group_name, {}).setdefault(group_value, []).append(group_candidate)

    result: dict[str, dict[str, QdrantFusionCaseGroupRecommendation]] = {}
    for group_name, group_values in sorted(grouped.items()):
        result[group_name] = {}
        for group_value, group_candidates in sorted(group_values.items()):
            ranked = sorted(group_candidates, key=case_group_candidate_rank_key, reverse=True)
            eligible = [candidate for candidate in ranked if candidate.globally_eligible]
            recommendation_pool = eligible or ranked
            recommended = recommendation_pool[0].name if recommendation_pool else None
            result[group_name][group_value] = QdrantFusionCaseGroupRecommendation(
                group_name=group_name,
                group_value=group_value,
                candidate_count=len(ranked),
                eligible_count=len(eligible),
                recommended=recommended,
                recommended_from_globally_eligible=bool(eligible and recommended),
                best_by_recall=best_case_group_candidate_name(ranked, "recall_at_k"),
                best_by_target_coverage=best_case_group_candidate_name(ranked, "target_coverage_at_k"),
                best_by_target_ndcg=best_case_group_candidate_name(ranked, "ndcg_at_k"),
                best_by_mrr=best_case_group_candidate_name(ranked, "mrr"),
                fastest_by_mean_latency=best_case_group_candidate_name(
                    ranked,
                    "mean_latency_ms",
                    prefer_lower=True,
                ),
                top_candidates=ranked[: max(top_k, 0)],
            )
    return result


def case_group_selection_score(
    recall_at_k: float,
    target_coverage_at_k: float,
    ndcg_at_k: float,
    mrr: float,
    precision_at_k: float,
    failed_query_count: int,
    mean_latency_ms: float,
    recall_weight: float,
    target_coverage_weight: float,
    target_ndcg_weight: float,
    mrr_weight: float,
    precision_weight: float,
    failed_query_penalty: float,
    latency_weight: float,
) -> float:
    return (
        recall_weight * recall_at_k
        + target_coverage_weight * target_coverage_at_k
        + target_ndcg_weight * ndcg_at_k
        + mrr_weight * mrr
        + precision_weight * precision_at_k
        - failed_query_penalty * failed_query_count
        - latency_weight * (mean_latency_ms / 1000.0)
    )


def fusion_candidate_excluded_hit_penalty(
    candidate: QdrantFusionSweepCandidate,
    excluded_query_hit_penalty: float,
    excluded_target_hit_penalty: float,
    source_excluded_target_hit_penalty: float,
    source_family_excluded_target_hit_penalty: float,
    chunk_strategy_excluded_target_hit_penalty: float,
    retrieval_role_excluded_target_hit_penalty: float,
) -> float:
    evaluation = candidate.evaluation
    return (
        excluded_query_hit_penalty * evaluation.excluded_query_hit_rate
        + excluded_target_hit_penalty * evaluation.excluded_target_hit_rate
        + source_excluded_target_hit_penalty * candidate.max_source_excluded_target_hit_rate
        + source_family_excluded_target_hit_penalty
        * candidate.max_source_family_excluded_target_hit_rate
        + chunk_strategy_excluded_target_hit_penalty
        * candidate.max_chunk_strategy_excluded_target_hit_rate
        + retrieval_role_excluded_target_hit_penalty
        * candidate.max_retrieval_role_excluded_target_hit_rate
    )


def case_group_candidate_rank_key(candidate: QdrantFusionCaseGroupCandidate) -> tuple:
    return (
        candidate.globally_eligible,
        candidate.selection_score,
        candidate.target_coverage_at_k,
        candidate.ndcg_at_k,
        candidate.mrr,
        candidate.recall_at_k,
        -candidate.mean_latency_ms,
        candidate.name,
    )


def best_case_group_candidate_name(
    candidates: list[QdrantFusionCaseGroupCandidate],
    metric_name: str,
    prefer_lower: bool = False,
) -> str | None:
    if not candidates:
        return None
    if prefer_lower:
        return min(candidates, key=lambda candidate: getattr(candidate, metric_name)).name
    return max(candidates, key=lambda candidate: getattr(candidate, metric_name)).name

from __future__ import annotations

from itertools import product
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import RetrievalEvaluation


class QdrantFusionSweepCandidate(BaseModel):
    name: str
    fusion_weights: dict[str, float] = Field(default_factory=dict)
    evaluation: RetrievalEvaluation
    selection_score: float = 0.0
    eligible: bool = True
    eligibility_failures: list[str] = Field(default_factory=list)
    rank: int = 0


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
    recall_weight: float = 1.0,
    target_coverage_weight: float = 2.0,
    target_ndcg_weight: float = 1.0,
    mrr_weight: float = 1.0,
    precision_weight: float = 0.5,
    failed_query_penalty: float = 0.02,
    latency_weight: float = 0.05,
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
            recall_weight=recall_weight,
            target_coverage_weight=target_coverage_weight,
            target_ndcg_weight=target_ndcg_weight,
            mrr_weight=mrr_weight,
            precision_weight=precision_weight,
            failed_query_penalty=failed_query_penalty,
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
    recall_weight: float,
    target_coverage_weight: float,
    target_ndcg_weight: float,
    mrr_weight: float,
    precision_weight: float,
    failed_query_penalty: float,
    latency_weight: float,
) -> QdrantFusionSweepCandidate:
    evaluation = candidate.evaluation
    failures = fusion_candidate_failures(
        evaluation,
        min_recall_at_k=min_recall_at_k,
        min_target_coverage_at_k=min_target_coverage_at_k,
        min_target_ndcg_at_k=min_target_ndcg_at_k,
        min_mrr=min_mrr,
        max_failed_queries=max_failed_queries,
        max_mean_latency_ms=max_mean_latency_ms,
    )
    score = (
        recall_weight * evaluation.recall_at_k
        + target_coverage_weight * evaluation.target_coverage_at_k
        + target_ndcg_weight * evaluation.mean_target_ndcg_at_k
        + mrr_weight * evaluation.mrr
        + precision_weight * evaluation.mean_precision_at_k
        - failed_query_penalty * len(evaluation.failed_queries)
        - latency_weight * (evaluation.mean_latency_ms / 1000.0)
    )
    return candidate.model_copy(
        update={
            "selection_score": score,
            "eligible": not failures,
            "eligibility_failures": failures,
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
    return failures


def fusion_candidate_rank_key(candidate: QdrantFusionSweepCandidate) -> tuple:
    evaluation = candidate.evaluation
    return (
        candidate.eligible,
        candidate.selection_score,
        evaluation.target_coverage_at_k,
        evaluation.mean_target_ndcg_at_k,
        evaluation.mrr,
        evaluation.recall_at_k,
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

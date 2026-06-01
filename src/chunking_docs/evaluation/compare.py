from __future__ import annotations

import random

from pydantic import BaseModel, Field

from chunking_docs.evaluation.chunking_quality import ChunkingQualityReport
from chunking_docs.evaluation.gate import (
    result_expected_target_keys,
    retrieval_case_group_metrics,
    retrieval_chunk_strategy_metrics,
    retrieval_rank_metrics,
    retrieval_role_metrics_payload,
    retrieval_source_family_metrics,
    retrieval_target_metrics,
)
from chunking_docs.evaluation.retrieval import RetrievalCaseResult


class ChunkingComparisonRow(BaseModel):
    name: str
    chunk_count: int
    quality_score: float
    retrieval_hit_rate: float | None
    retrieval_recall_at_k: float | None
    retrieval_mrr: float | None
    retrieval_target_coverage_at_k: float | None
    retrieval_mean_target_ndcg_at_k: float | None
    retrieval_mean_precision_at_k: float | None
    retrieval_mean_latency_ms: float | None
    retrieval_p95_latency_ms: float | None
    retrieval_mean_first_relevant_rank: float | None = None
    retrieval_p95_first_relevant_rank: float | None = None
    retrieval_mean_target_rank: float | None = None
    retrieval_p95_target_rank: float | None = None
    retrieval_ranked_expected_case_count: float | None = None
    retrieval_ranked_target_count: float | None = None
    retrieval_unstable_result_count: float | None = None
    retrieval_result_stability_rate: float | None = None
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    chunk_strategy_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    retrieval_role_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    case_group_metrics: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    failed_queries: list[str]
    page_coverage_ratio: float
    visual_annotation_ratio: float
    visual_text_asset_count: int = 0
    visual_text_covered_asset_count: int = 0
    visual_text_coverage_ratio: float = 1.0
    visual_text_part_count: int = 0
    visual_text_covered_part_count: int = 0
    visual_text_part_coverage_ratio: float = 1.0
    standalone_visual_chunk_count: int = 0
    standalone_visual_text_asset_count: int = 0
    visual_object_chunk_count: int = 0
    chunks_under_min_chars: int
    chunks_over_max_chars: int
    issue_codes: list[str]


class ChunkingPairwiseComparison(BaseModel):
    candidate: str
    baseline: str
    shared_query_count: int
    candidate_win_count: int = 0
    baseline_win_count: int = 0
    tie_count: int = 0
    candidate_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    mean_reciprocal_rank_delta: float = 0.0
    mean_target_coverage_delta: float = 0.0
    mean_target_ndcg_delta: float = 0.0
    mean_precision_delta: float = 0.0
    mean_first_relevant_rank_delta: float | None = None
    mean_target_rank_delta: float | None = None
    mean_latency_delta_ms: float | None = None
    bootstrap_samples: int = 0
    confidence_level: float = 0.95
    reciprocal_rank_delta_ci_low: float | None = None
    reciprocal_rank_delta_ci_high: float | None = None
    target_coverage_delta_ci_low: float | None = None
    target_coverage_delta_ci_high: float | None = None
    target_ndcg_delta_ci_low: float | None = None
    target_ndcg_delta_ci_high: float | None = None
    precision_delta_ci_low: float | None = None
    precision_delta_ci_high: float | None = None
    first_relevant_rank_delta_ci_low: float | None = None
    first_relevant_rank_delta_ci_high: float | None = None
    target_rank_delta_ci_low: float | None = None
    target_rank_delta_ci_high: float | None = None
    latency_delta_ci_low_ms: float | None = None
    latency_delta_ci_high_ms: float | None = None


PAIRWISE_BOOTSTRAP_SAMPLES = 1000
PAIRWISE_CONFIDENCE_LEVEL = 0.95


class ChunkingComparison(BaseModel):
    rows: list[ChunkingComparisonRow]
    best_by_quality: str | None
    best_by_retrieval: str | None
    fastest_by_mean_latency: str | None
    pairwise: list[ChunkingPairwiseComparison] = Field(default_factory=list)


def compare_chunking_reports(
    reports: dict[str, ChunkingQualityReport],
) -> ChunkingComparison:
    rows = []
    for name, report in reports.items():
        target_metrics = retrieval_target_metrics(report.retrieval)
        source_family_metrics = retrieval_source_family_metrics(report.retrieval)
        chunk_strategy_metrics = retrieval_chunk_strategy_metrics(report.retrieval)
        retrieval_role_metrics = retrieval_role_metrics_payload(report.retrieval)
        case_group_metrics = retrieval_case_group_metrics(report.retrieval)
        rank_metrics = retrieval_rank_metrics(report.retrieval) if report.retrieval else {}
        rows.append(
            ChunkingComparisonRow(
                name=name,
                chunk_count=report.chunk_count,
                quality_score=report.quality_score,
                retrieval_hit_rate=report.retrieval.hit_rate if report.retrieval else None,
                retrieval_recall_at_k=report.retrieval.recall_at_k if report.retrieval else None,
                retrieval_mrr=report.retrieval.mrr if report.retrieval else None,
                retrieval_target_coverage_at_k=report.retrieval.target_coverage_at_k
                if report.retrieval
                else None,
                retrieval_mean_target_ndcg_at_k=report.retrieval.mean_target_ndcg_at_k
                if report.retrieval
                else None,
                retrieval_mean_precision_at_k=report.retrieval.mean_precision_at_k
                if report.retrieval
                else None,
                retrieval_mean_latency_ms=report.retrieval.mean_latency_ms if report.retrieval else None,
                retrieval_p95_latency_ms=report.retrieval.p95_latency_ms if report.retrieval else None,
                retrieval_mean_first_relevant_rank=rank_metrics.get("mean_first_relevant_rank"),
                retrieval_p95_first_relevant_rank=rank_metrics.get("p95_first_relevant_rank"),
                retrieval_mean_target_rank=rank_metrics.get("mean_target_rank"),
                retrieval_p95_target_rank=rank_metrics.get("p95_target_rank"),
                retrieval_ranked_expected_case_count=rank_metrics.get(
                    "ranked_expected_case_count"
                ),
                retrieval_ranked_target_count=rank_metrics.get("ranked_target_count"),
                retrieval_unstable_result_count=float(report.retrieval.unstable_result_count)
                if report.retrieval
                else None,
                retrieval_result_stability_rate=report.retrieval.result_stability_rate
                if report.retrieval
                else None,
                target_metrics=target_metrics,
                source_family_metrics=source_family_metrics,
                chunk_strategy_metrics=chunk_strategy_metrics,
                retrieval_role_metrics=retrieval_role_metrics,
                case_group_metrics=case_group_metrics,
                failed_queries=report.retrieval.failed_queries if report.retrieval else [],
                page_coverage_ratio=report.page_coverage_ratio,
                visual_annotation_ratio=report.visual_annotation_ratio,
                visual_text_asset_count=report.visual_text_asset_count,
                visual_text_covered_asset_count=report.visual_text_covered_asset_count,
                visual_text_coverage_ratio=report.visual_text_coverage_ratio,
                visual_text_part_count=report.visual_text_part_count,
                visual_text_covered_part_count=report.visual_text_covered_part_count,
                visual_text_part_coverage_ratio=report.visual_text_part_coverage_ratio,
                standalone_visual_chunk_count=report.standalone_visual_chunk_count,
                standalone_visual_text_asset_count=report.standalone_visual_text_asset_count,
                visual_object_chunk_count=report.visual_object_chunk_count,
                chunks_under_min_chars=report.chunks_under_min_chars,
                chunks_over_max_chars=report.chunks_over_max_chars,
                issue_codes=[issue.code for issue in report.issues],
            )
        )
    rows.sort(
        key=lambda row: (
            row.retrieval_recall_at_k if row.retrieval_recall_at_k is not None else -1.0,
            row.retrieval_target_coverage_at_k
            if row.retrieval_target_coverage_at_k is not None
            else -1.0,
            row.retrieval_mean_target_ndcg_at_k
            if row.retrieval_mean_target_ndcg_at_k is not None
            else -1.0,
            rank_sort_value(row.retrieval_mean_target_rank),
            row.retrieval_mrr if row.retrieval_mrr is not None else -1.0,
            row.quality_score,
        ),
        reverse=True,
    )
    best_by_quality = max(rows, key=lambda row: row.quality_score).name if rows else None
    retrieval_rows = [row for row in rows if row.retrieval_recall_at_k is not None]
    best_by_retrieval = (
        max(
            retrieval_rows,
            key=lambda row: (
                row.retrieval_recall_at_k or 0.0,
                row.retrieval_target_coverage_at_k or 0.0,
                row.retrieval_mean_target_ndcg_at_k or 0.0,
                rank_sort_value(row.retrieval_mean_target_rank),
                row.retrieval_mrr or 0.0,
            ),
        ).name
        if retrieval_rows
        else None
    )
    latency_rows = [row for row in rows if row.retrieval_mean_latency_ms is not None]
    return ChunkingComparison(
        rows=rows,
        best_by_quality=best_by_quality,
        best_by_retrieval=best_by_retrieval,
        fastest_by_mean_latency=min(latency_rows, key=lambda row: row.retrieval_mean_latency_ms or 0.0).name
        if latency_rows
        else None,
        pairwise=pairwise_comparisons(reports),
    )


def pairwise_comparisons(
    reports: dict[str, ChunkingQualityReport],
) -> list[ChunkingPairwiseComparison]:
    comparisons = []
    for candidate_name, candidate_report in reports.items():
        for baseline_name, baseline_report in reports.items():
            if candidate_name == baseline_name:
                continue
            comparison = compare_retrieval_results_pairwise(
                candidate_name,
                candidate_report,
                baseline_name,
                baseline_report,
            )
            if comparison is not None:
                comparisons.append(comparison)
    return comparisons


def compare_retrieval_results_pairwise(
    candidate_name: str,
    candidate_report: ChunkingQualityReport,
    baseline_name: str,
    baseline_report: ChunkingQualityReport,
) -> ChunkingPairwiseComparison | None:
    if candidate_report.retrieval is None or baseline_report.retrieval is None:
        return None
    candidate_results = results_by_query(candidate_report.retrieval.results)
    baseline_results = results_by_query(baseline_report.retrieval.results)
    shared_queries = sorted(candidate_results.keys() & baseline_results.keys())
    if not shared_queries:
        return None

    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    reciprocal_rank_deltas = []
    target_coverage_deltas = []
    target_ndcg_deltas = []
    precision_deltas = []
    first_relevant_rank_deltas = []
    target_rank_deltas = []
    latency_deltas = []
    missing_rank = float(max(candidate_report.retrieval.top_k, baseline_report.retrieval.top_k) + 1)
    for query in shared_queries:
        candidate_result = candidate_results[query]
        baseline_result = baseline_results[query]
        winner = compare_case_results(candidate_result, baseline_result)
        if winner > 0:
            candidate_wins += 1
        elif winner < 0:
            baseline_wins += 1
        else:
            ties += 1
        reciprocal_rank_deltas.append(
            candidate_result.reciprocal_rank - baseline_result.reciprocal_rank
        )
        target_coverage_deltas.append(
            candidate_result.target_coverage_at_k - baseline_result.target_coverage_at_k
        )
        target_ndcg_deltas.append(
            candidate_result.target_ndcg_at_k - baseline_result.target_ndcg_at_k
        )
        precision_deltas.append(candidate_result.precision_at_k - baseline_result.precision_at_k)
        first_relevant_rank_deltas.append(
            first_relevant_rank(candidate_result, missing_rank)
            - first_relevant_rank(baseline_result, missing_rank)
        )
        target_rank_deltas.append(
            case_mean_target_rank(candidate_result, missing_rank)
            - case_mean_target_rank(baseline_result, missing_rank)
        )
        latency_deltas.append(candidate_result.latency_ms - baseline_result.latency_ms)

    shared_count = len(shared_queries)
    reciprocal_rank_ci = bootstrap_mean_interval(reciprocal_rank_deltas, seed=stable_seed(candidate_name, baseline_name, "mrr"))
    target_coverage_ci = bootstrap_mean_interval(target_coverage_deltas, seed=stable_seed(candidate_name, baseline_name, "coverage"))
    target_ndcg_ci = bootstrap_mean_interval(target_ndcg_deltas, seed=stable_seed(candidate_name, baseline_name, "ndcg"))
    precision_ci = bootstrap_mean_interval(precision_deltas, seed=stable_seed(candidate_name, baseline_name, "precision"))
    first_rank_ci = bootstrap_mean_interval(
        first_relevant_rank_deltas,
        seed=stable_seed(candidate_name, baseline_name, "first-rank"),
    )
    target_rank_ci = bootstrap_mean_interval(
        target_rank_deltas,
        seed=stable_seed(candidate_name, baseline_name, "target-rank"),
    )
    latency_ci = bootstrap_mean_interval(latency_deltas, seed=stable_seed(candidate_name, baseline_name, "latency"))
    return ChunkingPairwiseComparison(
        candidate=candidate_name,
        baseline=baseline_name,
        shared_query_count=shared_count,
        candidate_win_count=candidate_wins,
        baseline_win_count=baseline_wins,
        tie_count=ties,
        candidate_win_rate=candidate_wins / shared_count,
        baseline_win_rate=baseline_wins / shared_count,
        mean_reciprocal_rank_delta=mean(reciprocal_rank_deltas),
        mean_target_coverage_delta=mean(target_coverage_deltas),
        mean_target_ndcg_delta=mean(target_ndcg_deltas),
        mean_precision_delta=mean(precision_deltas),
        mean_first_relevant_rank_delta=mean(first_relevant_rank_deltas),
        mean_target_rank_delta=mean(target_rank_deltas),
        mean_latency_delta_ms=mean(latency_deltas),
        bootstrap_samples=PAIRWISE_BOOTSTRAP_SAMPLES,
        confidence_level=PAIRWISE_CONFIDENCE_LEVEL,
        reciprocal_rank_delta_ci_low=reciprocal_rank_ci[0],
        reciprocal_rank_delta_ci_high=reciprocal_rank_ci[1],
        target_coverage_delta_ci_low=target_coverage_ci[0],
        target_coverage_delta_ci_high=target_coverage_ci[1],
        target_ndcg_delta_ci_low=target_ndcg_ci[0],
        target_ndcg_delta_ci_high=target_ndcg_ci[1],
        precision_delta_ci_low=precision_ci[0],
        precision_delta_ci_high=precision_ci[1],
        first_relevant_rank_delta_ci_low=first_rank_ci[0],
        first_relevant_rank_delta_ci_high=first_rank_ci[1],
        target_rank_delta_ci_low=target_rank_ci[0],
        target_rank_delta_ci_high=target_rank_ci[1],
        latency_delta_ci_low_ms=latency_ci[0],
        latency_delta_ci_high_ms=latency_ci[1],
    )


def results_by_query(results: list[RetrievalCaseResult]) -> dict[str, RetrievalCaseResult]:
    return {result.query: result for result in results}


def compare_case_results(candidate: RetrievalCaseResult, baseline: RetrievalCaseResult) -> int:
    candidate_score = case_score(candidate)
    baseline_score = case_score(baseline)
    if candidate_score > baseline_score:
        return 1
    if candidate_score < baseline_score:
        return -1
    return 0


def case_score(result: RetrievalCaseResult) -> tuple[float, float, float, float]:
    return (
        result.target_coverage_at_k,
        result.target_ndcg_at_k,
        result.reciprocal_rank,
        result.precision_at_k,
    )


def rank_sort_value(value: float | None) -> float:
    return -float(value) if value is not None else float("-inf")


def first_relevant_rank(result: RetrievalCaseResult, missing_rank: float) -> float:
    return float(result.matched_rank) if result.matched_rank is not None else missing_rank


def case_mean_target_rank(result: RetrievalCaseResult, missing_rank: float) -> float:
    expected_targets = result_expected_target_keys(result)
    if not expected_targets:
        return 0.0
    ranks = [
        float(result.target_key_matched_ranks.get(target, missing_rank))
        for target in expected_targets
    ]
    return mean(ranks)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_mean_interval(
    values: list[float],
    samples: int = PAIRWISE_BOOTSTRAP_SAMPLES,
    confidence_level: float = PAIRWISE_CONFIDENCE_LEVEL,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    sample_count = max(1, samples)
    sample_means = []
    value_count = len(values)
    for _ in range(sample_count):
        sample_means.append(mean([values[rng.randrange(value_count)] for _ in range(value_count)]))
    sample_means.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    low_index = int((alpha / 2) * (sample_count - 1))
    high_index = int((1.0 - alpha / 2) * (sample_count - 1))
    return sample_means[low_index], sample_means[high_index]


def stable_seed(*parts: str) -> int:
    raw = "|".join(parts)
    return sum((index + 1) * ord(character) for index, character in enumerate(raw))

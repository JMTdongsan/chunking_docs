from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from chunking_docs.evaluation.compare import (
    ChunkingComparison,
    ChunkingComparisonRow,
    ChunkingPairwiseComparison,
)
from chunking_docs.evaluation.gate import (
    case_group_metric_key,
    chunk_strategy_metric_key,
    parse_case_group_spec,
    retrieval_role_metric_key,
    source_metric_key,
    source_family_metric_key,
    target_type_metric_key,
)


class ChunkingComparisonGateCheck(BaseModel):
    name: str
    candidate: str
    metric: str
    operator: str
    actual: float | None = None
    threshold: float | None = None
    baseline: float | None = None
    delta: float | None = None
    ratio: float | None = None
    passed: bool
    message: str | None = None


class ChunkingComparisonGateReport(BaseModel):
    passed: bool
    candidate: str
    baseline_candidate: str | None = None
    metrics: dict[str, float | None]
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    chunk_strategy_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    retrieval_role_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    case_group_metrics: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    baseline_metrics: dict[str, float | None] = Field(default_factory=dict)
    pairwise_metrics: dict[str, float | None] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[ChunkingComparisonGateCheck] = Field(default_factory=list)


def load_chunking_comparison(path: Path) -> ChunkingComparison:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "comparison" in payload:
        payload = payload["comparison"]
    if payload is None:
        raise ValueError(f"{path} does not contain a chunking comparison report.")
    return ChunkingComparison.model_validate(payload)


def gate_chunking_comparison(
    comparison: ChunkingComparison,
    candidate: str | None = None,
    baseline_candidate: str | None = None,
    require_retrieval: bool = False,
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
    min_result_stability_rate: float | None = None,
    max_unstable_result_count: int | None = None,
    max_failed_queries: int | None = 0,
    max_total_chunk_chars: float | None = None,
    max_embedding_text_kchars: float | None = None,
    min_retrieval_score_per_embedding_kchar: float | None = None,
    min_target_coverage_per_embedding_kchar: float | None = None,
    min_target_ndcg_per_embedding_kchar: float | None = None,
    min_retrieval_score_per_mean_latency_ms: float | None = None,
    min_target_coverage_per_mean_latency_ms: float | None = None,
    min_target_ndcg_per_mean_latency_ms: float | None = None,
    min_retrieval_score_per_p95_latency_ms: float | None = None,
    min_target_coverage_per_p95_latency_ms: float | None = None,
    min_target_ndcg_per_p95_latency_ms: float | None = None,
    max_chunks_under_min_chars: int | None = None,
    max_chunks_over_max_chars: int | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_target_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    min_chunk_strategy_target_coverage: dict[str, float] | None = None,
    min_retrieval_role_target_coverage: dict[str, float] | None = None,
    min_case_group_target_coverage: dict[str, float] | None = None,
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
) -> ChunkingComparisonGateReport:
    selected_name = select_candidate_name(comparison, candidate)
    selected_row = find_row(comparison, selected_name)
    metrics = row_metrics(selected_row)
    baseline_metrics: dict[str, float | None] = {}
    pairwise_metrics: dict[str, float | None] = {}
    checks = [
        minimum_check(
            "min_quality_score",
            selected_name,
            "quality_score",
            metrics,
            min_quality_score,
        ),
        minimum_check(
            "min_page_coverage_ratio",
            selected_name,
            "page_coverage_ratio",
            metrics,
            min_page_coverage_ratio,
        ),
    ]
    if require_retrieval:
        checks.append(available_check("require_retrieval", selected_name, "retrieval_recall_at_k", metrics))
    checks.extend(
        check
        for check in [
            optional_minimum_check(
                "min_recall_at_k",
                selected_name,
                "retrieval_recall_at_k",
                metrics,
                min_recall_at_k,
            ),
            optional_minimum_check(
                "min_target_coverage_at_k",
                selected_name,
                "retrieval_target_coverage_at_k",
                metrics,
                min_target_coverage_at_k,
            ),
            optional_minimum_check(
                "min_target_ndcg_at_k",
                selected_name,
                "retrieval_mean_target_ndcg_at_k",
                metrics,
                min_target_ndcg_at_k,
            ),
            optional_minimum_check("min_mrr", selected_name, "retrieval_mrr", metrics, min_mrr),
            optional_minimum_check(
                "min_precision_at_k",
                selected_name,
                "retrieval_mean_precision_at_k",
                metrics,
                min_precision_at_k,
            ),
            optional_maximum_check(
                "max_mean_first_relevant_rank",
                selected_name,
                "retrieval_mean_first_relevant_rank",
                metrics,
                max_mean_first_relevant_rank,
            ),
            optional_maximum_check(
                "max_p95_first_relevant_rank",
                selected_name,
                "retrieval_p95_first_relevant_rank",
                metrics,
                max_p95_first_relevant_rank,
            ),
            optional_maximum_check(
                "max_mean_target_rank",
                selected_name,
                "retrieval_mean_target_rank",
                metrics,
                max_mean_target_rank,
            ),
            optional_maximum_check(
                "max_p95_target_rank",
                selected_name,
                "retrieval_p95_target_rank",
                metrics,
                max_p95_target_rank,
            ),
            optional_minimum_check(
                "min_visual_annotation_ratio",
                selected_name,
                "visual_annotation_ratio",
                metrics,
                min_visual_annotation_ratio,
            ),
            optional_minimum_check(
                "min_visual_text_coverage_ratio",
                selected_name,
                "visual_text_coverage_ratio",
                metrics,
                min_visual_text_coverage_ratio,
            ),
            optional_minimum_check(
                "min_visual_text_part_coverage_ratio",
                selected_name,
                "visual_text_part_coverage_ratio",
                metrics,
                min_visual_text_part_coverage_ratio,
            ),
            optional_maximum_check(
                "max_mean_latency_ms",
                selected_name,
                "retrieval_mean_latency_ms",
                metrics,
                max_mean_latency_ms,
            ),
            optional_maximum_check(
                "max_p95_latency_ms",
                selected_name,
                "retrieval_p95_latency_ms",
                metrics,
                max_p95_latency_ms,
            ),
            optional_minimum_check(
                "min_result_stability_rate",
                selected_name,
                "retrieval_result_stability_rate",
                metrics,
                min_result_stability_rate,
            ),
            optional_maximum_check(
                "max_unstable_result_count",
                selected_name,
                "retrieval_unstable_result_count",
                metrics,
                float(max_unstable_result_count) if max_unstable_result_count is not None else None,
            ),
            optional_maximum_check(
                "max_failed_queries",
                selected_name,
                "failed_query_count",
                metrics,
                float(max_failed_queries) if max_failed_queries is not None else None,
            ),
            optional_maximum_check(
                "max_total_chunk_chars",
                selected_name,
                "total_chunk_chars",
                metrics,
                max_total_chunk_chars,
            ),
            optional_maximum_check(
                "max_embedding_text_kchars",
                selected_name,
                "embedding_text_kchars",
                metrics,
                max_embedding_text_kchars,
            ),
            optional_minimum_check(
                "min_retrieval_score_per_embedding_kchar",
                selected_name,
                "retrieval_score_per_embedding_kchar",
                metrics,
                min_retrieval_score_per_embedding_kchar,
            ),
            optional_minimum_check(
                "min_target_coverage_per_embedding_kchar",
                selected_name,
                "target_coverage_per_embedding_kchar",
                metrics,
                min_target_coverage_per_embedding_kchar,
            ),
            optional_minimum_check(
                "min_target_ndcg_per_embedding_kchar",
                selected_name,
                "target_ndcg_per_embedding_kchar",
                metrics,
                min_target_ndcg_per_embedding_kchar,
            ),
            optional_minimum_check(
                "min_retrieval_score_per_mean_latency_ms",
                selected_name,
                "retrieval_score_per_mean_latency_ms",
                metrics,
                min_retrieval_score_per_mean_latency_ms,
            ),
            optional_minimum_check(
                "min_target_coverage_per_mean_latency_ms",
                selected_name,
                "target_coverage_per_mean_latency_ms",
                metrics,
                min_target_coverage_per_mean_latency_ms,
            ),
            optional_minimum_check(
                "min_target_ndcg_per_mean_latency_ms",
                selected_name,
                "target_ndcg_per_mean_latency_ms",
                metrics,
                min_target_ndcg_per_mean_latency_ms,
            ),
            optional_minimum_check(
                "min_retrieval_score_per_p95_latency_ms",
                selected_name,
                "retrieval_score_per_p95_latency_ms",
                metrics,
                min_retrieval_score_per_p95_latency_ms,
            ),
            optional_minimum_check(
                "min_target_coverage_per_p95_latency_ms",
                selected_name,
                "target_coverage_per_p95_latency_ms",
                metrics,
                min_target_coverage_per_p95_latency_ms,
            ),
            optional_minimum_check(
                "min_target_ndcg_per_p95_latency_ms",
                selected_name,
                "target_ndcg_per_p95_latency_ms",
                metrics,
                min_target_ndcg_per_p95_latency_ms,
            ),
            optional_maximum_check(
                "max_chunks_under_min_chars",
                selected_name,
                "chunks_under_min_chars",
                metrics,
                float(max_chunks_under_min_chars) if max_chunks_under_min_chars is not None else None,
            ),
            optional_maximum_check(
                "max_chunks_over_max_chars",
                selected_name,
                "chunks_over_max_chars",
                metrics,
                float(max_chunks_over_max_chars) if max_chunks_over_max_chars is not None else None,
            ),
        ]
        if check is not None
    )
    checks.extend(target_type_coverage_checks(selected_name, metrics, min_target_type_coverage or {}))
    checks.extend(
        source_target_coverage_checks(
            selected_name,
            metrics,
            min_source_target_coverage or {},
        )
    )
    checks.extend(
        source_family_target_coverage_checks(
            selected_name,
            metrics,
            min_source_family_target_coverage or {},
        )
    )
    checks.extend(
        grouped_target_coverage_checks(
            selected_name,
            metrics,
            min_chunk_strategy_target_coverage or {},
            metric_key_fn=chunk_strategy_metric_key,
            check_prefix="min_chunk_strategy_target_coverage",
        )
    )
    checks.extend(
        grouped_target_coverage_checks(
            selected_name,
            metrics,
            min_retrieval_role_target_coverage or {},
            metric_key_fn=retrieval_role_metric_key,
            check_prefix="min_retrieval_role_target_coverage",
        )
    )
    checks.extend(
        case_group_target_coverage_checks(
            selected_name,
            metrics,
            min_case_group_target_coverage or {},
        )
    )

    if baseline_candidate is not None:
        baseline_row = find_row(comparison, baseline_candidate)
        baseline_metrics = row_metrics(baseline_row)
        checks.extend(
            baseline_drop_checks(
                selected_name,
                metrics,
                baseline_metrics,
                {
                    "quality_score": ("max_quality_score_drop", max_quality_drop),
                    "retrieval_recall_at_k": ("max_recall_at_k_drop", max_recall_drop),
                    "retrieval_target_coverage_at_k": (
                        "max_target_coverage_at_k_drop",
                        max_target_coverage_drop,
                    ),
                    "retrieval_mean_target_ndcg_at_k": (
                        "max_target_ndcg_at_k_drop",
                        max_target_ndcg_drop,
                    ),
                    "retrieval_mean_precision_at_k": (
                        "max_precision_at_k_drop",
                        max_precision_drop,
                    ),
                },
            )
        )
        checks.extend(
            latency_ratio_checks(
                selected_name,
                metrics,
                baseline_metrics,
                {
                    "retrieval_mean_latency_ms": ("max_mean_latency_ms_ratio", max_mean_latency_ratio),
                    "retrieval_p95_latency_ms": ("max_p95_latency_ms_ratio", max_p95_latency_ratio),
                },
            )
        )
        if any(
            value is not None
            for value in [
                min_pairwise_shared_queries,
                min_pairwise_win_rate,
                min_pairwise_target_coverage_lift,
                min_pairwise_target_ndcg_lift,
                min_pairwise_mrr_lift,
                min_pairwise_precision_lift,
                min_pairwise_target_coverage_ci_low,
                min_pairwise_target_ndcg_ci_low,
                min_pairwise_mrr_ci_low,
                min_pairwise_precision_ci_low,
                max_pairwise_mean_first_relevant_rank_delta,
                max_pairwise_mean_target_rank_delta,
                max_pairwise_first_relevant_rank_delta_ci_high,
                max_pairwise_target_rank_delta_ci_high,
                max_pairwise_mean_latency_delta_ms,
            ]
        ):
            pairwise = find_pairwise_comparison(comparison, selected_name, baseline_candidate)
            pairwise_metrics = pairwise_comparison_metrics(pairwise)
            checks.extend(
                check
                for check in [
                    optional_minimum_check(
                        "min_pairwise_shared_queries",
                        selected_name,
                        "pairwise_shared_query_count",
                        pairwise_metrics,
                        float(min_pairwise_shared_queries)
                        if min_pairwise_shared_queries is not None
                        else None,
                    ),
                    optional_minimum_check(
                        "min_pairwise_win_rate",
                        selected_name,
                        "pairwise_candidate_win_rate",
                        pairwise_metrics,
                        min_pairwise_win_rate,
                    ),
                    optional_minimum_check(
                        "min_pairwise_target_coverage_lift",
                        selected_name,
                        "pairwise_mean_target_coverage_delta",
                        pairwise_metrics,
                        min_pairwise_target_coverage_lift,
                    ),
                    optional_minimum_check(
                        "min_pairwise_target_ndcg_lift",
                        selected_name,
                        "pairwise_mean_target_ndcg_delta",
                        pairwise_metrics,
                        min_pairwise_target_ndcg_lift,
                    ),
                    optional_minimum_check(
                        "min_pairwise_mrr_lift",
                        selected_name,
                        "pairwise_mean_reciprocal_rank_delta",
                        pairwise_metrics,
                        min_pairwise_mrr_lift,
                    ),
                    optional_minimum_check(
                        "min_pairwise_precision_lift",
                        selected_name,
                        "pairwise_mean_precision_delta",
                        pairwise_metrics,
                        min_pairwise_precision_lift,
                    ),
                    optional_minimum_check(
                        "min_pairwise_target_coverage_ci_low",
                        selected_name,
                        "pairwise_target_coverage_delta_ci_low",
                        pairwise_metrics,
                        min_pairwise_target_coverage_ci_low,
                    ),
                    optional_minimum_check(
                        "min_pairwise_target_ndcg_ci_low",
                        selected_name,
                        "pairwise_target_ndcg_delta_ci_low",
                        pairwise_metrics,
                        min_pairwise_target_ndcg_ci_low,
                    ),
                    optional_minimum_check(
                        "min_pairwise_mrr_ci_low",
                        selected_name,
                        "pairwise_reciprocal_rank_delta_ci_low",
                        pairwise_metrics,
                        min_pairwise_mrr_ci_low,
                    ),
                    optional_minimum_check(
                        "min_pairwise_precision_ci_low",
                        selected_name,
                        "pairwise_precision_delta_ci_low",
                        pairwise_metrics,
                        min_pairwise_precision_ci_low,
                    ),
                    optional_maximum_check(
                        "max_pairwise_mean_first_relevant_rank_delta",
                        selected_name,
                        "pairwise_mean_first_relevant_rank_delta",
                        pairwise_metrics,
                        max_pairwise_mean_first_relevant_rank_delta,
                    ),
                    optional_maximum_check(
                        "max_pairwise_mean_target_rank_delta",
                        selected_name,
                        "pairwise_mean_target_rank_delta",
                        pairwise_metrics,
                        max_pairwise_mean_target_rank_delta,
                    ),
                    optional_maximum_check(
                        "max_pairwise_first_relevant_rank_delta_ci_high",
                        selected_name,
                        "pairwise_first_relevant_rank_delta_ci_high",
                        pairwise_metrics,
                        max_pairwise_first_relevant_rank_delta_ci_high,
                    ),
                    optional_maximum_check(
                        "max_pairwise_target_rank_delta_ci_high",
                        selected_name,
                        "pairwise_target_rank_delta_ci_high",
                        pairwise_metrics,
                        max_pairwise_target_rank_delta_ci_high,
                    ),
                    optional_maximum_check(
                        "max_pairwise_mean_latency_delta_ms",
                        selected_name,
                        "pairwise_mean_latency_delta_ms",
                        pairwise_metrics,
                        max_pairwise_mean_latency_delta_ms,
                    ),
                ]
                if check is not None
            )

    failed_checks = [check.name for check in checks if not check.passed]
    return ChunkingComparisonGateReport(
        passed=not failed_checks,
        candidate=selected_name,
        baseline_candidate=baseline_candidate,
        metrics=metrics,
        target_metrics=selected_row.target_metrics,
        source_metrics=selected_row.source_metrics,
        source_family_metrics=selected_row.source_family_metrics,
        chunk_strategy_metrics=selected_row.chunk_strategy_metrics,
        retrieval_role_metrics=selected_row.retrieval_role_metrics,
        case_group_metrics=selected_row.case_group_metrics,
        baseline_metrics=baseline_metrics,
        pairwise_metrics=pairwise_metrics,
        failed_checks=failed_checks,
        checks=checks,
    )


def select_candidate_name(comparison: ChunkingComparison, candidate: str | None) -> str:
    if candidate:
        return candidate
    if comparison.best_by_retrieval:
        return comparison.best_by_retrieval
    if comparison.best_by_quality:
        return comparison.best_by_quality
    if comparison.rows:
        return comparison.rows[0].name
    raise ValueError("Chunking comparison has no candidates.")


def find_row(comparison: ChunkingComparison, candidate: str) -> ChunkingComparisonRow:
    for row in comparison.rows:
        if row.name == candidate:
            return row
    available = ", ".join(row.name for row in comparison.rows)
    raise ValueError(f"Candidate '{candidate}' was not found. Available candidates: {available}")


def row_metrics(row: ChunkingComparisonRow) -> dict[str, float | None]:
    metrics = {
        "chunk_count": float(row.chunk_count),
        "total_chunk_chars": row.total_chunk_chars,
        "mean_chunk_chars": row.mean_chunk_chars,
        "p95_chunk_chars": row.p95_chunk_chars,
        "embedding_text_kchars": row.embedding_text_kchars,
        "quality_score": row.quality_score,
        "retrieval_score": row.retrieval_score,
        "retrieval_score_per_embedding_kchar": row.retrieval_score_per_embedding_kchar,
        "target_coverage_per_embedding_kchar": row.target_coverage_per_embedding_kchar,
        "target_ndcg_per_embedding_kchar": row.target_ndcg_per_embedding_kchar,
        "retrieval_score_per_mean_latency_ms": row.retrieval_score_per_mean_latency_ms,
        "target_coverage_per_mean_latency_ms": row.target_coverage_per_mean_latency_ms,
        "target_ndcg_per_mean_latency_ms": row.target_ndcg_per_mean_latency_ms,
        "retrieval_score_per_p95_latency_ms": row.retrieval_score_per_p95_latency_ms,
        "target_coverage_per_p95_latency_ms": row.target_coverage_per_p95_latency_ms,
        "target_ndcg_per_p95_latency_ms": row.target_ndcg_per_p95_latency_ms,
        "retrieval_hit_rate": row.retrieval_hit_rate,
        "retrieval_recall_at_k": row.retrieval_recall_at_k,
        "retrieval_mrr": row.retrieval_mrr,
        "retrieval_target_coverage_at_k": row.retrieval_target_coverage_at_k,
        "retrieval_mean_target_ndcg_at_k": row.retrieval_mean_target_ndcg_at_k,
        "retrieval_mean_precision_at_k": row.retrieval_mean_precision_at_k,
        "retrieval_mean_latency_ms": row.retrieval_mean_latency_ms,
        "retrieval_p95_latency_ms": row.retrieval_p95_latency_ms,
        "retrieval_mean_first_relevant_rank": row.retrieval_mean_first_relevant_rank,
        "retrieval_p95_first_relevant_rank": row.retrieval_p95_first_relevant_rank,
        "retrieval_mean_target_rank": row.retrieval_mean_target_rank,
        "retrieval_p95_target_rank": row.retrieval_p95_target_rank,
        "retrieval_ranked_expected_case_count": row.retrieval_ranked_expected_case_count,
        "retrieval_ranked_target_count": row.retrieval_ranked_target_count,
        "retrieval_unstable_result_count": row.retrieval_unstable_result_count,
        "retrieval_result_stability_rate": row.retrieval_result_stability_rate,
        "failed_query_count": float(len(row.failed_queries)),
        "page_coverage_ratio": row.page_coverage_ratio,
        "visual_annotation_ratio": row.visual_annotation_ratio,
        "visual_text_asset_count": float(row.visual_text_asset_count),
        "visual_text_covered_asset_count": float(row.visual_text_covered_asset_count),
        "visual_text_coverage_ratio": row.visual_text_coverage_ratio,
        "visual_text_part_count": float(row.visual_text_part_count),
        "visual_text_covered_part_count": float(row.visual_text_covered_part_count),
        "visual_text_part_coverage_ratio": row.visual_text_part_coverage_ratio,
        "standalone_visual_chunk_count": float(row.standalone_visual_chunk_count),
        "standalone_visual_text_asset_count": float(row.standalone_visual_text_asset_count),
        "chunks_under_min_chars": float(row.chunks_under_min_chars),
        "chunks_over_max_chars": float(row.chunks_over_max_chars),
    }
    for target_type, target_type_metrics in row.target_metrics.items():
        for key, value in target_type_metrics.items():
            metrics[target_type_metric_key(target_type, key)] = value
    for source, source_metrics in row.source_metrics.items():
        for key, value in source_metrics.items():
            metrics[source_metric_key(source, key)] = value
    for family, family_metrics in row.source_family_metrics.items():
        for key, value in family_metrics.items():
            metrics[source_family_metric_key(family, key)] = value
    for strategy, strategy_metrics in row.chunk_strategy_metrics.items():
        for key, value in strategy_metrics.items():
            metrics[chunk_strategy_metric_key(strategy, key)] = value
    for role, role_metrics in row.retrieval_role_metrics.items():
        for key, value in role_metrics.items():
            metrics[retrieval_role_metric_key(role, key)] = value
    for group_name, group_values in row.case_group_metrics.items():
        for group_value, group_metrics in group_values.items():
            for key, value in group_metrics.items():
                metrics[case_group_metric_key(group_name, group_value, key)] = value
    return metrics


def find_pairwise_comparison(
    comparison: ChunkingComparison,
    candidate: str,
    baseline: str,
) -> ChunkingPairwiseComparison | None:
    for pairwise in comparison.pairwise:
        if pairwise.candidate == candidate and pairwise.baseline == baseline:
            return pairwise
    return None


def pairwise_comparison_metrics(
    pairwise: ChunkingPairwiseComparison | None,
) -> dict[str, float | None]:
    if pairwise is None:
        return {
            "pairwise_shared_query_count": None,
            "pairwise_candidate_win_count": None,
            "pairwise_baseline_win_count": None,
            "pairwise_tie_count": None,
            "pairwise_candidate_win_rate": None,
            "pairwise_baseline_win_rate": None,
            "pairwise_mean_reciprocal_rank_delta": None,
            "pairwise_mean_target_coverage_delta": None,
            "pairwise_mean_target_ndcg_delta": None,
            "pairwise_mean_precision_delta": None,
            "pairwise_mean_first_relevant_rank_delta": None,
            "pairwise_mean_target_rank_delta": None,
            "pairwise_mean_latency_delta_ms": None,
            "pairwise_bootstrap_samples": None,
            "pairwise_confidence_level": None,
            "pairwise_reciprocal_rank_delta_ci_low": None,
            "pairwise_reciprocal_rank_delta_ci_high": None,
            "pairwise_target_coverage_delta_ci_low": None,
            "pairwise_target_coverage_delta_ci_high": None,
            "pairwise_target_ndcg_delta_ci_low": None,
            "pairwise_target_ndcg_delta_ci_high": None,
            "pairwise_precision_delta_ci_low": None,
            "pairwise_precision_delta_ci_high": None,
            "pairwise_first_relevant_rank_delta_ci_low": None,
            "pairwise_first_relevant_rank_delta_ci_high": None,
            "pairwise_target_rank_delta_ci_low": None,
            "pairwise_target_rank_delta_ci_high": None,
            "pairwise_latency_delta_ci_low_ms": None,
            "pairwise_latency_delta_ci_high_ms": None,
        }
    return {
        "pairwise_shared_query_count": float(pairwise.shared_query_count),
        "pairwise_candidate_win_count": float(pairwise.candidate_win_count),
        "pairwise_baseline_win_count": float(pairwise.baseline_win_count),
        "pairwise_tie_count": float(pairwise.tie_count),
        "pairwise_candidate_win_rate": pairwise.candidate_win_rate,
        "pairwise_baseline_win_rate": pairwise.baseline_win_rate,
        "pairwise_mean_reciprocal_rank_delta": pairwise.mean_reciprocal_rank_delta,
        "pairwise_mean_target_coverage_delta": pairwise.mean_target_coverage_delta,
        "pairwise_mean_target_ndcg_delta": pairwise.mean_target_ndcg_delta,
        "pairwise_mean_precision_delta": pairwise.mean_precision_delta,
        "pairwise_mean_first_relevant_rank_delta": pairwise.mean_first_relevant_rank_delta,
        "pairwise_mean_target_rank_delta": pairwise.mean_target_rank_delta,
        "pairwise_mean_latency_delta_ms": pairwise.mean_latency_delta_ms,
        "pairwise_bootstrap_samples": float(pairwise.bootstrap_samples),
        "pairwise_confidence_level": pairwise.confidence_level,
        "pairwise_reciprocal_rank_delta_ci_low": pairwise.reciprocal_rank_delta_ci_low,
        "pairwise_reciprocal_rank_delta_ci_high": pairwise.reciprocal_rank_delta_ci_high,
        "pairwise_target_coverage_delta_ci_low": pairwise.target_coverage_delta_ci_low,
        "pairwise_target_coverage_delta_ci_high": pairwise.target_coverage_delta_ci_high,
        "pairwise_target_ndcg_delta_ci_low": pairwise.target_ndcg_delta_ci_low,
        "pairwise_target_ndcg_delta_ci_high": pairwise.target_ndcg_delta_ci_high,
        "pairwise_precision_delta_ci_low": pairwise.precision_delta_ci_low,
        "pairwise_precision_delta_ci_high": pairwise.precision_delta_ci_high,
        "pairwise_first_relevant_rank_delta_ci_low": (
            pairwise.first_relevant_rank_delta_ci_low
        ),
        "pairwise_first_relevant_rank_delta_ci_high": (
            pairwise.first_relevant_rank_delta_ci_high
        ),
        "pairwise_target_rank_delta_ci_low": pairwise.target_rank_delta_ci_low,
        "pairwise_target_rank_delta_ci_high": pairwise.target_rank_delta_ci_high,
        "pairwise_latency_delta_ci_low_ms": pairwise.latency_delta_ci_low_ms,
        "pairwise_latency_delta_ci_high_ms": pairwise.latency_delta_ci_high_ms,
    }


def target_type_coverage_checks(
    candidate: str,
    metrics: dict[str, float | None],
    thresholds: dict[str, float],
) -> list[ChunkingComparisonGateCheck]:
    checks = []
    for target_type, threshold in sorted(thresholds.items()):
        normalized_target_type = target_type.strip().lower()
        metric = target_type_metric_key(normalized_target_type, "coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_target_type_coverage:{normalized_target_type}",
                candidate,
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def source_family_target_coverage_checks(
    candidate: str,
    metrics: dict[str, float | None],
    thresholds: dict[str, float],
) -> list[ChunkingComparisonGateCheck]:
    checks = []
    for family, threshold in sorted(thresholds.items()):
        normalized_family = family.strip().lower()
        metric = source_family_metric_key(normalized_family, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_source_family_target_coverage:{normalized_family}",
                candidate,
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def source_target_coverage_checks(
    candidate: str,
    metrics: dict[str, float | None],
    thresholds: dict[str, float],
) -> list[ChunkingComparisonGateCheck]:
    checks = []
    for source, threshold in sorted(thresholds.items()):
        normalized_source = source.strip().lower()
        metric = source_metric_key(normalized_source, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_source_target_coverage:{normalized_source}",
                candidate,
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def grouped_target_coverage_checks(
    candidate: str,
    metrics: dict[str, float | None],
    thresholds: dict[str, float],
    metric_key_fn,
    check_prefix: str,
) -> list[ChunkingComparisonGateCheck]:
    checks = []
    for group, threshold in sorted(thresholds.items()):
        normalized_group = group.strip().lower()
        metric = metric_key_fn(normalized_group, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"{check_prefix}:{normalized_group}",
                candidate,
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def case_group_target_coverage_checks(
    candidate: str,
    metrics: dict[str, float | None],
    thresholds: dict[str, float],
) -> list[ChunkingComparisonGateCheck]:
    checks = []
    for group_spec, threshold in sorted(thresholds.items()):
        group_name, group_value = parse_case_group_spec(group_spec)
        metric = case_group_metric_key(group_name, group_value, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_case_group_target_coverage:{group_name}:{group_value}",
                candidate,
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def optional_minimum_check(
    name: str,
    candidate: str,
    metric: str,
    metrics: dict[str, float | None],
    threshold: float | None,
) -> ChunkingComparisonGateCheck | None:
    if threshold is None:
        return None
    return minimum_check(name, candidate, metric, metrics, threshold)


def minimum_check(
    name: str,
    candidate: str,
    metric: str,
    metrics: dict[str, float | None],
    threshold: float,
) -> ChunkingComparisonGateCheck:
    actual = metrics.get(metric)
    return ChunkingComparisonGateCheck(
        name=name,
        candidate=candidate,
        metric=metric,
        operator=">=",
        actual=actual,
        threshold=threshold,
        passed=actual is not None and actual >= threshold,
    )


def optional_maximum_check(
    name: str,
    candidate: str,
    metric: str,
    metrics: dict[str, float | None],
    threshold: float | None,
) -> ChunkingComparisonGateCheck | None:
    if threshold is None:
        return None
    return maximum_check(name, candidate, metric, metrics, threshold)


def maximum_check(
    name: str,
    candidate: str,
    metric: str,
    metrics: dict[str, float | None],
    threshold: float,
) -> ChunkingComparisonGateCheck:
    actual = metrics.get(metric)
    return ChunkingComparisonGateCheck(
        name=name,
        candidate=candidate,
        metric=metric,
        operator="<=",
        actual=actual,
        threshold=threshold,
        passed=actual is not None and actual <= threshold,
    )


def available_check(
    name: str,
    candidate: str,
    metric: str,
    metrics: dict[str, float | None],
) -> ChunkingComparisonGateCheck:
    actual = metrics.get(metric)
    return ChunkingComparisonGateCheck(
        name=name,
        candidate=candidate,
        metric=metric,
        operator="exists",
        actual=actual,
        passed=actual is not None,
        message=None if actual is not None else "The selected candidate has no retrieval metrics.",
    )


def baseline_drop_checks(
    candidate: str,
    metrics: dict[str, float | None],
    baseline_metrics: dict[str, float | None],
    thresholds: dict[str, tuple[str, float | None]],
) -> list[ChunkingComparisonGateCheck]:
    checks = []
    for metric, (name, threshold) in thresholds.items():
        if threshold is None:
            continue
        actual = metrics.get(metric)
        baseline = baseline_metrics.get(metric)
        delta = actual - baseline if actual is not None and baseline is not None else None
        checks.append(
            ChunkingComparisonGateCheck(
                name=name,
                candidate=candidate,
                metric=metric,
                operator="baseline_drop<=",
                actual=actual,
                baseline=baseline,
                delta=delta,
                threshold=threshold,
                passed=actual is not None and baseline is not None and (baseline - actual) <= threshold,
            )
        )
    return checks


def latency_ratio_checks(
    candidate: str,
    metrics: dict[str, float | None],
    baseline_metrics: dict[str, float | None],
    thresholds: dict[str, tuple[str, float | None]],
) -> list[ChunkingComparisonGateCheck]:
    checks = []
    for metric, (name, threshold) in thresholds.items():
        if threshold is None:
            continue
        actual = metrics.get(metric)
        baseline = baseline_metrics.get(metric)
        ratio = safe_ratio(actual, baseline)
        checks.append(
            ChunkingComparisonGateCheck(
                name=name,
                candidate=candidate,
                metric=metric,
                operator="actual/baseline<=",
                actual=actual,
                baseline=baseline,
                delta=actual - baseline if actual is not None and baseline is not None else None,
                ratio=ratio,
                threshold=threshold,
                passed=actual is not None and (actual <= 0.0 if ratio is None else ratio <= threshold),
            )
        )
    return checks


def safe_ratio(actual: float | None, baseline: float | None) -> float | None:
    if actual is None or baseline is None or baseline <= 0:
        return None
    return actual / baseline


def chunking_gate_summary_payload(report: ChunkingComparisonGateReport) -> dict:
    return {
        "passed": report.passed,
        "candidate": report.candidate,
        "baseline_candidate": report.baseline_candidate,
        "failed_checks": report.failed_checks,
        "metrics": report.metrics,
        "target_metrics": report.target_metrics,
        "source_metrics": report.source_metrics,
        "source_family_metrics": report.source_family_metrics,
        "chunk_strategy_metrics": report.chunk_strategy_metrics,
        "retrieval_role_metrics": report.retrieval_role_metrics,
        "case_group_metrics": report.case_group_metrics,
        "baseline_metrics": report.baseline_metrics,
        "pairwise_metrics": report.pairwise_metrics,
    }

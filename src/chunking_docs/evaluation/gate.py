from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import RetrievalEvaluation


class RetrievalGateCheck(BaseModel):
    name: str
    metric: str
    operator: str
    actual: float
    threshold: float
    baseline: float | None = None
    delta: float | None = None
    ratio: float | None = None
    passed: bool


class RetrievalGateReport(BaseModel):
    passed: bool
    metrics: dict[str, float]
    source_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    case_group_source_metrics: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ] = Field(default_factory=dict)
    case_group_source_family_metrics: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ] = Field(default_factory=dict)
    chunk_strategy_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    retrieval_role_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    case_group_metrics: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[RetrievalGateCheck] = Field(default_factory=list)


def gate_retrieval_evaluation(
    evaluation: RetrievalEvaluation,
    baseline: RetrievalEvaluation | None = None,
    min_case_count: int = 0,
    min_expected_case_count: int = 0,
    min_expected_target_count: int = 0,
    min_passed_query_count: int = 0,
    max_failed_query_count: int | None = None,
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
    min_result_stability_rate: float = 0.0,
    max_unstable_result_count: int | None = None,
    max_excluded_target_hit_rate: float | None = None,
    max_excluded_query_hit_rate: float | None = None,
    max_excluded_hit_query_count: int | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_target_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    min_case_group_source_target_coverage: dict[str, float] | None = None,
    min_case_group_source_family_target_coverage: dict[str, float] | None = None,
    max_source_excluded_target_hit_rate: dict[str, float] | None = None,
    max_source_family_excluded_target_hit_rate: dict[str, float] | None = None,
    min_chunk_strategy_target_coverage: dict[str, float] | None = None,
    min_retrieval_role_target_coverage: dict[str, float] | None = None,
    max_chunk_strategy_excluded_target_hit_rate: dict[str, float] | None = None,
    max_retrieval_role_excluded_target_hit_rate: dict[str, float] | None = None,
    min_case_group_target_coverage: dict[str, float] | None = None,
    max_recall_drop: float | None = None,
    max_target_coverage_drop: float | None = None,
    max_target_ndcg_drop: float | None = None,
    max_precision_drop: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
) -> RetrievalGateReport:
    """Evaluate retrieval metrics against absolute floors and baseline regression limits."""

    source_metrics = retrieval_source_metrics(evaluation)
    source_family_metrics = retrieval_source_family_metrics(evaluation)
    case_group_source_metrics = retrieval_case_group_source_metrics(evaluation)
    case_group_source_family_metrics = retrieval_case_group_source_family_metrics(evaluation)
    target_metrics = retrieval_target_metrics(evaluation)
    chunk_strategy_metrics = retrieval_chunk_strategy_metrics(evaluation)
    retrieval_role_metrics = retrieval_role_metrics_payload(evaluation)
    case_group_metrics = retrieval_case_group_metrics(evaluation)
    metrics = retrieval_metrics(
        evaluation,
        source_metrics,
        source_family_metrics,
        case_group_source_metrics,
        case_group_source_family_metrics,
        target_metrics,
        chunk_strategy_metrics,
        retrieval_role_metrics,
        case_group_metrics,
    )
    baseline_metrics = (
        retrieval_metrics(
            baseline,
            retrieval_source_metrics(baseline),
            retrieval_source_family_metrics(baseline),
            retrieval_case_group_source_metrics(baseline),
            retrieval_case_group_source_family_metrics(baseline),
            retrieval_target_metrics(baseline),
            retrieval_chunk_strategy_metrics(baseline),
            retrieval_role_metrics_payload(baseline),
            retrieval_case_group_metrics(baseline),
        )
        if baseline is not None
        else {}
    )
    checks = [
        minimum_check("min_case_count", "case_count", metrics, float(min_case_count)),
        minimum_check(
            "min_expected_case_count",
            "expected_case_count",
            metrics,
            float(min_expected_case_count),
        ),
        minimum_check(
            "min_expected_target_count",
            "expected_target_count",
            metrics,
            float(min_expected_target_count),
        ),
        minimum_check(
            "min_passed_query_count",
            "passed_query_count",
            metrics,
            float(min_passed_query_count),
        ),
        minimum_check("min_recall_at_k", "recall_at_k", metrics, min_recall_at_k),
        minimum_check(
            "min_target_coverage_at_k",
            "target_coverage_at_k",
            metrics,
            min_target_coverage_at_k,
        ),
        minimum_check(
            "min_target_ndcg_at_k",
            "mean_target_ndcg_at_k",
            metrics,
            min_target_ndcg_at_k,
        ),
        minimum_check("min_mrr", "mrr", metrics, min_mrr),
        minimum_check("min_precision_at_k", "mean_precision_at_k", metrics, min_precision_at_k),
        minimum_check(
            "min_result_stability_rate",
            "result_stability_rate",
            metrics,
            min_result_stability_rate,
        ),
    ]
    if max_failed_query_count is not None:
        checks.append(
            maximum_check(
                "max_failed_query_count",
                "failed_query_count",
                metrics,
                float(max_failed_query_count),
            )
        )
    if max_unstable_result_count is not None:
        checks.append(
            maximum_check(
                "max_unstable_result_count",
                "unstable_result_count",
                metrics,
                float(max_unstable_result_count),
            )
        )
    if max_excluded_target_hit_rate is not None:
        checks.append(
            maximum_check(
                "max_excluded_target_hit_rate",
                "excluded_target_hit_rate",
                metrics,
                max_excluded_target_hit_rate,
            )
        )
    if max_excluded_query_hit_rate is not None:
        checks.append(
            maximum_check(
                "max_excluded_query_hit_rate",
                "excluded_query_hit_rate",
                metrics,
                max_excluded_query_hit_rate,
            )
        )
    if max_excluded_hit_query_count is not None:
        checks.append(
            maximum_check(
                "max_excluded_hit_query_count",
                "excluded_hit_query_count",
                metrics,
                float(max_excluded_hit_query_count),
            )
        )
    if max_mean_latency_ms is not None:
        checks.append(maximum_check("max_mean_latency_ms", "mean_latency_ms", metrics, max_mean_latency_ms))
    if max_p95_latency_ms is not None:
        checks.append(maximum_check("max_p95_latency_ms", "p95_latency_ms", metrics, max_p95_latency_ms))
    if max_mean_first_relevant_rank is not None:
        checks.append(
            maximum_check(
                "max_mean_first_relevant_rank",
                "mean_first_relevant_rank",
                metrics,
                max_mean_first_relevant_rank,
            )
        )
    if max_p95_first_relevant_rank is not None:
        checks.append(
            maximum_check(
                "max_p95_first_relevant_rank",
                "p95_first_relevant_rank",
                metrics,
                max_p95_first_relevant_rank,
            )
        )
    if max_mean_target_rank is not None:
        checks.append(maximum_check("max_mean_target_rank", "mean_target_rank", metrics, max_mean_target_rank))
    if max_p95_target_rank is not None:
        checks.append(maximum_check("max_p95_target_rank", "p95_target_rank", metrics, max_p95_target_rank))
    checks.extend(
        target_type_coverage_checks(
            metrics,
            min_target_type_coverage or {},
        )
    )
    checks.extend(
        source_target_coverage_checks(
            metrics,
            min_source_target_coverage or {},
        )
    )
    checks.extend(
        source_family_target_coverage_checks(
            metrics,
            min_source_family_target_coverage or {},
        )
    )
    checks.extend(
        case_group_source_target_coverage_checks(
            metrics,
            min_case_group_source_target_coverage or {},
            family=False,
        )
    )
    checks.extend(
        case_group_source_target_coverage_checks(
            metrics,
            min_case_group_source_family_target_coverage or {},
            family=True,
        )
    )
    checks.extend(
        source_excluded_target_hit_rate_checks(
            metrics,
            max_source_excluded_target_hit_rate or {},
        )
    )
    checks.extend(
        source_family_excluded_target_hit_rate_checks(
            metrics,
            max_source_family_excluded_target_hit_rate or {},
        )
    )
    checks.extend(
        grouped_target_coverage_checks(
            metrics,
            min_chunk_strategy_target_coverage or {},
            metric_key_fn=chunk_strategy_metric_key,
            check_prefix="min_chunk_strategy_target_coverage",
        )
    )
    checks.extend(
        grouped_target_coverage_checks(
            metrics,
            min_retrieval_role_target_coverage or {},
            metric_key_fn=retrieval_role_metric_key,
            check_prefix="min_retrieval_role_target_coverage",
        )
    )
    checks.extend(
        grouped_excluded_target_hit_rate_checks(
            metrics,
            max_chunk_strategy_excluded_target_hit_rate or {},
            metric_key_fn=chunk_strategy_metric_key,
            check_prefix="max_chunk_strategy_excluded_target_hit_rate",
        )
    )
    checks.extend(
        grouped_excluded_target_hit_rate_checks(
            metrics,
            max_retrieval_role_excluded_target_hit_rate or {},
            metric_key_fn=retrieval_role_metric_key,
            check_prefix="max_retrieval_role_excluded_target_hit_rate",
        )
    )
    checks.extend(
        case_group_target_coverage_checks(metrics, min_case_group_target_coverage or {})
    )
    if baseline is not None:
        checks.extend(
            baseline_drop_checks(
                metrics,
                baseline_metrics,
                {
                    "recall_at_k": max_recall_drop,
                    "target_coverage_at_k": max_target_coverage_drop,
                    "mean_target_ndcg_at_k": max_target_ndcg_drop,
                    "mean_precision_at_k": max_precision_drop,
                },
            )
        )
        checks.extend(
            latency_ratio_checks(
                metrics,
                baseline_metrics,
                {
                    "mean_latency_ms": max_mean_latency_ratio,
                    "p95_latency_ms": max_p95_latency_ratio,
                },
            )
        )
    failed_checks = [check.name for check in checks if not check.passed]
    return RetrievalGateReport(
        passed=not failed_checks,
        metrics=metrics,
        source_metrics=source_metrics,
        target_metrics=target_metrics,
        source_family_metrics=source_family_metrics,
        case_group_source_metrics=case_group_source_metrics,
        case_group_source_family_metrics=case_group_source_family_metrics,
        chunk_strategy_metrics=chunk_strategy_metrics,
        retrieval_role_metrics=retrieval_role_metrics,
        case_group_metrics=case_group_metrics,
        baseline_metrics=baseline_metrics,
        failed_checks=failed_checks,
        checks=checks,
    )


def retrieval_metrics(
    evaluation: RetrievalEvaluation | None,
    source_metrics: dict[str, dict[str, float]] | None = None,
    source_family_metrics: dict[str, dict[str, float]] | None = None,
    case_group_source_metrics: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ] | None = None,
    case_group_source_family_metrics: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ] | None = None,
    target_metrics: dict[str, dict[str, float]] | None = None,
    chunk_strategy_metrics: dict[str, dict[str, float]] | None = None,
    retrieval_role_metrics: dict[str, dict[str, float]] | None = None,
    case_group_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, float]:
    if evaluation is None:
        return {}
    metrics = {
        "case_count": float(evaluation.case_count),
        "expected_case_count": float(evaluation.expected_case_count),
        "passed_query_count": float(evaluation.passed_count),
        "failed_query_count": float(evaluation.failed_count),
        **retrieval_expected_target_count_metrics(evaluation),
        "hit_rate": evaluation.hit_rate,
        "recall_at_k": evaluation.recall_at_k,
        "mrr": evaluation.mrr,
        "target_coverage_at_k": evaluation.target_coverage_at_k,
        "mean_target_ndcg_at_k": evaluation.mean_target_ndcg_at_k,
        "mean_precision_at_k": evaluation.mean_precision_at_k,
        "excluded_query_count": float(evaluation.excluded_query_count),
        "excluded_hit_query_count": float(evaluation.excluded_hit_query_count),
        "excluded_query_hit_rate": evaluation.excluded_query_hit_rate,
        "excluded_target_count": float(evaluation.excluded_target_count),
        "excluded_matched_target_count": float(evaluation.excluded_matched_target_count),
        "excluded_target_hit_rate": evaluation.excluded_target_hit_rate,
        "mean_latency_ms": evaluation.mean_latency_ms,
        "p95_latency_ms": evaluation.p95_latency_ms,
        "unstable_result_count": float(evaluation.unstable_result_count),
        "result_stability_rate": evaluation.result_stability_rate,
    }
    metrics.update(retrieval_rank_metrics(evaluation))
    for source, source_metric_values in (source_metrics or {}).items():
        for key, value in source_metric_values.items():
            metrics[source_metric_key(source, key)] = value
    for target_type, target_type_metrics in (target_metrics or {}).items():
        for key, value in target_type_metrics.items():
            metrics[target_type_metric_key(target_type, key)] = value
    for family, family_metrics in (source_family_metrics or {}).items():
        for key, value in family_metrics.items():
            metrics[source_family_metric_key(family, key)] = value
    for group_name, group_values in (case_group_source_metrics or {}).items():
        for group_value, source_values in group_values.items():
            for source, source_metric_values in source_values.items():
                for key, value in source_metric_values.items():
                    metrics[case_group_source_metric_key(group_name, group_value, source, key)] = value
    for group_name, group_values in (case_group_source_family_metrics or {}).items():
        for group_value, family_values in group_values.items():
            for family, family_metric_values in family_values.items():
                for key, value in family_metric_values.items():
                    metrics[
                        case_group_source_family_metric_key(
                            group_name,
                            group_value,
                            family,
                            key,
                        )
                    ] = value
    for strategy, strategy_metrics in (chunk_strategy_metrics or {}).items():
        for key, value in strategy_metrics.items():
            metrics[chunk_strategy_metric_key(strategy, key)] = value
    for role, role_metrics in (retrieval_role_metrics or {}).items():
        for key, value in role_metrics.items():
            metrics[retrieval_role_metric_key(role, key)] = value
    for group_name, group_values in (case_group_metrics or {}).items():
        for group_value, group_metrics in group_values.items():
            for key, value in group_metrics.items():
                metrics[case_group_metric_key(group_name, group_value, key)] = value
    return metrics


def retrieval_expected_target_count_metrics(evaluation: RetrievalEvaluation) -> dict[str, float]:
    target_count = sum(metric.target_count for metric in evaluation.target_metrics.values())
    matched_target_count = sum(
        metric.matched_target_count for metric in evaluation.target_metrics.values()
    )
    if target_count <= 0 and evaluation.results:
        target_count = sum(result.expected_target_count for result in evaluation.results)
        matched_target_count = sum(result.matched_target_count for result in evaluation.results)
    return {
        "expected_target_count": float(target_count),
        "matched_target_count": float(matched_target_count),
    }


def retrieval_rank_metrics(evaluation: RetrievalEvaluation) -> dict[str, float]:
    missing_rank = float(evaluation.top_k + 1)
    expected_results = [result for result in evaluation.results if result.expected_target_count > 0]
    if not expected_results:
        if evaluation.expected_case_count <= 0:
            return {
                "mean_first_relevant_rank": 0.0,
                "p95_first_relevant_rank": 0.0,
                "mean_target_rank": 0.0,
                "p95_target_rank": 0.0,
            } | rank_count_metrics(expected_case_count=0, target_count=0)
        target_count = sum(metric.target_count for metric in evaluation.target_metrics.values())
        target_count = target_count or evaluation.expected_case_count
        return {
            "mean_first_relevant_rank": missing_rank,
            "p95_first_relevant_rank": missing_rank,
            "mean_target_rank": missing_rank,
            "p95_target_rank": missing_rank,
        } | rank_count_metrics(expected_case_count=evaluation.expected_case_count, target_count=int(target_count))

    first_relevant_ranks = [
        float(result.matched_rank) if result.matched_rank is not None else missing_rank
        for result in expected_results
    ]
    target_ranks: list[float] = []
    for result in expected_results:
        for target in result_expected_target_keys(result):
            target_ranks.append(float(result.target_key_matched_ranks.get(target, missing_rank)))
    return {
        "mean_first_relevant_rank": mean(first_relevant_ranks),
        "p95_first_relevant_rank": percentile(first_relevant_ranks, 0.95),
        "mean_target_rank": mean(target_ranks),
        "p95_target_rank": percentile(target_ranks, 0.95),
    } | rank_count_metrics(expected_case_count=len(expected_results), target_count=len(target_ranks))


def rank_count_metrics(expected_case_count: int, target_count: int) -> dict[str, float]:
    return {
        "ranked_expected_case_count": float(expected_case_count),
        "ranked_target_count": float(target_count),
    }


def result_expected_target_keys(result) -> list[str]:
    keys = [f"page:{page}" for page in result.expected_pages]
    keys.extend(f"chunk:{chunk_id}" for chunk_id in result.expected_chunk_ids)
    keys.extend(f"asset:{asset_id}" for asset_id in result.expected_asset_ids)
    keys.extend(f"triple:{triple_id}" for triple_id in result.expected_triple_ids)
    return keys


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def retrieval_target_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, float]]:
    if evaluation is None:
        return {}
    return {
        target_type: {
            "expected_count": float(metric.expected_count),
            "passed_count": float(metric.passed_count),
            "recall_at_k": metric.recall_at_k,
            "mrr": metric.mrr,
            "target_count": float(metric.target_count),
            "matched_target_count": float(metric.matched_target_count),
            "coverage_at_k": metric.coverage_at_k,
            "ndcg_at_k": metric.ndcg_at_k,
        }
        for target_type, metric in sorted(evaluation.target_metrics.items())
    }


def retrieval_source_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, float]]:
    if evaluation is None:
        return {}
    return {
        source.strip().lower(): {
            "query_count": float(metric.query_count),
            "relevant_query_count": float(metric.relevant_query_count),
            "excluded_query_count": float(metric.excluded_query_count),
            "hit_count": float(metric.hit_count),
            "relevant_hit_count": float(metric.relevant_hit_count),
            "excluded_hit_count": float(metric.excluded_hit_count),
            "expected_target_count": float(metric.expected_target_count),
            "matched_target_count": float(metric.matched_target_count),
            "excluded_target_count": float(metric.excluded_target_count),
            "excluded_matched_target_count": float(metric.excluded_matched_target_count),
            "precision_at_hits": metric.precision_at_hits,
            "excluded_precision_at_hits": metric.excluded_precision_at_hits,
            "target_coverage_at_k": metric.target_coverage_at_k,
            "excluded_target_hit_rate": metric.excluded_target_hit_rate,
            "mean_relevant_rank": metric.mean_relevant_rank,
        }
        for source, metric in sorted(evaluation.source_metrics.items())
    }


def retrieval_source_family_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, float]]:
    if evaluation is None:
        return {}
    return {
        family: {
            "query_count": float(metric.query_count),
            "relevant_query_count": float(metric.relevant_query_count),
            "excluded_query_count": float(metric.excluded_query_count),
            "hit_count": float(metric.hit_count),
            "relevant_hit_count": float(metric.relevant_hit_count),
            "excluded_hit_count": float(metric.excluded_hit_count),
            "expected_target_count": float(metric.expected_target_count),
            "matched_target_count": float(metric.matched_target_count),
            "excluded_target_count": float(metric.excluded_target_count),
            "excluded_matched_target_count": float(metric.excluded_matched_target_count),
            "precision_at_hits": metric.precision_at_hits,
            "excluded_precision_at_hits": metric.excluded_precision_at_hits,
            "target_coverage_at_k": metric.target_coverage_at_k,
            "excluded_target_hit_rate": metric.excluded_target_hit_rate,
            "mean_relevant_rank": metric.mean_relevant_rank,
        }
        for family, metric in sorted(evaluation.source_family_metrics.items())
    }


def retrieval_case_group_source_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    if evaluation is None:
        return {}
    return {
        normalize_case_group_key(group_name): {
            normalize_case_group_key(group_value): {
                source.strip().lower(): retrieval_source_metric_payload(metric)
                for source, metric in sorted(source_values.items())
            }
            for group_value, source_values in sorted(group_values.items())
        }
        for group_name, group_values in sorted(
            evaluation.case_group_source_metrics.items()
        )
    }


def retrieval_case_group_source_family_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    if evaluation is None:
        return {}
    return {
        normalize_case_group_key(group_name): {
            normalize_case_group_key(group_value): {
                family.strip().lower(): retrieval_source_metric_payload(metric)
                for family, metric in sorted(family_values.items())
            }
            for group_value, family_values in sorted(group_values.items())
        }
        for group_name, group_values in sorted(
            evaluation.case_group_source_family_metrics.items()
        )
    }


def retrieval_source_metric_payload(metric) -> dict[str, float]:
    return {
        "query_count": float(metric.query_count),
        "relevant_query_count": float(metric.relevant_query_count),
        "excluded_query_count": float(metric.excluded_query_count),
        "hit_count": float(metric.hit_count),
        "relevant_hit_count": float(metric.relevant_hit_count),
        "excluded_hit_count": float(metric.excluded_hit_count),
        "expected_target_count": float(metric.expected_target_count),
        "matched_target_count": float(metric.matched_target_count),
        "excluded_target_count": float(metric.excluded_target_count),
        "excluded_matched_target_count": float(metric.excluded_matched_target_count),
        "precision_at_hits": metric.precision_at_hits,
        "excluded_precision_at_hits": metric.excluded_precision_at_hits,
        "target_coverage_at_k": metric.target_coverage_at_k,
        "excluded_target_hit_rate": metric.excluded_target_hit_rate,
        "mean_relevant_rank": metric.mean_relevant_rank,
    }


def retrieval_chunk_strategy_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, float]]:
    if evaluation is None:
        return {}
    return retrieval_group_metrics(evaluation.chunk_strategy_metrics)


def retrieval_role_metrics_payload(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, float]]:
    if evaluation is None:
        return {}
    return retrieval_group_metrics(evaluation.retrieval_role_metrics)


def retrieval_case_group_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, dict[str, float]]]:
    if evaluation is None:
        return {}
    return {
        group_name: {
            group_value: {
                "case_count": float(metric.case_count),
                "expected_case_count": float(metric.expected_case_count),
                "passed_count": float(metric.passed_count),
                "failed_count": float(metric.failed_count),
                "recall_at_k": metric.recall_at_k,
                "mrr": metric.mrr,
                "target_count": float(metric.target_count),
                "matched_target_count": float(metric.matched_target_count),
                "target_coverage_at_k": metric.target_coverage_at_k,
                "ndcg_at_k": metric.ndcg_at_k,
                "precision_at_k": metric.precision_at_k,
                "mean_latency_ms": metric.mean_latency_ms,
            }
            for group_value, metric in sorted(group_values.items())
        }
        for group_name, group_values in sorted(evaluation.case_group_metrics.items())
    }


def retrieval_group_metrics(metrics_by_group) -> dict[str, dict[str, float]]:
    return {
        group: {
            "query_count": float(metric.query_count),
            "relevant_query_count": float(metric.relevant_query_count),
            "excluded_query_count": float(metric.excluded_query_count),
            "hit_count": float(metric.hit_count),
            "relevant_hit_count": float(metric.relevant_hit_count),
            "excluded_hit_count": float(metric.excluded_hit_count),
            "expected_target_count": float(metric.expected_target_count),
            "matched_target_count": float(metric.matched_target_count),
            "excluded_target_count": float(metric.excluded_target_count),
            "excluded_matched_target_count": float(metric.excluded_matched_target_count),
            "precision_at_hits": metric.precision_at_hits,
            "excluded_precision_at_hits": metric.excluded_precision_at_hits,
            "target_coverage_at_k": metric.target_coverage_at_k,
            "excluded_target_hit_rate": metric.excluded_target_hit_rate,
            "mean_relevant_rank": metric.mean_relevant_rank,
        }
        for group, metric in sorted(metrics_by_group.items())
    }


def target_type_coverage_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[RetrievalGateCheck]:
    checks = []
    for target_type, threshold in sorted(thresholds.items()):
        normalized_target_type = target_type.strip().lower()
        metric = target_type_metric_key(normalized_target_type, "coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_target_type_coverage:{normalized_target_type}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def target_type_metric_key(target_type: str, metric: str) -> str:
    return f"target_type.{target_type}.{metric}"


def source_family_target_coverage_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[RetrievalGateCheck]:
    checks = []
    for family, threshold in sorted(thresholds.items()):
        normalized_family = family.strip().lower()
        metric = source_family_metric_key(normalized_family, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_source_family_target_coverage:{normalized_family}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def source_target_coverage_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[RetrievalGateCheck]:
    checks = []
    for source, threshold in sorted(thresholds.items()):
        normalized_source = source.strip().lower()
        metric = source_metric_key(normalized_source, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_source_target_coverage:{normalized_source}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def source_excluded_target_hit_rate_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[RetrievalGateCheck]:
    checks = []
    for source, threshold in sorted(thresholds.items()):
        normalized_source = source.strip().lower()
        metric = source_metric_key(normalized_source, "excluded_target_hit_rate")
        metrics.setdefault(metric, 0.0)
        checks.append(
            maximum_check(
                f"max_source_excluded_target_hit_rate:{normalized_source}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def source_family_excluded_target_hit_rate_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[RetrievalGateCheck]:
    checks = []
    for family, threshold in sorted(thresholds.items()):
        normalized_family = family.strip().lower()
        metric = source_family_metric_key(normalized_family, "excluded_target_hit_rate")
        metrics.setdefault(metric, 0.0)
        checks.append(
            maximum_check(
                f"max_source_family_excluded_target_hit_rate:{normalized_family}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def source_metric_key(source: str, metric: str) -> str:
    return f"source.{source}.{metric}"


def source_family_metric_key(family: str, metric: str) -> str:
    return f"source_family.{family}.{metric}"


def grouped_target_coverage_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
    metric_key_fn,
    check_prefix: str,
) -> list[RetrievalGateCheck]:
    checks = []
    for group, threshold in sorted(thresholds.items()):
        normalized_group = group.strip().lower()
        metric = metric_key_fn(normalized_group, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"{check_prefix}:{normalized_group}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def grouped_excluded_target_hit_rate_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
    metric_key_fn,
    check_prefix: str,
) -> list[RetrievalGateCheck]:
    checks = []
    for group, threshold in sorted(thresholds.items()):
        normalized_group = group.strip().lower()
        metric = metric_key_fn(normalized_group, "excluded_target_hit_rate")
        metrics.setdefault(metric, 0.0)
        checks.append(
            maximum_check(
                f"{check_prefix}:{normalized_group}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def case_group_target_coverage_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[RetrievalGateCheck]:
    checks = []
    for group_spec, threshold in sorted(thresholds.items()):
        group_name, group_value = parse_case_group_spec(group_spec)
        metric = case_group_metric_key(group_name, group_value, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        checks.append(
            minimum_check(
                f"min_case_group_target_coverage:{group_name}:{group_value}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def case_group_source_target_coverage_checks(
    metrics: dict[str, float],
    thresholds: dict[str, float],
    family: bool = False,
) -> list[RetrievalGateCheck]:
    checks = []
    for spec, threshold in sorted(thresholds.items()):
        group_name, group_value, source = parse_case_group_source_spec(spec)
        metric_key_fn = (
            case_group_source_family_metric_key
            if family
            else case_group_source_metric_key
        )
        metric = metric_key_fn(group_name, group_value, source, "target_coverage_at_k")
        metrics.setdefault(metric, 0.0)
        check_prefix = (
            "min_case_group_source_family_target_coverage"
            if family
            else "min_case_group_source_target_coverage"
        )
        checks.append(
            minimum_check(
                f"{check_prefix}:{group_name}:{group_value}:{source}",
                metric,
                metrics,
                threshold,
            )
        )
    return checks


def parse_case_group_spec(value: str) -> tuple[str, str]:
    if ":" in value:
        group_name, group_value = value.split(":", 1)
    else:
        group_name, group_value = "case_source", value
    return normalize_case_group_key(group_name), normalize_case_group_key(group_value)


def parse_case_group_source_spec(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) < 3:
        return "case_source", normalize_case_group_key(parts[0]), ":".join(parts[1:])
    group_name, group_value, *source_parts = parts
    source = ":".join(source_parts).strip().lower()
    return (
        normalize_case_group_key(group_name),
        normalize_case_group_key(group_value),
        source or "unspecified",
    )


def normalize_case_group_key(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "_")
    return normalized or "unspecified"


def chunk_strategy_metric_key(strategy: str, metric: str) -> str:
    return f"chunk_strategy.{strategy}.{metric}"


def retrieval_role_metric_key(role: str, metric: str) -> str:
    return f"retrieval_role.{role}.{metric}"


def case_group_metric_key(group_name: str, group_value: str, metric: str) -> str:
    return f"case_group.{group_name}.{group_value}.{metric}"


def case_group_source_metric_key(
    group_name: str,
    group_value: str,
    source: str,
    metric: str,
) -> str:
    return f"case_group_source.{group_name}.{group_value}.{source}.{metric}"


def case_group_source_family_metric_key(
    group_name: str,
    group_value: str,
    family: str,
    metric: str,
) -> str:
    return f"case_group_source_family.{group_name}.{group_value}.{family}.{metric}"


def minimum_check(
    name: str,
    metric: str,
    metrics: dict[str, float],
    threshold: float,
) -> RetrievalGateCheck:
    actual = metrics[metric]
    return RetrievalGateCheck(
        name=name,
        metric=metric,
        operator=">=",
        actual=actual,
        threshold=threshold,
        passed=actual >= threshold,
    )


def maximum_check(
    name: str,
    metric: str,
    metrics: dict[str, float],
    threshold: float,
) -> RetrievalGateCheck:
    actual = metrics[metric]
    return RetrievalGateCheck(
        name=name,
        metric=metric,
        operator="<=",
        actual=actual,
        threshold=threshold,
        passed=actual <= threshold,
    )


def baseline_drop_checks(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: dict[str, float | None],
) -> list[RetrievalGateCheck]:
    checks = []
    for metric, threshold in thresholds.items():
        if threshold is None:
            continue
        actual = metrics[metric]
        baseline = baseline_metrics[metric]
        delta = actual - baseline
        checks.append(
            RetrievalGateCheck(
                name=f"max_{metric}_drop",
                metric=metric,
                operator="baseline_drop<=",
                actual=actual,
                baseline=baseline,
                delta=delta,
                threshold=threshold,
                passed=(baseline - actual) <= threshold,
            )
        )
    return checks


def latency_ratio_checks(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: dict[str, float | None],
) -> list[RetrievalGateCheck]:
    checks = []
    for metric, threshold in thresholds.items():
        if threshold is None:
            continue
        actual = metrics[metric]
        baseline = baseline_metrics[metric]
        ratio = safe_ratio(actual, baseline)
        passed = actual <= 0.0 if ratio is None else ratio <= threshold
        checks.append(
            RetrievalGateCheck(
                name=f"max_{metric}_ratio",
                metric=metric,
                operator="actual/baseline<=",
                actual=actual,
                baseline=baseline,
                delta=actual - baseline,
                ratio=ratio,
                threshold=threshold,
                passed=passed,
            )
        )
    return checks


def safe_ratio(actual: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return actual / baseline


def gate_summary_payload(report: RetrievalGateReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "failed_checks": report.failed_checks,
        "metrics": report.metrics,
        "target_metrics": report.target_metrics,
        "source_metrics": report.source_metrics,
        "source_family_metrics": report.source_family_metrics,
        "case_group_source_metrics": report.case_group_source_metrics,
        "case_group_source_family_metrics": report.case_group_source_family_metrics,
        "chunk_strategy_metrics": report.chunk_strategy_metrics,
        "retrieval_role_metrics": report.retrieval_role_metrics,
        "case_group_metrics": report.case_group_metrics,
        "baseline_metrics": report.baseline_metrics,
    }

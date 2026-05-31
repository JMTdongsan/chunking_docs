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
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[RetrievalGateCheck] = Field(default_factory=list)


def gate_retrieval_evaluation(
    evaluation: RetrievalEvaluation,
    baseline: RetrievalEvaluation | None = None,
    min_recall_at_k: float = 0.0,
    min_target_coverage_at_k: float = 0.0,
    min_target_ndcg_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_precision_at_k: float = 0.0,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    max_recall_drop: float | None = None,
    max_target_coverage_drop: float | None = None,
    max_target_ndcg_drop: float | None = None,
    max_precision_drop: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
) -> RetrievalGateReport:
    """Evaluate retrieval metrics against absolute floors and baseline regression limits."""

    source_family_metrics = retrieval_source_family_metrics(evaluation)
    target_metrics = retrieval_target_metrics(evaluation)
    metrics = retrieval_metrics(evaluation, source_family_metrics, target_metrics)
    baseline_metrics = (
        retrieval_metrics(
            baseline,
            retrieval_source_family_metrics(baseline),
            retrieval_target_metrics(baseline),
        )
        if baseline is not None
        else {}
    )
    checks = [
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
    ]
    if max_mean_latency_ms is not None:
        checks.append(maximum_check("max_mean_latency_ms", "mean_latency_ms", metrics, max_mean_latency_ms))
    if max_p95_latency_ms is not None:
        checks.append(maximum_check("max_p95_latency_ms", "p95_latency_ms", metrics, max_p95_latency_ms))
    checks.extend(
        target_type_coverage_checks(
            metrics,
            min_target_type_coverage or {},
        )
    )
    checks.extend(
        source_family_target_coverage_checks(
            metrics,
            min_source_family_target_coverage or {},
        )
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
        target_metrics=target_metrics,
        source_family_metrics=source_family_metrics,
        baseline_metrics=baseline_metrics,
        failed_checks=failed_checks,
        checks=checks,
    )


def retrieval_metrics(
    evaluation: RetrievalEvaluation | None,
    source_family_metrics: dict[str, dict[str, float]] | None = None,
    target_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    if evaluation is None:
        return {}
    metrics = {
        "hit_rate": evaluation.hit_rate,
        "recall_at_k": evaluation.recall_at_k,
        "mrr": evaluation.mrr,
        "target_coverage_at_k": evaluation.target_coverage_at_k,
        "mean_target_ndcg_at_k": evaluation.mean_target_ndcg_at_k,
        "mean_precision_at_k": evaluation.mean_precision_at_k,
        "mean_latency_ms": evaluation.mean_latency_ms,
        "p95_latency_ms": evaluation.p95_latency_ms,
    }
    for target_type, target_type_metrics in (target_metrics or {}).items():
        for key, value in target_type_metrics.items():
            metrics[target_type_metric_key(target_type, key)] = value
    for family, family_metrics in (source_family_metrics or {}).items():
        for key, value in family_metrics.items():
            metrics[source_family_metric_key(family, key)] = value
    return metrics


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


def retrieval_source_family_metrics(
    evaluation: RetrievalEvaluation | None,
) -> dict[str, dict[str, float]]:
    if evaluation is None:
        return {}
    return {
        family: {
            "query_count": float(metric.query_count),
            "relevant_query_count": float(metric.relevant_query_count),
            "hit_count": float(metric.hit_count),
            "relevant_hit_count": float(metric.relevant_hit_count),
            "expected_target_count": float(metric.expected_target_count),
            "matched_target_count": float(metric.matched_target_count),
            "precision_at_hits": metric.precision_at_hits,
            "target_coverage_at_k": metric.target_coverage_at_k,
            "mean_relevant_rank": metric.mean_relevant_rank,
        }
        for family, metric in sorted(evaluation.source_family_metrics.items())
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


def source_family_metric_key(family: str, metric: str) -> str:
    return f"source_family.{family}.{metric}"


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
        "source_family_metrics": report.source_family_metrics,
        "baseline_metrics": report.baseline_metrics,
    }

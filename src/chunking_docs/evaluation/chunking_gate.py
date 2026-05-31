from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from chunking_docs.evaluation.compare import ChunkingComparison, ChunkingComparisonRow
from chunking_docs.evaluation.gate import source_family_metric_key, target_type_metric_key


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
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_metrics: dict[str, float | None] = Field(default_factory=dict)
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
    min_recall_at_k: float | None = None,
    min_target_coverage_at_k: float | None = None,
    min_target_ndcg_at_k: float | None = None,
    min_mrr: float | None = None,
    min_precision_at_k: float | None = None,
    max_mean_latency_ms: float | None = None,
    max_p95_latency_ms: float | None = None,
    max_failed_queries: int | None = 0,
    max_chunks_under_min_chars: int | None = None,
    max_chunks_over_max_chars: int | None = None,
    min_target_type_coverage: dict[str, float] | None = None,
    min_source_family_target_coverage: dict[str, float] | None = None,
    max_quality_drop: float | None = None,
    max_recall_drop: float | None = None,
    max_target_coverage_drop: float | None = None,
    max_target_ndcg_drop: float | None = None,
    max_precision_drop: float | None = None,
    max_mean_latency_ratio: float | None = None,
    max_p95_latency_ratio: float | None = None,
) -> ChunkingComparisonGateReport:
    selected_name = select_candidate_name(comparison, candidate)
    selected_row = find_row(comparison, selected_name)
    metrics = row_metrics(selected_row)
    baseline_metrics: dict[str, float | None] = {}
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
            optional_minimum_check(
                "min_visual_annotation_ratio",
                selected_name,
                "visual_annotation_ratio",
                metrics,
                min_visual_annotation_ratio,
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
            optional_maximum_check(
                "max_failed_queries",
                selected_name,
                "failed_query_count",
                metrics,
                float(max_failed_queries) if max_failed_queries is not None else None,
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
        source_family_target_coverage_checks(
            selected_name,
            metrics,
            min_source_family_target_coverage or {},
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

    failed_checks = [check.name for check in checks if not check.passed]
    return ChunkingComparisonGateReport(
        passed=not failed_checks,
        candidate=selected_name,
        baseline_candidate=baseline_candidate,
        metrics=metrics,
        target_metrics=selected_row.target_metrics,
        source_family_metrics=selected_row.source_family_metrics,
        baseline_metrics=baseline_metrics,
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
        "quality_score": row.quality_score,
        "retrieval_hit_rate": row.retrieval_hit_rate,
        "retrieval_recall_at_k": row.retrieval_recall_at_k,
        "retrieval_mrr": row.retrieval_mrr,
        "retrieval_target_coverage_at_k": row.retrieval_target_coverage_at_k,
        "retrieval_mean_target_ndcg_at_k": row.retrieval_mean_target_ndcg_at_k,
        "retrieval_mean_precision_at_k": row.retrieval_mean_precision_at_k,
        "retrieval_mean_latency_ms": row.retrieval_mean_latency_ms,
        "retrieval_p95_latency_ms": row.retrieval_p95_latency_ms,
        "failed_query_count": float(len(row.failed_queries)),
        "page_coverage_ratio": row.page_coverage_ratio,
        "visual_annotation_ratio": row.visual_annotation_ratio,
        "chunks_under_min_chars": float(row.chunks_under_min_chars),
        "chunks_over_max_chars": float(row.chunks_over_max_chars),
    }
    for target_type, target_type_metrics in row.target_metrics.items():
        for key, value in target_type_metrics.items():
            metrics[target_type_metric_key(target_type, key)] = value
    for family, family_metrics in row.source_family_metrics.items():
        for key, value in family_metrics.items():
            metrics[source_family_metric_key(family, key)] = value
    return metrics


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
        "source_family_metrics": report.source_family_metrics,
        "baseline_metrics": report.baseline_metrics,
    }

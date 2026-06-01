from __future__ import annotations

from pydantic import BaseModel, Field

from chunking_docs.vision.jobs import VisualJobRunResult
from chunking_docs.vision.quality import VisualQualityReport, evaluate_visual_results


class VisualRunComparisonRow(BaseModel):
    name: str
    result_count: int
    completed_count: int
    failed_count: int
    skipped_count: int
    annotation_count: int
    completion_rate: float
    annotation_rate: float
    ocr_job_count: int
    ocr_text_coverage: float | None = None
    mean_ocr_text_chars: float | None = None
    vlm_job_count: int
    vlm_summary_coverage: float | None = None
    mean_vlm_summary_chars: float | None = None
    vlm_json_parse_rate: float | None = None
    vlm_object_count: int = 0
    vlm_object_coverage: float | None = None
    objects_per_vlm_job: float | None = None
    object_bbox_coverage: float | None = None
    triple_count: int
    triples_per_vlm_job: float | None = None
    total_mean_latency_ms: float | None = None
    total_p95_latency_ms: float | None = None
    ocr_mean_latency_ms: float | None = None
    ocr_p95_latency_ms: float | None = None
    vlm_mean_latency_ms: float | None = None
    vlm_p95_latency_ms: float | None = None
    parse_status_counts: dict[str, int] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    quality_score: float


class VisualRunComparison(BaseModel):
    rows: list[VisualRunComparisonRow]
    best_by_quality: str | None = None
    fastest_by_total_latency: str | None = None
    best_by_triple_density: str | None = None
    union_job_count: int = 0
    shared_job_count: int = 0
    job_set_mismatch: bool = False
    run_job_counts: dict[str, int] = Field(default_factory=dict)
    missing_job_ids_by_run: dict[str, list[str]] = Field(default_factory=dict)
    unshared_job_ids_by_run: dict[str, list[str]] = Field(default_factory=dict)


def compare_visual_runs(
    runs: dict[str, list[VisualJobRunResult]],
) -> VisualRunComparison:
    rows = [visual_run_row(name, results) for name, results in runs.items()]
    job_sets = visual_run_job_sets(runs)
    job_set_report = compare_visual_job_sets(job_sets)
    rows.sort(
        key=lambda row: (
            row.quality_score,
            row.completion_rate,
            row.vlm_summary_coverage if row.vlm_summary_coverage is not None else -1.0,
            row.vlm_json_parse_rate if row.vlm_json_parse_rate is not None else -1.0,
            row.vlm_object_coverage if row.vlm_object_coverage is not None else -1.0,
            row.triples_per_vlm_job if row.triples_per_vlm_job is not None else -1.0,
            -(row.total_mean_latency_ms or 0.0),
        ),
        reverse=True,
    )
    latency_rows = [row for row in rows if row.total_mean_latency_ms is not None]
    triple_rows = [row for row in rows if row.triples_per_vlm_job is not None]
    return VisualRunComparison(
        rows=rows,
        best_by_quality=max(rows, key=lambda row: row.quality_score).name if rows else None,
        fastest_by_total_latency=min(latency_rows, key=lambda row: row.total_mean_latency_ms or 0.0).name
        if latency_rows
        else None,
        best_by_triple_density=max(triple_rows, key=lambda row: row.triples_per_vlm_job or 0.0).name
        if triple_rows
        else None,
        **job_set_report,
    )


def visual_run_row(name: str, results: list[VisualJobRunResult]) -> VisualRunComparisonRow:
    report = evaluate_visual_results(results)
    total_durations = durations(results, "total_duration_ms")
    ocr_durations = durations(results, "ocr_duration_ms")
    vlm_durations = durations(results, "vlm_duration_ms")
    return VisualRunComparisonRow(
        name=name,
        result_count=report.result_count,
        completed_count=report.completed_count,
        failed_count=report.failed_count,
        skipped_count=report.skipped_count,
        annotation_count=report.annotation_count,
        completion_rate=report.completion_rate,
        annotation_rate=report.annotation_rate,
        ocr_job_count=report.ocr_job_count,
        ocr_text_coverage=report.ocr_text_coverage if report.ocr_job_count else None,
        mean_ocr_text_chars=report.mean_ocr_text_chars if report.ocr_job_count else None,
        vlm_job_count=report.vlm_job_count,
        vlm_summary_coverage=report.vlm_summary_coverage if report.vlm_job_count else None,
        mean_vlm_summary_chars=report.mean_vlm_summary_chars if report.vlm_job_count else None,
        vlm_json_parse_rate=report.vlm_json_parse_rate if report.vlm_job_count else None,
        vlm_object_count=report.vlm_object_count,
        vlm_object_coverage=report.vlm_object_coverage if report.vlm_job_count else None,
        objects_per_vlm_job=report.objects_per_vlm_job if report.vlm_job_count else None,
        object_bbox_coverage=report.object_bbox_coverage if report.vlm_object_count else None,
        triple_count=report.triple_count,
        triples_per_vlm_job=report.triples_per_vlm_job if report.vlm_job_count else None,
        total_mean_latency_ms=mean(total_durations),
        total_p95_latency_ms=percentile(sorted(total_durations), 0.95) if total_durations else None,
        ocr_mean_latency_ms=mean(ocr_durations),
        ocr_p95_latency_ms=percentile(sorted(ocr_durations), 0.95) if ocr_durations else None,
        vlm_mean_latency_ms=mean(vlm_durations),
        vlm_p95_latency_ms=percentile(sorted(vlm_durations), 0.95) if vlm_durations else None,
        parse_status_counts=report.parse_status_counts,
        failed_checks=report.failed_checks,
        quality_score=visual_run_quality_score(report),
    )


def visual_run_quality_score(report: VisualQualityReport) -> float:
    components = [
        (report.completion_rate, 0.25),
        (report.annotation_rate, 0.20),
    ]
    if report.ocr_job_count:
        components.extend(
            [
                (report.ocr_text_coverage, 0.15),
                (min(report.mean_ocr_text_chars / 120.0, 1.0), 0.05),
            ]
        )
    if report.vlm_job_count:
        components.extend(
            [
                (report.vlm_summary_coverage, 0.20),
                (report.vlm_json_parse_rate, 0.15),
                (report.vlm_object_coverage, 0.05),
                (min(report.triples_per_vlm_job, 1.0), 0.10),
                (min(report.mean_vlm_summary_chars / 240.0, 1.0), 0.05),
            ]
        )
    total_weight = sum(weight for _, weight in components)
    return sum(value * weight for value, weight in components) / total_weight if total_weight else 0.0


def visual_run_job_sets(
    runs: dict[str, list[VisualJobRunResult]],
) -> dict[str, set[str]]:
    return {
        name: {str(result.job_id).strip() for result in results if str(result.job_id).strip()}
        for name, results in runs.items()
    }


def compare_visual_job_sets(job_sets: dict[str, set[str]]) -> dict:
    if not job_sets:
        return {
            "union_job_count": 0,
            "shared_job_count": 0,
            "job_set_mismatch": False,
            "run_job_counts": {},
            "missing_job_ids_by_run": {},
            "unshared_job_ids_by_run": {},
        }

    union_ids = set().union(*job_sets.values())
    shared_ids = set.intersection(*job_sets.values()) if job_sets else set()
    return {
        "union_job_count": len(union_ids),
        "shared_job_count": len(shared_ids),
        "job_set_mismatch": any(ids != shared_ids for ids in job_sets.values()),
        "run_job_counts": {name: len(ids) for name, ids in sorted(job_sets.items())},
        "missing_job_ids_by_run": {
            name: sorted(union_ids - ids) for name, ids in sorted(job_sets.items())
        },
        "unshared_job_ids_by_run": {
            name: sorted(ids - shared_ids) for name, ids in sorted(job_sets.items())
        },
    }


def durations(results: list[VisualJobRunResult], key: str) -> list[float]:
    values = []
    for result in results:
        value = result.metadata.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

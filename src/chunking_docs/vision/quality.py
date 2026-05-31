from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.vision.jobs import VisualJobRunResult

STRUCTURED_VLM_PARSE_STATUSES = {"json_object", "json_list", "json_repaired"}


class VisualQualityCheck(BaseModel):
    name: str
    metric: str
    operator: str
    actual: float
    threshold: float
    passed: bool


class VisualQualityIssue(BaseModel):
    severity: str
    code: str
    message: str
    job_id: str | None = None
    asset_id: str | None = None
    page_no: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualQualityReport(BaseModel):
    passed: bool
    result_count: int
    completed_count: int
    failed_count: int
    skipped_count: int
    annotation_count: int
    completion_rate: float = 0.0
    annotation_rate: float = 0.0
    ocr_job_count: int = 0
    nonempty_ocr_count: int = 0
    ocr_text_coverage: float = 0.0
    mean_ocr_text_chars: float = 0.0
    vlm_job_count: int = 0
    nonempty_vlm_summary_count: int = 0
    vlm_summary_coverage: float = 0.0
    mean_vlm_summary_chars: float = 0.0
    vlm_json_parse_count: int = 0
    vlm_json_parse_rate: float = 0.0
    triple_count: int = 0
    triples_per_vlm_job: float = 0.0
    parse_status_counts: dict[str, int] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[VisualQualityCheck] = Field(default_factory=list)
    issues: list[VisualQualityIssue] = Field(default_factory=list)


def evaluate_visual_results(
    results: list[VisualJobRunResult],
    min_completion_rate: float = 0.0,
    min_annotation_rate: float = 0.0,
    min_ocr_text_coverage: float = 0.0,
    min_vlm_summary_coverage: float = 0.0,
    min_vlm_json_parse_rate: float = 0.0,
    min_triples_per_vlm_job: float = 0.0,
    min_mean_ocr_text_chars: float = 0.0,
    min_mean_vlm_summary_chars: float = 0.0,
    max_failed_count: int | None = None,
    max_skipped_count: int | None = None,
    max_issues: int = 200,
) -> VisualQualityReport:
    completed_count = sum(1 for result in results if result.status == "completed")
    failed_count = sum(1 for result in results if result.status == "failed")
    skipped_count = sum(1 for result in results if result.status == "skipped")
    annotation_count = sum(1 for result in results if result.annotation is not None)
    result_count = len(results)

    ocr_chars: list[int] = []
    vlm_summary_chars: list[int] = []
    parse_status_counts: dict[str, int] = {}
    ocr_job_count = 0
    vlm_job_count = 0
    vlm_json_parse_count = 0
    triple_count = 0
    issues: list[VisualQualityIssue] = []

    for result in results:
        operations = result_operations(result)
        if result.status == "failed":
            append_issue(
                issues,
                max_issues,
                result_issue(result, "error", "visual_job_failed", result.error or "Visual job failed."),
            )
        elif result.status == "skipped":
            append_issue(
                issues,
                max_issues,
                result_issue(result, "warning", "visual_job_skipped", result.error or "Visual job skipped."),
            )

        if "ocr" in operations:
            ocr_job_count += 1
            chars = ocr_text_chars(result)
            ocr_chars.append(chars)
            if result.status == "completed" and chars <= 0:
                append_issue(
                    issues,
                    max_issues,
                    result_issue(result, "warning", "empty_ocr_text", "OCR job produced no usable text."),
                )

        if "vlm" in operations:
            vlm_job_count += 1
            chars = vlm_summary_chars_count(result)
            vlm_summary_chars.append(chars)
            if result.status == "completed" and chars <= 0:
                append_issue(
                    issues,
                    max_issues,
                    result_issue(result, "warning", "empty_vlm_summary", "VLM job produced no usable summary."),
                )
            parse_status = vlm_parse_status(result)
            if parse_status:
                parse_status_counts[parse_status] = parse_status_counts.get(parse_status, 0) + 1
            if parse_status in STRUCTURED_VLM_PARSE_STATUSES:
                vlm_json_parse_count += 1

        if result.annotation is not None:
            triple_count += len(result.annotation.triples)

    metrics = {
        "completion_rate": completed_count / result_count if result_count else 0.0,
        "annotation_rate": annotation_count / result_count if result_count else 0.0,
        "ocr_text_coverage": nonempty_count(ocr_chars) / ocr_job_count if ocr_job_count else 0.0,
        "mean_ocr_text_chars": mean(ocr_chars),
        "vlm_summary_coverage": nonempty_count(vlm_summary_chars) / vlm_job_count if vlm_job_count else 0.0,
        "mean_vlm_summary_chars": mean(vlm_summary_chars),
        "vlm_json_parse_rate": vlm_json_parse_count / vlm_job_count if vlm_job_count else 0.0,
        "triples_per_vlm_job": triple_count / vlm_job_count if vlm_job_count else 0.0,
        "failed_count": float(failed_count),
        "skipped_count": float(skipped_count),
    }
    checks = [
        min_check("min_completion_rate", "completion_rate", metrics, min_completion_rate),
        min_check("min_annotation_rate", "annotation_rate", metrics, min_annotation_rate),
        min_check("min_ocr_text_coverage", "ocr_text_coverage", metrics, min_ocr_text_coverage),
        min_check(
            "min_vlm_summary_coverage",
            "vlm_summary_coverage",
            metrics,
            min_vlm_summary_coverage,
        ),
        min_check("min_vlm_json_parse_rate", "vlm_json_parse_rate", metrics, min_vlm_json_parse_rate),
        min_check("min_triples_per_vlm_job", "triples_per_vlm_job", metrics, min_triples_per_vlm_job),
        min_check("min_mean_ocr_text_chars", "mean_ocr_text_chars", metrics, min_mean_ocr_text_chars),
        min_check(
            "min_mean_vlm_summary_chars",
            "mean_vlm_summary_chars",
            metrics,
            min_mean_vlm_summary_chars,
        ),
    ]
    if max_failed_count is not None:
        checks.append(max_check("max_failed_count", "failed_count", metrics, float(max_failed_count)))
    if max_skipped_count is not None:
        checks.append(max_check("max_skipped_count", "skipped_count", metrics, float(max_skipped_count)))

    failed_checks = [check.name for check in checks if not check.passed]
    return VisualQualityReport(
        passed=not failed_checks and not any(issue.severity == "error" for issue in issues),
        result_count=result_count,
        completed_count=completed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        annotation_count=annotation_count,
        completion_rate=metrics["completion_rate"],
        annotation_rate=metrics["annotation_rate"],
        ocr_job_count=ocr_job_count,
        nonempty_ocr_count=nonempty_count(ocr_chars),
        ocr_text_coverage=metrics["ocr_text_coverage"],
        mean_ocr_text_chars=metrics["mean_ocr_text_chars"],
        vlm_job_count=vlm_job_count,
        nonempty_vlm_summary_count=nonempty_count(vlm_summary_chars),
        vlm_summary_coverage=metrics["vlm_summary_coverage"],
        mean_vlm_summary_chars=metrics["mean_vlm_summary_chars"],
        vlm_json_parse_count=vlm_json_parse_count,
        vlm_json_parse_rate=metrics["vlm_json_parse_rate"],
        triple_count=triple_count,
        triples_per_vlm_job=metrics["triples_per_vlm_job"],
        parse_status_counts=dict(sorted(parse_status_counts.items())),
        failed_checks=failed_checks,
        checks=checks,
        issues=issues,
    )


def result_operations(result: VisualJobRunResult) -> list[str]:
    operations = result.metadata.get("operations", [])
    if isinstance(operations, str):
        operations = [operations]
    return [str(operation) for operation in operations]


def ocr_text_chars(result: VisualJobRunResult) -> int:
    value = numeric_metadata(result.metadata, "ocr_text_chars")
    if value is not None:
        return int(value)
    if result.annotation and result.annotation.ocr_text:
        return len(result.annotation.ocr_text.strip())
    return 0


def vlm_summary_chars_count(result: VisualJobRunResult) -> int:
    if result.annotation and result.annotation.vlm_summary:
        return len(result.annotation.vlm_summary.strip())
    return 0


def vlm_parse_status(result: VisualJobRunResult) -> str:
    if result.annotation:
        value = result.annotation.metadata.get("vlm_parse_status")
        if value:
            return str(value)
    value = result.metadata.get("vlm_parse_status")
    return str(value) if value else ""


def numeric_metadata(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nonempty_count(values: list[int]) -> int:
    return sum(1 for value in values if value > 0)


def mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def min_check(
    name: str,
    metric: str,
    metrics: dict[str, float],
    threshold: float,
) -> VisualQualityCheck:
    actual = metrics[metric]
    return VisualQualityCheck(
        name=name,
        metric=metric,
        operator=">=",
        actual=actual,
        threshold=threshold,
        passed=actual >= threshold,
    )


def max_check(
    name: str,
    metric: str,
    metrics: dict[str, float],
    threshold: float,
) -> VisualQualityCheck:
    actual = metrics[metric]
    return VisualQualityCheck(
        name=name,
        metric=metric,
        operator="<=",
        actual=actual,
        threshold=threshold,
        passed=actual <= threshold,
    )


def result_issue(
    result: VisualJobRunResult,
    severity: str,
    code: str,
    message: str,
) -> VisualQualityIssue:
    return VisualQualityIssue(
        severity=severity,
        code=code,
        message=message,
        job_id=result.job_id,
        asset_id=result.asset_id,
        page_no=result.page_no,
        metadata={"status": result.status, "operations": result_operations(result)},
    )


def append_issue(
    issues: list[VisualQualityIssue],
    max_issues: int,
    issue: VisualQualityIssue,
) -> None:
    if len(issues) < max_issues:
        issues.append(issue)

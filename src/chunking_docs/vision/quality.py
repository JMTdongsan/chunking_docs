from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.models import VisualAsset
from chunking_docs.vision.jobs import VisualJobRunResult
from chunking_docs.vision.manual_annotations import AssetAnnotation

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
    vlm_object_count: int = 0
    vlm_object_job_count: int = 0
    vlm_object_coverage: float = 0.0
    objects_per_vlm_job: float = 0.0
    object_bbox_count: int = 0
    object_bbox_coverage: float = 0.0
    triple_count: int = 0
    triples_per_vlm_job: float = 0.0
    parse_status_counts: dict[str, int] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[VisualQualityCheck] = Field(default_factory=list)
    issues: list[VisualQualityIssue] = Field(default_factory=list)


def visual_results_from_assets(
    assets: list[VisualAsset],
    required_only: bool = True,
) -> list[VisualJobRunResult]:
    results = []
    for asset in assets:
        operations = visual_asset_operations(asset, required_only=required_only)
        if not operations:
            continue
        results.append(
            VisualJobRunResult(
                job_id=f"asset-state:{asset.asset_id}",
                asset_id=asset.asset_id,
                page_no=asset.page_no,
                status="completed",
                annotation=AssetAnnotation(
                    asset_id=asset.asset_id,
                    page_no=asset.page_no,
                    kind=asset.kind,
                    caption=asset.caption,
                    ocr_text=asset.ocr_text,
                    vlm_summary=asset.vlm_summary,
                    metadata=dict(asset.metadata),
                ),
                metadata=visual_asset_result_metadata(asset, operations),
            )
        )
    return results


def visual_asset_operations(asset: VisualAsset, required_only: bool = True) -> list[str]:
    operations = []
    if asset.metadata.get("requires_ocr") or (not required_only and asset.ocr_text is not None):
        operations.append("ocr")
    if asset.metadata.get("requires_vlm") or (not required_only and asset.vlm_summary is not None):
        operations.append("vlm")
    return operations


def visual_asset_result_metadata(asset: VisualAsset, operations: list[str]) -> dict[str, Any]:
    metadata = {
        "operations": operations,
        "ocr_backend": asset.metadata.get("ocr_backend"),
        "vlm_backend": asset.metadata.get("vlm_backend"),
        "ocr_text_chars": asset.metadata.get("ocr_text_chars", len((asset.ocr_text or "").strip())),
        "vlm_output_chars": asset.metadata.get("vlm_output_chars", len((asset.vlm_summary or "").strip())),
        "vlm_parse_status": asset.metadata.get("vlm_parse_status"),
        "object_count": asset.metadata.get("object_count"),
        "object_bbox_count": asset.metadata.get("object_bbox_count"),
    }
    for key in [
        "ocr_language",
        "vlm_prompt_name",
        "vlm_prompt_schema_version",
        "vlm_prompt_sha256",
        "vlm_prompt_chars",
        "ocr_duration_ms",
        "vlm_duration_ms",
        "total_duration_ms",
    ]:
        if key in asset.metadata:
            metadata[key] = asset.metadata[key]
    return metadata


def evaluate_visual_results(
    results: list[VisualJobRunResult],
    min_completion_rate: float = 0.0,
    min_annotation_rate: float = 0.0,
    min_ocr_text_coverage: float = 0.0,
    min_vlm_summary_coverage: float = 0.0,
    min_vlm_json_parse_rate: float = 0.0,
    min_vlm_object_coverage: float = 0.0,
    min_objects_per_vlm_job: float = 0.0,
    min_object_bbox_coverage: float = 0.0,
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
    object_count = 0
    object_job_count = 0
    object_bbox_count = 0
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
            result_object_count = vlm_object_count(result)
            result_object_bbox_count = vlm_object_bbox_count(result)
            object_count += result_object_count
            object_bbox_count += result_object_bbox_count
            if result_object_count > 0:
                object_job_count += 1

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
        "vlm_object_coverage": object_job_count / vlm_job_count if vlm_job_count else 0.0,
        "objects_per_vlm_job": object_count / vlm_job_count if vlm_job_count else 0.0,
        "object_bbox_coverage": object_bbox_count / object_count if object_count else 0.0,
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
        min_check("min_vlm_object_coverage", "vlm_object_coverage", metrics, min_vlm_object_coverage),
        min_check("min_objects_per_vlm_job", "objects_per_vlm_job", metrics, min_objects_per_vlm_job),
        min_check("min_object_bbox_coverage", "object_bbox_coverage", metrics, min_object_bbox_coverage),
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
        vlm_object_count=object_count,
        vlm_object_job_count=object_job_count,
        vlm_object_coverage=metrics["vlm_object_coverage"],
        objects_per_vlm_job=metrics["objects_per_vlm_job"],
        object_bbox_count=object_bbox_count,
        object_bbox_coverage=metrics["object_bbox_coverage"],
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


def vlm_object_count(result: VisualJobRunResult) -> int:
    value = object_count_from_metadata(result.annotation.metadata if result.annotation else {})
    if value is not None:
        return value
    value = object_count_from_metadata(result.metadata)
    return value or 0


def vlm_object_bbox_count(result: VisualJobRunResult) -> int:
    value = numeric_metadata(result.annotation.metadata if result.annotation else {}, "object_bbox_count")
    if value is not None:
        return max(0, int(value))
    value = numeric_metadata(result.metadata, "object_bbox_count")
    if value is not None:
        return max(0, int(value))
    objects = metadata_objects(result.annotation.metadata if result.annotation else {})
    if not objects:
        objects = metadata_objects(result.metadata)
    return sum(1 for item in objects if isinstance(item, dict) and item.get("bbox"))


def object_count_from_metadata(metadata: dict[str, Any]) -> int | None:
    value = numeric_metadata(metadata, "object_count")
    if value is not None:
        return max(0, int(value))
    objects = metadata_objects(metadata)
    if objects:
        return len(objects)
    return None


def metadata_objects(metadata: dict[str, Any]) -> list[Any]:
    objects = metadata.get("objects") or metadata.get("detected_objects") or metadata.get("visual_objects")
    if isinstance(objects, list):
        return objects
    return []


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

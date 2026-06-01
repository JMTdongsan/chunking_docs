from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from chunking_docs.models import AssetKind, VisualAsset
from chunking_docs.vision.annotate import prompt_for_asset, prompt_name_for_asset
from chunking_docs.vision.interfaces import OCRBackend, VLMBackend
from chunking_docs.vision.manual_annotations import AssetAnnotation
from chunking_docs.vision.prompts import VISUAL_PROMPT_SCHEMA_VERSION
from chunking_docs.vision.vlm_output import parse_vlm_output

VisualOperation = Literal["ocr", "vlm"]
VisualJobStatus = Literal["pending", "completed", "failed", "skipped"]


class VisualAnnotationJob(BaseModel):
    job_id: str
    asset_id: str
    doc_id: str
    page_no: int
    kind: AssetKind
    asset_path: Path
    operations: list[VisualOperation]
    priority: int
    reason: str
    section_label: str = ""
    status: VisualJobStatus = "pending"
    attempts: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualJobRunResult(BaseModel):
    job_id: str
    asset_id: str
    page_no: int
    status: VisualJobStatus
    annotation: AssetAnnotation | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def plan_visual_jobs(
    assets: list[VisualAsset],
    pages: set[int] | None = None,
    kinds: set[AssetKind] | None = None,
    include_ocr: bool = True,
    include_vlm: bool = True,
    limit: int | None = None,
) -> list[VisualAnnotationJob]:
    jobs = []
    for asset in assets:
        if pages is not None and asset.page_no not in pages:
            continue
        if kinds is not None and asset.kind not in kinds:
            continue
        job = visual_job_for_asset(asset, include_ocr=include_ocr, include_vlm=include_vlm)
        if job is not None:
            jobs.append(job)

    jobs.sort(key=visual_job_sort_key)
    if limit is not None:
        return jobs[:limit]
    return jobs


def visual_job_for_asset(
    asset: VisualAsset,
    include_ocr: bool = True,
    include_vlm: bool = True,
) -> VisualAnnotationJob | None:
    if asset.path is None:
        return None

    operations: list[VisualOperation] = []
    reasons = []
    if include_ocr and asset.metadata.get("requires_ocr", True) and asset_needs_ocr(asset):
        operations.append("ocr")
        reasons.append("missing OCR")
    if include_vlm and asset.metadata.get("requires_vlm", True) and asset_needs_vlm(asset):
        operations.append("vlm")
        reasons.append("missing or unstructured VLM summary")
    if not operations:
        return None

    return VisualAnnotationJob(
        job_id=make_visual_job_id(asset.asset_id, operations),
        asset_id=asset.asset_id,
        doc_id=asset.doc_id,
        page_no=asset.page_no,
        kind=asset.kind,
        asset_path=asset.path,
        operations=operations,
        priority=visual_job_priority(asset, operations),
        reason=", ".join(reasons),
        section_label=str(asset.metadata.get("section_label", "")),
        metadata=visual_job_metadata(asset),
    )


def visual_job_sort_key(job: VisualAnnotationJob):
    return (
        -job.priority,
        job.page_no,
        int(job.metadata.get("tile_index", -1)),
        job.asset_id,
    )


def visual_job_metadata(asset: VisualAsset) -> dict[str, Any]:
    metadata = {}
    for key in (
        "asset_scope",
        "parent_asset_id",
        "tile_index",
        "tile_row",
        "tile_col",
        "tile_rows",
        "tile_cols",
        "tile_overlap_ratio",
        "text_quality",
        "text_quality_reasons",
        "control_char_count",
        "control_char_ratio",
        "letter_or_number_ratio",
        "cjk_char_ratio",
        "image_block_count",
        "embedded_image_count",
        "drawing_count",
    ):
        if key in asset.metadata:
            metadata[key] = asset.metadata.get(key)
    if "text_quality" in metadata:
        metadata["text_quality"] = str(metadata["text_quality"])
    return metadata


def run_visual_jobs(
    jobs: list[VisualAnnotationJob],
    assets: list[VisualAsset],
    ocr_backend: OCRBackend | None = None,
    vlm_backend: VLMBackend | None = None,
    limit: int | None = None,
    ocr_language: str = "kor+eng",
    ocr_backend_name: str = "",
    vlm_backend_name: str = "",
) -> list[VisualJobRunResult]:
    assets_by_id = {asset.asset_id: asset for asset in assets}
    ocr_backend_config = backend_config(ocr_backend)
    vlm_backend_config = backend_config(vlm_backend)
    results = []
    processed = 0
    for job in jobs:
        if limit is not None and processed >= limit:
            results.append(skipped_result(job, "limit reached"))
            continue
        if job.status != "pending":
            results.append(skipped_result(job, f"job status is {job.status}"))
            continue

        asset = assets_by_id.get(job.asset_id)
        if asset is None:
            results.append(failed_result(job, "asset not found"))
            continue
        if "ocr" in job.operations and ocr_backend is None:
            results.append(failed_result(job, "ocr backend is required"))
            continue
        if "vlm" in job.operations and vlm_backend is None:
            results.append(failed_result(job, "vlm backend is required"))
            continue

        try:
            annotation = run_one_visual_job(
                job,
                asset,
                ocr_backend=ocr_backend,
                vlm_backend=vlm_backend,
                ocr_language=ocr_language,
                ocr_backend_name=ocr_backend_name,
                vlm_backend_name=vlm_backend_name,
                ocr_backend_config=ocr_backend_config,
                vlm_backend_config=vlm_backend_config,
            )
        except Exception as exc:  # pragma: no cover - exercised with real model failures.
            results.append(failed_result(job, str(exc)))
            continue

        processed += 1
        results.append(
            VisualJobRunResult(
                job_id=job.job_id,
                asset_id=job.asset_id,
                page_no=job.page_no,
                status="completed",
                annotation=annotation,
                metadata={
                    "operations": job.operations,
                    "ocr_backend": ocr_backend_name,
                    "vlm_backend": vlm_backend_name,
                    "ocr_language": annotation.metadata.get("ocr_language"),
                    "ocr_backend_config": annotation.metadata.get("ocr_backend_config", {}),
                    "vlm_backend_config": annotation.metadata.get("vlm_backend_config", {}),
                    "vlm_prompt_name": annotation.metadata.get("vlm_prompt_name"),
                    "vlm_prompt_schema_version": annotation.metadata.get("vlm_prompt_schema_version"),
                    "vlm_prompt_sha256": annotation.metadata.get("vlm_prompt_sha256"),
                    "vlm_prompt_chars": annotation.metadata.get("vlm_prompt_chars"),
                    "ocr_duration_ms": annotation.metadata.get("ocr_duration_ms"),
                    "vlm_duration_ms": annotation.metadata.get("vlm_duration_ms"),
                    "total_duration_ms": annotation.metadata.get("total_duration_ms"),
                    "ocr_text_chars": annotation.metadata.get("ocr_text_chars"),
                    "vlm_output_chars": annotation.metadata.get("vlm_output_chars"),
                    "triple_count": len(annotation.triples),
                    "vlm_parse_status": annotation.metadata.get("vlm_parse_status"),
                    "entity_count": annotation.metadata.get("entity_count"),
                    "visual_element_count": annotation.metadata.get("visual_element_count"),
                    "object_count": annotation.metadata.get("object_count"),
                    "object_bbox_count": annotation.metadata.get("object_bbox_count"),
                    "explicit_triple_count": annotation.metadata.get("explicit_triple_count"),
                    "derived_triple_count": annotation.metadata.get("derived_triple_count"),
                },
            )
        )
    return results


def run_one_visual_job(
    job: VisualAnnotationJob,
    asset: VisualAsset,
    ocr_backend: OCRBackend | None,
    vlm_backend: VLMBackend | None,
    ocr_language: str,
    ocr_backend_name: str = "",
    vlm_backend_name: str = "",
    ocr_backend_config: dict[str, Any] | None = None,
    vlm_backend_config: dict[str, Any] | None = None,
) -> AssetAnnotation:
    ocr_text = None
    vlm_summary = None
    caption = None
    triples = []
    vlm_metadata: dict[str, Any] = {}
    run_metadata: dict[str, Any] = {}
    started_at = time.perf_counter()
    if "ocr" in job.operations and ocr_backend is not None:
        ocr_started_at = time.perf_counter()
        ocr_text = ocr_backend.recognize(job.asset_path, language=ocr_language)
        run_metadata["ocr_language"] = ocr_language
        run_metadata["ocr_backend_config"] = ocr_backend_config or {}
        run_metadata["ocr_duration_ms"] = elapsed_ms(ocr_started_at)
        run_metadata["ocr_text_chars"] = len(ocr_text or "")
    if "vlm" in job.operations and vlm_backend is not None:
        prompt = prompt_for_asset(asset)
        run_metadata.update(vlm_prompt_metadata(asset, prompt))
        run_metadata["vlm_backend_config"] = vlm_backend_config or {}
        vlm_started_at = time.perf_counter()
        raw_vlm_output = vlm_backend.summarize(job.asset_path, prompt=prompt)
        run_metadata["vlm_duration_ms"] = elapsed_ms(vlm_started_at)
        run_metadata["vlm_output_chars"] = len(raw_vlm_output or "")
        parsed = parse_vlm_output(raw_vlm_output)
        vlm_summary = parsed.summary
        caption = parsed.caption
        triples = parsed.triples
        vlm_metadata = parsed.metadata
    run_metadata["total_duration_ms"] = elapsed_ms(started_at)

    return AssetAnnotation(
        asset_id=job.asset_id,
        page_no=job.page_no,
        kind=asset.kind,
        caption=caption,
        ocr_text=ocr_text,
        vlm_summary=vlm_summary,
        triples=triples,
        metadata={
            "annotation_source": "visual_job",
            "visual_job_id": job.job_id,
            "operations": job.operations,
            "priority": job.priority,
            "ocr_backend": ocr_backend_name,
            "vlm_backend": vlm_backend_name,
            **run_metadata,
            **vlm_metadata,
        },
    )


def visual_job_priority(asset: VisualAsset, operations: list[VisualOperation]) -> int:
    priority = {
        AssetKind.MAP: 1000,
        AssetKind.TABLE: 900,
        AssetKind.CHART: 850,
        AssetKind.FIGURE: 800,
        AssetKind.PAGE_IMAGE: 650,
        AssetKind.UNKNOWN: 500,
    }[asset.kind]
    if "vlm" in operations:
        priority += 200
    if "ocr" in operations:
        priority += 100
    priority += visual_complexity_bonus(asset)
    if asset.metadata.get("asset_scope") == "tile":
        priority += 80
    if str(asset.metadata.get("text_quality", "")) == "empty":
        priority += 50
    if asset.metadata.get("section_label"):
        priority += 20
    return priority


def visual_complexity_bonus(asset: VisualAsset) -> int:
    image_blocks = metadata_int(asset, "image_block_count")
    embedded_images = metadata_int(asset, "embedded_image_count")
    drawings = metadata_int(asset, "drawing_count")
    return min(image_blocks, 5) * 40 + min(embedded_images, 5) * 40 + min(drawings, 20) * 5


def asset_needs_ocr(asset: VisualAsset) -> bool:
    if asset.ocr_text:
        return False
    return not ocr_attempted(asset)


def asset_needs_vlm(asset: VisualAsset) -> bool:
    if not asset.vlm_summary:
        return True
    parse_status = str(asset.metadata.get("vlm_parse_status", "")).strip()
    return parse_status not in {"json_object", "json_list", "json_repaired"}


def ocr_attempted(asset: VisualAsset) -> bool:
    return "ocr_text_chars" in asset.metadata or bool(asset.metadata.get("ocr_backend"))


def metadata_int(asset: VisualAsset, key: str) -> int:
    value = asset.metadata.get(key, 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def make_visual_job_id(asset_id: str, operations: list[VisualOperation]) -> str:
    raw = f"{asset_id}:{','.join(operations)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def vlm_prompt_metadata(asset: VisualAsset, prompt: str | None = None) -> dict[str, Any]:
    prompt = prompt if prompt is not None else prompt_for_asset(asset)
    return {
        "vlm_prompt_name": prompt_name_for_asset(asset),
        "vlm_prompt_schema_version": VISUAL_PROMPT_SCHEMA_VERSION,
        "vlm_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "vlm_prompt_chars": len(prompt),
    }


def backend_config(backend: object | None) -> dict[str, Any]:
    if backend is None:
        return {}
    metadata = getattr(backend, "metadata", None)
    if callable(metadata):
        value = metadata()
    elif isinstance(metadata, dict):
        value = metadata
    else:
        value = {}
    if not isinstance(value, dict):
        return {"value": json_safe_value(value)}
    return {str(key): json_safe_value(item) for key, item in value.items()}


def json_safe_value(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value]
    return str(value)


def completed_annotations(results: list[VisualJobRunResult]) -> list[AssetAnnotation]:
    return [result.annotation for result in results if result.annotation is not None]


def failed_result(job: VisualAnnotationJob, error: str) -> VisualJobRunResult:
    return VisualJobRunResult(
        job_id=job.job_id,
        asset_id=job.asset_id,
        page_no=job.page_no,
        status="failed",
        error=error,
        metadata={"operations": job.operations},
    )


def skipped_result(job: VisualAnnotationJob, reason: str) -> VisualJobRunResult:
    return VisualJobRunResult(
        job_id=job.job_id,
        asset_id=job.asset_id,
        page_no=job.page_no,
        status="skipped",
        error=reason,
        metadata={"operations": job.operations},
    )


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.vision.jobs import VisualJobRunResult


class DurationSummary(BaseModel):
    count: int
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0


class VisualOperationSummary(BaseModel):
    operation: str
    backend: str
    status_counts: dict[str, int] = Field(default_factory=dict)
    duration: DurationSummary = Field(default_factory=lambda: DurationSummary(count=0))
    output_chars: int = 0


class VisualRunSummary(BaseModel):
    result_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    operation_counts: dict[str, int] = Field(default_factory=dict)
    annotation_count: int
    triple_count: int
    vlm_prompt_counts: dict[str, int] = Field(default_factory=dict)
    parse_status_counts: dict[str, int] = Field(default_factory=dict)
    operation_summaries: list[VisualOperationSummary] = Field(default_factory=list)


def summarize_visual_results(results: list[VisualJobRunResult]) -> VisualRunSummary:
    status_counts = Counter(result.status for result in results)
    operation_counts: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    vlm_prompt_counts: Counter[str] = Counter()
    durations_by_operation: dict[tuple[str, str], list[float]] = defaultdict(list)
    output_chars_by_operation: Counter[tuple[str, str]] = Counter()
    status_by_operation: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    triple_count = 0

    for result in results:
        operations = result.metadata.get("operations", [])
        if isinstance(operations, str):
            operations = [operations]
        for operation in operations:
            operation_counts[str(operation)] += 1
            backend = operation_backend(result.metadata, str(operation))
            key = (str(operation), backend)
            status_by_operation[key][result.status] += 1
            duration = numeric_metadata(result.metadata, f"{operation}_duration_ms")
            if duration is not None:
                durations_by_operation[key].append(duration)
            output_chars = numeric_metadata(result.metadata, operation_output_key(operation))
            if output_chars is not None:
                output_chars_by_operation[key] += int(output_chars)

        if result.annotation is None:
            continue
        triple_count += len(result.annotation.triples)
        prompt_key = vlm_prompt_key(result.annotation.metadata)
        if prompt_key:
            vlm_prompt_counts[prompt_key] += 1
        parse_status = result.annotation.metadata.get("vlm_parse_status")
        if parse_status:
            parse_status_counts[str(parse_status)] += 1

    operation_summaries = [
        VisualOperationSummary(
            operation=operation,
            backend=backend,
            status_counts=dict(status_by_operation[(operation, backend)]),
            duration=summarize_durations(durations_by_operation.get((operation, backend), [])),
            output_chars=int(output_chars_by_operation.get((operation, backend), 0)),
        )
        for operation, backend in sorted(status_by_operation)
    ]

    return VisualRunSummary(
        result_count=len(results),
        status_counts=dict(status_counts),
        operation_counts=dict(operation_counts),
        annotation_count=sum(1 for result in results if result.annotation is not None),
        triple_count=triple_count,
        vlm_prompt_counts=dict(vlm_prompt_counts),
        parse_status_counts=dict(parse_status_counts),
        operation_summaries=operation_summaries,
    )


def operation_backend(metadata: dict[str, Any], operation: str) -> str:
    backend = metadata.get(f"{operation}_backend")
    return str(backend) if backend else "unknown"


def operation_output_key(operation: str) -> str:
    if operation == "ocr":
        return "ocr_text_chars"
    if operation == "vlm":
        return "vlm_output_chars"
    return f"{operation}_output_chars"


def vlm_prompt_key(metadata: dict[str, Any]) -> str:
    name = metadata.get("vlm_prompt_name")
    digest = metadata.get("vlm_prompt_sha256")
    if not name and not digest:
        return ""
    if name and digest:
        return f"{name}:{str(digest)[:12]}"
    return str(name or digest)


def numeric_metadata(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_durations(values: list[float]) -> DurationSummary:
    if not values:
        return DurationSummary(count=0)
    ordered = sorted(values)
    return DurationSummary(
        count=len(values),
        mean_ms=round(statistics.fmean(values), 3),
        p50_ms=round(percentile(ordered, 0.50), 3),
        p95_ms=round(percentile(ordered, 0.95), 3),
        max_ms=round(ordered[-1], 3),
    )


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

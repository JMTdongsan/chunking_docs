from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import (
    CASE_GROUP_METADATA_KEYS,
    RetrievalCase,
    excluded_target_keys,
    expected_target_keys,
    sorted_target_keys,
)
from chunking_docs.graph.provenance import asset_ids_from_ref, string_values
from chunking_docs.retrieval.context import RAGContextBundle, RAGContextChunk


class RAGContextCaseResult(BaseModel):
    query: str
    case_metadata: dict[str, Any] = Field(default_factory=dict)
    passed: bool
    latency_ms: float = 0.0
    context_char_count: int = 0
    chunk_count: int = 0
    asset_count: int = 0
    triple_count: int = 0
    pages: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    triple_ids: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_asset_ids: list[str] = Field(default_factory=list)
    expected_triple_ids: list[str] = Field(default_factory=list)
    excluded_pages: list[int] = Field(default_factory=list)
    excluded_chunk_ids: list[str] = Field(default_factory=list)
    excluded_asset_ids: list[str] = Field(default_factory=list)
    excluded_triple_ids: list[str] = Field(default_factory=list)
    expected_target_count: int = 0
    matched_target_count: int = 0
    target_coverage: float = 0.0
    excluded_target_count: int = 0
    excluded_matched_target_count: int = 0
    excluded_target_hit_rate: float = 0.0
    target_matches: dict[str, bool] = Field(default_factory=dict)
    target_key_matches: dict[str, bool] = Field(default_factory=dict)
    excluded_target_key_matches: dict[str, bool] = Field(default_factory=dict)
    matched_targets: list[str] = Field(default_factory=list)
    excluded_matched_targets: list[str] = Field(default_factory=list)


class RAGContextTargetMetric(BaseModel):
    expected_count: int = 0
    passed_count: int = 0
    target_count: int = 0
    matched_target_count: int = 0
    coverage: float = 0.0
    failed_queries: list[str] = Field(default_factory=list)


class RAGContextCaseGroupMetric(BaseModel):
    case_count: int = 0
    expected_case_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    target_count: int = 0
    matched_target_count: int = 0
    target_coverage: float = 0.0
    excluded_target_count: int = 0
    excluded_matched_target_count: int = 0
    excluded_target_hit_rate: float = 0.0
    mean_latency_ms: float = 0.0
    mean_context_char_count: float = 0.0
    failed_queries: list[str] = Field(default_factory=list)


class RAGContextEvaluation(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    case_count: int
    expected_case_count: int
    passed_count: int
    failed_count: int
    hit_rate: float
    target_coverage: float = 0.0
    excluded_query_count: int = 0
    excluded_hit_query_count: int = 0
    excluded_query_hit_rate: float = 0.0
    excluded_target_count: int = 0
    excluded_matched_target_count: int = 0
    excluded_target_hit_rate: float = 0.0
    mean_latency_ms: float = 0.0
    mean_context_char_count: float = 0.0
    max_context_char_count: int = 0
    mean_chunk_count: float = 0.0
    mean_asset_count: float = 0.0
    mean_triple_count: float = 0.0
    target_metrics: dict[str, RAGContextTargetMetric] = Field(default_factory=dict)
    case_group_metrics: dict[str, dict[str, RAGContextCaseGroupMetric]] = Field(
        default_factory=dict
    )
    failed_queries: list[str] = Field(default_factory=list)
    results: list[RAGContextCaseResult] = Field(default_factory=list)


def evaluate_rag_contexts(
    cases: list[RetrievalCase],
    bundles: list[RAGContextBundle],
    latencies_ms: list[float] | None = None,
) -> RAGContextEvaluation:
    if len(cases) != len(bundles):
        raise ValueError("cases and bundles must have the same length")
    latencies = [0.0] * len(cases) if latencies_ms is None else latencies_ms
    if len(latencies) != len(cases):
        raise ValueError("latencies_ms must match cases length")

    results = [
        evaluate_rag_context_case(case, bundle, latency_ms=latency)
        for case, bundle, latency in zip(cases, bundles, latencies)
    ]
    expected_results = [result for result in results if result.expected_target_count > 0]
    excluded_results = [result for result in results if result.excluded_target_count > 0]
    excluded_hit_query_count = sum(
        1 for result in excluded_results if result.excluded_matched_target_count > 0
    )
    expected_target_count = sum(result.expected_target_count for result in expected_results)
    matched_target_count = sum(result.matched_target_count for result in expected_results)
    excluded_target_count = sum(result.excluded_target_count for result in excluded_results)
    excluded_matched_target_count = sum(
        result.excluded_matched_target_count for result in excluded_results
    )
    passed_count = sum(1 for result in results if result.passed)
    context_char_counts = [result.context_char_count for result in results]
    return RAGContextEvaluation(
        case_count=len(results),
        expected_case_count=len(expected_results),
        passed_count=passed_count,
        failed_count=len(results) - passed_count,
        hit_rate=passed_count / len(results) if results else 0.0,
        target_coverage=matched_target_count / expected_target_count
        if expected_target_count
        else 0.0,
        excluded_query_count=len(excluded_results),
        excluded_hit_query_count=excluded_hit_query_count,
        excluded_query_hit_rate=excluded_hit_query_count / len(excluded_results)
        if excluded_results
        else 0.0,
        excluded_target_count=excluded_target_count,
        excluded_matched_target_count=excluded_matched_target_count,
        excluded_target_hit_rate=excluded_matched_target_count / excluded_target_count
        if excluded_target_count
        else 0.0,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        mean_context_char_count=sum(context_char_counts) / len(context_char_counts)
        if context_char_counts
        else 0.0,
        max_context_char_count=max(context_char_counts) if context_char_counts else 0,
        mean_chunk_count=sum(result.chunk_count for result in results) / len(results)
        if results
        else 0.0,
        mean_asset_count=sum(result.asset_count for result in results) / len(results)
        if results
        else 0.0,
        mean_triple_count=sum(result.triple_count for result in results) / len(results)
        if results
        else 0.0,
        target_metrics=rag_context_target_metrics(results),
        case_group_metrics=rag_context_case_group_metrics(results),
        failed_queries=[result.query for result in results if not result.passed],
        results=results,
    )


def evaluate_rag_context_case(
    case: RetrievalCase,
    bundle: RAGContextBundle,
    latency_ms: float = 0.0,
) -> RAGContextCaseResult:
    expected_targets = expected_target_keys(case)
    excluded_targets = excluded_target_keys(case)
    context_keys = rag_context_target_keys(bundle)
    matched_targets = expected_targets & context_keys
    excluded_matched_targets = excluded_targets & context_keys
    expected_any = bool(expected_targets)
    excluded_any = bool(excluded_targets)
    if expected_any:
        passed = len(matched_targets) == len(expected_targets) and not excluded_matched_targets
    elif excluded_any:
        passed = not excluded_matched_targets
    else:
        passed = bool(bundle.chunks or bundle.assets or bundle.triples)
    return RAGContextCaseResult(
        query=case.query,
        case_metadata=case.metadata,
        passed=passed,
        latency_ms=latency_ms,
        context_char_count=rag_context_char_count(bundle),
        chunk_count=len(bundle.chunks),
        asset_count=len(bundle.assets),
        triple_count=len(bundle.triples),
        pages=context_pages(bundle),
        chunk_ids=context_chunk_ids(bundle),
        asset_ids=context_asset_ids(bundle),
        triple_ids=context_triple_ids(bundle),
        expected_pages=case.expected_pages,
        expected_chunk_ids=case.expected_chunk_ids,
        expected_asset_ids=case.expected_asset_ids,
        expected_triple_ids=case.expected_triple_ids,
        excluded_pages=case.excluded_pages,
        excluded_chunk_ids=case.excluded_chunk_ids,
        excluded_asset_ids=case.excluded_asset_ids,
        excluded_triple_ids=case.excluded_triple_ids,
        expected_target_count=len(expected_targets),
        matched_target_count=len(matched_targets),
        target_coverage=len(matched_targets) / len(expected_targets)
        if expected_targets
        else 0.0,
        excluded_target_count=len(excluded_targets),
        excluded_matched_target_count=len(excluded_matched_targets),
        excluded_target_hit_rate=len(excluded_matched_targets) / len(excluded_targets)
        if excluded_targets
        else 0.0,
        target_matches=target_family_matches(expected_targets, matched_targets),
        target_key_matches={
            target: target in matched_targets for target in sorted_target_keys(expected_targets)
        },
        excluded_target_key_matches={
            target: target in excluded_matched_targets
            for target in sorted_target_keys(excluded_targets)
        },
        matched_targets=sorted_target_keys(matched_targets),
        excluded_matched_targets=sorted_target_keys(excluded_matched_targets),
    )


def rag_context_target_keys(bundle: RAGContextBundle) -> set[str]:
    keys = {f"page:{page}" for page in context_pages(bundle)}
    keys.update(f"chunk:{chunk_id}" for chunk_id in context_chunk_ids(bundle))
    keys.update(f"asset:{asset_id}" for asset_id in context_asset_ids(bundle))
    keys.update(f"triple:{triple_id}" for triple_id in context_triple_ids(bundle))
    return keys


def context_pages(bundle: RAGContextBundle) -> list[int]:
    pages = set()
    for chunk in bundle.chunks:
        start = min(chunk.page_start, chunk.page_end)
        end = max(chunk.page_start, chunk.page_end)
        pages.update(range(start, end + 1))
    pages.update(asset.page_no for asset in bundle.assets)
    return sorted(pages)


def context_chunk_ids(bundle: RAGContextBundle) -> list[str]:
    values: list[str] = []
    for chunk in bundle.chunks:
        values.extend(context_chunk_alias_ids(chunk))
    return stable_ordered_values(values)


def context_chunk_alias_ids(chunk: RAGContextChunk) -> list[str]:
    values = [chunk.chunk_id]
    for key in ("source_chunk_id", "parent_chunk_id"):
        values.extend(string_values(chunk.metadata.get(key)))
    return stable_ordered_values(values)


def context_asset_ids(bundle: RAGContextBundle) -> list[str]:
    values: list[str] = [asset.asset_id for asset in bundle.assets]
    for chunk in bundle.chunks:
        values.extend(chunk.asset_ids)
        for ref in chunk.source_refs:
            values.extend(sorted(asset_ids_from_ref(ref)))
        values.extend(string_values(chunk.metadata.get("retrieved_asset_ids")))
        for ref in chunk.metadata.get("retrieval_payload_refs", []):
            if isinstance(ref, dict):
                values.extend(string_values(ref.get("asset_id")))
                values.extend(string_values(ref.get("asset_ids")))
    return stable_ordered_values(values)


def context_triple_ids(bundle: RAGContextBundle) -> list[str]:
    values: list[str] = [triple.triple_id for triple in bundle.triples]
    for chunk in bundle.chunks:
        values.extend(string_values(chunk.metadata.get("retrieved_triple_ids")))
    return stable_ordered_values(values)


def target_family_matches(
    expected_targets: set[str],
    matched_targets: set[str],
) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    for target in ("page", "chunk", "asset", "triple"):
        prefix = f"{target}:"
        if any(value.startswith(prefix) for value in expected_targets):
            matches[target] = any(value.startswith(prefix) for value in matched_targets)
    return matches


def rag_context_char_count(bundle: RAGContextBundle) -> int:
    total = sum(len(chunk.text or "") for chunk in bundle.chunks)
    for asset in bundle.assets:
        total += len(asset.caption or "")
        total += len(asset.ocr_text or "")
        total += len(asset.vlm_summary or "")
    for triple in bundle.triples:
        total += len(" ".join([triple.subject, triple.predicate, triple.object]))
    return total


def rag_context_target_metrics(
    results: list[RAGContextCaseResult],
) -> dict[str, RAGContextTargetMetric]:
    metrics = {}
    for target in ("page", "chunk", "asset", "triple"):
        prefix = f"{target}:"
        target_results = [
            result
            for result in results
            if any(value.startswith(prefix) for value in result.target_key_matches)
        ]
        if not target_results:
            continue
        target_count = sum(
            1
            for result in target_results
            for key in result.target_key_matches
            if key.startswith(prefix)
        )
        matched_count = sum(
            1
            for result in target_results
            for key, matched in result.target_key_matches.items()
            if key.startswith(prefix) and matched
        )
        metrics[target] = RAGContextTargetMetric(
            expected_count=len(target_results),
            passed_count=sum(1 for result in target_results if result.target_matches.get(target)),
            target_count=target_count,
            matched_target_count=matched_count,
            coverage=matched_count / target_count if target_count else 0.0,
            failed_queries=[
                result.query
                for result in target_results
                if not result.target_matches.get(target)
            ],
        )
    return metrics


def rag_context_case_group_metrics(
    results: list[RAGContextCaseResult],
) -> dict[str, dict[str, RAGContextCaseGroupMetric]]:
    grouped: dict[str, dict[str, list[RAGContextCaseResult]]] = {}
    for result in results:
        for key in CASE_GROUP_METADATA_KEYS:
            value = result.case_metadata.get(key)
            if value is None:
                continue
            grouped.setdefault(key, {}).setdefault(str(value), []).append(result)
    return {
        key: {value: rag_context_case_group_metric(group_results) for value, group_results in values.items()}
        for key, values in grouped.items()
    }


def rag_context_case_group_metric(
    results: list[RAGContextCaseResult],
) -> RAGContextCaseGroupMetric:
    expected_results = [result for result in results if result.expected_target_count > 0]
    excluded_results = [result for result in results if result.excluded_target_count > 0]
    target_count = sum(result.expected_target_count for result in expected_results)
    matched_target_count = sum(result.matched_target_count for result in expected_results)
    excluded_target_count = sum(result.excluded_target_count for result in excluded_results)
    excluded_matched_target_count = sum(
        result.excluded_matched_target_count for result in excluded_results
    )
    return RAGContextCaseGroupMetric(
        case_count=len(results),
        expected_case_count=len(expected_results),
        passed_count=sum(1 for result in results if result.passed),
        failed_count=sum(1 for result in results if not result.passed),
        target_count=target_count,
        matched_target_count=matched_target_count,
        target_coverage=matched_target_count / target_count if target_count else 0.0,
        excluded_target_count=excluded_target_count,
        excluded_matched_target_count=excluded_matched_target_count,
        excluded_target_hit_rate=excluded_matched_target_count / excluded_target_count
        if excluded_target_count
        else 0.0,
        mean_latency_ms=sum(result.latency_ms for result in results) / len(results)
        if results
        else 0.0,
        mean_context_char_count=sum(result.context_char_count for result in results)
        / len(results)
        if results
        else 0.0,
        failed_queries=[result.query for result in results if not result.passed],
    )


def stable_ordered_values(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result

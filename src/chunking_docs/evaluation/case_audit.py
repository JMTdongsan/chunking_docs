from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import RetrievalCase, case_group_labels, normalized_metric_label
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, VisualAsset

_SPACE_RE = re.compile(r"\s+")
_QUERY_TERM_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_./%+-]{1,}|"
    r"[0-9]+(?:[.,][0-9]+)*(?:[%A-Za-z]*)?|"
    r"[\uac00-\ud7a3]{2,}|"
    r"[\u4e00-\u9fff]{2,}"
)
_QUERY_WRAPPER_TERMS = {
    "about",
    "discussed",
    "evidence",
    "explains",
    "find",
    "is",
    "source",
    "where",
    "which",
    "관련",
    "근거가",
    "내용을",
    "다루는가",
    "설명하는가",
    "어디에서",
    "어떤",
    "있는가",
    "찾을",
    "항목을",
}


class RetrievalCaseAuditIssue(BaseModel):
    severity: str
    code: str
    message: str
    case_index: int | None = None
    query: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCaseAuditCheck(BaseModel):
    name: str
    metric: str
    operator: str
    actual: int | float
    threshold: int | float
    passed: bool


class RetrievalCaseAuditReport(BaseModel):
    passed: bool
    case_count: int
    expected_case_count: int
    target_counts: dict[str, int] = Field(default_factory=dict)
    distinct_target_counts: dict[str, int] = Field(default_factory=dict)
    excluded_target_counts: dict[str, int] = Field(default_factory=dict)
    excluded_distinct_target_counts: dict[str, int] = Field(default_factory=dict)
    max_cases_per_target: dict[str, int] = Field(default_factory=dict)
    case_group_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    case_group_distinct_target_counts: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)
    case_group_max_cases_per_target: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)
    visual_object_probe_count: int = 0
    visual_only_object_probe_count: int = 0
    non_visual_only_object_probe_count: int = 0
    graph_expand_count: int = 0
    duplicate_query_count: int = 0
    empty_query_count: int = 0
    todo_query_count: int = 0
    short_query_count: int = 0
    min_query_term_count: int = 0
    max_query_term_count: int = 0
    target_query_overlap_count: int = 0
    target_query_overlap_term_count: int = 0
    max_target_query_overlap_ratio: float = 0.0
    mean_target_query_overlap_ratio: float = 0.0
    max_target_query_overlap_terms: int = 0
    mean_target_query_overlap_terms: float = 0.0
    max_expected_targets_per_case: int = 0
    oversized_expected_target_case_count: int = 0
    missing_target_counts: dict[str, int] = Field(default_factory=dict)
    excluded_missing_target_counts: dict[str, int] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    checks: list[RetrievalCaseAuditCheck] = Field(default_factory=list)
    issues: list[RetrievalCaseAuditIssue] = Field(default_factory=list)


def audit_retrieval_cases(
    cases: list[RetrievalCase],
    profiles: list[PageProfile],
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    min_case_count: int = 1,
    min_page_cases: int = 0,
    min_chunk_cases: int = 0,
    min_asset_cases: int = 0,
    min_triple_cases: int = 0,
    min_distinct_page_targets: int = 0,
    min_distinct_chunk_targets: int = 0,
    min_distinct_asset_targets: int = 0,
    min_distinct_triple_targets: int = 0,
    max_page_cases_per_target: int | None = None,
    max_chunk_cases_per_target: int | None = None,
    max_asset_cases_per_target: int | None = None,
    max_triple_cases_per_target: int | None = None,
    min_case_group_counts: dict[str, int] | None = None,
    min_case_group_distinct_targets: dict[str, int] | None = None,
    max_case_group_cases_per_target: dict[str, int] | None = None,
    require_visual_only_object_probes: bool = False,
    min_query_terms_per_case: int = 0,
    max_target_query_overlap_ratio: float | None = None,
    max_target_query_overlap_terms: int | None = None,
    min_terms_for_target_overlap: int = 4,
    max_expected_targets_per_case: int | None = None,
    max_duplicate_queries: int = 0,
    max_issues: int = 200,
) -> RetrievalCaseAuditReport:
    page_numbers = known_pages(profiles, chunks)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    asset_ids = {asset.asset_id for asset in assets}
    triple_ids = {triple.triple_id for triple in triples}
    target_text_index = build_target_text_index(chunks, assets, triples)
    issues: list[RetrievalCaseAuditIssue] = []
    target_counts = count_retrieval_case_targets(cases)
    distinct_target_counts = count_retrieval_case_distinct_targets(cases)
    excluded_target_counts = count_retrieval_case_excluded_targets(cases)
    excluded_distinct_target_counts = count_retrieval_case_distinct_excluded_targets(cases)
    max_cases_per_target = count_retrieval_case_max_target_mentions(cases)
    group_counts = count_case_groups(cases)
    group_distinct_target_counts = count_case_group_distinct_targets(cases)
    group_max_cases_per_target = count_case_group_max_target_mentions(cases)
    visual_object_probe_counts = count_visual_object_probes(cases)
    missing_target_counts = {"page": 0, "chunk": 0, "asset": 0, "triple": 0}
    excluded_missing_target_counts = {"page": 0, "chunk": 0, "asset": 0, "triple": 0}
    normalized_queries: dict[str, list[int]] = {}
    query_term_counts: list[int] = []
    target_query_overlap_ratios: list[float] = []
    target_query_overlap_terms: list[int] = []
    target_query_overlap_check_ratios: list[float] = []
    target_query_overlap_check_terms: list[int] = []
    expected_target_counts: list[int] = []
    oversized_expected_target_case_count = 0
    target_query_overlap_count = 0
    target_query_overlap_term_count = 0

    for index, case in enumerate(cases):
        query = case.query.strip()
        query_terms = distinct_query_terms(query)
        query_term_counts.append(len(query_terms))
        target_overlap_terms = target_query_overlap_term_count_for_case(
            case,
            query_terms,
            target_text_index,
        )
        target_overlap_ratio = target_query_overlap_ratio_from_terms(
            target_overlap_terms,
            query_terms,
        )
        target_query_overlap_ratios.append(target_overlap_ratio)
        target_query_overlap_terms.append(target_overlap_terms)
        expected_target_count = retrieval_case_expected_target_count(case)
        expected_target_counts.append(expected_target_count)
        if (
            max_expected_targets_per_case is not None
            and expected_target_count > max_expected_targets_per_case
        ):
            oversized_expected_target_case_count += 1
            append_issue(
                issues,
                max_issues,
                issue(
                    "warning",
                    "too_many_expected_targets",
                    "Retrieval case has more expected targets than the configured per-case ceiling.",
                    index,
                    case,
                    {
                        "expected_target_count": expected_target_count,
                        "max_expected_targets_per_case": max_expected_targets_per_case,
                    },
                ),
            )
        if len(query_terms) >= min_terms_for_target_overlap:
            target_query_overlap_check_ratios.append(target_overlap_ratio)
            target_query_overlap_check_terms.append(target_overlap_terms)
            if (
                max_target_query_overlap_ratio is not None
                and target_overlap_ratio > max_target_query_overlap_ratio
            ):
                target_query_overlap_count += 1
                append_issue(
                    issues,
                    max_issues,
                    issue(
                        "warning",
                        "target_query_overlap",
                        "Retrieval case query terms overlap the expected target text too strongly.",
                        index,
                        case,
                        {
                            "target_query_overlap_ratio": target_overlap_ratio,
                            "max_target_query_overlap_ratio": max_target_query_overlap_ratio,
                            "min_terms_for_target_overlap": min_terms_for_target_overlap,
                        },
                    ),
                )
            if (
                max_target_query_overlap_terms is not None
                and target_overlap_terms > max_target_query_overlap_terms
            ):
                target_query_overlap_term_count += 1
                append_issue(
                    issues,
                    max_issues,
                    issue(
                        "warning",
                        "target_query_overlap_terms",
                        "Retrieval case query uses too many terms from the expected target text.",
                        index,
                        case,
                        {
                            "target_query_overlap_terms": target_overlap_terms,
                            "max_target_query_overlap_terms": max_target_query_overlap_terms,
                            "min_terms_for_target_overlap": min_terms_for_target_overlap,
                        },
                    ),
                )
        normalized_query = normalize_query(query)
        if normalized_query:
            normalized_queries.setdefault(normalized_query, []).append(index)
        if not query:
            append_issue(issues, max_issues, issue("error", "empty_query", "Retrieval case query is empty.", index, case))
        if query.upper().startswith("TODO:"):
            append_issue(
                issues,
                max_issues,
                issue("error", "todo_query", "Retrieval case still contains a TODO query.", index, case),
            )
        if not has_expected_target(case):
            append_issue(
                issues,
                max_issues,
                issue(
                    "error",
                    "missing_expected_target",
                    "Retrieval case does not declare an expected page, chunk, asset, or triple target.",
                    index,
                    case,
                ),
            )

        missing_target_counts["page"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "page",
            case.expected_pages,
            page_numbers,
        )
        missing_target_counts["chunk"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "chunk",
            case.expected_chunk_ids,
            chunk_ids,
        )
        missing_target_counts["asset"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "asset",
            case.expected_asset_ids,
            asset_ids,
        )
        missing_target_counts["triple"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "triple",
            case.expected_triple_ids,
            triple_ids,
        )
        excluded_missing_target_counts["page"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "excluded_page",
            case.excluded_pages,
            page_numbers,
        )
        excluded_missing_target_counts["chunk"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "excluded_chunk",
            case.excluded_chunk_ids,
            chunk_ids,
        )
        excluded_missing_target_counts["asset"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "excluded_asset",
            case.excluded_asset_ids,
            asset_ids,
        )
        excluded_missing_target_counts["triple"] += report_missing_values(
            issues,
            max_issues,
            index,
            case,
            "excluded_triple",
            case.excluded_triple_ids,
            triple_ids,
        )
        if case.expected_triple_ids and not case.graph_expand:
            append_issue(
                issues,
                max_issues,
                issue(
                    "warning",
                    "triple_case_without_graph_expand",
                    "Triple-target retrieval case should enable graph expansion.",
                    index,
                    case,
                ),
            )
        if require_visual_only_object_probes and is_visual_object_probe(case) and not is_visual_only_object_probe(case):
            append_issue(
                issues,
                max_issues,
                issue(
                    "error",
                    "non_visual_only_object_probe",
                    "Visual object probe case was not generated with visual-only object terms.",
                    index,
                    case,
                    {"object_probe_visual_only": case.metadata.get("object_probe_visual_only")},
                ),
            )
        if min_query_terms_per_case > 0 and len(query_terms) < min_query_terms_per_case:
            append_issue(
                issues,
                max_issues,
                issue(
                    "warning",
                    "short_query",
                    "Retrieval case query has fewer distinct terms than the configured minimum.",
                    index,
                    case,
                    {
                        "query_term_count": len(query_terms),
                        "min_query_terms_per_case": min_query_terms_per_case,
                    },
                ),
            )

    duplicate_query_count = duplicate_count(normalized_queries)
    for query, indexes in sorted(normalized_queries.items()):
        if len(indexes) <= 1:
            continue
        append_issue(
            issues,
            max_issues,
            RetrievalCaseAuditIssue(
                severity="warning",
                code="duplicate_query",
                message="Multiple retrieval cases use the same normalized query.",
                case_index=indexes[0],
                query=query,
                metadata={"case_indexes": indexes},
            ),
        )

    min_query_term_count = min(query_term_counts, default=0)
    max_query_term_count = max(query_term_counts, default=0)
    max_target_query_overlap = max(target_query_overlap_check_ratios, default=0.0)
    max_target_query_overlap_term_count = max(target_query_overlap_check_terms, default=0)
    mean_target_query_overlap = (
        sum(target_query_overlap_ratios) / len(target_query_overlap_ratios)
        if target_query_overlap_ratios
        else 0.0
    )
    mean_target_query_overlap_term_count = (
        sum(target_query_overlap_terms) / len(target_query_overlap_terms)
        if target_query_overlap_terms
        else 0.0
    )
    short_query_count = (
        sum(1 for count in query_term_counts if count < min_query_terms_per_case)
        if min_query_terms_per_case > 0
        else 0
    )
    metrics = {
        "case_count": len(cases),
        "page_cases": target_counts["page"],
        "chunk_cases": target_counts["chunk"],
        "asset_cases": target_counts["asset"],
        "triple_cases": target_counts["triple"],
        "distinct_page_targets": distinct_target_counts["page"],
        "distinct_chunk_targets": distinct_target_counts["chunk"],
        "distinct_asset_targets": distinct_target_counts["asset"],
        "distinct_triple_targets": distinct_target_counts["triple"],
        "max_page_cases_per_target": max_cases_per_target["page"],
        "max_chunk_cases_per_target": max_cases_per_target["chunk"],
        "max_asset_cases_per_target": max_cases_per_target["asset"],
        "max_triple_cases_per_target": max_cases_per_target["triple"],
        "non_visual_only_object_probe_count": visual_object_probe_counts["non_visual_only"],
        "duplicate_query_count": duplicate_query_count,
        "min_query_term_count": min_query_term_count,
        "max_target_query_overlap_ratio": max_target_query_overlap,
        "max_target_query_overlap_terms": max_target_query_overlap_term_count,
        "max_expected_targets_per_case": max(expected_target_counts, default=0),
        "oversized_expected_target_case_count": oversized_expected_target_case_count,
    }
    checks = [
        min_check("min_case_count", "case_count", metrics, min_case_count),
        min_check("min_page_cases", "page_cases", metrics, min_page_cases),
        min_check("min_chunk_cases", "chunk_cases", metrics, min_chunk_cases),
        min_check("min_asset_cases", "asset_cases", metrics, min_asset_cases),
        min_check("min_triple_cases", "triple_cases", metrics, min_triple_cases),
        min_check(
            "min_distinct_page_targets",
            "distinct_page_targets",
            metrics,
            min_distinct_page_targets,
        ),
        min_check(
            "min_distinct_chunk_targets",
            "distinct_chunk_targets",
            metrics,
            min_distinct_chunk_targets,
        ),
        min_check(
            "min_distinct_asset_targets",
            "distinct_asset_targets",
            metrics,
            min_distinct_asset_targets,
        ),
        min_check(
            "min_distinct_triple_targets",
            "distinct_triple_targets",
            metrics,
            min_distinct_triple_targets,
        ),
        max_check("max_duplicate_queries", "duplicate_query_count", metrics, max_duplicate_queries),
    ]
    if min_query_terms_per_case > 0:
        checks.append(
            min_check(
                "min_query_terms_per_case",
                "min_query_term_count",
                metrics,
                min_query_terms_per_case,
            )
        )
    if max_target_query_overlap_ratio is not None:
        checks.append(
            max_check(
                "max_target_query_overlap_ratio",
                "max_target_query_overlap_ratio",
                metrics,
                max_target_query_overlap_ratio,
            )
        )
    if max_target_query_overlap_terms is not None:
        checks.append(
            max_check(
                "max_target_query_overlap_terms",
                "max_target_query_overlap_terms",
                metrics,
                max_target_query_overlap_terms,
            )
        )
    if max_expected_targets_per_case is not None:
        checks.append(
            max_check(
                "max_expected_targets_per_case",
                "max_expected_targets_per_case",
                metrics,
                max_expected_targets_per_case,
            )
        )
    checks.extend(
        max_cases_per_target_checks(
            metrics,
            {
                "page": max_page_cases_per_target,
                "chunk": max_chunk_cases_per_target,
                "asset": max_asset_cases_per_target,
                "triple": max_triple_cases_per_target,
            },
        )
    )
    if require_visual_only_object_probes:
        checks.append(
            max_check(
                "require_visual_only_object_probes",
                "non_visual_only_object_probe_count",
                metrics,
                0,
            )
        )
    checks.extend(case_group_count_checks(group_counts, min_case_group_counts or {}))
    checks.extend(
        case_group_distinct_target_checks(
            group_distinct_target_counts,
            min_case_group_distinct_targets or {},
        )
    )
    checks.extend(
        case_group_max_cases_per_target_checks(
            group_max_cases_per_target,
            max_case_group_cases_per_target or {},
        )
    )
    failed_checks = [check.name for check in checks if not check.passed]
    return RetrievalCaseAuditReport(
        passed=not failed_checks and not any(item.severity == "error" for item in issues),
        case_count=len(cases),
        expected_case_count=sum(1 for case in cases if has_expected_target(case)),
        target_counts=target_counts,
        distinct_target_counts=distinct_target_counts,
        excluded_target_counts=excluded_target_counts,
        excluded_distinct_target_counts=excluded_distinct_target_counts,
        max_cases_per_target=max_cases_per_target,
        case_group_counts=group_counts,
        case_group_distinct_target_counts=group_distinct_target_counts,
        case_group_max_cases_per_target=group_max_cases_per_target,
        visual_object_probe_count=visual_object_probe_counts["total"],
        visual_only_object_probe_count=visual_object_probe_counts["visual_only"],
        non_visual_only_object_probe_count=visual_object_probe_counts["non_visual_only"],
        graph_expand_count=sum(1 for case in cases if case.graph_expand),
        duplicate_query_count=duplicate_query_count,
        empty_query_count=sum(1 for case in cases if not case.query.strip()),
        todo_query_count=sum(1 for case in cases if case.query.strip().upper().startswith("TODO:")),
        short_query_count=short_query_count,
        min_query_term_count=min_query_term_count,
        max_query_term_count=max_query_term_count,
        target_query_overlap_count=target_query_overlap_count,
        target_query_overlap_term_count=target_query_overlap_term_count,
        max_target_query_overlap_ratio=max_target_query_overlap,
        mean_target_query_overlap_ratio=mean_target_query_overlap,
        max_target_query_overlap_terms=max_target_query_overlap_term_count,
        mean_target_query_overlap_terms=mean_target_query_overlap_term_count,
        max_expected_targets_per_case=metrics["max_expected_targets_per_case"],
        oversized_expected_target_case_count=oversized_expected_target_case_count,
        missing_target_counts=missing_target_counts,
        excluded_missing_target_counts=excluded_missing_target_counts,
        failed_checks=failed_checks,
        checks=checks,
        issues=issues,
    )


def count_retrieval_case_targets(cases: list[RetrievalCase]) -> dict[str, int]:
    return {
        "page": sum(1 for case in cases if case.expected_pages),
        "chunk": sum(1 for case in cases if case.expected_chunk_ids),
        "asset": sum(1 for case in cases if case.expected_asset_ids),
        "triple": sum(1 for case in cases if case.expected_triple_ids),
    }


def retrieval_case_expected_target_count(case: RetrievalCase) -> int:
    return (
        len(case.expected_pages)
        + len(case.expected_chunk_ids)
        + len(case.expected_asset_ids)
        + len(case.expected_triple_ids)
    )


def count_retrieval_case_distinct_targets(cases: list[RetrievalCase]) -> dict[str, int]:
    return {
        "page": len({page for case in cases for page in case.expected_pages}),
        "chunk": len({chunk_id for case in cases for chunk_id in case.expected_chunk_ids}),
        "asset": len({asset_id for case in cases for asset_id in case.expected_asset_ids}),
        "triple": len({triple_id for case in cases for triple_id in case.expected_triple_ids}),
    }


def count_retrieval_case_excluded_targets(cases: list[RetrievalCase]) -> dict[str, int]:
    return {
        "page": sum(1 for case in cases if case.excluded_pages),
        "chunk": sum(1 for case in cases if case.excluded_chunk_ids),
        "asset": sum(1 for case in cases if case.excluded_asset_ids),
        "triple": sum(1 for case in cases if case.excluded_triple_ids),
    }


def count_retrieval_case_distinct_excluded_targets(cases: list[RetrievalCase]) -> dict[str, int]:
    return {
        "page": len({page for case in cases for page in case.excluded_pages}),
        "chunk": len({chunk_id for case in cases for chunk_id in case.excluded_chunk_ids}),
        "asset": len({asset_id for case in cases for asset_id in case.excluded_asset_ids}),
        "triple": len({triple_id for case in cases for triple_id in case.excluded_triple_ids}),
    }


def count_retrieval_case_max_target_mentions(cases: list[RetrievalCase]) -> dict[str, int]:
    return {
        "page": max_target_mentions(case.expected_pages for case in cases),
        "chunk": max_target_mentions(case.expected_chunk_ids for case in cases),
        "asset": max_target_mentions(case.expected_asset_ids for case in cases),
        "triple": max_target_mentions(case.expected_triple_ids for case in cases),
    }


def max_target_mentions(target_values: Any) -> int:
    counts: dict[Any, int] = {}
    for values in target_values:
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    return max(counts.values(), default=0)


def count_case_groups(cases: list[RetrievalCase]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for case in cases:
        for group_name, group_value in case_group_labels(case):
            counts.setdefault(group_name, {}).setdefault(group_value, 0)
            counts[group_name][group_value] += 1
    return {
        group_name: dict(sorted(values.items()))
        for group_name, values in sorted(counts.items())
    }


def count_case_group_distinct_targets(cases: list[RetrievalCase]) -> dict[str, dict[str, dict[str, int]]]:
    grouped: dict[str, dict[str, dict[str, set[Any]]]] = {}
    for case in cases:
        targets = retrieval_case_target_values(case)
        for group_name, group_value in case_group_labels(case):
            group_targets = grouped.setdefault(group_name, {}).setdefault(
                group_value,
                {"page": set(), "chunk": set(), "asset": set(), "triple": set()},
            )
            for target_name, values in targets.items():
                group_targets[target_name].update(values)
    return {
        group_name: {
            group_value: {
                target_name: len(values)
                for target_name, values in sorted(targets.items())
            }
            for group_value, targets in sorted(group_values.items())
        }
        for group_name, group_values in sorted(grouped.items())
    }


def count_case_group_max_target_mentions(
    cases: list[RetrievalCase],
) -> dict[str, dict[str, dict[str, int]]]:
    grouped: dict[str, dict[str, list[RetrievalCase]]] = {}
    for case in cases:
        for group_name, group_value in case_group_labels(case):
            grouped.setdefault(group_name, {}).setdefault(group_value, []).append(case)
    return {
        group_name: {
            group_value: count_retrieval_case_max_target_mentions(group_cases)
            for group_value, group_cases in sorted(group_values.items())
        }
        for group_name, group_values in sorted(grouped.items())
    }


def retrieval_case_target_values(case: RetrievalCase) -> dict[str, list[Any]]:
    return {
        "page": list(case.expected_pages),
        "chunk": list(case.expected_chunk_ids),
        "asset": list(case.expected_asset_ids),
        "triple": list(case.expected_triple_ids),
    }


def build_target_text_index(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
) -> dict[str, dict[Any, str]]:
    chunks_by_page: dict[int, list[DocumentChunk]] = {}
    for chunk in chunks:
        for page_no in range(chunk.page_start, chunk.page_end + 1):
            chunks_by_page.setdefault(page_no, []).append(chunk)
    assets_by_page: dict[int, list[VisualAsset]] = {}
    for asset in assets:
        assets_by_page.setdefault(asset.page_no, []).append(asset)

    return {
        "page": {
            page_no: "\n".join(
                [
                    *[target_text_for_chunk(chunk) for chunk in chunks_by_page.get(page_no, [])],
                    *[target_text_for_asset(asset) for asset in assets_by_page.get(page_no, [])],
                ]
            )
            for page_no in sorted(set(chunks_by_page) | set(assets_by_page))
        },
        "chunk": {chunk.chunk_id: target_text_for_chunk(chunk) for chunk in chunks},
        "asset": {asset.asset_id: target_text_for_asset(asset) for asset in assets},
        "triple": {triple.triple_id: target_text_for_triple(triple) for triple in triples},
    }


def target_query_overlap_ratio(
    case: RetrievalCase,
    query_terms: list[str],
    target_text_index: dict[str, dict[Any, str]],
) -> float:
    return target_query_overlap_ratio_from_terms(
        target_query_overlap_term_count_for_case(case, query_terms, target_text_index),
        query_terms,
    )


def target_query_overlap_ratio_from_terms(
    overlap_terms: int,
    query_terms: list[str],
) -> float:
    if not query_terms:
        return 0.0
    return overlap_terms / len(set(query_terms))


def target_query_overlap_term_count_for_case(
    case: RetrievalCase,
    query_terms: list[str],
    target_text_index: dict[str, dict[Any, str]],
) -> int:
    if not query_terms:
        return 0
    query_term_set = set(query_terms) - _QUERY_WRAPPER_TERMS
    if not query_term_set:
        return 0
    target_terms: set[str] = set()
    for target_name, target_values in retrieval_case_target_values(case).items():
        for target_value in target_values:
            target_terms.update(distinct_query_terms(target_text_index.get(target_name, {}).get(target_value, "")))
    if not target_terms:
        return 0
    return len(query_term_set.intersection(target_terms))


def target_text_for_chunk(chunk: DocumentChunk) -> str:
    return "\n".join([chunk.text, *text_fragments(chunk.metadata)])


def target_text_for_asset(asset: VisualAsset) -> str:
    return "\n".join(
        [
            asset.caption or "",
            asset.ocr_text or "",
            asset.vlm_summary or "",
            *text_fragments(asset.metadata),
        ]
    )


def target_text_for_triple(triple: GraphTriple) -> str:
    return "\n".join(
        [
            triple.subject,
            triple.predicate,
            triple.object,
            *text_fragments(triple.qualifiers),
        ]
    )


def text_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        fragments = []
        for item in value.values():
            fragments.extend(text_fragments(item))
        return fragments
    if isinstance(value, (list, tuple, set)):
        fragments = []
        for item in value:
            fragments.extend(text_fragments(item))
        return fragments
    return []


def count_visual_object_probes(cases: list[RetrievalCase]) -> dict[str, int]:
    total = 0
    visual_only = 0
    for case in cases:
        if not is_visual_object_probe(case):
            continue
        total += 1
        visual_only += int(is_visual_only_object_probe(case))
    return {
        "total": total,
        "visual_only": visual_only,
        "non_visual_only": total - visual_only,
    }


def count_visual_image_probes(cases: list[RetrievalCase]) -> int:
    return sum(1 for case in cases if is_visual_image_probe(case))


def is_visual_image_probe(case: RetrievalCase) -> bool:
    return ("case_source", "visual_image_probe") in case_group_labels(case)


def is_visual_object_probe(case: RetrievalCase) -> bool:
    return ("case_source", "visual_object_probe") in case_group_labels(case)


def is_visual_only_object_probe(case: RetrievalCase) -> bool:
    return case.metadata.get("object_probe_visual_only") is True


def case_group_count_checks(
    group_counts: dict[str, dict[str, int]],
    thresholds: dict[str, int],
) -> list[RetrievalCaseAuditCheck]:
    checks = []
    for group_spec, threshold in sorted(thresholds.items()):
        group_name, group_value = parse_case_group_count_spec(group_spec)
        actual = group_counts.get(group_name, {}).get(group_value, 0)
        checks.append(
            RetrievalCaseAuditCheck(
                name=f"min_case_group_count:{group_name}:{group_value}",
                metric=f"case_group.{group_name}.{group_value}.case_count",
                operator=">=",
                actual=actual,
                threshold=threshold,
                passed=actual >= threshold,
            )
        )
    return checks


def case_group_distinct_target_checks(
    group_target_counts: dict[str, dict[str, dict[str, int]]],
    thresholds: dict[str, int],
) -> list[RetrievalCaseAuditCheck]:
    checks = []
    for group_spec, threshold in sorted(thresholds.items()):
        group_name, group_value, target_name = parse_case_group_target_spec(group_spec)
        actual = group_target_counts.get(group_name, {}).get(group_value, {}).get(target_name, 0)
        checks.append(
            RetrievalCaseAuditCheck(
                name=f"min_case_group_distinct_targets:{group_name}:{group_value}:{target_name}",
                metric=f"case_group.{group_name}.{group_value}.distinct_{target_name}_targets",
                operator=">=",
                actual=actual,
                threshold=threshold,
                passed=actual >= threshold,
            )
        )
    return checks


def case_group_max_cases_per_target_checks(
    group_target_counts: dict[str, dict[str, dict[str, int]]],
    thresholds: dict[str, int],
) -> list[RetrievalCaseAuditCheck]:
    checks = []
    for group_spec, threshold in sorted(thresholds.items()):
        group_name, group_value, target_name = parse_case_group_target_spec(group_spec)
        actual = group_target_counts.get(group_name, {}).get(group_value, {}).get(target_name, 0)
        checks.append(
            RetrievalCaseAuditCheck(
                name=f"max_case_group_cases_per_target:{group_name}:{group_value}:{target_name}",
                metric=f"case_group.{group_name}.{group_value}.max_{target_name}_cases_per_target",
                operator="<=",
                actual=actual,
                threshold=threshold,
                passed=actual <= threshold,
            )
        )
    return checks


def max_cases_per_target_checks(
    metrics: dict[str, int | float],
    thresholds: dict[str, int | None],
) -> list[RetrievalCaseAuditCheck]:
    checks = []
    for target_name, threshold in thresholds.items():
        if threshold is None:
            continue
        metric = f"max_{target_name}_cases_per_target"
        checks.append(max_check(metric, metric, metrics, threshold))
    return checks


def parse_case_group_count_spec(value: str) -> tuple[str, str]:
    if ":" in value:
        group_name, group_value = value.split(":", 1)
    else:
        group_name, group_value = "case_source", value
    return normalized_metric_label(group_name), normalized_metric_label(group_value)


def parse_case_group_target_spec(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) == 3:
        group_name, group_value, target_name = parts
    elif len(parts) == 2:
        group_name = "case_source"
        group_value, target_name = parts
    else:
        raise ValueError("case group target spec must use group:value:target or value:target")
    target_name = normalized_metric_label(target_name)
    if target_name not in {"page", "chunk", "asset", "triple"}:
        raise ValueError("case group target type must be page, chunk, asset, or triple")
    return normalized_metric_label(group_name), normalized_metric_label(group_value), target_name


def known_pages(profiles: list[PageProfile], chunks: list[DocumentChunk]) -> set[int]:
    pages = {profile.page_no for profile in profiles}
    if pages:
        return pages
    return {page for chunk in chunks for page in range(chunk.page_start, chunk.page_end + 1)}


def report_missing_values(
    issues: list[RetrievalCaseAuditIssue],
    max_issues: int,
    index: int,
    case: RetrievalCase,
    target_name: str,
    expected_values: list,
    known_values: set,
) -> int:
    missing = sorted(value for value in expected_values if value not in known_values)
    if missing:
        append_issue(
            issues,
            max_issues,
            issue(
                "error",
                f"unknown_{target_name}_target",
                f"Retrieval case references {target_name} targets that are not in the package.",
                index,
                case,
                {"missing": missing},
            ),
        )
    return len(missing)


def has_expected_target(case: RetrievalCase) -> bool:
    return bool(
        case.expected_pages
        or case.expected_chunk_ids
        or case.expected_asset_ids
        or case.expected_triple_ids
    )


def normalize_query(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip().casefold())


def distinct_query_terms(value: str) -> list[str]:
    terms = []
    seen = set()
    for match in _QUERY_TERM_RE.finditer(value):
        term = match.group(0).casefold()
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def duplicate_count(queries: dict[str, list[int]]) -> int:
    return sum(len(indexes) - 1 for indexes in queries.values() if len(indexes) > 1)


def issue(
    severity: str,
    code: str,
    message: str,
    index: int,
    case: RetrievalCase,
    metadata: dict[str, Any] | None = None,
) -> RetrievalCaseAuditIssue:
    return RetrievalCaseAuditIssue(
        severity=severity,
        code=code,
        message=message,
        case_index=index,
        query=case.query,
        metadata=metadata or {},
    )


def append_issue(
    issues: list[RetrievalCaseAuditIssue],
    max_issues: int,
    value: RetrievalCaseAuditIssue,
) -> None:
    if len(issues) < max_issues:
        issues.append(value)


def min_check(
    name: str,
    metric: str,
    metrics: dict[str, int | float],
    threshold: int | float,
) -> RetrievalCaseAuditCheck:
    actual = metrics[metric]
    return RetrievalCaseAuditCheck(
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
    metrics: dict[str, int | float],
    threshold: int | float,
) -> RetrievalCaseAuditCheck:
    actual = metrics[metric]
    return RetrievalCaseAuditCheck(
        name=name,
        metric=metric,
        operator="<=",
        actual=actual,
        threshold=threshold,
        passed=actual <= threshold,
    )

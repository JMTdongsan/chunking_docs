from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, VisualAsset

_SPACE_RE = re.compile(r"\s+")


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
    actual: int
    threshold: int
    passed: bool


class RetrievalCaseAuditReport(BaseModel):
    passed: bool
    case_count: int
    expected_case_count: int
    target_counts: dict[str, int] = Field(default_factory=dict)
    graph_expand_count: int = 0
    duplicate_query_count: int = 0
    empty_query_count: int = 0
    todo_query_count: int = 0
    missing_target_counts: dict[str, int] = Field(default_factory=dict)
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
    max_duplicate_queries: int = 0,
    max_issues: int = 200,
) -> RetrievalCaseAuditReport:
    page_numbers = known_pages(profiles, chunks)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    asset_ids = {asset.asset_id for asset in assets}
    triple_ids = {triple.triple_id for triple in triples}
    issues: list[RetrievalCaseAuditIssue] = []
    target_counts = {"page": 0, "chunk": 0, "asset": 0, "triple": 0}
    missing_target_counts = {"page": 0, "chunk": 0, "asset": 0, "triple": 0}
    normalized_queries: dict[str, list[int]] = {}

    for index, case in enumerate(cases):
        query = case.query.strip()
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

        target_counts["page"] += int(bool(case.expected_pages))
        target_counts["chunk"] += int(bool(case.expected_chunk_ids))
        target_counts["asset"] += int(bool(case.expected_asset_ids))
        target_counts["triple"] += int(bool(case.expected_triple_ids))

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

    metrics = {
        "case_count": len(cases),
        "page_cases": target_counts["page"],
        "chunk_cases": target_counts["chunk"],
        "asset_cases": target_counts["asset"],
        "triple_cases": target_counts["triple"],
        "duplicate_query_count": duplicate_query_count,
    }
    checks = [
        min_check("min_case_count", "case_count", metrics, min_case_count),
        min_check("min_page_cases", "page_cases", metrics, min_page_cases),
        min_check("min_chunk_cases", "chunk_cases", metrics, min_chunk_cases),
        min_check("min_asset_cases", "asset_cases", metrics, min_asset_cases),
        min_check("min_triple_cases", "triple_cases", metrics, min_triple_cases),
        max_check("max_duplicate_queries", "duplicate_query_count", metrics, max_duplicate_queries),
    ]
    failed_checks = [check.name for check in checks if not check.passed]
    return RetrievalCaseAuditReport(
        passed=not failed_checks and not any(item.severity == "error" for item in issues),
        case_count=len(cases),
        expected_case_count=sum(1 for case in cases if has_expected_target(case)),
        target_counts=target_counts,
        graph_expand_count=sum(1 for case in cases if case.graph_expand),
        duplicate_query_count=duplicate_query_count,
        empty_query_count=sum(1 for case in cases if not case.query.strip()),
        todo_query_count=sum(1 for case in cases if case.query.strip().upper().startswith("TODO:")),
        missing_target_counts=missing_target_counts,
        failed_checks=failed_checks,
        checks=checks,
        issues=issues,
    )


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
    metrics: dict[str, int],
    threshold: int,
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
    metrics: dict[str, int],
    threshold: int,
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

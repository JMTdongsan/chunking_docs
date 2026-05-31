from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import RetrievalCaseResult, RetrievalEvaluation


class RetrievalDiagnosticRow(BaseModel):
    query: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    expected_targets: list[str] = Field(default_factory=list)
    matched_targets: list[str] = Field(default_factory=list)
    missing_targets: list[str] = Field(default_factory=list)
    target_coverage_at_k: float = 0.0
    precision_at_k: float = 0.0
    matched_rank: int | None = None
    top_pages: list[int] = Field(default_factory=list)
    top_chunk_ids: list[str] = Field(default_factory=list)
    top_sources: list[list[str]] = Field(default_factory=list)


class RetrievalDiagnosticsReport(BaseModel):
    case_count: int
    failed_count: int
    partial_count: int
    no_hit_count: int
    low_precision_count: int
    reason_counts: dict[str, int]
    missing_target_type_counts: dict[str, int]
    source_counts: dict[str, int]
    rows: list[RetrievalDiagnosticRow]


def analyze_retrieval_evaluation(
    evaluation: RetrievalEvaluation,
    precision_floor: float = 0.2,
    include_passed: bool = False,
) -> RetrievalDiagnosticsReport:
    all_rows = []
    rows = []
    reason_counter: Counter[str] = Counter()
    missing_type_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    for result in evaluation.results:
        for source_list in result.top_sources:
            source_counter.update(source_list)

        row = diagnostic_row(result, precision_floor=precision_floor)
        all_rows.append(row)
        if row.reasons or include_passed:
            rows.append(row)
        reason_counter.update(row.reasons)
        missing_type_counter.update(target_type(target) for target in row.missing_targets)

    return RetrievalDiagnosticsReport(
        case_count=evaluation.case_count,
        failed_count=sum(1 for result in evaluation.results if not result.passed),
        partial_count=sum(1 for row in all_rows if row.matched_targets and row.missing_targets),
        no_hit_count=sum(1 for result in evaluation.results if not result.top_chunk_ids),
        low_precision_count=sum(
            1
            for result in evaluation.results
            if result.top_chunk_ids and result.precision_at_k < precision_floor
        ),
        reason_counts=dict(sorted(reason_counter.items())),
        missing_target_type_counts=dict(sorted(missing_type_counter.items())),
        source_counts=dict(sorted(source_counter.items())),
        rows=rows,
    )


def diagnostic_row(
    result: RetrievalCaseResult,
    precision_floor: float = 0.2,
) -> RetrievalDiagnosticRow:
    expected_targets = sorted_targets(expected_result_targets(result))
    matched_targets = sorted_targets(matched_result_targets(result))
    missing_targets = sorted_targets(set(expected_targets) - set(matched_targets))
    reasons = diagnostic_reasons(
        result,
        matched_targets,
        missing_targets,
        precision_floor=precision_floor,
    )
    return RetrievalDiagnosticRow(
        query=result.query,
        passed=result.passed,
        reasons=reasons,
        expected_targets=expected_targets,
        matched_targets=matched_targets,
        missing_targets=missing_targets,
        target_coverage_at_k=result.target_coverage_at_k,
        precision_at_k=result.precision_at_k,
        matched_rank=result.matched_rank,
        top_pages=result.top_pages,
        top_chunk_ids=result.top_chunk_ids,
        top_sources=result.top_sources,
    )


def diagnostic_reasons(
    result: RetrievalCaseResult,
    matched_targets: list[str],
    missing_targets: list[str],
    precision_floor: float = 0.2,
) -> list[str]:
    reasons = []
    if not result.top_chunk_ids:
        reasons.append("no_hits")
    if missing_targets and not matched_targets:
        reasons.append("no_expected_target_retrieved")
    elif missing_targets:
        reasons.append("partial_target_coverage")
    if result.top_chunk_ids and result.precision_at_k < precision_floor:
        reasons.append("low_precision_at_k")
    for target in missing_targets:
        reason = f"missing_{target_type(target)}"
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def expected_result_targets(result: RetrievalCaseResult) -> set[str]:
    targets = {f"page:{page}" for page in result.expected_pages}
    targets.update(f"chunk:{chunk_id}" for chunk_id in result.expected_chunk_ids)
    targets.update(f"asset:{asset_id}" for asset_id in result.expected_asset_ids)
    targets.update(f"triple:{triple_id}" for triple_id in result.expected_triple_ids)
    return targets


def matched_result_targets(result: RetrievalCaseResult) -> set[str]:
    return {target for targets in result.top_matched_targets for target in targets}


def sorted_targets(targets) -> list[str]:
    order = {"page": 0, "chunk": 1, "asset": 2, "triple": 3}
    return sorted(targets, key=lambda target: (order.get(target_type(target), 99), target))


def target_type(target: str) -> str:
    return target.split(":", 1)[0]


def load_retrieval_evaluation(path: Path) -> RetrievalEvaluation:
    return RetrievalEvaluation.model_validate_json(path.read_text(encoding="utf-8"))

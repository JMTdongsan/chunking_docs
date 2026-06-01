from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from chunking_docs.evaluation.retrieval import (
    CASE_GROUP_METADATA_KEYS,
    RetrievalCaseResult,
    RetrievalEvaluation,
    metric_source_keys,
)


class RetrievalDiagnosticRow(BaseModel):
    query: str
    case_metadata: dict[str, object] = Field(default_factory=dict)
    case_groups: dict[str, str] = Field(default_factory=dict)
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    expected_targets: list[str] = Field(default_factory=list)
    matched_targets: list[str] = Field(default_factory=list)
    missing_targets: list[str] = Field(default_factory=list)
    excluded_targets: list[str] = Field(default_factory=list)
    matched_excluded_targets: list[str] = Field(default_factory=list)
    source_counts: dict[str, int] = Field(default_factory=dict)
    source_family_counts: dict[str, int] = Field(default_factory=dict)
    matched_source_counts: dict[str, int] = Field(default_factory=dict)
    matched_source_family_counts: dict[str, int] = Field(default_factory=dict)
    source_match_rates: dict[str, float] = Field(default_factory=dict)
    source_family_match_rates: dict[str, float] = Field(default_factory=dict)
    excluded_source_counts: dict[str, int] = Field(default_factory=dict)
    excluded_source_family_counts: dict[str, int] = Field(default_factory=dict)
    top_excluded_sources: list[list[str]] = Field(default_factory=list)
    top_source_families: list[list[str]] = Field(default_factory=list)
    target_coverage_at_k: float = 0.0
    target_ndcg_at_k: float = 0.0
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
    low_target_ndcg_count: int
    reason_counts: dict[str, int]
    missing_target_type_counts: dict[str, int]
    reason_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)
    missing_target_type_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    source_counts: dict[str, int]
    source_family_counts: dict[str, int] = Field(default_factory=dict)
    source_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    source_family_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    matched_source_counts: dict[str, int] = Field(default_factory=dict)
    matched_source_family_counts: dict[str, int] = Field(default_factory=dict)
    matched_source_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    matched_source_family_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    source_match_rates: dict[str, float] = Field(default_factory=dict)
    source_family_match_rates: dict[str, float] = Field(default_factory=dict)
    source_match_rates_by_case_group: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict
    )
    source_family_match_rates_by_case_group: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict
    )
    excluded_source_counts: dict[str, int] = Field(default_factory=dict)
    excluded_source_family_counts: dict[str, int] = Field(default_factory=dict)
    excluded_source_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    excluded_source_family_counts_by_case_group: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    rows: list[RetrievalDiagnosticRow]


def analyze_retrieval_evaluation(
    evaluation: RetrievalEvaluation,
    precision_floor: float = 0.2,
    target_ndcg_floor: float = 0.7,
    include_passed: bool = False,
) -> RetrievalDiagnosticsReport:
    all_rows = []
    rows = []
    reason_counter: Counter[str] = Counter()
    missing_type_counter: Counter[str] = Counter()
    group_reason_counters: dict[str, dict[str, Counter[str]]] = {}
    group_missing_type_counters: dict[str, dict[str, Counter[str]]] = {}
    source_counter: Counter[str] = Counter()
    source_family_counter: Counter[str] = Counter()
    group_source_counters: dict[str, dict[str, Counter[str]]] = {}
    group_source_family_counters: dict[str, dict[str, Counter[str]]] = {}
    matched_source_counter: Counter[str] = Counter()
    matched_source_family_counter: Counter[str] = Counter()
    group_matched_source_counters: dict[str, dict[str, Counter[str]]] = {}
    group_matched_source_family_counters: dict[str, dict[str, Counter[str]]] = {}
    excluded_source_counter: Counter[str] = Counter()
    excluded_source_family_counter: Counter[str] = Counter()
    group_excluded_source_counters: dict[str, dict[str, Counter[str]]] = {}
    group_excluded_source_family_counters: dict[str, dict[str, Counter[str]]] = {}

    for result in evaluation.results:
        row = diagnostic_row(
            result,
            precision_floor=precision_floor,
            target_ndcg_floor=target_ndcg_floor,
        )
        all_rows.append(row)
        if row.reasons or include_passed:
            rows.append(row)
        reason_counter.update(row.reasons)
        missing_types = [target_type(target) for target in row.missing_targets]
        missing_type_counter.update(missing_types)
        source_counter.update(row.source_counts)
        source_family_counter.update(row.source_family_counts)
        matched_source_counter.update(row.matched_source_counts)
        matched_source_family_counter.update(row.matched_source_family_counts)
        excluded_source_counter.update(row.excluded_source_counts)
        excluded_source_family_counter.update(row.excluded_source_family_counts)
        update_case_group_counters(group_reason_counters, row.case_groups, row.reasons)
        update_case_group_counters(group_missing_type_counters, row.case_groups, missing_types)
        update_case_group_counter_mapping(
            group_source_counters,
            row.case_groups,
            row.source_counts,
        )
        update_case_group_counter_mapping(
            group_source_family_counters,
            row.case_groups,
            row.source_family_counts,
        )
        update_case_group_counter_mapping(
            group_matched_source_counters,
            row.case_groups,
            row.matched_source_counts,
        )
        update_case_group_counter_mapping(
            group_matched_source_family_counters,
            row.case_groups,
            row.matched_source_family_counts,
        )
        update_case_group_counter_mapping(
            group_excluded_source_counters,
            row.case_groups,
            row.excluded_source_counts,
        )
        update_case_group_counter_mapping(
            group_excluded_source_family_counters,
            row.case_groups,
            row.excluded_source_family_counts,
        )

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
        low_target_ndcg_count=sum(
            1
            for result in evaluation.results
            if result.expected_target_count
            and result.top_chunk_ids
            and result.target_ndcg_at_k < target_ndcg_floor
        ),
        reason_counts=dict(sorted(reason_counter.items())),
        missing_target_type_counts=dict(sorted(missing_type_counter.items())),
        reason_counts_by_case_group=serializable_group_counters(group_reason_counters),
        missing_target_type_counts_by_case_group=serializable_group_counters(
            group_missing_type_counters
        ),
        source_counts=dict(sorted(source_counter.items())),
        source_family_counts=dict(sorted(source_family_counter.items())),
        source_counts_by_case_group=serializable_group_counters(group_source_counters),
        source_family_counts_by_case_group=serializable_group_counters(
            group_source_family_counters
        ),
        matched_source_counts=dict(sorted(matched_source_counter.items())),
        matched_source_family_counts=dict(sorted(matched_source_family_counter.items())),
        matched_source_counts_by_case_group=serializable_group_counters(
            group_matched_source_counters
        ),
        matched_source_family_counts_by_case_group=serializable_group_counters(
            group_matched_source_family_counters
        ),
        source_match_rates=source_match_rates(matched_source_counter, source_counter),
        source_family_match_rates=source_match_rates(
            matched_source_family_counter,
            source_family_counter,
        ),
        source_match_rates_by_case_group=source_match_rates_by_case_group(
            group_matched_source_counters,
            group_source_counters,
        ),
        source_family_match_rates_by_case_group=source_match_rates_by_case_group(
            group_matched_source_family_counters,
            group_source_family_counters,
        ),
        excluded_source_counts=dict(sorted(excluded_source_counter.items())),
        excluded_source_family_counts=dict(sorted(excluded_source_family_counter.items())),
        excluded_source_counts_by_case_group=serializable_group_counters(
            group_excluded_source_counters
        ),
        excluded_source_family_counts_by_case_group=serializable_group_counters(
            group_excluded_source_family_counters
        ),
        rows=rows,
    )


def diagnostic_row(
    result: RetrievalCaseResult,
    precision_floor: float = 0.2,
    target_ndcg_floor: float = 0.7,
) -> RetrievalDiagnosticRow:
    expected_targets = sorted_targets(expected_result_targets(result))
    matched_targets = sorted_targets(matched_result_targets(result))
    missing_targets = sorted_targets(set(expected_targets) - set(matched_targets))
    excluded_targets = sorted_targets(excluded_result_targets(result))
    matched_excluded_targets = sorted_targets(matched_excluded_result_targets(result))
    source_counts = sources_for_result(result, family=False)
    source_family_counts = sources_for_result(result, family=True)
    matched_source_counts = matched_sources_for_result(result, family=False)
    matched_source_family_counts = matched_sources_for_result(result, family=True)
    excluded_source_counts = excluded_sources_for_result(result, family=False)
    excluded_source_family_counts = excluded_sources_for_result(result, family=True)
    reasons = diagnostic_reasons(
        result,
        matched_targets,
        missing_targets,
        matched_excluded_targets,
        precision_floor=precision_floor,
        target_ndcg_floor=target_ndcg_floor,
    )
    return RetrievalDiagnosticRow(
        query=result.query,
        case_metadata=result.case_metadata,
        case_groups=case_groups(result.case_metadata),
        passed=result.passed,
        reasons=reasons,
        expected_targets=expected_targets,
        matched_targets=matched_targets,
        missing_targets=missing_targets,
        excluded_targets=excluded_targets,
        matched_excluded_targets=matched_excluded_targets,
        source_counts=source_counts,
        source_family_counts=source_family_counts,
        matched_source_counts=matched_source_counts,
        matched_source_family_counts=matched_source_family_counts,
        source_match_rates=source_match_rates(matched_source_counts, source_counts),
        source_family_match_rates=source_match_rates(
            matched_source_family_counts,
            source_family_counts,
        ),
        excluded_source_counts=excluded_source_counts,
        excluded_source_family_counts=excluded_source_family_counts,
        top_excluded_sources=top_excluded_sources(result),
        top_source_families=top_source_families(result),
        target_coverage_at_k=result.target_coverage_at_k,
        target_ndcg_at_k=result.target_ndcg_at_k,
        precision_at_k=result.precision_at_k,
        matched_rank=result.matched_rank,
        top_pages=result.top_pages,
        top_chunk_ids=result.top_chunk_ids,
        top_sources=result.top_sources,
    )


def case_groups(metadata: dict[str, object]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for key in CASE_GROUP_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            groups[key] = text
    return groups


def update_case_group_counters(
    counters: dict[str, dict[str, Counter[str]]],
    groups: dict[str, str],
    values: list[str],
) -> None:
    if not values:
        return
    for group_name, group_value in groups.items():
        counters.setdefault(group_name, {}).setdefault(group_value, Counter()).update(values)


def update_case_group_counter_mapping(
    counters: dict[str, dict[str, Counter[str]]],
    groups: dict[str, str],
    values: dict[str, int],
) -> None:
    if not values:
        return
    for group_name, group_value in groups.items():
        counters.setdefault(group_name, {}).setdefault(group_value, Counter()).update(values)


def serializable_group_counters(
    counters: dict[str, dict[str, Counter[str]]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        group_name: {
            group_value: dict(sorted(counter.items()))
            for group_value, counter in sorted(group_values.items())
        }
        for group_name, group_values in sorted(counters.items())
    }


def source_match_rates(
    matched_counts: dict[str, int] | Counter[str],
    source_counts: dict[str, int] | Counter[str],
) -> dict[str, float]:
    rates = {}
    for source, source_count in sorted(source_counts.items()):
        if source_count <= 0:
            continue
        rates[source] = float(matched_counts.get(source, 0)) / float(source_count)
    return rates


def source_match_rates_by_case_group(
    matched_counters: dict[str, dict[str, Counter[str]]],
    source_counters: dict[str, dict[str, Counter[str]]],
) -> dict[str, dict[str, dict[str, float]]]:
    rates: dict[str, dict[str, dict[str, float]]] = {}
    for group_name, group_values in sorted(source_counters.items()):
        for group_value, source_counts in sorted(group_values.items()):
            matched_counts = matched_counters.get(group_name, {}).get(group_value, Counter())
            group_rates = source_match_rates(matched_counts, source_counts)
            if group_rates:
                rates.setdefault(group_name, {})[group_value] = group_rates
    return rates


def diagnostic_reasons(
    result: RetrievalCaseResult,
    matched_targets: list[str],
    missing_targets: list[str],
    matched_excluded_targets: list[str],
    precision_floor: float = 0.2,
    target_ndcg_floor: float = 0.7,
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
    if (
        result.expected_target_count
        and result.top_chunk_ids
        and result.target_ndcg_at_k < target_ndcg_floor
    ):
        reasons.append("low_target_ndcg_at_k")
    if matched_excluded_targets:
        reasons.append("excluded_target_retrieved")
        for target in matched_excluded_targets:
            reason = f"excluded_{target_type(target)}_hit"
            if reason not in reasons:
                reasons.append(reason)
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


def excluded_result_targets(result: RetrievalCaseResult) -> set[str]:
    targets = {f"page:{page}" for page in result.excluded_pages}
    targets.update(f"chunk:{chunk_id}" for chunk_id in result.excluded_chunk_ids)
    targets.update(f"asset:{asset_id}" for asset_id in result.excluded_asset_ids)
    targets.update(f"triple:{triple_id}" for triple_id in result.excluded_triple_ids)
    return targets


def matched_excluded_result_targets(result: RetrievalCaseResult) -> set[str]:
    return {target for targets in result.top_excluded_targets for target in targets}


def excluded_sources_for_result(
    result: RetrievalCaseResult,
    family: bool = False,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index, targets in enumerate(result.top_excluded_targets):
        if not targets:
            continue
        sources = result.top_sources[index] if index < len(result.top_sources) else []
        target_count = len(set(targets))
        for source in sorted(metric_source_keys(sources, family=family)):
            counts[source] += target_count
    return dict(sorted(counts.items()))


def sources_for_result(
    result: RetrievalCaseResult,
    family: bool = False,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sources in result.top_sources:
        counts.update(metric_source_keys(sources, family=family))
    return dict(sorted(counts.items()))


def matched_sources_for_result(
    result: RetrievalCaseResult,
    family: bool = False,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index, targets in enumerate(result.top_matched_targets):
        if not targets:
            continue
        sources = result.top_sources[index] if index < len(result.top_sources) else []
        target_count = len(set(targets))
        for source in sorted(metric_source_keys(sources, family=family)):
            counts[source] += target_count
    return dict(sorted(counts.items()))


def top_excluded_sources(result: RetrievalCaseResult) -> list[list[str]]:
    values = []
    for index, targets in enumerate(result.top_excluded_targets):
        if not targets:
            values.append([])
            continue
        sources = result.top_sources[index] if index < len(result.top_sources) else []
        values.append(sorted(metric_source_keys(sources)))
    return values


def top_source_families(result: RetrievalCaseResult) -> list[list[str]]:
    return [sorted(metric_source_keys(sources, family=True)) for sources in result.top_sources]


def sorted_targets(targets) -> list[str]:
    order = {"page": 0, "chunk": 1, "asset": 2, "triple": 3}
    return sorted(targets, key=lambda target: (order.get(target_type(target), 99), target))


def target_type(target: str) -> str:
    return target.split(":", 1)[0]


def load_retrieval_evaluation(path: Path) -> RetrievalEvaluation:
    return RetrievalEvaluation.model_validate_json(path.read_text(encoding="utf-8"))

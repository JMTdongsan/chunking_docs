from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.chunking.multimodal import ChunkStrategy, build_strategy_chunks
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.chunking_quality import ChunkingQualityReport, evaluate_chunking_quality
from chunking_docs.evaluation.compare import ChunkingComparison, compare_chunking_reports
from chunking_docs.evaluation.gate import (
    case_group_metric_key,
    parse_case_group_spec,
    retrieval_case_group_metrics,
    retrieval_rank_metrics,
    retrieval_source_family_metrics,
    retrieval_target_metrics,
    source_family_metric_key,
    target_type_metric_key,
)
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.io import write_jsonl
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, VisualAsset


class ChunkingSweepCandidate(BaseModel):
    name: str
    strategy: str
    config: dict[str, Any] = Field(default_factory=dict)
    chunks_file: str | None = None
    chunk_count: int
    report: ChunkingQualityReport


class ChunkingSweepSelectionRow(BaseModel):
    name: str
    score: float
    pareto_efficient: bool = False
    eligible: bool = True
    failed_constraints: list[str] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class ChunkingSweepSelection(BaseModel):
    recommended: str | None = None
    pareto_front: list[str] = Field(default_factory=list)
    eligible_pareto_front: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    constraints: dict[str, float] = Field(default_factory=dict)
    eligible_count: int = 0
    rejected_count: int = 0
    ranking: list[ChunkingSweepSelectionRow] = Field(default_factory=list)


class ChunkingSweepReport(BaseModel):
    generated_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    candidates: list[ChunkingSweepCandidate]
    comparison: ChunkingComparison
    selection: ChunkingSweepSelection = Field(default_factory=ChunkingSweepSelection)


SWEEP_SELECTION_WEIGHTS = {
    "retrieval_recall_at_k": 0.22,
    "target_coverage_at_k": 0.22,
    "target_ndcg_at_k": 0.16,
    "target_rank_efficiency": 0.12,
    "precision_at_k": 0.09,
    "quality_score": 0.09,
    "visual_text_coverage_ratio": 0.05,
    "latency_efficiency": 0.03,
    "chunk_count_efficiency": 0.02,
}

PARETO_HIGHER_IS_BETTER = [
    "retrieval_recall_at_k",
    "target_coverage_at_k",
    "target_ndcg_at_k",
    "precision_at_k",
    "quality_score",
    "visual_text_coverage_ratio",
    "target_rank_efficiency",
]

PARETO_LOWER_IS_BETTER = [
    "mean_target_rank",
    "p95_target_rank",
    "mean_latency_ms",
    "chunk_count",
    "total_chunk_chars",
    "mean_chunk_chars",
    "p95_chunk_chars",
    "embedding_text_kchars",
    "standalone_visual_chunk_count",
]

MIN_SELECTION_CONSTRAINTS = {
    "min_retrieval_recall_at_k": "retrieval_recall_at_k",
    "min_target_coverage_at_k": "target_coverage_at_k",
    "min_target_ndcg_at_k": "target_ndcg_at_k",
    "min_precision_at_k": "precision_at_k",
    "min_quality_score": "quality_score",
    "min_visual_text_coverage_ratio": "visual_text_coverage_ratio",
}

MAX_SELECTION_CONSTRAINTS = {
    "max_mean_target_rank": "mean_target_rank",
    "max_p95_target_rank": "p95_target_rank",
    "max_mean_latency_ms": "mean_latency_ms",
    "max_chunk_count": "chunk_count",
    "max_total_chunk_chars": "total_chunk_chars",
    "max_mean_chunk_chars": "mean_chunk_chars",
    "max_p95_chunk_chars": "p95_chunk_chars",
    "max_embedding_text_kchars": "embedding_text_kchars",
    "max_standalone_visual_chunk_count": "standalone_visual_chunk_count",
}

DYNAMIC_MIN_SELECTION_PREFIXES = {
    "min_target_type_coverage": "target_type",
    "min_source_family_target_coverage": "source_family",
    "min_case_group_target_coverage": "case_group",
}


def run_chunking_sweep(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
    profiles: list[PageProfile],
    triples: list[GraphTriple],
    strategies: list[ChunkStrategy],
    max_chars_values: list[int],
    overlap_chars_values: list[int],
    min_chars: int = 180,
    parent_max_chars_values: list[int] | None = None,
    visual_context_chars_values: list[int] | None = None,
    retrieval_cases: list[RetrievalCase] | None = None,
    top_k: int = 5,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = True,
    retrieval_repeat: int = 1,
    fusion_weights: dict[str, float] | None = None,
    selection_constraints: dict[str, float | None] | None = None,
    output_dir: Path | None = None,
    write_candidates: bool = True,
) -> ChunkingSweepReport:
    reports: dict[str, ChunkingQualityReport] = {}
    candidates: list[ChunkingSweepCandidate] = []

    for config in sweep_configs(
        strategies=strategies,
        max_chars_values=max_chars_values,
        overlap_chars_values=overlap_chars_values,
        min_chars=min_chars,
        parent_max_chars_values=parent_max_chars_values or [900],
        visual_context_chars_values=visual_context_chars_values or [700],
    ):
        candidate_chunks = build_strategy_chunks(chunks, assets, **config)
        name = candidate_name(config)
        chunks_file = None
        if output_dir is not None and write_candidates:
            output_path = output_dir / f"chunks.{name}.jsonl"
            write_jsonl(output_path, candidate_chunks)
            chunks_file = str(output_path)

        report = evaluate_chunking_quality(
            chunks=candidate_chunks,
            profiles=profiles,
            assets=assets,
            triples=triples,
            retrieval_cases=retrieval_cases,
            top_k=top_k,
            min_chars=min_chars,
            max_chars=int(config["max_chars"]),
            tokenizer_config=tokenizer_config,
            collapse_hierarchical=collapse_hierarchical,
            retrieval_repeat=retrieval_repeat,
            fusion_weights=fusion_weights,
        )
        reports[name] = report
        candidates.append(
            ChunkingSweepCandidate(
                name=name,
                strategy=str(config["strategy"]),
                config=serializable_config(config),
                chunks_file=chunks_file,
                chunk_count=len(candidate_chunks),
                report=report,
            )
        )

    comparison = compare_chunking_reports(reports)
    selection = build_sweep_selection(candidates, selection_constraints=selection_constraints)
    rank_by_candidate = {row.name: index for index, row in enumerate(selection.ranking)}
    candidates.sort(key=lambda candidate: rank_by_candidate.get(candidate.name, len(candidates)))
    return ChunkingSweepReport(
        generated_at=datetime.now(UTC).isoformat(),
        config={
            "top_k": top_k,
            "min_chars": min_chars,
            "collapse_hierarchical": collapse_hierarchical,
            "retrieval_repeat": retrieval_repeat,
            "tokenizer": tokenizer_config.model_dump() if tokenizer_config else None,
            "fusion_weights": fusion_weights or {},
            "selection_constraints": selection.constraints,
        },
        candidates=candidates,
        comparison=comparison,
        selection=selection,
    )


def build_sweep_selection(
    candidates: list[ChunkingSweepCandidate],
    selection_constraints: dict[str, float | None] | None = None,
) -> ChunkingSweepSelection:
    constraints = normalize_selection_constraints(selection_constraints or {})
    if not candidates:
        return ChunkingSweepSelection(
            weights=SWEEP_SELECTION_WEIGHTS,
            constraints=constraints,
        )
    metrics_by_name = {candidate.name: selection_metrics(candidate) for candidate in candidates}
    pareto_front = [
        candidate.name
        for candidate in candidates
        if is_pareto_efficient(candidate.name, metrics_by_name)
    ]
    ranking = [
        ChunkingSweepSelectionRow(
            name=name,
            score=selection_score(metrics),
            pareto_efficient=name in pareto_front,
            failed_constraints=selection_constraint_failures(metrics, constraints),
            metrics=metrics,
        )
        for name, metrics in metrics_by_name.items()
    ]
    for row in ranking:
        row.eligible = not row.failed_constraints
    ranking.sort(
        key=lambda row: (
            row.eligible,
            row.pareto_efficient,
            row.score,
            row.metrics.get("retrieval_recall_at_k") or 0.0,
            row.metrics.get("target_coverage_at_k") or 0.0,
            row.metrics.get("target_ndcg_at_k") or 0.0,
            row.metrics.get("target_rank_efficiency") or 0.0,
        ),
        reverse=True,
    )
    recommended = None
    for row in ranking:
        if row.eligible:
            recommended = row.name
            break
    return ChunkingSweepSelection(
        recommended=recommended,
        pareto_front=[row.name for row in ranking if row.pareto_efficient],
        eligible_pareto_front=[
            row.name for row in ranking if row.pareto_efficient and row.eligible
        ],
        weights=SWEEP_SELECTION_WEIGHTS,
        constraints=constraints,
        eligible_count=sum(1 for row in ranking if row.eligible),
        rejected_count=sum(1 for row in ranking if not row.eligible),
        ranking=ranking,
    )


def normalize_selection_constraints(
    values: dict[str, float | None],
) -> dict[str, float]:
    supported = set(MIN_SELECTION_CONSTRAINTS) | set(MAX_SELECTION_CONSTRAINTS)
    constraints = {}
    for name, value in values.items():
        if value is None:
            continue
        normalized_name = normalize_selection_constraint_name(name)
        if normalized_name not in supported and dynamic_constraint_metric_name(normalized_name) is None:
            raise ValueError(f"Unsupported sweep selection constraint: {name}")
        constraints[normalized_name] = float(value)
    return constraints


def normalize_selection_constraint_name(name: str) -> str:
    normalized = name.strip()
    if ":" not in normalized:
        return normalized
    prefix, value = normalized.split(":", 1)
    prefix = prefix.strip()
    value = value.strip()
    if prefix == "min_case_group_target_coverage":
        group_name, group_value = parse_case_group_spec(value)
        return f"{prefix}:{group_name}:{group_value}"
    if prefix in DYNAMIC_MIN_SELECTION_PREFIXES:
        return f"{prefix}:{value.lower()}"
    return normalized


def dynamic_constraint_metric_name(name: str) -> str | None:
    if ":" not in name:
        return None
    prefix, value = name.split(":", 1)
    if prefix == "min_target_type_coverage" and value:
        return target_type_metric_key(value, "coverage_at_k")
    if prefix == "min_source_family_target_coverage" and value:
        return source_family_metric_key(value, "target_coverage_at_k")
    if prefix == "min_case_group_target_coverage" and value:
        group_name, group_value = parse_case_group_spec(value)
        return case_group_metric_key(group_name, group_value, "target_coverage_at_k")
    return None


def selection_constraint_failures(
    metrics: dict[str, float | None],
    constraints: dict[str, float],
) -> list[str]:
    failures = []
    for constraint_name, metric_name in MIN_SELECTION_CONSTRAINTS.items():
        if constraint_name not in constraints:
            continue
        value = metrics.get(metric_name)
        if value is None or float(value) < constraints[constraint_name]:
            failures.append(constraint_name)
    for constraint_name, metric_name in MAX_SELECTION_CONSTRAINTS.items():
        if constraint_name not in constraints:
            continue
        value = metrics.get(metric_name)
        if value is None or float(value) > constraints[constraint_name]:
            failures.append(constraint_name)
    handled = set(MIN_SELECTION_CONSTRAINTS) | set(MAX_SELECTION_CONSTRAINTS)
    for constraint_name, threshold in constraints.items():
        if constraint_name in handled:
            continue
        metric_name = dynamic_constraint_metric_name(constraint_name)
        if metric_name is None:
            continue
        value = metrics.get(metric_name)
        if value is None or float(value) < threshold:
            failures.append(constraint_name)
    return failures


def selection_metrics(candidate: ChunkingSweepCandidate) -> dict[str, float | None]:
    retrieval = candidate.report.retrieval
    mean_latency_ms = retrieval.mean_latency_ms if retrieval else None
    rank_metrics = retrieval_rank_metrics(retrieval) if retrieval else {}
    mean_target_rank = rank_metrics.get("mean_target_rank")
    p95_target_rank = rank_metrics.get("p95_target_rank")
    total_chunk_chars = candidate.report.char_count.count * candidate.report.char_count.mean
    metrics = {
        "retrieval_recall_at_k": retrieval.recall_at_k if retrieval else None,
        "target_coverage_at_k": retrieval.target_coverage_at_k if retrieval else None,
        "target_ndcg_at_k": retrieval.mean_target_ndcg_at_k if retrieval else None,
        "precision_at_k": retrieval.mean_precision_at_k if retrieval else None,
        "mean_first_relevant_rank": rank_metrics.get("mean_first_relevant_rank"),
        "p95_first_relevant_rank": rank_metrics.get("p95_first_relevant_rank"),
        "mean_target_rank": mean_target_rank,
        "p95_target_rank": p95_target_rank,
        "target_rank_efficiency": rank_efficiency(mean_target_rank),
        "mean_latency_ms": mean_latency_ms,
        "latency_efficiency": latency_efficiency(mean_latency_ms),
        "quality_score": candidate.report.quality_score,
        "visual_text_coverage_ratio": candidate.report.visual_text_coverage_ratio,
        "chunk_count": float(candidate.chunk_count),
        "chunk_count_efficiency": 1.0 / candidate.chunk_count if candidate.chunk_count > 0 else 0.0,
        "total_chunk_chars": total_chunk_chars,
        "mean_chunk_chars": candidate.report.char_count.mean,
        "p95_chunk_chars": candidate.report.char_count.p95,
        "embedding_text_kchars": total_chunk_chars / 1000.0,
        "standalone_visual_chunk_count": float(candidate.report.standalone_visual_chunk_count),
    }
    add_retrieval_breakdown_metrics(metrics, retrieval)
    return metrics


def add_retrieval_breakdown_metrics(
    metrics: dict[str, float | None],
    retrieval,
) -> None:
    if retrieval is None:
        return
    for target_type, values in retrieval_target_metrics(retrieval).items():
        for metric_name, value in values.items():
            metrics[target_type_metric_key(target_type, metric_name)] = value
    for family, values in retrieval_source_family_metrics(retrieval).items():
        for metric_name, value in values.items():
            metrics[source_family_metric_key(family, metric_name)] = value
    for group_name, group_values in retrieval_case_group_metrics(retrieval).items():
        for group_value, values in group_values.items():
            for metric_name, value in values.items():
                metrics[case_group_metric_key(group_name, group_value, metric_name)] = value


def selection_score(metrics: dict[str, float | None]) -> float:
    return sum(
        clamp01(metrics.get(metric_name)) * weight
        for metric_name, weight in SWEEP_SELECTION_WEIGHTS.items()
    )


def is_pareto_efficient(
    candidate_name: str,
    metrics_by_name: dict[str, dict[str, float | None]],
) -> bool:
    candidate = metrics_by_name[candidate_name]
    for other_name, other in metrics_by_name.items():
        if other_name == candidate_name:
            continue
        if dominates(other, candidate):
            return False
    return True


def dominates(
    candidate: dict[str, float | None],
    baseline: dict[str, float | None],
) -> bool:
    no_worse = all(
        metric_value(candidate, metric) >= metric_value(baseline, metric)
        for metric in PARETO_HIGHER_IS_BETTER
    ) and all(
        cost_value(candidate, metric) <= cost_value(baseline, metric)
        for metric in PARETO_LOWER_IS_BETTER
    )
    strictly_better = any(
        metric_value(candidate, metric) > metric_value(baseline, metric)
        for metric in PARETO_HIGHER_IS_BETTER
    ) or any(
        cost_value(candidate, metric) < cost_value(baseline, metric)
        for metric in PARETO_LOWER_IS_BETTER
    )
    return no_worse and strictly_better


def metric_value(metrics: dict[str, float | None], name: str) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else 0.0


def cost_value(metrics: dict[str, float | None], name: str) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else float("inf")


def latency_efficiency(latency_ms: float | None) -> float:
    if latency_ms is None:
        return 0.0
    if latency_ms <= 0:
        return 1.0
    return 1.0 / latency_ms


def rank_efficiency(rank: float | None) -> float:
    if rank is None:
        return 0.0
    if rank <= 0:
        return 1.0
    return 1.0 / rank


def clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def sweep_configs(
    strategies: list[ChunkStrategy],
    max_chars_values: list[int],
    overlap_chars_values: list[int],
    min_chars: int,
    parent_max_chars_values: list[int],
    visual_context_chars_values: list[int],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    seen = set()
    for strategy in strategies:
        if strategy == "page":
            config = {
                "strategy": strategy,
                "max_chars": max_chars_values[0],
                "overlap_chars": overlap_chars_values[0],
                "min_chars": min_chars,
            }
            add_unique_config(configs, seen, config)
            continue
        for max_chars in max_chars_values:
            for overlap_chars in overlap_chars_values:
                base = {
                    "strategy": strategy,
                    "max_chars": max_chars,
                    "overlap_chars": overlap_chars,
                    "min_chars": min_chars,
                }
                if strategy == "multimodal":
                    for visual_context_chars in visual_context_chars_values:
                        add_unique_config(
                            configs,
                            seen,
                            {
                                **base,
                                "visual_context_chars": visual_context_chars,
                            },
                        )
                    continue
                if strategy != "hierarchical":
                    add_unique_config(configs, seen, base)
                    continue
                for parent_max_chars in parent_max_chars_values:
                    for visual_context_chars in visual_context_chars_values:
                        add_unique_config(
                            configs,
                            seen,
                            {
                                **base,
                                "parent_max_chars": parent_max_chars,
                                "visual_context_chars": visual_context_chars,
                            },
                        )
    return configs


def add_unique_config(configs: list[dict[str, Any]], seen: set[tuple[tuple[str, Any], ...]], config):
    key = tuple(sorted(config.items()))
    if key in seen:
        return
    seen.add(key)
    configs.append(config)


def candidate_name(config: dict[str, Any]) -> str:
    strategy = str(config["strategy"])
    parts = [
        strategy,
        f"max{config['max_chars']}",
        f"ov{config['overlap_chars']}",
        f"min{config['min_chars']}",
    ]
    if strategy == "hierarchical":
        parts.extend(
            [
                f"parent{config.get('parent_max_chars')}",
                f"visual{config.get('visual_context_chars')}",
            ]
        )
    elif strategy == "multimodal" and "visual_context_chars" in config:
        parts.append(f"visual{config.get('visual_context_chars')}")
    return safe_name("-".join(parts))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if key == "strategy" else value for key, value in config.items()}

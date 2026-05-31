from __future__ import annotations

from pydantic import BaseModel, Field

from chunking_docs.evaluation.chunking_quality import ChunkingQualityReport
from chunking_docs.evaluation.gate import (
    retrieval_source_family_metrics,
    retrieval_target_metrics,
)


class ChunkingComparisonRow(BaseModel):
    name: str
    chunk_count: int
    quality_score: float
    retrieval_hit_rate: float | None
    retrieval_recall_at_k: float | None
    retrieval_mrr: float | None
    retrieval_target_coverage_at_k: float | None
    retrieval_mean_target_ndcg_at_k: float | None
    retrieval_mean_precision_at_k: float | None
    retrieval_mean_latency_ms: float | None
    retrieval_p95_latency_ms: float | None
    target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    failed_queries: list[str]
    page_coverage_ratio: float
    visual_annotation_ratio: float
    visual_text_asset_count: int = 0
    visual_text_covered_asset_count: int = 0
    visual_text_coverage_ratio: float = 1.0
    chunks_under_min_chars: int
    chunks_over_max_chars: int
    issue_codes: list[str]


class ChunkingComparison(BaseModel):
    rows: list[ChunkingComparisonRow]
    best_by_quality: str | None
    best_by_retrieval: str | None
    fastest_by_mean_latency: str | None


def compare_chunking_reports(
    reports: dict[str, ChunkingQualityReport],
) -> ChunkingComparison:
    rows = []
    for name, report in reports.items():
        target_metrics = retrieval_target_metrics(report.retrieval)
        source_family_metrics = retrieval_source_family_metrics(report.retrieval)
        rows.append(
            ChunkingComparisonRow(
                name=name,
                chunk_count=report.chunk_count,
                quality_score=report.quality_score,
                retrieval_hit_rate=report.retrieval.hit_rate if report.retrieval else None,
                retrieval_recall_at_k=report.retrieval.recall_at_k if report.retrieval else None,
                retrieval_mrr=report.retrieval.mrr if report.retrieval else None,
                retrieval_target_coverage_at_k=report.retrieval.target_coverage_at_k
                if report.retrieval
                else None,
                retrieval_mean_target_ndcg_at_k=report.retrieval.mean_target_ndcg_at_k
                if report.retrieval
                else None,
                retrieval_mean_precision_at_k=report.retrieval.mean_precision_at_k
                if report.retrieval
                else None,
                retrieval_mean_latency_ms=report.retrieval.mean_latency_ms if report.retrieval else None,
                retrieval_p95_latency_ms=report.retrieval.p95_latency_ms if report.retrieval else None,
                target_metrics=target_metrics,
                source_family_metrics=source_family_metrics,
                failed_queries=report.retrieval.failed_queries if report.retrieval else [],
                page_coverage_ratio=report.page_coverage_ratio,
                visual_annotation_ratio=report.visual_annotation_ratio,
                visual_text_asset_count=report.visual_text_asset_count,
                visual_text_covered_asset_count=report.visual_text_covered_asset_count,
                visual_text_coverage_ratio=report.visual_text_coverage_ratio,
                chunks_under_min_chars=report.chunks_under_min_chars,
                chunks_over_max_chars=report.chunks_over_max_chars,
                issue_codes=[issue.code for issue in report.issues],
            )
        )
    rows.sort(
        key=lambda row: (
            row.retrieval_recall_at_k if row.retrieval_recall_at_k is not None else -1.0,
            row.retrieval_target_coverage_at_k
            if row.retrieval_target_coverage_at_k is not None
            else -1.0,
            row.retrieval_mean_target_ndcg_at_k
            if row.retrieval_mean_target_ndcg_at_k is not None
            else -1.0,
            row.retrieval_mrr if row.retrieval_mrr is not None else -1.0,
            row.quality_score,
        ),
        reverse=True,
    )
    best_by_quality = max(rows, key=lambda row: row.quality_score).name if rows else None
    retrieval_rows = [row for row in rows if row.retrieval_recall_at_k is not None]
    best_by_retrieval = (
        max(
            retrieval_rows,
            key=lambda row: (
                row.retrieval_recall_at_k or 0.0,
                row.retrieval_target_coverage_at_k or 0.0,
                row.retrieval_mean_target_ndcg_at_k or 0.0,
                row.retrieval_mrr or 0.0,
            ),
        ).name
        if retrieval_rows
        else None
    )
    latency_rows = [row for row in rows if row.retrieval_mean_latency_ms is not None]
    return ChunkingComparison(
        rows=rows,
        best_by_quality=best_by_quality,
        best_by_retrieval=best_by_retrieval,
        fastest_by_mean_latency=min(latency_rows, key=lambda row: row.retrieval_mean_latency_ms or 0.0).name
        if latency_rows
        else None,
    )

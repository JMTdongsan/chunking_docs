from __future__ import annotations

from pydantic import BaseModel

from chunking_docs.evaluation.chunking_quality import ChunkingQualityReport


class ChunkingComparisonRow(BaseModel):
    name: str
    chunk_count: int
    quality_score: float
    retrieval_hit_rate: float | None
    page_coverage_ratio: float
    visual_annotation_ratio: float
    chunks_under_min_chars: int
    chunks_over_max_chars: int
    issue_codes: list[str]


class ChunkingComparison(BaseModel):
    rows: list[ChunkingComparisonRow]
    best_by_quality: str | None
    best_by_retrieval: str | None


def compare_chunking_reports(
    reports: dict[str, ChunkingQualityReport],
) -> ChunkingComparison:
    rows = [
        ChunkingComparisonRow(
            name=name,
            chunk_count=report.chunk_count,
            quality_score=report.quality_score,
            retrieval_hit_rate=report.retrieval.hit_rate if report.retrieval else None,
            page_coverage_ratio=report.page_coverage_ratio,
            visual_annotation_ratio=report.visual_annotation_ratio,
            chunks_under_min_chars=report.chunks_under_min_chars,
            chunks_over_max_chars=report.chunks_over_max_chars,
            issue_codes=[issue.code for issue in report.issues],
        )
        for name, report in reports.items()
    ]
    rows.sort(
        key=lambda row: (
            row.retrieval_hit_rate if row.retrieval_hit_rate is not None else -1.0,
            row.quality_score,
        ),
        reverse=True,
    )
    best_by_quality = max(rows, key=lambda row: row.quality_score).name if rows else None
    retrieval_rows = [row for row in rows if row.retrieval_hit_rate is not None]
    best_by_retrieval = (
        max(retrieval_rows, key=lambda row: row.retrieval_hit_rate or 0.0).name
        if retrieval_rows
        else None
    )
    return ChunkingComparison(
        rows=rows,
        best_by_quality=best_by_quality,
        best_by_retrieval=best_by_retrieval,
    )

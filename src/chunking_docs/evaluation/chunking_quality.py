from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from chunking_docs.embeddings.records import asset_text_parts
from chunking_docs.embeddings.tokenizers import LexicalTokenizerConfig
from chunking_docs.evaluation.retrieval import (
    RetrievalCase,
    RetrievalEvaluation,
    evaluate_retrieval,
)
from chunking_docs.graph.provenance import chunk_asset_ids
from chunking_docs.models import DocumentChunk, GraphTriple, PageProfile, VisualAsset


class NumericSummary(BaseModel):
    count: int
    minimum: int
    maximum: int
    mean: float
    p50: float
    p95: float


class QualityIssue(BaseModel):
    severity: str
    code: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingQualityReport(BaseModel):
    page_count: int
    chunk_count: int
    covered_page_count: int
    page_coverage_ratio: float
    char_count: NumericSummary
    empty_chunk_count: int
    chunks_under_min_chars: int
    chunks_over_max_chars: int
    section_coverage_ratio: float
    visual_asset_linkage_ratio: float
    visual_annotation_ratio: float
    visual_text_asset_count: int = 0
    visual_text_covered_asset_count: int = 0
    visual_text_coverage_ratio: float = 1.0
    visual_text_missing_asset_ids: list[str] = Field(default_factory=list)
    standalone_visual_chunk_count: int = 0
    standalone_visual_text_asset_count: int = 0
    standalone_visual_text_asset_ids: list[str] = Field(default_factory=list)
    retrieval: RetrievalEvaluation | None = None
    quality_score: float
    issues: list[QualityIssue] = Field(default_factory=list)


def evaluate_chunking_quality(
    chunks: list[DocumentChunk],
    profiles: list[PageProfile],
    assets: list[VisualAsset],
    triples: list[GraphTriple],
    retrieval_cases: list[RetrievalCase] | None = None,
    top_k: int = 5,
    min_chars: int = 120,
    max_chars: int = 1800,
    tokenizer_config: LexicalTokenizerConfig | None = None,
    collapse_hierarchical: bool = False,
    retrieval_repeat: int = 1,
    fusion_weights: dict[str, float] | None = None,
) -> ChunkingQualityReport:
    page_numbers = {profile.page_no for profile in profiles}
    if not page_numbers:
        page_numbers = {page for chunk in chunks for page in range(chunk.page_start, chunk.page_end + 1)}
    covered_pages = {page for chunk in chunks for page in range(chunk.page_start, chunk.page_end + 1)}
    char_counts = [len(chunk.text.strip()) for chunk in chunks]
    empty_count = sum(1 for count in char_counts if count == 0)
    under_min = sum(1 for count in char_counts if 0 < count < min_chars)
    over_max = sum(1 for count in char_counts if count > max_chars)
    section_coverage = ratio(
        sum(1 for chunk in chunks if chunk.section.label() or chunk.metadata.get("section_label")),
        len(chunks),
    )
    chunks_with_assets = sum(1 for chunk in chunks if chunk_asset_ids(chunk))
    visual_linkage = ratio(chunks_with_assets, len(chunks))
    visual_annotation = ratio(sum(1 for asset in assets if asset.ocr_text or asset.vlm_summary), len(assets))
    visual_text_coverage = visual_text_coverage_stats(chunks, assets)
    standalone_visual_text = standalone_visual_text_stats(chunks, assets)
    retrieval = None
    if retrieval_cases:
        retrieval = evaluate_retrieval(
            chunks=chunks,
            triples=triples,
            cases=retrieval_cases,
            top_k=top_k,
            tokenizer_config=tokenizer_config,
            collapse_hierarchical=collapse_hierarchical,
            repeat=retrieval_repeat,
            fusion_weights=fusion_weights,
        )

    issues = quality_issues(
        page_coverage_ratio=ratio(len(covered_pages & page_numbers), len(page_numbers)),
        empty_count=empty_count,
        under_min=under_min,
        over_max=over_max,
        chunk_count=len(chunks),
        visual_text_asset_count=visual_text_coverage["asset_count"],
        visual_text_coverage_ratio=visual_text_coverage["coverage_ratio"],
        visual_text_missing_asset_ids=visual_text_coverage["missing_asset_ids"],
        retrieval=retrieval,
    )
    quality_score = compute_quality_score(
        page_coverage_ratio=ratio(len(covered_pages & page_numbers), len(page_numbers)),
        size_ratio=1.0 - ratio(empty_count + under_min + over_max, len(chunks)),
        section_coverage_ratio=section_coverage,
        visual_asset_linkage_ratio=visual_linkage,
        visual_text_coverage_ratio=visual_text_coverage["coverage_ratio"]
        if visual_text_coverage["asset_count"]
        else None,
        retrieval_score=retrieval_quality_score(retrieval) if retrieval else None,
    )

    return ChunkingQualityReport(
        page_count=len(page_numbers),
        chunk_count=len(chunks),
        covered_page_count=len(covered_pages & page_numbers),
        page_coverage_ratio=ratio(len(covered_pages & page_numbers), len(page_numbers)),
        char_count=summarize_numbers(char_counts),
        empty_chunk_count=empty_count,
        chunks_under_min_chars=under_min,
        chunks_over_max_chars=over_max,
        section_coverage_ratio=section_coverage,
        visual_asset_linkage_ratio=visual_linkage,
        visual_annotation_ratio=visual_annotation,
        visual_text_asset_count=visual_text_coverage["asset_count"],
        visual_text_covered_asset_count=visual_text_coverage["covered_asset_count"],
        visual_text_coverage_ratio=visual_text_coverage["coverage_ratio"],
        visual_text_missing_asset_ids=visual_text_coverage["missing_asset_ids"],
        standalone_visual_chunk_count=standalone_visual_text["chunk_count"],
        standalone_visual_text_asset_count=standalone_visual_text["asset_count"],
        standalone_visual_text_asset_ids=standalone_visual_text["asset_ids"],
        retrieval=retrieval,
        quality_score=quality_score,
        issues=issues,
    )


def summarize_numbers(values: list[int]) -> NumericSummary:
    if not values:
        return NumericSummary(count=0, minimum=0, maximum=0, mean=0.0, p50=0.0, p95=0.0)
    ordered = sorted(values)
    return NumericSummary(
        count=len(values),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=statistics.fmean(values),
        p50=percentile(ordered, 0.50),
        p95=percentile(ordered, 0.95),
    )


def percentile(ordered: list[int], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def quality_issues(
    page_coverage_ratio: float,
    empty_count: int,
    under_min: int,
    over_max: int,
    chunk_count: int,
    visual_text_asset_count: int,
    visual_text_coverage_ratio: float,
    visual_text_missing_asset_ids: list[str],
    retrieval: RetrievalEvaluation | None,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if page_coverage_ratio < 1.0:
        issues.append(
            QualityIssue(
                severity="error",
                code="incomplete_page_coverage",
                message="Not every page is covered by at least one chunk.",
                metadata={"page_coverage_ratio": page_coverage_ratio},
            )
        )
    if empty_count:
        issues.append(
            QualityIssue(
                severity="warning",
                code="empty_chunks",
                message="Some chunks contain no searchable text.",
                metadata={"count": empty_count},
            )
        )
    if ratio(under_min + over_max, chunk_count) > 0.25:
        issues.append(
            QualityIssue(
                severity="warning",
                code="chunk_size_distribution",
                message="Many chunks fall outside the configured size window.",
                metadata={"under_min": under_min, "over_max": over_max},
            )
        )
    if visual_text_asset_count and visual_text_coverage_ratio < 0.8:
        issues.append(
            QualityIssue(
                severity="warning",
                code="visual_text_coverage",
                message="Some linked visual asset text is not present in candidate chunks.",
                metadata={
                    "visual_text_asset_count": visual_text_asset_count,
                    "visual_text_coverage_ratio": visual_text_coverage_ratio,
                    "missing_asset_ids": visual_text_missing_asset_ids[:50],
                },
            )
        )
    if retrieval is not None and retrieval.recall_at_k < 0.8:
        issues.append(
            QualityIssue(
                severity="warning",
                code="retrieval_hit_rate",
                message="Retrieval evaluation hit rate is below the recommended threshold.",
                metadata={
                    "recall_at_k": retrieval.recall_at_k,
                    "mrr": retrieval.mrr,
                    "target_coverage_at_k": retrieval.target_coverage_at_k,
                    "mean_target_ndcg_at_k": retrieval.mean_target_ndcg_at_k,
                    "mean_precision_at_k": retrieval.mean_precision_at_k,
                    "expected_case_count": retrieval.expected_case_count,
                    "failed_queries": retrieval.failed_queries[:10],
                },
            )
        )
    if retrieval is not None and retrieval.expected_case_count and retrieval.target_coverage_at_k < 0.8:
        issues.append(
            QualityIssue(
                severity="warning",
                code="retrieval_target_coverage",
                message="Retrieval target coverage is below the recommended threshold.",
                metadata={
                    "target_coverage_at_k": retrieval.target_coverage_at_k,
                    "recall_at_k": retrieval.recall_at_k,
                    "mean_target_ndcg_at_k": retrieval.mean_target_ndcg_at_k,
                    "mean_precision_at_k": retrieval.mean_precision_at_k,
                },
            )
        )
    if retrieval is not None and retrieval.expected_case_count and retrieval.mean_target_ndcg_at_k < 0.7:
        issues.append(
            QualityIssue(
                severity="warning",
                code="retrieval_target_ranking",
                message="Expected retrieval targets are not ranked highly enough.",
                metadata={
                    "mean_target_ndcg_at_k": retrieval.mean_target_ndcg_at_k,
                    "target_coverage_at_k": retrieval.target_coverage_at_k,
                    "recall_at_k": retrieval.recall_at_k,
                    "mean_precision_at_k": retrieval.mean_precision_at_k,
                },
            )
        )
    return issues


def compute_quality_score(
    page_coverage_ratio: float,
    size_ratio: float,
    section_coverage_ratio: float,
    visual_asset_linkage_ratio: float,
    visual_text_coverage_ratio: float | None,
    retrieval_score: float | None,
) -> float:
    components = [
        (page_coverage_ratio, 0.30),
        (max(0.0, size_ratio), 0.25),
        (section_coverage_ratio, 0.15),
        (visual_asset_linkage_ratio, 0.15),
    ]
    if visual_text_coverage_ratio is not None:
        components.append((visual_text_coverage_ratio, 0.15))
    if retrieval_score is not None:
        components.append((retrieval_score, 0.30))
    total_weight = sum(weight for _, weight in components)
    return sum(value * weight for value, weight in components) / total_weight if total_weight else 0.0


def retrieval_quality_score(retrieval: RetrievalEvaluation) -> float:
    return (
        retrieval.recall_at_k * 0.35
        + retrieval.target_coverage_at_k * 0.25
        + retrieval.mean_target_ndcg_at_k * 0.25
        + retrieval.mean_precision_at_k * 0.15
    )


def visual_text_coverage_stats(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
) -> dict[str, float | int | list[str]]:
    assets_with_text = {}
    for asset in assets:
        parts = asset_text_parts(asset)
        if parts:
            assets_with_text[asset.asset_id] = parts
    chunk_texts_by_asset: dict[str, list[str]] = {asset_id: [] for asset_id in assets_with_text}
    for chunk in chunks:
        for asset_id in chunk_asset_ids(chunk):
            if asset_id in chunk_texts_by_asset:
                chunk_texts_by_asset[asset_id].append(chunk.text)

    covered_asset_ids = []
    missing_asset_ids = []
    for asset_id, parts in assets_with_text.items():
        chunk_texts = chunk_texts_by_asset.get(asset_id, [])
        if any(visual_text_part_in_chunks(part, chunk_texts) for part in parts):
            covered_asset_ids.append(asset_id)
        else:
            missing_asset_ids.append(asset_id)

    asset_count = len(assets_with_text)
    covered_count = len(covered_asset_ids)
    return {
        "asset_count": asset_count,
        "covered_asset_count": covered_count,
        "coverage_ratio": ratio(covered_count, asset_count) if asset_count else 1.0,
        "missing_asset_ids": sorted(missing_asset_ids),
    }


def standalone_visual_text_stats(
    chunks: list[DocumentChunk],
    assets: list[VisualAsset],
) -> dict[str, int | list[str]]:
    text_asset_ids = {asset.asset_id for asset in assets if asset_text_parts(asset)}
    standalone_chunk_count = 0
    standalone_asset_ids: list[str] = []
    seen = set()
    for chunk in chunks:
        if not chunk.metadata.get("visual_asset_unlinked"):
            continue
        standalone_chunk_count += 1
        for asset_id in chunk_asset_ids(chunk):
            if asset_id not in text_asset_ids or asset_id in seen:
                continue
            standalone_asset_ids.append(asset_id)
            seen.add(asset_id)
    return {
        "chunk_count": standalone_chunk_count,
        "asset_count": len(standalone_asset_ids),
        "asset_ids": sorted(standalone_asset_ids),
    }


def visual_text_part_in_chunks(part: str, chunk_texts: list[str]) -> bool:
    normalized_part = normalize_for_coverage(part)
    if not normalized_part:
        return False
    for text in chunk_texts:
        normalized_text = normalize_for_coverage(text)
        if normalized_part in normalized_text:
            return True
        if len(normalized_part) > 80 and normalized_part[:80] in normalized_text:
            return True
    return False


def normalize_for_coverage(value: str) -> str:
    return " ".join(value.split()).casefold()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0

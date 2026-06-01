import pytest

from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.compare import compare_chunking_reports
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset


def test_compare_chunking_reports_ranks_by_retrieval_then_quality():
    weak_chunks = [
        DocumentChunk(
            chunk_id="weak",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="unrelated",
        )
    ]
    strong_chunks = [
        DocumentChunk(
            chunk_id="strong",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="river corridor station hub",
            asset_ids=["asset-1"],
            metadata={
                "chunking_strategy": "visual_asset_text",
                "retrieval_role": "child",
            },
        )
    ]
    cases = [
        RetrievalCase(
            query="river corridor",
            expected_pages=[1],
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_lexical_probe"},
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            caption="river corridor station hub",
        )
    ]
    reports = {
        "weak": evaluate_chunking_quality(weak_chunks, [], assets, [], retrieval_cases=cases),
        "strong": evaluate_chunking_quality(strong_chunks, [], assets, [], retrieval_cases=cases),
    }

    comparison = compare_chunking_reports(reports)

    assert comparison.rows[0].name == "strong"
    assert comparison.best_by_retrieval == "strong"
    assert comparison.fastest_by_mean_latency in {"weak", "strong"}
    assert comparison.rows[0].retrieval_recall_at_k == 1.0
    assert comparison.rows[0].retrieval_target_coverage_at_k == 1.0
    assert comparison.rows[0].retrieval_mean_target_ndcg_at_k == 1.0
    assert comparison.rows[0].retrieval_mean_precision_at_k is not None
    assert comparison.rows[0].retrieval_mean_first_relevant_rank == 1.0
    assert comparison.rows[0].retrieval_mean_target_rank == 1.0
    assert comparison.rows[0].retrieval_mean_latency_ms is not None
    assert comparison.rows[0].retrieval_unstable_result_count == 0.0
    assert comparison.rows[0].retrieval_result_stability_rate == 1.0
    assert comparison.rows[0].total_chunk_chars == len(strong_chunks[0].text)
    assert comparison.rows[0].embedding_text_kchars == len(strong_chunks[0].text) / 1000.0
    assert comparison.rows[0].retrieval_score == pytest.approx(
        comparison.rows[0].retrieval_recall_at_k * 0.35
        + comparison.rows[0].retrieval_target_coverage_at_k * 0.25
        + comparison.rows[0].retrieval_mean_target_ndcg_at_k * 0.25
        + comparison.rows[0].retrieval_mean_precision_at_k * 0.15
    )
    assert comparison.rows[0].retrieval_score_per_embedding_kchar == pytest.approx(
        comparison.rows[0].retrieval_score / comparison.rows[0].embedding_text_kchars
    )
    assert comparison.rows[0].retrieval_score_per_mean_latency_ms == pytest.approx(
        comparison.rows[0].retrieval_score
        / comparison.rows[0].retrieval_mean_latency_ms
    )
    assert comparison.rows[0].target_coverage_per_mean_latency_ms == pytest.approx(
        comparison.rows[0].retrieval_target_coverage_at_k
        / comparison.rows[0].retrieval_mean_latency_ms
    )
    assert comparison.rows[0].target_ndcg_per_mean_latency_ms == pytest.approx(
        comparison.rows[0].retrieval_mean_target_ndcg_at_k
        / comparison.rows[0].retrieval_mean_latency_ms
    )
    assert comparison.rows[0].retrieval_score_per_p95_latency_ms == pytest.approx(
        comparison.rows[0].retrieval_score / comparison.rows[0].retrieval_p95_latency_ms
    )
    assert comparison.rows[0].target_coverage_per_p95_latency_ms == pytest.approx(
        comparison.rows[0].retrieval_target_coverage_at_k
        / comparison.rows[0].retrieval_p95_latency_ms
    )
    assert comparison.rows[0].target_ndcg_per_p95_latency_ms == pytest.approx(
        comparison.rows[0].retrieval_mean_target_ndcg_at_k
        / comparison.rows[0].retrieval_p95_latency_ms
    )
    assert comparison.rows[0].target_metrics["asset"]["coverage_at_k"] == 1.0
    assert comparison.rows[0].source_family_metrics["lexical"]["target_coverage_at_k"] == 1.0
    assert comparison.rows[0].chunk_strategy_metrics["visual_asset_text"][
        "target_coverage_at_k"
    ] == 1.0
    assert comparison.rows[0].retrieval_role_metrics["child"]["target_coverage_at_k"] == 1.0
    assert comparison.rows[0].case_group_metrics["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ] == 1.0
    assert comparison.rows[0].visual_text_asset_count == 1
    assert comparison.rows[0].visual_text_covered_asset_count == 1
    assert comparison.rows[0].visual_text_coverage_ratio == 1.0
    assert comparison.rows[0].visual_text_part_count == 1
    assert comparison.rows[0].visual_text_covered_part_count == 1
    assert comparison.rows[0].visual_text_part_coverage_ratio == 1.0
    assert comparison.rows[0].standalone_visual_chunk_count == 0
    pairwise = next(
        item
        for item in comparison.pairwise
        if item.candidate == "strong" and item.baseline == "weak"
    )
    assert pairwise.shared_query_count == 1
    assert pairwise.candidate_win_rate == 1.0
    assert pairwise.mean_target_coverage_delta > 0.0
    assert pairwise.mean_target_ndcg_delta > 0.0
    assert pairwise.mean_first_relevant_rank_delta <= 0.0
    assert pairwise.mean_target_rank_delta < 0.0
    assert pairwise.bootstrap_samples == 1000
    assert pairwise.target_coverage_delta_ci_low == pairwise.mean_target_coverage_delta
    assert pairwise.target_ndcg_delta_ci_high == pairwise.mean_target_ndcg_delta
    assert comparison.rows[-1].failed_queries == ["river corridor"]
    assert comparison.rows[-1].visual_text_coverage_ratio == 0.0
    assert comparison.rows[-1].visual_text_part_coverage_ratio == 0.0

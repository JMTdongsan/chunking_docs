from chunking_docs.evaluation.audit import audit_package, degraded_page_ratio
from chunking_docs.evaluation.retrieval import RetrievalCase, evaluate_retrieval
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    TextQuality,
    VisualAsset,
)


def test_audit_package_detects_missing_vlm_annotations():
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=0,
            line_count=0,
            text_block_count=0,
            image_block_count=1,
            embedded_image_count=1,
            drawing_count=0,
            text_quality=TextQuality.EMPTY,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.PAGE_SUMMARY,
            text="",
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            metadata={"requires_vlm": True},
        )
    ]

    audit = audit_package(profiles, chunks, assets, [], require_annotations_for_visual_pages=True)

    assert not audit.passed
    assert audit.pages_requiring_vlm == [1]
    assert degraded_page_ratio(profiles) == 1.0


def test_evaluate_retrieval_hit_rate():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=12,
            page_end=12,
            kind=ChunkKind.TEXT,
            text="north district river corridor",
        )
    ]
    triples = [
        GraphTriple(
            triple_id="t",
            doc_id="doc",
            chunk_id="a",
            subject="north district",
            predicate="uses_axis",
            object="river corridor",
        )
    ]
    cases = [RetrievalCase(query="north district", expected_pages=[12], graph_expand=True)]

    result = evaluate_retrieval(chunks, triples, cases, top_k=3)

    assert result.hit_rate == 1.0
    assert result.recall_at_k == 1.0
    assert result.mrr == 1.0
    assert result.results[0].passed
    assert result.results[0].matched_rank == 1
    assert result.results[0].matched_page == 12


def test_evaluate_retrieval_reports_ranked_failures():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="alpha beta",
        ),
        DocumentChunk(
            chunk_id="b",
            doc_id="doc",
            page_start=3,
            page_end=4,
            kind=ChunkKind.TEXT,
            text="gamma delta",
        ),
    ]
    cases = [
        RetrievalCase(query="alpha", expected_pages=[2]),
        RetrievalCase(query="missing", expected_pages=[9]),
    ]

    result = evaluate_retrieval(chunks, [], cases, top_k=2)

    assert result.expected_case_count == 2
    assert result.passed_count == 1
    assert result.failed_count == 1
    assert result.recall_at_k == 0.5
    assert result.mrr == 0.5
    assert result.failed_queries == ["missing"]
    assert result.results[0].top_page_ranges == [(1, 2)]

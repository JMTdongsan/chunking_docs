from chunking_docs.evaluation.chunking_quality import evaluate_chunking_quality
from chunking_docs.evaluation.retrieval import RetrievalCase
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    TextQuality,
    VisualAsset,
)


def test_evaluate_chunking_quality_reports_retrieval_and_multimodal_metrics():
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=100,
            height=100,
            char_count=100,
            line_count=5,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="policy network transit corridor public space",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            ocr_text="transit corridor",
        )
    ]

    report = evaluate_chunking_quality(
        chunks=chunks,
        profiles=profiles,
        assets=assets,
        triples=[],
        retrieval_cases=[RetrievalCase(query="transit corridor", expected_pages=[1])],
    )

    assert report.page_coverage_ratio == 1.0
    assert report.visual_asset_linkage_ratio == 1.0
    assert report.visual_annotation_ratio == 1.0
    assert report.retrieval is not None
    assert report.retrieval.hit_rate == 1.0
    assert report.quality_score > 0.5


def test_evaluate_chunking_quality_flags_size_and_retrieval_issues():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="tiny",
        )
    ]

    report = evaluate_chunking_quality(
        chunks=chunks,
        profiles=[],
        assets=[],
        triples=[
            GraphTriple(
                triple_id="triple-1",
                doc_id="doc",
                chunk_id="chunk-1",
                subject="topic",
                predicate="related_to",
                object="other",
            )
        ],
        retrieval_cases=[RetrievalCase(query="missing", expected_pages=[99])],
        min_chars=20,
    )

    codes = {issue.code for issue in report.issues}
    assert "chunk_size_distribution" in codes
    assert "retrieval_hit_rate" in codes

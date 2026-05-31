import hashlib
import json
import math

import pytest
from typer.testing import CliRunner

import chunking_docs.cli as cli_module
from chunking_docs.cli import app
from chunking_docs.evaluation.audit import audit_package, degraded_page_ratio
from chunking_docs.evaluation.ablation import (
    QdrantVectorAblationMode,
    QdrantVectorAblationReport,
    QdrantVectorAblationRow,
    evaluate_retrieval_ablation,
    gate_retrieval_ablation,
    gate_qdrant_vector_ablation,
    parse_ablation_modes,
    parse_qdrant_vector_ablation_modes,
    qdrant_vector_names_for_modes,
)
from chunking_docs.evaluation.retrieval import RetrievalCase, evaluate_retrieval, evaluate_search_results
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    TextQuality,
    VisualAsset,
)
from chunking_docs.retrieval.local_hybrid import HybridSearchHit
from chunking_docs.storage.records import EmbeddingRecord


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


def test_audit_package_treats_unstructured_vlm_as_requiring_retry():
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=10,
            line_count=1,
            text_block_count=1,
            image_block_count=0,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.DEGRADED,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.PAGE_SUMMARY,
            text="summary",
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            vlm_summary="plain text fallback",
            metadata={"requires_vlm": True, "vlm_parse_status": "raw_text"},
        )
    ]

    audit = audit_package(profiles, chunks, assets, [], require_annotations_for_visual_pages=True)

    assert not audit.passed
    assert audit.pages_requiring_vlm == [1]


def test_audit_package_does_not_require_empty_completed_ocr_retry():
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
            kind=AssetKind.PAGE_IMAGE,
            ocr_text="",
            metadata={
                "requires_ocr": True,
                "ocr_text_chars": 0,
                "ocr_backend": "fake-ocr",
            },
        )
    ]

    audit = audit_package(profiles, chunks, assets, [])

    assert audit.pages_requiring_ocr == []


def test_audit_package_requires_vlm_retry_when_parse_status_is_missing():
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=10,
            line_count=1,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=1,
            drawing_count=0,
            text_quality=TextQuality.DEGRADED,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.PAGE_SUMMARY,
            text="summary",
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            vlm_summary="legacy visual summary",
            metadata={"requires_vlm": True},
        )
    ]

    audit = audit_package(profiles, chunks, assets, [], require_annotations_for_visual_pages=True)

    assert not audit.passed
    assert audit.pages_requiring_vlm == [1]


def test_audit_package_validates_qdrant_artifacts(tmp_path):
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=10,
            line_count=1,
            text_block_count=1,
            image_block_count=0,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="retrieval benchmark",
        )
    ]
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "documents",
                "named_vectors": {"text_dense": {"size": 3}},
                "payload_indexes": [{"field": "doc_id", "schema": "keyword"}],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "qdrant_text_records.jsonl",
        [
            EmbeddingRecord(
                point_id="point",
                chunk_id="chunk",
                doc_id="doc",
                vector_name="text_dense",
                vector=[1.0, 2.0],
                payload={
                    "chunk_id": "chunk",
                    "doc_id": "doc",
                    "page_start": 1,
                    "kind": "text",
                    "text": "retrieval benchmark",
                },
            )
        ],
    )

    audit = audit_package(profiles, chunks, [], [], package_dir=tmp_path)
    codes = {issue.code for issue in audit.issues}

    assert audit.qdrant_record_counts == {"text_dense": 1}
    assert audit.qdrant_vector_sizes == {"text_dense": 2}
    assert "qdrant_vector_size_mismatch" in codes
    assert "qdrant_missing_payload" in codes
    assert "missing_qdrant_payload_indexes" in codes
    assert "missing_embedding_manifest" in codes


def test_audit_package_validates_qdrant_target_coverage(tmp_path):
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=10,
            line_count=1,
            text_block_count=1,
            image_block_count=1,
            embedded_image_count=1,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk-a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="renewal strategy",
        ),
        DocumentChunk(
            chunk_id="chunk-b",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="mobility access",
        ),
    ]
    image_path = tmp_path / "asset.png"
    image_path.write_bytes(b"image")
    assets = [
        VisualAsset(
            asset_id="asset-a",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=image_path,
            vlm_summary="diagram summary",
        ),
        VisualAsset(
            asset_id="asset-b",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.FIGURE,
            path=image_path,
            vlm_summary="figure summary",
        ),
    ]
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "documents",
                "named_vectors": {
                    "text_dense": {"size": 2},
                    "image_dense": {"size": 2},
                    "caption_dense": {"size": 2},
                },
                "payload_indexes": [
                    {"field": "doc_id", "schema": "keyword"},
                    {"field": "chunk_id", "schema": "keyword"},
                    {"field": "asset_id", "schema": "keyword"},
                    {"field": "kind", "schema": "keyword"},
                    {"field": "page_no", "schema": "integer"},
                    {"field": "page_start", "schema": "integer"},
                    {"field": "page_end", "schema": "integer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "qdrant_text_records.jsonl",
        [
            EmbeddingRecord(
                point_id="text-a",
                chunk_id="chunk-a",
                doc_id="doc",
                vector_name="text_dense",
                vector=[1.0, 0.0],
                payload={
                    "chunk_id": "chunk-a",
                    "doc_id": "doc",
                    "page_start": 1,
                    "page_end": 1,
                    "kind": "text",
                    "text": "renewal strategy",
                },
            ),
            EmbeddingRecord(
                point_id="text-stale",
                chunk_id="missing-chunk",
                doc_id="doc",
                vector_name="text_dense",
                vector=[0.0, 1.0],
                payload={
                    "chunk_id": "missing-chunk",
                    "doc_id": "doc",
                    "page_start": 1,
                    "page_end": 1,
                    "kind": "text",
                    "text": "stale record",
                },
            ),
        ],
    )
    write_jsonl(
        tmp_path / "qdrant_image_records.jsonl",
        [
            EmbeddingRecord(
                point_id="image-a",
                chunk_id="asset-a",
                doc_id="doc",
                vector_name="image_dense",
                vector=[0.5, 0.5],
                payload={
                    "asset_id": "asset-a",
                    "doc_id": "doc",
                    "page_no": 1,
                    "kind": "page_image",
                },
            )
        ],
    )
    write_jsonl(
        tmp_path / "qdrant_caption_records.jsonl",
        [
            EmbeddingRecord(
                point_id="caption-a",
                chunk_id="asset-a",
                doc_id="doc",
                vector_name="caption_dense",
                vector=[0.25, 0.75],
                payload={
                    "asset_id": "asset-a",
                    "doc_id": "doc",
                    "page_no": 1,
                    "kind": "page_image",
                    "text": "diagram summary",
                },
            )
        ],
    )

    audit = audit_package(
        profiles,
        chunks,
        assets,
        [],
        package_dir=tmp_path,
        require_qdrant_records=True,
    )
    codes = {issue.code for issue in audit.issues}

    assert not audit.passed
    assert "qdrant_missing_chunk_records" in codes
    assert "qdrant_stale_chunk_records" in codes
    assert "qdrant_missing_image_asset_records" in codes
    assert "qdrant_missing_caption_asset_records" in codes


def test_audit_package_validates_embedding_manifest_contract(tmp_path):
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=10,
            line_count=1,
            text_block_count=1,
            image_block_count=0,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="retrieval benchmark",
        )
    ]
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "documents",
                "named_vectors": {"text_dense": {"size": 3}},
                "payload_indexes": [
                    {"field": "doc_id", "schema": "keyword"},
                    {"field": "chunk_id", "schema": "keyword"},
                    {"field": "kind", "schema": "keyword"},
                    {"field": "page_start", "schema": "integer"},
                    {"field": "page_end", "schema": "integer"},
                    {"field": "page_no", "schema": "integer"},
                    {"field": "asset_id", "schema": "keyword"},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "qdrant_text_records.jsonl",
        [
            EmbeddingRecord(
                point_id="point",
                chunk_id="chunk",
                doc_id="doc",
                vector_name="text_dense",
                vector=[1.0, 2.0, 3.0],
                payload={
                    "chunk_id": "chunk",
                    "doc_id": "doc",
                    "page_start": 1,
                    "page_end": 1,
                    "kind": "text",
                    "text": "retrieval benchmark",
                },
            )
        ],
    )
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "wrong_collection",
                "vectors": {
                    "text_dense": {
                        "file": "qdrant_text_records.jsonl",
                        "record_count": 2,
                        "dimension": 4,
                        "distance": "Cosine",
                        "bytes": 1,
                        "sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_package(profiles, chunks, [], [], package_dir=tmp_path)
    codes = {issue.code for issue in audit.issues}

    assert audit.embedding_manifest["collection"] == "wrong_collection"
    assert "embedding_manifest_collection_mismatch" in codes
    assert "embedding_manifest_record_count_mismatch" in codes
    assert "embedding_manifest_dimension_mismatch" in codes
    assert "embedding_manifest_observed_dimension_mismatch" in codes
    assert "embedding_manifest_bytes_mismatch" in codes
    assert "embedding_manifest_sha256_mismatch" in codes


def test_audit_package_accepts_matching_embedding_manifest(tmp_path):
    profiles = [
        PageProfile(
            doc_id="doc",
            page_no=1,
            width=1,
            height=1,
            char_count=10,
            line_count=1,
            text_block_count=1,
            image_block_count=0,
            embedded_image_count=0,
            drawing_count=0,
            text_quality=TextQuality.GOOD,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_id="chunk",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="retrieval benchmark",
        )
    ]
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "documents",
                "named_vectors": {"text_dense": {"size": 3}},
                "payload_indexes": [
                    {"field": "doc_id", "schema": "keyword"},
                    {"field": "chunk_id", "schema": "keyword"},
                    {"field": "kind", "schema": "keyword"},
                    {"field": "page_start", "schema": "integer"},
                    {"field": "page_end", "schema": "integer"},
                    {"field": "page_no", "schema": "integer"},
                    {"field": "asset_id", "schema": "keyword"},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "qdrant_text_records.jsonl",
        [
            EmbeddingRecord(
                point_id="point",
                chunk_id="chunk",
                doc_id="doc",
                vector_name="text_dense",
                vector=[1.0, 2.0, 3.0],
                payload={
                    "chunk_id": "chunk",
                    "doc_id": "doc",
                    "page_start": 1,
                    "page_end": 1,
                    "kind": "text",
                    "text": "retrieval benchmark",
                },
            )
        ],
    )
    record_content = (tmp_path / "qdrant_text_records.jsonl").read_bytes()
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "documents",
                "vectors": {
                    "text_dense": {
                        "file": "qdrant_text_records.jsonl",
                        "record_count": 1,
                        "dimension": 3,
                        "distance": "Cosine",
                        "bytes": len(record_content),
                        "sha256": hashlib.sha256(record_content).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_package(profiles, chunks, [], [], package_dir=tmp_path)
    codes = {issue.code for issue in audit.issues}

    assert audit.passed
    assert not any(code.startswith("embedding_manifest_") for code in codes)


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

    result = evaluate_retrieval(chunks, triples, cases, top_k=3, repeat=2)

    assert result.hit_rate == 1.0
    assert result.recall_at_k == 1.0
    assert result.mrr == 1.0
    assert result.mean_target_ndcg_at_k == 1.0
    assert result.repeat == 2
    assert result.mean_latency_ms >= 0.0
    assert result.p95_latency_ms >= 0.0
    assert result.target_metrics["page"].recall_at_k == 1.0
    assert result.target_metrics["page"].mrr == 1.0
    assert result.target_metrics["page"].ndcg_at_k == 1.0
    assert result.results[0].passed
    assert result.results[0].target_matches == {"page": True}
    assert result.results[0].target_matched_ranks == {"page": 1}
    assert result.results[0].target_key_matched_ranks == {"page:12": 1}
    assert result.results[0].target_ndcg_at_k == 1.0
    assert len(result.results[0].latency_samples_ms) == 2
    assert result.results[0].matched_rank == 1
    assert result.results[0].matched_page == 12


def test_evaluate_retrieval_reports_target_type_metrics():
    chunks = [
        DocumentChunk(
            chunk_id="a",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="station access corridor",
        )
    ]
    cases = [
        RetrievalCase(
            query="station access",
            expected_pages=[1],
            expected_asset_ids=["missing-asset"],
        )
    ]

    result = evaluate_retrieval(chunks, [], cases, top_k=1)

    assert result.passed_count == 1
    assert result.target_metrics["page"].recall_at_k == 1.0
    assert result.target_metrics["asset"].recall_at_k == 0.0
    assert result.target_metrics["asset"].failed_queries == ["station access"]
    assert result.results[0].target_matches == {"page": True, "asset": False}


def test_evaluate_search_results_reports_target_coverage_and_precision():
    chunk_a = DocumentChunk(
        chunk_id="a",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="alpha",
        asset_ids=["asset-a"],
    )
    chunk_b = DocumentChunk(
        chunk_id="b",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="beta",
    )

    class Hit:
        def __init__(self, chunk):
            self.chunk = chunk
            self.sources = ["test"]
            self.evidence_chunks = []
            self.payloads = []

    result = evaluate_search_results(
        cases=[
            RetrievalCase(
                query="multi target",
                expected_pages=[1, 2],
                expected_chunk_ids=["missing-chunk"],
                expected_asset_ids=["asset-a"],
            )
        ],
        search_fn=lambda case, graph_expand: [Hit(chunk_a), Hit(chunk_b)],
        top_k=3,
    )

    case_result = result.results[0]
    assert case_result.expected_target_count == 4
    assert case_result.matched_target_count == 3
    assert case_result.target_coverage_at_k == 0.75
    assert case_result.target_ndcg_at_k == pytest.approx((2 + 1 / math.log2(3)) / 4)
    assert case_result.relevant_hit_count == 2
    assert case_result.precision_at_k == 2 / 3
    assert case_result.top_matched_targets == [["page:1", "asset:asset-a"], ["page:2"]]
    assert case_result.target_key_matched_ranks == {
        "page:1": 1,
        "asset:asset-a": 1,
        "page:2": 2,
    }
    assert result.target_coverage_at_k == 0.75
    assert result.mean_target_ndcg_at_k == case_result.target_ndcg_at_k
    assert result.mean_precision_at_k == 2 / 3
    assert result.target_metrics["page"].target_count == 2
    assert result.target_metrics["page"].matched_target_count == 2
    assert result.target_metrics["page"].coverage_at_k == 1.0
    assert result.target_metrics["page"].ndcg_at_k == pytest.approx(
        (1 + 1 / math.log2(3)) / 2
    )
    assert result.target_metrics["chunk"].coverage_at_k == 0.0
    assert result.target_metrics["chunk"].ndcg_at_k == 0.0
    assert result.target_metrics["asset"].coverage_at_k == 1.0
    assert result.target_metrics["asset"].ndcg_at_k == 1.0
    assert result.source_metrics["test"].query_count == 1
    assert result.source_metrics["test"].hit_count == 2
    assert result.source_metrics["test"].relevant_hit_count == 2
    assert result.source_metrics["test"].matched_target_count == 3
    assert result.source_metrics["test"].target_coverage_at_k == 0.75


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


def test_evaluate_retrieval_matches_collapsed_hierarchical_evidence_chunk():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.PAGE_SUMMARY,
        text="summary",
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access benchmark evidence",
        metadata={"hierarchical_parent_chunk_id": "parent"},
    )
    cases = [RetrievalCase(query="station access", expected_chunk_ids=["child"])]

    result = evaluate_retrieval([parent, child], [], cases, collapse_hierarchical=True)

    assert result.recall_at_k == 1.0
    assert result.results[0].top_chunk_ids == ["parent"]
    assert result.results[0].top_evidence_chunk_ids == [["child"]]
    assert result.results[0].matched_chunk_id == "child"


def test_evaluate_retrieval_matches_hierarchical_source_chunk_alias():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.PAGE_SUMMARY,
        text="summary",
        metadata={"source_chunk_id": "source-1"},
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access benchmark evidence",
        metadata={
            "source_chunk_id": "source-1",
            "hierarchical_parent_chunk_id": "parent",
        },
    )
    cases = [RetrievalCase(query="station access", expected_chunk_ids=["source-1"])]

    result = evaluate_retrieval([parent, child], [], cases, collapse_hierarchical=True)

    assert result.recall_at_k == 1.0
    assert result.results[0].matched_chunk_id == "source-1"
    assert result.results[0].target_matches["chunk"] is True


def test_evaluate_retrieval_matches_visual_asset_id_from_evidence_chunk():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.PAGE_SUMMARY,
        text="summary",
    )
    child = DocumentChunk(
        chunk_id="child",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access map benchmark evidence",
        asset_ids=["asset-map"],
        metadata={"hierarchical_parent_chunk_id": "parent"},
    )
    cases = [RetrievalCase(query="station access map", expected_asset_ids=["asset-map"])]

    result = evaluate_retrieval([parent, child], [], cases, collapse_hierarchical=True)

    assert result.expected_case_count == 1
    assert result.recall_at_k == 1.0
    assert result.results[0].top_chunk_ids == ["parent"]
    assert result.results[0].top_asset_ids == [["asset-map"]]
    assert result.results[0].matched_asset_id == "asset-map"
    assert result.target_metrics["asset"].recall_at_k == 1.0


def test_evaluate_retrieval_matches_expected_triple_id():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=4,
        page_end=4,
        kind=ChunkKind.TEXT,
        text="unrelated text",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="north district",
        predicate="connects_to",
        object="river corridor",
    )
    cases = [RetrievalCase(query="north district", expected_triple_ids=["triple-1"])]

    result = evaluate_retrieval(
        [chunk],
        [triple],
        cases,
        use_dense=False,
        use_bm25=False,
        use_graph=True,
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].top_triple_ids == [["triple-1"]]
    assert result.results[0].matched_triple_id == "triple-1"
    assert result.results[0].top_sources == [["graph"]]
    assert result.target_metrics["triple"].recall_at_k == 1.0


def test_evaluate_retrieval_matches_triples_attached_to_source_chunk_alias():
    parent = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=4,
        page_end=4,
        kind=ChunkKind.PAGE_SUMMARY,
        text="north district river corridor",
        metadata={"source_chunk_id": "source-1"},
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="source-1",
        subject="north district",
        predicate="connects_to",
        object="river corridor",
    )
    cases = [RetrievalCase(query="north district", expected_triple_ids=["triple-1"])]

    result = evaluate_retrieval(
        [parent],
        [triple],
        cases,
        use_dense=False,
        use_bm25=False,
        use_graph=True,
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].matched_triple_id == "triple-1"
    assert result.results[0].top_triple_ids == [["triple-1"]]


def test_evaluate_search_results_matches_visual_asset_id_from_payload():
    chunk = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access",
    )

    class PayloadHit:
        def __init__(self):
            self.chunk = chunk
            self.sources = ["qdrant:caption_dense"]
            self.evidence_chunks = []
            self.payloads = [{"asset_id": "asset-map"}]

    result = evaluate_search_results(
        cases=[RetrievalCase(query="station access map", expected_asset_ids=["asset-map"])],
        search_fn=lambda case, graph_expand: [PayloadHit()],
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].top_asset_ids == [["asset-map"]]
    assert result.results[0].matched_asset_id == "asset-map"
    assert result.source_metrics["qdrant:caption_dense"].target_coverage_at_k == 1.0
    assert result.source_family_metrics["visual"].target_coverage_at_k == 1.0
    assert result.source_family_metrics["visual"].mean_relevant_rank == 1.0


def test_evaluate_search_results_matches_visual_asset_ids_from_payload_list():
    chunk = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access",
    )

    class PayloadHit:
        def __init__(self):
            self.chunk = chunk
            self.sources = ["qdrant:text_dense"]
            self.evidence_chunks = []
            self.payloads = [{"asset_id": ["asset-map", "asset-diagram"]}]

    result = evaluate_search_results(
        cases=[RetrievalCase(query="station access map", expected_asset_ids=["asset-diagram"])],
        search_fn=lambda case, graph_expand: [PayloadHit()],
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].top_asset_ids == [["asset-map", "asset-diagram"]]
    assert result.results[0].matched_asset_id == "asset-diagram"


def test_evaluate_search_results_matches_visual_asset_provenance_triple_from_payload():
    chunk = DocumentChunk(
        chunk_id="parent",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access",
    )
    triple = GraphTriple(
        triple_id="triple-map",
        doc_id="doc",
        chunk_id="visual-annotation-chunk",
        subject="station access map",
        predicate="shows",
        object="transfer corridor",
        qualifiers={"source": "visual_annotation", "asset_id": "asset-map"},
    )

    class PayloadHit:
        def __init__(self):
            self.chunk = chunk
            self.sources = ["qdrant:caption_dense"]
            self.evidence_chunks = []
            self.payloads = [{"asset_id": "asset-map"}]

    result = evaluate_search_results(
        cases=[RetrievalCase(query="station access map", expected_triple_ids=["triple-map"])],
        search_fn=lambda case, graph_expand: [PayloadHit()],
        triples=[triple],
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].top_triple_ids == [["triple-map"]]
    assert result.results[0].top_matched_targets == [["triple:triple-map"]]
    assert result.results[0].matched_triple_id == "triple-map"
    assert result.target_metrics["triple"].recall_at_k == 1.0
    assert result.source_family_metrics["visual"].target_coverage_at_k == 1.0


def test_evaluate_retrieval_ablation_compares_modes():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="capital budget transit corridor",
        )
    ]
    cases = [RetrievalCase(query="capital budget", expected_pages=[1])]

    report = evaluate_retrieval_ablation(
        chunks,
        [],
        cases,
        modes=parse_ablation_modes("dense,bm25,hybrid"),
        repeat=2,
    )

    assert [row.mode.name for row in report.rows]
    assert report.best_by_recall in {"dense", "bm25", "hybrid"}
    assert report.best_by_target_coverage in {"dense", "bm25", "hybrid"}
    assert report.fastest_by_mean_latency in {"dense", "bm25", "hybrid"}
    assert all(row.evaluation.case_count == 1 for row in report.rows)
    assert all(row.evaluation.repeat == 2 for row in report.rows)


def test_evaluate_retrieval_ablation_can_measure_visual_lexical_gain():
    report = visual_lexical_ablation_report()
    rows = {row.mode.name: row for row in report.rows}

    assert rows["bm25_text"].evaluation.recall_at_k == 0.0
    assert rows["bm25_visual"].evaluation.recall_at_k == 1.0
    assert rows["bm25_text"].evaluation.metadata["include_asset_text"] is False
    assert rows["bm25_visual"].evaluation.metadata["include_asset_text"] is True
    assert report.best_by_recall == "bm25_visual"


def test_gate_retrieval_ablation_can_require_visual_lift():
    report = visual_lexical_ablation_report()

    gate = gate_retrieval_ablation(
        report,
        mode="bm25_visual",
        baseline_mode="bm25_text",
        min_recall_at_k=1.0,
        min_target_type_coverage={"asset": 1.0},
        min_source_family_target_coverage={"lexical": 1.0},
        min_recall_lift=1.0,
        min_target_coverage_lift=1.0,
        require_best_by_recall=True,
    )

    assert gate.passed is True
    assert gate.mode == "bm25_visual"
    assert gate.baseline_mode == "bm25_text"
    assert gate.metrics["recall_at_k"] == 1.0
    assert gate.baseline_metrics["recall_at_k"] == 0.0
    assert gate.failed_checks == []


def test_gate_retrieval_ablation_reports_failed_lift():
    report = visual_lexical_ablation_report()

    gate = gate_retrieval_ablation(
        report,
        mode="bm25_visual",
        baseline_mode="bm25_text",
        min_recall_lift=1.1,
    )

    assert gate.passed is False
    assert "min_recall_at_k_lift" in gate.failed_checks


def visual_lexical_ablation_report():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="reference overview",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            caption="north river corridor diagram",
        )
    ]
    cases = [RetrievalCase(query="north river corridor diagram", expected_asset_ids=["asset-1"])]
    return evaluate_retrieval_ablation(
        chunks,
        [],
        cases,
        assets=assets,
        modes=parse_ablation_modes("bm25_text,bm25_visual"),
    )


def test_eval_retrieval_cli_writes_latency_report(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    output = tmp_path / "retrieval.json"
    cases_path = tmp_path / "cases.jsonl"
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="capital budget transit corridor",
        )
    ]
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "triples.jsonl", [])
    write_jsonl(cases_path, [RetrievalCase(query="capital budget", expected_pages=[1])])

    result = CliRunner().invoke(
        app,
        [
            "eval-retrieval",
            str(cases_path),
            "--package-dir",
            str(package_dir),
            "--repeat",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["repeat"] == 2
    assert payload["mean_target_ndcg_at_k"] == 1.0
    assert payload["mean_latency_ms"] >= 0.0
    assert len(payload["results"][0]["latency_samples_ms"]) == 2


def test_eval_qdrant_retrieval_cli_writes_report(monkeypatch, tmp_path):
    output = tmp_path / "qdrant_retrieval.json"
    cases_path = tmp_path / "cases.jsonl"
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="capital budget transit corridor",
        asset_ids=["asset-1"],
    )
    write_jsonl(cases_path, [RetrievalCase(query="capital budget", expected_asset_ids=["asset-1"])])

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            assert kwargs["vector_names"] == ["text_dense"]
            assert kwargs["top_k"] == 5
            return [HybridSearchHit(chunk=chunk, score=0.8, sources=["qdrant:text_dense"])]

    def fake_prepare(**kwargs):
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": "documents",
            "selected_vectors": ["text_dense"],
            "query_encoders": {"text_dense": "default_text"},
            "query_encoder_details": {
                "text_dense": {
                    "encoder": "default text query encoder",
                    "backend": "sentence-transformers",
                    "model": "BAAI/bge-m3",
                    "dimension": 1024,
                }
            },
            "upserted": 1,
            "triples": [],
        }

    monkeypatch.setattr(cli_module, "prepare_qdrant_hybrid_search", fake_prepare)

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-retrieval",
            str(cases_path),
            "--package-dir",
            str(tmp_path),
            "--vector-names",
            "text_dense",
            "--repeat",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["backend"] == "qdrant_hybrid"
    assert payload["metadata"]["collection"] == "documents"
    assert payload["metadata"]["query_encoder_details"]["text_dense"]["model"] == "BAAI/bge-m3"
    assert payload["metadata"]["query_encoder_details"]["text_dense"]["dimension"] == 1024
    assert payload["repeat"] == 2
    assert payload["recall_at_k"] == 1.0
    assert payload["results"][0]["matched_asset_id"] == "asset-1"
    assert payload["results"][0]["top_sources"] == [["qdrant:text_dense"]]
    assert payload["source_family_metrics"]["dense_text"]["target_coverage_at_k"] == 1.0
    assert len(payload["results"][0]["latency_samples_ms"]) == 2


def test_parse_qdrant_vector_ablation_modes_returns_union():
    modes = parse_qdrant_vector_ablation_modes("text,caption,text_caption_graph")

    assert [mode.name for mode in modes] == ["text", "caption", "text_caption_graph"]
    assert modes[-1].graph_expand is True
    assert qdrant_vector_names_for_modes(modes) == ["text_dense", "caption_dense"]


def test_eval_qdrant_vector_ablation_cli_writes_report(monkeypatch, tmp_path):
    output = tmp_path / "qdrant_vector_ablation.json"
    cases_path = tmp_path / "cases.jsonl"
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual caption evidence",
        asset_ids=["asset-1"],
    )
    write_jsonl(cases_path, [RetrievalCase(query="visual evidence", expected_asset_ids=["asset-1"])])
    calls = []

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            calls.append((tuple(kwargs["vector_names"]), kwargs["graph_expand"]))
            if kwargs["vector_names"] == ["caption_dense"]:
                return [
                    HybridSearchHit(
                        chunk=chunk,
                        score=0.9,
                        sources=["qdrant:caption_dense"],
                    )
                ]
            return []

    def fake_prepare(**kwargs):
        assert kwargs["vector_names"] == "text_dense,caption_dense"
        return {
            "searcher": FakeSearcher(),
            "store": FakeStore(),
            "collection_name": "documents",
            "selected_vectors": ["text_dense", "caption_dense"],
            "query_encoders": {
                "text_dense": "default_text",
                "caption_dense": "default_text",
            },
            "query_encoder_details": {
                "text_dense": {
                    "encoder": "default text query encoder",
                    "backend": "sentence-transformers",
                    "model": "BAAI/bge-m3",
                    "dimension": 1024,
                },
                "caption_dense": {
                    "encoder": "default text query encoder",
                    "backend": "sentence-transformers",
                    "model": "BAAI/bge-m3",
                    "dimension": 1024,
                },
            },
            "upserted": 1,
            "triples": [],
        }

    monkeypatch.setattr(cli_module, "prepare_qdrant_hybrid_search", fake_prepare)

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-vector-ablation",
            str(cases_path),
            "--package-dir",
            str(tmp_path),
            "--modes",
            "text,caption",
            "--repeat",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["mode"]["name"]: row for row in payload["rows"]}
    assert payload["best_by_recall"] == "caption"
    assert payload["best_by_target_coverage"] == "caption"
    assert payload["best_by_target_ndcg"] == "caption"
    assert rows["text"]["evaluation"]["recall_at_k"] == 0.0
    assert rows["caption"]["evaluation"]["recall_at_k"] == 1.0
    assert rows["caption"]["evaluation"]["target_coverage_at_k"] == 1.0
    assert rows["caption"]["evaluation"]["source_family_metrics"]["visual"]["target_coverage_at_k"] == 1.0
    assert rows["caption"]["evaluation"]["metadata"]["vector_names"] == ["caption_dense"]
    assert rows["caption"]["evaluation"]["metadata"]["query_encoder_details"] == {
        "caption_dense": {
            "encoder": "default text query encoder",
            "backend": "sentence-transformers",
            "model": "BAAI/bge-m3",
            "dimension": 1024,
        }
    }
    assert calls.count((("caption_dense",), False)) == 2


def qdrant_vector_ablation_report_for_gate():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual evidence",
        asset_ids=["asset-1"],
    )
    cases = [RetrievalCase(query="visual evidence", expected_asset_ids=["asset-1"])]
    caption_image_eval = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=chunk,
                score=0.9,
                sources=["qdrant:caption_dense", "qdrant:image_dense"],
            )
        ],
        top_k=5,
    )
    image_eval = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    return QdrantVectorAblationReport(
        rows=[
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(
                    name="caption_image",
                    vector_names=["caption_dense", "image_dense"],
                ),
                evaluation=caption_image_eval,
            ),
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(
                    name="image",
                    vector_names=["image_dense"],
                ),
                evaluation=image_eval,
            ),
        ],
        best_by_recall="caption_image",
        best_by_target_coverage="caption_image",
        best_by_target_ndcg="caption_image",
        best_by_mrr="caption_image",
        fastest_by_mean_latency="image",
    )


def test_gate_qdrant_vector_ablation_passes_required_mode():
    report = qdrant_vector_ablation_report_for_gate()

    gate = gate_qdrant_vector_ablation(
        report,
        mode="caption_image",
        min_recall_at_k=1.0,
        min_target_coverage_at_k=1.0,
        min_target_ndcg_at_k=1.0,
        max_failed_queries=0,
        min_target_type_coverage={"asset": 1.0},
        min_source_family_target_coverage={"visual": 1.0},
        require_best_by_recall=True,
        require_best_by_target_coverage=True,
    )

    assert gate.passed
    assert gate.mode == "caption_image"
    assert gate.vector_names == ["caption_dense", "image_dense"]
    assert gate.metrics["failed_query_count"] == 0.0
    assert gate.metrics["target_type.asset.coverage_at_k"] == 1.0
    assert gate.target_metrics["asset"]["coverage_at_k"] == 1.0
    assert gate.metrics["source_family.visual.target_coverage_at_k"] == 1.0
    assert gate.source_family_metrics["visual"]["target_coverage_at_k"] == 1.0
    assert gate.failed_checks == []


def test_gate_qdrant_vector_ablation_reports_failed_checks():
    report = qdrant_vector_ablation_report_for_gate()

    gate = gate_qdrant_vector_ablation(
        report,
        mode="image",
        min_recall_at_k=1.0,
        max_failed_queries=0,
        min_target_type_coverage={"asset": 1.0},
        min_source_family_target_coverage={"visual": 1.0},
        require_best_by_recall=True,
    )

    assert not gate.passed
    assert gate.metrics["failed_query_count"] == 1.0
    assert gate.metrics["target_type.asset.coverage_at_k"] == 0.0
    assert gate.metrics["source_family.visual.target_coverage_at_k"] == 0.0
    assert set(gate.failed_checks) == {
        "min_recall_at_k",
        "max_failed_queries",
        "min_target_type_coverage:asset",
        "min_source_family_target_coverage:visual",
        "require_best_by_recall",
    }


def test_gate_qdrant_vector_ablation_cli_writes_report(tmp_path):
    report_path = tmp_path / "qdrant_vector_ablation.json"
    output = tmp_path / "qdrant_vector_ablation_gate.json"
    report_path.write_text(
        qdrant_vector_ablation_report_for_gate().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-qdrant-vector-ablation",
            str(report_path),
            "--mode",
            "caption_image",
            "--min-recall-at-k",
            "1.0",
            "--min-target-coverage-at-k",
            "1.0",
            "--max-failed-queries",
            "0",
            "--min-target-type-coverage",
            "asset=1.0",
            "--min-source-family-target-coverage",
            "visual=1.0",
            "--require-best-by-recall",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["mode"] == "caption_image"
    assert payload["vector_names"] == ["caption_dense", "image_dense"]
    assert payload["target_metrics"]["asset"]["coverage_at_k"] == 1.0
    assert payload["source_family_metrics"]["visual"]["target_coverage_at_k"] == 1.0


def test_gate_qdrant_vector_ablation_cli_can_report_without_failing(tmp_path):
    report_path = tmp_path / "qdrant_vector_ablation.json"
    output = tmp_path / "qdrant_vector_ablation_gate.json"
    report_path.write_text(
        qdrant_vector_ablation_report_for_gate().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-qdrant-vector-ablation",
            str(report_path),
            "--mode",
            "image",
            "--min-recall-at-k",
            "1.0",
            "--max-failed-queries",
            "0",
            "--min-target-type-coverage",
            "asset=1.0",
            "--min-source-family-target-coverage",
            "visual=1.0",
            "--require-best-by-recall",
            "--no-fail",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert set(payload["failed_checks"]) == {
        "min_recall_at_k",
        "max_failed_queries",
        "min_target_type_coverage:asset",
        "min_source_family_target_coverage:visual",
        "require_best_by_recall",
    }


def test_eval_retrieval_ablation_cli_writes_report(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    output = tmp_path / "ablation.json"
    cases_path = tmp_path / "cases.jsonl"
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="capital budget transit corridor",
        )
    ]
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "triples.jsonl", [])
    write_jsonl(cases_path, [RetrievalCase(query="capital budget", expected_pages=[1])])

    result = CliRunner().invoke(
        app,
        [
            "eval-retrieval-ablation",
            str(cases_path),
            "--package-dir",
            str(package_dir),
            "--modes",
            "bm25,hybrid",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_by_recall"] in {"bm25", "hybrid"}
    assert payload["best_by_target_coverage"] in {"bm25", "hybrid"}
    assert payload["best_by_target_ndcg"] in {"bm25", "hybrid"}
    assert payload["fastest_by_mean_latency"] in {"bm25", "hybrid"}
    assert {row["mode"]["name"] for row in payload["rows"]} == {"bm25", "hybrid"}


def test_eval_retrieval_ablation_cli_compares_visual_lexical_modes(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    output = tmp_path / "ablation.json"
    cases_path = tmp_path / "cases.jsonl"
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="reference overview",
            asset_ids=["asset-1"],
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            caption="north river corridor diagram",
        )
    ]
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    write_jsonl(package_dir / "assets.jsonl", assets)
    write_jsonl(package_dir / "triples.jsonl", [])
    write_jsonl(cases_path, [RetrievalCase(query="north river corridor diagram", expected_asset_ids=["asset-1"])])

    result = CliRunner().invoke(
        app,
        [
            "eval-retrieval-ablation",
            str(cases_path),
            "--package-dir",
            str(package_dir),
            "--modes",
            "bm25_text,bm25_visual",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["mode"]["name"]: row for row in payload["rows"]}
    assert rows["bm25_text"]["evaluation"]["recall_at_k"] == 0.0
    assert rows["bm25_visual"]["evaluation"]["recall_at_k"] == 1.0
    assert payload["best_by_recall"] == "bm25_visual"


def test_gate_retrieval_ablation_cli_writes_lift_report(tmp_path):
    report_path = tmp_path / "ablation.json"
    output = tmp_path / "ablation_gate.json"
    report_path.write_text(
        visual_lexical_ablation_report().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval-ablation",
            str(report_path),
            "--mode",
            "bm25_visual",
            "--baseline-mode",
            "bm25_text",
            "--min-recall-at-k",
            "1.0",
            "--min-recall-lift",
            "1.0",
            "--min-target-type-coverage",
            "asset=1.0",
            "--min-source-family-target-coverage",
            "lexical=1.0",
            "--require-best-by-recall",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["mode"] == "bm25_visual"
    assert payload["baseline_mode"] == "bm25_text"
    assert payload["metrics"]["recall_at_k"] == 1.0
    assert payload["baseline_metrics"]["recall_at_k"] == 0.0

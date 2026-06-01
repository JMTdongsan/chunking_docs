import hashlib
import json
import math

import pytest
from typer.testing import CliRunner

import chunking_docs.cli as cli_module
from chunking_docs.cli import app
from chunking_docs.evaluation.audit import (
    audit_package,
    degraded_page_ratio,
    normalize_payload_indexes,
    required_payload_indexes,
)
from chunking_docs.evaluation.ablation import (
    QdrantRerankerAblationMode,
    QdrantRerankerAblationRow,
    QdrantVectorAblationMode,
    QdrantVectorAblationRow,
    build_qdrant_reranker_ablation_report,
    build_qdrant_vector_ablation_report,
    evaluate_retrieval_ablation,
    gate_qdrant_reranker_ablation,
    gate_retrieval_ablation,
    gate_qdrant_vector_ablation,
    parse_ablation_modes,
    parse_qdrant_reranker_ablation_modes,
    parse_qdrant_vector_ablation_modes,
    qdrant_vector_names_for_modes,
)
from chunking_docs.evaluation.retrieval import (
    RetrievalCase,
    RetrievalEvaluation,
    evaluate_retrieval,
    evaluate_search_results,
)
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


def test_required_payload_indexes_include_chunk_strategy_fields():
    fields = required_payload_indexes()

    assert normalize_payload_indexes([]) == set()
    assert {
        "chunking_strategy",
        "retrieval_role",
        "parent_chunk_id",
        "source_chunk_id",
        "hierarchical_parent_chunk_id",
        "visual_asset_unlinked",
        "text_quality",
        "text_quality_reasons",
        "requires_ocr",
        "requires_vlm",
        "asset_scope",
        "parent_asset_id",
        "tile_index",
        "object_id",
        "label",
        "bbox_region",
        "source_key",
        "control_char_ratio",
        "section.chapter",
        "section.issue",
    }.issubset(fields)


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


def test_audit_package_counts_source_ref_visual_asset_links():
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual context",
        source_refs=["asset:asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
    )

    audit = audit_package([], [chunk], [asset], [])

    assert audit.chunks_with_assets == 1


def test_audit_package_accepts_asset_backed_graph_triples():
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual context",
        source_refs=["asset:asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
    )
    triple = GraphTriple(
        triple_id="visual",
        doc_id="doc",
        chunk_id="vlm-annotation",
        subject="diagram",
        predicate="depicts",
        object="process",
        qualifiers={"asset_id": "asset"},
    )

    audit = audit_package([], [chunk], [asset], [triple])

    assert audit.passed
    assert "orphan_triples" not in {issue.code for issue in audit.issues}


def test_audit_package_warns_when_vlm_metadata_triples_are_missing():
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual context",
        asset_ids=["asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.PAGE_IMAGE,
        metadata={
            "title": "map panel",
            "entities": ["station marker"],
            "objects": [{"label": "route line"}],
        },
    )

    audit = audit_package([], [chunk], [asset], [])

    issue = next(issue for issue in audit.issues if issue.code == "missing_visual_derived_triples")
    assert audit.passed
    assert issue.severity == "warning"
    assert issue.metadata["assets"][0]["missing_triple_count"] == 2


def test_audit_package_can_require_vlm_metadata_triples():
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual context",
        asset_ids=["asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.PAGE_IMAGE,
        metadata={"title": "map panel", "visual_elements": ["legend"]},
    )

    audit = audit_package([], [chunk], [asset], [], require_visual_derived_triples=True)

    issue = next(issue for issue in audit.issues if issue.code == "missing_visual_derived_triples")
    assert not audit.passed
    assert issue.severity == "error"


def test_audit_package_accepts_vlm_metadata_triples_with_asset_provenance():
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual context",
        asset_ids=["asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.PAGE_IMAGE,
        metadata={"title": "map panel", "entities": ["station marker"]},
    )
    triple = GraphTriple(
        triple_id="visual-entity",
        doc_id="doc",
        chunk_id="chunk",
        subject="map panel",
        predicate="mentions_entity",
        object="station marker",
        qualifiers={"asset_id": "asset", "derived_from_vlm_field": True, "source_field": "entities"},
    )

    audit = audit_package([], [chunk], [asset], [triple], require_visual_derived_triples=True)

    assert audit.passed
    assert "missing_visual_derived_triples" not in {issue.code for issue in audit.issues}


def test_audit_package_uses_asset_caption_for_vlm_metadata_triple_subject():
    chunk = DocumentChunk(
        chunk_id="chunk",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual context",
        asset_ids=["asset"],
    )
    asset = VisualAsset(
        asset_id="asset",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.PAGE_IMAGE,
        caption="map panel",
        metadata={"entities": ["station marker"]},
    )
    triple = GraphTriple(
        triple_id="visual-entity",
        doc_id="doc",
        chunk_id="chunk",
        subject="map panel",
        predicate="mentions_entity",
        object="station marker",
        qualifiers={"asset_id": "asset"},
    )

    audit = audit_package([], [chunk], [asset], [triple], require_visual_derived_triples=True)

    assert audit.passed
    assert "missing_visual_derived_triples" not in {issue.code for issue in audit.issues}


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


def test_audit_package_validates_qdrant_triple_records(tmp_path):
    chunk = DocumentChunk(
        chunk_id="chunk-a",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual graph evidence",
    )
    triples = [
        GraphTriple(
            triple_id="triple-a",
            doc_id="doc",
            chunk_id="chunk-a",
            subject="map panel",
            predicate="mentions_entity",
            object="station marker",
        ),
        GraphTriple(
            triple_id="triple-b",
            doc_id="doc",
            chunk_id="chunk-a",
            subject="map panel",
            predicate="contains_object",
            object="route line",
        ),
    ]
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "documents",
                "named_vectors": {"triple_dense": {"size": 2}},
                "payload_indexes": [{"field": "triple_id", "schema": "keyword"}],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "qdrant_triple_records.jsonl",
        [
            EmbeddingRecord(
                point_id="triple-a",
                chunk_id="chunk-a",
                doc_id="doc",
                vector_name="triple_dense",
                vector=[1.0, 0.0],
                payload={
                    "triple_id": "triple-a",
                    "chunk_id": "chunk-a",
                    "doc_id": "doc",
                    "kind": "text",
                    "record_kind": "graph_triple",
                    "page_start": 1,
                    "page_end": 1,
                    "subject": "map panel",
                    "predicate": "mentions_entity",
                    "object": "station marker",
                    "text": "stale graph text",
                },
            )
        ],
    )

    audit = audit_package([], [chunk], [], triples, package_dir=tmp_path, require_qdrant_records=True)
    codes = {issue.code for issue in audit.issues}

    assert not audit.passed
    assert "qdrant_missing_triple_records" in codes
    assert "qdrant_stale_triple_payload_text" in codes


def test_audit_package_validates_qdrant_object_records(tmp_path):
    asset = VisualAsset(
        asset_id="asset-a",
        doc_id="doc",
        page_no=2,
        kind=AssetKind.FIGURE,
        caption="marker diagram",
        metadata={
            "objects": [
                {
                    "label": "transfer marker",
                    "attributes": ["blue square"],
                    "bbox": [0.6, 0.1, 0.8, 0.3],
                },
                {"label": "route line"},
            ]
        },
    )
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "documents",
                "named_vectors": {"object_dense": {"size": 2}},
                "payload_indexes": [{"field": "object_id", "schema": "keyword"}],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "qdrant_object_records.jsonl",
        [
            EmbeddingRecord(
                point_id="object-a",
                chunk_id="asset-a",
                doc_id="doc",
                vector_name="object_dense",
                vector=[1.0, 0.0],
                payload={
                    "record_kind": "visual_object",
                    "object_id": "asset-a:object:0",
                    "asset_id": "asset-a",
                    "doc_id": "doc",
                    "page_no": 2,
                    "kind": "figure",
                    "label": "transfer marker",
                    "object_index": 0,
                    "source_key": "objects",
                    "bbox_region": "upper right",
                    "text": "old object text",
                },
            )
        ],
    )

    audit = audit_package([], [], [asset], [], package_dir=tmp_path, require_qdrant_records=True)
    codes = {issue.code for issue in audit.issues}

    assert not audit.passed
    assert "qdrant_missing_visual_object_records" in codes
    assert "qdrant_stale_visual_object_payload_text" in codes


def test_audit_package_detects_stale_qdrant_payload_text(tmp_path):
    image_path = tmp_path / "asset.png"
    image_path.write_bytes(b"image")
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
            text="current chunk evidence",
        )
    ]
    assets = [
        VisualAsset(
            asset_id="asset-a",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.FIGURE,
            path=image_path,
            caption="current caption",
            ocr_text="current OCR text",
            vlm_summary="current VLM object summary",
            metadata={"objects": [{"label": "current object", "description": "visible marker"}]},
        )
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
                    {"field": field, "schema": "keyword"}
                    for field in sorted(required_payload_indexes())
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
                    "text": "old chunk evidence",
                },
            )
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
                vector=[0.25, 0.75],
                payload={
                    "asset_id": "asset-a",
                    "doc_id": "doc",
                    "page_no": 1,
                    "kind": "figure",
                    "caption": "current caption",
                    "ocr_text": "old OCR text",
                    "vlm_summary": "old VLM object summary",
                    "objects": [{"label": "old object", "description": "old marker"}],
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
                vector=[0.5, 0.5],
                payload={
                    "asset_id": "asset-a",
                    "doc_id": "doc",
                    "page_no": 1,
                    "kind": "figure",
                    "text": "old visual summary",
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
    assert "qdrant_stale_chunk_payload_text" in codes
    assert "qdrant_stale_image_asset_payload_fields" in codes
    assert "qdrant_stale_caption_asset_payload_text" in codes
    assert "qdrant_stale_caption_asset_payload_fields" in codes
    stale_image = next(
        issue for issue in audit.issues if issue.code == "qdrant_stale_image_asset_payload_fields"
    )
    assert stale_image.metadata["mismatches"][0]["fields"] == [
        "objects",
        "ocr_text",
        "vlm_summary",
    ]
    stale_caption = next(
        issue for issue in audit.issues if issue.code == "qdrant_stale_caption_asset_payload_text"
    )
    assert stale_caption.metadata["mismatches"][0]["id"] == "asset-a"


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
    assert result.unstable_result_count == 0
    assert result.result_stability_rate == 1.0
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
    assert result.results[0].result_consistent is True
    assert result.results[0].distinct_result_sets == 1
    assert result.results[0].matched_rank == 1
    assert result.results[0].matched_page == 12


def test_evaluate_search_results_reports_unstable_repeated_results():
    chunk_a = DocumentChunk(
        chunk_id="a",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="alpha",
    )
    chunk_b = DocumentChunk(
        chunk_id="b",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="alpha alternative",
    )
    calls = 0

    def alternating_search(case, graph_expand):
        nonlocal calls
        calls += 1
        chunk = chunk_a if calls % 2 else chunk_b
        return [HybridSearchHit(chunk=chunk, score=1.0, sources=["test"])]

    result = evaluate_search_results(
        cases=[RetrievalCase(query="alpha", expected_pages=[1])],
        search_fn=alternating_search,
        top_k=1,
        repeat=2,
    )

    assert result.unstable_result_count == 1
    assert result.result_stability_rate == 0.0
    assert result.results[0].result_consistent is False
    assert result.results[0].distinct_result_sets == 2


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
                metadata={"case_source": "visual_lexical_probe", "query_mode": "salient_terms"},
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
    assert (
        result.case_group_source_metrics["case_source"]["visual_lexical_probe"]["test"]
        .target_coverage_at_k
        == 0.75
    )
    assert (
        result.case_group_source_family_metrics["case_source"]["visual_lexical_probe"]["test"]
        .target_coverage_at_k
        == 0.75
    )
    assert result.case_group_metrics["case_source"]["visual_lexical_probe"].case_count == 1
    assert (
        result.case_group_metrics["case_source"]["visual_lexical_probe"].target_coverage_at_k
        == 0.75
    )
    assert result.case_group_metrics["query_mode"]["salient_terms"].ndcg_at_k == pytest.approx(
        case_result.target_ndcg_at_k
    )
    assert result.case_group_metrics["graph_expand"]["false"].precision_at_k == 2 / 3


def test_evaluate_search_results_reports_excluded_target_hits():
    chunk_a = DocumentChunk(
        chunk_id="a",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="alpha",
    )
    chunk_b = DocumentChunk(
        chunk_id="b",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="beta",
        asset_ids=["asset-b"],
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
                query="hard negative target",
                expected_pages=[1],
                excluded_pages=[2],
                excluded_chunk_ids=["b"],
                excluded_asset_ids=["asset-b"],
            )
        ],
        search_fn=lambda case, graph_expand: [Hit(chunk_a), Hit(chunk_b)],
        top_k=2,
    )

    case_result = result.results[0]
    assert case_result.passed is False
    assert result.passed_count == 0
    assert result.recall_at_k == 1.0
    assert case_result.excluded_target_count == 3
    assert case_result.excluded_matched_target_count == 3
    assert case_result.excluded_hit_count == 1
    assert case_result.excluded_target_hit_rate == 1.0
    assert case_result.top_excluded_targets == [
        [],
        ["page:2", "chunk:b", "asset:asset-b"],
    ]
    assert result.excluded_query_count == 1
    assert result.excluded_hit_query_count == 1
    assert result.excluded_query_hit_rate == 1.0
    assert result.excluded_target_count == 3
    assert result.excluded_matched_target_count == 3
    assert result.excluded_target_hit_rate == 1.0
    assert result.source_metrics["test"].excluded_query_count == 1
    assert result.source_metrics["test"].excluded_hit_count == 1
    assert result.source_metrics["test"].excluded_target_count == 3
    assert result.source_metrics["test"].excluded_matched_target_count == 3
    assert result.source_metrics["test"].excluded_precision_at_hits == 0.5
    assert result.source_metrics["test"].excluded_target_hit_rate == 1.0
    assert result.source_family_metrics["test"].excluded_target_hit_rate == 1.0
    assert result.chunk_strategy_metrics["unspecified"].excluded_query_count == 1
    assert result.chunk_strategy_metrics["unspecified"].excluded_hit_count == 1
    assert result.chunk_strategy_metrics["unspecified"].excluded_target_hit_rate == 1.0
    assert result.retrieval_role_metrics["unspecified"].excluded_target_hit_rate == 1.0
    assert result.failed_queries == ["hard negative target"]
    assert result.case_group_metrics["graph_expand"]["false"].recall_at_k == 1.0
    assert result.case_group_metrics["graph_expand"]["false"].passed_count == 0


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


def test_evaluate_search_results_reports_chunk_strategy_metrics():
    chunk = DocumentChunk(
        chunk_id="visual-chunk",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.FIGURE,
        text="station access map",
        asset_ids=["asset-map"],
        metadata={
            "chunking_strategy": "visual_asset_text",
            "retrieval_role": "child",
        },
    )

    class PayloadHit:
        def __init__(self):
            self.chunk = chunk
            self.sources = ["qdrant:text_dense"]
            self.evidence_chunks = []
            self.payloads = []

    result = evaluate_search_results(
        cases=[RetrievalCase(query="station access map", expected_asset_ids=["asset-map"])],
        search_fn=lambda case, graph_expand: [PayloadHit()],
    )

    assert result.results[0].top_chunking_strategies == [["visual_asset_text"]]
    assert result.results[0].top_retrieval_roles == [["child"]]
    assert result.chunk_strategy_metrics["visual_asset_text"].target_coverage_at_k == 1.0
    assert result.retrieval_role_metrics["child"].target_coverage_at_k == 1.0


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


def test_evaluate_search_results_matches_triple_id_from_payload():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=6,
        page_end=6,
        kind=ChunkKind.TEXT,
        text="station access",
    )

    class PayloadHit:
        def __init__(self):
            self.chunk = chunk
            self.sources = ["qdrant:triple_dense"]
            self.evidence_chunks = []
            self.payloads = [{"triple_id": "triple-map"}]

    result = evaluate_search_results(
        cases=[RetrievalCase(query="station access map", expected_triple_ids=["triple-map"])],
        search_fn=lambda case, graph_expand: [PayloadHit()],
    )

    assert result.recall_at_k == 1.0
    assert result.results[0].top_triple_ids == [["triple-map"]]
    assert result.results[0].top_matched_targets == [["triple:triple-map"]]
    assert result.results[0].matched_triple_id == "triple-map"
    assert result.source_metrics["qdrant:triple_dense"].target_coverage_at_k == 1.0
    assert result.source_family_metrics["graph"].target_coverage_at_k == 1.0


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
    visual_probe_best = report.case_group_best_modes["case_source"]["visual_lexical_probe"]
    assert visual_probe_best["target_coverage_at_k"].mode == "bm25_visual"
    assert visual_probe_best["target_coverage_at_k"].value == 1.0
    assert visual_probe_best["ndcg_at_k"].mode == "bm25_visual"
    pairwise = next(
        comparison
        for comparison in report.pairwise
        if comparison.candidate == "bm25_visual" and comparison.baseline == "bm25_text"
    )
    assert pairwise.shared_query_count == 1
    assert pairwise.candidate_win_rate == 1.0
    assert pairwise.mean_target_coverage_delta == 1.0
    assert pairwise.mean_target_rank_delta == -5.0
    assert pairwise.target_coverage_delta_ci_low == 1.0
    assert pairwise.target_rank_delta_ci_high == -5.0


def test_gate_retrieval_ablation_can_require_visual_lift():
    report = visual_lexical_ablation_report()

    gate = gate_retrieval_ablation(
        report,
        mode="bm25_visual",
        baseline_mode="bm25_text",
        min_recall_at_k=1.0,
        min_target_type_coverage={"asset": 1.0},
        min_source_target_coverage={"bm25": 1.0},
        min_source_family_target_coverage={"lexical": 1.0},
        min_source_precision_at_hits={"bm25": 1.0},
        min_source_family_precision_at_hits={"lexical": 1.0},
        min_case_group_target_coverage={"case_source:visual_lexical_probe": 1.0},
        min_case_group_source_target_coverage={
            "case_source:visual_lexical_probe:bm25": 1.0
        },
        min_case_group_source_family_target_coverage={
            "case_source:visual_lexical_probe:lexical": 1.0
        },
        min_case_group_source_precision_at_hits={
            "case_source:visual_lexical_probe:bm25": 1.0
        },
        min_case_group_source_family_precision_at_hits={
            "case_source:visual_lexical_probe:lexical": 1.0
        },
        max_mean_target_rank=1.0,
        min_recall_lift=1.0,
        min_target_coverage_lift=1.0,
        min_pairwise_shared_queries=1,
        min_pairwise_win_rate=1.0,
        min_pairwise_target_coverage_lift=1.0,
        min_pairwise_target_coverage_ci_low=1.0,
        max_pairwise_mean_target_rank_delta=0.0,
        max_pairwise_target_rank_delta_ci_high=0.0,
        require_best_by_recall=True,
    )

    assert gate.passed is True
    assert gate.mode == "bm25_visual"
    assert gate.baseline_mode == "bm25_text"
    assert gate.metrics["recall_at_k"] == 1.0
    assert gate.metrics["mean_target_rank"] == 1.0
    assert gate.baseline_metrics["recall_at_k"] == 0.0
    assert gate.baseline_metrics["mean_target_rank"] == 6.0
    assert gate.metrics["source.bm25.target_coverage_at_k"] == 1.0
    assert gate.metrics["source.bm25.precision_at_hits"] == 1.0
    assert gate.source_metrics["bm25"]["target_coverage_at_k"] == 1.0
    assert gate.source_metrics["bm25"]["precision_at_hits"] == 1.0
    assert gate.metrics["source_family.lexical.precision_at_hits"] == 1.0
    assert gate.source_family_metrics["lexical"]["precision_at_hits"] == 1.0
    assert "bm25" not in gate.baseline_source_metrics
    assert gate.metrics["chunk_strategy.visual_asset_text.target_coverage_at_k"] == 1.0
    assert gate.metrics["retrieval_role.child.target_coverage_at_k"] == 1.0
    assert gate.metrics[
        "case_group.case_source.visual_lexical_probe.target_coverage_at_k"
    ] == 1.0
    assert gate.metrics[
        "case_group_source.case_source.visual_lexical_probe.bm25.target_coverage_at_k"
    ] == 1.0
    assert gate.metrics[
        "case_group_source.case_source.visual_lexical_probe.bm25.precision_at_hits"
    ] == 1.0
    assert gate.metrics[
        "case_group_source_family.case_source.visual_lexical_probe.lexical.target_coverage_at_k"
    ] == 1.0
    assert gate.metrics[
        "case_group_source_family.case_source.visual_lexical_probe.lexical.precision_at_hits"
    ] == 1.0
    assert gate.chunk_strategy_metrics["visual_asset_text"]["target_coverage_at_k"] == 1.0
    assert gate.retrieval_role_metrics["child"]["target_coverage_at_k"] == 1.0
    assert gate.case_group_metrics["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ] == 1.0
    assert gate.case_group_source_metrics["case_source"]["visual_lexical_probe"]["bm25"][
        "target_coverage_at_k"
    ] == 1.0
    assert gate.case_group_source_metrics["case_source"]["visual_lexical_probe"]["bm25"][
        "precision_at_hits"
    ] == 1.0
    assert gate.case_group_source_family_metrics["case_source"]["visual_lexical_probe"][
        "lexical"
    ]["target_coverage_at_k"] == 1.0
    assert gate.case_group_source_family_metrics["case_source"]["visual_lexical_probe"][
        "lexical"
    ]["precision_at_hits"] == 1.0
    assert gate.case_group_best_modes["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ].mode == "bm25_visual"
    assert gate.pairwise_metrics["pairwise_candidate_win_rate"] == 1.0
    assert gate.pairwise_metrics["pairwise_mean_target_coverage_delta"] == 1.0
    assert gate.pairwise_metrics["pairwise_mean_target_rank_delta"] == -5.0
    assert gate.pairwise_metrics["pairwise_target_rank_delta_ci_high"] == -5.0
    assert {check.name for check in gate.checks if check.name.startswith("min_pairwise")} == {
        "min_pairwise_shared_queries",
        "min_pairwise_win_rate",
        "min_pairwise_target_coverage_lift",
        "min_pairwise_target_coverage_ci_low",
    }
    assert {check.name for check in gate.checks if check.name.startswith("max_pairwise")} == {
        "max_pairwise_mean_target_rank_delta",
        "max_pairwise_target_rank_delta_ci_high",
    }
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


def test_gate_retrieval_ablation_reports_failed_rank_gate():
    report = visual_lexical_ablation_report()

    gate = gate_retrieval_ablation(
        report,
        mode="bm25_text",
        max_mean_target_rank=5.0,
    )

    assert gate.passed is False
    assert gate.metrics["mean_target_rank"] == 6.0
    assert gate.failed_checks == ["max_mean_target_rank"]


def test_gate_retrieval_ablation_checks_excluded_target_hits():
    report = retrieval_ablation_report_with_excluded_hits()

    gate = gate_retrieval_ablation(
        report,
        mode="bm25",
        max_excluded_target_hit_rate=0.0,
        max_excluded_query_hit_rate=0.0,
        max_excluded_hit_query_count=0,
    )

    assert gate.passed is False
    assert gate.metrics["excluded_target_hit_rate"] == 1.0
    assert gate.metrics["excluded_query_hit_rate"] == 1.0
    assert gate.metrics["excluded_hit_query_count"] == 1.0
    assert set(gate.failed_checks) == {
        "max_excluded_target_hit_rate",
        "max_excluded_query_hit_rate",
        "max_excluded_hit_query_count",
    }


def test_gate_retrieval_ablation_checks_source_excluded_target_hits():
    report = retrieval_ablation_report_with_excluded_hits()

    gate = gate_retrieval_ablation(
        report,
        mode="bm25",
        max_source_excluded_target_hit_rate={"bm25": 0.0},
        max_source_family_excluded_target_hit_rate={"lexical": 0.0},
        max_chunk_strategy_excluded_target_hit_rate={"visual_asset_text": 0.0},
        max_retrieval_role_excluded_target_hit_rate={"child": 0.0},
    )

    assert gate.passed is False
    assert gate.metrics["source.bm25.excluded_target_hit_rate"] == 1.0
    assert gate.metrics["source_family.lexical.excluded_target_hit_rate"] == 1.0
    assert gate.metrics["chunk_strategy.visual_asset_text.excluded_target_hit_rate"] == 1.0
    assert gate.metrics["retrieval_role.child.excluded_target_hit_rate"] == 1.0
    assert set(gate.failed_checks) == {
        "max_source_excluded_target_hit_rate:bm25",
        "max_source_family_excluded_target_hit_rate:lexical",
        "max_chunk_strategy_excluded_target_hit_rate:visual_asset_text",
        "max_retrieval_role_excluded_target_hit_rate:child",
    }


def test_gate_retrieval_ablation_reports_failed_pairwise_check():
    report = visual_lexical_ablation_report()

    gate = gate_retrieval_ablation(
        report,
        mode="bm25_text",
        baseline_mode="bm25_visual",
        min_pairwise_win_rate=0.5,
        max_pairwise_mean_target_rank_delta=0.0,
    )

    assert gate.passed is False
    assert gate.pairwise_metrics["pairwise_candidate_win_rate"] == 0.0
    assert gate.pairwise_metrics["pairwise_mean_target_rank_delta"] == 5.0
    assert "min_pairwise_win_rate" in gate.failed_checks
    assert "max_pairwise_mean_target_rank_delta" in gate.failed_checks


def test_gate_retrieval_ablation_requires_baseline_for_pairwise_checks():
    report = visual_lexical_ablation_report()

    with pytest.raises(ValueError, match="baseline mode"):
        gate_retrieval_ablation(
            report,
            mode="bm25_visual",
            min_pairwise_win_rate=0.5,
        )


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
            metadata={
                "chunking_strategy": "visual_asset_text",
                "retrieval_role": "child",
            },
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
    cases = [
        RetrievalCase(
            query="north river corridor diagram",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_lexical_probe"},
        )
    ]
    return evaluate_retrieval_ablation(
        chunks,
        [],
        cases,
        assets=assets,
        modes=parse_ablation_modes("bm25_text,bm25_visual"),
    )


def retrieval_ablation_report_with_excluded_hits():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="north river corridor diagram",
            asset_ids=["asset-1"],
            metadata={"chunking_strategy": "visual_asset_text", "retrieval_role": "child"},
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="north river corridor diagram",
            asset_ids=["asset-2"],
            metadata={"chunking_strategy": "visual_asset_text", "retrieval_role": "child"},
        ),
    ]
    cases = [
        RetrievalCase(
            query="north river corridor diagram",
            expected_asset_ids=["asset-1"],
            excluded_asset_ids=["asset-2"],
        )
    ]
    return evaluate_retrieval_ablation(
        chunks,
        [],
        cases,
        modes=parse_ablation_modes("bm25"),
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
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="capital budget",
                expected_pages=[1],
                metadata={"case_source": "page", "query_mode": "snippet"},
            )
        ],
    )

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
    assert payload["case_group_metrics"]["case_source"]["page"]["target_coverage_at_k"] == 1.0
    assert payload["case_group_metrics"]["query_mode"]["snippet"]["recall_at_k"] == 1.0
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
    modes = parse_qdrant_vector_ablation_modes("text,caption,text_object_graph,triple")

    assert [mode.name for mode in modes] == ["text", "caption", "text_object_graph", "triple"]
    assert modes[-2].graph_expand is True
    assert qdrant_vector_names_for_modes(modes) == [
        "text_dense",
        "caption_dense",
        "object_dense",
        "triple_dense",
    ]


def test_parse_qdrant_reranker_ablation_modes_defaults_to_none_and_lexical():
    modes = parse_qdrant_reranker_ablation_modes("")

    assert [mode.name for mode in modes] == ["none", "lexical"]
    assert modes[0].reranker == "none"
    assert modes[1].reranker == "lexical"
    assert modes[1].rerank_top_k == 0


def test_build_qdrant_reranker_ablation_report_ranks_by_retrieval_quality():
    none_mode, lexical_mode = parse_qdrant_reranker_ablation_modes("none,lexical")

    report = build_qdrant_reranker_ablation_report(
        [
            QdrantRerankerAblationRow(
                mode=none_mode,
                evaluation=RetrievalEvaluation(
                    case_count=1,
                    expected_case_count=1,
                    passed_count=0,
                    failed_count=1,
                    hit_rate=0.0,
                    recall_at_k=0.0,
                    mrr=0.0,
                    target_coverage_at_k=0.0,
                    mean_target_ndcg_at_k=0.0,
                    top_k=5,
                    failed_queries=["visual evidence"],
                    results=[],
                ),
            ),
            QdrantRerankerAblationRow(
                mode=lexical_mode.model_copy(update={"rerank_top_k": 20}),
                evaluation=RetrievalEvaluation(
                    case_count=1,
                    expected_case_count=1,
                    passed_count=1,
                    failed_count=0,
                    hit_rate=1.0,
                    recall_at_k=1.0,
                    mrr=1.0,
                    target_coverage_at_k=1.0,
                    mean_target_ndcg_at_k=1.0,
                    top_k=5,
                    failed_queries=[],
                    results=[],
                ),
            ),
        ]
    )

    assert report.best_by_recall == "lexical"
    assert report.best_by_target_coverage == "lexical"
    assert report.best_by_target_ndcg == "lexical"
    assert report.best_by_mrr == "lexical"
    assert report.rows[0].mode.name == "lexical"
    assert report.rows[0].mode.rerank_top_k == 20


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
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="visual evidence",
                expected_asset_ids=["asset-1"],
                metadata={"case_source": "visual_object_probe"},
            )
        ],
    )
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
    assert (
        payload["case_group_best_modes"]["case_source"]["visual_object_probe"][
            "target_coverage_at_k"
        ]["mode"]
        == "caption"
    )
    assert rows["text"]["evaluation"]["recall_at_k"] == 0.0
    assert rows["caption"]["evaluation"]["recall_at_k"] == 1.0
    assert rows["caption"]["evaluation"]["target_coverage_at_k"] == 1.0
    assert rows["caption"]["evaluation"]["source_family_metrics"]["visual"]["target_coverage_at_k"] == 1.0
    assert (
        rows["caption"]["evaluation"]["case_group_metrics"]["case_source"][
            "visual_object_probe"
        ]["target_coverage_at_k"]
        == 1.0
    )
    assert rows["caption"]["evaluation"]["metadata"]["vector_names"] == ["caption_dense"]
    assert rows["caption"]["evaluation"]["metadata"]["query_encoder_details"] == {
        "caption_dense": {
            "encoder": "default text query encoder",
            "backend": "sentence-transformers",
            "model": "BAAI/bge-m3",
            "dimension": 1024,
        }
    }
    pairwise = next(
        comparison
        for comparison in payload["pairwise"]
        if comparison["candidate"] == "caption" and comparison["baseline"] == "text"
    )
    assert pairwise["shared_query_count"] == 1
    assert pairwise["candidate_win_rate"] == 1.0
    assert pairwise["mean_target_coverage_delta"] == 1.0
    assert calls.count((("caption_dense",), False)) == 2


def test_eval_qdrant_reranker_ablation_cli_writes_report(monkeypatch, tmp_path):
    output = tmp_path / "qdrant_reranker_ablation.json"
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
    write_jsonl(
        cases_path,
        [
            RetrievalCase(
                query="visual evidence",
                expected_asset_ids=["asset-1"],
                metadata={"case_source": "visual_object_probe"},
            )
        ],
    )
    calls = []

    class FakeStore:
        def count(self):
            return 1

    class FakeSearcher:
        def search(self, **kwargs):
            calls.append(kwargs)
            if kwargs["reranker"] is None:
                return []
            return [
                HybridSearchHit(
                    chunk=chunk,
                    score=0.9,
                    sources=["qdrant:caption_dense", "rerank:lexical"],
                )
            ]

    def fake_prepare(**kwargs):
        assert kwargs["vector_names"] == "text_dense,caption_dense"
        assert kwargs["ngram_max"] == 3
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
                "text_dense": {"backend": "hashing"},
                "caption_dense": {"backend": "hashing"},
            },
            "upserted": 1,
            "triples": [],
        }

    monkeypatch.setattr(cli_module, "prepare_qdrant_hybrid_search", fake_prepare)

    result = CliRunner().invoke(
        app,
        [
            "eval-qdrant-reranker-ablation",
            str(cases_path),
            "--package-dir",
            str(tmp_path),
            "--vector-names",
            "text_dense,caption_dense",
            "--modes",
            "none,lexical",
            "--rerank-top-k",
            "7",
            "--ngram-max",
            "3",
            "--fusion-weight",
            "bm25=1.5",
            "--repeat",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 4
    assert calls[0]["reranker"] is None
    assert calls[0]["rerank_top_k"] is None
    assert calls[0]["fusion_weights"] == {"bm25": 1.5}
    assert calls[0]["vector_names"] == ["text_dense", "caption_dense"]
    assert calls[2]["reranker"].source == "rerank:lexical"
    assert calls[2]["rerank_top_k"] == 7

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["mode"]["name"]: row for row in payload["rows"]}
    assert payload["best_by_recall"] == "lexical"
    assert payload["best_by_target_coverage"] == "lexical"
    assert payload["best_by_target_ndcg"] == "lexical"
    assert rows["none"]["evaluation"]["recall_at_k"] == 0.0
    assert rows["lexical"]["mode"]["rerank_top_k"] == 7
    assert rows["lexical"]["evaluation"]["recall_at_k"] == 1.0
    assert rows["lexical"]["evaluation"]["metadata"]["reranker"] == "rerank:lexical"
    assert rows["lexical"]["evaluation"]["metadata"]["rerank_top_k"] == 7
    assert rows["lexical"]["evaluation"]["metadata"]["fusion_weights"] == {"bm25": 1.5}
    pairwise = next(
        comparison
        for comparison in payload["pairwise"]
        if comparison["candidate"] == "lexical" and comparison["baseline"] == "none"
    )
    assert pairwise["shared_query_count"] == 1
    assert pairwise["candidate_win_rate"] == 1.0
    assert pairwise["mean_target_coverage_delta"] == 1.0


def qdrant_reranker_ablation_report_for_gate():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual evidence",
        asset_ids=["asset-1"],
        metadata={
            "chunking_strategy": "visual_asset_text",
            "retrieval_role": "child",
        },
    )
    cases = [
        RetrievalCase(
            query="visual evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe"},
        )
    ]
    lexical_eval = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=chunk,
                score=0.9,
                sources=["bm25", "rerank:lexical"],
            )
        ],
        top_k=5,
    )
    none_eval = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [],
        top_k=5,
    )
    return build_qdrant_reranker_ablation_report(
        [
            QdrantRerankerAblationRow(
                mode=QdrantRerankerAblationMode(
                    name="lexical",
                    reranker="lexical",
                    rerank_top_k=20,
                ),
                evaluation=lexical_eval,
            ),
            QdrantRerankerAblationRow(
                mode=QdrantRerankerAblationMode(name="none"),
                evaluation=none_eval,
            ),
        ]
    )


def test_gate_qdrant_reranker_ablation_passes_required_mode():
    report = qdrant_reranker_ablation_report_for_gate()

    gate = gate_qdrant_reranker_ablation(
        report,
        mode="lexical",
        baseline_mode="none",
        min_recall_at_k=1.0,
        min_target_coverage_at_k=1.0,
        min_target_ndcg_at_k=1.0,
        min_mrr=1.0,
        min_precision_at_k=0.2,
        max_failed_queries=0,
        max_mean_target_rank=1.0,
        min_target_type_coverage={"asset": 1.0},
        min_source_target_coverage={"rerank:lexical": 1.0},
        min_source_family_target_coverage={"reranker": 1.0},
        min_source_precision_at_hits={"rerank:lexical": 1.0},
        min_source_family_precision_at_hits={"reranker": 1.0},
        min_case_group_target_coverage={"case_source:visual_object_probe": 1.0},
        min_case_group_source_target_coverage={
            "case_source:visual_object_probe:rerank:lexical": 1.0
        },
        min_case_group_source_family_target_coverage={
            "case_source:visual_object_probe:reranker": 1.0
        },
        min_case_group_source_precision_at_hits={
            "case_source:visual_object_probe:rerank:lexical": 1.0
        },
        min_case_group_source_family_precision_at_hits={
            "case_source:visual_object_probe:reranker": 1.0
        },
        min_pairwise_shared_queries=1,
        min_pairwise_win_rate=1.0,
        min_pairwise_target_coverage_lift=1.0,
        min_pairwise_target_ndcg_lift=1.0,
        min_pairwise_mrr_lift=1.0,
        max_pairwise_mean_target_rank_delta=0.0,
        require_best_by_recall=True,
        require_best_by_target_coverage=True,
        require_best_by_target_ndcg=True,
    )

    assert gate.passed is True
    assert gate.mode == "lexical"
    assert gate.baseline_mode == "none"
    assert gate.reranker == "lexical"
    assert gate.rerank_top_k == 20
    assert gate.metrics["failed_query_count"] == 0.0
    assert gate.metrics["target_coverage_at_k"] == 1.0
    assert gate.pairwise_metrics["pairwise_candidate_win_rate"] == 1.0
    assert gate.pairwise_metrics["pairwise_mean_target_rank_delta"] < 0
    assert gate.target_metrics["asset"]["coverage_at_k"] == 1.0
    assert gate.source_metrics["rerank:lexical"]["target_coverage_at_k"] == 1.0
    assert gate.source_metrics["rerank:lexical"]["precision_at_hits"] == 1.0
    assert gate.source_family_metrics["reranker"]["target_coverage_at_k"] == 1.0
    assert gate.source_family_metrics["reranker"]["precision_at_hits"] == 1.0
    assert (
        gate.case_group_metrics["case_source"]["visual_object_probe"][
            "target_coverage_at_k"
        ]
        == 1.0
    )
    assert gate.case_group_source_metrics["case_source"]["visual_object_probe"][
        "rerank:lexical"
    ]["target_coverage_at_k"] == 1.0
    assert gate.case_group_source_metrics["case_source"]["visual_object_probe"][
        "rerank:lexical"
    ]["precision_at_hits"] == 1.0
    assert gate.case_group_source_family_metrics["case_source"]["visual_object_probe"][
        "reranker"
    ]["target_coverage_at_k"] == 1.0
    assert gate.case_group_source_family_metrics["case_source"]["visual_object_probe"][
        "reranker"
    ]["precision_at_hits"] == 1.0


def test_gate_qdrant_reranker_ablation_cli_writes_report(tmp_path):
    report_path = tmp_path / "qdrant_reranker_ablation.json"
    output = tmp_path / "qdrant_reranker_ablation_gate.json"
    report_path.write_text(
        qdrant_reranker_ablation_report_for_gate().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-qdrant-reranker-ablation",
            str(report_path),
            "--mode",
            "lexical",
            "--baseline-mode",
            "none",
            "--min-target-coverage-at-k",
            "1.0",
            "--min-pairwise-win-rate",
            "1.0",
            "--max-pairwise-mean-target-rank-delta",
            "0.0",
            "--min-case-group-source-target-coverage",
            "case_source:visual_object_probe:rerank:lexical=1.0",
            "--min-case-group-source-family-target-coverage",
            "case_source:visual_object_probe:reranker=1.0",
            "--min-source-precision-at-hits",
            "rerank:lexical=1.0",
            "--min-source-family-precision-at-hits",
            "reranker=1.0",
            "--min-case-group-source-precision-at-hits",
            "case_source:visual_object_probe:rerank:lexical=1.0",
            "--min-case-group-source-family-precision-at-hits",
            "case_source:visual_object_probe:reranker=1.0",
            "--require-best-by-target-ndcg",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["mode"] == "lexical"
    assert payload["baseline_mode"] == "none"
    assert payload["reranker"] == "lexical"
    assert payload["rerank_top_k"] == 20
    assert payload["pairwise_metrics"]["pairwise_candidate_win_rate"] == 1.0
    assert payload["case_group_source_metrics"]["case_source"]["visual_object_probe"][
        "rerank:lexical"
    ]["target_coverage_at_k"] == 1.0
    assert payload["source_metrics"]["rerank:lexical"]["precision_at_hits"] == 1.0
    assert payload["source_family_metrics"]["reranker"]["precision_at_hits"] == 1.0
    assert payload["case_group_source_metrics"]["case_source"]["visual_object_probe"][
        "rerank:lexical"
    ]["precision_at_hits"] == 1.0
    assert payload["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["reranker"]["target_coverage_at_k"] == 1.0
    assert payload["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["reranker"]["precision_at_hits"] == 1.0


def qdrant_vector_ablation_report_for_gate():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual evidence",
        asset_ids=["asset-1"],
        metadata={
            "chunking_strategy": "visual_asset_text",
            "retrieval_role": "child",
        },
    )
    cases = [
        RetrievalCase(
            query="visual evidence",
            expected_asset_ids=["asset-1"],
            metadata={"case_source": "visual_object_probe"},
        )
    ]
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
    return build_qdrant_vector_ablation_report(
        [
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
        ]
    )


def qdrant_vector_ablation_report_with_excluded_hits():
    expected_chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="visual evidence",
        asset_ids=["asset-1"],
        metadata={"chunking_strategy": "visual_asset_text", "retrieval_role": "child"},
    )
    excluded_chunk = DocumentChunk(
        chunk_id="chunk-2",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="visual evidence",
        asset_ids=["asset-2"],
        metadata={"chunking_strategy": "visual_asset_text", "retrieval_role": "child"},
    )
    cases = [
        RetrievalCase(
            query="visual evidence",
            expected_asset_ids=["asset-1"],
            excluded_asset_ids=["asset-2"],
        )
    ]
    clean_eval = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=expected_chunk,
                score=0.9,
                sources=["qdrant:caption_dense"],
            )
        ],
        top_k=5,
    )
    leaky_eval = evaluate_search_results(
        cases=cases,
        search_fn=lambda case, graph_expand: [
            HybridSearchHit(
                chunk=expected_chunk,
                score=0.9,
                sources=["qdrant:caption_dense"],
            ),
            HybridSearchHit(
                chunk=excluded_chunk,
                score=0.8,
                sources=["qdrant:image_dense"],
            ),
        ],
        top_k=5,
    )
    return build_qdrant_vector_ablation_report(
        [
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(
                    name="clean",
                    vector_names=["caption_dense"],
                ),
                evaluation=clean_eval,
            ),
            QdrantVectorAblationRow(
                mode=QdrantVectorAblationMode(
                    name="leaky",
                    vector_names=["caption_dense", "image_dense"],
                ),
                evaluation=leaky_eval,
            ),
        ]
    )


def test_gate_qdrant_vector_ablation_passes_required_mode():
    report = qdrant_vector_ablation_report_for_gate()

    gate = gate_qdrant_vector_ablation(
        report,
        mode="caption_image",
        baseline_mode="image",
        min_recall_at_k=1.0,
        min_target_coverage_at_k=1.0,
        min_target_ndcg_at_k=1.0,
        max_failed_queries=0,
        min_target_type_coverage={"asset": 1.0},
        min_source_target_coverage={
            "qdrant:caption_dense": 1.0,
            "qdrant:image_dense": 1.0,
        },
        min_source_family_target_coverage={"visual": 1.0},
        min_source_precision_at_hits={
            "qdrant:caption_dense": 1.0,
            "qdrant:image_dense": 1.0,
        },
        min_source_family_precision_at_hits={"visual": 1.0},
        min_case_group_target_coverage={"case_source:visual_object_probe": 1.0},
        min_case_group_source_target_coverage={
            "case_source:visual_object_probe:qdrant:image_dense": 1.0
        },
        min_case_group_source_family_target_coverage={
            "case_source:visual_object_probe:visual": 1.0
        },
        min_case_group_source_precision_at_hits={
            "case_source:visual_object_probe:qdrant:image_dense": 1.0
        },
        min_case_group_source_family_precision_at_hits={
            "case_source:visual_object_probe:visual": 1.0
        },
        max_mean_target_rank=1.0,
        min_pairwise_shared_queries=1,
        min_pairwise_win_rate=1.0,
        min_pairwise_target_coverage_lift=1.0,
        min_pairwise_target_coverage_ci_low=1.0,
        max_pairwise_mean_target_rank_delta=0.0,
        max_pairwise_target_rank_delta_ci_high=0.0,
        require_best_by_recall=True,
        require_best_by_target_coverage=True,
    )

    assert gate.passed
    assert gate.mode == "caption_image"
    assert gate.baseline_mode == "image"
    assert gate.vector_names == ["caption_dense", "image_dense"]
    assert gate.metrics["failed_query_count"] == 0.0
    assert gate.metrics["mean_target_rank"] == 1.0
    assert gate.baseline_metrics["failed_query_count"] == 1.0
    assert gate.baseline_metrics["mean_target_rank"] == 6.0
    assert gate.pairwise_metrics["pairwise_candidate_win_rate"] == 1.0
    assert gate.pairwise_metrics["pairwise_mean_target_coverage_delta"] == 1.0
    assert gate.pairwise_metrics["pairwise_mean_target_rank_delta"] == -5.0
    assert gate.pairwise_metrics["pairwise_target_rank_delta_ci_high"] == -5.0
    assert {check.name for check in gate.checks if check.name.startswith("min_pairwise")} == {
        "min_pairwise_shared_queries",
        "min_pairwise_win_rate",
        "min_pairwise_target_coverage_lift",
        "min_pairwise_target_coverage_ci_low",
    }
    assert {check.name for check in gate.checks if check.name.startswith("max_pairwise")} == {
        "max_pairwise_mean_target_rank_delta",
        "max_pairwise_target_rank_delta_ci_high",
    }
    assert gate.metrics["target_type.asset.coverage_at_k"] == 1.0
    assert gate.target_metrics["asset"]["coverage_at_k"] == 1.0
    assert gate.metrics["source.qdrant:caption_dense.target_coverage_at_k"] == 1.0
    assert gate.metrics["source.qdrant:image_dense.target_coverage_at_k"] == 1.0
    assert gate.metrics["source.qdrant:image_dense.precision_at_hits"] == 1.0
    assert gate.source_metrics["qdrant:image_dense"]["target_coverage_at_k"] == 1.0
    assert gate.source_metrics["qdrant:image_dense"]["precision_at_hits"] == 1.0
    assert gate.metrics["source_family.visual.target_coverage_at_k"] == 1.0
    assert gate.metrics["source_family.visual.precision_at_hits"] == 1.0
    assert gate.source_family_metrics["visual"]["target_coverage_at_k"] == 1.0
    assert gate.source_family_metrics["visual"]["precision_at_hits"] == 1.0
    assert gate.metrics[
        "case_group.case_source.visual_object_probe.target_coverage_at_k"
    ] == 1.0
    assert gate.case_group_metrics["case_source"]["visual_object_probe"][
        "target_coverage_at_k"
    ] == 1.0
    assert gate.case_group_source_metrics["case_source"]["visual_object_probe"][
        "qdrant:image_dense"
    ]["target_coverage_at_k"] == 1.0
    assert gate.case_group_source_metrics["case_source"]["visual_object_probe"][
        "qdrant:image_dense"
    ]["precision_at_hits"] == 1.0
    assert gate.case_group_source_family_metrics["case_source"]["visual_object_probe"][
        "visual"
    ]["target_coverage_at_k"] == 1.0
    assert gate.case_group_source_family_metrics["case_source"]["visual_object_probe"][
        "visual"
    ]["precision_at_hits"] == 1.0
    assert gate.metrics["chunk_strategy.visual_asset_text.target_coverage_at_k"] == 1.0
    assert gate.metrics["retrieval_role.child.target_coverage_at_k"] == 1.0
    assert gate.chunk_strategy_metrics["visual_asset_text"]["target_coverage_at_k"] == 1.0
    assert gate.retrieval_role_metrics["child"]["target_coverage_at_k"] == 1.0
    assert gate.failed_checks == []


def test_gate_qdrant_vector_ablation_reports_failed_checks():
    report = qdrant_vector_ablation_report_for_gate()

    gate = gate_qdrant_vector_ablation(
        report,
        mode="image",
        min_recall_at_k=1.0,
        max_failed_queries=0,
        min_target_type_coverage={"asset": 1.0},
        min_source_target_coverage={"qdrant:image_dense": 1.0},
        min_source_family_target_coverage={"visual": 1.0},
        min_source_precision_at_hits={"qdrant:image_dense": 1.0},
        min_source_family_precision_at_hits={"visual": 1.0},
        min_case_group_target_coverage={"case_source:visual_object_probe": 1.0},
        min_case_group_source_target_coverage={
            "case_source:visual_object_probe:qdrant:image_dense": 1.0
        },
        min_case_group_source_family_target_coverage={
            "case_source:visual_object_probe:visual": 1.0
        },
        min_case_group_source_precision_at_hits={
            "case_source:visual_object_probe:qdrant:image_dense": 1.0
        },
        min_case_group_source_family_precision_at_hits={
            "case_source:visual_object_probe:visual": 1.0
        },
        max_mean_target_rank=5.0,
        require_best_by_recall=True,
    )

    assert not gate.passed
    assert gate.metrics["failed_query_count"] == 1.0
    assert gate.metrics["mean_target_rank"] == 6.0
    assert gate.metrics["target_type.asset.coverage_at_k"] == 0.0
    assert gate.metrics["source_family.visual.target_coverage_at_k"] == 0.0
    assert gate.metrics["source.qdrant:image_dense.precision_at_hits"] == 0.0
    assert gate.metrics["source_family.visual.precision_at_hits"] == 0.0
    assert set(gate.failed_checks) == {
        "min_recall_at_k",
        "max_failed_queries",
        "min_target_type_coverage:asset",
        "min_source_target_coverage:qdrant:image_dense",
        "min_source_family_target_coverage:visual",
        "min_source_precision_at_hits:qdrant:image_dense",
        "min_source_family_precision_at_hits:visual",
        "min_case_group_target_coverage:case_source:visual_object_probe",
        "min_case_group_source_target_coverage:"
        "case_source:visual_object_probe:qdrant:image_dense",
        "min_case_group_source_family_target_coverage:"
        "case_source:visual_object_probe:visual",
        "min_case_group_source_precision_at_hits:"
        "case_source:visual_object_probe:qdrant:image_dense",
        "min_case_group_source_family_precision_at_hits:"
        "case_source:visual_object_probe:visual",
        "max_mean_target_rank",
        "require_best_by_recall",
    }


def test_gate_qdrant_vector_ablation_checks_excluded_target_hits():
    report = qdrant_vector_ablation_report_with_excluded_hits()

    gate = gate_qdrant_vector_ablation(
        report,
        mode="leaky",
        max_excluded_target_hit_rate=0.0,
        max_excluded_query_hit_rate=0.0,
        max_excluded_hit_query_count=0,
    )

    assert gate.passed is False
    assert gate.metrics["excluded_target_hit_rate"] == 1.0
    assert gate.metrics["excluded_query_hit_rate"] == 1.0
    assert gate.metrics["excluded_hit_query_count"] == 1.0
    assert set(gate.failed_checks) == {
        "max_excluded_target_hit_rate",
        "max_excluded_query_hit_rate",
        "max_excluded_hit_query_count",
    }


def test_gate_qdrant_vector_ablation_checks_source_excluded_target_hits():
    report = qdrant_vector_ablation_report_with_excluded_hits()

    gate = gate_qdrant_vector_ablation(
        report,
        mode="leaky",
        max_source_excluded_target_hit_rate={"qdrant:image_dense": 0.0},
        max_source_family_excluded_target_hit_rate={"visual": 0.0},
        max_chunk_strategy_excluded_target_hit_rate={"visual_asset_text": 0.0},
        max_retrieval_role_excluded_target_hit_rate={"child": 0.0},
    )

    assert gate.passed is False
    assert gate.metrics["source.qdrant:image_dense.excluded_target_hit_rate"] == 1.0
    assert gate.metrics["source_family.visual.excluded_target_hit_rate"] == 1.0
    assert gate.metrics["chunk_strategy.visual_asset_text.excluded_target_hit_rate"] == 1.0
    assert gate.metrics["retrieval_role.child.excluded_target_hit_rate"] == 1.0
    assert set(gate.failed_checks) == {
        "max_source_excluded_target_hit_rate:qdrant:image_dense",
        "max_source_family_excluded_target_hit_rate:visual",
        "max_chunk_strategy_excluded_target_hit_rate:visual_asset_text",
        "max_retrieval_role_excluded_target_hit_rate:child",
    }


def test_gate_qdrant_vector_ablation_requires_baseline_for_pairwise_checks():
    report = qdrant_vector_ablation_report_for_gate()

    with pytest.raises(ValueError, match="baseline mode"):
        gate_qdrant_vector_ablation(
            report,
            mode="caption_image",
            min_pairwise_win_rate=0.5,
        )


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
            "--baseline-mode",
            "image",
            "--min-recall-at-k",
            "1.0",
            "--min-target-coverage-at-k",
            "1.0",
            "--max-failed-queries",
            "0",
            "--min-target-type-coverage",
            "asset=1.0",
            "--min-source-target-coverage",
            "qdrant:image_dense=1.0",
            "--min-source-family-target-coverage",
            "visual=1.0",
            "--min-source-precision-at-hits",
            "qdrant:image_dense=1.0",
            "--min-source-family-precision-at-hits",
            "visual=1.0",
            "--min-case-group-target-coverage",
            "case_source:visual_object_probe=1.0",
            "--min-case-group-source-target-coverage",
            "case_source:visual_object_probe:qdrant:image_dense=1.0",
            "--min-case-group-source-family-target-coverage",
            "case_source:visual_object_probe:visual=1.0",
            "--min-case-group-source-precision-at-hits",
            "case_source:visual_object_probe:qdrant:image_dense=1.0",
            "--min-case-group-source-family-precision-at-hits",
            "case_source:visual_object_probe:visual=1.0",
            "--max-mean-target-rank",
            "1.0",
            "--min-pairwise-shared-queries",
            "1",
            "--min-pairwise-win-rate",
            "1.0",
            "--min-pairwise-target-coverage-lift",
            "1.0",
            "--max-pairwise-mean-target-rank-delta",
            "0.0",
            "--require-best-by-recall",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["mode"] == "caption_image"
    assert payload["baseline_mode"] == "image"
    assert payload["vector_names"] == ["caption_dense", "image_dense"]
    assert payload["metrics"]["mean_target_rank"] == 1.0
    assert payload["baseline_metrics"]["recall_at_k"] == 0.0
    assert payload["baseline_metrics"]["mean_target_rank"] == 6.0
    assert payload["pairwise_metrics"]["pairwise_candidate_win_rate"] == 1.0
    assert payload["pairwise_metrics"]["pairwise_mean_target_rank_delta"] == -5.0
    assert payload["target_metrics"]["asset"]["coverage_at_k"] == 1.0
    assert payload["source_metrics"]["qdrant:image_dense"]["target_coverage_at_k"] == 1.0
    assert payload["source_metrics"]["qdrant:image_dense"]["precision_at_hits"] == 1.0
    assert payload["metrics"]["source.qdrant:image_dense.target_coverage_at_k"] == 1.0
    assert payload["metrics"]["source.qdrant:image_dense.precision_at_hits"] == 1.0
    assert payload["source_family_metrics"]["visual"]["target_coverage_at_k"] == 1.0
    assert payload["source_family_metrics"]["visual"]["precision_at_hits"] == 1.0
    assert payload["case_group_metrics"]["case_source"]["visual_object_probe"][
        "target_coverage_at_k"
    ] == 1.0
    assert payload["case_group_source_metrics"]["case_source"]["visual_object_probe"][
        "qdrant:image_dense"
    ]["target_coverage_at_k"] == 1.0
    assert payload["case_group_source_metrics"]["case_source"]["visual_object_probe"][
        "qdrant:image_dense"
    ]["precision_at_hits"] == 1.0
    assert payload["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["visual"]["target_coverage_at_k"] == 1.0
    assert payload["case_group_source_family_metrics"]["case_source"][
        "visual_object_probe"
    ]["visual"]["precision_at_hits"] == 1.0
    assert payload["case_group_best_modes"]["case_source"]["visual_object_probe"][
        "target_coverage_at_k"
    ]["mode"] == "caption_image"
    assert payload["chunk_strategy_metrics"]["visual_asset_text"]["target_coverage_at_k"] == 1.0
    assert payload["retrieval_role_metrics"]["child"]["target_coverage_at_k"] == 1.0


def test_gate_qdrant_vector_ablation_cli_checks_excluded_target_hits(tmp_path):
    report_path = tmp_path / "qdrant_vector_ablation.json"
    output = tmp_path / "qdrant_vector_ablation_gate.json"
    report_path.write_text(
        qdrant_vector_ablation_report_with_excluded_hits().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-qdrant-vector-ablation",
            str(report_path),
            "--mode",
            "leaky",
            "--max-excluded-target-hit-rate",
            "0",
            "--max-excluded-query-hit-rate",
            "0",
            "--max-excluded-hit-query-count",
            "0",
            "--max-source-excluded-target-hit-rate",
            "qdrant:image_dense=0",
            "--max-source-family-excluded-target-hit-rate",
            "visual=0",
            "--max-chunk-strategy-excluded-target-hit-rate",
            "visual_asset_text=0",
            "--max-retrieval-role-excluded-target-hit-rate",
            "child=0",
            "--no-fail",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["metrics"]["excluded_target_hit_rate"] == 1.0
    assert payload["metrics"]["source.qdrant:image_dense.excluded_target_hit_rate"] == 1.0
    assert payload["metrics"]["source_family.visual.excluded_target_hit_rate"] == 1.0
    assert payload["metrics"]["chunk_strategy.visual_asset_text.excluded_target_hit_rate"] == 1.0
    assert payload["metrics"]["retrieval_role.child.excluded_target_hit_rate"] == 1.0
    assert set(payload["failed_checks"]) == {
        "max_excluded_target_hit_rate",
        "max_excluded_query_hit_rate",
        "max_excluded_hit_query_count",
        "max_source_excluded_target_hit_rate:qdrant:image_dense",
        "max_source_family_excluded_target_hit_rate:visual",
        "max_chunk_strategy_excluded_target_hit_rate:visual_asset_text",
        "max_retrieval_role_excluded_target_hit_rate:child",
    }


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
            "--min-case-group-target-coverage",
            "case_source:visual_object_probe=1.0",
            "--min-case-group-source-target-coverage",
            "case_source:visual_object_probe:qdrant:image_dense=1.0",
            "--min-case-group-source-family-target-coverage",
            "case_source:visual_object_probe:visual=1.0",
            "--max-mean-target-rank",
            "5.0",
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
        "min_case_group_target_coverage:case_source:visual_object_probe",
        "min_case_group_source_target_coverage:"
        "case_source:visual_object_probe:qdrant:image_dense",
        "min_case_group_source_family_target_coverage:"
        "case_source:visual_object_probe:visual",
        "max_mean_target_rank",
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
    pairwise = next(
        comparison
        for comparison in payload["pairwise"]
        if comparison["candidate"] == "bm25_visual"
        and comparison["baseline"] == "bm25_text"
    )
    assert pairwise["candidate_win_rate"] == 1.0
    assert pairwise["mean_target_coverage_delta"] == 1.0


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
            "--min-source-target-coverage",
            "bm25=1.0",
            "--min-source-family-target-coverage",
            "lexical=1.0",
            "--min-source-precision-at-hits",
            "bm25=1.0",
            "--min-source-family-precision-at-hits",
            "lexical=1.0",
            "--min-case-group-target-coverage",
            "case_source:visual_lexical_probe=1.0",
            "--min-case-group-source-target-coverage",
            "case_source:visual_lexical_probe:bm25=1.0",
            "--min-case-group-source-family-target-coverage",
            "case_source:visual_lexical_probe:lexical=1.0",
            "--min-case-group-source-precision-at-hits",
            "case_source:visual_lexical_probe:bm25=1.0",
            "--min-case-group-source-family-precision-at-hits",
            "case_source:visual_lexical_probe:lexical=1.0",
            "--max-mean-target-rank",
            "1.0",
            "--min-pairwise-shared-queries",
            "1",
            "--min-pairwise-win-rate",
            "1.0",
            "--min-pairwise-target-coverage-lift",
            "1.0",
            "--max-pairwise-mean-target-rank-delta",
            "0.0",
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
    assert payload["metrics"]["mean_target_rank"] == 1.0
    assert payload["baseline_metrics"]["recall_at_k"] == 0.0
    assert payload["baseline_metrics"]["mean_target_rank"] == 6.0
    assert payload["pairwise_metrics"]["pairwise_candidate_win_rate"] == 1.0
    assert payload["pairwise_metrics"]["pairwise_mean_target_coverage_delta"] == 1.0
    assert payload["pairwise_metrics"]["pairwise_mean_target_rank_delta"] == -5.0
    assert payload["source_metrics"]["bm25"]["target_coverage_at_k"] == 1.0
    assert payload["source_metrics"]["bm25"]["precision_at_hits"] == 1.0
    assert payload["metrics"]["source.bm25.target_coverage_at_k"] == 1.0
    assert payload["metrics"]["source.bm25.precision_at_hits"] == 1.0
    assert payload["case_group_metrics"]["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ] == 1.0
    assert payload["case_group_source_metrics"]["case_source"]["visual_lexical_probe"][
        "bm25"
    ]["target_coverage_at_k"] == 1.0
    assert payload["case_group_source_metrics"]["case_source"]["visual_lexical_probe"][
        "bm25"
    ]["precision_at_hits"] == 1.0
    assert payload["case_group_source_family_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["target_coverage_at_k"] == 1.0
    assert payload["case_group_source_family_metrics"]["case_source"][
        "visual_lexical_probe"
    ]["lexical"]["precision_at_hits"] == 1.0
    assert payload["case_group_best_modes"]["case_source"]["visual_lexical_probe"][
        "target_coverage_at_k"
    ]["mode"] == "bm25_visual"


def test_gate_retrieval_ablation_cli_checks_source_excluded_target_hits(tmp_path):
    report_path = tmp_path / "ablation.json"
    output = tmp_path / "ablation_gate.json"
    report_path.write_text(
        retrieval_ablation_report_with_excluded_hits().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "gate-retrieval-ablation",
            str(report_path),
            "--mode",
            "bm25",
            "--max-source-excluded-target-hit-rate",
            "bm25=0",
            "--max-source-family-excluded-target-hit-rate",
            "lexical=0",
            "--max-chunk-strategy-excluded-target-hit-rate",
            "visual_asset_text=0",
            "--max-retrieval-role-excluded-target-hit-rate",
            "child=0",
            "--output",
            str(output),
            "--no-fail",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["metrics"]["source.bm25.excluded_target_hit_rate"] == 1.0
    assert payload["metrics"]["source_family.lexical.excluded_target_hit_rate"] == 1.0
    assert payload["metrics"]["chunk_strategy.visual_asset_text.excluded_target_hit_rate"] == 1.0
    assert payload["metrics"]["retrieval_role.child.excluded_target_hit_rate"] == 1.0
    assert set(payload["failed_checks"]) == {
        "max_source_excluded_target_hit_rate:bm25",
        "max_source_family_excluded_target_hit_rate:lexical",
        "max_chunk_strategy_excluded_target_hit_rate:visual_asset_text",
        "max_retrieval_role_excluded_target_hit_rate:child",
    }

import json
from pathlib import Path

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.evaluation.case_audit import audit_retrieval_cases
from chunking_docs.evaluation.casegen import generate_retrieval_case_skeleton
from chunking_docs.io import write_jsonl
from chunking_docs.models import (
    AssetKind,
    ChunkKind,
    DocumentChunk,
    GraphTriple,
    PageProfile,
    ProcessingManifest,
    SourceDocument,
    TextQuality,
    VisualAsset,
)


def test_generate_retrieval_case_skeleton_targets_pages_assets_and_triples():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map",
        vlm_summary="Shows corridor links.",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="corridor",
        predicate="connects",
        object="station",
    )

    cases = generate_retrieval_case_skeleton([chunk], [asset], [triple])

    assert len(cases) == 3
    assert cases[0].expected_pages == [1]
    assert cases[0].expected_chunk_ids == ["chunk-1"]
    assert cases[1].expected_asset_ids == ["asset-1"]
    assert cases[2].expected_triple_ids == ["triple-1"]
    assert cases[2].graph_expand is True


def test_generate_retrieval_case_skeleton_merges_duplicate_triple_queries():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="first page",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="second page",
        ),
    ]
    triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="shared section",
            predicate="includes_issue",
            object="common topic",
        ),
        GraphTriple(
            triple_id="triple-2",
            doc_id="doc",
            chunk_id="chunk-2",
            subject="shared section",
            predicate="includes_issue",
            object="common topic",
        ),
    ]

    cases = generate_retrieval_case_skeleton(
        chunks,
        [],
        triples,
        page_limit=0,
        triple_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].expected_chunk_ids == ["chunk-1", "chunk-2"]
    assert cases[0].expected_triple_ids == ["triple-1", "triple-2"]
    assert cases[0].metadata["merged_case_count"] == 2


def test_generate_retrieval_case_skeleton_targets_visual_asset_from_triple_provenance():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="station access map",
        predicate="shows",
        object="corridor link",
        qualifiers={"source": "visual_annotation", "asset_id": "asset-1"},
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [triple],
        include_pages=False,
        include_assets=False,
    )

    assert len(cases) == 1
    assert cases[0].expected_chunk_ids == ["chunk-1"]
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].expected_triple_ids == ["triple-1"]
    assert cases[0].graph_expand is True


def test_generate_retrieval_case_skeleton_resolves_visual_triple_chunk_from_asset():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        source_refs=["asset:asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="vlm-annotation",
        subject="station access map",
        predicate="shows",
        object="corridor link",
        qualifiers={"source": "visual_annotation", "asset_id": "asset-1"},
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [triple],
        include_pages=False,
        include_assets=False,
    )

    assert len(cases) == 1
    assert cases[0].expected_chunk_ids == ["chunk-1"]
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].expected_triple_ids == ["triple-1"]


def test_generate_retrieval_case_skeleton_can_emit_todo_cases():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.PAGE_SUMMARY,
        text="[empty text layer] OCR/VLM processing required for page 2.",
    )

    cases = generate_retrieval_case_skeleton([chunk], [], [], include_todo=True)

    assert cases[0].query == "TODO: write query for page 2"
    assert cases[0].expected_pages == [2]


def test_generate_retrieval_case_skeleton_can_use_salient_terms():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="Common overview alpha corridor terminal transfer evidence.",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="Common overview beta retention wetland basin evidence.",
        ),
    ]

    cases = generate_retrieval_case_skeleton(
        chunks,
        [],
        [],
        page_limit=2,
        query_mode="salient_terms",
        min_query_terms=3,
    )

    assert len(cases) == 2
    assert "overview" not in cases[0].query.lower()
    assert {"alpha", "corridor", "terminal"}.issubset(set(cases[0].query.lower().split()))
    assert cases[0].metadata["query_mode"] == "salient_terms"
    assert cases[0].metadata["case_source"] == "page"
    assert cases[0].metadata["selection_score"] > 0


def test_generate_retrieval_case_skeleton_can_use_question_queries():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [],
        [],
        page_limit=1,
        query_mode="question",
        min_query_terms=3,
        max_query_terms=3,
    )
    report = audit_retrieval_cases(
        cases,
        profiles=[],
        chunks=[chunk],
        assets=[],
        triples=[],
        max_target_query_overlap_ratio=0.75,
    )

    assert cases[0].query == "Where is Transit corridor evidence discussed?"
    assert cases[0].metadata["query_mode"] == "question"
    assert cases[0].metadata["query_terms"] == ["Transit", "corridor", "evidence"]
    assert report.passed is True
    assert report.max_target_query_overlap_ratio <= 0.75


def test_generate_retrieval_case_skeleton_can_filter_target_overlap_terms():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [],
        [],
        page_limit=1,
        query_mode="question",
        min_query_terms=3,
        max_query_terms=3,
        max_target_query_overlap_terms=1,
        min_terms_for_target_overlap=2,
    )

    assert cases == []


def test_generate_retrieval_case_skeleton_can_filter_target_concentration():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map legend signal",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        page_limit=1,
        asset_limit=1,
        include_triples=False,
        max_page_cases_per_target=1,
    )

    assert len(cases) == 1
    assert cases[0].metadata["case_source"] == "page"
    assert cases[0].expected_pages == [1]


def test_generate_retrieval_case_skeleton_attaches_hard_negative_targets():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="Transit corridor station access evidence.",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="Transit corridor freight depot evidence.",
        ),
        DocumentChunk(
            chunk_id="chunk-3",
            doc_id="doc",
            page_start=3,
            page_end=3,
            kind=ChunkKind.TEXT,
            text="Wetland basin retention monitoring.",
        ),
    ]

    cases = generate_retrieval_case_skeleton(
        chunks,
        [],
        [],
        page_limit=1,
        include_assets=False,
        include_triples=False,
        hard_negative_limit=1,
        hard_negative_min_overlap_terms=2,
    )

    assert len(cases) == 1
    assert cases[0].expected_pages == [1]
    assert cases[0].excluded_pages == [2]
    assert cases[0].excluded_chunk_ids == ["chunk-2"]
    assert cases[0].metadata["hard_negative_target_counts"] == {
        "page": 1,
        "chunk": 1,
        "asset": 0,
        "triple": 0,
    }
    assert cases[0].metadata["hard_negative_target_types"] == ["page", "chunk"]


def test_generate_retrieval_case_skeleton_attaches_asset_hard_negative():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map corridor legend",
    )
    similar_asset = VisualAsset(
        asset_id="asset-2",
        doc_id="doc",
        page_no=2,
        kind=AssetKind.MAP,
        caption="Station access map bicycle legend",
    )
    unrelated_asset = VisualAsset(
        asset_id="asset-3",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.CHART,
        caption="Housing supply forecast chart",
    )

    cases = generate_retrieval_case_skeleton(
        [],
        [asset, similar_asset, unrelated_asset],
        [],
        include_pages=False,
        asset_limit=1,
        include_triples=False,
        hard_negative_limit=1,
        hard_negative_min_overlap_terms=2,
    )

    assert len(cases) == 1
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].excluded_asset_ids == ["asset-2"]
    assert cases[0].excluded_pages == [2]
    assert cases[0].metadata["hard_negative_target_types"] == ["page", "asset"]


def test_generate_retrieval_case_skeleton_merges_excluded_targets():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="Shared retrieval phrase.",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="Shared retrieval phrase.",
        ),
        DocumentChunk(
            chunk_id="chunk-3",
            doc_id="doc",
            page_start=3,
            page_end=3,
            kind=ChunkKind.TEXT,
            text="Shared retrieval alternate phrase.",
        ),
    ]

    cases = generate_retrieval_case_skeleton(
        chunks,
        [],
        [],
        page_limit=2,
        include_assets=False,
        include_triples=False,
        hard_negative_limit=1,
        hard_negative_min_overlap_terms=1,
    )

    assert len(cases) == 1
    assert cases[0].expected_pages == [1, 2]
    assert cases[0].expected_chunk_ids == ["chunk-1", "chunk-2"]
    assert cases[0].excluded_pages == [3]
    assert cases[0].excluded_chunk_ids == ["chunk-3"]


def test_generate_retrieval_case_skeleton_can_rank_by_salience():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="Overview summary contents report reference.",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="Biofilter retention basin infiltration monitoring.",
        ),
    ]

    cases = generate_retrieval_case_skeleton(
        chunks,
        [],
        [],
        page_limit=1,
        query_mode="salient_terms",
        selection_strategy="salience",
        min_query_terms=3,
    )

    assert cases[0].expected_pages == [2]
    assert "biofilter" in cases[0].query.lower()


def test_generate_retrieval_case_skeleton_ignores_identifier_like_triple_terms():
    chunk = DocumentChunk(
        chunk_id="abc123def456",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Placeholder text.",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id=chunk.chunk_id,
        subject=chunk.chunk_id,
        predicate="belongs_to_section",
        object="Watershed retention strategy",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [],
        [triple],
        include_pages=False,
        query_mode="salient_terms",
        min_query_terms=2,
    )

    assert cases[0].query == "Watershed retention strategy"
    assert chunk.chunk_id not in cases[0].query
    assert "belongs" not in cases[0].query


def test_generate_retrieval_case_skeleton_dedupes_queries_by_default():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="Shared retrieval phrase.",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="Shared retrieval phrase.",
        ),
    ]

    cases = generate_retrieval_case_skeleton(chunks, [], [], page_limit=2)

    assert len(cases) == 1
    assert cases[0].expected_chunk_ids == ["chunk-1", "chunk-2"]


def test_generate_retrieval_case_skeleton_merges_case_group_metadata():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Shared retrieval phrase.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Shared retrieval phrase.",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        page_limit=1,
        asset_limit=1,
    )
    report = audit_retrieval_cases(
        cases,
        profiles=[],
        chunks=[chunk],
        assets=[asset],
        triples=[],
    )

    assert len(cases) == 1
    assert cases[0].expected_chunk_ids == ["chunk-1"]
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].metadata["case_source"] == ["page", "asset"]
    assert cases[0].metadata["merged_case_count"] == 2
    assert report.case_group_counts["case_source"]["page"] == 1
    assert report.case_group_counts["case_source"]["asset"] == 1
    assert report.case_group_distinct_target_counts["case_source"]["asset"]["asset"] == 1


def test_generate_retrieval_case_skeleton_can_create_visual_lexical_probes():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map legend signal",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        visual_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "map legend signal"
    assert cases[0].expected_pages == [1]
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].metadata["case_source"] == "visual_lexical_probe"
    assert cases[0].metadata["case_family"] == "visual"
    assert cases[0].metadata["evidence_family"] == "visual_text"
    assert cases[0].metadata["modality"] == "visual_text"
    assert cases[0].metadata["linked_chunk_ids"] == ["chunk-1"]
    assert cases[0].metadata["query_terms"] == ["map", "legend", "signal"]


def test_generate_retrieval_case_skeleton_can_create_visual_probe_from_source_ref():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        source_refs=["asset:asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        caption="Station access map legend signal",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        visual_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "map legend signal"
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].metadata["linked_chunk_ids"] == ["chunk-1"]


def test_generate_retrieval_case_skeleton_can_create_visual_image_probes(tmp_path):
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        path=tmp_path / "asset.png",
        caption="Station access map legend signal",
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        image_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "Station access map legend signal"
    assert cases[0].expected_pages == [1]
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].metadata["case_source"] == "visual_image_probe"
    assert cases[0].metadata["case_family"] == "visual"
    assert cases[0].metadata["evidence_family"] == "visual_image"
    assert cases[0].metadata["modality"] == "image"
    assert cases[0].metadata["target_vector"] == "image_dense"
    assert cases[0].metadata["asset_kind"] == "map"
    assert cases[0].metadata["linked_chunk_ids"] == ["chunk-1"]


def test_generate_retrieval_case_skeleton_can_create_visual_object_probes():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        metadata={
            "objects": [
                {
                    "label": "transfer hub marker",
                    "attributes": ["blue circle", "north gate"],
                    "location": "upper right quadrant",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                }
            ]
        },
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        object_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "transfer hub marker blue circle north gate upper"
    assert cases[0].expected_pages == [1]
    assert cases[0].expected_asset_ids == ["asset-1"]
    assert cases[0].metadata["case_source"] == "visual_object_probe"
    assert cases[0].metadata["case_family"] == "visual"
    assert cases[0].metadata["evidence_family"] == "visual_object"
    assert cases[0].metadata["modality"] == "vision_object"
    assert cases[0].metadata["object_label"] == "transfer hub marker"
    assert cases[0].metadata["object_has_bbox"] is True
    assert cases[0].metadata["object_probe_visual_only"] is True
    assert cases[0].metadata["linked_chunk_ids"] == ["chunk-1"]


def test_visual_object_probe_duplicate_queries_keep_one_expected_target_per_case():
    chunk_1 = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor evidence.",
        asset_ids=["asset-1"],
    )
    chunk_2 = DocumentChunk(
        chunk_id="chunk-2",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="Station area evidence.",
        asset_ids=["asset-2"],
    )
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.MAP,
            metadata={"visual_elements": [{"label": "legend color symbol meaning"}]},
        ),
        VisualAsset(
            asset_id="asset-2",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            metadata={"visual_elements": [{"label": "legend color symbol meaning"}]},
        ),
    ]

    cases = generate_retrieval_case_skeleton(
        [chunk_1, chunk_2],
        assets,
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        object_probe_limit=5,
    )

    assert len(cases) == 1
    assert len(cases[0].expected_pages) == 1
    assert len(cases[0].expected_asset_ids) == 1
    assert cases[0].metadata["duplicate_query_candidate_count"] == 2
    assert "merged_case_count" not in cases[0].metadata


def test_generate_retrieval_case_skeleton_uses_bbox_region_for_object_probes():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        metadata={
            "objects": [
                {
                    "label": "route icon",
                    "attributes": ["green square"],
                    "bbox": [0.05, 0.1, 0.2, 0.3],
                }
            ]
        },
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        object_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "route icon green square upper left"
    assert cases[0].metadata["object_has_bbox"] is True


def test_generate_retrieval_case_skeleton_uses_visual_elements_for_object_probes():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text=(
            "Transit evidence.\n\n"
            "[VLM page 1 map]\n"
            "Visual elements: station access corridor\n\n"
            "[Visual asset page 1 map asset-1]\n"
            "Visual elements: station access corridor"
        ),
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        metadata={"visual_elements": ["station access corridor"]},
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        object_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "station access corridor"
    assert cases[0].metadata["object_label"] == "station access corridor"
    assert cases[0].metadata["object_source_key"] == "visual_elements"
    assert cases[0].metadata["object_visual_feature_type"] == "visual_element"
    assert cases[0].metadata["object_has_bbox"] is False


def test_generate_retrieval_case_skeleton_object_probes_prefer_visual_only_terms():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor transfer hub marker evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        metadata={
            "objects": [
                {
                    "label": "transfer hub marker",
                    "attributes": ["blue circle", "north gate", "elevated deck"],
                }
            ]
        },
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        object_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "blue circle north gate elevated deck"
    assert "transfer" not in cases[0].query
    assert cases[0].metadata["query_terms"] == [
        "blue",
        "circle",
        "north",
        "gate",
        "elevated",
        "deck",
    ]


def test_generate_retrieval_case_skeleton_object_probes_ignore_generated_visual_context():
    base_chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text=(
            "Transit corridor station access evidence.\n\n"
            "Visual context:\nObjects: transfer hub marker: blue circle, north gate"
        ),
        asset_ids=["asset-1"],
        metadata={"visual_context_added": True},
    )
    visual_chunk = DocumentChunk(
        chunk_id="visual-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.MAP,
        text="Visual asset kind: map\nObjects: transfer hub marker: blue circle, north gate",
        asset_ids=["asset-1"],
        source_refs=["asset:asset-1"],
        metadata={"chunking_strategy": "visual_asset_text"},
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        metadata={
            "objects": [
                {
                    "label": "transfer hub marker",
                    "attributes": ["blue circle", "north gate"],
                }
            ]
        },
    )

    cases = generate_retrieval_case_skeleton(
        [base_chunk, visual_chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        object_probe_limit=1,
    )

    assert len(cases) == 1
    assert cases[0].query == "transfer hub marker blue circle north gate"
    assert cases[0].metadata["linked_chunk_ids"] == ["chunk-1", "visual-1"]
    assert cases[0].metadata["object_probe_visual_only"] is True


def test_generate_retrieval_case_skeleton_can_disable_visual_only_object_probe_terms():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor transfer hub marker evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        metadata={"objects": [{"label": "transfer hub marker"}]},
    )

    cases = generate_retrieval_case_skeleton(
        [chunk],
        [asset],
        [],
        include_pages=False,
        include_assets=False,
        include_triples=False,
        object_probe_limit=1,
        object_probe_visual_only=False,
    )

    assert len(cases) == 1
    assert cases[0].query == "transfer hub marker"
    assert cases[0].metadata["object_probe_visual_only"] is False


def test_generate_retrieval_cases_cli_writes_jsonl(tmp_path):
    package_dir = write_case_package(tmp_path)
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--page-limit",
            "1",
            "--asset-limit",
            "1",
            "--triple-limit",
            "1",
            "--query-mode",
            "salient_terms",
            "--selection-strategy",
            "salience",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["expected_pages"] == [1]
    assert rows[0]["metadata"]["query_mode"] == "salient_terms"
    assert rows[1]["expected_asset_ids"] == ["asset-1"]
    assert rows[2]["expected_triple_ids"] == ["triple-1"]


def test_generate_retrieval_cases_cli_accepts_target_concentration_limits(tmp_path):
    package_dir = write_case_package(tmp_path)
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--page-limit",
            "1",
            "--asset-limit",
            "1",
            "--no-include-triples",
            "--max-page-cases-per-target",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["case_source"] == "page"
    assert "'max_page_cases_per_target': 1" in result.output


def test_generate_retrieval_cases_cli_writes_visual_probe_cases(tmp_path):
    package_dir = write_case_package(tmp_path)
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--no-include-pages",
            "--no-include-assets",
            "--no-include-triples",
            "--visual-probe-limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["query"].startswith("map legend signal")
    assert rows[0]["metadata"]["case_source"] == "visual_lexical_probe"


def test_generate_retrieval_cases_cli_writes_object_probe_cases(tmp_path):
    package_dir = write_case_package(tmp_path)
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--no-include-pages",
            "--no-include-assets",
            "--no-include-triples",
            "--object-probe-limit",
            "1",
            "--no-object-probe-visual-only",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["case_source"] == "visual_object_probe"
    assert rows[0]["metadata"]["modality"] == "vision_object"
    assert rows[0]["metadata"]["object_probe_visual_only"] is False
    assert "'target_counts':" in result.output
    assert "'distinct_target_counts':" in result.output
    assert "'max_cases_per_target':" in result.output
    assert "'asset': 1" in result.output
    assert "'case_group_counts':" in result.output
    assert "'visual_object_probe': 1" in result.output
    assert "'visual_object_probe_count': 1" in result.output
    assert "'visual_only_object_probe_count': 0" in result.output
    assert "'non_visual_only_object_probe_count': 1" in result.output


def test_generate_retrieval_cases_cli_writes_image_probe_cases(tmp_path):
    package_dir = write_case_package(tmp_path)
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--no-include-pages",
            "--no-include-assets",
            "--no-include-triples",
            "--image-probe-limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["case_source"] == "visual_image_probe"
    assert rows[0]["metadata"]["target_vector"] == "image_dense"
    assert "'visual_image_probe_count': 1" in result.output
    assert "'image_probe_limit': 1" in result.output


def test_generate_retrieval_cases_cli_writes_hard_negative_targets(tmp_path):
    package_dir = write_case_package(tmp_path)
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc",
            page_start=1,
            page_end=1,
            kind=ChunkKind.TEXT,
            text="Transit corridor station access evidence.",
            asset_ids=["asset-1"],
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc",
            page_start=2,
            page_end=2,
            kind=ChunkKind.TEXT,
            text="Transit corridor freight depot evidence.",
        ),
    ]
    write_jsonl(package_dir / "chunks.jsonl", chunks)
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
            "--page-limit",
            "1",
            "--no-include-assets",
            "--no-include-triples",
            "--hard-negative-limit",
            "1",
            "--hard-negative-min-overlap-terms",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["excluded_pages"] == [2]
    assert rows[0]["excluded_chunk_ids"] == ["chunk-2"]
    assert "'excluded_target_counts':" in result.output
    assert "'page': 1" in result.output
    assert "'hard_negative_limit': 1" in result.output


def test_generate_retrieval_cases_cli_accepts_candidate_chunks(tmp_path):
    package_dir = write_case_package(tmp_path)
    candidate_chunks = tmp_path / "candidate_chunks.jsonl"
    write_jsonl(
        candidate_chunks,
        [
            DocumentChunk(
                chunk_id="candidate-1",
                doc_id="doc",
                page_start=2,
                page_end=2,
                kind=ChunkKind.TEXT,
                text="Distinct candidate retrieval evidence.",
            )
        ],
    )
    output = tmp_path / "cases.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "generate-retrieval-cases",
            "--package-dir",
            str(package_dir),
            "--chunks",
            str(candidate_chunks),
            "--output",
            str(output),
            "--page-limit",
            "1",
            "--no-include-assets",
            "--no-include-triples",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["expected_chunk_ids"] == ["candidate-1"]


def write_case_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    doc = SourceDocument(
        doc_id="doc",
        title="Reference Document",
        local_path=tmp_path / "reference.pdf",
    )
    profile = PageProfile(
        doc_id="doc",
        page_no=1,
        width=100,
        height=100,
        char_count=100,
        line_count=4,
        text_block_count=1,
        image_block_count=1,
        embedded_image_count=0,
        drawing_count=0,
        text_quality=TextQuality.GOOD,
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="Transit corridor station access evidence.",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        path=package_dir / "assets/page.png",
        caption="Station access map legend signal",
        metadata={
            "objects": [
                {
                    "label": "transfer hub marker",
                    "attributes": ["blue circle", "north gate"],
                    "location": "upper right quadrant",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                }
            ]
        },
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="corridor",
        predicate="connects",
        object="station",
    )
    manifest = ProcessingManifest(
        doc=doc,
        profiles=[profile],
        chunks=[chunk],
        assets=[asset],
        triples=[triple],
    )
    (package_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_jsonl(package_dir / "pages.jsonl", [profile])
    write_jsonl(package_dir / "chunks.jsonl", [chunk])
    write_jsonl(package_dir / "assets.jsonl", [asset])
    write_jsonl(package_dir / "triples.jsonl", [triple])
    return package_dir

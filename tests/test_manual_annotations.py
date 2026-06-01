from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.vision.manual_annotations import AssetAnnotation, apply_asset_annotations


def test_apply_asset_annotations_updates_assets_chunks_and_triples():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=12,
        kind=AssetKind.PAGE_IMAGE,
        caption="page",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=12,
        page_end=12,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base",
        asset_ids=["asset-1"],
    )

    assets, chunks, triples = apply_asset_annotations(
        [asset],
        [chunk],
        [
            AssetAnnotation(
                asset_id="asset-1",
                page_no=12,
                kind=AssetKind.MAP,
                vlm_summary="north district river corridor",
                triples=[
                    {
                        "subject": "north district",
                        "predicate": "has_development_concept",
                        "object": "river corridor development concept",
                        "confidence": 0.8,
                    }
                ],
                metadata={
                    "annotation_source": "visual_job",
                    "visual_job_id": "job-1",
                    "vlm_prompt_name": "map_summary",
                },
            )
        ],
        existing_triples=[
            GraphTriple(
                triple_id="existing",
                doc_id="doc",
                chunk_id="chunk-1",
                subject="a",
                predicate="b",
                object="c",
            )
        ],
    )

    assert assets[0].kind == AssetKind.MAP
    assert "north district river" in chunks[0].text
    assert len(triples) == 2
    visual_triple = next(triple for triple in triples if triple.subject == "north district")
    assert visual_triple.qualifiers["source"] == "visual_annotation"
    assert visual_triple.qualifiers["asset_id"] == "asset-1"
    assert visual_triple.qualifiers["page_no"] == 12
    assert visual_triple.qualifiers["asset_kind"] == "map"
    assert visual_triple.qualifiers["visual_job_id"] == "job-1"
    assert "confidence" not in visual_triple.qualifiers
    assert visual_triple.confidence == 0.8


def test_apply_asset_annotations_uses_source_ref_for_triple_chunk():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=12,
        kind=AssetKind.PAGE_IMAGE,
        caption="page",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=12,
        page_end=12,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base",
        source_refs=["asset:asset-1"],
    )

    assets, chunks, triples = apply_asset_annotations(
        [asset],
        [chunk],
        [
            AssetAnnotation(
                asset_id="asset-1",
                vlm_summary="source ref visual summary",
                triples=[
                    {
                        "subject": "source ref visual",
                        "predicate": "shows",
                        "object": "mapped condition",
                    }
                ],
            )
        ],
    )

    assert assets[0].vlm_summary == "source ref visual summary"
    assert "source ref visual summary" in chunks[0].text
    assert triples[0].chunk_id == "chunk-1"
    assert triples[0].qualifiers["asset_id"] == "asset-1"


def test_apply_asset_annotations_derives_triples_from_visual_metadata():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=12,
        kind=AssetKind.PAGE_IMAGE,
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=12,
        page_end=12,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base",
        asset_ids=["asset-1"],
    )

    assets, chunks, triples = apply_asset_annotations(
        [asset],
        [chunk],
        [
            AssetAnnotation(
                asset_id="asset-1",
                page_no=12,
                caption="map panel",
                vlm_summary="The panel shows a route line and a station marker.",
                metadata={
                    "annotation_source": "visual_job",
                    "visual_job_id": "job-1",
                    "entities": ["station marker"],
                    "objects": [{"label": "route line"}],
                },
            )
        ],
    )

    assert assets[0].caption == "map panel"
    assert "station marker" in chunks[0].text
    assert {
        (triple.subject, triple.predicate, triple.object)
        for triple in triples
    } == {
        ("map panel", "mentions_entity", "station marker"),
        ("map panel", "contains_object", "route line"),
    }
    assert {triple.qualifiers["asset_id"] for triple in triples} == {"asset-1"}
    assert {triple.qualifiers["visual_job_id"] for triple in triples} == {"job-1"}
    assert all(triple.qualifiers["derived_from_vlm_field"] is True for triple in triples)


def test_apply_asset_annotations_derives_page_level_triples_with_asset_ids():
    assets = [
        VisualAsset(
            asset_id="asset-1",
            doc_id="doc",
            page_no=12,
            kind=AssetKind.PAGE_IMAGE,
        ),
        VisualAsset(
            asset_id="asset-2",
            doc_id="doc",
            page_no=12,
            kind=AssetKind.TABLE,
        ),
    ]
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=12,
        page_end=12,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base",
        asset_ids=["asset-1", "asset-2"],
    )

    _, _, triples = apply_asset_annotations(
        assets,
        [chunk],
        [
            AssetAnnotation(
                page_no=12,
                caption="overview panel",
                metadata={"visual_elements": ["legend"]},
            )
        ],
    )

    assert len(triples) == 1
    assert triples[0].qualifiers["asset_ids"] == ["asset-1", "asset-2"]

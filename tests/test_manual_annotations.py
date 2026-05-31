from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, GraphTriple, VisualAsset
from chunking_docs.vision.manual_annotations import AssetAnnotation, apply_asset_annotations


def test_apply_asset_annotations_updates_assets_chunks_and_triples():
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=188,
        kind=AssetKind.PAGE_IMAGE,
        caption="page",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=188,
        page_end=188,
        kind=ChunkKind.PAGE_SUMMARY,
        text="base",
        asset_ids=["asset-1"],
    )

    assets, chunks, triples = apply_asset_annotations(
        [asset],
        [chunk],
        [
            AssetAnnotation(
                page_no=188,
                kind=AssetKind.MAP,
                vlm_summary="동북권 발전구상 중랑천",
                triples=[
                    {
                        "subject": "동북권",
                        "predicate": "has_development_concept",
                        "object": "중랑천 중심 발전구상",
                    }
                ],
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
    assert "동북권 발전구상" in chunks[0].text
    assert len(triples) == 2

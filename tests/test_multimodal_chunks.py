from pathlib import Path

from chunking_docs.chunking.multimodal import (
    add_visual_context_to_chunks,
    build_strategy_chunks,
    visual_asset_chunks,
)
from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, SectionPath, VisualAsset


def test_multimodal_strategy_adds_context_and_visual_chunks():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=3,
        page_end=3,
        kind=ChunkKind.TEXT,
        text="base text",
        section=SectionPath(chapter="Strategy", section="Mobility"),
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.MAP,
        path=Path("map.png"),
        caption="corridor map",
        vlm_summary="river corridor links station hubs",
    )

    chunks = build_strategy_chunks([chunk], [asset], strategy="multimodal")

    assert len(chunks) == 2
    assert chunks[0].text.startswith("Section: Strategy > Mobility")
    assert "Visual context" in chunks[0].text
    assert "corridor map" in chunks[0].text
    assert chunks[0].metadata["visual_context_added"] is True
    assert chunks[1].kind == ChunkKind.MAP
    assert chunks[1].metadata["chunking_strategy"] == "visual_asset_text"
    assert "river corridor" in chunks[1].text


def test_visual_asset_chunks_skip_assets_without_text():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="base",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.FIGURE,
        path=Path("figure.png"),
    )

    assert visual_asset_chunks([chunk], [asset]) == []


def test_visual_context_can_be_disabled_for_multimodal_text_chunks():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=2,
        page_end=2,
        kind=ChunkKind.TEXT,
        text="base text",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=2,
        kind=AssetKind.CHART,
        caption="chart caption",
    )

    assert add_visual_context_to_chunks([chunk], [asset], max_chars=0) == [chunk]


def test_multimodal_strategy_uses_structured_visual_metadata_as_context():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=3,
        page_end=3,
        kind=ChunkKind.TEXT,
        text="base text",
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.MAP,
        metadata={
            "entities": ["transfer hub"],
            "objects": [{"label": "station marker", "attributes": ["red circle"]}],
        },
    )

    chunks = build_strategy_chunks([chunk], [asset], strategy="multimodal")

    assert len(chunks) == 2
    assert "Entities: transfer hub" in chunks[0].text
    assert "Objects: station marker: red circle" in chunks[1].text


def test_multimodal_strategy_uses_source_ref_visual_asset_links():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=3,
        page_end=3,
        kind=ChunkKind.TEXT,
        text="base text",
        source_refs=["asset:asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.MAP,
        caption="corridor map",
        vlm_summary="river corridor links station hubs",
    )

    chunks = build_strategy_chunks([chunk], [asset], strategy="multimodal")

    assert len(chunks) == 2
    assert "Visual context" in chunks[0].text
    assert "corridor map" in chunks[0].text
    assert chunks[1].metadata["parent_chunk_id"] == "chunk-1"
    assert chunks[1].asset_ids == ["asset-1"]
    assert chunks[1].source_refs == ["asset:asset-1"]


def test_visual_asset_chunks_preserve_tile_metadata_for_text_filters():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=3,
        page_end=3,
        kind=ChunkKind.TEXT,
        text="base text",
        source_refs=["asset:tile-1"],
    )
    asset = VisualAsset(
        asset_id="tile-1",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.MAP,
        caption="tile corridor map",
        metadata={
            "asset_scope": "tile",
            "parent_asset_id": "page-asset",
            "tile_index": 2,
            "tile_row": 1,
            "tile_col": 0,
            "tile_rows": 2,
            "tile_cols": 2,
            "text_quality": "degraded",
            "text_quality_reasons": ["high_control_char_ratio"],
            "requires_ocr": True,
            "requires_vlm": True,
        },
    )

    chunks = visual_asset_chunks([chunk], [asset])

    assert len(chunks) == 1
    visual_chunk = chunks[0]
    assert visual_chunk.metadata["asset_scope"] == "tile"
    assert visual_chunk.metadata["parent_asset_id"] == "page-asset"
    assert visual_chunk.metadata["tile_index"] == 2
    assert visual_chunk.metadata["tile_row"] == 1
    assert visual_chunk.metadata["tile_col"] == 0
    assert visual_chunk.metadata["text_quality"] == "degraded"
    assert visual_chunk.metadata["text_quality_reasons"] == ["high_control_char_ratio"]
    assert visual_chunk.metadata["requires_ocr"] is True
    assert visual_chunk.metadata["requires_vlm"] is True
    assert "Asset scope: tile" in visual_chunk.text
    assert "Tile: 3/4, row 2 col 1" in visual_chunk.text
    assert "Text quality: degraded" in visual_chunk.text


def test_multimodal_strategy_keeps_unlinked_visual_assets_searchable():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=3,
        page_end=3,
        kind=ChunkKind.TEXT,
        text="base text",
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=3,
        kind=AssetKind.MAP,
        caption="standalone corridor map",
        vlm_summary="shows access coverage with station markers",
        metadata={"section_label": "Strategy > Access"},
    )

    chunks = build_strategy_chunks([chunk], [asset], strategy="multimodal")

    assert len(chunks) == 2
    visual_chunk = chunks[1]
    assert visual_chunk.metadata["visual_asset_unlinked"] is True
    assert visual_chunk.metadata["chunking_strategy"] == "visual_asset_text"
    assert visual_chunk.metadata["section_label"] == "Strategy > Access"
    assert visual_chunk.asset_ids == ["asset-1"]
    assert visual_chunk.source_refs == ["asset:asset-1"]
    assert "standalone corridor map" in visual_chunk.text
    assert "Page range: 3-3" in visual_chunk.text


def test_hierarchical_strategy_adds_parent_and_child_context():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=4,
        page_end=4,
        kind=ChunkKind.TEXT,
        text="\n\n".join(
            [
                "Transit corridor overview " + ("station access " * 12),
                "Investment table notes " + ("capital program " * 12),
            ]
        ),
        section=SectionPath(chapter="Strategy", section="Access"),
        asset_ids=["asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=4,
        kind=AssetKind.TABLE,
        caption="capital investment table",
        vlm_summary="table lists investment priorities by access corridor",
    )

    chunks = build_strategy_chunks(
        [chunk],
        [asset],
        strategy="hierarchical",
        max_chars=180,
        overlap_chars=20,
        min_chars=40,
        parent_max_chars=120,
        visual_context_chars=160,
    )

    parent = chunks[0]
    children = chunks[1:]
    assert parent.metadata["retrieval_role"] == "parent"
    assert parent.metadata["chunking_strategy"] == "hierarchical_parent"
    assert "Visual context" in parent.text
    assert len(children) > 1
    assert all(child.metadata["retrieval_role"] == "child" for child in children)
    assert all(child.metadata["hierarchical_parent_chunk_id"] == parent.chunk_id for child in children)
    assert any("capital investment table" in child.text for child in children)
    assert all(f"parent:{parent.chunk_id}" in child.source_refs for child in children)


def test_hierarchical_strategy_uses_source_ref_visual_context():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=4,
        page_end=4,
        kind=ChunkKind.TEXT,
        text="Transit corridor overview " + ("station access " * 12),
        source_refs=["asset:asset-1"],
    )
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=4,
        kind=AssetKind.CHART,
        caption="access chart",
        vlm_summary="chart compares station access coverage",
        metadata={
            "asset_scope": "tile",
            "tile_index": 1,
            "tile_row": 0,
            "tile_col": 1,
            "tile_rows": 2,
            "tile_cols": 2,
        },
    )

    chunks = build_strategy_chunks(
        [chunk],
        [asset],
        strategy="hierarchical",
        max_chars=120,
        overlap_chars=20,
        min_chars=40,
        parent_max_chars=80,
        visual_context_chars=160,
    )

    assert "access chart" in chunks[0].text
    assert "chart tile 2/4, row 1 col 2 page 4" in chunks[0].text
    assert any("access chart" in child.text for child in chunks[1:])


def test_hierarchical_strategy_keeps_unlinked_visual_assets_searchable():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=4,
        page_end=4,
        kind=ChunkKind.TEXT,
        text="Transit corridor overview " + ("station access " * 12),
    )
    linked_asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=4,
        kind=AssetKind.CHART,
        caption="linked access chart",
    )
    unlinked_asset = VisualAsset(
        asset_id="asset-2",
        doc_id="doc",
        page_no=4,
        kind=AssetKind.FIGURE,
        caption="standalone station diagram",
        vlm_summary="shows station entrances and transfer paths",
    )
    chunk = chunk.model_copy(update={"source_refs": ["asset:asset-1"]})

    chunks = build_strategy_chunks(
        [chunk],
        [linked_asset, unlinked_asset],
        strategy="hierarchical",
        max_chars=120,
        overlap_chars=20,
        min_chars=40,
        parent_max_chars=80,
        visual_context_chars=160,
    )

    standalone_chunks = [
        candidate
        for candidate in chunks
        if candidate.metadata.get("chunking_strategy") == "visual_asset_text"
    ]
    assert len(standalone_chunks) == 1
    assert standalone_chunks[0].asset_ids == ["asset-2"]
    assert "standalone station diagram" in standalone_chunks[0].text
    assert all("asset-1" not in candidate.asset_ids for candidate in standalone_chunks)

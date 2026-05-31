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
    assert any("access chart" in child.text for child in chunks[1:])

import fitz

from chunking_docs.graph.heuristics import section_triples
from chunking_docs.models import (
    ChunkKind,
    DocumentChunk,
    PageProfile,
    SectionPath,
    TextQuality,
)
from chunking_docs.vision.assets import build_page_tile_assets, page_kind, page_tiles, should_render_page


def test_page_kind_detects_visual_dense_page_as_map():
    profile = PageProfile(
        doc_id="doc",
        page_no=150,
        width=595,
        height=842,
        char_count=100,
        line_count=10,
        text_block_count=3,
        image_block_count=2,
        embedded_image_count=2,
        drawing_count=2,
        text_quality=TextQuality.DEGRADED,
    )

    assert page_kind(profile).value == "map"
    assert should_render_page(profile)


def test_page_kind_keeps_cover_as_page_image():
    profile = PageProfile(
        doc_id="doc",
        page_no=1,
        width=595,
        height=842,
        char_count=0,
        line_count=0,
        text_block_count=0,
        image_block_count=1,
        embedded_image_count=1,
        drawing_count=120,
        text_quality=TextQuality.EMPTY,
    )

    assert page_kind(profile).value == "page_image"


def test_page_tiles_adds_overlap_without_leaving_page_bounds():
    tiles = page_tiles(page_width=100, page_height=200, rows=2, cols=2, overlap_ratio=0.1)

    assert len(tiles) == 4
    assert tiles[0].bbox == (0.0, 0.0, 55.0, 110.0)
    assert tiles[-1].bbox == (45.0, 90.0, 100.0, 200.0)


def test_build_page_tile_assets_renders_tiles(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page(width=100, height=200)
    doc.save(pdf_path)
    doc.close()
    profile = PageProfile(
        doc_id="doc",
        page_no=1,
        width=100,
        height=200,
        char_count=0,
        line_count=0,
        text_block_count=0,
        image_block_count=0,
        embedded_image_count=0,
        drawing_count=0,
        text_quality=TextQuality.EMPTY,
        control_char_count=3,
        control_char_ratio=0.2,
        text_quality_reasons=["empty_text"],
    )

    assets = build_page_tile_assets(
        pdf_path=pdf_path,
        doc_id="doc",
        profiles=[profile],
        output_dir=tmp_path / "assets",
        rows=2,
        cols=2,
        overlap_ratio=0.1,
        zoom=1.0,
    )

    assert len(assets) == 4
    assert all(asset.path and asset.path.exists() for asset in assets)
    assert assets[0].metadata["asset_scope"] == "tile"
    assert assets[0].metadata["tile_rows"] == 2
    assert assets[0].metadata["tile_cols"] == 2
    assert assets[0].metadata["requires_ocr"] is True
    assert assets[0].metadata["control_char_count"] == 3
    assert assets[0].metadata["control_char_ratio"] == 0.2
    assert assets[0].metadata["text_quality_reasons"] == ["empty_text"]


def test_section_triples_from_chunk_metadata():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=37,
        page_end=37,
        kind=ChunkKind.TEXT,
        text="",
        section=SectionPath(
            chapter="제3장 핵심이슈별 계획",
            issue="핵심이슈 1 차별없이 더불어 사는 사람중심 도시",
        ),
    )

    triples = section_triples([chunk])

    assert any(triple.predicate == "includes_chapter" for triple in triples)
    assert any(triple.predicate == "includes_issue" for triple in triples)

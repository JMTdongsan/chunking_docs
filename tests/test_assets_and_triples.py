from chunking_docs.graph.heuristics import section_triples
from chunking_docs.models import (
    ChunkKind,
    DocumentChunk,
    PageProfile,
    SectionPath,
    TextQuality,
)
from chunking_docs.vision.assets import page_kind, should_render_page


def test_page_kind_detects_spatial_page_as_map():
    profile = PageProfile(
        doc_id="doc",
        page_no=150,
        width=595,
        height=842,
        char_count=100,
        line_count=10,
        text_block_count=3,
        image_block_count=1,
        embedded_image_count=1,
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

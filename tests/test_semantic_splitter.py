from chunking_docs.chunking.semantic_splitter import semantic_subchunks, split_text
from chunking_docs.models import ChunkKind, DocumentChunk, SectionPath


def test_split_text_keeps_overlap_for_long_blocks():
    text = "\n\n".join([f"{index}. " + ("서울 도시계획 " * 35) for index in range(8)])

    chunks = split_text(text, max_chars=500, overlap_chars=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 560 for chunk in chunks)


def test_semantic_subchunks_preserve_metadata_and_parent():
    chunk = DocumentChunk(
        chunk_id="parent-1",
        doc_id="doc",
        page_start=10,
        page_end=10,
        kind=ChunkKind.PAGE_SUMMARY,
        text="\n\n".join([f"제{index}절 " + ("공간구조 " * 60) for index in range(1, 4)]),
        section=SectionPath(chapter="제4장 공간구조 및 토지이용계획"),
        metadata={"source": "test"},
    )

    chunks = semantic_subchunks([chunk], max_chars=450, overlap_chars=40)

    assert len(chunks) > 1
    assert chunks[0].kind == ChunkKind.TEXT
    assert chunks[0].metadata["parent_chunk_id"] == "parent-1"
    assert chunks[0].section.chapter == "제4장 공간구조 및 토지이용계획"
